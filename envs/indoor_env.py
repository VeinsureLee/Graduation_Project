"""多智能体室内导航环境（基于 numpy 网格 + Gymnasium 接口）。

为支持多智能体，我们直接基于 map_builder 生成的网格实现 reset/step，
未直接复用 MiniGrid 单智能体环境，但沿用其房间/门/格子世界设计风格。

接口契约：
    obs:    dict[agent_id, np.ndarray]
    action: dict[agent_id, int]   动作 0..3 表示四个方向
"""
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.map_builder import DOOR, EMPTY, WALL, build_grid_array, load_map_spec

# 动作编码
ACTION_RIGHT = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_UP = 3
ACTION_DELTAS: dict[int, tuple[int, int]] = {
    ACTION_RIGHT: (0, 1),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_UP: (-1, 0),
}

# 视图编码（用于观测）
VIEW_EMPTY = 0
VIEW_WALL = 1
VIEW_DOOR = 2
VIEW_SELF = 3
VIEW_OTHER = 4
VIEW_GOAL = 5


class IndoorEnv(gym.Env):
    """多智能体室内导航环境。"""

    metadata = {"render_modes": ["rgb_array"]}

    # 随机采样起点/终点的最大重试次数；超限说明地图可能不连通。
    MAX_SPAWN_RETRIES = 200

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        env_cfg = cfg["env"]
        self.spec = load_map_spec(env_cfg["map_file"])
        self.grid = build_grid_array(self.spec)
        self.rows, self.cols = self.grid.shape
        self.num_agents = self.spec["num_agents"]

        self.observation_mode = env_cfg["observation_mode"]
        self.view_size = env_cfg["partial_view_size"]
        self.random_spawn = env_cfg.get("random_spawn", False)
        self.reward_mode = env_cfg["reward_mode"]
        self.r_goal = env_cfg["reward_goal"]
        self.r_step = env_cfg["reward_step"]
        self.r_collision = env_cfg["reward_collision"]
        self.r_team = env_cfg["reward_team_bonus"]

        self.action_space = spaces.Discrete(4)
        obs_dim = self._obs_dim()
        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # 预存所有可站立的空地格（EMPTY），供随机采样使用。
        self._floor_cells = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r, c] == EMPTY
        ]

        seed = cfg.get("seed", None)
        self._rng = np.random.default_rng(seed)
        # 当前 episode 的起点与目标（reset 时填充）。
        self.agent_positions: list[list[int]] = []
        self.goals: list[list[int]] = []
        self._goal_reached: dict[int, bool] = {}

    def _obs_dim(self) -> int:
        if self.observation_mode == "partial":
            return self.view_size * self.view_size
        return self.rows * self.cols

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> dict[int, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.random_spawn:
            self.agent_positions, self.goals = self._sample_spawn()
        else:
            self.agent_positions = [list(p) for p in self.spec["agents_start"]]
            self.goals = [list(g) for g in self.spec["goals"]]
        self._goal_reached = {i: False for i in range(self.num_agents)}
        return self._get_obs()

    # ---------------- 随机起点/终点采样 ----------------

    def _is_passable(self, r: int, c: int) -> bool:
        """空地与门均可通行（用于可达性 BFS）。"""
        return self._in_bounds(r, c) and self.grid[r, c] in (EMPTY, DOOR)

    def _reachable(self, start: list[int], goal: list[int]) -> bool:
        """BFS 判断 start 是否能走到 goal（四连通）。"""
        src, dst = tuple(start), tuple(goal)
        seen = {src}
        queue = deque([src])
        while queue:
            r, c = queue.popleft()
            if (r, c) == dst:
                return True
            for dr, dc in ACTION_DELTAS.values():
                nr, nc = r + dr, c + dc
                if (nr, nc) not in seen and self._is_passable(nr, nc):
                    seen.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def _sample_spawn(self) -> tuple[list[list[int]], list[list[int]]]:
        """随机采样互不重叠的起点与目标，并保证每个 agent 起点可达其目标。

        采样自空地格（不含墙/门）。前 N 个为起点、后 N 个为目标，2N 格互不重叠。
        超过 MAX_SPAWN_RETRIES 仍不满足可达性则报错（提示检查地图连通性）。
        """
        n = self.num_agents
        if len(self._floor_cells) < 2 * n:
            raise RuntimeError(
                f"Map has too few floor cells ({len(self._floor_cells)}) "
                f"to place {n} starts and {n} goals."
            )
        for _ in range(self.MAX_SPAWN_RETRIES):
            idx = self._rng.choice(len(self._floor_cells), size=2 * n, replace=False)
            picks = [self._floor_cells[i] for i in idx]
            starts = [list(p) for p in picks[:n]]
            goals = [list(p) for p in picks[n:]]
            if all(self._reachable(starts[i], goals[i]) for i in range(n)):
                return starts, goals
        raise RuntimeError(
            "Failed to sample reachable start/goal pairs after "
            f"{self.MAX_SPAWN_RETRIES} retries; check map connectivity."
        )

    def step(self, actions: dict[int, int]):
        rewards = {i: 0.0 for i in range(self.num_agents)}
        dones = {i: self._goal_reached[i] for i in range(self.num_agents)}
        infos: dict[int, dict] = {i: {} for i in range(self.num_agents)}

        for agent_id, action in actions.items():
            if self._goal_reached[agent_id]:
                continue
            dr, dc = ACTION_DELTAS[action]
            r, c = self.agent_positions[agent_id]
            nr, nc = r + dr, c + dc
            if not self._in_bounds(nr, nc) or self.grid[nr, nc] == WALL:
                rewards[agent_id] += self.r_collision
                infos[agent_id]["collision"] = True
            else:
                self.agent_positions[agent_id] = [nr, nc]
            rewards[agent_id] += self.r_step

            goal = self.goals[agent_id]
            if self.agent_positions[agent_id] == list(goal):
                rewards[agent_id] += self.r_goal
                self._goal_reached[agent_id] = True
                dones[agent_id] = True

        if self.reward_mode == "cooperative":
            bonus = self._compute_team_bonus()
            if bonus > 0:
                for i in rewards:
                    rewards[i] += bonus

        return self._get_obs(), rewards, dones, infos

    def _compute_team_bonus(self) -> float:
        if all(self._goal_reached.values()):
            return self.r_team
        return 0.0

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _get_obs(self) -> dict[int, np.ndarray]:
        obs = {}
        for i in range(self.num_agents):
            if self.observation_mode == "partial":
                obs[i] = self._partial_view(i).flatten().astype(np.float32)
            else:
                obs[i] = self._full_view(i).flatten().astype(np.float32)
        return obs

    def _partial_view(self, agent_id: int) -> np.ndarray:
        r, c = self.agent_positions[agent_id]
        half = self.view_size // 2
        view = np.full((self.view_size, self.view_size), VIEW_WALL, dtype=np.int8)
        for i in range(self.view_size):
            for j in range(self.view_size):
                gr, gc = r - half + i, c - half + j
                if self._in_bounds(gr, gc):
                    cell = self.grid[gr, gc]
                    if cell == EMPTY:
                        view[i, j] = VIEW_EMPTY
                    elif cell == DOOR:
                        view[i, j] = VIEW_DOOR
                    else:
                        view[i, j] = VIEW_WALL
                    # 覆盖：其他 agent / goal
                    for other_id, pos in enumerate(self.agent_positions):
                        if other_id != agent_id and pos == [gr, gc]:
                            view[i, j] = VIEW_OTHER
                    goal = self.goals[agent_id]
                    if list(goal) == [gr, gc]:
                        view[i, j] = VIEW_GOAL
        view[half, half] = VIEW_SELF
        return view

    def _full_view(self, agent_id: int) -> np.ndarray:
        view = self.grid.copy().astype(np.int8)
        for i, pos in enumerate(self.agent_positions):
            view[pos[0], pos[1]] = VIEW_SELF if i == agent_id else VIEW_OTHER
        goal = self.goals[agent_id]
        if self.grid[goal[0], goal[1]] in (EMPTY, DOOR):
            view[goal[0], goal[1]] = VIEW_GOAL
        return view

    def render(self) -> np.ndarray:
        """返回 RGB 数组用于可视化。"""
        rgb = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        # 黑白底色：空地=白，墙=黑；门保持打开，渲染为白色通道
        rgb[self.grid == EMPTY] = (255, 255, 255)
        rgb[self.grid == WALL] = (0, 0, 0)
        rgb[self.grid == DOOR] = (255, 255, 255)
        # 先画 goal（避免被 agent 遮挡）
        for i, (r, c) in enumerate(self.goals):
            if self.grid[r, c] in (EMPTY, DOOR):
                rgb[r, c] = (50, 200, 50)
        # 再画 agents
        agent_colors = [(50, 100, 220), (220, 50, 50), (200, 100, 200), (50, 200, 200)]
        for i, (r, c) in enumerate(self.agent_positions):
            rgb[r, c] = agent_colors[i % len(agent_colors)]
        return rgb
