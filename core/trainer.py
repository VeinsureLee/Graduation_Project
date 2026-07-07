"""统一训练循环。

通过 algo.required_buffer() 选择缓冲区类型；通过 algo.is_on_policy 选择
on/off-policy 执行分支。
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.base import BaseAlgorithm
from algorithms.common.buffers import JointReplayBuffer, ReplayBuffer


@dataclass
class TrainResult:
    run_id: str
    model_path: str
    history: list[dict] = field(default_factory=list)


# ====================================================
# Off-policy
# ====================================================

def _make_buffer(algo: BaseAlgorithm, env, algo_cfg: dict) -> Any:
    kind = algo.required_buffer()
    cap = algo_cfg["replay_buffer_size"]
    if kind == "single":
        return ReplayBuffer(cap)
    if kind == "per_agent":
        return {i: ReplayBuffer(cap) for i in range(env.num_agents)}
    if kind == "joint":
        return JointReplayBuffer(cap, env.num_agents, env.observation_space.shape[0])
    return None


def _push(buffer, kind: str, algo, states, actions, rewards, next_states, dones) -> None:
    if kind == "single":
        a = algo.controlled_agent
        buffer.push(states[a], actions[a], rewards[a], next_states[a], dones[a])
    elif kind == "per_agent":
        for i in buffer:
            buffer[i].push(states[i], actions[i], rewards[i], next_states[i], dones[i])
    elif kind == "joint":
        buffer.push(states, actions, rewards, next_states, dones)


def _sample(buffer, kind: str, batch_size: int):
    if kind in ("single", "joint"):
        return buffer.sample(batch_size)
    if kind == "per_agent":
        return {i: buffer[i].sample(batch_size) for i in buffer}
    return None


def _ready(buffer, kind: str, batch_size: int) -> bool:
    if kind in ("single", "joint"):
        return len(buffer) >= batch_size
    if kind == "per_agent":
        return all(len(b) >= batch_size for b in buffer.values())
    return False


def _run_off_policy_episode(
    algo: BaseAlgorithm,
    env,
    buffer,
    kind: str,
    max_steps: int,
    train_interval: int,
    batch_size: int,
    learn: bool,
) -> dict:
    states = env.reset()
    total_reward = 0.0
    step_count = 0
    losses: list[float] = []

    for step in range(max_steps):
        actions = algo.take_action(states, explore=True)
        next_states, rewards, dones, _ = env.step(actions)
        step_count += 1
        total_reward += sum(rewards.values())

        _push(buffer, kind, algo, states, actions, rewards, next_states, dones)
        states = next_states

        if learn and _ready(buffer, kind, batch_size) and step % train_interval == 0:
            batch = _sample(buffer, kind, batch_size)
            info = algo.update(batch)
            losses.append(info.get("loss", 0.0))

        if all(dones.values()):
            break

    return {
        "reward": total_reward,
        "steps": step_count,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
    }


def _train_off_policy(
    algo: BaseAlgorithm, env, cfg: dict, run_dir: Path
) -> TrainResult:
    algo_cfg = cfg["algorithm"]
    num_episodes = algo_cfg["num_episodes"]
    episode_length = algo_cfg["episode_length"]
    batch_size = algo_cfg["batch_size"]
    train_interval = algo_cfg["train_interval"]
    save_interval = cfg["logging"]["save_interval"]

    kind = algo.required_buffer()
    buffer = _make_buffer(algo, env, algo_cfg)
    history: list[dict] = []
    model_path = str(run_dir / "model.pth")

    for ep in range(num_episodes):
        t0 = time.time()
        info = _run_off_policy_episode(
            algo, env, buffer, kind, episode_length, train_interval, batch_size, True
        )
        info["episode"] = ep
        info["wall_time"] = time.time() - t0
        history.append(info)

        if ep % save_interval == 0 or ep == num_episodes - 1:
            algo.save(model_path)

    return TrainResult(run_id=run_dir.name, model_path=model_path, history=history)


# ====================================================
# On-policy (PPO / MAPPO)
# ====================================================

def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 GAE 优势与折扣回报。

    输入均为 1D 时序数组（长度 T）。返回 (advantages, returns) 同长度。
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_v = last_value
    for t in reversed(range(T)):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        adv[t] = gae
        next_v = values[t]
    returns = adv + values
    return adv, returns


def _train_ppo(algo, env, cfg: dict, run_dir: Path) -> TrainResult:
    """PPO 单智能体训练：仅控制 agent 0，其他随机。"""
    algo_cfg = cfg["algorithm"]
    num_episodes = algo_cfg["num_episodes"]
    episode_length = algo_cfg["episode_length"]
    update_interval = algo_cfg["update_interval"]
    save_interval = cfg["logging"]["save_interval"]
    ca = algo.controlled_agent

    history: list[dict] = []
    model_path = str(run_dir / "model.pth")

    rb_states: list = []
    rb_actions: list = []
    rb_log_probs: list = []
    rb_rewards: list = []
    rb_values: list = []
    rb_dones: list = []
    steps_accum = 0

    for ep in range(num_episodes):
        t0 = time.time()
        states = env.reset()
        total_reward = 0.0
        ep_steps = 0
        last_value = 0.0

        for _ in range(episode_length):
            action_ca, log_prob_ca, value_ca = algo.evaluate(states[ca])
            actions = {ca: action_ca}
            for i in range(algo.num_agents):
                if i != ca:
                    actions[i] = int(np.random.randint(algo.n_actions))
            next_states, rewards, dones, _ = env.step(actions)
            ep_steps += 1
            total_reward += sum(rewards.values())

            rb_states.append(states[ca])
            rb_actions.append(action_ca)
            rb_log_probs.append(log_prob_ca)
            rb_rewards.append(rewards[ca])
            rb_values.append(value_ca)
            rb_dones.append(float(dones[ca]))
            steps_accum += 1
            states = next_states

            if all(dones.values()):
                break

            if steps_accum >= update_interval:
                # 用当前状态估算 bootstrap value
                _, _, last_value = algo.evaluate(states[ca])
                _update_ppo_buffer(
                    algo, rb_states, rb_actions, rb_log_probs, rb_rewards,
                    rb_values, rb_dones, last_value
                )
                rb_states.clear(); rb_actions.clear(); rb_log_probs.clear()
                rb_rewards.clear(); rb_values.clear(); rb_dones.clear()
                steps_accum = 0

        history.append({
            "episode": ep,
            "reward": total_reward,
            "steps": ep_steps,
            "mean_loss": 0.0,
            "wall_time": time.time() - t0,
        })
        if ep % save_interval == 0 or ep == num_episodes - 1:
            algo.save(model_path)

    return TrainResult(run_id=run_dir.name, model_path=model_path, history=history)


def _update_ppo_buffer(algo, states, actions, log_probs, rewards, values, dones, last_value):
    adv, ret = _compute_gae(
        np.array(rewards, dtype=np.float32),
        np.array(values, dtype=np.float32),
        np.array(dones, dtype=np.float32),
        last_value, algo.gamma, algo.gae_lambda,
    )
    rollout = {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "log_probs": np.array(log_probs, dtype=np.float32),
        "returns": ret,
        "advantages": adv,
    }
    algo.update(rollout)


def _train_mappo(algo, env, cfg: dict, run_dir: Path) -> TrainResult:
    """MAPPO：所有 agent 共享 actor，中心化 critic。"""
    algo_cfg = cfg["algorithm"]
    num_episodes = algo_cfg["num_episodes"]
    episode_length = algo_cfg["episode_length"]
    update_interval = algo_cfg["update_interval"]
    save_interval = cfg["logging"]["save_interval"]

    history: list[dict] = []
    model_path = str(run_dir / "model.pth")

    # rollout 缓冲
    rb_local: list = []      # per-step per-agent local obs
    rb_global: list = []     # per-step per-agent global state (重复 N 次/步)
    rb_actions: list = []
    rb_log_probs: list = []
    rb_rewards: list = []    # per-step per-agent team reward (重复 N 次/步)
    rb_values: list = []     # per-step per-agent value (来自同一个全局 critic)
    rb_dones: list = []
    steps_accum = 0

    for ep in range(num_episodes):
        t0 = time.time()
        states = env.reset()
        total_reward = 0.0
        ep_steps = 0

        for _ in range(episode_length):
            per_agent, value, global_s = algo.evaluate_joint(states)
            actions = {i: per_agent[i][0] for i in range(algo.num_agents)}
            next_states, rewards, dones, _ = env.step(actions)
            ep_steps += 1
            r_team = sum(rewards.values())
            total_reward += r_team

            for i in range(algo.num_agents):
                rb_local.append(states[i])
                rb_global.append(global_s)
                rb_actions.append(per_agent[i][0])
                rb_log_probs.append(per_agent[i][1])
                rb_rewards.append(r_team)
                rb_values.append(value)
                rb_dones.append(float(dones[i]))
            steps_accum += algo.num_agents
            states = next_states

            if all(dones.values()):
                break

            if steps_accum >= update_interval:
                _, last_value, _ = algo.evaluate_joint(states)
                _update_mappo_buffer(
                    algo, rb_local, rb_global, rb_actions, rb_log_probs,
                    rb_rewards, rb_values, rb_dones, last_value
                )
                rb_local.clear(); rb_global.clear(); rb_actions.clear()
                rb_log_probs.clear(); rb_rewards.clear(); rb_values.clear()
                rb_dones.clear()
                steps_accum = 0

        history.append({
            "episode": ep,
            "reward": total_reward,
            "steps": ep_steps,
            "mean_loss": 0.0,
            "wall_time": time.time() - t0,
        })
        if ep % save_interval == 0 or ep == num_episodes - 1:
            algo.save(model_path)

    return TrainResult(run_id=run_dir.name, model_path=model_path, history=history)


def _update_mappo_buffer(algo, local, global_s, actions, log_probs, rewards, values, dones, last_value):
    adv, ret = _compute_gae(
        np.array(rewards, dtype=np.float32),
        np.array(values, dtype=np.float32),
        np.array(dones, dtype=np.float32),
        last_value, algo.gamma, algo.gae_lambda,
    )
    rollout = {
        "local_states": np.array(local, dtype=np.float32),
        "global_states": np.array(global_s, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "log_probs": np.array(log_probs, dtype=np.float32),
        "returns": ret,
        "advantages": adv,
    }
    algo.update(rollout)


def _train_on_policy(algo, env, cfg: dict, run_dir: Path) -> TrainResult:
    """Dispatch by algo class name (PPO vs MAPPO)."""
    name = algo.__class__.__name__.lower()
    if name == "ppo":
        return _train_ppo(algo, env, cfg, run_dir)
    if name == "mappo":
        return _train_mappo(algo, env, cfg, run_dir)
    raise NotImplementedError(f"On-policy training not implemented for {name}")


def train(algo: BaseAlgorithm, env, cfg: dict, run_dir: Path) -> TrainResult:
    if algo.is_on_policy:
        return _train_on_policy(algo, env, cfg, run_dir)
    return _train_off_policy(algo, env, cfg, run_dir)
