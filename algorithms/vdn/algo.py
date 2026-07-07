"""VDN (Value Decomposition Network)：CTDE 中最简单的值分解方法。

每个 agent 有独立 Q 网络，联合 Q 值 = 各 agent Q 值之和：
    Q_tot(s, a) = sum_i Q_i(o_i, a_i)
团队 TD 损失：(Q_tot - (r_team + gamma * max_a' Q_tot(s', a')))^2
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import BaseAlgorithm
from algorithms.common.networks import MlpQNet


class VDN(BaseAlgorithm):
    def __init__(self, env, cfg: dict):
        algo_cfg = cfg["algorithm"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = algo_cfg["gamma"]
        self.lr = algo_cfg["lr"]
        self.epsilon = algo_cfg["epsilon"]
        self.epsilon_min = algo_cfg["epsilon_min"]
        self.epsilon_decay = algo_cfg["epsilon_decay"]
        self.update_freq = algo_cfg["update_freq"]
        self._update_count = 0

        self.n_actions = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.num_agents = env.num_agents

        self.qs = nn.ModuleList(
            [
                MlpQNet(self.state_dim, algo_cfg["hidden_dim"], self.n_actions)
                for _ in range(self.num_agents)
            ]
        ).to(self.device)
        self.q_targets = deepcopy(self.qs)
        for p in self.q_targets.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.qs.parameters(), lr=self.lr)

    def required_buffer(self) -> str:
        return "joint"

    def take_action(self, states, explore=True):
        actions = {}
        for i in range(self.num_agents):
            if explore and np.random.rand() < self.epsilon:
                actions[i] = int(np.random.randint(self.n_actions))
            else:
                s = torch.from_numpy(states[i]).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    actions[i] = int(self.qs[i](s).argmax(dim=1).item())
        return actions

    def update(self, batch) -> dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        # shapes: states (B, N, state_dim), actions (B, N), rewards (B, N), dones (B, N)
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        q_taken_list = []
        q_next_max_list = []
        for i in range(self.num_agents):
            q_i = self.qs[i](states[:, i, :])
            q_taken_list.append(q_i.gather(1, actions[:, i].unsqueeze(1)).squeeze(1))
            with torch.no_grad():
                q_n = self.q_targets[i](next_states[:, i, :])
                q_next_max_list.append(q_n.max(dim=1)[0])
        q_tot = torch.stack(q_taken_list, dim=1).sum(dim=1)
        q_next_tot = torch.stack(q_next_max_list, dim=1).sum(dim=1)
        r_team = rewards.sum(dim=1)
        d_team = (dones.sum(dim=1) >= self.num_agents).float()
        target = r_team + self.gamma * q_next_tot * (1 - d_team)
        loss = F.mse_loss(q_tot, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.update_freq == 0:
            self.q_targets.load_state_dict(self.qs.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return {"loss": float(loss.item()), "epsilon": float(self.epsilon)}

    def save(self, path: str) -> None:
        torch.save(self.qs.state_dict(), path)

    def load(self, path: str) -> None:
        self.qs.load_state_dict(torch.load(path, map_location=self.device))
        self.q_targets.load_state_dict(self.qs.state_dict())
