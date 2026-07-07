"""Independent MADQN：每个智能体一套独立的 Q 网络 + 目标网络 + 优化器 + buffer。

DTDE（Decentralized Training, Decentralized Execution）的最简实现：
所有 agent 独立学习，无任何参数共享或全局信息。
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.base import BaseAlgorithm
from algorithms.common.networks import MlpQNet


class MADQN(BaseAlgorithm):
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

        self.qs = [
            MlpQNet(self.state_dim, algo_cfg["hidden_dim"], self.n_actions).to(self.device)
            for _ in range(self.num_agents)
        ]
        self.q_targets = [deepcopy(q) for q in self.qs]
        for q_t in self.q_targets:
            for p in q_t.parameters():
                p.requires_grad = False
        self.optimizers = [
            torch.optim.Adam(q.parameters(), lr=self.lr) for q in self.qs
        ]

    def required_buffer(self) -> str:
        return "per_agent"

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

    def update(self, batches: dict[int, tuple]) -> dict[str, float]:
        losses = []
        for i in range(self.num_agents):
            s, a, r, s_next, d = batches[i]
            s = torch.from_numpy(s).float().to(self.device)
            a = torch.from_numpy(a).long().to(self.device)
            r = torch.from_numpy(r).float().to(self.device)
            s_next = torch.from_numpy(s_next).float().to(self.device)
            d = torch.from_numpy(d).float().to(self.device)

            q = self.qs[i](s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = self.q_targets[i](s_next).max(dim=1)[0]
                target = r + self.gamma * q_next * (1 - d)
            loss = F.mse_loss(q, target)
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()
            losses.append(float(loss.item()))

        self._update_count += 1
        if self._update_count % self.update_freq == 0:
            for q, q_t in zip(self.qs, self.q_targets):
                q_t.load_state_dict(q.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return {"loss": float(np.mean(losses)), "epsilon": float(self.epsilon)}

    def save(self, path: str) -> None:
        torch.save([q.state_dict() for q in self.qs], path)

    def load(self, path: str) -> None:
        states = torch.load(path, map_location=self.device)
        for q, sd in zip(self.qs, states):
            q.load_state_dict(sd)
        for q, q_t in zip(self.qs, self.q_targets):
            q_t.load_state_dict(q.state_dict())
