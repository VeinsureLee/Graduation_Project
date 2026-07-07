"""Parameter-shared MADQN：所有 agent 共用同一个 Q 网络。

参数共享版独立 Q-learning：训练时所有 agent 的经验都用来更新同一个 Q 网络，
执行时所有 agent 用同一个网络选动作。在同质多智能体任务中通常比 MADQN 更稳定，
样本效率更高。
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.base import BaseAlgorithm
from algorithms.common.networks import MlpQNet


class SharedMADQN(BaseAlgorithm):
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

        self.q = MlpQNet(self.state_dim, algo_cfg["hidden_dim"], self.n_actions).to(
            self.device
        )
        self.q_target = deepcopy(self.q)
        for p in self.q_target.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)

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
                    actions[i] = int(self.q(s).argmax(dim=1).item())
        return actions

    def update(self, batches: dict[int, tuple]) -> dict[str, float]:
        # 拼接所有 agent 的 batch 到一起做一次大更新
        all_s, all_a, all_r, all_s_next, all_d = [], [], [], [], []
        for i in range(self.num_agents):
            s, a, r, s_next, d = batches[i]
            all_s.append(s)
            all_a.append(a)
            all_r.append(r)
            all_s_next.append(s_next)
            all_d.append(d)
        s = torch.from_numpy(np.concatenate(all_s)).float().to(self.device)
        a = torch.from_numpy(np.concatenate(all_a)).long().to(self.device)
        r = torch.from_numpy(np.concatenate(all_r)).float().to(self.device)
        s_next = torch.from_numpy(np.concatenate(all_s_next)).float().to(self.device)
        d = torch.from_numpy(np.concatenate(all_d)).float().to(self.device)

        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(s_next).max(dim=1)[0]
            target = r + self.gamma * q_next * (1 - d)
        loss = F.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return {"loss": float(loss.item()), "epsilon": float(self.epsilon)}

    def save(self, path: str) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path: str) -> None:
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.q_target.load_state_dict(self.q.state_dict())
