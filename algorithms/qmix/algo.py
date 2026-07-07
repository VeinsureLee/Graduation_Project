"""QMIX：CTDE 中带 monotonic mixer 的值分解方法。

每个 agent 独立 Q 网络；mixer 网络（带绝对值权重）保证：
    d Q_tot / d Q_i >= 0   (单调性)
联合 Q 通过 mixer 计算：Q_tot = mixer(Q_1, ..., Q_N, s_global)
团队 TD 损失：(Q_tot - (r_team + gamma * Q_tot_target(s', a')))^2
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import BaseAlgorithm
from algorithms.common.networks import MlpQNet, QMixer


class QMIX(BaseAlgorithm):
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
        # 全局状态用所有 agent 观测拼接
        self.global_dim = self.state_dim * self.num_agents

        self.qs = nn.ModuleList(
            [
                MlpQNet(self.state_dim, algo_cfg["hidden_dim"], self.n_actions)
                for _ in range(self.num_agents)
            ]
        ).to(self.device)
        self.mixer = QMixer(self.num_agents, self.global_dim).to(self.device)

        self.qs_target = deepcopy(self.qs)
        self.mixer_target = deepcopy(self.mixer)
        for p in self.qs_target.parameters():
            p.requires_grad = False
        for p in self.mixer_target.parameters():
            p.requires_grad = False

        params = list(self.qs.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

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
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        bs = states.size(0)
        global_s = states.view(bs, -1)
        global_s_next = next_states.view(bs, -1)

        q_taken_list = []
        q_next_max_list = []
        for i in range(self.num_agents):
            q_i = self.qs[i](states[:, i, :])
            q_taken_list.append(q_i.gather(1, actions[:, i].unsqueeze(1)).squeeze(1))
            with torch.no_grad():
                q_n = self.qs_target[i](next_states[:, i, :])
                q_next_max_list.append(q_n.max(dim=1)[0])
        q_taken = torch.stack(q_taken_list, dim=1)
        q_next_max = torch.stack(q_next_max_list, dim=1)

        q_tot = self.mixer(q_taken, global_s)
        with torch.no_grad():
            q_next_tot = self.mixer_target(q_next_max, global_s_next)
            r_team = rewards.sum(dim=1)
            d_team = (dones.sum(dim=1) >= self.num_agents).float()
            target = r_team + self.gamma * q_next_tot * (1 - d_team)
        loss = F.mse_loss(q_tot, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.update_freq == 0:
            self.qs_target.load_state_dict(self.qs.state_dict())
            self.mixer_target.load_state_dict(self.mixer.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return {"loss": float(loss.item()), "epsilon": float(self.epsilon)}

    def save(self, path: str) -> None:
        torch.save(
            {"qs": self.qs.state_dict(), "mixer": self.mixer.state_dict()}, path
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.qs.load_state_dict(ckpt["qs"])
        self.mixer.load_state_dict(ckpt["mixer"])
        self.qs_target.load_state_dict(self.qs.state_dict())
        self.mixer_target.load_state_dict(self.mixer.state_dict())
