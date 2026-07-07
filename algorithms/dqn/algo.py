"""标准 DQN：epsilon-greedy 探索 + 目标网络 + 经验回放。

在多智能体环境中，本类只控制 agent 0，其他智能体随机动作。
作为入门理解 Q-learning 流程的首选算法。
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.base import BaseAlgorithm
from algorithms.common.networks import MlpQNet


class DQN(BaseAlgorithm):
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
        self.controlled_agent = 0

        self.q = MlpQNet(self.state_dim, algo_cfg["hidden_dim"], self.n_actions).to(
            self.device
        )
        self.q_target = deepcopy(self.q)
        for p in self.q_target.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)

    def required_buffer(self) -> str:
        return "single"

    def take_action(self, states, explore=True):
        actions = {}
        for agent_id in range(self.num_agents):
            if agent_id == self.controlled_agent:
                if explore and np.random.rand() < self.epsilon:
                    actions[agent_id] = int(np.random.randint(self.n_actions))
                else:
                    s = (
                        torch.from_numpy(states[agent_id])
                        .float()
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    with torch.no_grad():
                        q_values = self.q(s)
                    actions[agent_id] = int(q_values.argmax(dim=1).item())
            else:
                actions[agent_id] = int(np.random.randint(self.n_actions))
        return actions

    def update(self, batch) -> dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)

        q = self.q(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(next_states_t).max(dim=1)[0]
            target = rewards_t + self.gamma * q_next * (1 - dones_t)
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
