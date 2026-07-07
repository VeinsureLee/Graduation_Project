"""PPO (Proximal Policy Optimization)：单智能体 on-policy 策略梯度方法。

在多智能体环境中，本类只控制 agent 0，其他随机。
更新流程：收集 update_interval 步 → 计算 GAE → 多 epoch clipped surrogate 更新。
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from algorithms.base import BaseAlgorithm
from algorithms.common.networks import ActorCritic


class PPO(BaseAlgorithm):
    def __init__(self, env, cfg: dict):
        algo_cfg = cfg["algorithm"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = algo_cfg["gamma"]
        self.lr = algo_cfg["lr"]
        self.clip_eps = algo_cfg["clip_epsilon"]
        self.gae_lambda = algo_cfg["gae_lambda"]
        self.entropy_coef = algo_cfg["entropy_coef"]
        self.value_coef = algo_cfg["value_coef"]
        self.ppo_epochs = algo_cfg["ppo_epochs"]
        self.batch_size = algo_cfg["batch_size"]

        self.n_actions = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.num_agents = env.num_agents
        self.controlled_agent = 0

        self.net = ActorCritic(self.state_dim, algo_cfg["hidden_dim"], self.n_actions).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    @property
    def is_on_policy(self) -> bool:
        return True

    def required_buffer(self) -> str:
        return "none"

    def take_action(self, states, explore=True):
        actions = {}
        for agent_id in range(self.num_agents):
            if agent_id == self.controlled_agent:
                s = (
                    torch.from_numpy(states[agent_id])
                    .float()
                    .unsqueeze(0)
                    .to(self.device)
                )
                with torch.no_grad():
                    logits, _ = self.net(s)
                    if explore:
                        dist = Categorical(logits=logits)
                        actions[agent_id] = int(dist.sample().item())
                    else:
                        actions[agent_id] = int(logits.argmax(dim=1).item())
            else:
                actions[agent_id] = int(np.random.randint(self.n_actions))
        return actions

    def evaluate(self, state: np.ndarray):
        """返回 (action, log_prob, value)，用于 rollout 收集。"""
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(s)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def update(self, rollout: dict) -> dict[str, float]:
        """rollout: 包含 states, actions, log_probs, returns, advantages 的 dict。"""
        states = torch.from_numpy(rollout["states"]).float().to(self.device)
        actions = torch.from_numpy(rollout["actions"]).long().to(self.device)
        old_log_probs = torch.from_numpy(rollout["log_probs"]).float().to(self.device)
        returns = torch.from_numpy(rollout["returns"]).float().to(self.device)
        advantages = torch.from_numpy(rollout["advantages"]).float().to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = states.size(0)
        losses = []
        for _ in range(self.ppo_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                b = idx[start : start + self.batch_size]
                logits, values = self.net(states[b])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[b])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs[b])
                surr1 = ratio * advantages[b]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[b]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns[b])
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.item()))
        return {"loss": float(np.mean(losses))}

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))
