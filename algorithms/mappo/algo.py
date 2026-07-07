"""MAPPO (Multi-Agent PPO)：所有 agent 共享 actor 参数 + 共享中心化 critic。

CTDE 范式：
    actor 输入：单个 agent 的局部观测 -> 离散动作分布
    critic 输入：所有 agent 观测的拼接（全局状态）-> 全局价值
所有 agent 共享 actor 参数（同质 agent 假设）；critic 仅在训练时使用。
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from algorithms.base import BaseAlgorithm
from algorithms.common.networks import Actor, Critic


class MAPPO(BaseAlgorithm):
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
        self.global_dim = self.state_dim * self.num_agents

        self.actor = Actor(self.state_dim, algo_cfg["hidden_dim"], self.n_actions).to(
            self.device
        )
        self.critic = Critic(self.global_dim, algo_cfg["hidden_dim"]).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr
        )

    @property
    def is_on_policy(self) -> bool:
        return True

    def required_buffer(self) -> str:
        return "none"

    def take_action(self, states, explore=True):
        actions = {}
        for i in range(self.num_agents):
            s = torch.from_numpy(states[i]).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.actor(s)
            if explore:
                dist = Categorical(logits=logits)
                actions[i] = int(dist.sample().item())
            else:
                actions[i] = int(logits.argmax(dim=1).item())
        return actions

    def evaluate_joint(self, joint_states: dict[int, np.ndarray]):
        """返回 {agent_id: (action, log_prob)} 和 value (基于全局状态)."""
        global_s = np.concatenate([joint_states[i] for i in range(self.num_agents)])
        per_agent = {}
        for i in range(self.num_agents):
            s = torch.from_numpy(joint_states[i]).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.actor(s)
            dist = Categorical(logits=logits)
            a = dist.sample()
            per_agent[i] = (int(a.item()), float(dist.log_prob(a).item()))
        with torch.no_grad():
            g = torch.from_numpy(global_s).float().unsqueeze(0).to(self.device)
            value = float(self.critic(g).item())
        return per_agent, value, global_s

    def update(self, rollout: dict) -> dict[str, float]:
        """rollout: per-agent flattened trajectories + global states.

        Keys:
          local_states (T*N, state_dim)
          actions      (T*N,)
          log_probs    (T*N,)
          advantages   (T*N,) -- shared from team return
          returns      (T*N,)
          global_states (T*N, global_dim)
        """
        local_s = torch.from_numpy(rollout["local_states"]).float().to(self.device)
        actions = torch.from_numpy(rollout["actions"]).long().to(self.device)
        old_log_probs = torch.from_numpy(rollout["log_probs"]).float().to(self.device)
        returns = torch.from_numpy(rollout["returns"]).float().to(self.device)
        advantages = torch.from_numpy(rollout["advantages"]).float().to(self.device)
        global_s = torch.from_numpy(rollout["global_states"]).float().to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = local_s.size(0)
        losses = []
        for _ in range(self.ppo_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                b = idx[start : start + self.batch_size]
                logits = self.actor(local_s[b])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[b])
                entropy = dist.entropy().mean()
                values = self.critic(global_s[b])

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
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
