"""共享神经网络模块。

合并了原来分散在各算法目录下的 QNet/ActorCritic/Actor/Critic/QMixer，
消除重复代码。

网络清单：
    MlpQNet     — 通用 3 层 MLP (Linear→ReLU→Linear→ReLU→Linear)，用于 DQN/MADQN/SharedMADQN/VDN/QMIX
    ActorCritic — PPO 的 Actor-Critic 共用体网络
    Actor       — MAPPO 的 Actor 网络（局部观测 → 动作分布）
    Critic      — MAPPO 的 Critic 网络（全局状态 → 价值）
    QMixer      — QMIX 的单调混合网络（hypernetwork 生成非负权重）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpQNet(nn.Module):
    """通用 Q 网络：3 层全连接 MLP，中间 ReLU 激活。

    参数：
        state_dim: 观测（状态）维度
        hidden_dim: 隐藏层宽度
        n_actions: 动作空间大小
    """

    def __init__(self, state_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """PPO 的 Actor-Critic 网络：共享躯干 + 独立 actor/critic 头。

    参数：
        state_dim: 观测维度
        hidden_dim: 隐藏层宽度
        n_actions: 动作空间大小
    """

    def __init__(self, state_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


class Actor(nn.Module):
    """MAPPO 的 Actor 网络：局部观测 → 动作 logits。

    参数：
        state_dim: 单个 agent 的局部观测维度
        hidden_dim: 隐藏层宽度
        n_actions: 动作空间大小
    """

    def __init__(self, state_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    """MAPPO 的 Critic 网络：全局状态 → 标量价值。

    参数：
        global_dim: 全局状态维度（通常 = state_dim * num_agents）
        hidden_dim: 隐藏层宽度
    """

    def __init__(self, global_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class QMixer(nn.Module):
    """QMIX 的 Monotonic Mixer：用 hypernetwork 生成非负权重保证单调性。

    输入：
        agent_qs: (B, N)    各 agent 选定动作的 Q 值
        state:    (B, S)    全局状态（用各 agent 观测的拼接近似）

    输出：
        q_tot:    (B,)      联合 Q 值
    """

    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # hypernet：state → 权重（|w| 保证非负 → 单调性）
        self.hyper_w1 = nn.Linear(state_dim, num_agents * embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        bs = agent_qs.size(0)
        # 第一层
        w1 = torch.abs(self.hyper_w1(state)).view(bs, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(bs, 1, self.embed_dim)
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = F.elu(hidden)
        # 第二层
        w2 = torch.abs(self.hyper_w2(state)).view(bs, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(bs, 1, 1)
        y = torch.bmm(hidden, w2) + b2
        return y.view(bs)
