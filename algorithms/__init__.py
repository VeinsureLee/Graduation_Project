"""算法注册表。新增算法只需在此添加一行，无需修改 trainer。"""
from algorithms.base import BaseAlgorithm
from algorithms.dqn import DQN
from algorithms.madqn import MADQN
from algorithms.mappo import MAPPO
from algorithms.ppo import PPO
from algorithms.qmix import QMIX
from algorithms.shared_madqn import SharedMADQN
from algorithms.vdn import VDN

ALGORITHM_REGISTRY: dict[str, type[BaseAlgorithm]] = {
    # value-based
    "dqn": DQN,
    "madqn": MADQN,
    "shared_madqn": SharedMADQN,
    "vdn": VDN,
    "qmix": QMIX,
    # policy-based
    "ppo": PPO,
    "mappo": MAPPO,
}


def build_algorithm(name: str, env, cfg: dict) -> BaseAlgorithm:
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm: {name!r}. Available: {sorted(ALGORITHM_REGISTRY)}"
        )
    return ALGORITHM_REGISTRY[name](env, cfg)


__all__ = ["BaseAlgorithm", "ALGORITHM_REGISTRY", "build_algorithm"]
