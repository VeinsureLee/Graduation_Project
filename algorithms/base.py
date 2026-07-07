"""所有算法的抽象基类。统一接口便于训练/测试循环复用。"""
from abc import ABC, abstractmethod
from typing import Any


class BaseAlgorithm(ABC):
    """RL 算法基类。

    所有算法需实现 take_action / update / save / load 四个方法。
    is_on_policy 与 required_buffer 用于让 trainer 决定训练流程。
    """

    @abstractmethod
    def take_action(
        self, states: dict[int, Any], explore: bool = True
    ) -> dict[int, int]:
        """根据当前观测返回每个智能体的动作。"""

    @abstractmethod
    def update(self, *args, **kwargs) -> dict[str, float]:
        """执行一次参数更新，返回训练指标字典（至少含 loss）。"""

    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型权重。"""

    @abstractmethod
    def load(self, path: str) -> None:
        """加载模型权重。"""

    @property
    def is_on_policy(self) -> bool:
        """on/off-policy 标识；trainer 用此选择训练流程。"""
        return False

    def required_buffer(self) -> str:
        """返回所需 buffer 类型：'single' | 'per_agent' | 'joint' | 'none'。"""
        return "single"
