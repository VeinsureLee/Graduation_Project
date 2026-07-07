"""核心模块：训练器、测试器、绘图工具。"""
from core.plot import plot_training
from core.tester import TestResult, test
from core.trainer import TrainResult, train

__all__ = ["train", "test", "plot_training", "TrainResult", "TestResult"]
