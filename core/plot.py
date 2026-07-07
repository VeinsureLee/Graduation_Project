"""训练曲线绘图：episode reward + steps per episode。"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["axes.unicode_minus"] = False


def plot_training(history: list[dict], out_dir: Path) -> None:
    if not history:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = [h["episode"] for h in history]
    rewards = [h["reward"] for h in history]
    steps = [h["steps"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(eps, rewards, label="episode reward")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_title("Training reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reward.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(eps, steps, label="steps")
    ax.set_xlabel("episode")
    ax.set_ylabel("steps")
    ax.set_title("Steps per episode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "steps.png", dpi=120)
    plt.close(fig)
