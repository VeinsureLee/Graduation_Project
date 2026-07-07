"""测试单次 episode 并生成 gif/png 可视化。"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from algorithms.base import BaseAlgorithm


@dataclass
class TestResult:
    success: bool
    steps: int
    total_reward: float
    gif_path: str
    png_path: str


def _render_to_pil(frame: np.ndarray, scale: int = 20) -> Image.Image:
    img = Image.fromarray(frame)
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.NEAREST)


def test(
    algo: BaseAlgorithm, env, cfg: dict, out_dir: Path, max_steps: int = 500
) -> TestResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    states = env.reset()
    frames: list[np.ndarray] = [env.render()]
    total_reward = 0.0
    success = False
    step = 0

    for step in range(max_steps):
        actions = algo.take_action(states, explore=False)
        states, rewards, dones, _ = env.step(actions)
        frames.append(env.render())
        total_reward += sum(rewards.values())
        if all(dones.values()):
            success = True
            break

    gif_path = out_dir / "nav.gif"
    png_path = out_dir / "nav_final.png"
    pil_frames = [_render_to_pil(f) for f in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=200,
        loop=0,
    )
    pil_frames[-1].save(png_path)
    return TestResult(
        success=success,
        steps=step + 1,
        total_reward=total_reward,
        gif_path=str(gif_path),
        png_path=str(png_path),
    )
