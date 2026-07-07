"""地图可视化工具：把 maps/<name>.yml 渲染成 PNG。

用法::
    python scripts/visualize_map.py                       # 静态户型骨架 -> maps/preview/default.png
    python scripts/visualize_map.py --map default --scale 30
    python scripts/visualize_map.py --samples 6           # 6 组随机起终点 -> maps/preview/default_samples.png
    python scripts/visualize_map.py --out /tmp/map.png

输出说明：
    - 黑色      墙（保持 1 格厚度）
    - 白色      空地
    - 白色通道  门（叠加建筑平面图开门符号：细线门扇 + 四分之一圆弧）
    - 蓝/红圆点 智能体起点（多个 agent 按颜色区分）
    - 绿色方块  目标点

默认（不带 --samples）只画户型骨架（墙/空地/门，不含起终点）。
带 --samples N 时，用不同子种子随机采样 N 组起点/终点，拼成一张蒙太奇，
直观展示「起终点每次都在随机变化且都合法连通」。
"""
import argparse
from math import ceil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envs.indoor_env import IndoorEnv
from envs.map_builder import DOOR, EMPTY, WALL, build_grid_array, load_map_spec
from utils.paths import project_root


def _build_env(map_name: str, random_spawn: bool, seed: int = 0) -> IndoorEnv:
    cfg = {
        "env": {
            "map_file": map_name,
            "observation_mode": "full",
            "partial_view_size": 7,
            "random_spawn": random_spawn,
            "reward_mode": "independent",
            "reward_goal": 10.0,
            "reward_step": -0.01,
            "reward_collision": -1.0,
            "reward_team_bonus": 5.0,
        },
        "seed": seed,
    }
    env = IndoorEnv(cfg)
    env.reset()
    return env


def _grid_rgb(grid: np.ndarray) -> np.ndarray:
    """只渲染地图骨架（墙/空地/门），不含智能体与目标。

    黑白底色：空地=白，墙=黑；门格渲染为白色通道（开门符号由 _draw_door_symbols 叠加）。
    """
    rgb = np.zeros((*grid.shape, 3), dtype=np.uint8)
    rgb[grid == EMPTY] = (255, 255, 255)
    rgb[grid == WALL] = (0, 0, 0)
    rgb[grid == DOOR] = (255, 255, 255)
    return rgb


def _draw_door_symbols(img: Image.Image, grid: np.ndarray, scale: int) -> None:
    """在缩放后的图像上叠加建筑平面图风格的开门符号。

    每扇门：白色通道保持不变，叠加细线（门扇折叠后靠墙位置）+ 四分之一圆弧（旋转扫过范围）。
    线与弧均放在远离相邻实墙的那侧，避免视觉上遮挡通道。

    竖向墙（门洞左右通行，上下邻格为墙）：
      铰链在右上角 (x1,y0)，门扇沿右边竖线，弧在门格内 90°→180°。
    横向墙（门洞上下通行，左右邻格为墙）：
      铰链在左下角 (x0,y1)，门扇沿下边横线，弧在门格内 270°→360°。
    """
    draw = ImageDraw.Draw(img)
    rows, cols = grid.shape
    lw = max(1, scale // 12)

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != DOOR:
                continue

            # 上下邻格均为墙 → 竖向墙（门洞左右通行）
            in_vertical_wall = (
                0 < r < rows - 1
                and grid[r - 1, c] == WALL
                and grid[r + 1, c] == WALL
            )

            x0, y0 = c * scale, r * scale
            x1, y1 = x0 + scale, y0 + scale

            if in_vertical_wall:
                # 门扇：右边竖线，铰链在右上角 (x1,y0)
                # 弧：圆心 (x1,y0)，90°→180°，扫过门格内左下区域
                draw.line([(x1, y0), (x1, y1)], fill=(0, 0, 0), width=lw)
                draw.arc(
                    [x1 - scale, y0 - scale, x1 + scale, y0 + scale],
                    start=90,
                    end=180,
                    fill=(0, 0, 0),
                    width=lw,
                )
            else:
                # 门扇：下边横线，铰链在左下角 (x0,y1)
                # 弧：圆心 (x0,y1)，270°→360°，扫过门格内右上区域
                draw.line([(x0, y1), (x1, y1)], fill=(0, 0, 0), width=lw)
                draw.arc(
                    [x0 - scale, y1 - scale, x0 + scale, y1 + scale],
                    start=270,
                    end=360,
                    fill=(0, 0, 0),
                    width=lw,
                )


def _scale_img(rgb: np.ndarray, scale: int) -> Image.Image:
    img = Image.fromarray(rgb, mode="RGB")
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.NEAREST)


def _label_font(scale: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("arial.ttf", size=max(10, scale // 2))
    except OSError:
        return ImageFont.load_default()


def _draw_labels(img: Image.Image, env: IndoorEnv, scale: int) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = _label_font(scale)
    for i, (r, c) in enumerate(env.agent_positions):
        x, y = c * scale + scale // 2, r * scale + scale // 2
        draw.text((x - scale // 4, y - scale // 4), f"A{i}", fill="white", font=font)
    for i, (r, c) in enumerate(env.goals):
        x, y = c * scale + scale // 2, r * scale + scale // 2
        draw.text((x - scale // 4, y - scale // 4), f"G{i}", fill="black", font=font)
    return img


def _sample_image(env: IndoorEnv, scale: int, show_labels: bool) -> Image.Image:
    """渲染一帧带起终点的图（用于随机采样展示）。"""
    img = _scale_img(env.render(), scale)
    _draw_door_symbols(img, env.grid, scale)
    if show_labels:
        img = _draw_labels(img, env, scale)
    return img


def _montage(images: list[Image.Image], cols: int, pad: int = 8) -> Image.Image:
    rows = ceil(len(images) / cols)
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    canvas = Image.new(
        "RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), (255, 255, 255)
    )
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        canvas.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    return canvas


def _summary_strings(env: IndoorEnv) -> list[str]:
    spec = env.spec
    return [
        f"Map size      : {spec['size'][0]} x {spec['size'][1]}",
        f"Rooms         : {len(spec['rooms'])}",
        f"Doors         : {len(spec['doors'])}",
        f"Agents        : {spec['num_agents']}",
        f"Floor cells   : {(env.grid == EMPTY).sum()}",
        f"Walls / cells : {(env.grid == WALL).sum()} / {env.grid.size}",
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="default", help="Map file name (without .yml)")
    p.add_argument("--scale", type=int, default=30, help="Pixel scale per grid cell")
    p.add_argument("--samples", type=int, default=0, help="Render N random start/goal samples")
    p.add_argument("--seed", type=int, default=0, help="Base seed for random samples")
    p.add_argument("--cols", type=int, default=3, help="Columns in the sample montage")
    p.add_argument("--out", default=None, help="Output PNG path")
    p.add_argument("--no-labels", action="store_true", help="Disable A0/G0 text labels")
    args = p.parse_args()

    preview_dir = project_root() / "maps" / "preview"

    if args.samples > 0:
        # 随机起终点：每个样本用不同子种子，证明每次都在变化且合法。
        env = _build_env(args.map, random_spawn=True, seed=args.seed)
        sample_scale = max(6, args.scale // 2)
        images = []
        for i in range(args.samples):
            env.reset(seed=args.seed + i)
            images.append(
                _sample_image(env, sample_scale, show_labels=not args.no_labels)
            )
        canvas = _montage(images, cols=args.cols)
        out_path = Path(args.out) if args.out else preview_dir / f"{args.map}_samples.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
        print(f"Saved: {out_path}  ({args.samples} random start/goal samples)")
        for line in _summary_strings(env):
            print(f"  {line}")
        return

    # 默认：静态户型骨架（不含起终点）。
    env = _build_env(args.map, random_spawn=False, seed=args.seed)
    img = _scale_img(_grid_rgb(env.grid), args.scale)
    _draw_door_symbols(img, env.grid, args.scale)
    out_path = Path(args.out) if args.out else preview_dir / f"{args.map}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Saved: {out_path}  (static floor plan)")
    for line in _summary_strings(env):
        print(f"  {line}")


if __name__ == "__main__":
    main()
