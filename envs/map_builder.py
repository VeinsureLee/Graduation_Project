"""地图加载与构建：从 yml 读取房间/门/墙定义，生成 numpy 网格。

网格编码：
    0 = 空地（可通行）
    1 = 墙
    2 = 门（可通行，仅用于可视化区分）
"""
from typing import Any

import numpy as np
import yaml

from utils.paths import map_path

EMPTY = 0
WALL = 1
DOOR = 2


def load_map_spec(name: str) -> dict[str, Any]:
    """加载地图 yml，返回 `map` 段下的 dict。"""
    with open(map_path(name), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["map"]


def build_grid_array(spec: dict) -> np.ndarray:
    """根据 spec 构建网格：先填墙，再按房间挖空，最后开门。

    房间边界由墙包围（top_left 起 size 大小的矩形，最外一圈是墙）。
    门在 doors 列表中给出坐标，覆盖任何已有的墙。
    """
    rows, cols = spec["size"]
    grid = np.full((rows, cols), WALL, dtype=np.int8)

    # 房间内部置空（保留四周一圈墙）
    for room in spec["rooms"]:
        r0, c0 = room["top_left"]
        h, w = room["size"]
        r1 = min(r0 + h, rows)
        c1 = min(c0 + w, cols)
        # 内部空地：[r0+1, r1-1) × [c0+1, c1-1)
        if r1 - 1 > r0 + 1 and c1 - 1 > c0 + 1:
            grid[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1] = EMPTY

    # 门：在指定位置打开通道
    for door in spec["doors"]:
        r, c = door["position"]
        grid[r, c] = DOOR
    return grid
