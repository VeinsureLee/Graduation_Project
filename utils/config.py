"""配置加载器：读取单一 config.yml + 支持 CLI/API 覆盖。"""
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from utils.paths import config_path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载 YAML 配置。默认读取 config/config.yml。"""
    p = Path(path) if path else config_path()
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_overrides(base: dict, overrides: dict) -> dict:
    """递归合并 overrides 到 base 的副本，返回新 dict。"""
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_overrides(result[key], value)
        else:
            result[key] = value
    return result
