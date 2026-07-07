"""路径工具：基于 pyproject.toml 定位项目根目录。"""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    """向上查找包含 pyproject.toml 的目录作为项目根。"""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Cannot locate project root (no pyproject.toml found)")


def config_path(name: str = "config") -> Path:
    return project_root() / "config" / f"{name}.yml"


def map_path(name: str) -> Path:
    return project_root() / "maps" / f"{name}.yml"


def experiments_dir() -> Path:
    d = project_root() / "experiments"
    d.mkdir(exist_ok=True)
    return d
