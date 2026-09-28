"""Configuration loading and path resolution (no user-specific absolute paths)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

RUN_DIR = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else RUN_DIR / "configs" / "default.yaml"
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path)
    return cfg


def data_dir(cfg: dict) -> Path:
    """SKIPPER_DATA_DIR wins; otherwise the config path relative to the run dir."""
    env = os.environ.get("SKIPPER_DATA_DIR")
    p = Path(env) if env else (RUN_DIR / cfg["data"]["data_dir"])
    p = p.resolve()
    if not p.is_dir():
        raise FileNotFoundError(
            f"Skipper data directory not found: {p}. Set SKIPPER_DATA_DIR=/path/to/pp_proc")
    return p


def cache_dir(cfg: dict) -> Path:
    env = os.environ.get("SKBG_CACHE_DIR")
    p = Path(env) if env else (RUN_DIR / cfg["data"]["cache_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir() -> Path:
    p = RUN_DIR / "results"
    p.mkdir(exist_ok=True)
    return p


def figures_dir() -> Path:
    p = RUN_DIR / "figures"
    p.mkdir(exist_ok=True)
    return p
