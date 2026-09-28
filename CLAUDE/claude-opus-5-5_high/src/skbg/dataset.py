"""Dataset-level helpers: records, quality cut, acquisition-level split."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, io


def calibration_table(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(config.cache_dir(cfg) / "calibration_table.csv")


def quality_flags(cal: pd.DataFrame, max_gain_dev: float = 0.05) -> pd.DataFrame:
    """Flag quadrant-images whose fitted gain deviates > max_gain_dev from the
    module/HDU median. Decided from calibration alone (blind to the ROI)."""
    cal = cal.copy()
    med = cal.groupby(["module", "hdu"]).gain_adu.transform("median")
    cal["gain_dev"] = cal.gain_adu / med - 1.0
    cal["bad_quality"] = cal.gain_dev.abs() > max_gain_dev
    return cal


def records(cfg: dict) -> list[dict]:
    """File records with the acquisition-level quality decision attached.
    A file is dropped if any of its HDUs fails the quality cut."""
    recs = io.discover(config.data_dir(cfg), cfg["data"]["file_glob"])
    cal = quality_flags(calibration_table(cfg), cfg.get("quality", {}).get("max_gain_dev", 0.05))
    bad = set(cal.loc[cal.bad_quality, "file"])
    for r in recs:
        r["usable"] = r["file"] not in bad
    return recs


def split_of(img: int, cfg: dict) -> str:
    return "test" if img >= cfg["split"]["test_first_img"] else "dev"


def dev_folds(cfg: dict, imgs: list[int]) -> dict[int, int]:
    """Contiguous (chronological) blocks of development acquisitions -> fold id."""
    dev = sorted(i for i in set(imgs) if split_of(i, cfg) == "dev")
    blocks = np.array_split(np.array(dev), cfg["split"]["n_cv_folds"])
    return {int(i): k for k, b in enumerate(blocks) for i in b}
