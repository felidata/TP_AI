"""Per-fold stacked count and exposure maps (written for this project).

Group ids: dev folds 0..K-1 and 'test'. Stacks are additive, so the static
maps for any training set are sums over its groups. Test stacks are built only
for descriptive baseline figures; no model or threshold ever reads them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import cache, config, dataset, exposure
from .features import SHAPE

SOURCES = {"roi": (4, 7), "side": (2, 3), "one": (1, 1), "mid": (8, 99)}
QUADRANTS = [(m, h) for m in ("S101", "S99") for h in range(4)]


def group_of(rec: dict, cfg: dict, folds: dict) -> str:
    return "test" if dataset.split_of(rec["img"], cfg) == "test" else f"f{folds[rec['img']]}"


def build(cfg: dict) -> Path:
    cd = config.cache_dir(cfg)
    out = cd / "stacks"
    out.mkdir(exist_ok=True)
    recs = [r for r in dataset.records(cfg) if r["usable"]]
    folds = dataset.dev_folds(cfg, [r["img"] for r in recs])
    groups = sorted({group_of(r, cfg, folds) for r in recs})
    kmap = exposure.exposure_map_days({"exposure_s": 0.0, "readout_s": 86400.0, "nrow": 515, "ncol": 3200},
                                      tuple(cfg["geometry"]["active_rows"]), tuple(cfg["geometry"]["active_cols"]))
    for g in groups:
        acc = {(m, h): {s: np.zeros(SHAPE, np.int32) for s in SOURCES} for m, h in QUADRANTS}
        tsum = {m: 0.0 for m in ("S101", "S99")}
        esum_exp = {m: 0.0 for m in ("S101", "S99")}
        nimg = {m: 0 for m in ("S101", "S99")}
        for r in recs:
            if group_of(r, cfg, folds) != g:
                continue
            z = cache.load_image(cd, r)
            t = cache.load_meta(cd, r)["timing"]
            tsum[r["module"]] += t["readout_s"] / 86400.0
            esum_exp[r["module"]] += t["exposure_s"] / 86400.0
            nimg[r["module"]] += 1
            for h in range(4):
                Q = z[f"cl{h}_Q"]
                rc = np.rint(z[f"cl{h}_rc"]).astype(int)
                cc = np.rint(z[f"cl{h}_cc"]).astype(int)
                for s, (lo, hi) in SOURCES.items():
                    sel = (Q >= lo) & (Q <= hi)
                    np.add.at(acc[(r["module"], h)][s], (rc[sel], cc[sel]), 1)
        for (m, h), d in acc.items():
            E = (kmap * tsum[m] + esum_exp[m]).astype(np.float32)
            np.savez_compressed(out / f"{g}_{m}_{h}.npz", E=E, nimg=nimg[m], **d)
    return out


def load(cfg: dict, groups: list[str], m: str, h: int) -> dict:
    out = config.cache_dir(cfg) / "stacks"
    tot = None
    for g in groups:
        with np.load(out / f"{g}_{m}_{h}.npz") as z:
            d = {k: z[k].astype(np.float64) for k in z.files}
        tot = d if tot is None else {k: tot[k] + d[k] for k in tot}
    return tot
