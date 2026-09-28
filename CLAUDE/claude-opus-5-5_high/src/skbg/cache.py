"""Derived per-image cache: calibration, sparse charged pixels and cluster catalogues.

For each FITS file, one ``<stem>.npz`` holds, for each HDU h:
  pix{h}_r, pix{h}_c, pix{h}_n : active-region coordinates and integer charge of pixels with n >= 1
  cl{h}_<field>                : cluster catalogue (see events.reconstruct)
and a JSON sidecar with calibration fits and timing. Raw FITS are never modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import calib, events, io


def cache_path(cache_dir: Path, rec: dict) -> Path:
    return cache_dir / "images" / (Path(rec["file"]).stem + ".npz")


def process_file(rec: dict, cfg: dict, cache_dir: Path, overwrite: bool = False) -> dict:
    out = cache_path(cache_dir, rec)
    side = out.with_suffix(".json")
    if out.exists() and side.exists() and not overwrite:
        return json.loads(side.read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    data, hdrs = io.read_hdus(rec["path"])
    arrays = {}
    meta = {"file": rec["file"], "module": rec["module"], "img": rec["img"],
            "timing": io.read_timing(hdrs[0]), "hdus": []}
    for h, img in enumerate(data):
        q, n, fit = calib.calibrate_quadrant(img, cfg)
        nz = np.nonzero(n)
        arrays[f"pix{h}_r"] = nz[0].astype(np.uint16)
        arrays[f"pix{h}_c"] = nz[1].astype(np.uint16)
        arrays[f"pix{h}_n"] = n[nz].astype(np.int32)
        _, cl = events.reconstruct(n, cfg["events"]["connectivity"])
        for k, v in cl.items():
            arrays[f"cl{h}_{k}"] = v.astype(np.float32 if k in ("rc", "cc") else np.int32)
        Q = cl["Q"]
        fit.update({"hdu": h, "chid": int(hdrs[h].get("CHID", -1)),
                    "n_charged_pix": int(len(nz[0])), "n_1e_pix": int((n == 1).sum()),
                    "n_clusters": int(len(Q)),
                    "n_roi": int(((Q >= cfg["events"]["roi_min_e"]) & (Q <= cfg["events"]["roi_max_e"])).sum()),
                    "n_hee": int((Q >= cfg["events"]["hee_min_e"]).sum()),
                    "active_shape": list(n.shape)})
        meta["hdus"].append(fit)
    np.savez_compressed(out, **arrays)
    side.write_text(json.dumps(meta, indent=1))
    return meta


def load_image(cache_dir: Path, rec: dict) -> dict:
    with np.load(cache_path(cache_dir, rec)) as z:
        return {k: z[k] for k in z.files}


def load_meta(cache_dir: Path, rec: dict) -> dict:
    return json.loads(cache_path(cache_dir, rec).with_suffix(".json").read_text())
