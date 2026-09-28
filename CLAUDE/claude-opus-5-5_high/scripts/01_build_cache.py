"""Stage 1: calibrate every quadrant, reconstruct clusters, write the derived cache.

Usage:  python scripts/01_build_cache.py [--config configs/default.yaml] [--workers 4] [--overwrite]
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from skbg import cache, config, io  # noqa: E402


def _work(args):
    rec, cfg, cdir, ow = args
    try:
        return cache.process_file(rec, cfg, cdir, ow)
    except Exception as exc:  # recorded, not hidden
        return {"file": rec["file"], "error": repr(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = config.load_config(a.config)
    dd, cdir = config.data_dir(cfg), config.cache_dir(cfg)
    recs = io.discover(dd, cfg["data"]["file_glob"])
    print(f"{len(recs)} FITS files in {dd}")
    t0 = time.time()
    with Pool(a.workers) as pool:
        metas = pool.map(_work, [(r, cfg, cdir, a.overwrite) for r in recs], chunksize=2)
    errs = [m for m in metas if "error" in m]
    rows = []
    for m in metas:
        if "error" in m:
            continue
        for h in m["hdus"]:
            row = {"file": m["file"], "module": m["module"], "img": m["img"], **m["timing"]}
            row.update({k: v for k, v in h.items() if k not in ("amps", "active_shape")})
            row.update({f"amp{i}": a_ for i, a_ in enumerate(h["amps"])})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(cdir / "calibration_table.csv", index=False)
    (cdir / "build_errors.json").write_text(json.dumps(errs, indent=1))
    print(f"done in {time.time()-t0:.1f}s; {len(df)} quadrant-images; {len(errs)} errors")


if __name__ == "__main__":
    main()
