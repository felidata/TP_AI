"""Calibration check: position of the n = 1..8 e- single-pixel peaks in calibrated charge q.

For a fixed sample of quadrant-images (every 9th acquisition, all 8 quadrants), the median of q
in the window n +- 0.5 is compared with n. Writes results/check_linearity.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd  # noqa: E402
from skbg import calib, config, dataset, io  # noqa: E402


def main():
    cfg = config.load_config()
    recs = [r for r in dataset.records(cfg) if r["usable"] and r["img"] % 9 == 0]
    rows = []
    for r in recs:
        data, _ = io.read_hdus(r["path"])
        for h in range(4):
            q, n, fit = calib.calibrate_quadrant(data[h], cfg)
            q = q.ravel()
            for k in range(1, 9):
                s = q[(q > k - 0.5) & (q < k + 0.5)]
                rows.append({"module": r["module"], "img": r["img"], "hdu": h, "n": k, "count": len(s),
                             "median_q": float(np.median(s)) if len(s) > 20 else np.nan})
    d = pd.DataFrame(rows)
    d["bias_e"] = d.median_q - d.n
    d.to_csv(config.results_dir() / "check_linearity.csv", index=False)
    print(d.groupby("n").bias_e.describe().round(3).to_string())


if __name__ == "__main__":
    main()
