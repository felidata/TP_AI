"""Cross-check (labeled external comparison): our per-HDU gain/noise fits vs the GAIN/NOISE_E
headers written by the pre-existing MOSKITA calibration (outputs/calibrated in the parent project).
Needs MOSKITA_CALIBRATED_DIR; writes results/check_calibration_vs_moskita.csv. Not part of run_all.sh."""
import os, sys, glob
from pathlib import Path
sys.path.insert(0, 'src')
import pandas as pd, numpy as np
from astropy.io import fits
from skbg import config, dataset
cfg = config.load_config()
mdir = Path(os.environ.get("MOSKITA_CALIBRATED_DIR", config.RUN_DIR / "../../../outputs/calibrated")).resolve()
cal = dataset.calibration_table(cfg)
rows = []
for f in sorted(mdir.glob("cal_*.fits"))[::7]:
    with fits.open(f, memmap=True) as h:
        for i in range(min(4, len(h))):
            hd = h[i].header
            mine = cal[(cal.file == f.name[4:]) & (cal.hdu == i)]
            if len(mine) and "GAIN" in hd:
                rows.append({"file": f.name[4:], "hdu": i, "gain_moskita": hd["GAIN"], "gain_ours": float(mine.gain_adu.iloc[0]),
                             "noise_moskita": hd.get("NOISE_E"), "noise_ours": float(mine.noise_e.iloc[0])})
d = pd.DataFrame(rows)
d["gain_rel_diff"] = d.gain_ours / d.gain_moskita - 1
d.to_csv(config.results_dir() / "check_calibration_vs_moskita.csv", index=False)
print(len(d), "HDU-images compared from", mdir)
print(d[["gain_rel_diff"]].describe().T.to_string()); print((d.noise_ours - d.noise_moskita).describe().to_string())
