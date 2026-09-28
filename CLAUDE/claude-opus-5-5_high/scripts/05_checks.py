"""Stage 5: sanity checks on the held-out test result.

1. leakage invariance: deleting every 4-7 e- cluster from a test image leaves its score map unchanged
2. area vs location: random-mask and shifted-mask controls at the selected threshold
3. pathological image: leave-one-acquisition-out range of r_B on the test block
4. quadrant dominance: per-quadrant metrics; global metrics without S101 HDU0
5. charge-window generalisation: r_B for other cluster-charge windows under the same mask
6. model calibration: predicted vs observed 4-7 e- counts in bins of predicted rate
Writes results/checks.json and results/checks_by_quadrant.csv.
"""
import json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd  # noqa: E402
from skbg import cache, config, dataset, features, metrics, pipeline, stacks  # noqa: E402


def leakage_invariance(cfg, spec, mdl, imgs=(105, 120, 128)):
    cd = config.cache_dir(cfg)
    recs = [r for r in dataset.records(cfg) if r["usable"] and r["img"] in imgs]
    worst = 0.0
    n_removed = 0
    for r in recs:
        z = cache.load_image(cd, r)
        for h in range(4):
            qi = stacks.QUADRANTS.index((r["module"], h))
            st = pipeline._static(cfg, spec, ["f0", "f1", "f2", "f3"], r["module"], h)
            arr, dyn = pipeline._dynamic(z, h, spec)
            eta = mdl.log_rate_maps({**st, **dyn}, qi, features.SHAPE)
            roi = (arr["Q"] >= 4) & (arr["Q"] <= 7)
            kill = np.isin(arr["lab"], np.flatnonzero(roi) + 1)
            z2 = dict(z)
            keep_pix = ~kill[z[f"pix{h}_r"], z[f"pix{h}_c"]]
            for k in ("r", "c", "n"):
                z2[f"pix{h}_{k}"] = z[f"pix{h}_{k}"][keep_pix]
            for k in ("Q", "npix", "rc", "cc", "qmax", "r0", "r1", "c0", "c1"):
                z2[f"cl{h}_{k}"] = z[f"cl{h}_{k}"][~roi]
            _, dyn2 = pipeline._dynamic(z2, h, spec)
            eta2 = mdl.log_rate_maps({**st, **dyn2}, qi, features.SHAPE)
            worst = max(worst, float(np.abs(eta - eta2).max()))
            n_removed += int(roi.sum())
    return {"images": list(imgs), "roi_clusters_removed": n_removed, "max_abs_score_change": worst}


def main():
    cfg = config.load_config(); rd = config.results_dir(); fd = config.cache_dir(cfg) / "final"
    fin = json.loads((rd / "final_test_metrics.json").read_text())
    name = fin["config"]
    with np.load(fd / f"test_{name}.npz") as zz:
        z = {k: zz[k] for k in zz.files}
    with open(fd / "model.pkl", "rb") as fh:
        m = pickle.load(fh)
    thr_idx = int(round((fin["log_rate_threshold"] - pipeline.ETA_EDGES[0]) / pipeline.ETA_STEP)) - 1
    out = {"config": name}

    out["leakage_invariance"] = leakage_invariance(cfg, m["spec"], m["model"])

    sh = z["shift"]  # E_keep, N_keep, E_keep_shift, N_keep_shift, E_tot, N_tot
    Et, Nt = sh[:, 4].sum(), sh[:, 5].sum()
    f_m, r_m = sh[:, 0].sum() / Et, (sh[:, 1].sum() / sh[:, 0].sum()) / (Nt / Et)
    f_s, r_s = sh[:, 2].sum() / Et, (sh[:, 3].sum() / sh[:, 2].sum()) / (Nt / Et)
    cr = metrics.curve(z["Er"].sum(0), z["Nr"].sum(0))
    ir = metrics.index_at_fexp(cr, f_m)
    out["area_controls"] = {"method": {"f_exp": f_m, "r_B": r_m},
                            "shifted_mask": {"f_exp": f_s, "r_B": r_s},
                            "random_mask_same_f": {"f_exp": float(cr["f_exp"][ir]), "r_B": float(cr["r_B"][ir])}}

    cE = np.cumsum(z["E"], 1)[:, thr_idx]; cN = np.cumsum(z["N"], 1)[:, thr_idx]
    tE, tN = z["E"].sum(1), z["N"].sum(1)
    loo = []
    for i in np.unique(z["img"]):
        s = z["img"] != i
        f = cE[s].sum() / tE[s].sum(); r = (cN[s].sum() / cE[s].sum()) / (tN[s].sum() / tE[s].sum())
        loo.append((int(i), f, r, f / r))
    loo = np.array(loo)
    out["leave_one_acquisition_out"] = {"r_B_min": loo[:, 2].min(), "r_B_max": loo[:, 2].max(),
                                        "F_min": loo[:, 3].min(), "F_max": loo[:, 3].max(),
                                        "most_influential_img": int(loo[np.argmax(np.abs(loo[:, 2] - fin["r_B"])), 0])}
    per_acq = []
    for i in np.unique(z["img"]):
        s = z["img"] == i
        per_acq.append({"img": int(i), "f_exp": cE[s].sum() / tE[s].sum(),
                        "r_B": (cN[s].sum() / cE[s].sum()) / (tN[s].sum() / tE[s].sum()) if tN[s].sum() else np.nan,
                        "N_before": int(tN[s].sum()), "N_after": int(cN[s].sum())})
    pd.DataFrame(per_acq).to_csv(rd / "checks_by_acquisition.csv", index=False)

    rows = []
    for qi, (mod, h) in enumerate(stacks.QUADRANTS):
        s = z["q"] == qi
        rows.append({"module": mod, "hdu": h, "E_before": tE[s].sum(), "N_before": int(tN[s].sum()),
                     "E_after": cE[s].sum(), "N_after": int(cN[s].sum()),
                     "f_exp": cE[s].sum() / tE[s].sum(), "r_B": (cN[s].sum() / cE[s].sum()) / (tN[s].sum() / tE[s].sum())})
    dq = pd.DataFrame(rows); dq["F_equal"] = dq.f_exp / dq.r_B
    dq.to_csv(rd / "checks_by_quadrant.csv", index=False)
    s = z["q"] != 0
    f7 = cE[s].sum() / tE[s].sum(); r7 = (cN[s].sum() / cE[s].sum()) / (tN[s].sum() / tE[s].sum())
    c7 = metrics.curve(z["E"][s].sum(0), z["N"][s].sum(0))
    for mod in ("S101", "S99"):
        sm = np.isin(z["q"], [i for i, (mm, _) in enumerate(stacks.QUADRANTS) if mm == mod])
        out[f"module_{mod}"] = {"f_exp": cE[sm].sum() / tE[sm].sum(),
                                "r_B": (cN[sm].sum() / cE[sm].sum()) / (tN[sm].sum() / tE[sm].sum())}
    out["without_S101_HDU0"] = {"same_threshold": {"f_exp": f7, "r_B": r7, "F_equal": f7 / r7},
                                "curve_r_B_at_f": {str(f): float(c7["r_B"][metrics.index_at_fexp(c7, f)]) for f in (0.8, 0.9, 0.95, 0.99)}}

    spec_h = z["spec"]  # (Q bin, eta bin)
    kept = spec_h[:, : thr_idx + 1].sum(1); allq = spec_h.sum(1)
    E_b, E_a = z["E"].sum(), cE.sum()
    win = {}
    for lo, hi in ((1, 1), (2, 3), (4, 7), (8, 20), (21, 62)):
        nb, na = allq[lo:hi + 1].sum(), kept[lo:hi + 1].sum()
        win[f"{lo}-{hi}"] = {"N_before": int(nb), "N_after": int(na), "r_B": float((na / E_a) / (nb / E_b))}
    out["charge_windows_same_mask"] = win

    centers = pipeline.ETA_EDGES[:-1] + pipeline.ETA_STEP / 2
    pred = (z["E"].sum(0) * np.exp(centers)); obs = z["N"].sum(0)
    grp = np.floor(centers).astype(int)
    cal = []
    for g in np.unique(grp):
        s = grp == g
        if z["E"].sum(0)[s].sum() > 0:
            cal.append({"log_rate_bin": int(g), "E": float(z["E"].sum(0)[s].sum()),
                        "predicted": float(pred[s].sum()), "observed": int(obs[s].sum())})
    out["calibration"] = cal
    out["calibration_total"] = {"predicted": float(pred.sum()), "observed": int(obs.sum())}
    (rd / "checks.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "calibration"}, indent=1, default=float))
    print(dq.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
