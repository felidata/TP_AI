"""Stage 4: final model on all development acquisitions, evaluated once on the held-out test block.

Usage: python scripts/04_final.py [--workers 4] [--extra glm_static_only,glm_dynamic_only]
Reads results/cv/selection.json (stage 3b). Writes cache/final/*.npz,
results/final_test_metrics.json, results/operating_points.csv, results/final_coefficients.json.
--extra evaluates additional configs on the test block as *reported ablations* (never used for selection).
"""
import argparse, json, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, yaml  # noqa: E402
from skbg import config, metrics, pipeline, stacks  # noqa: E402

DEV = ["f0", "f1", "f2", "f3"]


def run(cfg, name, overrides, workers, keep_maps=None, control_threshold=None):
    spec = {**pipeline.default_feature_spec(cfg), **(overrides or {}), "control_threshold": control_threshold}
    mdl = pipeline.fit_model(cfg, spec, DEV, workers=workers)
    parts = pipeline.evaluate(cfg, spec, mdl, DEV, ["test"], workers=workers, keep_maps=keep_maps)
    imgs, qs, E, N = metrics.stack_items(parts)
    _, _, Er, Nr = metrics.stack_items(parts, "Erand", "Nrand")
    spec_h = sum(o["spec"] for o in parts.values())
    maps = {(qi, img): m for qi, o in parts.items() for img, m in o["maps"].items()}
    z = dict(img=imgs, q=qs, E=E, N=N, Er=Er, Nr=Nr, spec=spec_h)
    if control_threshold is not None:
        z["shift"] = np.vstack([o["shift"] for qi, o in sorted(parts.items()) if len(o["img"])])
        for qi, o in parts.items():
            for k, v in o.get("dens", {}).items():
                z[f"dens_q{qi}_{k}"] = v
    return spec, mdl, z, maps


def curve_rows(label, split, z, thr_idx=None, n_boot=500, seed=0, max_points=400):
    c = metrics.curve(z["E"].sum(0), z["N"].sum(0))
    nz = np.flatnonzero(np.diff(np.concatenate([[0], c["E_after"]])) > 0)  # bins that change the curve
    if len(nz) > max_points:
        nz = nz[np.unique(np.linspace(0, len(nz) - 1, max_points).astype(int))]
    if thr_idx is not None:
        nz = np.unique(np.concatenate([nz, [thr_idx]]))
    bs = metrics.bootstrap(z["img"], z["E"], z["N"], nz, n_boot=n_boot, seed=seed)
    lo, hi = np.percentile(bs, [16, 84], axis=0)
    df = pd.DataFrame({"method": label, "split": split,
                       "log_rate_threshold": pipeline.ETA_EDGES[nz + 1],
                       "E_before_pixday": c["E_before"], "N_before": c["N_before"],
                       "E_after_pixday": c["E_after"][nz], "N_after": c["N_after"][nz],
                       "f_exp": c["f_exp"][nz], "r_B": c["r_B"][nz], "F_equal": c["F_equal"][nz],
                       "f_exp_lo": lo[0], "f_exp_hi": hi[0], "r_B_lo": lo[1], "r_B_hi": hi[1],
                       "F_equal_lo": lo[2], "F_equal_hi": hi[2]})
    df["B_before_per_pixday"] = df.N_before / df.E_before_pixday
    df["B_after_per_pixday"] = df.N_after / df.E_after_pixday
    df["selected"] = (nz == thr_idx) if thr_idx is not None else False
    return df, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--extra", default="glm_static_only,glm_dynamic_only,glm_intercept_only")
    a = ap.parse_args()
    t0 = time.time()
    cfg = config.load_config()
    rd = config.results_dir(); fd = config.cache_dir(cfg) / "final"; fd.mkdir(exist_ok=True)
    sel = json.loads((rd / "cv" / "selection.json").read_text())
    search = yaml.safe_load(open(config.RUN_DIR / "configs" / "cv_search.yaml"))["configs"]
    name = sel["selected_config"]
    thr = sel["cv_row"]["sel_log_rate_threshold"]
    thr_idx = int(round((thr - pipeline.ETA_EDGES[0]) / pipeline.ETA_STEP)) - 1
    keep = cfg.get("report", {}).get("example_images", [110, 125])
    spec, mdl, z, maps = run(cfg, name, search[name], a.workers, keep_maps=keep, control_threshold=thr)
    np.savez_compressed(fd / f"test_{name}.npz", **z)
    with open(fd / "model.pkl", "wb") as fh:
        pickle.dump({"name": name, "spec": spec, "model": mdl}, fh)
    np.savez_compressed(fd / "example_maps.npz", **{f"q{qi}_img{img}_{k}": v for (qi, img), m in maps.items() for k, v in m.items()})
    (rd / "final_coefficients.json").write_text(json.dumps({"config": name, "spec": spec, **mdl.coefficients()}, indent=1, default=float))

    ops, c = curve_rows(name, "test", z, thr_idx)
    rand, _ = curve_rows("random_mask_control", "test", {"img": z["img"], "E": z["Er"], "N": z["Nr"]})
    tables = [ops, rand]
    # dev out-of-fold curves of every CV config (validation curves)
    for p in sorted((config.cache_dir(cfg) / "cv").glob("*.npz")):
        with np.load(p) as zz:
            d, _ = curve_rows(p.stem, "dev_cv_out_of_fold", {k: zz[k] for k in ("img", "E", "N")})
        tables.append(d)
    extra_rows = {}
    for ex in [e for e in a.extra.split(",") if e and e != name]:
        _, _, ze, _ = run(cfg, ex, search[ex], a.workers)
        np.savez_compressed(fd / f"test_{ex}.npz", **ze)
        d, ce = curve_rows(ex, "test_ablation", ze)
        tables.append(d)
        i = metrics.index_at_fexp(ce, float(c["f_exp"][thr_idx]))
        extra_rows[ex] = {"f_exp": float(ce["f_exp"][i]), "r_B": float(ce["r_B"][i]), "F_equal": float(ce["F_equal"][i])}
    allops = pd.concat(tables, ignore_index=True)
    allops["pareto"] = False
    for (m_, s_), g in allops.groupby(["method", "split"]):
        allops.loc[g.index, "pareto"] = metrics.pareto_mask(g.f_exp.values, g.r_B.values)
    allops.to_csv(rd / "operating_points.csv", index=False)

    bs = metrics.bootstrap(z["img"], z["E"], z["N"], [thr_idx], n_boot=2000, seed=cfg["seed"])[:, :, 0]
    q16, q50, q84 = np.percentile(bs, [16, 50, 84], axis=0)
    res = {"config": name, "log_rate_threshold": thr, "threshold_rate_per_pixday": float(np.exp(thr)),
           "E_before_pixday": float(c["E_before"]), "E_after_pixday": float(c["E_after"][thr_idx]),
           "N_before": int(c["N_before"]), "N_after": int(c["N_after"][thr_idx]),
           "f_exp": float(c["f_exp"][thr_idx]), "r_B": float(c["r_B"][thr_idx]), "F_equal": float(c["F_equal"][thr_idx]),
           "bootstrap_16_50_84": {"f_exp": [q16[0], q50[0], q84[0]], "r_B": [q16[1], q50[1], q84[1]],
                                  "F_equal": [q16[2], q50[2], q84[2]]},
           "significance_gain_sqrtF": float(np.sqrt(c["F_equal"][thr_idx])),
           "ablations_at_same_f_exp": extra_rows,
           "runtime_s": time.time() - t0}
    (rd / "final_test_metrics.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
