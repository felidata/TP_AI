"""Stage 3b: summarise the dev cross-validation and fix the model + operating point.

Pre-declared rules (written before the CV results were seen):
  model:  minimise S = mean_{f in 0.80..0.99 step 0.01} log r_B(f)  (pooled out-of-fold, dev only)
          among the supervised intensity models (glm_*, hgb_*); iforest/intercept-only are comparators.
  point:  among thresholds with f_exp >= 0.5, F_max = max F_equal; choose the largest f_exp with
          F_equal >= 0.95 F_max. The threshold is stored as an absolute predicted rate density
          (log events / pixel-day) and applied unchanged to the held-out test block.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd  # noqa: E402
from skbg import config, metrics, pipeline  # noqa: E402

GRID = np.round(np.arange(0.80, 0.991, 0.01), 2)


def summarise(name, z, n_boot=500, seed=0):
    E, N = z["E"], z["N"]
    c = metrics.curve(E.sum(0), N.sum(0))
    S = float(np.mean([np.log(c["r_B"][metrics.index_at_fexp(c, f)]) for f in GRID]))
    ok = c["f_exp"] >= 0.5
    Fmax = np.nanmax(np.where(ok, c["F_equal"], np.nan))
    cand = np.flatnonzero(ok & (c["F_equal"] >= 0.95 * Fmax))
    i_sel = int(cand[np.argmax(c["f_exp"][cand])])
    idx = [metrics.index_at_fexp(c, f) for f in (0.90, 0.95, 0.99)] + [i_sel]
    bs = metrics.bootstrap(z["img"], E, N, idx, n_boot=n_boot, seed=seed)
    lo, hi = np.percentile(bs, [16, 84], axis=0)
    # per-fold spread at f=0.90
    per_fold = []
    for k in np.unique(z["fold"]):
        s = z["fold"] == k
        ck = metrics.curve(E[s].sum(0), N[s].sum(0))
        per_fold.append(float(ck["r_B"][metrics.index_at_fexp(ck, 0.90)]))
    row = {"config": name, "S_mean_log_rB_80_99": S, "F_max": float(Fmax),
           "sel_f_exp": float(c["f_exp"][i_sel]), "sel_r_B": float(c["r_B"][i_sel]), "sel_F": float(c["F_equal"][i_sel]),
           "sel_log_rate_threshold": float(pipeline.ETA_EDGES[i_sel + 1]),
           "sel_r_B_lo": float(lo[1, 3]), "sel_r_B_hi": float(hi[1, 3])}
    for j, f in enumerate((0.90, 0.95, 0.99)):
        row[f"r_B@{f}"] = float(c["r_B"][idx[j]]); row[f"r_B@{f}_lo"] = float(lo[1, j]); row[f"r_B@{f}_hi"] = float(hi[1, j])
    row["r_B@0.90_per_fold"] = per_fold
    return row, c


def main():
    cfg = config.load_config()
    cvd = config.cache_dir(cfg) / "cv"
    rows = []
    for p in sorted(cvd.glob("*.npz")):
        with np.load(p) as z:
            row, _ = summarise(p.stem, {k: z[k] for k in z.files})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("S_mean_log_rB_80_99")
    rd = config.results_dir() / "cv"; rd.mkdir(exist_ok=True)
    df.to_csv(rd / "cv_summary.csv", index=False)
    elig = df[df.config.str.match(r"^(glm|hgb)_") & (df.config != "glm_intercept_only")]
    best = elig.iloc[0].to_dict()
    sel = {"selected_config": best["config"], "rule": __doc__.strip(), "cv_row": best}
    (rd / "selection.json").write_text(json.dumps(sel, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(df.drop(columns=["r_B@0.90_per_fold"]).round(4).to_string(index=False))
    print("selected:", best["config"], "threshold log-rate", best["sel_log_rate_threshold"])


if __name__ == "__main__":
    main()
