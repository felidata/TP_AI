"""Stage 6: result figures (reads only products of stages 2-5)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402
from skbg import config, metrics, pipeline, stacks, plotstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

plotstyle.apply()
C = plotstyle.SERIES
LABEL = {"glm_full": "Poisson intensity model (full)", "glm_static_only": "static covariates only",
         "glm_dynamic_only": "same-image covariates only", "glm_intercept_only": "quadrant rates only (quadrant veto)",
         "glm_sideband_only": "trained on 2–3 e⁻ sideband only", "glm_static_roi_side": "static ROI + sideband maps",
         "glm_hee20": "Poisson intensity model, HEE ≥ 20 e⁻ (selected)", "glm_l2_100": "L2 = 100", "hgb_full": "gradient-boosted intensity",
         "iforest_full": "Isolation Forest (unsupervised)", "random_mask_control": "random mask (control)"}


def main():
    cfg = config.load_config(); rd = config.results_dir(); fd = config.figures_dir()
    cd = config.cache_dir(cfg)
    ops = pd.read_csv(rd / "operating_points.csv")
    fin = json.loads((rd / "final_test_metrics.json").read_text())
    chk = json.loads((rd / "checks.json").read_text())
    name = fin["config"]
    with np.load(cd / "final" / f"test_{name}.npz") as zz:
        z = {k: zz[k] for k in zz.files}
    thr_idx = int(round((fin["log_rate_threshold"] - pipeline.ETA_EDGES[0]) / pipeline.ETA_STEP)) - 1

    # ---------------- 1. background vs exposure: test + controls + ablations -------------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    t = ops[(ops.method == name) & (ops.split == "test")].sort_values("f_exp")
    ax.fill_between(t.f_exp, t.r_B_lo, t.r_B_hi, color=C[0], alpha=0.2, lw=0)
    ax.plot(t.f_exp, t.r_B, color=C[0], label=f"{LABEL.get(name, name)} — test")
    d = ops[(ops.method == name) & (ops.split == "dev_cv_out_of_fold")].sort_values("f_exp")
    ax.plot(d.f_exp, d.r_B, color=C[0], ls="--", lw=1.5, label="same model — dev cross-validation")
    for i, ex in enumerate(["glm_static_only", "glm_dynamic_only", "glm_intercept_only"]):
        e = ops[(ops.method == ex) & (ops.split == "test_ablation")].sort_values("f_exp")
        if len(e):
            ax.plot(e.f_exp, e.r_B, color=C[i + 1], lw=1.6, label=f"{LABEL[ex]} — test")
    r = ops[(ops.method == "random_mask_control")].sort_values("f_exp")
    ax.plot(r.f_exp, r.r_B, color=plotstyle.INK2, lw=1.2, ls=":", label=LABEL["random_mask_control"])
    ax.plot(fin["f_exp"], fin["r_B"], "o", ms=10, color=C[7], mec="white", mew=2, zorder=5,
            label=f"selected: f_exp={fin['f_exp']:.3f}, r_B={fin['r_B']:.3f}")
    ax.set(yscale="log", xlim=(0.5, 1.001), ylim=(0.01, 1.5), xlabel="f_exp = retained exposure fraction",
           ylabel="r_B = 4–7 e⁻ rate after / before", title="Background vs retained exposure (held-out test block)")
    ax.legend(fontsize=8.5, loc="lower right"); fig.savefig(fd / "result_background_vs_exposure.png"); plt.close(fig)

    # ---------------- 2. Pareto frontier (all methods on test) + F_equal ------------------
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    tt = ops[ops.split.isin(["test", "test_ablation"])]
    for i, (m_, g) in enumerate(tt.groupby("method")):
        g = g.sort_values("f_exp"); p = g[g.pareto]
        col = C[0] if m_ == name else plotstyle.INK2 if m_ == "random_mask_control" else C[(i % 6) + 1]
        axs[0].plot(p.f_exp, p.r_B, color=col, lw=2 if m_ == name else 1.3, label=LABEL.get(m_, m_))
        axs[1].plot(g.f_exp, g.F_equal, color=col, lw=2 if m_ == name else 1.3, label=LABEL.get(m_, m_))
    axs[1].fill_between(t.f_exp, t.F_equal_lo, t.F_equal_hi, color=C[0], alpha=0.2, lw=0)
    axs[1].plot(d.f_exp, d.F_equal, color=C[0], ls="--", lw=1.5, label="selected model — dev CV")
    axs[0].set(yscale="log", xlim=(0.5, 1.001), xlabel="f_exp", ylabel="r_B", title="Pareto frontiers (test)")
    axs[1].set(xlim=(0.5, 1.001), xlabel="f_exp", ylabel="F_equal = f_exp / r_B", title="Equal-weight trade-off (test)")
    axs[1].axvline(fin["f_exp"], color=C[7], lw=1, ls="--")
    for a in axs:
        a.legend(fontsize=8)
    fig.savefig(fd / "result_pareto_and_Fequal.png"); plt.close(fig)

    # ---------------- 3. model comparison on dev CV -----------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    dv = ops[ops.split == "dev_cv_out_of_fold"]
    for i, (m_, g) in enumerate(sorted(dv.groupby("method"), key=lambda x: x[0])):
        g = g.sort_values("f_exp")
        ax.plot(g.f_exp, g.r_B, lw=2.2 if m_ == name else 1.3, color=C[i % 8], ls="-" if i < 8 else "--",
                label=LABEL.get(m_, m_))
    ax.set(yscale="log", xlim=(0.5, 1.001), ylim=(0.01, 1.5), xlabel="f_exp", ylabel="r_B",
           title="Development set, 4-fold blocked CV (out-of-fold)")
    ax.legend(fontsize=8); fig.savefig(fd / "cv_model_comparison.png"); plt.close(fig)

    # ---------------- 4. spectrum before / after + ratio -------------------------------------
    sp = z["spec"]; kept = sp[:, : thr_idx + 1].sum(1); allq = sp.sum(1)
    E_b = z["E"].sum(); E_a = np.cumsum(z["E"], 1)[:, thr_idx].sum()
    q = np.arange(sp.shape[0])
    fig, axs = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    axs[0].step(q[1:63], allq[1:63] / E_b, where="mid", color=C[1], label="before (no mask)")
    axs[0].step(q[1:63], kept[1:63] / E_a, where="mid", color=C[0], label="after (selected mask)")
    axs[0].axvspan(3.5, 7.5, color=C[3], alpha=0.18, lw=0, label="4–7 e⁻ ROI")
    axs[0].set(yscale="log", ylabel="clusters / pixel·day", title="Cluster charge spectrum, test block")
    axs[0].legend()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (kept / E_a) / (allq / E_b)
        err = ratio * np.sqrt(1 / np.maximum(kept, 1))
    axs[1].errorbar(q[1:63], ratio[1:63], yerr=err[1:63], fmt="o", ms=4, color=C[0])
    axs[1].axvspan(3.5, 7.5, color=C[3], alpha=0.18, lw=0)
    axs[1].axhline(1, color=plotstyle.INK2, lw=1)
    axs[1].set(yscale="log", xlabel="cluster charge Q [e⁻] (last bin: overflow)", ylabel="rate ratio after/before")
    fig.savefig(fd / "result_spectrum_before_after.png"); plt.close(fig)

    # ---------------- 5. density before/after (test) -----------------------------------------
    fig, axs = plt.subplots(8, 2, figsize=(14, 17), layout="constrained")
    for qi, (m_, h) in enumerate(stacks.QUADRANTS):
        k = f"dens_q{qi}_"
        if k + "E_all" not in z:
            continue
        for j, (nk, ek, lab) in enumerate((("N_all", "E_all", "before"), ("N_kept", "E_kept", "after mask"))):
            Nb = uniform_filter(z[k + nk].astype(float), 32); Eb = uniform_filter(z[k + ek].astype(float), 32)
            with np.errstate(divide="ignore", invalid="ignore"):
                im = axs[qi, j].imshow(np.log10(Nb / Eb + 1e-6), origin="lower", aspect="auto", cmap=plotstyle.SEQ_CMAP,
                                       vmin=-5, vmax=-1)
            masked = z[k + "E_kept"] < 0.5 * z[k + "E_all"]
            axs[qi, j].imshow(np.where(masked, 1.0, np.nan), origin="lower", aspect="auto",
                              cmap=ListedColormap(["#9ec5f0"]), alpha=0.35 if j else 0.0)
            axs[qi, j].set_title(f"{m_} HDU{h} — {lab}: {int(z[k + nk].sum())} clusters", fontsize=9)
            axs[qi, j].grid(False); axs[qi, j].tick_params(labelsize=7)
    fig.colorbar(im, ax=axs, shrink=0.4, label="log10 (4–7 e⁻ / pixel·day), 32 px smoothing; light blue: masked in >50% of test exposure")
    fig.suptitle("4–7 e⁻ cluster density on the held-out test block, before and after the selected mask")
    fig.savefig(fd / "result_density_before_after.png", dpi=110); plt.close(fig)

    # ---------------- 6. example image / score / mask ---------------------------------------
    ex = np.load(cd / "final" / "example_maps.npz")
    keys = sorted({k.rsplit("_", 1)[0] for k in ex.files if k.endswith("_eta")})
    keys = [k.replace("_eta", "") for k in keys]
    for key in keys:
        qi = int(key.split("_")[0][1:]); img = int(key.split("img")[1])
        m_, h = stacks.QUADRANTS[qi]
        n = ex[f"{key}_n"]; eta = ex[f"{key}_eta"]; rr = ex[f"{key}_roi_rc"]; cc = ex[f"{key}_roi_cc"]
        keep = eta < fin["log_rate_threshold"]
        fig, axs = plt.subplots(3, 1, figsize=(14, 10.5), layout="constrained")
        im0 = axs[0].imshow(np.log10(1 + n), origin="lower", aspect="auto", cmap="gray_r", vmin=0, vmax=2, interpolation="nearest")
        fig.colorbar(im0, ax=axs[0], label="log10(1 + charge [e⁻])")
        axs[0].set_title(f"{m_} HDU{h}, acquisition {img} (test): integer charge per pixel")
        im1 = axs[1].imshow(eta / np.log(10), origin="lower", aspect="auto", cmap=plotstyle.SEQ_CMAP, vmin=-6, vmax=0, interpolation="nearest")
        fig.colorbar(im1, ax=axs[1], label="log10 predicted 4–7 e⁻ rate [/pixel·day]")
        axs[1].set_title("anomaly score = predicted rate density (built without this image's 4–7 e⁻ clusters)")
        axs[2].imshow(np.log10(1 + n), origin="lower", aspect="auto", cmap="gray_r", vmin=0, vmax=2, interpolation="nearest")
        axs[2].imshow(np.where(keep, np.nan, 1.0), origin="lower", aspect="auto", cmap=ListedColormap([C[0]]), alpha=0.45, interpolation="nearest")
        k_ev = keep[rr, cc]
        axs[2].scatter(cc[k_ev], rr[k_ev], s=22, facecolors="none", edgecolors=C[2], lw=1.2, label=f"4–7 e⁻ kept ({k_ev.sum()})")
        axs[2].scatter(cc[~k_ev], rr[~k_ev], s=22, marker="x", color=C[7], lw=1.2, label=f"4–7 e⁻ removed ({(~k_ev).sum()})")
        axs[2].set_title(f"mask overlay (blue = masked, {100 * (1 - keep.mean()):.1f}% of pixels)")
        axs[2].legend(loc="upper right", fontsize=9)
        for a in axs:
            a.grid(False); a.set_ylabel("row")
        axs[2].set_xlabel("column")
        fig.savefig(fd / f"example_{m_}_HDU{h}_img{img}.png", dpi=110); plt.close(fig)

    # ---------------- 7. zoomed accepted / rejected regions ---------------------------------
    key = next((k for k in keys if k.startswith("q5_")), keys[0])  # S99 HDU1 if available
    n = ex[f"{key}_n"]; eta = ex[f"{key}_eta"]; keep = eta < fin["log_rate_threshold"]
    rng = np.random.default_rng(cfg["seed"])
    W = 64
    cand_rej, cand_acc = [], []
    for _ in range(4000):
        r0 = rng.integers(0, 512 - W); c0 = rng.integers(0, 3072 - W)
        frac = 1 - keep[r0:r0 + W, c0:c0 + W].mean()
        (cand_rej if 0.3 < frac < 0.9 else cand_acc if frac == 0 else []).append((r0, c0))
    picks = [("rejected (partly masked)", p) for p in cand_rej[:4]] + [("accepted (unmasked)", p) for p in cand_acc[:4]]
    fig, axs = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    for a, (lab, (r0, c0)) in zip(axs.flat, picks):
        a.imshow(np.log10(1 + n[r0:r0 + W, c0:c0 + W]), origin="lower", cmap="gray_r", vmin=0, vmax=2, interpolation="nearest")
        a.imshow(np.where(keep[r0:r0 + W, c0:c0 + W], np.nan, 1.0), origin="lower", cmap=ListedColormap([C[0]]), alpha=0.4, interpolation="nearest")
        a.set_title(f"{lab}\nrows {r0}–{r0 + W}, cols {c0}–{c0 + W}", fontsize=9); a.grid(False); a.set_xticks([]); a.set_yticks([])
    qi = int(key.split("_")[0][1:]); m_, h = stacks.QUADRANTS[qi]
    fig.suptitle(f"64×64 px cut-outs, {m_} HDU{h}, acquisition {key.split('img')[1]}: charge (grey) and mask (blue)")
    fig.savefig(fd / "example_cutouts.png", dpi=110); plt.close(fig)

    # ---------------- 8. per-quadrant and per-acquisition ------------------------------------
    dq = pd.read_csv(rd / "checks_by_quadrant.csv")
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    lab = [f"{a}\nHDU{b}" for a, b in zip(dq.module, dq.hdu)]
    axs[0].bar(np.arange(8) - 0.2, dq.f_exp, 0.4, color=C[0], label="f_exp")
    axs[0].bar(np.arange(8) + 0.2, dq.r_B, 0.4, color=C[1], label="r_B")
    axs[0].set_xticks(range(8), lab, fontsize=8); axs[0].set(title="Per quadrant at the selected threshold (test)"); axs[0].legend()
    pa = pd.read_csv(rd / "checks_by_acquisition.csv")
    axs[1].plot(pa.img, pa.r_B, "o-", ms=4, color=C[1], label="r_B")
    axs[1].plot(pa.img, pa.f_exp, "s-", ms=4, color=C[0], label="f_exp")
    axs[1].set(xlabel="test acquisition", title="Per acquisition at the selected threshold", yscale="log"); axs[1].legend()
    fig.savefig(fd / "result_per_quadrant_and_acquisition.png"); plt.close(fig)

    # ---------------- 9. calibration of the predicted rate -----------------------------------
    cal = pd.DataFrame(chk["calibration"]); cal = cal[cal.E > 0]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.errorbar(cal.predicted, cal.observed, yerr=np.sqrt(np.maximum(cal.observed, 1)), fmt="o", color=C[0])
    lim = [max(0.1, cal.predicted.min() * 0.5), cal[["predicted", "observed"]].max().max() * 2]
    ax.plot(lim, lim, color=plotstyle.INK2, lw=1, ls="--", label="observed = predicted")
    for _, rw in cal.iterrows():
        ax.annotate(f"{rw.log_rate_bin}", (rw.predicted, max(rw.observed, 0.5)), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(xscale="log", yscale="log", xlabel="predicted 4–7 e⁻ clusters (sum of exposure × rate)", ylabel="observed",
           title="Calibration on the test block\n(label: floor of ln predicted rate)")
    ax.legend(); fig.savefig(fd / "result_calibration.png"); plt.close(fig)

    # ---------------- 10. coefficients -------------------------------------------------------
    co = json.loads((rd / "final_coefficients.json").read_text())
    if "beta_standardized" in co:
        b = pd.Series(co["beta_standardized"]).sort_values()
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(b) + 1.5))
        ax.barh(b.index, b.values, color=[C[0] if v > 0 else C[1] for v in b.values])
        ax.axvline(0, color=plotstyle.INK2, lw=1)
        ax.set(xlabel="standardized coefficient (log-rate per 1 s.d.)", title="Final model coefficients")
        ax.tick_params(axis="y", labelsize=8)
        fig.savefig(fd / "model_coefficients.png"); plt.close(fig)
    print("figures written to", fd)


if __name__ == "__main__":
    main()
