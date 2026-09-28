"""Stage 2: dataset audit and baseline (no mask) quantities.

Descriptive only: nothing here is used to choose a method or a threshold.
Writes results/baseline.json, results/baseline_by_quadrant.csv,
results/baseline_by_acquisition.csv and figures/baseline_*.png.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd  # noqa: E402
from scipy import stats as sst  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402
from skbg import cache, config, dataset, exposure, stacks, plotstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

plotstyle.apply()


def poisson_ci(n, cl=0.6827):
    a = (1 - cl) / 2
    lo = 0.0 if n == 0 else sst.chi2.ppf(a, 2 * n) / 2
    hi = sst.chi2.ppf(1 - a, 2 * n + 2) / 2
    return lo, hi


def main():
    cfg = config.load_config(); cd = config.cache_dir(cfg)
    fig_d, res_d = config.figures_dir(), config.results_dir()
    pix_g = cfg["geometry"]["pixel_mass_g"]
    recs_all = dataset.records(cfg)
    recs = [r for r in recs_all if r["usable"]]
    cal = dataset.quality_flags(dataset.calibration_table(cfg))
    kmap = exposure.exposure_map_days({"exposure_s": 0.0, "readout_s": 86400.0, "nrow": 515, "ncol": 3200},
                                      (0, 512), (8, 3080))
    rows = []
    specQ = {s: np.zeros(201) for s in ("dev", "test")}
    npix_roi = np.zeros(10)
    for r in recs:
        z = cache.load_image(cd, r); t = cache.load_meta(cd, r)["timing"]
        Eimg = float(kmap.sum() * t["readout_s"] / 86400.0)
        sp = dataset.split_of(r["img"], cfg)
        for h in range(4):
            Q = z[f"cl{h}_Q"]; npx = z[f"cl{h}_npix"]
            roi = (Q >= 4) & (Q <= 7)
            specQ[sp] += np.bincount(np.clip(Q, 0, 200), minlength=201)
            npix_roi += np.bincount(np.clip(npx[roi], 0, 9), minlength=10)
            rows.append({"img": r["img"], "module": r["module"], "hdu": h, "split": sp,
                         "exposure_pixday": Eimg, "n_roi": int(roi.sum()),
                         "n_roi_le4pix": int((roi & (npx <= 4)).sum()),
                         "n_q1": int((Q == 1).sum()), "n_q23": int(((Q >= 2) & (Q <= 3)).sum()),
                         "n_hee100": int((Q >= 100).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(res_d / "baseline_by_acquisition.csv", index=False)

    def summ(g):
        n, E = g.n_roi.sum(), g.exposure_pixday.sum()
        lo, hi = poisson_ci(n)
        disp = g.groupby("img").n_roi.sum()
        return pd.Series({"n_quadrant_images": len(g), "exposure_pixday": E, "exposure_gday": E * pix_g,
                          "n_roi": n, "rate_per_pixday": n / E, "rate_lo": lo / E, "rate_hi": hi / E,
                          "rate_per_gday": n / (E * pix_g),
                          "dispersion_index": disp.var() / disp.mean() if disp.mean() > 0 else np.nan})
    byq = df.groupby(["split", "module", "hdu"]).apply(summ, include_groups=False).reset_index()
    byq["share_of_split_roi"] = byq.n_roi / byq.groupby("split").n_roi.transform("sum")
    byq.to_csv(res_d / "baseline_by_quadrant.csv", index=False)
    tot = {s: summ(g).to_dict() for s, g in df.groupby("split")}
    tot["all"] = summ(df).to_dict()
    base = {
        "n_fits_files": len(recs_all), "n_usable_files": len(recs),
        "unusable_files": [r["file"] for r in recs_all if not r["usable"]],
        "unusable_reason": "calibration quality cut: fitted gain deviates >5% from module/HDU median in all HDUs "
                           "(S99 img114: gain ~180 vs ~214 ADU/e-, noise 0.20-0.21 e-)",
        "hdus_per_file": 4, "modules": sorted({r["module"] for r in recs_all}),
        "acquisitions": [min(r["img"] for r in recs_all), max(r["img"] for r in recs_all)],
        "active_region_per_hdu": [512, 3072],
        "readout_hours_median": float(cal.readout_s.median() / 3600),
        "calibration_by_quadrant": cal[~cal.bad_quality].groupby(["module", "hdu"])[["gain_adu", "noise_e", "chi2_ndf", "see_a1a0"]]
            .median().round(5).reset_index().to_dict(orient="records"),
        "roi_definition": "8-connected cluster of pixels with integer charge n>=1 (n=floor(q+0.5)); 4<=sum(n)<=7",
        "roi_cluster_size_distribution": {int(k): int(v) for k, v in enumerate(npix_roi) if v},
        "totals": tot,
        "pixel_mass_g": pix_g,
    }
    (res_d / "baseline.json").write_text(json.dumps(base, indent=1, default=float))

    # ---- figure: cluster charge spectrum -----------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    q = np.arange(201)
    for i, s in enumerate(("dev", "test")):
        E = tot[s]["exposure_pixday"]
        ax.step(q[1:101], specQ[s][1:101] / E, where="mid", label=f"{s} acquisitions", color=plotstyle.SERIES[i])
    ax.axvspan(3.5, 7.5, color=plotstyle.SERIES[3], alpha=0.18, lw=0, label="ROI 4–7 e⁻")
    ax.set(yscale="log", xscale="log", xlabel="cluster charge Q [e⁻]", ylabel="clusters / pixel·day",
           title="Cluster charge spectrum, no mask (all quadrants)")
    ax.legend(); fig.savefig(fig_d / "baseline_spectrum.png"); plt.close(fig)

    # ---- figure: rates by quadrant ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = [f"{m}\nHDU{h}" for m, h in stacks.QUADRANTS]
    for i, s in enumerate(("dev", "test")):
        b = byq[byq.split == s].set_index(["module", "hdu"])
        y = [b.loc[k, "rate_per_pixday"] for k in stacks.QUADRANTS]
        lo = [b.loc[k, "rate_per_pixday"] - b.loc[k, "rate_lo"] for k in stacks.QUADRANTS]
        hi = [b.loc[k, "rate_hi"] - b.loc[k, "rate_per_pixday"] for k in stacks.QUADRANTS]
        ax.errorbar(np.arange(8) + (i - 0.5) * 0.2, y, yerr=[lo, hi], fmt="o", ms=6, label=s, color=plotstyle.SERIES[i])
    ax.set_xticks(range(8), labels); ax.set_yscale("log")
    ax.set(ylabel="4–7 e⁻ clusters / pixel·day", title="Baseline 4–7 e⁻ rate per quadrant")
    ax.legend(); fig.savefig(fig_d / "baseline_rate_by_quadrant.png"); plt.close(fig)

    # ---- figure: rate per acquisition (time series) ------------------------
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ai, m in enumerate(("S101", "S99")):
        for h in range(4):
            g = df[(df.module == m) & (df.hdu == h)].sort_values("img")
            axs[ai].plot(g.img, g.n_roi / g.exposure_pixday, lw=1.3, label=f"HDU{h}", color=plotstyle.SERIES[h])
        axs[ai].set(yscale="log", ylabel="rate / pixel·day", title=f"{m}: 4–7 e⁻ rate per acquisition")
        axs[ai].axvline(cfg["split"]["test_first_img"] - 0.5, color=plotstyle.INK2, ls="--", lw=1)
        axs[ai].legend(ncol=4, fontsize=9)
    axs[1].set_xlabel("acquisition (image number); dashed: start of held-out test block")
    fig.savefig(fig_d / "baseline_rate_vs_acquisition.png"); plt.close(fig)

    # ---- figure: spatial maps (dev and test stacks, descriptive) -----------
    for s, groups in (("dev", [f"f{k}" for k in range(cfg["split"]["n_cv_folds"])]), ("test", ["test"])):
        fig, axs = plt.subplots(4, 2, figsize=(14, 11), layout="constrained")
        for (m, h), ax in zip(stacks.QUADRANTS, axs.T.flat):
            st = stacks.load(cfg, groups, m, h)
            N = uniform_filter(st["roi"], 32) * 32 * 32
            Eb = uniform_filter(st["E"], 32) * 32 * 32
            im = ax.imshow(np.log10(N / Eb + 1e-6), origin="lower", aspect="auto", cmap=plotstyle.SEQ_CMAP, vmin=-5, vmax=-1)
            ax.set_title(f"{m} HDU{h}: {int(st['roi'].sum())} clusters", fontsize=10); ax.grid(False)
        fig.colorbar(im, ax=axs, label="log10 (4–7 e⁻ clusters / pixel·day), 32×32 px smoothing", shrink=0.6)
        fig.suptitle(f"4–7 e⁻ cluster density, {s} acquisitions, no mask (x: column, y: row)")
        fig.savefig(fig_d / f"baseline_roi_density_{s}.png"); plt.close(fig)

    # ---- persistence: block counts, dev vs test ----------------------------
    pers = []
    for (m, h) in stacks.QUADRANTS:
        a = stacks.load(cfg, [f"f{k}" for k in range(cfg["split"]["n_cv_folds"])], m, h)
        b = stacks.load(cfg, ["test"], m, h)
        for B in (8, 32, 128):
            ca = a["roi"][: 512 // B * B, : 3072 // B * B].reshape(512 // B, B, 3072 // B, B).sum((1, 3)).ravel()
            cb = b["roi"][: 512 // B * B, : 3072 // B * B].reshape(512 // B, B, 3072 // B, B).sum((1, 3)).ravel()
            ea = a["E"][: 512 // B * B, : 3072 // B * B].reshape(512 // B, B, 3072 // B, B).sum((1, 3)).ravel()
            pers.append({"module": m, "hdu": h, "block": B, "spearman_dev_vs_test": float(sst.spearmanr(ca / ea, cb / ea).statistic),
                         "top1pct_blocks_share_dev": float(np.sort(ca)[::-1][: max(1, len(ca) // 100)].sum() / ca.sum())})
    pd.DataFrame(pers).to_csv(res_d / "baseline_persistence.csv", index=False)
    print(json.dumps(tot, indent=1, default=float))
    print(byq.to_string())
    print(pd.DataFrame(pers).to_string())


if __name__ == "__main__":
    main()
