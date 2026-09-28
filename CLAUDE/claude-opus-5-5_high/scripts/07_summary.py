"""Stage 7: standardized machine-readable summary (results/summary.json)."""
import json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skbg import config, dataset  # noqa: E402


def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=config.RUN_DIR, text=True).strip()
    except Exception:
        return None


def main():
    cfg = config.load_config(); rd = config.results_dir()
    fin = json.loads((rd / "final_test_metrics.json").read_text())
    sel = json.loads((rd / "cv" / "selection.json").read_text())
    base = json.loads((rd / "baseline.json").read_text())
    chk = json.loads((rd / "checks.json").read_text())
    coef = json.loads((rd / "final_coefficients.json").read_text())
    timing = {}
    tf = rd / "run_timing.json"
    if tf.exists():
        timing = json.loads(tf.read_text())
    recs = dataset.records(cfg)
    pm = cfg["geometry"]["pixel_mass_g"]
    B0 = fin["N_before"] / fin["E_before_pixday"]; B1 = fin["N_after"] / fin["E_after_pixday"]
    summary = {
        "agent": "CLAUDE",
        "model": "claude-opus-5-5",
        "effort": "high",
        "run_id": "claude-opus-5-5_high",
        "method_name": "Cross-fitted log-linear Poisson point-process model of the 4-7 e- rate density "
                       "(static multiscale rate maps from other acquisitions + same-image high-energy-trail and "
                       "single-electron covariates); mask = pixels with predicted rate above a threshold",
        "selected_config": fin["config"],
        "data_files": {"n_files": len(recs), "n_usable": sum(r["usable"] for r in recs),
                       "excluded": [r["file"] for r in recs if not r["usable"]],
                       "manifest": "data/manifest.csv", "pattern": cfg["data"]["file_glob"]},
        "split_strategy": ("Acquisition-level split (both modules and all 4 HDUs of an image number stay together). "
                           f"Development: acquisitions 22-{cfg['split']['test_first_img'] - 1} (76), 4 contiguous "
                           "chronological folds for cross-validation, model/feature/threshold selection; static maps "
                           "cross-fitted so no image contributes to its own covariates. Held-out test: acquisitions "
                           f"{cfg['split']['test_first_img']}-130 (33 acquisitions, 65 files after the quality cut), "
                           "evaluated once with the frozen model and threshold."),
        "evaluation_set": "held-out test block",
        "units": {"exposure": "pixel-day (1 pixel-day = %.4e g-day)" % pm, "rate": "4-7 e- clusters per pixel-day"},
        "baseline_exposure": fin["E_before_pixday"],
        "baseline_exposure_gday": fin["E_before_pixday"] * pm,
        "retained_exposure": fin["E_after_pixday"],
        "retained_exposure_gday": fin["E_after_pixday"] * pm,
        "exposure_fraction": fin["f_exp"],
        "baseline_roi_events": fin["N_before"],
        "final_roi_events": fin["N_after"],
        "baseline_roi_background_rate": B0,
        "final_roi_background_rate": B1,
        "background_ratio": fin["r_B"],
        "F_equal": fin["F_equal"],
        "uncertainties_bootstrap_acquisitions_16_50_84": fin["bootstrap_16_50_84"],
        "significance_gain_background_dominated": fin["significance_gain_sqrtF"],
        "selected_hyperparameters": {"spec": coef.get("spec"), "log_rate_threshold": fin["log_rate_threshold"],
                                     "rate_threshold_per_pixday": fin["threshold_rate_per_pixday"],
                                     "selection_rule": sel["rule"]},
        "dev_cv_at_selected_threshold": {k: sel["cv_row"][k] for k in ("sel_f_exp", "sel_r_B", "sel_F", "sel_r_B_lo", "sel_r_B_hi")},
        "whole_dataset_baseline": base["totals"]["all"],
        "sanity_checks": {k: chk[k] for k in ("leakage_invariance", "area_controls", "leave_one_acquisition_out",
                                              "without_S101_HDU0", "charge_windows_same_mask")},
        "random_seed": cfg["seed"],
        "runtime_if_available": timing or {"final_stage_s": fin.get("runtime_s")},
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_at_run": bool(git("status", "--porcelain", ".")),
        "notes": ("ROI = 8-connected clusters with integer total charge 4..7 e- (pixel charge floor(q+0.5)); "
                  "a cluster is kept if its charge-weighted centroid pixel is unmasked. Exposure per pixel from "
                  "the continuous-readout time map. The test block baseline differs from the whole-dataset "
                  "baseline (see whole_dataset_baseline). S101 HDU0 carries ~73% of baseline 4-7 e- clusters; "
                  "see sanity_checks.without_S101_HDU0. git_commit is the commit checked out when this summary "
                  "was generated (the code commit); results are committed in a following commit."),
    }
    (rd / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: summary[k] for k in ("exposure_fraction", "background_ratio", "F_equal", "git_commit")}, indent=1))


if __name__ == "__main__":
    main()
