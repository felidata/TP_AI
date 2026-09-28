# Provenance

This file maps each reported artefact to its inputs, script and function, configuration, algorithm, libraries, literature and the
check behind it. Each number printed in `report.html` also has a record (value, statement, producer, from-scratch versus library,
choices) in `provenance/numbers.json`, written by `scripts/08_report.py`.

**Inputs for everything:** the 218 FITS files listed in `data/manifest.csv`, with SHA-256 hashes (`scripts/00_manifest.py --verify`);
217 are used (quality rule in DATA.md). **Configuration:** `configs/default.yaml` (seed 20260927) and `configs/cv_search.yaml`.

## What was written here, what is a library, and what comes from MOSKITA

| item | origin |
|---|---|
| calibration fit model (4 Gaussians, common σ, linear gain), rounding, quality rule | written here (`src/skbg/calib.py`, `dataset.py`); fitting via `scipy.optimize.curve_fit` |
| cluster catalogue (charge, centroid, bbox) | written here (`events.py`); connected components via `scipy.ndimage.label` |
| exposure-time map | re-implemented here; **same physical convention as MOSKITA `moskita.exposure.make_pixel_time_map`** (definition reused, code not imported) |
| geometry slices (active 0:512 × 8:3080, pedestal columns 3100:3200), 8-connected cluster definition | **MOSKITA definitions** (docs/notes/reading_skipper_fits.md, CalibrationConfig, hotzones note), reused as definitions |
| static covariates (shrunk multiscale relative rates), trail covariates, single-electron covariates | written here (`features.py`); box sums via `scipy.ndimage.uniform_filter`; trail loops JIT-compiled with `numba` |
| Poisson point-process likelihood, gradient, quadrature, cross-fitting | written here (`model.py`, `pipeline.py`); optimizer `scipy.optimize.minimize(L-BFGS-B)`; method from Berman & Turner (1992), Baddeley & Turner (2000) |
| gradient-boosted / Isolation-Forest alternatives (not completed) | wrappers written here; models are `sklearn.ensemble.HistGradientBoostingClassifier` / `IsolationForest`; logistic identity from Baddeley et al. (2014); IF from Liu, Ting & Zhou (2008) |
| operating points, acquisition bootstrap, Pareto mask, √F proxy | written here (`metrics.py`); Asimov formula from Cowan et al. (2011) |
| FITS I/O, arrays, tables, plots | `astropy.io.fits`, `numpy`, `pandas`, `matplotlib` |

No MOSKITA mask (hot columns, bleed, halo, SRE, transfer gate, hot zones, crosstalk) and no MOSKITA threshold is used anywhere.
MOSKITA calibrated products are read only by the external cross-check `notebooks/check_calibration_vs_moskita.py`.

## Artefacts

| artefact | script :: function | inputs | algorithm / choices | check |
|---|---|---|---|---|
| `cache/images/*.npz`, `cache/calibration_table.csv` (not committed) | `scripts/01_build_cache.py` → `cache.process_file` | FITS | per-HDU calibration, clusters | χ²/ndf 0.8–3.4; `check_linearity.csv` (peaks within 0.04 e⁻ mean); MOSKITA gain cross-check |
| `cache/stacks/*.npz` | `stacks.build` | cache | per-fold count maps | totals equal baseline counts |
| `results/baseline*.{json,csv}`, `figures/baseline_*.png` | `scripts/02_baseline.py::main` | cache, stacks | no mask; Poisson CIs (χ² quantiles) | dev/test rates per quadrant agree (table) |
| `results/explore_hee_proximity*.{csv,json}` | `notebooks/explore_02_hee_proximity.py` | cache (dev, even acquisitions) | EDT distance to Q ≥ 100 clusters (`scipy.ndimage.distance_transform_edt`) | exploratory, motivates covariates |
| `figures/explore_dev_stack_maps.png` | `notebooks/explore_01_maps.py` | cache (dev) | stacked maps | exploratory |
| `results/check_linearity.csv` | `scripts/check_linearity.py::main` | FITS (every 9th acquisition) | peak medians n = 1…8 | — |
| `results/check_calibration_vs_moskita.csv` | `notebooks/check_calibration_vs_moskita.py` | our table + MOSKITA `outputs/calibrated` headers | gain comparison | disagreements checked against direct peak medians |
| `cache/cv/*.npz`, `results/cv/*_coefficients.json` | `scripts/03_cv_search.py::run_cv` | cache, stacks (dev only) | 4-fold chronological CV, cross-fitted static maps | random-mask channel in every fold |
| `results/cv/cv_summary.csv`, `results/cv/selection.json` | `scripts/03b_select.py::summarise/main` | `cache/cv` | pre-declared rule S; F within 5% of max | per-fold r_B spread in the CSV |
| `figures/cv_model_comparison.png` | `scripts/06_figures.py` §3 | `operating_points.csv` (dev CV rows) | — | — |
| `results/final_test_metrics.json`, `results/final_coefficients.json`, `cache/final/*` | `scripts/04_final.py::main` | dev for fitting; test for evaluation, once | frozen threshold ln ρ* = −8.91 | bootstrap 2000 × acquisitions |
| `results/operating_points.csv` | `scripts/04_final.py::curve_rows` | test + dev-CV histograms | ≤ 400 thresholds per curve, bootstrap 500; `pareto` column from `metrics.pareto_mask` | recomputable from columns E/N before/after |
| `results/checks.json`, `checks_by_quadrant.csv`, `checks_by_acquisition.csv` | `scripts/05_checks.py::main`, `leakage_invariance` | test, final model | leakage invariance, random/shifted masks, LOAO, without S101 HDU0, charge windows, calibration | these are the checks |
| `figures/result_*.png`, `example_*.png`, `model_coefficients.png` | `scripts/06_figures.py::main` (sections 1–10) | results + `cache/final` | — | visual inspection done for the main figures |
| `results/summary.json` | `scripts/07_summary.py::main` | all results | — | fields cross-checked against `final_test_metrics.json` |
| `report.html`, `provenance/numbers.json` | `scripts/08_report.py` + `scripts/report_body.py::body` | results, figures | numbers read at build time | — |

## Checks done by hand during the run (for the record)

* The numba trail filter was compared with `scipy.signal.lfilter`. It differs only by the documented constant factor e^{1/L}
  (kernel exp(−d/L) versus exp(−(d−1)/L)).
* Speed-ups (in-place linear predictor, arithmetic binning, inverse-CDF dummy sampling) reproduced the fold-3 test curve to
  every printed digit.
* S99 acquisition 114: the gain drop was verified in 4 row ranges (≈176–188 ADU/e⁻ vs ≈212–214 in acquisitions 113 and 115).
* README numbers were compared with `results/operating_points.csv` and `results/final_test_metrics.json`.
