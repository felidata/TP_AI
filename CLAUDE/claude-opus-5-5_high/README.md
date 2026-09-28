# Skipper-CCD 4–7 e⁻ background rejection — run `claude-opus-5-5_high`

Independent analysis by the CLAUDE agent for the UBA course *Gravitational Waves and AI-Assisted Research*.
The task is in [ORIGINAL_PROMPT.md](ORIGINAL_PROMPT.md), the full report in [report.html](report.html)
(self-contained; open it in a browser), definitions in [METHOD.md](METHOD.md), data in [DATA.md](DATA.md), and the
trace of every figure and number in [PROVENANCE.md](PROVENANCE.md) and [provenance/numbers.json](provenance/numbers.json).

## Run identification

| field | value | source |
|---|---|---|
| agent | CLAUDE (Claude Code 2.1.283, CLI) | `claude --version` |
| model | `claude-opus-5-5` | runtime model id |
| effort | `high` | `CLAUDE_EFFORT=high` in the session environment and `effortLevel: high` in `~/.claude/settings.json` |
| branch | `claude/claude-opus-5-5_high` | orphan branch; never merged into `main` |

## Result (held-out test block, acquisitions 98–130)

Method: a **cross-fitted log-linear Poisson point-process model of the 4–7 e⁻ rate density**. It combines static multiscale rate
maps from *other* acquisitions with same-image covariates: exponential trails around deposits with Q ≥ 20 e⁻ and
single-electron row/column activity. The mask is every pixel whose predicted rate exceeds a threshold fixed on the development set.

| quantity | value (68% acquisition-bootstrap interval) |
|---|---|
| retained exposure f_exp | 0.731 (2.95×10⁷ of 4.04×10⁷ pixel·day) |
| background ratio r_B | 0.046 [0.044, 0.048] |
| F_equal = f_exp / r_B | 16.0 [15.4, 16.6] |
| 4–7 e⁻ clusters | 34 091 → 1 140 |
| rate | 8.43×10⁻⁴ → 3.86×10⁻⁵ per pixel·day |
| same mask without S101 HDU0 | f_exp 0.727, r_B 0.134, F 5.4 |

Other points of the same test curve: f_exp 0.903 → r_B 0.072 [0.070, 0.073]; f_exp 0.951 → r_B 0.090 [0.088, 0.093]; f_exp 0.990 → r_B 0.145 [0.142, 0.148]. The complete curve is in
`results/operating_points.csv`, and the machine-readable summary in `results/summary.json`.

Checks: deleting every 4–7 e⁻ cluster from an image leaves its score unchanged (max |Δ| = 0). Random masks and the same
masks shifted along columns give r_B ≈ 1.00–1.04. Leave-one-acquisition-out r_B stays within 0.045–0.046. Predicted and observed
test counts are 33 770 vs 34 091.

**Main limitation:** one defective quadrant (S101 HDU0) holds ~73% of the baseline 4–7 e⁻ clusters and dominates the headline
ratio. In addition, pixels on high-energy tracks are blind to 4–7 e⁻ clusters but are counted as retained exposure, which makes
r_B optimistic by an estimated ~4%.

## Reproduce

```bash
cd CLAUDE/claude-opus-5-5_high
pip install -r requirements.txt              # Python 3.13; versions pinned
export SKIPPER_DATA_DIR=/path/to/pp_proc     # 218 FITS files, see DATA.md / data/manifest.csv
./run_all.sh                                 # everything: cache, baseline, 8-config CV, selection, test, checks, figures, report
SKIP_CV=1 ./run_all.sh                       # reuse cache/cv/*.npz (final stages only, ~15 min)
```

Without `SKIPPER_DATA_DIR`, the default is `../pp_proc` relative to the repository root. The FITS files and the derived
`cache/` are not committed. `scripts/download_data.py` is ready for a future Google Drive copy; its URL is a documented
placeholder in `configs/data_source.yaml`.

## Layout

```
configs/      default.yaml (all parameters, seed), cv_search.yaml (8 configurations run), cv_search_pending.yaml, data_source.yaml
src/skbg/     calib, events, exposure, features, stacks, model, pipeline, metrics, dataset, io, cache, config, plotstyle
scripts/      00_manifest … 08_report (+ report_body.py), check_linearity.py, download_data.py
notebooks/    exploratory scripts (dev set only) and the MOSKITA calibration cross-check
results/      summary.json, operating_points.csv, baseline_*, cv/, checks*, final_*, explore_*, check_*
figures/      all figures (PNG)
provenance/   numbers.json (generated with the report)
data/         manifest.csv (names, sizes, SHA-256, headers)
```

## Pending work (not done in this session)

* **Gradient boosting (`hgb_full`) and Isolation Forest (`iforest_full`)** were implemented in `src/skbg/model.py`, but their
  cross-validation was not completed. The first attempt hung for 2 h 20 min: OpenMP deadlocked in `fork`ed workers, which was
  fixed by switching to `spawn`. The second attempt was stopped at the user's request after one of four folds (~18 min per fold).
  To run them: `python scripts/03_cv_search.py --search configs/cv_search_pending.yaml`, then `03b_select.py` and the later stages.
  This may change the selected model.
* Test-block ablations (`EXTRA=glm_static_only,glm_dynamic_only,glm_intercept_only ./run_all.sh`) were skipped for time. The
  ablation comparison is therefore from development CV only.
* Excluding track-blind pixels from both E_before and E_after (see the main limitation).
* A PDF version of the report (the course asks for HTML + PDF).

## Environment note

The prompt assumed a pre-made Claude worktree. None existed, so the empty `felidata/TP_AI` repository was cloned inside the
MOSKITA project. Later the Codex agent checked out its own branch in that same clone. This run was then moved to a separate
worktree (`MOSKITA_pipeline/TP_AI_claude`) on the orphan branch above, so Codex's working tree was never modified. No `CODEX/`
file was opened. The only Codex information seen is the commit titles printed once by `git log --all` while diagnosing the
branch change.
