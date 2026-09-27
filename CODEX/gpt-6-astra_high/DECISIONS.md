# Research log

## Before fitting methods (2026-09-27)

Runtime turn_context confirms gpt-6-astra, high. Backend build is not exposed.
Initial checkout had no commits and was on claude/claude-opus-5-5_high, contrary
to the prompt's setup description. Created codex/gpt-6-astra_high; CLAUDE untouched.
Prompt was extracted verbatim from the session user message, not paraphrased.

Read the course final-project, day4 and day5 material from repository commit
75ba1e0b01df221c7325226d030b22e3a0855e30. Requirements emphasize provenance AND
checks, readable HTML, reproducibility, and a PDF for presentation. A separate
verification implementation will provide computational checks; this single-model
run cannot claim an independent human or different-model scientific review.

218 files; 109 matched image IDs in each of S99 and S101, four HDUs each.
One paired acquisition has a one-second timestamp discrepancy: group by matching
image index, never by exact timestamp or random pixels. Keep all eight HDUs in
the same split. Training IDs 22..86 (65), validation 87..108 (22), test 109..130 (22).
Initial quantitative samples: image22 both modules and image100 S101/image99 S99
(all development). No test charge inspection before selection.

Geometry [0:512,8:3080], row-median reverse overscan [3100:3200], readout-index
exposure informed by parent reading_skipper_fits.md, calibration.py, exposure.py.
No parent code is imported. No existing mask outputs/optimized masks are used.
Two-peak weighted Gaussian calibration implemented independently; gain is ~210
ADU/e and noise ~0.17e on initial samples. Mass unavailable: exposure in pixel-days.
EXPOSURE=0 does not mean zero effective exposure: timestamp readout lasts ~4.77h.

Before optimizing, fix connected-component reconstruction: 8-neighbor >0.7e,
sum above-threshold calibrated charges, ROI [3.5,7.5), maximum-pixel anchor.
Masks operate on entire 32x32 or 64x64 tiles; events evaluated by anchor, with
no reclustering. This is an explicit reconstructed-charge/anchor fiducial result,
not a detector-corrected interaction rate or a calibrated signal-efficiency result.

Explore sideband-only patch diagnostics: 0.7..2.5e component counts, occupied
pixels of components >20e, and negative noise <-0.7e. Intermediate charges never
enter diagnostic channels. Persistent maps learned from training only; dynamic
patch scores and Isolation Forest provide alternatives. ROI catalogs are used
only to evaluate configurations on validation, never as per-event mask labels.
Compare static sideband density, robust multivariate dynamic scores, combined
persistence/dynamic scores, and Isolation Forest plus feature ablations.

Freeze candidate grid before test: training-score quantiles at 0..40% masking,
tile sizes 32/64; select among multivariate robust dynamic/combined or Isolation
Forest candidates with f_exp>=0.8 and maximum F_equal (ties favor exposure).
Static sideband ranking and one-channel ablations are comparisons only, since a
static occupancy cut alone is too close to the existing hot-zone approach to be
the primary contribution. Display the global validation/test Pareto frontier even
if a comparison dominates an eligible primary candidate. The constraint is a declared working preference,
not universal experimental optimum. Always show all operating points and frontier.
