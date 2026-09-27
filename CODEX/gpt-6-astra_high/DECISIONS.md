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
One paired acquisition has a ten-second timestamp discrepancy (ID114): group by matching
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

## Held-out result and checks

Frozen Isolation Forest tile32, threshold0.5133267106012511: test f_exp0.819080,
r_B0.214287, F_equal3.822350; 22237 -> 3903 ROI clusters. Exposure22.4191 of
27.3711 million pixel-days. First test acquisition ROI erasure:941 clusters,
identical six diagnostic inputs after fixed calibration/reconstruction. Independent
exposure arithmetic agrees to3.3e-15 relative. No retuning after test.

Negative results: negative-tail-only score is weak. Post-freeze ablations show
no-negative and no-neighbor descriptors improve both validation and test at the
same training budget (test F3.916 and3.971 vs3.822). Thus the full feature set is
not optimal. These are follow-up hypotheses for new data, not replacements for
the frozen result. Removing persistence/high-charge features worsens the result.
S101HDU0 dominates the pooled gain; preserving80% globally does not preserve80%
per quadrant. Fixed-mask gain+/-1% and thresholds0.6/0.8 leave r_B near0.214.

Course-inspired verification is separate arithmetic/code paths in this same
model's run; it is not claimed to be independent human or alternate-model review.
Only a clean-cache repeat in the observed environment is tested, not a new OS.

## Visual QA finding during clean reproduction

The gain-versus-acquisition figure exposed a transient in all four S99 HDUs at
image114: fitted gains174.57,170.27,180.51,179.44 ADU/e. Input ADU histograms for
IDs113/114/115 show a shifted1e feature (e.g. HDU0 bin maxima218/178/206 ADU);
this is not just a plot artifact. Its paired end timestamps differ by10seconds.
No causal explanation is established. Fits pass the original declared checks;
we retain the original sample and add a paired-acquisition exclusion diagnostic,
not a revised main result. The first end-to-end repeat was still running when
this diagnostic/report addition was made; the unchanged core numeric products
are compared to the first-run reference.

## Final reproduction and presentation QA

Moved the first-run cache outside the repository; ran reproduce.sh with no
analysis cache. Input hashes verified; all5 unit checks and all artifact checks
passed. Fifteen numerical/table/manifest/figure-hash products reproduced exactly,
excluding naturally changing runtime and git metadata. Record:
results/reproduction_check.json; complete run log: results/reproduction.log.
This is the same installed environment, not a fresh package installation.

Viewed all8 rendered PDF pages, representative PNGs and the HTML in Firefox.
The initial headless Firefox profile under /tmp was inaccessible to the Snap
package; a disposable ignored cache profile resolved the preview issue. No
scientific result depended on the browser. Report PNGs reproduced byte-identically.
