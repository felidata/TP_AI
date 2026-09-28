# Method

This file defines every quantity and decision. Numbers quoted here are produced by
the scripts named in [PROVENANCE.md](PROVENANCE.md); the results themselves are in
`results/` and in [report.html](report.html).

## 1. From FITS to events

**Calibration** (`src/skbg/calib.py`, written for this project). For each HDU of
each file: subtract a per-row pedestal (median of stored columns 3100–3199, the
reverse overscan; the same *region* MOSKITA uses); fit the histogram of the active
pixels (rows 0–511, columns 8–3079) between −200 and 900 ADU (4 ADU bins) with four
Gaussians at μ₀ + n·g (n = 0…3) sharing one width σ, with free amplitudes
(Poisson-weighted least squares, `scipy.optimize.curve_fit`). Charge q = (x − μ₀)/g.

**Integer charge.** n = floor(q + 0.5), n ≥ 0. With σ ≈ 0.17 e⁻ the 0/1 boundary
sits at ≈2.9σ.

**Events.** An event is an 8-connected component of pixels with n ≥ 1
(`scipy.ndimage.label`). Its charge is Q = Σ n over its pixels; its position is the
charge-weighted centroid, and it belongs to the pixel containing the rounded
centroid.

**ROI ("4–7 e⁻").** Events with integer total charge Q ∈ {4, 5, 6, 7}. There is no
cluster-size cut. Because every pixel is rounded, this is not the same as a cut on
the summed continuous charge. Events whose centroid pixel is masked are removed.

**Checks on this definition.** The calibrated positions of the n = 2…8 e⁻
single-pixel peaks lie within 0.04 e⁻ (mean) and 0.13 e⁻ (worst) of n
(`scripts/check_linearity.py`), so rounding assigns the ROI boundaries correctly.
The fitted gains agree with the pre-existing MOSKITA calibration headers to within
1% at the median. Where they differ by up to 10% (S99 HDU2), the direct peak medians
support the gains fitted here (`notebooks/check_calibration_vs_moskita.py`).

**Exposure.** Readout is continuous (EXPOSURE = 0), so pixel (r, c) of the stored
array has integrated charge for t = T_read·(r·NCOL + c)/(NROW·NCOL), where T_read =
DATEEND − DATESTART (≈ 4.77 h). This is the same physical convention as MOSKITA's
`make_pixel_time_map`, re-implemented in `src/skbg/exposure.py`. Exposure is
reported in pixel·day; 1 pixel·day = 3.537 × 10⁻⁷ g·day, assuming 15 × 15 × 675 µm³
silicon pixels.

**Rates and metrics.** For an evaluated set of quadrant-images, E is the summed
exposure of kept pixels and N the number of kept ROI events.
B = N/E; f_exp = E_after/E_before; r_B = B_after/B_before; F_equal = f_exp/r_B.

## 2. Why this method (evidence from the development data)

The exploration was done on development acquisitions only (`notebooks/explore_*.py`).
It showed two kinds of structure:

1. **Persistent (static) structure.** Stacked 4–7 e⁻ density maps show a bright
   corner and a band of hot columns in S101 HDU0 (≈ 71% of all ROI events). Other
   quadrants have hot-column segments that begin at a fixed row and extend to higher
   rows, which is the signature of a charge-emitting defect that packets pass
   through during vertical transfer. There is also a horizontal band near row 70 in
   S99 HDU2. Block rates are strongly correlated between development and test
   acquisitions (`results/baseline_persistence.csv`).
2. **Transient structure tied to high-energy deposits in the same image.** Pixels
   within 20 px of a cluster with Q ≥ 100 e⁻ hold ≈ 20% of the exposure but 34–43% of
   ROI events (outside S101 HDU0). The excess is 5–8× at 2–5 px and is directional:
   it appears to the right (serial readout direction), to the left, and above
   (vertical transfer direction), but not below the deposit.

Both kinds of structure are spatially smooth or geometrically simple, and they act
at very different scales. A classifier on individual events is ruled out, because
it would label the target itself. The natural object is instead a **rate density**:
ρ(p) = expected number of 4–7 e⁻ events per pixel·day at pixel p. With ρ known,
the mask that minimizes B_after for a given E_after removes pixels in decreasing
order of ρ(p). This is a fractional knapsack: each pixel costs its exposure and
carries ρ·E expected background. So a *good estimate of ρ is the anomaly score*,
and thresholding it traces the background–exposure curve.

## 3. The model: a log-linear inhomogeneous Poisson point process

log ρ(p) = b_q + β · x(p)

Here q is the quadrant (module × HDU), with its own intercept, and x(p) are
standardized covariates (`src/skbg/features.py`):

* **Static covariates** are built from the stacked events of *other acquisitions*
  only. For count map N and exposure map E, each window w gives
  s_w = log[(N_w + k)/(ρ̄_q E_w + k)]: a Gamma–Poisson-shrunk relative rate with
  k = 1 pseudo-count. The windows are 1×1, 5×5, 17×17, 65×65 and 257×257 squares,
  65×1 and full-height column segments, and a 1×65 row segment. Together they let
  the model represent hot pixels, hot columns, hot zones and glow.
* **Same-image covariates** never use the image's ROI events:
  * high-energy-deposit (HEE, Q ≥ 100 e⁻) proximity, computed as one-sided
    exponential trails (length 3 and 15 px, band ±1 px) of log(1 + n) over HEE
    pixels in the +x, −x, +y and −y directions, plus a separable isotropic kernel;
  * single-electron activity (Q = 1 events, which are disjoint from ROI events by
    construction) in the same row and in the same column, relative to the image
    expectation. These respond to serial-register events and to transient hot
    columns.

**Fitting.** The Poisson log-likelihood Σᵢ log ρ(xᵢ) − ∫ρ dE is estimated by
Monte-Carlo quadrature. In each training quadrant-image, 3000 dummy points are
drawn with probability ∝ exposure, each weighted by E_image/3000 (Berman & Turner
1992; Baddeley & Turner 2000). The code adds an L2 penalty (λ = 1 on the
standardized β) and minimizes with scipy's L-BFGS-B using an analytic gradient
(`src/skbg/model.py`; the likelihood code is ours, the optimizer is scipy's).

**Cross-fitting.** Without it, the static covariates of a training image would
contain that image's own events, and the model would over-trust them. So for
training images in fold j, the static maps are built from the other *training*
folds only. For evaluation images they are built from all training folds.

## 4. Validation design (leakage control)

* **Unit of splitting: the complete acquisition** (image number). Both modules and
  all four HDUs of an acquisition always fall in the same group, because they are
  read simultaneously.
* **Development block:** acquisitions 22–97 (76), split into 4 contiguous
  chronological folds of 19. All exploration, feature choices, model comparisons and
  the operating-point choice use out-of-fold predictions from this block only.
* **Held-out test block:** acquisitions 98–130 (33; the S99 file of acquisition 114
  is excluded by the calibration quality rule). The test block is scored once with
  the model fitted on all development folds and the threshold frozen from the
  development CV. Only descriptive baseline figures (stage 2) looked at test images
  before that point; no parameter was taken from them.
* **Quadrant structure** enters through per-quadrant intercepts and per-quadrant
  static maps. The results are broken down per quadrant and recomputed without
  S101 HDU0.
* **Chronological split.** A mask learned from earlier data is applied to later
  data, which is how it would be used in practice. Temporal drift therefore shows up
  as a test-set degradation instead of being averaged away.

**Model selection rule (fixed before the CV results were available).** Choose the
supervised configuration with the smallest S = mean log r_B over
f_exp ∈ {0.80, 0.81, …, 0.99}, pooled out-of-fold. **Operating point:** among
thresholds with f_exp ≥ 0.5, take F_max = max F_equal and choose the *largest*
f_exp with F_equal ≥ 0.95 F_max. Where F_equal is flat, this prefers keeping
exposure. The threshold is stored as an absolute predicted rate (events per
pixel·day), so it transfers unchanged to the test block.

## 5. Alternatives investigated (same covariates, same harness)

| family | configuration | role |
|---|---|---|
| random mask | uniform random score per pixel | area-only control |
| quadrant veto | intercept-only intensity model | coarse baseline: rank whole quadrants |
| static-only | persistent maps from other acquisitions | multiscale hot-region detection |
| dynamic-only | same-image HEE-trail + single-electron covariates | morphology around tracks |
| sideband-trained | target and static maps from 2–3 e⁻ events, never 4–7 e⁻ | target-leakage robustness |
| gradient boosting | `HistGradientBoostingClassifier`, events vs dummies (logistic point-process identity) | non-linear flexibility |
| Isolation Forest | unsupervised on exposure-sampled pixels (Liu et al. 2008) | generic anomaly detection |

Neural networks (autoencoders or CNNs) were not trained. The structures are
low-dimensional and already captured by explicit covariates. There are only 76
development acquisitions, and a network would need its own cross-fitting scheme to
avoid memorizing hot pixels. The report gives the results of each alternative.

## 6. Uncertainties and sensitivity proxy

Uncertainties on f_exp, r_B and F_equal come from a **bootstrap over complete
acquisitions** of the evaluated block (500–2000 resamples; 16th/84th
percentiles). Resampling acquisitions keeps the over-dispersion between images and
the correlations between the quadrants of one acquisition.

**Sensitivity proxy.** Assume a signal that is uniform per unit exposure with rate
s in the 4–7 e⁻ window, and a known background rate b. For a counting experiment in
the background-dominated regime (N_B ≫ 1), the median significance is
Z ≈ sE/√(bE) = s√(E/b). The gain from a mask is therefore
Z_after/Z_before = √(f_exp / r_B) = √F_equal. F_equal is thus not an arbitrary
equal weighting: it is the squared gain in background-dominated sensitivity,
under these assumptions:

* the signal has the same efficiency in masked and kept pixels;
* the background is known, with no systematic uncertainty;
* N_B ≫ 1 after masking.

The test block keeps thousands of ROI events after masking, so the last assumption
holds. In the zero-background limit the gain would instead be f_exp, and with a
dominant fractional background systematic it would be 1/r_B. These limits bracket
the proxy. The code also provides the Asimov formula
Z = √(2[(s+b)ln(1+s/b) − s]) (Cowan et al. 2011) in `metrics.asimov_Z`.

## 7. Limitations known in advance

* The ROI events a mask removes are not inspected event by event, so a real signal
  that happened to cluster near HEE or defects would be removed too. This is an
  efficiency question for a signal model, not for the background.
* HEE footprints and their 1-px border are intrinsically blind to 4–7 e⁻ events,
  because such events would merge with the track. That blind exposure is counted in
  E_before, as in a plain pixel-count exposure. Masking it costs "exposure" that was
  never sensitive, so f_exp is slightly conservative (it understates the retained *useful* exposure).
* The static maps are sensor-specific and must be re-derived for another module.
