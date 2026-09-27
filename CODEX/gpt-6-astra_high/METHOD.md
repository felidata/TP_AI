# Sideband patch anomaly rejection

## Scope and observable

This is an independent, reproducible detector-region anomaly study on processed
MOSKITA images, not a replacement for the production mask pipeline. No production
hot-column, halo, bleed, SRE, transfer-gate, or hot-zone mask is run, copied, or used
as a training label. The baseline is the geometrically active, calibrated detector.
Consequently, improvements are relative to that baseline, not incremental gains
over the full existing MOSKITA cleaning chain. ROI candidates are treated as a
background proxy; their physical origin is not identified by this experiment.

## Data and calibration

The 218 input files cover paired modules S99 and S101, image IDs 22 through 130.
Each has four 515 by 3200 ADU images. The active region is rows [0,512), columns
[8,3080): 1,572,864 pixels/HDU. HDUs remain in stored orientation; there is no
unverified physical mosaic. Four channels per module are analyzed separately in
diagnostics and share an anomaly detector. Geometry and the overscan convention
come from the parent project's documented instrument conventions.

Subtract each row's median over columns [3100,3200). On the active pixels fit a
two-Gaussian-plus-constant histogram model over [-140,340) ADU in 2 ADU bins.
The Gaussians share width; the second mean is pedestal plus gain. Fit weights are
1/sqrt(max(bin count,1)). The gain is the fitted 0/1e peak separation, and
q=(ADU-pedestal)/gain. Bounds and quality checks are explicit in core.calibrate.
Only the 0/1e calibration peaks enter the fit, not the 4..7e ROI. The constant is
a nuisance approximation to non-Gaussian tails; reduced chi-square is recorded,
not used as proof of a perfect noise model. This is newly written calibration
code, informed by the same physical electron-peak principle used in MOSKITA.

Label 8-connected pixels q>0.7e. A cluster's reconstructed charge Q is the sum of
these above-threshold pixels only. Assign its position to its largest-charge pixel
(first stored pixel breaks exact ties). The ROI is **3.5 <= Q < 7.5e**, the nearest
integer bins 4,5,6,7. This is not the alternative literal continuous interval
[4,7]. Charge below the clustering threshold is not recovered; charge sharing,
noise and accidental merging can change reconstructed multiplicity. All spectra
and counts use this fixed definition. Masks accept/reject a whole pre-reconstructed
cluster by its anchor's tile. They never truncate a cluster or recluster the
surviving pixels. Exposure is therefore an anchor-fiducial exposure, with no
claim of an independently measured efficiency for extended true interactions.

## Effective exposure and sensitivity interpretation

For stored active coordinates (y,x), with x including the 8 prescan columns,
define i=3200*y+x and dt=(DATEEND-DATESTART)/(515*3200). The per-pixel exposure is
t(y,x)=[EXPOSURE+i*dt]/86400 days. Sum t over retained pixels and all images/HDUs.
EXPOSURE is zero, but readout lasts about 4.77 hours: later pixels have more
exposure. This follows the local historical readout-order model in its active
region; uniform clock and zero initial integration are assumptions. The active
geometry here is 3072 columns, not an extrapolation into overscan. Boundary timing,
clock overhead, clearing efficiency and non-exposure-dependent spurious charge
are systematic limitations. Exposure fractions are not simple pixel fractions.
No verified module mass/thickness is available for this run: kg-days are null.

Let N be ROI counts and E be pixel-days. B=N/E. Report f_exp=E1/E0,
r_B=(N1/E1)/(N0/E0), F_equal=f_exp/r_B. Also N1/N0=f_exp*r_B, an independent
consistency identity. Counts and rates must not be interchanged.

Under the additional assumptions of uniform signal generation per pixel-day,
unchanged reconstruction efficiency, a known background, and the large-background
Gaussian counting approximation, S=sE and Z approximately S/sqrt(N)=s*sqrt(E/B).
The relative expected significance is sqrt(f_exp/r_B)=sqrt(F_equal). This is a
conditional sensitivity proxy, not a dark-matter/neutrino exclusion limit. A
collision-correlated or readout-induced signal need not follow E; its acceptance
would require an appropriate signal model, and this proxy would then be invalid.

## Independent method development

Quantitative inspection precedes model fitting. The data show clear electron
peaks, sparse high-charge structures, large module/HDU differences, and a strong
ROI excess in S101 HDU0. Unsupervised **patch** detection is appropriate because
regions, rather than isolated ROI candidates, must be selected. Coarse spatial
patches also limit the ability to select individual target events.

At each 32x32 or 64x64 tile, compute three diagnostics:

1. Number of clusters with 0.7 <= Q < 2.5e, assigned by their anchors.
2. Number of pixels belonging to clusters with Q>20e (not their total charge).
3. Number of pixels with q<-0.7e, tracing electronic/non-Gaussian negative tails.

No positive cluster with 2.5 <= Q <= 20e contributes to these features; individual
pixels of excluded clusters are not repurposed as low-charge labels. Divide
counts by tile area and multiply by 1024. Add training-only temporal mean maps of
the first two features for each module/HDU, and a 3x3 reflected box-smoothed map
of the current first feature. Apply log(1+feature) to all six channels. A transient
diagnostic and a learned persistence map can therefore identify different kinds
of contamination. These are occupancy diagnostics, not estimated physical rates;
the final evaluation separately uses the exposure-time map.

The primary candidate families are (a) the maximum positive robust standardized
score over dynamic channels, (b) the same over all six channels, and (c) Isolation
Forest over all six channels. Robust scales are IQR/1.349 with a 0.15 floor after
the log transform. Forest: 100 trees, 256 training samples/tree, seed 20260927,
one computational thread, default other sklearn 1.6.1 parameters. Use minus
score_samples, larger means more anomalous; sklearn's contamination threshold is
not used. One global forest is fitted to training tiles across eight channels;
no module identity, coordinates, ROI counts or ROI positions are input features.
Persistent maps preserve each module/HDU's detector coordinates.

Each candidate has an absolute threshold from its exposure-weighted training
score quantile, targeting 0,1,2.5,5,10,15,20,30,40 percent removal. Scores equal to
threshold survive, so tied/discrete scores need not attain the target budget.
The threshold is held fixed across validation/test; observed exposure retention
may drift. Entire tiles are removed, not small neighborhoods around ROI events.

Comparisons: static training sideband occupancy, single-channel high/low/negative
scores, and matched-budget Forest feature ablations. The static score is explicitly
a comparison rather than a primary candidate because of its conceptual closeness
to an occupancy-based hot-zone mask. Ablations of Forest omit persistence, high
charge, negative tail or the neighborhood channel, without test-driven selection.

Alternatives considered but not implemented: robust PCA/low-rank plus sparse
decomposition would require a detector stack model and risks classifying ordinary
particle events as sparse defects; no evidence here demands that extra complexity.
Laplacian cosmic-ray rejection identifies sharp individual structures, which is
not the target-region task and could reject particle signals. Autoencoders/CNNs
would require stronger training targets, augmentation and out-of-distribution
checks than these 109 time-correlated acquisitions justify. These are reasoned
rejections, not failed experiments or benchmarked comparisons.

## Validation and uncertainty

Train on image IDs 22..86 (65 acquisitions, 130 files, 520 HDUs). Select on
87..108 (22,44,176). Test once on 109..130 (22,44,176) after committing selection.
Paired modules and all HDUs stay together. ID114 has matching starts and end
times differing by 10 seconds, so matching exact start/end pairs would incorrectly
split an acquisition. File identity is grouped by shared image index. This is a
chronological holdout, not random-pixel validation, and does not establish
generalization to new detectors or another operating campaign.

Choose maximum validation F_equal among primary candidates with retained
exposure >=80%; break ties by greater exposure. This resource constraint and
candidate set are declared before test, and trade-off plots display the full
empirical Pareto frontier (including comparison methods). The selected point is
not claimed to be universally optimal. The test curve is descriptive only: no
test-point retuning. All absolute and relative test metrics are reported.

Uncertainty at the selected point: 2000 paired circular moving-block bootstrap
resamples of the 22 test acquisition groups, block length four. Sum retained and
baseline exposures/counts in each resample before forming ratios. This retains
within-acquisition module correlations and some serial dependence. Percentile
95% intervals condition on the trained/selected model and do not cover detector
systematics or all longer-term drift. Charge-spectrum error bars sqrt(N)/E are
only Poisson illustrations, not independent errors on before/after ratios.

## Checks and limitations

An independent Python flood-fill validates scipy component reconstruction on
synthetic images. Arithmetic series validate exposure sums, and direct catalog
loops verify final counts. Erasing ROI components on all eight HDUs of test
acquisition109 must leave feature arrays exactly unchanged. Synthetic isolated
4..7e injections also leave features unchanged. Uniform exposure-weighted anchor
injections validate mask acceptance but do not calibrate detector efficiency.
Spatially shifting masks breaks spatial alignment as a negative control; because
it preserves HDU-specific masking, it is not expected to return r_B=1 when
different HDUs have different baseline rates. Fixed-mask +/-1% gain and 0.6/0.8e
clustering-threshold checks test reconstructed-charge sensitivity. These are not
full retraining/systematic-coverage studies. Region inference does not prove a
physical mechanism or distinguish unknown signal from background.

## References that influenced choices

- Liu, Ting & Zhou (2008), Isolation Forest, ICDM, doi:10.1109/ICDM.2008.17;
  https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
- scikit-learn IsolationForest implementation/documentation:
  https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.IsolationForest.html
- Candès et al. (2011), Robust Principal Component Analysis?, JACM 58(3);
  https://arxiv.org/abs/0912.3599 (considered, not implemented).
- van Dokkum (2001), Cosmic-Ray Rejection by Laplacian Edge Detection;
  https://arxiv.org/abs/astro-ph/0108003 (considered, not implemented).
- SENSEI (2021), Characterization of Single-Electron Events Using a Skipper-CCD;
  https://arxiv.org/abs/2106.08347 (motivation to distinguish exposure-dependent
  and spurious readout contributions, not calibration imported for these modules).
- UBA course final-project requirements and provenance practice:
  https://github.com/matiaszaldarriaga/GW-AI-course/blob/main/final-project.html
  and day4/session-ai4.html, day5/session-ai5.html, how-a-page-is-built/page.py;
  inspected repository commit 75ba1e0b01df221c7325226d030b22e3a0855e30.
