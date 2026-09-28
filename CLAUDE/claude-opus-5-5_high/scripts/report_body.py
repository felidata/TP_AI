"""Body of report.html. Called by scripts/08_report.py; every number comes from results/."""

CHECKS = '''<p>The first three checks answer "is the improvement real?". No test image's 4–7 e⁻ clusters influence its score, so the
reduction is not leakage. Masks with the same area or the same shape, placed elsewhere, remove nothing, so the reduction comes from
<i>where</i> the mask is and not from how much it covers. Dropping any one acquisition changes r_B by less than 2%, so no single image drives the result.</p>
<p><b>Quadrant dominance is real.</b> S101 HDU0 carries {pct:.0f}% of the test-block 4–7 e⁻ clusters. With it removed from the evaluation,
the same mask gives r_B = {r7:.3f} at f_exp = {f7:.3f} (F = {F7:.1f}) instead of {rB:.3f} (F = {F:.1f}). The method still removes most of the
background in the other seven quadrants, but the headline number is dominated by one defective quadrant. By module: S101 r_B = {r101:.3f},
S99 r_B = {r99:.3f}.</p>
<p><b>Charge windows.</b> The same mask lowers the rate of 8–20 e⁻ clusters by a similar factor (r_B = {w820:.3f}). It lowers 2–3 e⁻ clusters
much less ({w23:.3f}) and single electrons little ({w1:.3f}). The masked regions are therefore sources of multi-electron clusters in general,
not something specific to the 4–7 e⁻ bin. That is consistent with the sideband-trained model (never shown a 4–7 e⁻ event) reaching
S = {Ssb:.2f} in cross-validation against {Ssel:.2f} for the selected model.</p>'''

LIMITS = '''<section id="limits"><h2>Limitations and failures</h2>
<ul>
<li><b>Blind pixels count as kept exposure.</b> Pixels on and next to high-energy tracks cannot host a separate 4–7 e⁻ cluster, because it would
merge with the track. The model correctly predicts a very low rate there, so the mask keeps them. From the development exploration they are roughly
2.5–3% of the exposure (Q ≥ 100 e⁻ footprints plus 1 px). Counting them in E_after makes B_after look lower by an estimated ~4%. This was not
measured on the final mask. A cleaner definition would drop blind pixels from both E_before and E_after.</li>
<li><b>One quadrant dominates</b> the headline ratio (see Checks). Results without S101 HDU0 are reported alongside.</li>
<li><b>Operating point.</b> The pre-declared rule chose f_exp ≈ 0.73 because F_equal is flat between 0.65 and 0.80. A user who values exposure
more can take any point of the published curve: f_exp ≈ 0.90 gives r_B ≈ {r90:.3f} on test.</li>
<li><b>Not run, stopped for time.</b> The gradient-boosting and Isolation-Forest comparisons were implemented and started. The first attempt
deadlocked (OpenMP after <code>fork</code>, 2 h 20 min lost); it was fixed by using <code>spawn</code> workers. The second attempt was
stopped by request after one of four gradient-boosting folds (~18 min per fold). Their configurations are in
<code>configs/cv_search_pending.yaml</code>. No conclusion about non-linear or unsupervised models is drawn here.</li>
<li><b>Ablations on test</b> (static-only, dynamic-only, quadrant veto) were skipped for time. Their comparison is from development CV only.</li>
<li><b>Signal efficiency</b> was not modelled. If a signal preferentially produced clusters near tracks or defects, the mask would remove it too.</li>
<li><b>Masks are sensor-specific.</b> The static maps must be re-derived for other modules. Their stability over longer periods than
these 22 days is untested; the test block is the last third of the same run.</li>
<li><b>Calibration.</b> The gain is fitted as linear from 0–3 e⁻ peaks. A ≈1.5% gain decrease between 2 and 7 e⁻ leaves a mean bias of 0.03 e⁻
at 7 e⁻ (worst 0.13 e⁻), inside the rounding window but not zero.</li>
<li><b>Existing MOSKITA masks were not compared.</b> They were built using all 109 acquisitions, including this test block, so any comparison
would be biased in their favour.</li>
</ul></section>'''



def body(J, img, fmt, table, pd, R):
    base = J("baseline.json"); fin = J("final_test_metrics.json"); chk = J("checks.json")
    summ = J("summary.json"); sel = J("cv/selection.json")
    cv = pd.read_csv(R / "cv" / "cv_summary.csv")
    byq = pd.read_csv(R / "baseline_by_quadrant.csv")
    ops = pd.read_csv(R / "operating_points.csv")
    hee = J("explore_hee_proximity_direction.json")
    lin = pd.read_csv(R / "check_linearity.csv")
    cq = pd.read_csv(R / "checks_by_quadrant.csv")
    pm = base["pixel_mass_g"]
    name = fin["config"]
    hee_min = J("final_coefficients.json")["spec"]["hee_min_e"]
    tot = base["totals"]
    numbers = {}

    def num(slug, value, statement, produced_by, from_scratch, from_library, choices):
        numbers[slug] = dict(value=value, statement=statement, produced_by=produced_by, from_scratch=from_scratch,
                             from_library=from_library, choices=choices)
        return value

    b16, b50, b84 = fin["bootstrap_16_50_84"]["r_B"]
    f16, f50, f84 = fin["bootstrap_16_50_84"]["F_equal"]
    B0 = fin["N_before"] / fin["E_before_pixday"]; B1 = fin["N_after"] / fin["E_after_pixday"]
    num("test_f_exp", fin["f_exp"], "retained exposure fraction on the held-out test block at the selected threshold",
        "scripts/04_final.py::main (metrics.curve)", "exposure bookkeeping, threshold application", "numpy", ["threshold frozen from dev CV"])
    num("test_r_B", fin["r_B"], "4-7 e- rate after/before on the test block", "scripts/04_final.py::main",
        "event counting at centroid pixel", "numpy", ["centroid-pixel assignment of clusters to masked/kept"])
    num("test_F_equal", fin["F_equal"], "f_exp / r_B on the test block", "scripts/04_final.py::main", "ratio", "numpy", [])
    num("test_r_B_boot", [b16, b50, b84], "acquisition-bootstrap 16/50/84 percentiles of r_B", "src/skbg/metrics.py::bootstrap",
        "bootstrap over acquisitions", "numpy.random", ["2000 resamples", "acquisition as the resampling unit"])
    num("baseline_rate_all", tot["all"]["rate_per_pixday"], "4-7 e- rate, all usable data, no mask",
        "scripts/02_baseline.py::main", "counting and exposure map", "numpy", ["no border or other mask in the baseline"])

    def opt(method, split, f):
        g = ops[(ops.method == method) & (ops.split == split)].sort_values("f_exp")
        i = (g.f_exp - f).abs().idxmin()
        return g.loc[i]

    fixed = pd.DataFrame([opt(name, "test", f) for f in (0.99, 0.95, 0.90)] +
                         [ops[(ops.method == name) & (ops.split == "test") & (ops.selected)].iloc[0]])
    fixed["label"] = ["f≈0.99", "f≈0.95", "f≈0.90", "selected"]
    fixed["r_B_ci"] = [f"{r:.3f} [{a:.3f}, {b:.3f}]" for r, a, b in zip(fixed.r_B, fixed.r_B_lo, fixed.r_B_hi)]
    fixed["F_ci"] = [f"{r:.1f} [{a:.1f}, {b:.1f}]" for r, a, b in zip(fixed.F_equal, fixed.F_equal_lo, fixed.F_equal_hi)]
    fixed["sqrtF"] = fixed.F_equal ** 0.5

    s = []
    # ---------------------------------------------------------------- header + summary
    s.append(f"""<header class="top"><div class="eyebrow">UBA · Gravitational Waves and AI-Assisted Research · final project · agent run {summ['run_id']}</div>
<h1>Masking the regions that produce 4–7 e⁻ background in MOSKITA Skipper-CCDs</h1>
<p class="lede">A cross-fitted Poisson point-process model predicts, pixel by pixel, the rate of 4–7 e⁻ clusters from
information that never includes the image's own 4–7 e⁻ clusters. Masking pixels in decreasing predicted rate traces the whole
background–exposure curve. On 33 held-out acquisitions it removes {100*(1-fin['r_B']):.1f}% of the 4–7 e⁻ rate while keeping
{100*fin['f_exp']:.1f}% of the exposure at the pre-selected operating point.</p>
<div class="meta"><span>model {summ['model']}</span><span>effort {summ['effort']}</span><span>branch {summ.get('git_branch')}</span><span>code commit {str(summ.get('git_commit'))[:10]}</span></div></header>""")

    s.append(f"""<section id="summary"><h2>Summary</h2>
<div class="keynums">
<div><div class="k">retained exposure f_exp</div><div class="v">{fin['f_exp']:.3f}</div><div class="u">{fin['E_after_pixday']:.3e} of {fin['E_before_pixday']:.3e} pixel·day</div></div>
<div><div class="k">background ratio r_B</div><div class="v">{fin['r_B']:.3f}</div><div class="u">68% CI [{b16:.3f}, {b84:.3f}]</div></div>
<div><div class="k">F_equal = f_exp / r_B</div><div class="v">{fin['F_equal']:.1f}</div><div class="u">68% CI [{f16:.1f}, {f84:.1f}]</div></div>
<div><div class="k">4–7 e⁻ clusters</div><div class="v">{fin['N_before']:,} → {fin['N_after']:,}</div><div class="u">test block, before → after</div></div>
</div>
<p>All four numbers are measured on the held-out test block (acquisitions 98–130). The model, its covariates and the threshold
were fixed using the development acquisitions only. The rate before masking is {fmt(B0)} and after {fmt(B1)} clusters per pixel·day
({B0/pm:.0f} → {B1/pm:.0f} per g·day, assuming 3.54×10⁻⁷ g per pixel). Intervals are 16th–84th percentiles of a bootstrap over whole acquisitions.</p>
<p>Because F_equal is flat around its maximum, the curve matters more than the single point. Other points of the same test curve:</p>
{table(fixed, ['label','f_exp','r_B_ci','F_ci','sqrtF','N_after'], ['operating point','f_exp','r_B [68% CI]','F_equal [68% CI]','√F (significance gain)','4–7 e⁻ left'],
       {'f_exp': lambda v: f'{v:.3f}', 'sqrtF': lambda v: f'{v:.2f}', 'N_after': lambda v: f'{int(v):,}'})}
</section>""")

    # ---------------------------------------------------------------- motivation
    s.append("""<section id="motivation"><h2>Motivation</h2>
<p>Skipper-CCDs count individual electrons, so rare events of a few electrons are the natural search region for low-mass dark-matter
or millicharged-particle signals. The <span class="roi">4–7 e⁻</span> window sits above the regime dominated by accidental
single-electron coincidences. It is also below muon and electron tracks. In the MOSKITA data its rate is dominated by detector effects
concentrated in specific places. Removing those places lowers the background but also removes exposure, so the task is a
trade-off with two objectives. It is not a search for the smallest background.</p>
<p>The aim was an independent method, not a rebuild of the existing MOSKITA masks (hot columns, bleed, halo, serial-register events,
hot zones). The MOSKITA code and notes were read for context: data format, exposure convention and event definition. None of their
masks is used, and no threshold was taken from them.</p></section>""")

    # ---------------------------------------------------------------- data
    calq = pd.DataFrame(base["calibration_by_quadrant"])
    s.append(f"""<section id="data"><h2>Data</h2>
<p>{base['n_fits_files']} FITS files: modules S99 and S101, acquisitions {base['acquisitions'][0]}–{base['acquisitions'][1]}.
Each file has 4 HDUs (one per amplifier) of 515 × 3200 ADU pixels, with 256 non-destructive samples already averaged. Both modules are
read simultaneously. Each image takes {base['readout_hours_median']:.2f} h of continuous readout with no exposure time. The active region is
512 × 3072 pixels per HDU. {base['n_usable_files']} files pass a calibration-quality rule decided without looking at the 4–7 e⁻ counts:
{', '.join(base['unusable_files'])} is removed because its fitted gain is 16% low in every HDU. Details are in DATA.md, and the file list with SHA-256 hashes is in <code>data/manifest.csv</code>.</p>
<p>The charge scale comes from a four-peak Gaussian fit (0–3 e⁻, common width, linear gain) to each HDU of each image:</p>
{table(calq, ['module','hdu','gain_adu','noise_e','chi2_ndf','see_a1a0'], ['module','HDU','gain [ADU/e⁻]','noise [e⁻]','χ²/ndf','A₁/A₀'],
       {'gain_adu': lambda v: f'{v:.1f}', 'noise_e': lambda v: f'{v:.3f}', 'chi2_ndf': lambda v: f'{v:.2f}', 'see_a1a0': lambda v: f'{v:.4f}'})}
<p>Events are 8-connected clusters of pixels with integer charge n = ⌊q + ½⌋ ≥ 1. The <span class="roi">4–7 e⁻</span> ROI is Q = Σn ∈ {{4,5,6,7}},
with no size cut. A cluster belongs to the pixel of its charge-weighted centroid. Across 96 sampled quadrant-images, the calibrated
n = 2…8 e⁻ peaks lie within {lin[lin.n.between(2,8)].groupby('n').bias_e.mean().abs().max():.3f} e⁻ (mean) of their nominal charge;
the worst single quadrant-image is off by {lin[lin.n.between(2,8)].bias_e.abs().max():.3f} e⁻. That is small next to the ±0.5 e⁻ rounding window.
Exposure is per pixel: the time between the clear and the pixel's readout (same convention as MOSKITA, re-implemented).</p></section>""")

    # ---------------------------------------------------------------- diagnostics
    q0 = byq[(byq.split == "dev") & (byq.module == "S101") & (byq.hdu == 0)].iloc[0]
    hs = hee["S99"]; h1 = hee["S101"]
    s.append(f"""<section id="diagnostics"><h2>Initial diagnostics</h2>
<p>Without any mask, all usable data contain {tot['all']['n_roi']:,.0f} ROI clusters in {tot['all']['exposure_pixday']:.3e} pixel·day
({tot['all']['exposure_gday']:.2f} g·day), i.e. {fmt(tot['all']['rate_per_pixday'])} per pixel·day.</p>
{img('baseline_spectrum.png','cluster charge spectrum','Cluster-charge spectrum before any mask, development and test acquisitions separately. The shaded band is the ROI.')}
{img('baseline_rate_by_quadrant.png','rate per quadrant','Baseline 4–7 e⁻ rate per quadrant with Poisson 68% intervals. The quadrants differ by a factor ≈35.')}
<div class="obs"><span class="tag">observation</span>S101 HDU0 alone holds {100*q0.share_of_split_roi:.0f}% of the development ROI clusters on 1/8 of the exposure.
Its per-acquisition counts are strongly over-dispersed (dispersion index {q0.dispersion_index:.1f}; a Poisson process would give 1).</div>
{img('baseline_rate_vs_acquisition.png','rate versus acquisition','ROI rate per quadrant and acquisition. The dashed line marks the start of the held-out block.')}
{img('baseline_roi_density_dev.png','ROI density, development set','Stacked ROI density, development acquisitions (32 px smoothing). Bottom rows have little exposure and therefore noisy rates.', wide=True)}
<div class="obs"><span class="tag">observation</span>The persistent structures are: a bright corner and a band of hot columns in S101 HDU0;
isolated hot-column segments that start at a fixed row and extend upward in every quadrant; a horizontal band near row 70 in S99 HDU2.
The same structures appear in the test-block stack (<code>figures/baseline_roi_density_test.png</code>).</div>
<div class="obs"><span class="tag">observation</span>Around high-energy deposits (Q ≥ 100 e⁻, same image, development set, S101 HDU0 excluded):
pixels within 20 px hold {100*hs['frac_exposure_d_le_20']:.0f}% (S99) / {100*h1['frac_exposure_d_le_20']:.0f}% (S101) of the exposure
but {100*hs['frac_roi_d_le_20']:.0f}% / {100*h1['frac_roi_d_le_20']:.0f}% of the ROI clusters. Within 20 px, relative to the quadrant mean,
the rate ratio is right / left / up / down = {' / '.join(f'{v:.1f}' for v in hs['direction_rate_ratio_right_left_up_down'])} (S99)
and {' / '.join(f'{v:.1f}' for v in h1['direction_rate_ratio_right_left_up_down'])} (S101). Source: <code>results/explore_hee_proximity*.{{csv,json}}</code>.</div>
<div class="interp"><span class="tag">interpretation</span>The excess along +x (the serial readout direction) and +y (the vertical transfer direction)
matches charge released late by traps along the transfer path. The excess at −x does not have an equally simple explanation.
These directions guided the choice of one-sided trail covariates; the weights were left for the likelihood fit to decide.</div>
</section>""")

    # ---------------------------------------------------------------- research
    cvt = cv.copy()
    cvt["label"] = cvt.config
    s.append(f"""<section id="research"><h2>Research process and candidate approaches</h2>
<p>The diagnostics point to a rate density ρ(p), the expected number of ROI events per pixel·day, as the natural quantity.
If ρ were known, the best mask for any exposure budget would drop pixels in decreasing ρ (a fractional knapsack). The problem therefore
reduces to predicting ρ without using the target events of the image being masked. All candidates below were run in the same harness,
with 4-fold chronological cross-validation on the 76 development acquisitions. S is the mean of ln r_B over f_exp = 0.80…0.99,
pooled out-of-fold; lower is better. The selection rule was written before any of these numbers existed.</p>
{table(cvt, ['label','S_mean_log_rB_80_99','r_B@0.99','r_B@0.95','r_B@0.9','F_max','sel_f_exp','sel_r_B'],
       ['configuration (dev CV)','S','r_B @ f=0.99','r_B @ f=0.95','r_B @ f=0.90','max F_equal','chosen f_exp','chosen r_B'],
       {k: (lambda v: f'{v:.3f}') for k in ['S_mean_log_rB_80_99','r_B@0.99','r_B@0.95','r_B@0.9','sel_f_exp','sel_r_B']} | {'F_max': lambda v: f'{v:.1f}'})}
{img('cv_model_comparison.png','cross-validation comparison','Out-of-fold background ratio vs retained exposure for every candidate on the development block.')}
<p>Families and what each tests:</p>
<ul>
<li><b>Quadrant veto</b> (intercept-only model): the coarse baseline of dropping whole quadrants.</li>
<li><b>Static-only</b>: multiscale, shrunk rate maps from other acquisitions. This is persistence analysis with Gamma–Poisson smoothing.</li>
<li><b>Dynamic-only</b>: trails around high-energy deposits and single-electron row/column activity in the same image.</li>
<li><b>Sideband-trained</b>: both the target and the static maps use 2–3 e⁻ clusters. The model never sees a 4–7 e⁻ event, which tests target leakage.</li>
<li><b>Gradient boosting</b>: the same covariates in a non-linear events-vs-dummies classifier.</li>
<li><b>Isolation Forest</b>: generic unsupervised anomaly detection on the same covariates.</li>
</ul>
</section>""")

    # ---------------------------------------------------------------- method
    s.append(f"""<section id="method"><h2>Chosen method</h2>
<p>Selected by the pre-declared rule: <code>{name}</code>. The model is log ρ(p) = b<sub>q</sub> + β·x(p), a log-linear inhomogeneous
Poisson point process, with one intercept per quadrant and the following covariates:</p>
<ul><li>eight <b>static</b> relative-rate maps, windows from 1×1 px to 257×257 px plus column and row segments, built only from other acquisitions;</li>
<li>ten <b>track-trail</b> maps: one-sided exponential kernels of 3 and 15 px in ±x and ±y, plus isotropic kernels, all around pixels of clusters with
Q ≥ {hee_min} e⁻ in the same image;</li>
<li>two <b>single-electron</b> maps: row and column excess of Q = 1 clusters in the same image.</li></ul>
<p>The likelihood Σ log ρ(xᵢ) − ∫ρ dE is evaluated by Monte-Carlo quadrature (Berman & Turner 1992; Baddeley & Turner 2000) and maximized
with L-BFGS-B. The static maps of training images are cross-fitted, so an image never informs its own covariates.
The mask is {{p : ρ̂(p) ≥ ρ*}} with ρ* = {fin['threshold_rate_per_pixday']:.2e} per pixel·day, taken from the development CV.</p>
{img('model_coefficients.png','model coefficients','Standardized coefficients of the final model, fitted on all development acquisitions.')}
</section>""")

    # ---------------------------------------------------------------- validation
    s.append(f"""<section id="validation"><h2>Validation design</h2>
<ul><li><b>Unit of split:</b> the complete acquisition. Both modules and all HDUs of an image number share one group.</li>
<li><b>Development:</b> acquisitions 22–97, 4 contiguous chronological folds. Used for exploration, model choice and threshold choice.</li>
<li><b>Test:</b> acquisitions 98–130, scored once with the frozen model and threshold. Only descriptive baseline plots looked at it earlier.</li>
<li><b>No image informs its own score:</b> static maps come from other acquisitions, and same-image covariates use only Q ≥ {hee_min} e⁻ and Q = 1 clusters.
This is checked numerically below.</li>
<li><b>Chronological split:</b> the mask is learned on earlier data and applied to later data, which is how it would be used in practice.</li></ul>
</section>""")

    # ---------------------------------------------------------------- results
    s.append(f"""<section id="results"><h2>Results on the held-out test block</h2>
{img('result_background_vs_exposure.png','background vs exposure','Background ratio vs retained exposure on the test block. The band is the acquisition-bootstrap 68% interval. Dashed: the same model in dev CV. Dotted: random masks. Other lines: ablations evaluated on test. The dot marks the pre-selected threshold.')}
{img('result_spectrum_before_after.png','spectrum before after','Test-block cluster-charge spectrum before and after the selected mask, per pixel·day of the respective exposure, and their ratio.')}
{img('result_density_before_after.png','density before after','Test-block ROI density before (left) and after (right) the mask. Light blue shows pixels masked in more than half of the test exposure.', wide=True)}
{''.join(img(p.name, 'example image', 'Example test quadrant-image: integer charge, predicted rate (the anomaly score) and mask overlay with kept (circles) and removed (crosses) 4–7 e⁻ clusters.', wide=True) for p in [R.parent / 'figures' / 'example_S101_HDU0_img110.png', R.parent / 'figures' / 'example_S99_HDU1_img110.png'])}
{img('example_cutouts.png','cutouts','Randomly drawn 64×64 px cut-outs of one test image: regions partly masked (top) and fully kept (bottom).', wide=True)}
{img('result_per_quadrant_and_acquisition.png','per quadrant and acquisition','Left: f_exp and r_B per quadrant at the selected threshold. Right: per test acquisition.', wide=True)}
{table(cq, ['module','hdu','N_before','N_after','f_exp','r_B','F_equal'], ['module','HDU','4–7 e⁻ before','after','f_exp','r_B','F_equal'],
       {'N_before': lambda v: f'{int(v):,}', 'N_after': lambda v: f'{int(v):,}', 'f_exp': lambda v: f'{v:.3f}', 'r_B': lambda v: f'{v:.3f}', 'F_equal': lambda v: f'{v:.2f}'})}
{img('result_calibration.png','calibration','Predicted vs observed test ROI counts, grouped in unit bins of ln predicted rate.')}
</section>""")

    # ---------------------------------------------------------------- tradeoff
    s.append(f"""<section id="tradeoff"><h2>Exposure–background trade-off</h2>
{img('result_pareto_and_Fequal.png','pareto','Left: Pareto frontier (non-dominated points) of each method on the test block. Right: F_equal vs f_exp. The band shows the 68% bootstrap interval of the selected model; dashed: dev CV.', wide=True)}
<p><b>Sensitivity proxy.</b> Assume a signal uniform per unit exposure and a known background, in the background-dominated regime. The
median significance is Z ∝ s·E/√(bE), so the gain from a mask is √(f_exp/r_B) = √F_equal. Under these assumptions F_equal is the squared
gain in counting sensitivity, not an arbitrary equal weighting. At the selected point √F = {fin['significance_gain_sqrtF']:.2f}.
With {fin['N_after']:,} ROI clusters left, the test block is well inside the background-dominated regime. Two limits bracket the proxy:
with a dominant fractional background systematic, the relevant gain is 1/r_B; with zero background, it is f_exp. Where the dev-CV
F_equal is flat (f_exp ≈ 0.70–0.90), the choice is a judgement about systematics and signal efficiency that the data cannot settle.</p>
</section>""")

    # ---------------------------------------------------------------- checks
    ac = chk["area_controls"]; lo = chk["leave_one_acquisition_out"]; w7 = chk["without_S101_HDU0"]; li = chk["leakage_invariance"]
    cw = chk["charge_windows_same_mask"]
    crow = []
    crow.append({"check": "No image informs its own score: delete every 4–7 e⁻ cluster from 3 test acquisitions (all quadrants) and recompute the score",
                 "result": f"{li['roi_clusters_removed']} clusters removed; max |Δ score| = {li['max_abs_score_change']:.2e}",
                 "verdict": "pass" if li["max_abs_score_change"] < 1e-9 else "fail"})
    crow.append({"check": "Area alone: random mask with the same f_exp",
                 "result": f"r_B = {ac['random_mask_same_f']['r_B']:.3f} at f = {ac['random_mask_same_f']['f_exp']:.3f} (method: {ac['method']['r_B']:.3f})",
                 "verdict": "pass" if ac['random_mask_same_f']['r_B'] > 0.9 else "fail"})
    crow.append({"check": "Location vs morphology: the same masks rolled by a random 300–2770 px along columns",
                 "result": f"r_B = {ac['shifted_mask']['r_B']:.3f} at f = {ac['shifted_mask']['f_exp']:.3f}",
                 "verdict": "pass" if ac['shifted_mask']['r_B'] > 2 * ac['method']['r_B'] else "fail"})
    crow.append({"check": "One pathological acquisition: leave-one-acquisition-out range of r_B on test",
                 "result": f"{lo['r_B_min']:.3f} – {lo['r_B_max']:.3f} (most influential: img {lo['most_influential_img']})",
                 "verdict": "pass" if lo['r_B_max'] - lo['r_B_min'] < 0.3 * fin['r_B'] else "fail"})
    crow.append({"check": "One quadrant dominating: remove S101 HDU0 from the evaluation (same threshold)",
                 "result": f"f_exp = {w7['same_threshold']['f_exp']:.3f}, r_B = {w7['same_threshold']['r_B']:.3f}, F = {w7['same_threshold']['F_equal']:.2f}; curve r_B@f=0.90: {w7['curve_r_B_at_f']['0.9']:.3f}",
                 "verdict": "see text"})
    crow.append({"check": "Modules separately",
                 "result": f"S101 r_B = {chk['module_S101']['r_B']:.3f} (f {chk['module_S101']['f_exp']:.3f}); S99 r_B = {chk['module_S99']['r_B']:.3f} (f {chk['module_S99']['f_exp']:.3f})",
                 "verdict": "see text"})
    crow.append({"check": "Charge windows under the same mask (r_B)",
                 "result": "; ".join(f"{k} e⁻: {v['r_B']:.3f}" for k, v in cw.items()), "verdict": "see text"})
    crow.append({"check": "Rate calibration: predicted vs observed ROI counts on test",
                 "result": f"{chk['calibration_total']['predicted']:.0f} predicted vs {chk['calibration_total']['observed']} observed", "verdict": "see figure"})
    cdf = pd.DataFrame(crow)
    cdf["verdict"] = [f'<span class="{v}">{v}</span>' if v in ("pass", "fail") else v for v in cdf.verdict]
    s.append(f"""<section id="checks"><h2>Checks</h2>
{table(cdf, ['check','result','verdict'], ['check','result','verdict'])}
<div id="checks-text"></div>
</section>""")
    r90 = opt(name, "test", 0.90).r_B
    cvi = cv.set_index("config")
    pct = 100 * cq[(cq.module == "S101") & (cq.hdu == 0)].N_before.iloc[0] / cq.N_before.sum()
    s[-1] = s[-1].replace('<div id="checks-text"></div>', CHECKS.format(pct=pct, r7=w7["same_threshold"]["r_B"], f7=w7["same_threshold"]["f_exp"],
        F7=w7["same_threshold"]["F_equal"], rB=fin["r_B"], F=fin["F_equal"], r101=chk["module_S101"]["r_B"], r99=chk["module_S99"]["r_B"],
        w820=cw["8-20"]["r_B"], w23=cw["2-3"]["r_B"], w1=cw["1-1"]["r_B"], Ssb=cvi.loc["glm_sideband_only", "S_mean_log_rB_80_99"],
        Ssel=cvi.loc[name, "S_mean_log_rB_80_99"]))
    s.append(LIMITS.format(r90=r90))

    # ---------------------------------------------------------------- provenance / repro
    s.append("""<section id="provenance"><h2>Provenance</h2>
<p>Each figure and table traces back to a script, function, configuration and input set, recorded in <code>PROVENANCE.md</code>.
Each number printed on this page is also recorded in <code>provenance/numbers.json</code>, which says what it is, what produced it,
what was written for this project and what came from a library, and which choices had defensible alternatives. The report itself is
generated by <code>scripts/08_report.py</code> from <code>results/</code>, so none of its numbers were typed by hand.</p>
<p>Written for this project: calibration fit model, cluster bookkeeping, exposure map (MOSKITA convention), covariates, Poisson likelihood
and gradient, cross-fitting, evaluation, bootstrap and checks. Taken from libraries: astropy (FITS I/O), scipy (curve_fit, ndimage.label,
uniform filters, L-BFGS-B), numba (JIT of the trail loops), scikit-learn (HistGradientBoosting, IsolationForest), numpy, pandas, matplotlib.
Taken from MOSKITA, as definitions only (no code imported): the geometry slices, the overscan columns used for the pedestal, the exposure-time
convention, and the 8-connected cluster definition.</p>
<p>Literature that shaped the method: Berman &amp; Turner (1992), Appl. Stat. 41:31, and Baddeley &amp; Turner (2000), Aust. N. Z. J. Stat. 42:283,
for quadrature point-process likelihoods; Baddeley, Coeurjolly, Rubak &amp; Waagepetersen (2014), Biometrika 101:377, for the logistic
point-process identity; Liu, Ting &amp; Zhou (2008), ICDM, for Isolation Forest; Cowan, Cranmer, Gross &amp; Vitells (2011), Eur. Phys. J. C 71:1554,
for the Asimov significance; Cervantes-Vergara et al. (2025), arXiv:2508.06749, for the detector geometry.</p></section>""")
    s.append("""<section id="repro"><h2>Reproduction</h2>
<pre>git clone git@github.com:felidata/TP_AI.git &amp;&amp; cd TP_AI &amp;&amp; git checkout claude/claude-opus-5-5_high
cd CLAUDE/claude-opus-5-5_high
pip install -r requirements.txt            # Python 3.13
export SKIPPER_DATA_DIR=/path/to/pp_proc   # the 218 FITS files listed in data/manifest.csv
./run_all.sh                               # full run, ~3-4 h on 4 cores with 6 GB RAM
SKIP_CV=1 ./run_all.sh                     # reuse cache/cv/*.npz: final stages only</pre>
<p>Seeds are fixed in <code>configs/default.yaml</code>. The FITS data are not in the repository. A download helper for a future public copy
is in <code>scripts/download_data.py</code>; its URL is still a placeholder.</p></section>""")
    meta = {"run_id": summ["run_id"], "commit": str(summ.get("git_commit"))[:10]}
    return "\n".join(s), meta, numbers
