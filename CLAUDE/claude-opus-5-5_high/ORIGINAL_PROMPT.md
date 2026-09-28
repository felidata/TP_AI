# Independent autonomous research project — Skipper CCD background rejection

You are an autonomous scientific research agent responsible for completing an entire reproducible analysis from raw experimental data to validated scientific results.

This is a final project for the UBA course "Gravitational Waves and AI-Assisted Research".

Another independent AI agent, Codex, is simultaneously receiving the same scientific problem.

DO NOT inspect, imitate or use the CODEX/ implementation.

The purpose of the experiment is precisely to see what method you develop independently.

## RUN IDENTIFICATION

AGENT = CLAUDE

Determine the actual Claude model being used in this session and any relevant effort/reasoning configuration available from the runtime.

Your complete work must live under:

CLAUDE/<model>_<effort>/

Use the most precise run identifier available.

If an exact effort setting does not exist for this runtime, use an informative identifier rather than inventing one and explain it in the README.

Store this complete prompt verbatim at:

CLAUDE/<model>_<effort>/ORIGINAL_PROMPT.md

## Repository and isolation

Repository:

git@github.com:felidata/TP_AI.git

You are running inside a dedicated Claude Git worktree/branch.

Do not modify CODEX/.

Do not push directly to main.

Commit and push only your Claude branch.

The instructors' repository is:

matiaszaldarriaga/GW-AI-course

Inspect its final-project material and relevant course material to understand the intended philosophy concerning:
- AI-assisted scientific research;
- provenance;
- independent verification;
- reproducibility;
- HTML scientific reporting.

Do not copy its scientific examples. Apply its research methodology to this problem.

## Available data and context

The local Skipper-CCD dataset is available at:

../pp_proc

The parent MOSKITA project also contains detector documentation and an existing image-analysis pipeline.

You may inspect that material to understand the experiment and data format.

Treat it as scientific context, not as the solution.

The primary contribution of this run MUST NOT be a reproduction of the existing MOSKITA masking pipeline.

Do not simply rebuild the existing:
- hot-column masks;
- bleed masks;
- halo masks;
- SRE masks;
- transfer-gate masks;
- hot-zone criteria.

They may be studied or used as a labeled baseline only when useful.

Instead, independently investigate techniques from computer vision, image processing, anomaly detection, astronomical imaging, spatial statistics or machine learning that can plausibly identify regions responsible for excess low-energy events.

## Scientific question

Develop a method that reduces the background rate of events with reconstructed charge between 4 and 7 electrons while retaining as much useful Skipper-CCD exposure as possible.

This is a multi-objective problem.

An aggressive mask that removes almost all events by destroying detector exposure is not a good solution.

For every candidate operating point calculate at least:

f_exp = E_after / E_before

r_B = B_after / B_before

where B is the 4–7 e- background RATE normalized to surviving effective exposure.

Also calculate the common comparison metric:

F_equal = f_exp / r_B

with larger values representing a better equal-weight trade-off.

This is only an auxiliary metric.

The primary comparison should show the complete background-vs-exposure trade-off and its Pareto frontier rather than hiding the problem behind a single score.

If detector and physics information permit construction of a better sensitivity proxy, derive it and state the assumptions explicitly.

Do not silently substitute an arbitrary significance formula.

## Scientific integrity / leakage control

Do not directly classify individual 4–7 e- events as bad solely because they fall in the target ROI.

The goal is to identify anomalous detector/image regions or contamination processes.

Do not optimize and report performance on the same data.

Do not make random train/test splits among pixels from the same FITS file.

Split at the complete image/acquisition level.

Take detector module/quadrant structure into account when relevant.

If there are too few files for a conventional train/validation/test split, use an appropriate image-level cross-validation strategy such as leave-one-image/acquisition-out.

Hyperparameter tuning must never use the final evaluation image(s).

Explicitly document this procedure.

## Research phase

Before choosing an implementation:

1. Audit the actual FITS data.
2. Quantify the baseline.
3. Characterize spatial and inter-image structure.
4. Investigate multiple plausible methodological families.
5. Select an approach based on evidence rather than novelty alone.

You may search the scientific/technical literature.

Record bibliographic information for any method that materially influences the analysis.

You have freedom to invent or adapt an approach.

Potentially relevant families include, but are not limited to:
- patch-based anomaly detection;
- robust decomposition;
- image segmentation;
- unsupervised defect detection;
- multiscale or wavelet methods;
- robust local statistics;
- clustering and density estimation;
- spatial persistence analysis;
- Isolation Forest / LOF;
- autoencoders;
- self-supervised representations;
- convolutional models;
- morphological methods.

This is NOT a menu you must follow.

Choose what the data justify.

A clever robust statistical/CV method is preferable to an unjustified neural network.

Document rejected alternatives and negative experiments.

## Establish the baseline

Determine and document:
- number of usable files;
- FITS/HDU organization;
- relevant metadata;
- detector/module/quadrant organization;
- charge representation;
- baseline effective exposure;
- baseline charge spectrum;
- baseline event count in 4–7 e-;
- baseline exposure-normalized 4–7 e- background;
- spatial distribution of those events;
- persistence or variability of anomalous regions across images.

Define precisely how events are reconstructed/counting if this is not already explicit in the data.

Define exactly how "4–7 e-" is interpreted.

Never silently choose bin boundaries.

## Method development

Construct an end-to-end pipeline whose stages can be reproduced non-interactively.

It should expose relevant parameters in configuration files or command-line arguments.

Search important hyperparameters scientifically.

Do not optimize an excessive number of arbitrary thresholds.

Where possible, construct a continuous anomaly score from which different operating points can be generated.

This is desirable because it naturally produces the background-vs-exposure Pareto curve.

Investigate whether the learned/detected patterns generalize across images and detector sections.

## Required validation and plots

Generate quantitative and visual evidence including, where appropriate:

- charge spectrum before cleaning;
- charge spectrum after cleaning;
- ratio or difference;
- representative detector image;
- anomaly-score image;
- final mask overlay;
- spatial density of 4–7 e- candidates before/after;
- retained exposure;
- 4–7 e- background rate;
- background ratio;
- F_equal;
- background versus retained exposure;
- Pareto frontier;
- validation versus held-out performance;
- statistical uncertainties;
- relevant ablations;
- examples of regions accepted/rejected by the method.

Run sanity checks to determine whether apparent improvement is caused by:
- actual removal of anomalous regions;
- simple area removal;
- leakage from the target ROI;
- one pathological FITS image;
- one module/quadrant dominating the result.

## Reproducibility

Never commit the large FITS files.

No final script may require a user-specific absolute path.

Support:

SKIPPER_DATA_DIR=/path/to/pp_proc

Create enough data documentation that another researcher knows what data are required.

Create a manifest of the local dataset and useful metadata; include hashes when practical.

Prepare a configurable download helper for a future Google Drive copy of the dataset.

Do NOT invent the Drive link.

If the URL is unavailable, leave a documented placeholder while keeping the local-data workflow complete.

Provide:
- environment specification;
- deterministic seeds when relevant;
- one-command or very short reproduction instructions;
- non-interactive scripts for all final results.

Exploratory notebooks are permitted but cannot be the only source of final numbers.

## Provenance

Create:

CLAUDE/<model>_<effort>/PROVENANCE.md

Track each final figure/table/result back to:
- input files;
- script;
- function;
- configuration;
- model or algorithm;
- external libraries;
- relevant literature;
- output artifact;
- validation/check.

Explicitly label any pre-existing MOSKITA code or definitions that are reused.

Do not present library functionality as code invented during this project.

## Standardized machine-readable results

Write:

CLAUDE/<model>_<effort>/results/summary.json

containing at least:

agent
model
effort
method_name
data_files
split_strategy
baseline_exposure
retained_exposure
exposure_fraction
baseline_roi_events
final_roi_events
baseline_roi_background_rate
final_roi_background_rate
background_ratio
F_equal
selected_hyperparameters
random_seed
runtime_if_available
git_commit
notes

If a physical quantity cannot be obtained reliably, use null and explain why.

Do not fabricate values.

Also produce:

CLAUDE/<model>_<effort>/results/operating_points.csv

containing sufficient information to independently reconstruct the background/exposure curve and Pareto frontier.

## Run directory

At minimum:

CLAUDE/<model>_<effort>/
    README.md
    ORIGINAL_PROMPT.md
    METHOD.md
    PROVENANCE.md
    src/
    notebooks/          # optional
    configs/
    figures/
    results/
        summary.json
        operating_points.csv
    report.html

You may improve this structure if useful without writing outside CLAUDE/.

## HTML scientific report

Create:

CLAUDE/<model>_<effort>/report.html

It should be a polished scientific report covering:
- scientific motivation;
- data;
- initial diagnostics;
- independent research process;
- candidate approaches;
- chosen method;
- validation;
- results;
- exposure/background trade-off;
- checks;
- limitations and failures;
- provenance;
- reproduction instructions.

Make the figures readable.

Avoid promotional language.

Distinguish observations from interpretation.

## Deadline-aware autonomy

A complete simple result is worth more than an unfinished sophisticated result.

First build and validate an end-to-end analysis.

Only then add complexity.

Do not spend the entire project training a deep network unless there is compelling evidence that this is appropriate and computationally feasible.

If an attempted method is unsuccessful, record why and move to the next defensible approach.

Do not ask the user to select algorithms or routine implementation details.

Investigate and decide.

Run all important code yourself.

Inspect the generated results rather than assuming success.

Fix failures and numerical inconsistencies.

## Completion requirements

Before declaring the project finished:

- reproduce the analysis from its documented entry point;
- verify all final numbers;
- make sure no FITS data or secrets are committed;
- check that every important figure has provenance;
- check the machine-readable summary;
- commit all intended files;
- push the Claude branch.

At completion print:
- branch;
- final commit hash;
- selected method;
- principal quantitative outcome;
- retained exposure;
- main scientific limitation;
- exact reproduction command.
