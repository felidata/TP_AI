# Autonomous research project — Skipper CCD background rejection

You are acting as an autonomous scientific research and software-development agent.

This is a final project for the UBA course "Gravitational Waves and AI-Assisted Research".

Your task is not merely to write code. You must independently investigate, design, implement, validate, document and version-control a complete scientific analysis.

Do not wait for the user to tell you what algorithm to implement. Make defensible scientific and technical decisions yourself, test them, and document those decisions.

## RUN INFORMATION

AGENT = CODEX

Determine the exact model and reasoning/effort setting being used in this session if that information is available from the runtime or CLI configuration.

The run directory MUST be:

CODEX/<model>_<effort>/

For example:

CODEX/astra_high/

If the exact model identifier cannot be determined automatically, use the most accurate identifier available and explicitly document the uncertainty.

Store a verbatim copy of this prompt in:

CODEX/<model>_<effort>/ORIGINAL_PROMPT.md

## Repository

The Git repository is:

git@github.com:felidata/TP_AI.git

You are running inside a dedicated Git worktree/branch for Codex.

Do not modify CLAUDE/.

Do not push directly to main.

Work on the current Codex branch, commit your work regularly with meaningful commit messages, and push the branch to origin.

The repository of the course instructors is available as:

matiaszaldarriaga/GW-AI-course

You may inspect it with GitHub, `gh`, or git.

In particular inspect the material concerning:
- final project requirements;
- provenance;
- autonomous agents;
- independent verification;
- producing scientific HTML reports;
- reproducibility.

Do not copy their scientific examples mechanically. Use them to understand the philosophy and expected level of documentation.

## Local scientific context

This worktree is located beside an existing MOSKITA Skipper-CCD analysis project.

The raw/preprocessed Skipper-CCD images are located at:

../pp_proc

There is also documentation and existing MOSKITA analysis code in the parent project.

You MAY inspect those materials in read-only fashion to understand:
- FITS organization;
- detector geometry;
- quadrants/HDU structure;
- charge units;
- exposure information;
- calibration conventions;
- detector-specific terminology;
- the physical context of the experiment.

However:

THE PRIMARY METHOD DEVELOPED IN THIS PROJECT MUST NOT SIMPLY REIMPLEMENT THE EXISTING MOSKITA MASK PIPELINE.

In particular, do not make the main contribution a rewrite of the existing hot-column, halo, bleed, SRE, transfer-gate-trap or hot-zone masks.

Existing algorithms may be used only:
1. to understand the data;
2. as a clearly labeled baseline;
3. for comparison.

The objective is to independently investigate image-processing, computer-vision, anomaly-detection or machine-learning techniques that could identify detector regions associated with excess low-energy background.

## Scientific objective

The ultimate experimental objective is to reduce background events with reconstructed charge between 4 and 7 electrons while preserving as much effective exposure as possible.

A method that removes background by masking a large fraction of the detector is NOT automatically better.

The optimization is explicitly multi-objective:

1. Reduce the background rate in the 4–7 e- region.
2. Preserve effective detector exposure.

Do not optimize only the raw number of surviving 4–7 e- events.

For each method/configuration calculate at least:

f_exp = E_after / E_before

r_B = B_after / B_before

where B is the 4–7 e- background RATE normalized by surviving effective exposure.

Also report the auxiliary equal-weight figure of merit

F_equal = f_exp / r_B

where larger is better.

Do not use this scalar as the only criterion.

The primary scientific visualization must include the trade-off between background rate and retained exposure and, where possible, the Pareto frontier.

If enough detector/exposure information exists to construct a more physically meaningful expected-sensitivity proxy, derive it explicitly and report it IN ADDITION to the metrics above. Do not silently assume a sensitivity formula.

## Prevent a trivial or biased solution

A method is not allowed to "solve" the task by identifying and masking individual 4–7 e- candidate events.

The mask or cleaning criterion should identify detector/image behavior indicating anomalous regions or contamination, rather than use the target ROI as a direct label for every event.

Separate method development from final evaluation.

Do NOT randomly split pixels from the same image between training and test sets.

Perform splitting or cross-validation at the level of complete FITS images/acquisitions, and preserve detector/module structure when relevant.

If the dataset is small, prefer file-level cross-validation or leave-one-image/acquisition-out validation over an unreliable random pixel split.

Hyperparameters must be selected without using the final held-out evaluation images.

Clearly document the exact validation protocol.

## Independent method development

Act as an independent researcher.

First inspect the images quantitatively.

Then investigate potentially relevant methods from:
- computer vision;
- astronomical image processing;
- anomaly detection;
- defect detection;
- robust statistics;
- unsupervised or self-supervised learning;
- spatial statistics;
- segmentation;
- multiscale image processing.

You may search scientific literature and technical references.

Maintain citations/links or bibliographic information for ideas that materially influenced the method.

Do NOT select a neural network simply because machine learning sounds sophisticated.

Prefer the simplest method that is defensible and works, but explore multiple candidate approaches before committing to the primary one.

Examples of families you MAY consider, but are explicitly NOT requirements:

- robust PCA / low-rank + sparse decomposition;
- patch-based anomaly detection;
- local robust statistics;
- multiscale/wavelet filtering;
- morphological image analysis;
- connected-component analysis;
- clustering;
- Isolation Forest / Local Outlier Factor;
- autoencoder reconstruction error;
- convolutional or self-supervised representations;
- spatial-temporal persistence maps;
- combinations of these.

You are free to reject all of these and develop something better suited to the data.

Document which alternatives were considered and why the final method was chosen.

## Scientific controls

Before developing the new method, establish a baseline.

Quantify at minimum:
- number of usable images;
- image/HDU dimensions and relevant metadata;
- baseline exposure;
- baseline 4–7 e- event count;
- baseline 4–7 e- background rate;
- spatial distribution of events;
- charge spectrum.

Inspect whether artifacts are spatially persistent between images or transient.

Determine whether module/quadrant differences matter.

All definitions must be explicit.

In particular, define exactly what "4–7 e-" means for the calibrated continuous charge variable or integer charge bins used by these data.

## Evaluation

Your final evaluation should contain, whenever scientifically meaningful:

- baseline vs cleaned charge spectra;
- retained exposure;
- absolute and relative 4–7 e- background;
- background vs retained-exposure curve;
- Pareto frontier;
- representative raw/calibrated image;
- anomaly score or mask overlay;
- spatial event maps before and after cleaning;
- validation/test comparison;
- uncertainty or statistical error bars where meaningful;
- ablation study for important components;
- sanity checks showing that the method is not merely masking the ROI events themselves.

Record negative results too.

Do not hide approaches that failed.

## Reproducibility

THE GITHUB REPOSITORY MUST NOT CONTAIN THE LARGE FITS DATA.

Code must not depend on a hard-coded absolute path.

Support at least:

SKIPPER_DATA_DIR=/path/to/pp_proc

Create documentation describing where the files should be placed.

Create a reproducible data manifest containing filenames and useful metadata and, if practical, hashes.

Prepare a mechanism such as:

data/fetch_data.py

or equivalent, capable of obtaining the dataset once a Google Drive URL or file ID is supplied.

Do NOT invent a Google Drive URL.

If no URL is available, put an obvious placeholder and make the local SKIPPER_DATA_DIR workflow fully functional.

A new person cloning the repository should be able to understand exactly how to:
1. create the environment;
2. obtain/provide the data;
3. reproduce the analysis;
4. regenerate every figure;
5. regenerate the HTML report.

Pin important dependencies using pyproject.toml, requirements.txt or environment.yml.

Prefer scripts/workflows over results that exist only inside an interactive notebook.

Notebooks are welcome for exploration but every final quantitative result must be reproducible non-interactively.

Set deterministic random seeds where relevant.

## Provenance

Create:

CODEX/<model>_<effort>/PROVENANCE.md

For every major result/figure/table document:
- source data;
- script/function that produced it;
- configuration/hyperparameters;
- output file;
- external library functionality used;
- methodological choice involved;
- validation/check performed.

Clearly distinguish:
- code written specifically for this project;
- library routines;
- algorithms adapted from references;
- pre-existing MOSKITA code if any is used.

Every figure in the final report should be traceable to code.

## Machine-readable output

Produce:

CODEX/<model>_<effort>/results/summary.json

with a clean schema containing at minimum:
- agent;
- model;
- effort;
- method_name;
- data_files;
- split_strategy;
- baseline_exposure;
- retained_exposure;
- exposure_fraction;
- baseline_roi_events;
- final_roi_events;
- baseline_roi_background_rate;
- final_roi_background_rate;
- background_ratio;
- F_equal;
- selected_hyperparameters;
- random_seed;
- runtime if available;
- git commit;
- notes.

Use null plus an explanation rather than inventing unavailable physical quantities.

Also generate a CSV containing the tested operating points so another program can reconstruct the background/exposure Pareto curve.

## Required run-directory structure

At minimum create:

CODEX/<model>_<effort>/
    README.md
    ORIGINAL_PROMPT.md
    METHOD.md
    PROVENANCE.md
    environment information
    src/
    notebooks/              # optional
    configs/
    figures/
    results/
        summary.json
        operating_points.csv
    report.html

You may add additional files/directories when useful.

## HTML report

Generate a self-contained or easily portable scientific report:

CODEX/<model>_<effort>/report.html

It should explain:
- problem;
- dataset;
- independent method-selection process;
- method;
- validation design;
- results;
- background/exposure trade-off;
- checks;
- failures/limitations;
- reproducibility;
- provenance.

The report should be useful both to a physicist and to someone evaluating the use of an AI coding/research agent.

Do not write marketing language.

## Time management

There is a hard project deadline.

Prioritize a complete, validated, reproducible end-to-end result over an ambitious unfinished model.

Use a staged strategy:

First obtain a simple reproducible baseline and evaluation pipeline.

Then implement the strongest promising independent method.

Only attempt computationally expensive deep-learning approaches if the dataset, compute availability and expected benefit justify them.

If an ambitious approach fails or would take unreasonable resources, document that result and continue with a simpler method.

Do not stop merely because one experiment fails.

## Autonomous behavior

Do not ask the user routine implementation questions.

Inspect the available files and make reasonable scientific choices.

When uncertain:
- investigate;
- test alternatives;
- document the assumption.

Run the code yourself.

Inspect outputs.

Fix errors.

Verify numerical consistency.

Do not claim success simply because the script runs.

Continue until the repository contains a defensible end-to-end scientific result.

## Git

Commit incrementally.

Use descriptive commits such as:
- baseline/data audit
- candidate method implementation
- validation and optimization
- figures and provenance
- final report

Before finishing:
- run the complete reproduction workflow from a clean entry point;
- check git status;
- verify that no large FITS files or secrets are tracked;
- commit all intended files;
- push the Codex branch.

At the end, print a concise final status containing:
- branch;
- commit hash;
- method selected;
- principal numerical result;
- retained exposure;
- main limitations;
- exact command required to reproduce the analysis.