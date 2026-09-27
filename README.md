# Skipper-CCD background rejection — UBA final project

This branch contains the independent CODEX analysis in
[`CODEX/gpt-6-astra_high/`](CODEX/gpt-6-astra_high/README.md).

- [Scientific HTML report](CODEX/gpt-6-astra_high/report.html)
- [Presentation PDF](CODEX/gpt-6-astra_high/report.pdf)
- [Method and validation protocol](CODEX/gpt-6-astra_high/METHOD.md)
- [Provenance and checks](CODEX/gpt-6-astra_high/PROVENANCE.md)
- [Machine-readable result](CODEX/gpt-6-astra_high/results/summary.json)

See the run README for the pinned environment and data-access instructions.
FITS data are deliberately external. With the environment installed, reproduce
from the repository root using:

```bash
SKIPPER_DATA_DIR=/path/to/pp_proc bash CODEX/gpt-6-astra_high/reproduce.sh
```

The main result uses a fixed acquisition-level holdout. It measures improvement
relative to the active-geometry baseline, not the existing MOSKITA production
pipeline. Limitations, negative ablations and a calibration transient are reported.
