#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/skipper-codex-matplotlib"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/skipper-codex-xdg"
python "$RUN_DIR/src/verify_data.py"
python -m unittest discover -s "$RUN_DIR/tests" -v
python "$RUN_DIR/src/audit.py"
python "$RUN_DIR/src/experiment.py"
python "$RUN_DIR/src/audit.py" --include-test
python "$RUN_DIR/src/experiment.py" --test
python "$RUN_DIR/src/verify.py"
python "$RUN_DIR/src/report.py"
python "$RUN_DIR/src/check_artifacts.py"
