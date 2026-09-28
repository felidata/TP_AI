#!/usr/bin/env bash
# One-command reproduction of every number and figure of this run.
#   SKIPPER_DATA_DIR=/path/to/pp_proc ./run_all.sh            (full: cache + 8-config CV + final, ~2.5 h on 4 cores)
#   SKIP_CV=1 ./run_all.sh     reuse cache/cv/*.npz if present (final stages only, ~15 min)
set -euo pipefail
cd "$(dirname "$0")"
W=${WORKERS:-4}
t0=$(date +%s)
python scripts/00_manifest.py ${NO_HASH:+--no-hash}
python scripts/01_build_cache.py --workers "$W"
python -c "import sys; sys.path.insert(0,'src'); from skbg import config, stacks; stacks.build(config.load_config())"
python scripts/02_baseline.py
if [ -z "${SKIP_CV:-}" ]; then python scripts/03_cv_search.py --workers "$W"; fi
python scripts/03b_select.py
python scripts/04_final.py --workers "$W" --extra "${EXTRA:-}"   # EXTRA=glm_static_only,... adds test ablations
python scripts/05_checks.py
python scripts/06_figures.py
t1=$(date +%s)
echo "{\"total_wall_s\": $((t1-t0)), \"workers\": $W, \"skip_cv\": \"${SKIP_CV:-}\"}" > results/run_timing.json
python scripts/07_summary.py
python scripts/08_report.py
