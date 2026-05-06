#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/.venv_lab/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv_lab/bin/python"
else
  echo "Missing project Python environment (.venv or .venv_lab)." >&2
  exit 1
fi

PHASE_TAG="${PHASE_TAG:-score_fusion_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/data/results/$PHASE_TAG}"
LOG_ROOT="$RESULTS_ROOT/logs"
mkdir -p "$RESULTS_ROOT" "$LOG_ROOT"
export PYTHONPATH="$ROOT_DIR/src"

SOURCE_RESULTS_ROOT="${SOURCE_RESULTS_ROOT:?Set SOURCE_RESULTS_ROOT to the mixed-matrix results root}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"

run_pair() {
  local pair_name="$1"
  local cubical_source="$2"
  local morse_source="$3"
  "$PYTHON_BIN" "$ROOT_DIR/src/scripts/run_score_fusion_experiment.py" \
    --out-dir "$RESULTS_ROOT/$pair_name" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --calibration "asv2019dev::$SOURCE_RESULTS_ROOT/${cubical_source}_cubical_asv2019dev::$SOURCE_RESULTS_ROOT/${morse_source}_morse_asv2019dev" \
    --target "asv2021::$SOURCE_RESULTS_ROOT/${cubical_source}_cubical_asv2021::$SOURCE_RESULTS_ROOT/${morse_source}_morse_asv2021" \
    --target "mlaad_en_test::$SOURCE_RESULTS_ROOT/${cubical_source}_cubical_mlaad_en_test::$SOURCE_RESULTS_ROOT/${morse_source}_morse_mlaad_en_test" \
    --target "in_the_wild::$SOURCE_RESULTS_ROOT/${cubical_source}_cubical_in_the_wild::$SOURCE_RESULTS_ROOT/${morse_source}_morse_in_the_wild" \
    > "$LOG_ROOT/${pair_name}.log" 2>&1
}

run_pair "asv_only_pair" "asv_only" "asv_only"
run_pair "mixed_pair" "mixed" "mixed"
run_pair "asv_cubical_mixed_morse" "asv_only" "mixed"
