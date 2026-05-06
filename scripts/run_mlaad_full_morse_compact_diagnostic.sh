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

PHASE_TAG="${PHASE_TAG:-mlaad_full_morse_diag_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/data/results/$PHASE_TAG}"
LOG_ROOT="$RESULTS_ROOT/logs"
SUMMARY_LOG="$LOG_ROOT/${PHASE_TAG}.log"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/${USER}/tda_deepfake_runtime}"
CACHE_ROOT="${CACHE_ROOT:-$RUNTIME_ROOT/feature_cache/$PHASE_TAG}"
TRAIN_WORKERS="${TRAIN_WORKERS:-20}"
EVAL_WORKERS="${EVAL_WORKERS:-20}"
PROGRESS_EVERY="${PROGRESS_EVERY:-500}"
CLASSIFIER_QUEUE_ROOT="${CLASSIFIER_QUEUE_ROOT:-}"
DATASET_TAG="${DATASET_TAG:-mlaad_full_20260504_en}"
PROTO_ROOT="${PROTO_ROOT:-$ROOT_DIR/data/protocols/mlaad_full_20260504}"
RUNTIME_DATASET_ROOT="${RUNTIME_DATASET_ROOT:-$ROOT_DIR/data/raw/runtime_datasets/tda_datasets}"

mkdir -p "$RESULTS_ROOT" "$LOG_ROOT" "$CACHE_ROOT" "$RUNTIME_ROOT/mpl" "$RUNTIME_ROOT/pycache" "$RUNTIME_ROOT/xdg"
export PYTHONPATH="$ROOT_DIR/src"
export MPLCONFIGDIR="$RUNTIME_ROOT/mpl"
export PYTHONPYCACHEPREFIX="$RUNTIME_ROOT/pycache"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

DEV_PROTOCOL="$PROTO_ROOT/${DATASET_TAG}_splits/${DATASET_TAG}_dev.txt"
TEST_PROTOCOL="$PROTO_ROOT/${DATASET_TAG}_splits/${DATASET_TAG}_test.txt"
AUDIO_DIR="$RUNTIME_DATASET_ROOT/${DATASET_TAG}_materialized/audio"

metric_triplet() {
  local out_dir="$1"
  "$PYTHON_BIN" - "$out_dir" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "eval_results.json"
d = json.loads(p.read_text())
print(f"auc={d['auc']:.4f} eer={d['eer']:.4f} n_eval={d['n_eval']}")
PY
}

wait_for_result() {
  local out_dir="$1"
  while [[ ! -f "$out_dir/eval_results.json" ]]; do
    sleep 10
  done
}

generate_morse_config() {
  local run_name="$1"
  local band_mode="$2"
  local gate_pct="$3"
  local feature_subset="$4"
  local out_path="$RESULTS_ROOT/generated_configs/${run_name}.yaml"
  mkdir -p "$RESULTS_ROOT/generated_configs"
  "$PYTHON_BIN" - <<PY
from pathlib import Path
import yaml
base = Path(r"$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml")
out = Path(r"$out_path")
cfg = yaml.safe_load(base.read_text())
cfg.setdefault("spectrogram", {})
cfg.setdefault("morse_smale", {})
cfg["morse_smale"]["feature_subset"] = "$feature_subset"
if "$band_mode" == "none":
    cfg["spectrogram"].pop("band_mask_mode", None)
else:
    cfg["spectrogram"]["band_mask_mode"] = "$band_mode"
if "$gate_pct" == "none":
    cfg["spectrogram"].pop("energy_gate_percentile", None)
else:
    cfg["spectrogram"]["energy_gate_percentile"] = int("$gate_pct")
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(out)
PY
}

run_one() {
  local run_name="$1"
  local config_path="$2"
  local out_dir="$RESULTS_ROOT/${run_name}_dev"
  local log_file="$LOG_ROOT/${run_name}_dev.log"
  local -a queue_args=()
  if [[ -n "$CLASSIFIER_QUEUE_ROOT" ]]; then
    queue_args+=(--classifier-queue-root "$CLASSIFIER_QUEUE_ROOT")
  fi
  echo "[$(date -Is)] START $run_name" | tee -a "$SUMMARY_LOG"
  "$PYTHON_BIN" -u "$ROOT_DIR/src/scripts/run_pipeline.py" \
    --config "$config_path" \
    --train-protocol "$PROTO_ROOT/${DATASET_TAG}_splits/${DATASET_TAG}_train.txt" \
    --train-audio-dir "$AUDIO_DIR" \
    --eval-protocol "$DEV_PROTOCOL" \
    --eval-audio-dir "$AUDIO_DIR" \
    --out-dir "$out_dir" \
    --cache-dir "$CACHE_ROOT/$run_name" \
    --train-workers "$TRAIN_WORKERS" \
    --eval-workers "$EVAL_WORKERS" \
    --progress-every "$PROGRESS_EVERY" \
    "${queue_args[@]}" \
    > "$log_file" 2>&1
  if [[ -n "$CLASSIFIER_QUEUE_ROOT" ]]; then
    wait_for_result "$out_dir"
  fi
  echo "[$(date -Is)] DONE  $run_name $(metric_triplet "$out_dir")" | tee -a "$SUMMARY_LOG"
}

{
  echo "Phase tag: $PHASE_TAG"
  echo "Dataset tag: $DATASET_TAG"
  echo "Classifier queue root: ${CLASSIFIER_QUEUE_ROOT:-inline}"
} | tee "$SUMMARY_LOG"

run_one "morse_full_reference" "$ROOT_DIR/configs/experiments/morse_smale_best_field_matched_svm.yaml"
run_one "morse_gate_off" "$(generate_morse_config morse_gate_off keep_low none full)"
run_one "morse_keep_low" "$(generate_morse_config morse_keep_low keep_low 10 full)"
run_one "morse_basin_fractions" "$(generate_morse_config morse_basin_fractions keep_low 10 basin_fractions)"
run_one "morse_counts_entropy" "$(generate_morse_config morse_counts_entropy keep_low 10 counts_entropy)"

echo "[$(date -Is)] PHASE COMPLETE" | tee -a "$SUMMARY_LOG"
