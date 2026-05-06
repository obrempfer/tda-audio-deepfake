#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv_lab/bin/python}"
DATASET_TAG="${DATASET_TAG:?Set DATASET_TAG, e.g. mlaad_full_20260504_ende}"
PHASE_TAG="${PHASE_TAG:-mlaad_full_anchor_subset_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/data/results/$PHASE_TAG}"
LOG_ROOT="$RESULTS_ROOT/logs"
SUMMARY_LOG="$LOG_ROOT/${DATASET_TAG}.log"
PROTO_ROOT="${PROTO_ROOT:-$ROOT_DIR/data/protocols/mlaad_full_20260504}"
RUNTIME_DATASET_ROOT="${RUNTIME_DATASET_ROOT:-$ROOT_DIR/data/raw/runtime_datasets/tda_datasets}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/${USER}/tda_deepfake_runtime}"
CACHE_ROOT="${CACHE_ROOT:-$RUNTIME_ROOT/feature_cache/$PHASE_TAG/$DATASET_TAG}"
TRAIN_WORKERS="${TRAIN_WORKERS:-16}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
PROGRESS_EVERY="${PROGRESS_EVERY:-500}"
CLASSIFIER_QUEUE_ROOT="${CLASSIFIER_QUEUE_ROOT:-}"

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

PROTOCOL_DIR="$PROTO_ROOT/${DATASET_TAG}_splits"
TRAIN_PROTOCOL="$PROTOCOL_DIR/${DATASET_TAG}_train.txt"
DEV_PROTOCOL="$PROTOCOL_DIR/${DATASET_TAG}_dev.txt"
TEST_PROTOCOL="$PROTOCOL_DIR/${DATASET_TAG}_test.txt"
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
  local timeout_seconds="${2:-0}"
  local waited=0
  while [[ ! -f "$out_dir/eval_results.json" || ! -f "$out_dir/model.pkl" ]]; do
    sleep 10
    waited=$((waited + 10))
    if [[ "$timeout_seconds" -gt 0 && "$waited" -ge "$timeout_seconds" ]]; then
      echo "Timed out waiting for queued classifier output in $out_dir" >&2
      return 1
    fi
  done
}

run_branch() {
  local branch="$1"
  local config_path="$2"
  local branch_cache="$CACHE_ROOT/$branch"
  local dev_out="$RESULTS_ROOT/${DATASET_TAG}_${branch}_dev"
  local test_out="$RESULTS_ROOT/${DATASET_TAG}_${branch}_test"
  local dev_log="$LOG_ROOT/${DATASET_TAG}_${branch}_dev.log"
  local test_log="$LOG_ROOT/${DATASET_TAG}_${branch}_test.log"

  echo "[$(date -Is)] START ${DATASET_TAG} ${branch} -> dev" | tee -a "$SUMMARY_LOG"
  local -a queue_args=()
  if [[ -n "$CLASSIFIER_QUEUE_ROOT" ]]; then
    queue_args+=(--classifier-queue-root "$CLASSIFIER_QUEUE_ROOT")
  fi
  "$PYTHON_BIN" -u "$ROOT_DIR/src/scripts/run_pipeline.py" \
    --config "$config_path" \
    --train-protocol "$TRAIN_PROTOCOL" \
    --train-audio-dir "$AUDIO_DIR" \
    --eval-protocol "$DEV_PROTOCOL" \
    --eval-audio-dir "$AUDIO_DIR" \
    --out-dir "$dev_out" \
    --train-cache-dir "$branch_cache/train" \
    --eval-cache-dir "$branch_cache/dev" \
    --train-workers "$TRAIN_WORKERS" \
    --eval-workers "$EVAL_WORKERS" \
    --progress-every "$PROGRESS_EVERY" \
    "${queue_args[@]}" \
    > "$dev_log" 2>&1
  if [[ -n "$CLASSIFIER_QUEUE_ROOT" ]]; then
    wait_for_result "$dev_out"
  fi
  echo "[$(date -Is)] DONE  ${DATASET_TAG} ${branch} -> dev $(metric_triplet "$dev_out")" | tee -a "$SUMMARY_LOG"
  rm -rf "$branch_cache/train" "$branch_cache/dev"

  echo "[$(date -Is)] START ${DATASET_TAG} ${branch} -> test" | tee -a "$SUMMARY_LOG"
  "$PYTHON_BIN" -u "$ROOT_DIR/src/scripts/run_pipeline.py" \
    --config "$config_path" \
    --load-model "$dev_out/model.pkl" \
    --eval-protocol "$TEST_PROTOCOL" \
    --eval-audio-dir "$AUDIO_DIR" \
    --out-dir "$test_out" \
    --eval-cache-dir "$branch_cache/test" \
    --eval-workers "$EVAL_WORKERS" \
    --progress-every "$PROGRESS_EVERY" \
    > "$test_log" 2>&1
  echo "[$(date -Is)] DONE  ${DATASET_TAG} ${branch} -> test $(metric_triplet "$test_out")" | tee -a "$SUMMARY_LOG"
  rm -rf "$branch_cache/test"
}

{
  echo "Phase tag: $PHASE_TAG"
  echo "Dataset tag: $DATASET_TAG"
  echo "Results root: $RESULTS_ROOT"
  echo "Protocol dir: $PROTOCOL_DIR"
  echo "Audio dir: $AUDIO_DIR"
  echo "Train workers: $TRAIN_WORKERS"
  echo "Eval workers: $EVAL_WORKERS"
  echo "Classifier queue root: ${CLASSIFIER_QUEUE_ROOT:-inline}"
} | tee "$SUMMARY_LOG"

while [[ ! -f "$TRAIN_PROTOCOL" || ! -d "$AUDIO_DIR" ]]; do
  sleep 10
done

run_branch "cubical" "$ROOT_DIR/configs/experiments/ablation/cubical_best_band_keep_low_gate12.yaml" &
PID_C=$!
run_branch "morse" "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" &
PID_M=$!
wait $PID_C
wait $PID_M

echo "[$(date -Is)] DATASET COMPLETE ${DATASET_TAG}" | tee -a "$SUMMARY_LOG"
