#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv_lab/bin/python}"
PHASE_TAG="${PHASE_TAG:-mlaad_full_mixed_matrix_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/data/results/$PHASE_TAG}"
LOG_ROOT="$RESULTS_ROOT/logs"
SUMMARY_LOG="$LOG_ROOT/${PHASE_TAG}.log"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/${USER}/tda_deepfake_runtime}"
CACHE_ROOT="${CACHE_ROOT:-$RUNTIME_ROOT/feature_cache/$PHASE_TAG}"
TRAIN_WORKERS="${TRAIN_WORKERS:-12}"
EVAL_WORKERS="${EVAL_WORKERS:-12}"
PROGRESS_EVERY="${PROGRESS_EVERY:-500}"
CLASSIFIER_QUEUE_ROOT="${CLASSIFIER_QUEUE_ROOT:-}"

ASV_TRAIN_PROTOCOL="${ASV_TRAIN_PROTOCOL:-$ROOT_DIR/data/protocols/mixed_source_20260502/asv2019_balanced_2580pc.txt}"
ASV_TRAIN_AUDIO_DIR="${ASV_TRAIN_AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac}"
MLAAD_TRAIN_PROTOCOL="${MLAAD_TRAIN_PROTOCOL:-$ROOT_DIR/data/protocols/mlaad_full_20260504/mlaad_full_20260504_en_splits/mlaad_full_20260504_en_train.txt}"
MLAAD_TEST_PROTOCOL="${MLAAD_TEST_PROTOCOL:-$ROOT_DIR/data/protocols/mlaad_full_20260504/mlaad_full_20260504_en_splits/mlaad_full_20260504_en_test.txt}"
MLAAD_AUDIO_DIR="${MLAAD_AUDIO_DIR:-$ROOT_DIR/data/raw/runtime_datasets/tda_datasets/mlaad_full_20260504_en_materialized/audio}"
MIXED_DATASET_TAG="${MIXED_DATASET_TAG:-mixed_source_full_mlaad_en_20260505}"
MIXED_RUNTIME_DATASET_ROOT="${MIXED_RUNTIME_DATASET_ROOT:-$ROOT_DIR/data/raw/runtime_datasets/tda_datasets}"

ASV2019_EVAL_PROTOCOL="${ASV2019_EVAL_PROTOCOL:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019.LA.cm.dev.trl.txt}"
ASV2019_EVAL_AUDIO_DIR="${ASV2019_EVAL_AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019_LA_dev/flac}"
ASV2021_EVAL_PROTOCOL="${ASV2021_EVAL_PROTOCOL:-$ROOT_DIR/data/raw/ASVspoof2021_LA/keys/LA/CM/trial_metadata.txt}"
ASV2021_EVAL_AUDIO_DIR="${ASV2021_EVAL_AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2021_LA/ASVspoof2021_LA_eval/flac}"
ITW_EVAL_PROTOCOL="${ITW_EVAL_PROTOCOL:-$ROOT_DIR/data/protocols/in_the_wild/protocol.txt}"
ITW_EVAL_AUDIO_DIR="${ITW_EVAL_AUDIO_DIR:-$ROOT_DIR/data/raw/In-The-Wild/release_in_the_wild}"

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

MIXED_PROTOCOL_PATH="$MIXED_RUNTIME_DATASET_ROOT/${MIXED_DATASET_TAG}_materialized/${MIXED_DATASET_TAG}.txt"
MIXED_AUDIO_DIR="$MIXED_RUNTIME_DATASET_ROOT/${MIXED_DATASET_TAG}_materialized/audio"

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

run_eval() {
  local source_tag="$1"
  local branch="$2"
  local config_path="$3"
  local train_protocol="$4"
  local train_audio_dir="$5"
  local cache_dir="$6"
  local primary_tag="$7"
  local primary_protocol="$8"
  local primary_audio_dir="$9"
  shift 9

  local primary_out="$RESULTS_ROOT/${source_tag}_${branch}_${primary_tag}"
  local primary_log="$LOG_ROOT/${source_tag}_${branch}_${primary_tag}.log"
  local -a queue_args=()
  if [[ -n "$CLASSIFIER_QUEUE_ROOT" ]]; then
    queue_args+=(--classifier-queue-root "$CLASSIFIER_QUEUE_ROOT")
  fi
  if [[ -f "$primary_out/eval_results.json" && -f "$primary_out/model.pkl" ]]; then
    echo "[$(date -Is)] SKIP  ${source_tag} ${branch} -> ${primary_tag} $(metric_triplet "$primary_out")" | tee -a "$SUMMARY_LOG"
  else
    echo "[$(date -Is)] START ${source_tag} ${branch} -> ${primary_tag}" | tee -a "$SUMMARY_LOG"
    "$PYTHON_BIN" -u "$ROOT_DIR/src/scripts/run_pipeline.py" \
      --config "$config_path" \
      --train-protocol "$train_protocol" \
      --train-audio-dir "$train_audio_dir" \
      --eval-protocol "$primary_protocol" \
      --eval-audio-dir "$primary_audio_dir" \
      --out-dir "$primary_out" \
      --train-cache-dir "$cache_dir/train" \
      --eval-cache-dir "$cache_dir/${primary_tag}" \
      --train-workers "$TRAIN_WORKERS" \
      --eval-workers "$EVAL_WORKERS" \
      --progress-every "$PROGRESS_EVERY" \
      "${queue_args[@]}" \
      > "$primary_log" 2>&1
    if [[ -n "$CLASSIFIER_QUEUE_ROOT" ]]; then
      wait_for_result "$primary_out"
    fi
    echo "[$(date -Is)] DONE  ${source_tag} ${branch} -> ${primary_tag} $(metric_triplet "$primary_out")" | tee -a "$SUMMARY_LOG"
    rm -rf "$cache_dir/train" "$cache_dir/${primary_tag}"
  fi

  while (($#)); do
    local eval_tag="$1"
    local eval_protocol="$2"
    local eval_audio_dir="$3"
    shift 3

    local out_dir="$RESULTS_ROOT/${source_tag}_${branch}_${eval_tag}"
    local log_file="$LOG_ROOT/${source_tag}_${branch}_${eval_tag}.log"
    if [[ -f "$out_dir/eval_results.json" ]]; then
      echo "[$(date -Is)] SKIP  ${source_tag} ${branch} -> ${eval_tag} $(metric_triplet "$out_dir")" | tee -a "$SUMMARY_LOG"
    else
      echo "[$(date -Is)] START ${source_tag} ${branch} -> ${eval_tag}" | tee -a "$SUMMARY_LOG"
      "$PYTHON_BIN" -u "$ROOT_DIR/src/scripts/run_pipeline.py" \
        --config "$config_path" \
        --load-model "$primary_out/model.pkl" \
        --eval-protocol "$eval_protocol" \
        --eval-audio-dir "$eval_audio_dir" \
        --out-dir "$out_dir" \
        --eval-cache-dir "$cache_dir/${eval_tag}" \
        --eval-workers "$EVAL_WORKERS" \
        --progress-every "$PROGRESS_EVERY" \
        > "$log_file" 2>&1
      echo "[$(date -Is)] DONE  ${source_tag} ${branch} -> ${eval_tag} $(metric_triplet "$out_dir")" | tee -a "$SUMMARY_LOG"
      rm -rf "$cache_dir/${eval_tag}"
    fi
  done
}

run_source_pair() {
  local source_tag="$1"
  local train_protocol="$2"
  local train_audio_dir="$3"
  local source_cache="$CACHE_ROOT/$source_tag"
  echo "[$(date -Is)] SOURCE $source_tag" | tee -a "$SUMMARY_LOG"

  run_eval "$source_tag" "cubical" \
    "$ROOT_DIR/configs/experiments/ablation/cubical_best_band_keep_low_gate12.yaml" \
    "$train_protocol" "$train_audio_dir" "$source_cache/cubical" \
    "asv2019dev" "$ASV2019_EVAL_PROTOCOL" "$ASV2019_EVAL_AUDIO_DIR" \
    "asv2021" "$ASV2021_EVAL_PROTOCOL" "$ASV2021_EVAL_AUDIO_DIR" \
    "mlaad_en_test" "$MLAAD_TEST_PROTOCOL" "$MLAAD_AUDIO_DIR" \
    "in_the_wild" "$ITW_EVAL_PROTOCOL" "$ITW_EVAL_AUDIO_DIR" &
  PID_C=$!

  run_eval "$source_tag" "morse" \
    "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" \
    "$train_protocol" "$train_audio_dir" "$source_cache/morse" \
    "asv2019dev" "$ASV2019_EVAL_PROTOCOL" "$ASV2019_EVAL_AUDIO_DIR" \
    "asv2021" "$ASV2021_EVAL_PROTOCOL" "$ASV2021_EVAL_AUDIO_DIR" \
    "mlaad_en_test" "$MLAAD_TEST_PROTOCOL" "$MLAAD_AUDIO_DIR" \
    "in_the_wild" "$ITW_EVAL_PROTOCOL" "$ITW_EVAL_AUDIO_DIR" &
  PID_M=$!

  wait $PID_C
  wait $PID_M
  rm -rf "$source_cache"
}

{
  echo "Phase tag: $PHASE_TAG"
  echo "Results root: $RESULTS_ROOT"
  echo "Mixed dataset tag: $MIXED_DATASET_TAG"
  echo "Train workers: $TRAIN_WORKERS"
  echo "Eval workers: $EVAL_WORKERS"
  echo "Classifier queue root: ${CLASSIFIER_QUEUE_ROOT:-inline}"
} | tee "$SUMMARY_LOG"

while [[ ! -f "$MIXED_PROTOCOL_PATH" || ! -d "$MIXED_AUDIO_DIR" ]]; do
  sleep 10
done

run_source_pair "asv_only" "$ASV_TRAIN_PROTOCOL" "$ASV_TRAIN_AUDIO_DIR"
run_source_pair "mlaad_only" "$MLAAD_TRAIN_PROTOCOL" "$MLAAD_AUDIO_DIR"
run_source_pair "mixed" "$MIXED_PROTOCOL_PATH" "$MIXED_AUDIO_DIR"

echo "[$(date -Is)] PHASE COMPLETE" | tee -a "$SUMMARY_LOG"
