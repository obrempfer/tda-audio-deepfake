#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
PHASE_TAG="${PHASE_TAG:-weighted_morse_mixed_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/data/results/$PHASE_TAG}"
LOG_ROOT="$RESULTS_ROOT/logs"
SUMMARY_LOG="$LOG_ROOT/${PHASE_TAG}.log"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/${USER}/tda_deepfake_runtime}"
CACHE_ROOT="${CACHE_ROOT:-$RUNTIME_ROOT/feature_cache/$PHASE_TAG}"
RUNTIME_DATASET_ROOT="${RUNTIME_DATASET_ROOT:-$ROOT_DIR/data/raw/runtime_datasets/tda_datasets}"
TRAIN_WORKERS="${TRAIN_WORKERS:-16}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
PROGRESS_EVERY="${PROGRESS_EVERY:-500}"
CLASSIFIER_QUEUE_ROOT="${CLASSIFIER_QUEUE_ROOT:-}"

ASV_TRAIN_PROTOCOL="${ASV_TRAIN_PROTOCOL:-$ROOT_DIR/data/protocols/mixed_source_20260502/asv2019_balanced_2580pc.txt}"
ASV_TRAIN_AUDIO_DIR="${ASV_TRAIN_AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac}"
MLAAD_TRAIN_PROTOCOL="${MLAAD_TRAIN_PROTOCOL:-$ROOT_DIR/data/protocols/mlaad_full_20260504/mlaad_full_20260504_en_splits/mlaad_full_20260504_en_train.txt}"
MLAAD_TRAIN_AUDIO_DIR="${MLAAD_TRAIN_AUDIO_DIR:-$ROOT_DIR/data/raw/runtime_datasets/tda_datasets/mlaad_full_20260504_en_materialized/audio}"
MLAAD_TEST_PROTOCOL="${MLAAD_TEST_PROTOCOL:-$ROOT_DIR/data/protocols/mlaad_full_20260504/mlaad_full_20260504_en_splits/mlaad_full_20260504_en_test.txt}"
MLAAD_TEST_AUDIO_DIR="${MLAAD_TEST_AUDIO_DIR:-$ROOT_DIR/data/raw/runtime_datasets/tda_datasets/mlaad_full_20260504_en_materialized/audio}"
ASV2021_EVAL_PROTOCOL="${ASV2021_EVAL_PROTOCOL:-$ROOT_DIR/data/raw/ASVspoof2021_LA/keys/LA/CM/trial_metadata.txt}"
ASV2021_EVAL_AUDIO_DIR="${ASV2021_EVAL_AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2021_LA/ASVspoof2021_LA_eval/flac}"
ITW_EVAL_PROTOCOL="${ITW_EVAL_PROTOCOL:-$ROOT_DIR/data/protocols/in_the_wild/protocol.txt}"
ITW_EVAL_AUDIO_DIR="${ITW_EVAL_AUDIO_DIR:-$ROOT_DIR/data/raw/In-The-Wild/release_in_the_wild}"

mkdir -p "$RESULTS_ROOT" "$LOG_ROOT" "$CACHE_ROOT" "$RUNTIME_ROOT/mpl" "$RUNTIME_ROOT/pycache" "$RUNTIME_ROOT/xdg" "$RUNTIME_DATASET_ROOT"
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
  while [[ ! -f "$out_dir/eval_results.json" || ! -f "$out_dir/model.pkl" ]]; do
    sleep 10
  done
}

build_mix() {
  local mix_tag="$1"
  local asv_per_label="$2"
  local mlaad_per_label="$3"
  "$PYTHON_BIN" "$ROOT_DIR/src/scripts/build_full_mlaad_mixed_source_protocols.py" \
    --asv-train-protocol "$ASV_TRAIN_PROTOCOL" \
    --asv-train-audio-dir "$ASV_TRAIN_AUDIO_DIR" \
    --mlaad-train-protocol "$MLAAD_TRAIN_PROTOCOL" \
    --mlaad-train-audio-dir "$MLAAD_TRAIN_AUDIO_DIR" \
    --runtime-dataset-root "$RUNTIME_DATASET_ROOT" \
    --dataset-tag "$mix_tag" \
    --asv-per-label "$asv_per_label" \
    --mlaad-per-label "$mlaad_per_label"
}

run_mix() {
  local mix_name="$1"
  local asv_per_label="$2"
  local mlaad_per_label="$3"
  local dataset_tag="${PHASE_TAG}_${mix_name}"
  local cache_dir="$CACHE_ROOT/$mix_name"
  local train_out="$RESULTS_ROOT/${mix_name}_asv2021"
  local mlaad_out="$RESULTS_ROOT/${mix_name}_mlaad_en_test"
  local itw_out="$RESULTS_ROOT/${mix_name}_in_the_wild"
  local mix_root="$RUNTIME_DATASET_ROOT/${dataset_tag}_materialized"
  local mix_protocol="$mix_root/${dataset_tag}.txt"
  local mix_audio_dir="$mix_root/audio"
  local -a queue_args=()

  build_mix "$dataset_tag" "$asv_per_label" "$mlaad_per_label" > "$LOG_ROOT/${mix_name}_build.log" 2>&1
  if [[ -n "$CLASSIFIER_QUEUE_ROOT" ]]; then
    queue_args+=(--classifier-queue-root "$CLASSIFIER_QUEUE_ROOT")
  fi

  echo "[$(date -Is)] START ${mix_name} -> asv2021" | tee -a "$SUMMARY_LOG"
  "$PYTHON_BIN" -u "$ROOT_DIR/src/scripts/run_pipeline.py" \
    --config "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" \
    --train-protocol "$mix_protocol" \
    --train-audio-dir "$mix_audio_dir" \
    --eval-protocol "$ASV2021_EVAL_PROTOCOL" \
    --eval-audio-dir "$ASV2021_EVAL_AUDIO_DIR" \
    --out-dir "$train_out" \
    --train-cache-dir "$cache_dir/train" \
    --eval-cache-dir "$cache_dir/asv2021" \
    --train-workers "$TRAIN_WORKERS" \
    --eval-workers "$EVAL_WORKERS" \
    --progress-every "$PROGRESS_EVERY" \
    "${queue_args[@]}" \
    > "$LOG_ROOT/${mix_name}_asv2021.log" 2>&1
  if [[ -n "$CLASSIFIER_QUEUE_ROOT" ]]; then
    wait_for_result "$train_out"
  fi
  echo "[$(date -Is)] DONE  ${mix_name} -> asv2021 $(metric_triplet "$train_out")" | tee -a "$SUMMARY_LOG"

  echo "[$(date -Is)] START ${mix_name} -> mlaad_en_test" | tee -a "$SUMMARY_LOG"
  "$PYTHON_BIN" -u "$ROOT_DIR/src/scripts/run_pipeline.py" \
    --config "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" \
    --load-model "$train_out/model.pkl" \
    --eval-protocol "$MLAAD_TEST_PROTOCOL" \
    --eval-audio-dir "$MLAAD_TEST_AUDIO_DIR" \
    --out-dir "$mlaad_out" \
    --eval-cache-dir "$cache_dir/mlaad_en_test" \
    --eval-workers "$EVAL_WORKERS" \
    --progress-every "$PROGRESS_EVERY" \
    > "$LOG_ROOT/${mix_name}_mlaad_en_test.log" 2>&1
  echo "[$(date -Is)] DONE  ${mix_name} -> mlaad_en_test $(metric_triplet "$mlaad_out")" | tee -a "$SUMMARY_LOG"

  echo "[$(date -Is)] START ${mix_name} -> in_the_wild" | tee -a "$SUMMARY_LOG"
  "$PYTHON_BIN" -u "$ROOT_DIR/src/scripts/run_pipeline.py" \
    --config "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" \
    --load-model "$train_out/model.pkl" \
    --eval-protocol "$ITW_EVAL_PROTOCOL" \
    --eval-audio-dir "$ITW_EVAL_AUDIO_DIR" \
    --out-dir "$itw_out" \
    --eval-cache-dir "$cache_dir/in_the_wild" \
    --eval-workers "$EVAL_WORKERS" \
    --progress-every "$PROGRESS_EVERY" \
    > "$LOG_ROOT/${mix_name}_in_the_wild.log" 2>&1
  echo "[$(date -Is)] DONE  ${mix_name} -> in_the_wild $(metric_triplet "$itw_out")" | tee -a "$SUMMARY_LOG"
}

{
  echo "Phase tag: $PHASE_TAG"
  echo "Results root: $RESULTS_ROOT"
  echo "Classifier queue root: ${CLASSIFIER_QUEUE_ROOT:-inline}"
} | tee "$SUMMARY_LOG"

run_mix "asv75_mlaad25" 1935 645
run_mix "asv50_mlaad50" 1290 1290
run_mix "asv25_mlaad75" 645 1935

echo "[$(date -Is)] PHASE COMPLETE" | tee -a "$SUMMARY_LOG"
