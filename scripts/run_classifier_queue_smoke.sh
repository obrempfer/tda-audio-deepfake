#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv_lab/bin/python}"
RUN_TAG="${RUN_TAG:-classifier_queue_smoke_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/data/results/$RUN_TAG}"
QUEUE_ROOT="$RESULTS_ROOT/train_queue"
FEATURE_DIR="$RESULTS_ROOT/features/cached_cubical_bundle"
LOG_ROOT="$RESULTS_ROOT/logs"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/${USER}/tda_deepfake_runtime}"

TRAIN_PROTOCOL="${TRAIN_PROTOCOL:-$ROOT_DIR/data/protocols/mlaad_full_20260504/mlaad_full_20260504_de_splits/mlaad_full_20260504_de_train.txt}"
TRAIN_AUDIO_DIR="${TRAIN_AUDIO_DIR:-$ROOT_DIR/data/raw/runtime_datasets/tda_datasets/mlaad_full_20260504_de_materialized/audio}"
TRAIN_CACHE_DIR="${TRAIN_CACHE_DIR:-$RUNTIME_ROOT/feature_cache/mlaad_full_phase1_bg18_20260505/mlaad_full_20260504_de/cubical/train}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-$ROOT_DIR/data/protocols/mlaad_full_20260504/mlaad_full_20260504_de_splits/mlaad_full_20260504_de_dev.txt}"
EVAL_AUDIO_DIR="${EVAL_AUDIO_DIR:-$ROOT_DIR/data/raw/runtime_datasets/tda_datasets/mlaad_full_20260504_de_materialized/audio}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-$RUNTIME_ROOT/feature_cache/mlaad_full_phase1_bg18_20260505/mlaad_full_20260504_de/cubical/dev}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/experiments/ablation/cubical_best_band_keep_low_gate12.yaml}"

mkdir -p "$RESULTS_ROOT" "$LOG_ROOT" "$FEATURE_DIR" "$QUEUE_ROOT/ready" "$QUEUE_ROOT/claimed" "$QUEUE_ROOT/done" "$QUEUE_ROOT/failed"
export PYTHONPATH="$ROOT_DIR/src"

echo "[smoke] building cached feature bundle"
"$PYTHON_BIN" "$ROOT_DIR/src/scripts/build_cached_classifier_bundle.py" \
  --config "$CONFIG_PATH" \
  --train-protocol "$TRAIN_PROTOCOL" \
  --train-audio-dir "$TRAIN_AUDIO_DIR" \
  --train-cache-dir "$TRAIN_CACHE_DIR" \
  --eval-protocol "$EVAL_PROTOCOL" \
  --eval-audio-dir "$EVAL_AUDIO_DIR" \
  --eval-cache-dir "$EVAL_CACHE_DIR" \
  --out-dir "$FEATURE_DIR" \
  > "$LOG_ROOT/bundle_build.log" 2>&1

python_job() {
  local run_id="$1"
  local classifier="$2"
  local probability="$3"
  local cache_size="$4"
  local result_dir="$RESULTS_ROOT/results/$run_id"
  mkdir -p "$result_dir"
  cat > "$RESULTS_ROOT/${run_id}.json" <<EOF
{
  "run_id": "$run_id",
  "feature_dir": "$FEATURE_DIR",
  "result_dir": "$result_dir",
  "classifier": "$classifier",
  "params": {
    "C": 1.0,
    "gamma": "scale",
    "probability": $probability,
    "cache_size": $cache_size,
    "max_iter": 1000,
    "random_state": 42
  }
}
EOF
  /usr/bin/time -f "ELAPSED=%E USER=%U SYS=%S MAXRSS_KB=%M" \
    -o "$result_dir/time.txt" \
    "$PYTHON_BIN" "$ROOT_DIR/src/scripts/train_classifier_job.py" \
    --job-json "$RESULTS_ROOT/${run_id}.json" \
    > "$LOG_ROOT/${run_id}.log" 2>&1
}

echo "[smoke] direct comparisons"
python_job "svc_prob_true" "sklearn_svc_rbf" "true" "200"
python_job "svc_prob_false" "sklearn_svc_rbf" "false" "200"
python_job "svc_prob_false_cache8000" "sklearn_svc_rbf" "false" "8000"

echo "[smoke] queue simulation"
for idx in 1 2 3; do
  cat > "$QUEUE_ROOT/ready/run_${idx}.ready.json" <<EOF
{
  "run_id": "queue_run_${idx}",
  "feature_dir": "$FEATURE_DIR",
  "result_dir": "$RESULTS_ROOT/results/queue_run_${idx}",
  "classifier": "sklearn_svc_rbf",
  "params": {
    "C": 1.0,
    "gamma": "scale",
    "probability": false,
    "cache_size": 8000,
    "max_iter": 1000,
    "random_state": 42
  }
}
EOF
done

/usr/bin/time -f "ELAPSED=%E USER=%U SYS=%S MAXRSS_KB=%M" \
  -o "$RESULTS_ROOT/queue_one_trainer.time" \
  "$PYTHON_BIN" "$ROOT_DIR/src/scripts/train_queue_worker.py" \
  --queue-root "$QUEUE_ROOT" --once \
  > "$LOG_ROOT/queue_one_trainer.log" 2>&1 || true

echo "[smoke] complete"
