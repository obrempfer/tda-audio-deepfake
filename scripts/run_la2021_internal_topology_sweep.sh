#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SOURCE_PROTOCOL="${SOURCE_PROTOCOL:-$ROOT_DIR/data/raw/ASVspoof2021_LA/keys/LA/CM/trial_metadata.txt}"
EVAL_AUDIO_DIR="${EVAL_AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2021_LA/ASVspoof2021_LA_eval/flac}"
DERIVED_DIR="${DERIVED_DIR:-$ROOT_DIR/data/raw/ASVspoof2021_LA/derived/internal_seed42}"
SPLIT_PREFIX="${SPLIT_PREFIX:-asvspoof2021_la_internal_seed42}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TRAIN_RATIO="${TRAIN_RATIO:-0.6}"
DEV_RATIO="${DEV_RATIO:-0.2}"
TEST_RATIO="${TEST_RATIO:-0.2}"
PARTITIONS="${PARTITIONS:-progress,eval}"
TRAIN_PROTOCOL="${TRAIN_PROTOCOL:-$DERIVED_DIR/${SPLIT_PREFIX}_train.txt}"
DEV_PROTOCOL="${DEV_PROTOCOL:-$DERIVED_DIR/${SPLIT_PREFIX}_dev.txt}"
TEST_PROTOCOL="${TEST_PROTOCOL:-$DERIVED_DIR/${SPLIT_PREFIX}_test.txt}"
EVAL_SPLIT="${EVAL_SPLIT:-dev}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/data/results}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/${USER}/tda_deepfake_runtime}"
CACHE_ROOT="${CACHE_ROOT:-$RUNTIME_ROOT/feature_cache}"
CACHE_NAMESPACE="${CACHE_NAMESPACE:-la2021_internal_topology_shared}"
SHARED_CACHE_DIR="${SHARED_CACHE_DIR:-$CACHE_ROOT/$CACHE_NAMESPACE}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-20000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-10000}"
PIPELINE_WORKERS="${PIPELINE_WORKERS:-40}"
TRAIN_WORKERS="${TRAIN_WORKERS:-$PIPELINE_WORKERS}"
EVAL_WORKERS="${EVAL_WORKERS:-$PIPELINE_WORKERS}"
PROGRESS_EVERY="${PROGRESS_EVERY:-250}"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/.venv_lab/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv_lab/bin/python"
else
  echo "Missing project Python environment (.venv or .venv_lab)." >&2
  exit 1
fi

mkdir -p "$RESULTS_ROOT/logs" "$SHARED_CACHE_DIR" "$RUNTIME_ROOT/mpl" "$RUNTIME_ROOT/pycache" "$RUNTIME_ROOT/xdg" "$DERIVED_DIR"
export MPLCONFIGDIR="$RUNTIME_ROOT/mpl"
export PYTHONPYCACHEPREFIX="$RUNTIME_ROOT/pycache"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg"
export PYTHONPATH="$ROOT_DIR/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

if [[ ! -f "$TRAIN_PROTOCOL" || ! -f "$DEV_PROTOCOL" || ! -f "$TEST_PROTOCOL" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/src/scripts/build_internal_protocol_split.py" \
    --protocol "$SOURCE_PROTOCOL" \
    --out-dir "$DERIVED_DIR" \
    --prefix "$SPLIT_PREFIX" \
    --seed "$SPLIT_SEED" \
    --train-ratio "$TRAIN_RATIO" \
    --dev-ratio "$DEV_RATIO" \
    --test-ratio "$TEST_RATIO" \
    --allowed-partitions "$PARTITIONS"
fi

case "$EVAL_SPLIT" in
  dev)
    EVAL_PROTOCOL="$DEV_PROTOCOL"
    ;;
  test)
    EVAL_PROTOCOL="$TEST_PROTOCOL"
    ;;
  *)
    echo "Unsupported EVAL_SPLIT='$EVAL_SPLIT'; expected dev or test." >&2
    exit 1
    ;;
esac

RUN_TAG="${RUN_TAG:-la2021_internal_topology_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_LOG="$RESULTS_ROOT/logs/${RUN_TAG}_summary.log"

declare -a RUNS=(
  "full_reference|configs/experiments/cubical_mel_best_field_svm.yaml"
  "keep_low|configs/experiments/ablation/cubical_best_band_keep_low.yaml"
  "keep_low_h1|configs/experiments/ablation/cubical_best_band_keep_low_h1_only.yaml"
  "keep_low_h0|configs/experiments/ablation/cubical_best_band_keep_low_h0_only.yaml"
  "drop_low|configs/experiments/ablation/cubical_best_band_drop_low.yaml"
  "drop_mid|configs/experiments/ablation/cubical_best_band_drop_mid.yaml"
  "drop_high|configs/experiments/ablation/cubical_best_band_drop_high.yaml"
  "gate_off|configs/experiments/ablation/cubical_best_band_keep_low_gate_off.yaml"
  "gate10|configs/experiments/ablation/cubical_best_band_keep_low.yaml"
  "gate12|configs/experiments/ablation/cubical_best_band_keep_low_gate12.yaml"
)

echo "Run tag: $RUN_TAG" | tee "$SUMMARY_LOG"
echo "Source protocol: $SOURCE_PROTOCOL" | tee -a "$SUMMARY_LOG"
echo "Train protocol: $TRAIN_PROTOCOL" | tee -a "$SUMMARY_LOG"
echo "Eval protocol: $EVAL_PROTOCOL" | tee -a "$SUMMARY_LOG"
echo "Eval split: $EVAL_SPLIT" | tee -a "$SUMMARY_LOG"
echo "Max train samples: $MAX_TRAIN_SAMPLES" | tee -a "$SUMMARY_LOG"
echo "Max eval samples: $MAX_EVAL_SAMPLES" | tee -a "$SUMMARY_LOG"
echo "Shared cache dir: $SHARED_CACHE_DIR" | tee -a "$SUMMARY_LOG"
echo "Train workers: $TRAIN_WORKERS" | tee -a "$SUMMARY_LOG"
echo "Eval workers: $EVAL_WORKERS" | tee -a "$SUMMARY_LOG"

fail_count=0

for item in "${RUNS[@]}"; do
  IFS="|" read -r name config_rel <<< "$item"
  config_path="$ROOT_DIR/$config_rel"
  out_dir="$RESULTS_ROOT/${RUN_TAG}_${name}"
  log_file="$RESULTS_ROOT/logs/${RUN_TAG}_${name}.log"

  echo "" | tee -a "$SUMMARY_LOG"
  echo "[$(date -Is)] START $name" | tee -a "$SUMMARY_LOG"
  echo "  config=$config_rel" | tee -a "$SUMMARY_LOG"
  echo "  out_dir=${out_dir#$ROOT_DIR/}" | tee -a "$SUMMARY_LOG"
  echo "  cache_dir=${SHARED_CACHE_DIR#$ROOT_DIR/}" | tee -a "$SUMMARY_LOG"
  echo "  mode=train_eval" | tee -a "$SUMMARY_LOG"

  cmd=(
    "$PYTHON_BIN" -u "$ROOT_DIR/src/scripts/run_pipeline.py"
    --config "$config_path"
    --train-protocol "$TRAIN_PROTOCOL"
    --train-audio-dir "$EVAL_AUDIO_DIR"
    --eval-protocol "$EVAL_PROTOCOL"
    --eval-audio-dir "$EVAL_AUDIO_DIR"
    --out-dir "$out_dir"
    --cache-dir "$SHARED_CACHE_DIR"
    --max-train-samples "$MAX_TRAIN_SAMPLES"
    --max-eval-samples "$MAX_EVAL_SAMPLES"
    --train-workers "$TRAIN_WORKERS"
    --eval-workers "$EVAL_WORKERS"
    --progress-every "$PROGRESS_EVERY"
  )

  "${cmd[@]}" >"$log_file" 2>&1
  status=$?

  if [[ $status -ne 0 ]]; then
    echo "[$(date -Is)] FAIL  $name (exit=$status)" | tee -a "$SUMMARY_LOG"
    fail_count=$((fail_count + 1))
  else
    metrics="$("$PYTHON_BIN" - <<PY
import json, pathlib
p = pathlib.Path("$out_dir") / "eval_results.json"
if p.exists():
    d = json.loads(p.read_text())
    print(f"auc={d.get('auc'):.3f} eer={d.get('eer'):.3f} n_eval={d.get('n_eval')}")
else:
    print("metrics=missing")
PY
)"
    echo "[$(date -Is)] DONE  $name ${metrics}" | tee -a "$SUMMARY_LOG"
  fi
done

echo "" | tee -a "$SUMMARY_LOG"
echo "Total failures: $fail_count" | tee -a "$SUMMARY_LOG"
echo "Summary log: ${SUMMARY_LOG#$ROOT_DIR/}" | tee -a "$SUMMARY_LOG"

if [[ $fail_count -ne 0 ]]; then
  exit 1
fi
