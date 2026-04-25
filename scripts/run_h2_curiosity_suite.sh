#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAIN_PROTOCOL="${TRAIN_PROTOCOL:-$ROOT_DIR/data/raw/ASVspoof2019_LA/derived/ASVspoof2019.LA.cm.train.all_bonafide_balanced.seed42.txt}"
TRAIN_AUDIO_DIR="${TRAIN_AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019.LA.cm.dev.trl.txt}"
EVAL_AUDIO_DIR="${EVAL_AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019_LA_dev/flac}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/data/results}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/${USER}/tda_deepfake_runtime}"
CACHE_ROOT="${CACHE_ROOT:-$RUNTIME_ROOT/feature_cache}"
CACHE_NAMESPACE="${CACHE_NAMESPACE:-h2_curiosity_2019_keep_low_shared}"
SHARED_CACHE_DIR="${SHARED_CACHE_DIR:-$CACHE_ROOT/$CACHE_NAMESPACE}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1000}"
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

mkdir -p "$RESULTS_ROOT/logs" "$SHARED_CACHE_DIR" "$RUNTIME_ROOT/mpl" "$RUNTIME_ROOT/pycache" "$RUNTIME_ROOT/xdg"
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

RUN_TAG="${RUN_TAG:-h2_curiosity_2019_keep_low_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_LOG="$RESULTS_ROOT/logs/${RUN_TAG}_summary.log"

declare -a NEW_RUNS=(
  "keep_low_h2|configs/experiments/ablation/cubical_best_band_keep_low_h2_only.yaml"
  "keep_low_h1_h2|configs/experiments/ablation/cubical_best_band_keep_low_h1_h2.yaml"
  "keep_low_h0_h1_h2|configs/experiments/ablation/cubical_best_band_keep_low_h0_h1_h2.yaml"
)

declare -a BASELINES=(
  "keep_low_h0|data/results/holdout_all_full_20260418_154849_phaseB_holdout_keep_low_h0"
  "keep_low_h1|data/results/holdout_all_full_20260418_154849_phaseB_holdout_keep_low_h1"
  "keep_low_h0_h1|data/results/holdout_all_full_20260418_154849_phaseA_holdout_keep_low"
)

echo "Run tag: $RUN_TAG" | tee "$SUMMARY_LOG"
echo "Train protocol: $TRAIN_PROTOCOL" | tee -a "$SUMMARY_LOG"
echo "Eval protocol: $EVAL_PROTOCOL" | tee -a "$SUMMARY_LOG"
echo "Max train samples: $MAX_TRAIN_SAMPLES" | tee -a "$SUMMARY_LOG"
echo "Shared cache dir: $SHARED_CACHE_DIR" | tee -a "$SUMMARY_LOG"
echo "Train workers: $TRAIN_WORKERS" | tee -a "$SUMMARY_LOG"
echo "Eval workers: $EVAL_WORKERS" | tee -a "$SUMMARY_LOG"
echo "Note: H2 on a 2-D cubical image is a bounded curiosity check and may be empty or nearly empty." | tee -a "$SUMMARY_LOG"

fail_count=0

metrics_for_dir() {
  local run_dir="$1"
  "$PYTHON_BIN" - <<PY
import json, pathlib, re

run_dir = pathlib.Path("$run_dir")
results_path = run_dir / "eval_results.json"
report_path = run_dir / "eval_report.txt"
if not results_path.exists():
    print("status=missing")
    raise SystemExit(0)

data = json.loads(results_path.read_text())
accuracy = None
if report_path.exists():
    match = re.search(r"^\\s*accuracy\\s+([0-9.]+)\\s+[0-9]+\\s*$", report_path.read_text(), re.MULTILINE)
    if match:
        accuracy = float(match.group(1))

parts = [
    f"auc={data.get('auc'):.4f}",
    f"eer={data.get('eer'):.4f}",
]
if accuracy is not None:
    parts.append(f"acc={accuracy:.4f}")
parts.append(f"n_eval={data.get('n_eval')}")
print(" ".join(parts))
PY
}

echo "" | tee -a "$SUMMARY_LOG"
echo "Existing baselines:" | tee -a "$SUMMARY_LOG"
for item in "${BASELINES[@]}"; do
  IFS="|" read -r name run_dir_rel <<< "$item"
  metrics="$(metrics_for_dir "$ROOT_DIR/$run_dir_rel")"
  echo "  $name $metrics" | tee -a "$SUMMARY_LOG"
done

for item in "${NEW_RUNS[@]}"; do
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
    --train-audio-dir "$TRAIN_AUDIO_DIR"
    --eval-protocol "$EVAL_PROTOCOL"
    --eval-audio-dir "$EVAL_AUDIO_DIR"
    --out-dir "$out_dir"
    --cache-dir "$SHARED_CACHE_DIR"
    --max-train-samples "$MAX_TRAIN_SAMPLES"
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
    metrics="$(metrics_for_dir "$out_dir")"
    echo "[$(date -Is)] DONE  $name $metrics" | tee -a "$SUMMARY_LOG"
  fi
done

echo "" | tee -a "$SUMMARY_LOG"
echo "Combined comparison:" | tee -a "$SUMMARY_LOG"
for item in "${BASELINES[@]}"; do
  IFS="|" read -r name run_dir_rel <<< "$item"
  metrics="$(metrics_for_dir "$ROOT_DIR/$run_dir_rel")"
  echo "  $name $metrics" | tee -a "$SUMMARY_LOG"
done
for item in "${NEW_RUNS[@]}"; do
  IFS="|" read -r name _ <<< "$item"
  run_dir="$RESULTS_ROOT/${RUN_TAG}_${name}"
  metrics="$(metrics_for_dir "$run_dir")"
  echo "  $name $metrics" | tee -a "$SUMMARY_LOG"
done

echo "" | tee -a "$SUMMARY_LOG"
echo "Total failures: $fail_count" | tee -a "$SUMMARY_LOG"
echo "Summary log: ${SUMMARY_LOG#$ROOT_DIR/}" | tee -a "$SUMMARY_LOG"

if [[ $fail_count -ne 0 ]]; then
  exit 1
fi
