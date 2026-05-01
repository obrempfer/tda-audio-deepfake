#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAIN_PROTOCOL="${TRAIN_PROTOCOL:-$ROOT_DIR/data/raw/ASVspoof2019_LA/derived/ASVspoof2019.LA.cm.train.all_bonafide_balanced.seed42.txt}"
TRAIN_AUDIO_DIR="${TRAIN_AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-$ROOT_DIR/data/raw/ASVspoof2019_LA/derived/ASVspoof2019.LA.cm.dev.balanced_5000.seed42.txt}"
EVAL_AUDIO_DIR="${EVAL_AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019_LA_dev/flac}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/data/results}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/${USER}/tda_deepfake_runtime}"
CACHE_ROOT="${CACHE_ROOT:-$RUNTIME_ROOT/feature_cache}"
CACHE_NAMESPACE="${CACHE_NAMESPACE:-takens_lowband_sweep_shared}"
SHARED_CACHE_DIR="${SHARED_CACHE_DIR:-$CACHE_ROOT/$CACHE_NAMESPACE}"
CONFIG_TMP_DIR="${CONFIG_TMP_DIR:-$RUNTIME_ROOT/generated_takens_configs}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-5000}"
PIPELINE_WORKERS="${PIPELINE_WORKERS:-24}"
TRAIN_WORKERS="${TRAIN_WORKERS:-$PIPELINE_WORKERS}"
EVAL_WORKERS="${EVAL_WORKERS:-$PIPELINE_WORKERS}"
PROGRESS_EVERY="${PROGRESS_EVERY:-250}"
SIGNAL_TYPES="${SIGNAL_TYPES:-low_wave low_env}"
EMBEDDING_DIMS="${EMBEDDING_DIMS:-3 5}"
LOW_WAVE_DELAYS="${LOW_WAVE_DELAYS:-2 4 8}"
LOW_ENV_DELAYS="${LOW_ENV_DELAYS:-1 2 4}"
LOW_WAVE_STRIDE="${LOW_WAVE_STRIDE:-80}"
LOW_ENV_STRIDE="${LOW_ENV_STRIDE:-1}"
BASE_CONFIG="${BASE_CONFIG:-$ROOT_DIR/configs/experiments/takens_ph_base_svm.yaml}"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/.venv_lab/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv_lab/bin/python"
else
  echo "Missing project Python environment (.venv or .venv_lab)." >&2
  exit 1
fi

mkdir -p "$RESULTS_ROOT/logs" "$SHARED_CACHE_DIR" "$CONFIG_TMP_DIR" "$RUNTIME_ROOT/mpl" "$RUNTIME_ROOT/pycache" "$RUNTIME_ROOT/xdg"
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

RUN_TAG="${RUN_TAG:-takens_lowband_sweep_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_LOG="$RESULTS_ROOT/logs/${RUN_TAG}_summary.log"
config_run_dir="$CONFIG_TMP_DIR/$RUN_TAG"
mkdir -p "$config_run_dir"

echo "Run tag: $RUN_TAG" | tee "$SUMMARY_LOG"
echo "Base config: ${BASE_CONFIG#$ROOT_DIR/}" | tee -a "$SUMMARY_LOG"
echo "Train protocol: $TRAIN_PROTOCOL" | tee -a "$SUMMARY_LOG"
echo "Eval protocol: $EVAL_PROTOCOL" | tee -a "$SUMMARY_LOG"
echo "Max train samples: $MAX_TRAIN_SAMPLES" | tee -a "$SUMMARY_LOG"
echo "Max eval samples: $MAX_EVAL_SAMPLES" | tee -a "$SUMMARY_LOG"
echo "Shared cache dir: $SHARED_CACHE_DIR" | tee -a "$SUMMARY_LOG"
echo "Generated config dir: $config_run_dir" | tee -a "$SUMMARY_LOG"
echo "Train workers: $TRAIN_WORKERS" | tee -a "$SUMMARY_LOG"
echo "Eval workers: $EVAL_WORKERS" | tee -a "$SUMMARY_LOG"
echo "Signal types: $SIGNAL_TYPES" | tee -a "$SUMMARY_LOG"
echo "Embedding dims: $EMBEDDING_DIMS" | tee -a "$SUMMARY_LOG"
echo "Low-wave delays: $LOW_WAVE_DELAYS (stride=$LOW_WAVE_STRIDE)" | tee -a "$SUMMARY_LOG"
echo "Low-env delays: $LOW_ENV_DELAYS (stride=$LOW_ENV_STRIDE)" | tee -a "$SUMMARY_LOG"

fail_count=0
run_count=0

for signal_type in $SIGNAL_TYPES; do
  if [[ "$signal_type" == "low_wave" ]]; then
    delay_values="$LOW_WAVE_DELAYS"
    stride="$LOW_WAVE_STRIDE"
  elif [[ "$signal_type" == "low_env" ]]; then
    delay_values="$LOW_ENV_DELAYS"
    stride="$LOW_ENV_STRIDE"
  else
    echo "Unsupported signal type in sweep: $signal_type" >&2
    exit 1
  fi

  for embedding_dim in $EMBEDDING_DIMS; do
    for delay in $delay_values; do
      run_name="${signal_type}_m${embedding_dim}_d${delay}"
      out_dir="$RESULTS_ROOT/${RUN_TAG}_${run_name}"
      log_file="$RESULTS_ROOT/logs/${RUN_TAG}_${run_name}.log"
      generated_config="$config_run_dir/${run_name}.yaml"
      run_count=$((run_count + 1))

      "$PYTHON_BIN" - <<PY
from pathlib import Path
import yaml

base = Path("$BASE_CONFIG")
out = Path("$generated_config")
cfg = yaml.safe_load(base.read_text())
cfg.setdefault("takens", {})
cfg["takens"]["signal_type"] = "$signal_type"
cfg["takens"]["embedding_dim"] = int("$embedding_dim")
cfg["takens"]["delay"] = int("$delay")
cfg["takens"]["stride"] = int("$stride")
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

      echo "" | tee -a "$SUMMARY_LOG"
      echo "[$(date -Is)] START $run_name" | tee -a "$SUMMARY_LOG"
      echo "  signal_type=$signal_type" | tee -a "$SUMMARY_LOG"
      echo "  embedding_dim=$embedding_dim" | tee -a "$SUMMARY_LOG"
      echo "  delay=$delay" | tee -a "$SUMMARY_LOG"
      echo "  stride=$stride" | tee -a "$SUMMARY_LOG"
      echo "  generated_config=$generated_config" | tee -a "$SUMMARY_LOG"
      echo "  out_dir=${out_dir#$ROOT_DIR/}" | tee -a "$SUMMARY_LOG"
      echo "  cache_dir=${SHARED_CACHE_DIR#$ROOT_DIR/}" | tee -a "$SUMMARY_LOG"
      echo "  mode=train_eval" | tee -a "$SUMMARY_LOG"

      cmd=(
        "$PYTHON_BIN" -u "$ROOT_DIR/src/scripts/run_pipeline.py"
        --config "$generated_config"
        --train-protocol "$TRAIN_PROTOCOL"
        --train-audio-dir "$TRAIN_AUDIO_DIR"
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
        echo "[$(date -Is)] FAIL  $run_name (exit=$status)" | tee -a "$SUMMARY_LOG"
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
        echo "[$(date -Is)] DONE  $run_name ${metrics}" | tee -a "$SUMMARY_LOG"
      fi
    done
  done
done

echo "" | tee -a "$SUMMARY_LOG"
echo "Total runs: $run_count" | tee -a "$SUMMARY_LOG"
echo "Total failures: $fail_count" | tee -a "$SUMMARY_LOG"
echo "Summary log: ${SUMMARY_LOG#$ROOT_DIR/}" | tee -a "$SUMMARY_LOG"

if [[ $fail_count -ne 0 ]]; then
  exit 1
fi
