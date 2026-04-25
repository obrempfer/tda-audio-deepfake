#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAIN_PROTOCOL="${TRAIN_PROTOCOL:-$ROOT_DIR/data/raw/ASVspoof2019_LA/derived/ASVspoof2019.LA.cm.train.all_bonafide_balanced.seed42.txt}"
TRAIN_AUDIO_DIR="${TRAIN_AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac}"
VAL_PROTOCOL="${VAL_PROTOCOL:-$ROOT_DIR/data/raw/ASVspoof2019_LA/derived/ASVspoof2019.LA.cm.dev.balanced_5000.seed42.txt}"
VAL_AUDIO_DIR="${VAL_AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019_LA_dev/flac}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019.LA.cm.dev.trl.txt}"
EVAL_AUDIO_DIR="${EVAL_AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019_LA_dev/flac}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/data/results}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/${USER}/tda_deepfake_runtime}"
CACHE_ROOT="${CACHE_ROOT:-$RUNTIME_ROOT/topology_nn_cache}"

RUN_TAG="${RUN_TAG:-topology_nn_2019_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$RESULTS_ROOT/$RUN_TAG}"
LOG_FILE="${LOG_FILE:-$RESULTS_ROOT/logs/${RUN_TAG}.log}"

MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1000}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-5000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
WORKERS="${WORKERS:-40}"
PROGRESS_EVERY="${PROGRESS_EVERY:-250}"
MODEL_KIND="${MODEL_KIND:-all}"

LINEAR_C="${LINEAR_C:-1.0}"
HIDDEN_DIMS="${HIDDEN_DIMS:-128,64}"
FEATURE_DROPOUT="${FEATURE_DROPOUT:-0.10}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MONITOR="${MONITOR:-eer}"
PATIENCE="${PATIENCE:-8}"
FLAT_EPOCHS="${FLAT_EPOCHS:-40}"
FLAT_LR="${FLAT_LR:-1e-3}"
STAGE_EPOCHS="${STAGE_EPOCHS:-25,20,15}"
STAGE_LRS="${STAGE_LRS:-1e-3,3e-4,1e-4}"

CORE_CONFIG="${CORE_CONFIG:-$ROOT_DIR/configs/experiments/ablation/cubical_best_band_keep_low_h1_only.yaml}"
AUX_A_CONFIG="${AUX_A_CONFIG:-$ROOT_DIR/configs/experiments/ablation/cubical_best_band_keep_low_h0_only.yaml}"
AUX_B_CONFIG="${AUX_B_CONFIG:-$ROOT_DIR/configs/experiments/cubical_mel_best_field_svm.yaml}"
SKIP_AUX_B="${SKIP_AUX_B:-0}"

EXTRA_EVAL_2021_LA_PROTOCOL="${EXTRA_EVAL_2021_LA_PROTOCOL:-}"
EXTRA_EVAL_2021_LA_AUDIO_DIR="${EXTRA_EVAL_2021_LA_AUDIO_DIR:-}"
EXTRA_EVAL_2021_DF_PROTOCOL="${EXTRA_EVAL_2021_DF_PROTOCOL:-}"
EXTRA_EVAL_2021_DF_AUDIO_DIR="${EXTRA_EVAL_2021_DF_AUDIO_DIR:-}"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/.venv_lab/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv_lab/bin/python"
else
  echo "Missing project Python environment (.venv or .venv_lab)." >&2
  exit 1
fi

mkdir -p "$RESULTS_ROOT/logs" "$CACHE_ROOT" "$RUNTIME_ROOT/mpl" "$RUNTIME_ROOT/pycache" "$RUNTIME_ROOT/xdg"
export MPLCONFIGDIR="$RUNTIME_ROOT/mpl"
export PYTHONPYCACHEPREFIX="$RUNTIME_ROOT/pycache"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

cmd=(
  "$PYTHON_BIN" "$ROOT_DIR/src/scripts/run_topology_nn_experiment.py"
  --train-protocol "$TRAIN_PROTOCOL"
  --train-audio-dir "$TRAIN_AUDIO_DIR"
  --val-protocol "$VAL_PROTOCOL"
  --val-audio-dir "$VAL_AUDIO_DIR"
  --eval-protocol "$EVAL_PROTOCOL"
  --eval-audio-dir "$EVAL_AUDIO_DIR"
  --out-dir "$OUT_DIR"
  --cache-root "$CACHE_ROOT"
  --max-train-samples "$MAX_TRAIN_SAMPLES"
  --max-val-samples "$MAX_VAL_SAMPLES"
  --workers "$WORKERS"
  --progress-every "$PROGRESS_EVERY"
  --model-kind "$MODEL_KIND"
  --linear-c "$LINEAR_C"
  --hidden-dims "$HIDDEN_DIMS"
  --feature-dropout "$FEATURE_DROPOUT"
  --weight-decay "$WEIGHT_DECAY"
  --batch-size "$BATCH_SIZE"
  --monitor "$MONITOR"
  --patience "$PATIENCE"
  --flat-epochs "$FLAT_EPOCHS"
  --flat-learning-rate "$FLAT_LR"
  --stage-epochs "$STAGE_EPOCHS"
  --stage-learning-rates "$STAGE_LRS"
  --core-config "$CORE_CONFIG"
  --aux-a-config "$AUX_A_CONFIG"
  --aux-b-config "$AUX_B_CONFIG"
)

if [[ -n "$MAX_EVAL_SAMPLES" ]]; then
  cmd+=(--max-eval-samples "$MAX_EVAL_SAMPLES")
fi

if [[ "$SKIP_AUX_B" == "1" ]]; then
  cmd+=(--skip-aux-b)
fi

if [[ -n "$EXTRA_EVAL_2021_LA_PROTOCOL" && -n "$EXTRA_EVAL_2021_LA_AUDIO_DIR" ]]; then
  cmd+=(--extra-eval "transfer_2021_la=${EXTRA_EVAL_2021_LA_PROTOCOL}::${EXTRA_EVAL_2021_LA_AUDIO_DIR}")
fi

if [[ -n "$EXTRA_EVAL_2021_DF_PROTOCOL" && -n "$EXTRA_EVAL_2021_DF_AUDIO_DIR" ]]; then
  cmd+=(--extra-eval "transfer_2021_df=${EXTRA_EVAL_2021_DF_PROTOCOL}::${EXTRA_EVAL_2021_DF_AUDIO_DIR}")
fi

printf 'Run tag: %s\n' "$RUN_TAG" | tee "$LOG_FILE"
printf 'Output dir: %s\n' "$OUT_DIR" | tee -a "$LOG_FILE"
printf 'Train protocol: %s\n' "$TRAIN_PROTOCOL" | tee -a "$LOG_FILE"
printf 'Val protocol: %s\n' "$VAL_PROTOCOL" | tee -a "$LOG_FILE"
printf 'Eval protocol: %s\n' "$EVAL_PROTOCOL" | tee -a "$LOG_FILE"
printf 'Model kind: %s\n' "$MODEL_KIND" | tee -a "$LOG_FILE"
printf 'Cache root: %s\n' "$CACHE_ROOT" | tee -a "$LOG_FILE"
printf 'Workers: %s\n' "$WORKERS" | tee -a "$LOG_FILE"

"${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
