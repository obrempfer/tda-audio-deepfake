#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAIN_PROTOCOL="${TRAIN_PROTOCOL:-$ROOT_DIR/data/raw/ASVspoof2019_LA/derived/ASVspoof2019.LA.cm.train.all_bonafide_balanced.seed42.txt}"
TRAIN_AUDIO_DIR="${TRAIN_AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-$ROOT_DIR/data/raw/ASVspoof2021_LA/keys/LA/CM/trial_metadata.txt}"
EVAL_AUDIO_DIR="${EVAL_AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2021_LA/ASVspoof2021_LA_eval/flac}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/data/results}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/${USER}/tda_deepfake_runtime}"
CACHE_ROOT="${CACHE_ROOT:-$RUNTIME_ROOT/feature_cache}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-1000}"

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
export PYTHONPATH="$ROOT_DIR/src"

RUN_TAG="${RUN_TAG:-cross_dataset_2019_to_2021_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_LOG="$RESULTS_ROOT/logs/${RUN_TAG}_summary.log"

declare -a RUNS=(
  "cubical_mel_landscape|configs/experiments/cubical_mel_landscape_svm.yaml|"
  "cubical_best_field|configs/experiments/cubical_mel_best_field_svm.yaml|data/results/holdout_all_full_20260418_154849_phaseA_holdout_reference/model.pkl"
  "cubical_keep_low|configs/experiments/ablation/cubical_best_band_keep_low.yaml|data/results/holdout_all_full_20260418_154849_phaseA_holdout_keep_low/model.pkl"
  "cubical_keep_low_h1|configs/experiments/ablation/cubical_best_band_keep_low_h1_only.yaml|data/results/holdout_all_full_20260418_154849_phaseB_holdout_keep_low_h1/model.pkl"
  "cubical_keep_low_h0|configs/experiments/ablation/cubical_best_band_keep_low_h0_only.yaml|data/results/holdout_all_full_20260418_154849_phaseB_holdout_keep_low_h0/model.pkl"
)

echo "Run tag: $RUN_TAG" | tee "$SUMMARY_LOG"
echo "Train protocol: $TRAIN_PROTOCOL" | tee -a "$SUMMARY_LOG"
echo "Eval protocol: $EVAL_PROTOCOL" | tee -a "$SUMMARY_LOG"
echo "Max train samples: $MAX_TRAIN_SAMPLES" | tee -a "$SUMMARY_LOG"
echo "Max eval samples: $MAX_EVAL_SAMPLES" | tee -a "$SUMMARY_LOG"
echo "Cache root: $CACHE_ROOT" | tee -a "$SUMMARY_LOG"

fail_count=0

for item in "${RUNS[@]}"; do
  IFS="|" read -r name config_rel model_rel <<< "$item"
  config_path="$ROOT_DIR/$config_rel"
  out_dir="$RESULTS_ROOT/${RUN_TAG}_${name}"
  cache_dir="$CACHE_ROOT/${RUN_TAG}_${name}"
  log_file="$RESULTS_ROOT/logs/${RUN_TAG}_${name}.log"

  echo "" | tee -a "$SUMMARY_LOG"
  echo "[$(date -Is)] START $name" | tee -a "$SUMMARY_LOG"
  echo "  config=$config_rel" | tee -a "$SUMMARY_LOG"
  echo "  out_dir=${out_dir#$ROOT_DIR/}" | tee -a "$SUMMARY_LOG"

  cmd=(
    "$PYTHON_BIN" -u "$ROOT_DIR/src/scripts/run_pipeline.py"
    --config "$config_path"
    --eval-protocol "$EVAL_PROTOCOL"
    --eval-audio-dir "$EVAL_AUDIO_DIR"
    --out-dir "$out_dir"
    --cache-dir "$cache_dir"
  )

  if [[ -n "$MAX_EVAL_SAMPLES" ]]; then
    cmd+=(--max-eval-samples "$MAX_EVAL_SAMPLES")
  fi

  if [[ -n "$model_rel" ]]; then
    cmd+=(--load-model "$ROOT_DIR/$model_rel")
    echo "  mode=eval_only_saved_model" | tee -a "$SUMMARY_LOG"
    echo "  model=$model_rel" | tee -a "$SUMMARY_LOG"
  else
    cmd+=(
      --train-protocol "$TRAIN_PROTOCOL"
      --train-audio-dir "$TRAIN_AUDIO_DIR"
      --max-train-samples "$MAX_TRAIN_SAMPLES"
    )
    echo "  mode=train_eval" | tee -a "$SUMMARY_LOG"
  fi

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
