#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PROTOCOL="${TRAIN_PROTOCOL:-$ROOT_DIR/data/raw/ASVspoof2019_LA/derived/ASVspoof2019.LA.cm.train.all_bonafide_balanced.seed42.txt}"
TRAIN_AUDIO_DIR="${TRAIN_AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019.LA.cm.dev.trl.txt}"
EVAL_AUDIO_DIR="${EVAL_AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019_LA_dev/flac}"
CV_PROTOCOL="${CV_PROTOCOL:-$TRAIN_PROTOCOL}"
CV_AUDIO_DIR="${CV_AUDIO_DIR:-$TRAIN_AUDIO_DIR}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/data/results}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/${USER}/tda_deepfake_runtime}"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/.venv_lab/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv_lab/bin/python"
else
  echo "Missing project Python environment (.venv or .venv_lab)." >&2
  exit 1
fi

mkdir -p "$RESULTS_ROOT/logs" "$RUNTIME_ROOT/mpl" "$RUNTIME_ROOT/pycache" "$RUNTIME_ROOT/xdg"
export MPLCONFIGDIR="$RUNTIME_ROOT/mpl"
export PYTHONPYCACHEPREFIX="$RUNTIME_ROOT/pycache"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg"
export PYTHONPATH="$ROOT_DIR/src"

RUN_TAG="${RUN_TAG:-overnight_followup_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_LOG="$RESULTS_ROOT/logs/${RUN_TAG}_summary.log"

declare -a HOLDOUT_RUNS=(
  "phase3_holdout_reference|configs/experiments/cubical_mel_best_field_svm.yaml"
  "phase3_holdout_band_keep_low|configs/experiments/ablation/cubical_best_band_keep_low.yaml"
  "phase3_holdout_gate12|configs/experiments/ablation/cubical_best_gate12.yaml"
  "phase3_holdout_norm_zscore|configs/experiments/ablation/cubical_best_norm_zscore.yaml"
  "phase3_holdout_energy_weight_soft|configs/experiments/ablation/cubical_best_energy_weight_soft.yaml"
  "phase3_holdout_temporal_transition|configs/experiments/ablation/cubical_best_temporal_transition.yaml"
  "phase3_holdout_temporal_sustained|configs/experiments/ablation/cubical_best_temporal_sustained.yaml"
)

declare -a HOMOLOGY_RUNS=(
  "phase3_homology_reference_h0|configs/experiments/ablation/cubical_best_h0_only.yaml"
  "phase3_homology_reference_h1|configs/experiments/ablation/cubical_best_h1_only.yaml"
  "phase3_homology_keep_low_h0|configs/experiments/ablation/cubical_best_band_keep_low_h0_only.yaml"
  "phase3_homology_keep_low_h1|configs/experiments/ablation/cubical_best_band_keep_low_h1_only.yaml"
)

declare -a ROBUSTNESS_RUNS=(
  "phase4_robust_reference_c2|configs/experiments/ablation/cubical_best_c2.yaml"
  "phase4_robust_reference_c4|configs/experiments/cubical_mel_best_field_svm.yaml"
  "phase4_robust_reference_c8|configs/experiments/ablation/cubical_best_c8.yaml"
)

echo "Run tag: $RUN_TAG" | tee "$SUMMARY_LOG"
echo "Train protocol: $TRAIN_PROTOCOL" | tee -a "$SUMMARY_LOG"
echo "Eval protocol: $EVAL_PROTOCOL" | tee -a "$SUMMARY_LOG"
echo "CV protocol: $CV_PROTOCOL" | tee -a "$SUMMARY_LOG"
echo "Max train samples: $MAX_SAMPLES" | tee -a "$SUMMARY_LOG"

fail_count=0

run_holdout() {
  local name="$1"
  local config_rel="$2"
  local config_path="$ROOT_DIR/$config_rel"
  local out_dir="$RESULTS_ROOT/${RUN_TAG}_${name}"
  local log_file="$RESULTS_ROOT/logs/${RUN_TAG}_${name}.log"

  echo "" | tee -a "$SUMMARY_LOG"
  echo "[$(date -Is)] START $name" | tee -a "$SUMMARY_LOG"
  echo "  config=$config_rel" | tee -a "$SUMMARY_LOG"
  echo "  mode=train_eval" | tee -a "$SUMMARY_LOG"

  "$PYTHON_BIN" -u "$ROOT_DIR/src/scripts/run_pipeline.py" \
    --train-protocol "$TRAIN_PROTOCOL" \
    --train-audio-dir "$TRAIN_AUDIO_DIR" \
    --eval-protocol "$EVAL_PROTOCOL" \
    --eval-audio-dir "$EVAL_AUDIO_DIR" \
    --config "$config_path" \
    --max-samples "$MAX_SAMPLES" \
    --out-dir "$out_dir" >"$log_file" 2>&1
  local status=$?

  if [[ $status -ne 0 ]]; then
    echo "[$(date -Is)] FAIL  $name (exit=$status)" | tee -a "$SUMMARY_LOG"
    fail_count=$((fail_count + 1))
  else
    local metrics
    metrics="$(python3 - <<PY
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
}

run_cv() {
  local name="$1"
  local config_rel="$2"
  local config_path="$ROOT_DIR/$config_rel"
  local out_dir="$RESULTS_ROOT/${RUN_TAG}_${name}"
  local log_file="$RESULTS_ROOT/logs/${RUN_TAG}_${name}.log"

  echo "" | tee -a "$SUMMARY_LOG"
  echo "[$(date -Is)] START $name" | tee -a "$SUMMARY_LOG"
  echo "  config=$config_rel" | tee -a "$SUMMARY_LOG"
  echo "  mode=cv" | tee -a "$SUMMARY_LOG"

  "$PYTHON_BIN" -u "$ROOT_DIR/src/scripts/run_pipeline.py" \
    --protocol "$CV_PROTOCOL" \
    --audio-dir "$CV_AUDIO_DIR" \
    --config "$config_path" \
    --max-samples "$MAX_SAMPLES" \
    --out-dir "$out_dir" >"$log_file" 2>&1
  local status=$?

  if [[ $status -ne 0 ]]; then
    echo "[$(date -Is)] FAIL  $name (exit=$status)" | tee -a "$SUMMARY_LOG"
    fail_count=$((fail_count + 1))
  else
    local metrics
    metrics="$(python3 - <<PY
import json, pathlib
p = pathlib.Path("$out_dir") / "cv_results.json"
if p.exists():
    d = json.loads(p.read_text())
    print(f"acc={d.get('accuracy_mean'):.3f} auc={d.get('auc_mean'):.3f} eer={d.get('eer_mean'):.3f}")
else:
    print("metrics=missing")
PY
)"
    echo "[$(date -Is)] DONE  $name ${metrics}" | tee -a "$SUMMARY_LOG"
  fi
}

for item in "${HOLDOUT_RUNS[@]}"; do
  run_holdout "${item%%|*}" "${item#*|}"
done

for item in "${HOMOLOGY_RUNS[@]}"; do
  run_cv "${item%%|*}" "${item#*|}"
done

for item in "${ROBUSTNESS_RUNS[@]}"; do
  run_cv "${item%%|*}" "${item#*|}"
done

echo "" | tee -a "$SUMMARY_LOG"
echo "Total failures: $fail_count" | tee -a "$SUMMARY_LOG"
echo "Summary log: ${SUMMARY_LOG#$ROOT_DIR/}" | tee -a "$SUMMARY_LOG"

if [[ $fail_count -ne 0 ]]; then
  exit 1
fi
