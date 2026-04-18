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
INCLUDE_A5="${INCLUDE_A5:-0}"

MODE="${1:-${RUN_MODE:-all}}"

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

RUN_TAG="${RUN_TAG:-lowband_plan_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_LOG="$RESULTS_ROOT/logs/${RUN_TAG}_summary.log"

declare -a PHASE_A_CORE=(
  "phaseA_holdout_reference|configs/experiments/cubical_mel_best_field_svm.yaml"
  "phaseA_holdout_keep_low|configs/experiments/ablation/cubical_best_band_keep_low.yaml"
  "phaseA_holdout_gate12|configs/experiments/ablation/cubical_best_gate12.yaml"
  "phaseA_holdout_norm_zscore|configs/experiments/ablation/cubical_best_norm_zscore.yaml"
)

declare -a PHASE_A_OPTIONAL=(
  "phaseA_holdout_energy_weight_soft|configs/experiments/ablation/cubical_best_energy_weight_soft.yaml"
  "phaseA_holdout_temporal_transition|configs/experiments/ablation/cubical_best_temporal_transition.yaml"
  "phaseA_holdout_temporal_sustained|configs/experiments/ablation/cubical_best_temporal_sustained.yaml"
)

declare -a PHASE_B_HOLDOUT=(
  "phaseB_holdout_keep_low_full|configs/experiments/ablation/cubical_best_band_keep_low.yaml"
  "phaseB_holdout_low_lowest_third|configs/experiments/ablation/cubical_best_band_keep_low_lowest_third.yaml"
  "phaseB_holdout_low_upper_third|configs/experiments/ablation/cubical_best_band_keep_low_upper_third.yaml"
  "phaseB_holdout_low_lower_half|configs/experiments/ablation/cubical_best_band_keep_low_lower_half.yaml"
  "phaseB_holdout_low_upper_half|configs/experiments/ablation/cubical_best_band_keep_low_upper_half.yaml"
  "phaseB_holdout_low_plus_lower_mid|configs/experiments/ablation/cubical_best_band_keep_low_plus_lower_mid.yaml"
  "phaseB_holdout_low_minus_bottom_edge|configs/experiments/ablation/cubical_best_band_keep_low_minus_bottom_edge.yaml"
  "phaseB_holdout_keep_low_h0|configs/experiments/ablation/cubical_best_band_keep_low_h0_only.yaml"
  "phaseB_holdout_keep_low_h1|configs/experiments/ablation/cubical_best_band_keep_low_h1_only.yaml"
)

declare -a PHASE_C_CV=(
  "phaseC_cv_keep_low_c2|configs/experiments/ablation/cubical_best_band_keep_low_c2.yaml"
  "phaseC_cv_keep_low_c4|configs/experiments/ablation/cubical_best_band_keep_low.yaml"
  "phaseC_cv_keep_low_c8|configs/experiments/ablation/cubical_best_band_keep_low_c8.yaml"
  "phaseC_cv_keep_low_gate_off|configs/experiments/ablation/cubical_best_band_keep_low_gate_off.yaml"
  "phaseC_cv_keep_low_gate10|configs/experiments/ablation/cubical_best_band_keep_low.yaml"
  "phaseC_cv_keep_low_gate12|configs/experiments/ablation/cubical_best_band_keep_low_gate12.yaml"
  "phaseC_cv_keep_low_gate16|configs/experiments/ablation/cubical_best_band_keep_low_gate16.yaml"
)

echo "Run tag: $RUN_TAG" | tee "$SUMMARY_LOG"
echo "Mode: $MODE" | tee -a "$SUMMARY_LOG"
echo "Train protocol: $TRAIN_PROTOCOL" | tee -a "$SUMMARY_LOG"
echo "Eval protocol: $EVAL_PROTOCOL" | tee -a "$SUMMARY_LOG"
echo "CV protocol: $CV_PROTOCOL" | tee -a "$SUMMARY_LOG"
echo "Max train samples: $MAX_SAMPLES" | tee -a "$SUMMARY_LOG"
echo "Include A5 optional holdouts: $INCLUDE_A5" | tee -a "$SUMMARY_LOG"

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

run_phase_a() {
  for item in "${PHASE_A_CORE[@]}"; do
    run_holdout "${item%%|*}" "${item#*|}"
  done
  if [[ "$INCLUDE_A5" == "1" ]]; then
    for item in "${PHASE_A_OPTIONAL[@]}"; do
      run_holdout "${item%%|*}" "${item#*|}"
    done
  fi
}

run_phase_b() {
  for item in "${PHASE_B_HOLDOUT[@]}"; do
    run_holdout "${item%%|*}" "${item#*|}"
  done
}

run_phase_c() {
  for item in "${PHASE_C_CV[@]}"; do
    run_cv "${item%%|*}" "${item#*|}"
  done
}

case "$MODE" in
  phaseA)
    run_phase_a
    ;;
  phaseB)
    run_phase_b
    ;;
  phaseC)
    run_phase_c
    ;;
  all)
    run_phase_a
    run_phase_b
    run_phase_c
    ;;
  *)
    echo "Unknown mode: $MODE (expected: phaseA, phaseB, phaseC, all)" >&2
    exit 2
    ;;
esac

echo "" | tee -a "$SUMMARY_LOG"
echo "Total failures: $fail_count" | tee -a "$SUMMARY_LOG"
echo "Summary log: ${SUMMARY_LOG#$ROOT_DIR/}" | tee -a "$SUMMARY_LOG"

if [[ $fail_count -ne 0 ]]; then
  exit 1
fi
