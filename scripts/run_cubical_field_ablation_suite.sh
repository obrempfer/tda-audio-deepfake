#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTOCOL_PATH="${PROTOCOL_PATH:-$ROOT_DIR/data/raw/ASVspoof2019_LA/derived/ASVspoof2019.LA.cm.train.all_bonafide_balanced.seed42.txt}"
AUDIO_DIR="${AUDIO_DIR:-$ROOT_DIR/data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac}"
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

RUN_TAG="${RUN_TAG:-overnight_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_LOG="$RESULTS_ROOT/logs/${RUN_TAG}_summary.log"

declare -a RUNS=(
  "phase1_reference|configs/experiments/cubical_mel_best_field_svm.yaml"
  "phase1_band_drop_low|configs/experiments/ablation/cubical_best_band_drop_low.yaml"
  "phase1_band_drop_mid|configs/experiments/ablation/cubical_best_band_drop_mid.yaml"
  "phase1_band_drop_high|configs/experiments/ablation/cubical_best_band_drop_high.yaml"
  "phase1_band_keep_low|configs/experiments/ablation/cubical_best_band_keep_low.yaml"
  "phase1_band_keep_mid|configs/experiments/ablation/cubical_best_band_keep_mid.yaml"
  "phase1_band_keep_high|configs/experiments/ablation/cubical_best_band_keep_high.yaml"
  "phase2_energy_raw_db|configs/experiments/ablation/cubical_best_gate_off.yaml"
  "phase2_energy_gated_db|configs/experiments/cubical_mel_best_field_svm.yaml"
  "phase2_energy_weight_soft|configs/experiments/ablation/cubical_best_energy_weight_soft.yaml"
  "phase2_energy_weight_strong|configs/experiments/ablation/cubical_best_energy_weight_strong.yaml"
  "phase2_temporal_transition|configs/experiments/ablation/cubical_best_temporal_transition.yaml"
  "phase2_temporal_sustained|configs/experiments/ablation/cubical_best_temporal_sustained.yaml"
)

echo "Run tag: $RUN_TAG" | tee "$SUMMARY_LOG"
echo "Protocol: $PROTOCOL_PATH" | tee -a "$SUMMARY_LOG"
echo "Audio dir: $AUDIO_DIR" | tee -a "$SUMMARY_LOG"
echo "Max samples: $MAX_SAMPLES" | tee -a "$SUMMARY_LOG"

fail_count=0

for item in "${RUNS[@]}"; do
  name="${item%%|*}"
  config_rel="${item#*|}"
  config_path="$ROOT_DIR/$config_rel"
  out_dir="$RESULTS_ROOT/${RUN_TAG}_${name}"
  log_file="$RESULTS_ROOT/logs/${RUN_TAG}_${name}.log"

  echo "" | tee -a "$SUMMARY_LOG"
  echo "[$(date -Is)] START $name" | tee -a "$SUMMARY_LOG"
  echo "  config=$config_rel" | tee -a "$SUMMARY_LOG"
  echo "  out_dir=${out_dir#$ROOT_DIR/}" | tee -a "$SUMMARY_LOG"

  "$PYTHON_BIN" -u "$ROOT_DIR/src/scripts/run_pipeline.py" \
    --protocol "$PROTOCOL_PATH" \
    --audio-dir "$AUDIO_DIR" \
    --config "$config_path" \
    --max-samples "$MAX_SAMPLES" \
    --out-dir "$out_dir" >"$log_file" 2>&1
  status=$?

  if [[ $status -ne 0 ]]; then
    echo "[$(date -Is)] FAIL  $name (exit=$status)" | tee -a "$SUMMARY_LOG"
    fail_count=$((fail_count + 1))
  else
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
done

echo "" | tee -a "$SUMMARY_LOG"
echo "Total failures: $fail_count" | tee -a "$SUMMARY_LOG"
echo "Summary log: ${SUMMARY_LOG#$ROOT_DIR/}" | tee -a "$SUMMARY_LOG"

if [[ $fail_count -ne 0 ]]; then
  exit 1
fi
