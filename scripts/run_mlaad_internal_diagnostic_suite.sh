#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MLAAD_ROOT="${MLAAD_ROOT:-$ROOT_DIR/data/raw/MLAAD-tiny}"
MLAAD_LANGUAGES="${MLAAD_LANGUAGES:-en}"
MLAAD_MAX_PER_CLASS="${MLAAD_MAX_PER_CLASS:-1000}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/data/results}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/${USER}/tda_deepfake_runtime}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$ROOT_DIR/data/protocols}"
MATERIALIZED_ROOT="${MATERIALIZED_ROOT:-$PROTOCOL_ROOT/mlaad_tiny}"
CACHE_ROOT="${CACHE_ROOT:-$RUNTIME_ROOT/feature_cache}"
CONFIG_TMP_DIR="${CONFIG_TMP_DIR:-$RUNTIME_ROOT/generated_mlaad_diagnostic_configs}"
PIPELINE_WORKERS="${PIPELINE_WORKERS:-32}"
TRAIN_WORKERS="${TRAIN_WORKERS:-$PIPELINE_WORKERS}"
EVAL_WORKERS="${EVAL_WORKERS:-$PIPELINE_WORKERS}"
PROGRESS_EVERY="${PROGRESS_EVERY:-500}"
PROFILE="${PROFILE:-english_full}"
TRAIN_WITH_DEV="${TRAIN_WITH_DEV:-1}"
SEED="${SEED:-42}"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/.venv_lab/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv_lab/bin/python"
else
  echo "Missing project Python environment (.venv or .venv_lab)." >&2
  exit 1
fi

mkdir -p \
  "$RESULTS_ROOT/logs" \
  "$CONFIG_TMP_DIR" \
  "$RUNTIME_ROOT/mpl" \
  "$RUNTIME_ROOT/pycache" \
  "$RUNTIME_ROOT/xdg"
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

RUN_TAG="${RUN_TAG:-mlaad_diag_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_LOG="$RESULTS_ROOT/logs/${RUN_TAG}_summary.log"
MATERIALIZED_DIR="$MATERIALIZED_ROOT/${RUN_TAG}_materialized"
SPLIT_DIR="$MATERIALIZED_ROOT/${RUN_TAG}_splits"
CACHE_DIR="${CACHE_DIR:-$CACHE_ROOT/${RUN_TAG}_shared}"
CONFIG_RUN_DIR="$CONFIG_TMP_DIR/$RUN_TAG"

mkdir -p "$MATERIALIZED_DIR" "$SPLIT_DIR" "$CACHE_DIR" "$CONFIG_RUN_DIR"

"$PYTHON_BIN" "$ROOT_DIR/src/scripts/materialize_mlaad_subset.py" \
  --src-root "$MLAAD_ROOT" \
  --out-dir "$MATERIALIZED_DIR" \
  --languages "$MLAAD_LANGUAGES" \
  --max-per-class "$MLAAD_MAX_PER_CLASS" \
  --seed "$SEED"

"$PYTHON_BIN" "$ROOT_DIR/src/scripts/build_internal_protocol_split.py" \
  --protocol "$MATERIALIZED_DIR/protocol.txt" \
  --out-dir "$SPLIT_DIR" \
  --prefix "$RUN_TAG" \
  --seed "$SEED" \
  --train-ratio 0.7 \
  --dev-ratio 0.15 \
  --test-ratio 0.15 \
  --label-only

if [[ "$TRAIN_WITH_DEV" == "1" ]]; then
  TRAIN_PROTOCOL="$SPLIT_DIR/${RUN_TAG}_trainplusdev.txt"
  cat "$SPLIT_DIR/${RUN_TAG}_train.txt" "$SPLIT_DIR/${RUN_TAG}_dev.txt" > "$TRAIN_PROTOCOL"
else
  TRAIN_PROTOCOL="$SPLIT_DIR/${RUN_TAG}_train.txt"
fi

EVAL_PROTOCOL="$SPLIT_DIR/${RUN_TAG}_test.txt"
AUDIO_DIR="$MATERIALIZED_DIR/audio"

echo "Run tag: $RUN_TAG" | tee "$SUMMARY_LOG"
echo "Profile: $PROFILE" | tee -a "$SUMMARY_LOG"
echo "MLAAD root: $MLAAD_ROOT" | tee -a "$SUMMARY_LOG"
echo "Languages: $MLAAD_LANGUAGES" | tee -a "$SUMMARY_LOG"
echo "Max per class: $MLAAD_MAX_PER_CLASS" | tee -a "$SUMMARY_LOG"
echo "Train with dev: $TRAIN_WITH_DEV" | tee -a "$SUMMARY_LOG"
echo "Train protocol: $TRAIN_PROTOCOL" | tee -a "$SUMMARY_LOG"
echo "Eval protocol: $EVAL_PROTOCOL" | tee -a "$SUMMARY_LOG"
echo "Audio dir: $AUDIO_DIR" | tee -a "$SUMMARY_LOG"
echo "Cache dir: $CACHE_DIR" | tee -a "$SUMMARY_LOG"
echo "Generated config dir: $CONFIG_RUN_DIR" | tee -a "$SUMMARY_LOG"
echo "Train workers: $TRAIN_WORKERS" | tee -a "$SUMMARY_LOG"
echo "Eval workers: $EVAL_WORKERS" | tee -a "$SUMMARY_LOG"

declare -a RUNS=()

add_run() {
  RUNS+=("$1|$2")
}

generate_morse_config() {
  local run_name="$1"
  local base_config="$2"
  local band_mode="$3"
  local gate_pct="$4"
  local feature_subset="$5"
  local out_path="$CONFIG_RUN_DIR/${run_name}.yaml"

  "$PYTHON_BIN" - <<PY
from pathlib import Path
import yaml

base = Path(r"$base_config")
out = Path(r"$out_path")
cfg = yaml.safe_load(base.read_text())
cfg.setdefault("spectrogram", {})
cfg.setdefault("morse_smale", {})
cfg["morse_smale"]["graph_max_neighbors"] = 4
cfg["morse_smale"]["normalization"] = None
cfg["morse_smale"]["feature_subset"] = "$feature_subset"
if "$band_mode" == "none":
    cfg["spectrogram"].pop("band_mask_mode", None)
else:
    cfg["spectrogram"]["band_mask_mode"] = "$band_mode"
if "$gate_pct" == "none":
    cfg["spectrogram"].pop("energy_gate_percentile", None)
else:
    cfg["spectrogram"]["energy_gate_percentile"] = int("$gate_pct")
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(out)
PY
}

if [[ "$PROFILE" == "english_full" ]]; then
  add_run "cubical_full_reference" "$ROOT_DIR/configs/experiments/cubical_mel_best_field_svm.yaml"
  add_run "cubical_keep_low_gate10" "$ROOT_DIR/configs/experiments/ablation/cubical_best_band_keep_low.yaml"
  add_run "cubical_keep_low_gate12" "$ROOT_DIR/configs/experiments/ablation/cubical_best_band_keep_low_gate12.yaml"
  add_run "cubical_drop_low" "$ROOT_DIR/configs/experiments/ablation/cubical_best_band_drop_low.yaml"
  add_run "cubical_drop_mid" "$ROOT_DIR/configs/experiments/ablation/cubical_best_band_drop_mid.yaml"
  add_run "cubical_drop_high" "$ROOT_DIR/configs/experiments/ablation/cubical_best_band_drop_high.yaml"
  add_run "cubical_gate_off" "$ROOT_DIR/configs/experiments/ablation/cubical_best_band_keep_low_gate_off.yaml"
  add_run "cubical_h0_only" "$ROOT_DIR/configs/experiments/ablation/cubical_best_band_keep_low_h0_only.yaml"
  add_run "cubical_h1_only" "$ROOT_DIR/configs/experiments/ablation/cubical_best_band_keep_low_h1_only.yaml"

  add_run "morse_full_reference" "$(generate_morse_config morse_full_reference "$ROOT_DIR/configs/experiments/morse_smale_best_field_matched_svm.yaml" none 10 full)"
  add_run "morse_keep_low_gate10" "$(generate_morse_config morse_keep_low_gate10 "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" keep_low 10 full)"
  add_run "morse_keep_low_gate12" "$(generate_morse_config morse_keep_low_gate12 "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" keep_low 12 full)"
  add_run "morse_drop_low" "$(generate_morse_config morse_drop_low "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" drop_low 10 full)"
  add_run "morse_drop_mid" "$(generate_morse_config morse_drop_mid "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" drop_mid 10 full)"
  add_run "morse_drop_high" "$(generate_morse_config morse_drop_high "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" drop_high 10 full)"
  add_run "morse_gate_off" "$(generate_morse_config morse_gate_off "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" keep_low none full)"
  add_run "morse_counts_entropy" "$(generate_morse_config morse_counts_entropy "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" keep_low 10 counts_entropy)"
  add_run "morse_basin_fractions" "$(generate_morse_config morse_basin_fractions "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" keep_low 10 basin_fractions)"
  add_run "morse_merge_sequence" "$(generate_morse_config morse_merge_sequence "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" keep_low 10 merge_sequence)"
  add_run "morse_extrema_values" "$(generate_morse_config morse_extrema_values "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" keep_low 10 extrema_values)"
elif [[ "$PROFILE" == "compact" ]]; then
  add_run "cubical_full_reference" "$ROOT_DIR/configs/experiments/cubical_mel_best_field_svm.yaml"
  add_run "cubical_keep_low_gate10" "$ROOT_DIR/configs/experiments/ablation/cubical_best_band_keep_low.yaml"
  add_run "cubical_keep_low_gate12" "$ROOT_DIR/configs/experiments/ablation/cubical_best_band_keep_low_gate12.yaml"
  add_run "cubical_drop_low" "$ROOT_DIR/configs/experiments/ablation/cubical_best_band_drop_low.yaml"
  add_run "cubical_gate_off" "$ROOT_DIR/configs/experiments/ablation/cubical_best_band_keep_low_gate_off.yaml"

  add_run "morse_full_reference" "$(generate_morse_config morse_full_reference "$ROOT_DIR/configs/experiments/morse_smale_best_field_matched_svm.yaml" none 10 full)"
  add_run "morse_keep_low_gate10" "$(generate_morse_config morse_keep_low_gate10 "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" keep_low 10 full)"
  add_run "morse_keep_low_gate12" "$(generate_morse_config morse_keep_low_gate12 "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" keep_low 12 full)"
  add_run "morse_drop_low" "$(generate_morse_config morse_drop_low "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" drop_low 10 full)"
  add_run "morse_gate_off" "$(generate_morse_config morse_gate_off "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" keep_low none full)"
  add_run "morse_counts_entropy" "$(generate_morse_config morse_counts_entropy "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" keep_low 10 counts_entropy)"
  add_run "morse_basin_fractions" "$(generate_morse_config morse_basin_fractions "$ROOT_DIR/configs/experiments/ablation/morse_smale_best_band_keep_low_k4_norm_none.yaml" keep_low 10 basin_fractions)"
else
  echo "Unsupported PROFILE='$PROFILE'. Use 'english_full' or 'compact'." >&2
  exit 1
fi

fail_count=0
run_count=0

for item in "${RUNS[@]}"; do
  IFS="|" read -r run_name config_path <<< "$item"
  out_dir="$RESULTS_ROOT/${RUN_TAG}_${run_name}"
  log_file="$RESULTS_ROOT/logs/${RUN_TAG}_${run_name}.log"
  run_count=$((run_count + 1))

  echo "" | tee -a "$SUMMARY_LOG"
  echo "[$(date -Is)] START $run_name" | tee -a "$SUMMARY_LOG"
  echo "  config=$config_path" | tee -a "$SUMMARY_LOG"
  echo "  out_dir=${out_dir#$ROOT_DIR/}" | tee -a "$SUMMARY_LOG"

  cmd=(
    "$PYTHON_BIN" -u "$ROOT_DIR/src/scripts/run_pipeline.py"
    --config "$config_path"
    --train-protocol "$TRAIN_PROTOCOL"
    --train-audio-dir "$AUDIO_DIR"
    --eval-protocol "$EVAL_PROTOCOL"
    --eval-audio-dir "$AUDIO_DIR"
    --out-dir "$out_dir"
    --cache-dir "$CACHE_DIR"
    --train-workers "$TRAIN_WORKERS"
    --eval-workers "$EVAL_WORKERS"
    --progress-every "$PROGRESS_EVERY"
  )

  "${cmd[@]}" >"$log_file" 2>&1
  status=$?

  if [[ $status -ne 0 ]]; then
    echo "[$(date -Is)] FAIL  $run_name (exit=$status)" | tee -a "$SUMMARY_LOG"
    fail_count=$((fail_count + 1))
    continue
  fi

  metrics="$("$PYTHON_BIN" - <<PY
import json, pathlib
p = pathlib.Path(r"$out_dir") / "eval_results.json"
if p.exists():
    d = json.loads(p.read_text())
    print(f"auc={d.get('auc'):.3f} eer={d.get('eer'):.3f} n_eval={d.get('n_eval')}")
else:
    print("metrics=missing")
PY
)"
  echo "[$(date -Is)] DONE  $run_name ${metrics}" | tee -a "$SUMMARY_LOG"
done

echo "" | tee -a "$SUMMARY_LOG"
echo "Total runs: $run_count" | tee -a "$SUMMARY_LOG"
echo "Total failures: $fail_count" | tee -a "$SUMMARY_LOG"
echo "Summary log: ${SUMMARY_LOG#$ROOT_DIR/}" | tee -a "$SUMMARY_LOG"

if [[ $fail_count -ne 0 ]]; then
  exit 1
fi
