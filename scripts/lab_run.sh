#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv_lab/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing $PYTHON_BIN. Create the lab venv first." >&2
  exit 1
fi

# Keep import/cache churn off home quota and on local disk for faster startup.
RUNTIME_ROOT="/tmp/${USER}/tda_deepfake_runtime"
export MPLCONFIGDIR="$RUNTIME_ROOT/mpl"
export PYTHONPYCACHEPREFIX="$RUNTIME_ROOT/pycache"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg"
mkdir -p "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX" "$XDG_CACHE_HOME"

if [[ "${1:-}" == "--warm-only" ]]; then
  shift
  "$PYTHON_BIN" -c "import matplotlib, scipy, sklearn, gtda, topopy"
  echo "Warm import complete."
  exit 0
fi

if [[ "${1:-}" == "--python" ]]; then
  shift
  exec "$PYTHON_BIN" "$@"
fi

if [[ "${1:-}" != "run-pipeline" ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/lab_run.sh --warm-only
  scripts/lab_run.sh --python <args...>
  scripts/lab_run.sh run-pipeline <pipeline args...>
EOF
  exit 2
fi

shift
exec env PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" -u "$ROOT_DIR/src/scripts/run_pipeline.py" "$@"
