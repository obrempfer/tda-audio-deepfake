#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/.venv_lab/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv_lab/bin/python"
else
  echo "Missing project Python environment (.venv or .venv_lab)." >&2
  exit 1
fi
QUEUE_ROOT=""
NUM_WORKERS=1
POLL_SECONDS="${POLL_SECONDS:-5}"

while (($#)); do
  case "$1" in
    --queue-root)
      QUEUE_ROOT="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$QUEUE_ROOT" ]]; then
  echo "--queue-root is required" >&2
  exit 2
fi

mkdir -p "$QUEUE_ROOT"/{ready,claimed,done,failed}
export PYTHONPATH="$ROOT_DIR/src"

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait "${pids[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for idx in $(seq 1 "$NUM_WORKERS"); do
  "$PYTHON_BIN" "$ROOT_DIR/src/scripts/train_queue_worker.py" \
    --queue-root "$QUEUE_ROOT" \
    --poll-seconds "$POLL_SECONDS" \
    > "$QUEUE_ROOT/worker_${idx}.log" 2>&1 &
  pids+=("$!")
done

echo "Started $NUM_WORKERS classifier worker(s) for queue root: $QUEUE_ROOT"
wait "${pids[@]}"
