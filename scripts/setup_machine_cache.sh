#!/usr/bin/env bash
set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$SCRATCH_ROOT/tda_deepfake_runtime}"

mkdir -p \
  "$RUNTIME_ROOT/feature_cache" \
  "$RUNTIME_ROOT/topology_nn_cache" \
  "$RUNTIME_ROOT/sample_explanations" \
  "$RUNTIME_ROOT/sample_explanation_cache" \
  "$RUNTIME_ROOT/pycache" \
  "$RUNTIME_ROOT/xdg" \
  "$RUNTIME_ROOT/mpl" \
  "$SCRATCH_ROOT/hf-home" \
  "$SCRATCH_ROOT/hf-home-auth" \
  "$SCRATCH_ROOT/pip-cache" \
  "/tmp/$USER"

ln -sfn "$RUNTIME_ROOT" "/tmp/$USER/tda_deepfake_runtime"
ln -sfn "$SCRATCH_ROOT/hf-home" "/tmp/$USER/hf-home"
ln -sfn "$SCRATCH_ROOT/hf-home-auth" "/tmp/$USER/hf-home-auth"
ln -sfn "$SCRATCH_ROOT/pip-cache" "/tmp/$USER/pip-cache"

cat <<EOF
SCRATCH_ROOT=$SCRATCH_ROOT
RUNTIME_ROOT=$RUNTIME_ROOT
export RUNTIME_ROOT=$RUNTIME_ROOT
export XDG_CACHE_HOME=$RUNTIME_ROOT/xdg
export PYTHONPYCACHEPREFIX=$RUNTIME_ROOT/pycache
export MPLCONFIGDIR=$RUNTIME_ROOT/mpl
export PIP_CACHE_DIR=$SCRATCH_ROOT/pip-cache
EOF
