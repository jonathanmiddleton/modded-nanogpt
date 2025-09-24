#!/usr/bin/env bash
set -euo pipefail

# Only run heavy setup once
INIT_MARKER="$CONDA_PREFIX/.modded_initialized"
if [[ -f "$INIT_MARKER" ]]; then
  return 0
fi

# Choose an install location inside the env
MODDED_HOME="$HOME/modded"
REPO_URL="https://github.com/jonathanmiddleton/modded-nanogpt.git"

mkdir -p "$(dirname "$MODDED_HOME")"

if [[ ! -d "$MODDED_HOME/.git" ]]; then
  echo "[modded] Cloning repository into $MODDED_HOME ..."
  git clone "$REPO_URL" "$MODDED_HOME"
else
  echo "[modded] Repository already present at $MODDED_HOME"
fi

# Install Python requirements from the repo
if [[ -f "$MODDED_HOME/requirements.txt" ]]; then
  echo "[modded] Installing root Python requirements..."
  pip install -r "$MODDED_HOME/requirements.txt"
fi

if [[ -f "$MODDED_HOME/data/requirements.txt" ]]; then
  echo "[modded] Installing data Python requirements..."
  pip install -r "$MODDED_HOME/data/requirements.txt"
fi

# We get the entire 10B pre-tokenized dataset
if [[ -f "$MODDED_HOME/data/cached_fineweb10B.py" ]]; then
  echo "[modded] Running data preparation script..."
  python "$MODDED_HOME/data/cached_fineweb10B.py"
fi

# Torch nightly with CUDA 12.6
echo "[modded] Installing nightly torch (CUDA 12.6)..."
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade

# Mark as initialized
touch "$INIT_MARKER"

# Export a helper env var for convenience
export MODDED_HOME
echo "[modded] Setup complete. MODDED_HOME=$MODDED_HOME"