#!/bin/bash
# One-time setup, step 1 of 2: create the conda env and install the Python packages.
# Light work (a pip install), normally fine on the login node. From the repo root:
#   cp slurm/cluster.env.example slurm/cluster.env   # check values first
#   bash slurm/setup_env.sh
# Step 2 (models + datasets, ~30 GB) is slurm/download_data.sh - read its header first.
set -eo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/slurm/cluster.env"

mkdir -p "$REPO_DIR/logs"   # sbatch --output=logs/... needs this to exist
mkdir -p "$PA_SCRATCH" "$HF_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$PA_RESULTS"

module load $PA_MODULES
source "$PA_CONDA_SH"
# Create the env unless a complete one exists. A folder left behind by a failed or interrupted
# create (no conda-meta/history inside) is removed first; it only ever lives in your own ~/.conda.
ENV_DIR="$HOME/.conda/envs/$PA_CONDA_ENV"
if [ -d "$ENV_DIR" ] && [ ! -f "$ENV_DIR/conda-meta/history" ]; then
  echo "Removing incomplete environment folder $ENV_DIR (left by an earlier failed attempt)"
  rm -rf "$ENV_DIR"
fi
if ! conda env list | awk '{print $1}' | grep -qx "$PA_CONDA_ENV"; then
  conda create -y -n "$PA_CONDA_ENV" python=3.11
fi
conda activate "$PA_CONDA_ENV"
pip install --upgrade pip
pip install -r "$REPO_DIR/requirements.txt"

echo "Environment ready. Next: slurm/download_data.sh (read its header: where downloads are allowed)"
