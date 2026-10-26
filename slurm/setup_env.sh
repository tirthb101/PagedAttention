#!/bin/bash
# One-time setup. Run on the LOGIN node (it has internet), from the repo root:
#   cp slurm/cluster.env.example slurm/cluster.env   # check values first
#   bash slurm/setup_env.sh
# Creates the conda env and pre-downloads models + datasets so jobs can run offline.
set -eo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/slurm/cluster.env"

mkdir -p "$REPO_DIR/logs"   # sbatch --output=logs/... needs this to exist
mkdir -p "$PA_SCRATCH" "$HF_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$PA_RESULTS"

module load $PA_MODULES
source "$PA_CONDA_SH"
if ! conda env list | grep -q "^$PA_CONDA_ENV "; then
  conda create -y -n "$PA_CONDA_ENV" python=3.11
fi
conda activate "$PA_CONDA_ENV"
pip install --upgrade pip
pip install -r "$REPO_DIR/requirements.txt"

# Models: small ones first so smoke tests can start while the big one downloads.
for m in facebook/opt-125m facebook/opt-1.3b facebook/opt-13b; do
  echo "Downloading $m ..."
  huggingface-cli download "$m" --exclude "*.msgpack" "*.h5" "*tf_model*" "*flax*"
done

# Datasets used by benchmarks/trace.py
huggingface-cli download anon8231489123/ShareGPT_Vicuna_unfiltered \
  ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset
huggingface-cli download tatsu-lab/alpaca --repo-type dataset
huggingface-cli download wmt/wmt16 --repo-type dataset --include "de-en/*"

echo "Setup done. Next: sbatch slurm/check_gpu.sbatch"
