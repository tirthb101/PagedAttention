# Sourced at the top of every job script: loads config, conda env, offline mode.
set -eo pipefail   # no -u: conda/module scripts use unset variables

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_DIR="$(pwd)"
source "$REPO_DIR/slurm/cluster.env"

module load $PA_MODULES
source "$PA_CONDA_SH"
conda activate "$PA_CONDA_ENV"

# Compute nodes may have no internet: use only pre-downloaded models/datasets.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1          # logs appear instantly (same as python -u)
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
mkdir -p "$PA_RESULTS" "$REPO_DIR/logs"

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) at $(date) ==="
echo "git commit: $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo none)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}  CPUs: ${SLURM_CPUS_PER_TASK:-?}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
