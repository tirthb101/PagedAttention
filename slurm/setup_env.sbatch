#!/bin/bash
#SBATCH --job-name=pa-setup
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Setup step 1 as a job: create the conda env and pip-install requirements.txt.
#   sbatch slurm/setup_env.sbatch
# Needs internet on the compute node (pip). The log says clearly if there is none.
# No GPU is requested. If the cluster rejects jobs without a GPU on this partition,
# either add  #SBATCH --gres=gpu:1  here, or use a CPU partition (see `sinfo`).
cd "$SLURM_SUBMIT_DIR"
echo "=== setup job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
if ! timeout 10 curl -sI https://pypi.org >/dev/null 2>&1; then
  echo "ERROR: this compute node has no internet, so pip cannot install packages."
  echo "Ask the admins how students should install Python packages (login node, data-transfer node, or a module)."
  exit 1
fi
bash slurm/setup_env.sh
