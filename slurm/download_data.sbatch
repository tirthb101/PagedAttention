#!/bin/bash
#SBATCH --job-name=pa-download
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Setup step 2 as a job: download models + datasets into $PA_SCRATCH.
#   sbatch slurm/download_data.sbatch                         # everything incl. OPT-13B (~30 GB)
#   sbatch --export=ALL,SMALL=1 slurm/download_data.sbatch    # OPT-125m + OPT-1.3B + datasets (~3 GB)
# Run only after setup_env.sbatch succeeded. Needs internet on the compute node.
# Resubmitting resumes an interrupted download. No GPU is requested (see setup_env.sbatch).
cd "$SLURM_SUBMIT_DIR"
echo "=== download job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
if ! timeout 10 curl -sI https://huggingface.co >/dev/null 2>&1; then
  echo "ERROR: this compute node cannot reach huggingface.co. Ask the admins where downloads are allowed."
  exit 1
fi
if [ -n "${SMALL:-}" ]; then bash slurm/download_data.sh --small; else bash slurm/download_data.sh; fi
