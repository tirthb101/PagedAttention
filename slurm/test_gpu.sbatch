#!/bin/bash
#SBATCH --job-name=pa-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# GPU-only tests (Triton kernel vs reference, e2e equivalence on CUDA).
# From the repo root:  sbatch slurm/test_gpu.sbatch

source slurm/common.sh
python -m pytest -q -m gpu tests/
