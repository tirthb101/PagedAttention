#!/bin/bash
#SBATCH --job-name=pa-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# GPU tests: Triton kernel vs reference, and the engine with the Triton backend, on CUDA.
# From the repo root:  sbatch slurm/test_gpu.sbatch

source slurm/common.sh
python -m pytest -q tests/test_paged_triton.py   # Triton kernel on the real GPU (~1 min)
