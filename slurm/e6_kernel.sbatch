#!/bin/bash
#SBATCH --job-name=pa-e6-kernel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#
# E6 (Fig. 18a): decode attention kernel latency, Triton paged kernel vs contiguous PyTorch SDPA,
# OPT-13B dims, batch 8 and 32. Runs twice: CUDA-graph timing (kernel time, the headline) and
# eager timing (includes Python/launch overhead, the old method), one CSV each.
#
# From the repo root:  sbatch slurm/e6_kernel.sbatch
# Overrides go in the environment, not --export (sbatch --export splits on commas), e.g.
#   BATCHES=8 CONTEXTS=64,128,256 sbatch slurm/e6_kernel.sbatch

source slurm/common.sh

BATCHES="${BATCHES:-8,32}"
CONTEXTS="${CONTEXTS:-64,128,256,512,1024}"     # paper's Fig. 18a range is 64-256
SDPA_BACKENDS="${SDPA_BACKENDS:-flash,efficient,auto}"   # first one is the contiguous_us baseline
TRITON_WARPS="${TRITON_WARPS:-default,1,2,8}"   # first one is paged_triton_us (default = Triton's 4)
MODEL_DIMS="${MODEL_DIMS:-opt-13b}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"

OUT_DIR="$PA_RESULTS/fig18a_kernel"
mkdir -p "$OUT_DIR"
echo "batches=$BATCHES contexts=$CONTEXTS sdpa=$SDPA_BACKENDS warps=$TRITON_WARPS out=$OUT_DIR"

python -m pytest -q tests/test_paged_triton.py -k "not engine"   # kernel numerics + out= + graph capture

for TIMING in graph eager; do
  python -u benchmarks/kernel_bench.py \
    --model-dims "$MODEL_DIMS" --block-size "$BLOCK_SIZE" \
    --batches "$BATCHES" --contexts "$CONTEXTS" \
    --sdpa-backends "$SDPA_BACKENDS" --triton-warps "$TRITON_WARPS" \
    --timing "$TIMING" --out "$OUT_DIR/e6_kernel_${TIMING}.csv"
done
