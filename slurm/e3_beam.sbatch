#!/bin/bash
#SBATCH --job-name=pa-e3-beam
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-2%2     # beam width 2, 4, 6
#
# E3 beam search (paper Fig. 15 memory saving; Fig. 14b latency): OPT-13B, Alpaca, paged
# only (the Orca baselines do not implement beam forking). Optional. ~1-1.5 GPU-h.
# The CSV columns mem_saving_avg_pct / mem_saving_max_pct give the Fig. 15 numbers.
#
#   sbatch slurm/e3_beam.sbatch
#   RATES_B2=2,4,8 sbatch --array=0 slurm/e3_beam.sbatch    # lists via env vars (commas!)

source slurm/common.sh

BEAMS=(2 4 6)
BEAM="${BEAM:-${BEAMS[${SLURM_ARRAY_TASK_ID:-0}]}}"

EXP="${EXP:-e3_beam}"
MODEL="${MODEL:-facebook/opt-13b}"
TRACE="${TRACE:-alpaca}"
RATES_B2="${RATES_B2:-4,8,12}"
RATES_B4="${RATES_B4:-2,4,6}"
RATES_B6="${RATES_B6:-1,2,4}"
RATES_VAR="RATES_B${BEAM}"
RATES="${RATES:-${!RATES_VAR}}"
DURATION="${DURATION:-180}"
KV_GB="${KV_GB:-12}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
WALL_FACTOR="${WALL_FACTOR:-2}"

OUT="$PA_RESULTS/$EXP/${TRACE}_paged_beam${BEAM}_bs${BLOCK_SIZE}.csv"
mkdir -p "$(dirname "$OUT")"
echo "E3 beam: width=$BEAM model=$MODEL trace=$TRACE rates=$RATES duration=$DURATION out=$OUT"

python -u benchmarks/run_serving.py \
  --model "$MODEL" --trace "$TRACE" --rates "$RATES" --duration "$DURATION" \
  --memory-manager paged --kv-cache-gb "$KV_GB" --block-size "$BLOCK_SIZE" \
  --preemption recompute --beam "$BEAM" --max-wall-factor "$WALL_FACTOR" \
  --seed 0 --out "$OUT"
