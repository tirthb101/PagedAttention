#!/bin/bash
#SBATCH --job-name=pa-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00   # cluster max is 5 days; final 1-hour-trace runs: --time=1-00:00:00
#SBATCH --output=logs/%x-%A_%a.out
# (no --mem: matches our known-working job. If a 13B job is OOM-killed while
#  loading weights, add:  #SBATCH --mem=64G)
#
# One array task = one memory manager (system), looping over all request rates.
# The model is loaded once per task; one CSV row is appended per finished rate,
# so a job that hits its time limit keeps the rates it finished.
#
# Example (E1, ShareGPT, OPT-13B, 4 systems), from the repo root:
#   sbatch --array=0-3 --export=ALL,EXP=e1,MODEL=facebook/opt-13b,TRACE=sharegpt slurm/serve_sweep.sbatch
#
# Smoke test first:
#   sbatch --array=0 --time=00:20:00 \
#     --export=ALL,EXP=smoke,MODEL=facebook/opt-125m,TRACE=sharegpt,RATES=1.0,DURATION=60 slurm/serve_sweep.sbatch

source slurm/common.sh

SYSTEMS=(paged orca-oracle orca-pow2 orca-max)
SYSTEM="${SYSTEM:-${SYSTEMS[${SLURM_ARRAY_TASK_ID:-0}]}}"

EXP="${EXP:?set EXP, e.g. e1}"
MODEL="${MODEL:-facebook/opt-1.3b}"
TRACE="${TRACE:-sharegpt}"
RATES="${RATES:-0.25,0.5,0.75,1.0,1.5,2.0,3.0}"   # requests/second; tune after the OPT-1.3B dry run
DURATION="${DURATION:-600}"                      # seconds of trace per rate; final headline runs use 3600 (= paper)
KV_GB="${KV_GB:-12}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
PREEMPTION="${PREEMPTION:-recompute}"
EXTRA_ARGS="${EXTRA_ARGS:-}"                     # e.g. "--n 4" or "--beam 4" or "--prefix 5shot"

OUT="$PA_RESULTS/$EXP/${TRACE}_${SYSTEM}_bs${BLOCK_SIZE}_${PREEMPTION}.csv"
mkdir -p "$(dirname "$OUT")"
echo "system=$SYSTEM model=$MODEL trace=$TRACE rates=$RATES out=$OUT"

python -u benchmarks/run_serving.py \
  --model "$MODEL" --trace "$TRACE" --rates "$RATES" --duration "$DURATION" \
  --memory-manager "$SYSTEM" --kv-cache-gb "$KV_GB" --block-size "$BLOCK_SIZE" \
  --preemption "$PREEMPTION" --seed 0 --out "$OUT" $EXTRA_ARGS
