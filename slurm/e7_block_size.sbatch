#!/bin/bash
#SBATCH --job-name=pa-e7
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-17%2    # 2 traces x 9 block sizes, at most 2 running at once (our 2-GPU limit)
#
# E7 (paper Fig. 18b): normalized latency vs block size under MEMORY PRESSURE, ShareGPT and Alpaca.
#
# Why a second E7: the first sweep (jobs 7694-7711) used a 12 GB KV cache. OPT-1.3B needs
# 192 KiB of KV per token, so 12 GB = 65,536 token slots: even with 256-token blocks the cache
# never limited the batch at 12 req/s, so the large-block penalty (internal fragmentation ->
# fewer sequences fit -> smaller batches / preemption) could not appear. Here the KV budget is
# fixed in BYTES (same for every block size, as in the paper) and small enough to bind:
#   ShareGPT: 1.5 GiB =  8,192 token slots (= 32 blocks of 256). The 1.5 GiB E8b runs at 12 req/s
#             saturated at 8.2 req/s (bs16) / 8.9 (bs64) / 6.3 (bs256), so 6 and 8 req/s sit
#             just below / at the knee for good block sizes and above it for bs256.
#   Alpaca:   0.75 GiB = 4,096 token slots (= 16 blocks of 256). An Alpaca request is ~80 tokens,
#             so bs256 wastes ~3/4 of every block: at most ~16 requests run at once (~30 req/s of
#             capacity at ~0.55 s per request), whereas bs16 fits ~70. 32 req/s is at the bs256
#             knee and 48 req/s is beyond it; E1 (12 GB) served both with batch 13-24, no preemption.
#
# One array task = one (trace, block size); the model is loaded once and both rates run in
# turn (one log line per rate). Task order interleaves traces and starts with the block sizes
# that matter most, so a partial run is already plottable:
#   task i -> trace = (sharegpt, alpaca)[i % 2], block = ORDER[i / 2]
#
# Usage (from the repo root; see slurm/e7_submit.sh for the wrapper):
#   sbatch slurm/e7_block_size.sbatch                        # all 18 tasks
#   sbatch --array=0,1 slurm/e7_block_size.sbatch            # pilot: bs16 on both traces
#   TRACE=alpaca BLOCK_SIZE=256 sbatch --array=0 slurm/e7_block_size.sbatch   # one point by hand
# Per-trace overrides: KV_GB_SHAREGPT, KV_GB_ALPACA, RATES_SHAREGPT, RATES_ALPACA (a plain RATES or
# KV_GB left over from serve_sweep is deliberately ignored). Multi-value lists must be set in the environment BEFORE sbatch: --export splits on commas.

source slurm/common.sh

TRACES=(sharegpt alpaca)
ORDER=(16 256 64 128 32 8 4 2 1)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
TRACE="${TRACE:-${TRACES[$((IDX % 2))]}}"
BLOCK_SIZE="${BLOCK_SIZE:-${ORDER[$((IDX / 2))]}}"

MODEL="${MODEL:-facebook/opt-1.3b}"
if [ "$TRACE" = "alpaca" ]; then
  KV_GB="${KV_GB_ALPACA:-0.75}"
  RATES="${RATES_ALPACA:-32,48}"
else
  KV_GB="${KV_GB_SHAREGPT:-1.5}"
  RATES="${RATES_SHAREGPT:-6,8}"
fi
DURATION="${DURATION:-150}"          # seconds of Poisson arrivals per rate (ShareGPT@8 -> 1,200 requests)
WALL_FACTOR="${WALL_FACTOR:-1.5}"    # overloaded runs stop at DURATION*WALL_FACTOR+120 s; unfinished
                                     # requests are dropped (timed_out=1 -> latency is a lower bound)
PREEMPTION="${PREEMPTION:-recompute}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TAG="$(basename "$MODEL")_kv${KV_GB}"
OUT="$PA_RESULTS/e7_mem/${TAG}/${TRACE}_paged_bs${BLOCK_SIZE}_${PREEMPTION}.csv"
mkdir -p "$(dirname "$OUT")"
# Header lines read by plot_e7_e8b.py (and compatible with parse_logs.py's regex).
echo "e7: kv_cache_gb=$KV_GB duration=$DURATION wall_factor=$WALL_FACTOR"
echo "system=paged model=$MODEL trace=$TRACE rates=$RATES out=$OUT"

python -u benchmarks/run_serving.py \
  --model "$MODEL" --trace "$TRACE" --rates "$RATES" --duration "$DURATION" \
  --memory-manager paged --kv-cache-gb "$KV_GB" --block-size "$BLOCK_SIZE" \
  --preemption "$PREEMPTION" --max-wall-factor "$WALL_FACTOR" --seed 0 --out "$OUT" $EXTRA_ARGS
