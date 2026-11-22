#!/bin/bash
#SBATCH --job-name=pa-e3-par
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-8%2     # 3 sample counts x 3 systems, at most 2 at once (our 2-GPU limit)
#
# E3 parallel sampling (paper Fig. 14a): OPT-13B, Alpaca, n = 2, 4, 6, paged vs Orca
# (Oracle) vs Orca (Max). Array task i -> n = NS[i / 3], system = SYSTEMS[i % 3]:
#   0 paged n2   1 oracle n2   2 max n2
#   3 paged n4   4 oracle n4   5 max n4
#   6 paged n6   7 oracle n6   8 max n6
# Each task loads the model once and appends one CSV row per rate (incl. the memory-saving
# columns mem_saving_avg_pct / mem_saving_max_pct; 0 for Orca). Samples use temperature 1.0.
#
# Budget: 3-minute traces, 3 rates per task, wall cap 2x duration (+120 s) per rate:
# ~15-20 min for paged/Oracle tasks, <= ~30 min for Orca-Max (it saturates early) -> ~3 GPU-h.
#
# Submit from the repo root. sbatch --export splits on commas, so pass rate lists as
# environment variables in front of sbatch (the default --export=ALL forwards them):
#   sbatch slurm/e3_parallel.sbatch
#   RATES_N2=4,8,12,16 DURATION=300 sbatch slurm/e3_parallel.sbatch
#   sbatch --array=0,3,6 slurm/e3_parallel.sbatch          # paged only
# Smoke test (OPT-125m, 1 rate, 30 s):
#   MODEL=facebook/opt-125m EXP=e3_smoke RATES_N2=2 DURATION=30 sbatch --array=0-2 --time=00:20:00 slurm/e3_parallel.sbatch

source slurm/common.sh

SYSTEMS=(paged orca-oracle orca-max)
NS=(2 4 6)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
SYSTEM="${SYSTEM:-${SYSTEMS[$((TASK % 3))]}}"
N="${N:-${NS[$((TASK / 3))]}}"

EXP="${EXP:-e3_parallel}"
MODEL="${MODEL:-facebook/opt-13b}"
TRACE="${TRACE:-alpaca}"
# Request rates (req/s) per n. Paged OPT-13B Alpaca saturates at ~36 req/s with n=1 (E1);
# n samples cost ~n x the decode work, so the ranges shrink with n.
RATES_N2="${RATES_N2:-6,12,18}"
RATES_N4="${RATES_N4:-3,6,9}"
RATES_N6="${RATES_N6:-2,4,6}"
RATES_VAR="RATES_N${N}"
RATES="${RATES:-${!RATES_VAR}}"
DURATION="${DURATION:-180}"
KV_GB="${KV_GB:-12}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
PREEMPTION="${PREEMPTION:-recompute}"   # n>1 groups are swapped anyway (recompute if swap space is full)
WALL_FACTOR="${WALL_FACTOR:-2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

OUT="$PA_RESULTS/$EXP/${TRACE}_${SYSTEM}_n${N}_bs${BLOCK_SIZE}.csv"
mkdir -p "$(dirname "$OUT")"
echo "E3 parallel: system=$SYSTEM n=$N model=$MODEL trace=$TRACE rates=$RATES duration=$DURATION out=$OUT"

python -u benchmarks/run_serving.py \
  --model "$MODEL" --trace "$TRACE" --rates "$RATES" --duration "$DURATION" \
  --memory-manager "$SYSTEM" --kv-cache-gb "$KV_GB" --block-size "$BLOCK_SIZE" \
  --preemption "$PREEMPTION" --n "$N" --max-wall-factor "$WALL_FACTOR" \
  --seed 0 --out "$OUT" $EXTRA_ARGS
