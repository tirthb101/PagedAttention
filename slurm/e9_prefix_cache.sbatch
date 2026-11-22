#!/bin/bash
#SBATCH --job-name=pa-e9
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-8%2     # at most 2 at once (our 2-GPU limit)
#
# E9 = extension X1, automatic prefix caching. OPT-13B, paged engine only.
#
#   Workload B, shared system prompt (Alpaca + a SYS_LEN-token system prompt), rates RATES_B:
#     0  no caching          (paper's vLLM, every request prefills the system prompt)
#     1  --register-prefix   (paper §4.4: operator declares the system prompt in advance)
#     2  --prefix-caching    (ours, X1: found automatically)
#   Workload A, multi-turn ShareGPT chat (closed loop, rate = new conversations/s), rates RATES_A:
#     3  no caching
#     4  --prefix-caching
#   Hit rate vs KV pool size (workload A, caching on, rate RATE_SWEEP):
#     5..8  KV_GB = 2, 4, 6, 8
#
# Budget: ~30-45 min for tasks 0-4, ~15 min for 5-8 -> ~3.5 GPU-h, ~2 h wall on 2 GPUs.
#
# Submit from the repo root (pass comma lists as env vars in FRONT of sbatch):
#   sbatch slurm/e9_prefix_cache.sbatch
#   RATES_B=2,4,8 sbatch --array=0-2 slurm/e9_prefix_cache.sbatch
# Smoke test (OPT-125m, 30 s, one rate per task):
#   MODEL=facebook/opt-125m EXP=e9_smoke RATES_B=4 RATES_A=0.5 RATE_SWEEP=0.5 DURATION=30 \
#     sbatch --time=00:20:00 slurm/e9_prefix_cache.sbatch

source slurm/common.sh

TASK="${SLURM_ARRAY_TASK_ID:-0}"
EXP="${EXP:-e9_prefix_cache}"
MODEL="${MODEL:-facebook/opt-13b}"
DURATION="${DURATION:-180}"
KV_GB="${KV_GB:-12}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
WALL_FACTOR="${WALL_FACTOR:-2}"
SYS_LEN="${SYS_LEN:-1000}"          # system-prompt tokens (workload B)
NUM_SYS="${NUM_SYS:-1}"             # distinct system prompts (workload B)
TURNS="${TURNS:-5}"                 # max turns per conversation (workload A)
THINK="${THINK:-5}"                 # mean user think time between turns, s (workload A)
RATES_B="${RATES_B:-2,4,8,16,24}"
RATES_A="${RATES_A:-0.25,0.5,1,1.5}"
RATE_SWEEP="${RATE_SWEEP:-1}"
SWEEP_KV=(2 4 6 8)
EXTRA_ARGS="${EXTRA_ARGS:-}"

case "$TASK" in
  0) WL=sysprompt; MODE=nocache;  TRACE=alpaca;   RATES="$RATES_B"; FLAGS="" ;;
  1) WL=sysprompt; MODE=manual;   TRACE=alpaca;   RATES="$RATES_B"; FLAGS="--register-prefix" ;;
  2) WL=sysprompt; MODE=auto;     TRACE=alpaca;   RATES="$RATES_B"; FLAGS="--prefix-caching" ;;
  3) WL=chat;      MODE=nocache;  TRACE=sharegpt; RATES="$RATES_A"; FLAGS="" ;;
  4) WL=chat;      MODE=auto;     TRACE=sharegpt; RATES="$RATES_A"; FLAGS="--prefix-caching" ;;
  5|6|7|8)
     WL=chat; MODE=auto; TRACE=sharegpt; RATES="$RATE_SWEEP"; FLAGS="--prefix-caching"
     KV_GB="${SWEEP_KV[$((TASK - 5))]}" ;;
  *) echo "unknown task $TASK"; exit 1 ;;
esac

if [ "$WL" = sysprompt ]; then
  FLAGS="$FLAGS --sys-prompt-len $SYS_LEN --num-sys-prompts $NUM_SYS"
  OUT="$PA_RESULTS/$EXP/${WL}${SYS_LEN}_${MODE}_bs${BLOCK_SIZE}.csv"
else
  FLAGS="$FLAGS --chat-turns $TURNS --think-time $THINK"
  OUT="$PA_RESULTS/$EXP/${WL}_${MODE}_kv${KV_GB}_bs${BLOCK_SIZE}.csv"
fi
mkdir -p "$(dirname "$OUT")"
echo "E9: task=$TASK workload=$WL mode=$MODE model=$MODEL trace=$TRACE rates=$RATES kv=${KV_GB}GB out=$OUT"

python -u benchmarks/run_serving.py \
  --model "$MODEL" --trace "$TRACE" --rates "$RATES" --duration "$DURATION" \
  --memory-manager paged --kv-cache-gb "$KV_GB" --block-size "$BLOCK_SIZE" \
  --max-wall-factor "$WALL_FACTOR" --seed 0 --out "$OUT" $FLAGS $EXTRA_ARGS
