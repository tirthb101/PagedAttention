#!/bin/bash
# Submit the E7 memory-pressure block-size sweep (paper Fig. 18b): 2 traces x 9 block sizes.
# Run from the repo root on the login node.
#
#   bash slurm/e7_submit.sh pilot     # 2 jobs: bs16 on ShareGPT and Alpaca (~15 min). Check first:
#                                     #   bs16 should show few/no preemptions and latency near E1's
#                                     #   (ShareGPT ~0.01 s/tok, Alpaca ~0.007-0.009 s/tok).
#   bash slurm/e7_submit.sh rest      # after a good pilot: the other 16 runs (array 2-17%2)
#   bash slurm/e7_submit.sh           # the full sweep as ONE array job, throttled to 2 GPUs (%2)
#   bash slurm/e7_submit.sh loop      # same 18 runs as separate jobs, chained in 2 lanes with
#                                     #   --dependency=afterany (for a QOS that dislikes arrays)
#
# Overrides go in the environment (comma lists must NOT go through --export), e.g.
#   RATES_ALPACA=24,40 KV_GB_ALPACA=0.5 bash slurm/e7_submit.sh
#   MODEL=facebook/opt-13b KV_GB_SHAREGPT=6.25 KV_GB_ALPACA=3.1 RATES_SHAREGPT=1.5,2 \
#     RATES_ALPACA=8,12 bash slurm/e7_submit.sh       # optional OPT-13B version (see report)
#
# Afterwards: copy logs/pa-e7-*.out to the PC under logs/fig18b_e7_mem/ and run
#   python "Claude outputs/report_src/plot_e7_e8b.py" .
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
export MODEL="${MODEL:-facebook/opt-1.3b}"
for v in RATES_SHAREGPT RATES_ALPACA KV_GB_SHAREGPT KV_GB_ALPACA DURATION WALL_FACTOR PREEMPTION EXTRA_ARGS; do
  if [ -n "${!v:-}" ]; then export "$v"; fi
done

unset TRACE BLOCK_SIZE   # set per job below; a leftover value would pin every array task
MODE="${1:-array}"
case "$MODE" in
  pilot)
    sbatch --job-name=pa-e7 --array=0,1 slurm/e7_block_size.sbatch ;;
  rest)
    sbatch --job-name=pa-e7 --array=2-17%2 slurm/e7_block_size.sbatch ;;
  array)
    sbatch --job-name=pa-e7 --array=0-17%2 slurm/e7_block_size.sbatch ;;
  loop)
    # Same order as the array (ShareGPT on lane 0, Alpaca on lane 1); each job waits for the
    # previous job in its lane, so at most 2 GPUs are in use.
    prev=("" "")
    for b in 16 256 64 128 32 8 4 2 1; do
      for lane in 0 1; do
        trace=$([ "$lane" = 0 ] && echo sharegpt || echo alpaca)
        dep=${prev[$lane]:+--dependency=afterany:${prev[$lane]}}
        id=$(TRACE=$trace BLOCK_SIZE=$b sbatch --parsable $dep --job-name="pa-e7-${trace}-bs$b" \
               --array=0 --output="logs/%x-%j.out" slurm/e7_block_size.sbatch)
        echo "submitted $trace bs$b -> $id"
        prev[$lane]=$id
      done
    done ;;
  *) echo "usage: $0 [pilot|rest|array|loop]" >&2; exit 1 ;;
esac
squeue -u "$USER" -o "%.12i %.22j %.8T %.10M %R" || true
