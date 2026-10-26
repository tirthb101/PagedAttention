# Experiments

Every result we reproduce from the paper, how we run it on one L40S (college SLURM cluster), and what we expect to see.
All runs write a CSV to `results/` and are plotted by `benchmarks/plot.py`, so the slides can show our curve next to the paper's figure.

## Common setup

| Item | Paper | Ours |
| --- | --- | --- |
| GPU | A100 40/80 GB | L40S 48 GB (864 GB/s — slower than A100, so absolute rates will be lower) |
| Main model | OPT-13B on 1 GPU | OPT-13B on 1 GPU (FP16, ≈26 GB) |
| Fast-iteration model | — | OPT-1.3B |
| KV cache pool | 12 GB (13B config, Table 1) | **12 GB cap** to match paper; one extra run uncapped (~18 GB) |
| Block size | 16 | 16 (swept in Fig. 18b) |
| Arrivals | Poisson, varying rate | Poisson, 6–8 rates from low load to saturation |
| Trace length | 1 hour | 10 min per point for development and ablations; **1 hour (same as paper) for the final E1 headline runs** — jobs may run up to 5 days |
| Metric | Normalized latency = mean over requests of (end-to-end latency / output tokens), s/token | same |
| Baselines | FasterTransformer; Orca Oracle / Pow2 / Max (their reimplementation) | Orca Oracle / Pow2 / Max as `memory_manager` flags; static batching as optional FT stand-in |

**Datasets** (downloaded once on the cluster login node by `slurm/setup_env.sh`):
- ShareGPT: `ShareGPT_V3_unfiltered_cleaned_split.json` (HF dataset `anon8231489123/ShareGPT_Vicuna_unfiltered`) — use input/output token lengths of real conversations.
- Alpaca: HF `tatsu-lab/alpaca` — instruction as input, reference response length as output.
- WMT16 En–De: HF `wmt/wmt16` (`de-en`) — for the shared-prefix experiment.

Outputs are forced to the trace's output length (`ignore_eos=True`) so that every system does the same work, as in the paper.

**How to read a latency-vs-rate plot:** latency is flat at low load, then shoots up when the system cannot keep up. The request rate at which the curve "takes off" is the system's capacity. The paper's claim is that vLLM's knee is 1.7–2.7× further right than Orca's.

---

## E0 — Memory waste snapshot (Fig. 2) — no GPU needed

- **Script:** `benchmarks/simulate_memory.py --trace sharegpt --model opt-13b --kv-gb 12`
- **What:** replay the trace through each block manager (no model); every step record used slots / reserved slots, and the number of running requests.
- **Paper:** existing systems use only 20.4–38.2% of KV memory for real tokens; vLLM ≈96% *(verify on figure)*.
- **Expect:** paged ≫ Oracle > Pow2 > Max on utilisation and on batch size.
- **Why first:** runs on a laptop on Day 7, gives the talk's motivating chart before any GPU run.

## E1 — Basic sampling, latency vs request rate (Fig. 12) — **headline result**

- **Script:** `benchmarks/run_serving.py --model opt-13b --trace sharegpt --rates 0.25,0.5,...` × `--memory-manager paged|orca-oracle|orca-pow2|orca-max`
- **Also:** same with `--trace alpaca`.
- **Paper:** ShareGPT: 1.7–2.7× rate vs Orca (Oracle), 2.7–8× vs Orca (Max). Alpaca: similar trend.
- **Expect:** same ordering; absolute rates lower than paper (slower GPU).
- **GPU time:** two passes.
  1. *Quick pass* (10-min traces): 4 systems × 2 traces × ~7 rates ≈ 11 GPU-h. Run on OPT-1.3B first (~1.5 h) to find the right rate range, then OPT-13B.
  2. *Final pass* (1-hour traces, exactly like the paper): ≈ 56–70 GPU-h → about 1.5 days of wall-clock with 2 GPUs. Submit as `--array=0-3%2 --time=1-00:00:00 --export=...,DURATION=3600` by **Sat 14 Nov** so it finishes with margin. If it doesn't finish, the quick-pass results are the fallback.

## E2 — Batch size (Fig. 13)

- **Same runs as E1** — log the average number of running requests per step.
- **Paper:** OPT-13B ShareGPT: 2.2× more requests than Orca (Oracle), 4.3× vs Orca (Max).

## E3 — Parallel sampling and beam search (Fig. 14, 15)

- **Script:** `run_serving.py --trace alpaca --n 2|4|6` and `--beam 2|4|6`, paged vs orca-oracle.
- **Metrics:** normalized latency vs rate; **memory saving** = 1 − (blocks with sharing / blocks without sharing), from block-manager counters.
- **Paper:** gain over Oracle 1.3× (n=1) → 2.3× (beam 6); saving 6.1–9.8% parallel, 37.6–55.2% beam (Alpaca); 16.2–30.5% / 44.3–66.3% (ShareGPT).
- **Cut:** beam search part is optional (see PLAN cut order).

## E4 — Shared prefix (Fig. 16)

- **Script:** `run_serving.py --trace wmt16 --prefix 1shot|5shot`
- **Model:** paper uses LLaMA-13B; we use OPT-13B (state the change) or a LLaMA-family model if access is easy.
- **Paper:** 1.67× (1-shot, 80-token prefix), 3.58× (5-shot, 341-token prefix) over Orca (Oracle).
- **Expect:** gain grows with prefix length.

## E5 — Chatbot (Fig. 17) — optional

- ShareGPT conversations; prompt = history truncated to last 1024 tokens; output ≤ 1024; no KV kept between turns.
- **Paper:** 2× rate vs all Orca variants.

## E6 — Kernel microbenchmark (Fig. 18a)

- **Script:** `benchmarks/kernel_bench.py` — decode attention latency for context 64–256, batch 8 and 32.
- **Compare:** our Triton paged kernel vs contiguous attention (PyTorch SDPA / FlashAttention) and vs our PyTorch reference.
- **Paper:** paged kernel 20–26% slower than FasterTransformer's.
- **Expect:** some overhead; if ours is much slower, explain why (no warp-level tuning) — honest is fine.

## E7 — Block size (Fig. 18b)

- **Script:** `run_serving.py --block-size 1,2,4,8,16,32,64,128,256` at one fixed rate per trace.
- **Paper:** best 16–128 on ShareGPT, 16–32 on Alpaca.
- **Use OPT-1.3B** with a reduced KV cap to keep runs short.

## E8 — Recompute vs swap (Fig. 19)

- **(a) Microbenchmark:** time to swap out+in vs recompute N tokens, for block sizes 1–256.
- **(b) End-to-end:** `run_serving.py --preemption recompute|swap` at a rate that forces preemption, block sizes 1–256.
- **Paper:** recompute ≤ 20% of swap latency at small blocks; swap better at large blocks; similar for 16–64.
- **Note:** L40S uses PCIe 4.0 — same as A100 PCIe, so swap costs should be comparable.

## E9 — Extension experiment

See [EXTENSIONS.md](EXTENSIONS.md).

---

## GPU-time budget (approx.)

| Experiment | Model | Hours |
| --- | --- | --- |
| E1–E2 dry run | OPT-1.3B | 1.5 |
| E1–E2 quick pass | OPT-13B | 11 |
| E1–E2 final pass (1-h traces) | OPT-13B | 56–70 |
| E3 | OPT-13B | 3 |
| E4 | OPT-13B | 2 |
| E6 | — | 0.5 |
| E7–E8 | OPT-1.3B | 3 |
| E9 | 8B | 3 |
| **Total** | | **≈85–95 GPU-hours** (≈2–3 days wall-clock on 2 GPUs) — unattended `sbatch` job arrays (see CLUSTER.md) |

If the queue is busy near the deadline: skip the final 1-hour pass and present the 10-minute results (say so on the slide); if still short, use 5 rates instead of 7, or run E1 on OPT-6.7B with a 6 GB KV cap.

## Results table (fill in)

| Exp | Paper result | Our result | Matches? | Notes |
| --- | --- | --- | --- | --- |
| E0 | 20.4–38.2% vs ≈96% | Simulator, synthetic ShareGPT, OPT-13B 12 GB: Max 19.9%, Pow2 29.6%, Oracle 38.2%, paged 97.0% | Yes | 26 Oct; `results/e0_sharegpt.png`. Rerun with the real ShareGPT file |
| E1 ShareGPT | 1.7–2.7× vs Oracle | | | |
| E1 Alpaca | | | | |
| E2 | 2.2× vs Oracle | Simulator (memory-bound, offline): 2.37× vs Oracle, 4.43× vs Max | Close | Simulator only; GPU run still needed |
| E3 parallel | 6.1–9.8% saved | | | |
| E3 beam | 37.6–55.2% saved | | | |
| E4 | 1.67× / 3.58× | | | |
| E6 | +20–26% kernel time | | | |
| E7 | best 16–128 | | | |
| E8 | crossover ~16–64 | | | |
| E9 | — (new) | | | |
