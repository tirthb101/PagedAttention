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
- **Waste categories (paper §3 / Fig. 2):** *reserved* = slots for tokens not generated yet; *internal fragmentation* = slots the policy reserved that the request never uses (≈0 for Oracle, which knows the true length); *external fragmentation & others* = buddy rounding up to a power-of-two range + free blocks no request can use. Changed 22 Nov: buddy rounding was previously counted as internal; real-token shares are unchanged.
- **Result (22 Nov, ShareGPT):** tokens / reserved / internal / external & others = paged 97.0 / 0 / 1.4 / 1.6 %, Oracle 38.2 / 25.5 / 0.6 / 35.6 %, Pow2 29.6 / 19.7 / 15.1 / 35.6 %, Max 19.9 / 13.3 / 57.7 / 9.2 %. Alpaca (2000 requests): Oracle 37.4 / 26.9 / 4.4 / 31.4 %.
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
- **Metrics:** normalized latency vs rate; **memory saving** = 1 − (blocks with sharing / blocks without sharing), from block-manager counters: `BlockSpaceManager.sharing_stats()` per model step (logical = sum of GPU block-table lengths, physical = distinct GPU blocks), averaged over steps by `LLMEngine.stats()`; `run_serving.py` writes `mem_saving_avg_pct`, `mem_saving_max_pct` (and block-weighted `mem_saving_blocks_pct`). Orca rows are 0 by construction.
- **Status (22 Nov): code ready, not run.** Parallel: `sbatch slurm/e3_parallel.sbatch` (9 array tasks: n = 2/4/6 × paged / orca-oracle / orca-max; OPT-13B Alpaca; 3-min traces, 3 rates per n; ~3 GPU-h). Beam (optional): `sbatch slurm/e3_beam.sbatch` (paged, width 2/4/6, ~1–1.5 GPU-h). Parallel samples use temperature 1.0 with one RNG per sample (`--temperature` overrides).
- **Paper:** gain over Oracle 1.3× (n=1) → 2.3× (beam 6); saving 6.1–9.8% parallel, 37.6–55.2% beam (Alpaca); 16.2–30.5% / 44.3–66.3% (ShareGPT).
- **Beam search** is implemented (20 Nov) for the paged manager only; the Orca baselines would need a full private copy per beam on every fork, which is not built. So the beam half can show paged memory saving (Fig. 15) but not the paged-vs-Orca latency curve (Fig. 14b). Runs are still optional (see PLAN cut order).

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
| E3 parallel (reinstated 22 Nov) + optional beam | OPT-13B | 3 + 1.5 |
| E4 | OPT-13B | 2 |
| E6 | — | 0.5 |
| E7–E8 | OPT-1.3B | 3 |
| E9 | 8B | 3 |
| **Total** | | **≈85–95 GPU-hours** (≈2–3 days wall-clock on 2 GPUs) — unattended `sbatch` job arrays (see CLUSTER.md) |

If the queue is busy near the deadline: skip the final 1-hour pass and present the 10-minute results (say so on the slide); if still short, use 5 rates instead of 7, or run E1 on OPT-6.7B with a 6 GB KV cap.

## Results table (fill in)

| Exp | Paper result | Our result | Matches? | Notes |
| --- | --- | --- | --- | --- |
| E0 (Fig. 2) | 20.4–38.2% vs ≈96% | Simulator, synthetic ShareGPT, OPT-13B 12 GB: Max 19.9%, Pow2 29.6%, Oracle 38.2%, paged 97.0% | Yes | 26 Oct; `results/fig02_memory_waste/e0_sharegpt.png`. Rerun with the real ShareGPT file |
| E1 ShareGPT (Fig. 12) | 1.7–2.7× vs Oracle, 2.7–8× vs Max | OPT-13B, 10-min trace, L40S: saturated throughput paged **2.52 req/s** (plateau confirmed: 2.51 at 4.0 and 2.52 at 5.0 req/s) vs Oracle 1.44 (**1.75×**), Pow2 1.12 (2.25×), Max 0.81 (**3.11×**). At 2 req/s: paged 0.055 s/tok vs Oracle 3.06. OPT-1.3B: 1.57× Oracle, 2.47× Max | Yes (low end of range) | 27 Oct jobs 6919_0–3 / 6902_0–3 (88e0a17); plateau job 7693, 20 Nov (c151a7c). `results/fig12_latency_vs_rate/e1_opt13b_sharegpt.png`. Orca-Max 13B at 3 req/s timed out (1552/1800). Quick pass only (10 min, not 1 h). No FasterTransformer line (paged vs Orca Oracle/Pow2/Max only) |
| E1 Alpaca (Fig. 12) | similar trend | **OPT-13B**, 10-min trace, rates 2–40 req/s: saturated throughput paged **35.7 req/s** vs Oracle 23.7 (**1.50×**), Pow2 15.3 (2.34×), Max 3.26 (**10.9×**). Capacity at ≤ 0.1 s/tok: paged 36, Oracle 20 (1.8×), Pow2 12 (3×), Max 2 (18×). OPT-1.3B: only Max saturates by 64 req/s (paged ≥ 2.97× Max) | Yes (gain vs Oracle smaller than ShareGPT, as expected for short sequences) | 20–21 Nov, jobs 7692 array + 7732 array (paged/Oracle 24–40), commit c151a7c. `results/fig12_latency_vs_rate/e1_opt13b_alpaca.png`. Orca-Max at ≥ 12 req/s hit the wall-time cap (~6240 finished). No FasterTransformer line (paged vs Orca Oracle/Pow2/Max only) |
| E2 (Fig. 13) | 2.2× vs Oracle, 4.3× vs Max | **GPU, OPT-13B ShareGPT** (peak mean running requests): paged 35.3 vs Oracle 14.5 (**2.43×**), Pow2 10.4 (3.39×), Max 7.0 (**5.04×**). **Alpaca 13B:** paged 175.7 vs Oracle 73.8 (2.38×), Pow2 41.2 (4.26×), Max 7.0 (25×). Simulator: 2.37× / 4.43× | Yes | `results/fig13_batch_size/e2_opt13b_{sharegpt,alpaca}_batch_throughput.png` |
| E3 parallel (Fig. 14) | 6.1–9.8% saved | — | Not run | Cut 21 Nov, reinstated 22 Nov: **code ready, not run** (`slurm/e3_parallel.sbatch`; memory-saving counter now in the CSV). Parallel sampling is tested on CPU (sharing / COW / free for n = 2, 4, 6; swap, recompute and fallback preemption; Orca private reservations) |
| E3 beam (Fig. 15) | 37.6–55.2% saved | — | Not run | Code ready, not run (`slurm/e3_beam.sbatch`, paged only) |
| E4 (Fig. 16) | 1.67× / 3.58× | | | |
| E6 (Fig. 18a) | +20–26% kernel time (vs FasterTransformer, batch 8 and 32) | Triton paged vs contiguous SDPA, OPT-13B dims, at the paper's contexts 64/128/256: batch 32 **+4 / +3 / +9%**; batch 8 **+108 / +37 / +29%** (whole kernel only 12–40 µs, fixed launch overhead dominates). Beyond the paper's range: batch 32 +9–10% at ctx 512–1024; batch 8 +28% at 512, −8% at 1024. PyTorch gather reference is 2.6–3.3× slower than Triton | Partial: matches at batch 32 (smaller overhead), not at batch 8 | 26 Oct, job 6894, commit b163d12. Baseline is PyTorch SDPA, not FasterTransformer. `results/fig18a_kernel/e6_kernel.png`; side by side with the paper's figure: `results/fig18a_kernel/e6_vs_fig18a.png` |
| E7 (Fig. 18b) | best 16–128 | OPT-1.3B ShareGPT, 12 req/s, 12 GB KV: normalized latency bs1 **7.0**, bs2 0.115, bs4 0.026, bs8 0.017, bs16 0.014, bs32 0.012, bs64–256 0.011–0.012 s/tok. Throughput 6.1 req/s at bs1, ≥ 11.8 from bs16 up | Mostly — small blocks clearly bad; flat from 32; no large-block penalty seen on ShareGPT | 20 Nov, jobs 7694–7711, c151a7c. `results/fig18b_block_size/e7_block_size.png`. Alpaca block sweep not run (paper's large-block penalty shows on Alpaca) |
| E8b (Fig. 19b) | recompute better at small blocks, swap at large | OPT-1.3B ShareGPT, 12 req/s, **1.5 GB KV** (forces 4–36k preemptions): recompute vs swap s/tok — bs4 8.84 vs 9.12, bs16 3.75 vs 3.97, bs64 **2.75** vs 2.99, bs256 7.53 vs **5.07**. Recompute wins by 3–8% up to bs64; swap wins by 33% at bs256 | Yes (qualitative crossover between 64 and 256) | 21 Nov, jobs 7724–7731, c151a7c. `results/fig19b_swap_vs_recompute/e8b_swap_vs_recompute.png`. At 12 GB (jobs 7694–7711) only bs1/bs2 preempt (recompute 0.115 vs swap 0.186 at bs2). Open: recompute at bs256 preempts 35.6k times vs 6.0k for swap (likely re-admission thrashing with only ~32 blocks). E8a microbenchmark cut |
| E9 (new) | — (new) | | | |
