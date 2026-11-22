# Results

One folder per paper figure. All runs: 1× NVIDIA L40S 48 GB, KV pool capped at 12 GB (unless stated), 10-minute traces (unless stated).
Per-figure setup and the full results table are in [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).

## Paper figures reproduced

7 clean matches (E0 (Fig. 2), E1 (Fig. 12) ShareGPT, E1 (Fig. 12) Alpaca, E2 (Fig. 13), E6 (Fig. 18a), E7 (Fig. 18b), E8b (Fig. 19b)) and 1 partial (E3 (Fig. 14)).

| Paper figure | Folder | Ours | Paper | Verdict |
| --- | --- | --- | --- | --- |
| **E0 (Fig. 2)** memory waste | [`fig02_memory_waste/`](fig02_memory_waste) | real tokens: paged 97.0%, Orca 19.9–38.2% | ~96% vs 20.4–38.2% | ✅ Match |
| **E1 (Fig. 12)** latency vs rate, OPT-13B ShareGPT | [`fig12_latency_vs_rate/`](fig12_latency_vs_rate) | 1.75× Oracle, 3.11× Max | 1.7–2.7×, 2.7–8× | ✅ Match (no FasterTransformer line: paged vs Orca Oracle/Pow2/Max only) |
| **E1 (Fig. 12)** latency vs rate, OPT-13B Alpaca | [`fig12_latency_vs_rate/`](fig12_latency_vs_rate) | 1.50× Oracle, 10.9× Max | same trend | ✅ Match (smaller gain vs Oracle: short sequences; no FasterTransformer line) |
| **E2 (Fig. 13)** batch size | [`fig13_batch_size/`](fig13_batch_size) | 2.43× Oracle, 5.04× Max (ShareGPT) | 2.2×, 4.3× | ✅ Match |
| **E6 (Fig. 18a)** kernel overhead | [`fig18a_kernel/`](fig18a_kernel) | CUDA-graph timing, PyTorch SDPA (flash) baseline, context 64/128/256: batch 8 **+0% / +10% / +17%**, batch 32 **−8% / −1% / +8%**. Side by side with the paper: [`e6_vs_fig18a.png`](fig18a_kernel/e6_vs_fig18a.png) | +20–26% vs FasterTransformer (batch 8 and 32) | ✅ Match (overhead at or below the paper's). The earlier +29–108% at batch 8 was host-side launch overhead (int64→int32 casts, output alloc, Triton Python launcher) dominating 12–40 µs kernels. Caveat: configs up to ~84 MB of KV fit in the L40S's 96 MB L2 (bs 8 ctx ≤ 512, bs 32 ctx ≤ 128), which affects both sides equally; `num_warps=1` looked 2× faster in L2-resident configs but about equal once KV exceeds L2, so the default is kept. Baseline is SDPA, not FT |
| **E7 (Fig. 18b)** block size | [`fig18b_block_size/`](fig18b_block_size) | memory-bound sweep, OPT-1.3B (ShareGPT 1.5 GB KV, Alpaca 0.75 GB KV), s/token at block 1/2/4/8/16/32/64/128/256 — ShareGPT 8 req/s: 3.06* / 1.88 / 0.68 / 0.21 / 0.035 / 0.017 / **0.015** / 0.079 / 0.39; Alpaca 48 req/s: 0.317 / 0.0117 / 0.0093 / 0.0085 / 0.0082 / **0.0081** / **0.0081** / 0.0935 / 1.91. [`e7_block_size_mem.png`](fig18b_block_size/e7_block_size_mem.png) | best 16–128 | ✅ Match: U-shaped, best 32–64 (ShareGPT) and 16–64 (Alpaca). Large blocks lose to internal fragmentation → preemptions (Alpaca block 256: 12.7k preemptions); small blocks to per-block kernel/bookkeeping overhead (much steeper than the paper's at block 1–2: implementation overhead). *Block 1 ShareGPT hit the wall limit (970/1200 finished), lower bound |
| **E8b (Fig. 19b)** swap vs recompute | [`fig19b_swap_vs_recompute/`](fig19b_swap_vs_recompute) | recompute wins at block 4–64, swap at 256 | recompute small, swap large | ✅ Match |
| **E3 (Fig. 14)** parallel sampling | [`fig14_parallel_sampling/`](fig14_parallel_sampling) | OPT-13B Alpaca, 3-min traces, 12 GB KV, n = 2/4/6. KV memory saved by sharing (paged, avg): **4.6–5.9%** (n=2), **7.9–10.1%** (n=4), **9.2–10.9%** (n=6) ([`e3_mem_saving.png`](fig14_parallel_sampling/e3_mem_saving.png)). Saturated throughput paged vs Orca Oracle: n=2 5.38 vs 4.65 (**1.16×**), n=4 2.64 vs 2.22 (**1.19×**), n=6 1.76 vs 1.51 (**1.17×**); Orca Max 1.14 / 0.42 / 0.41 req/s (4.7× / 6.3× / 4.3× lower) ([`e3_parallel.png`](fig14_parallel_sampling/e3_parallel.png)) | saving 6.1–9.8% (Fig. 15a); paged clearly ahead of Oracle (Fig. 14a) | ☑️ Partial: memory saving matches; paged beats both Orca variants, but the gap to Oracle is smaller than the paper's. Every system is throttled far below E1 Alpaca (n=1 peak 35.7 req/s) by a per-sequence sampler bottleneck (each sequence's sampling copies the full-vocab probabilities GPU→CPU and syncs). Even the lowest rate is past saturation, so the curves have no low-latency region. Fix pending → rerun |

Not reproduced: Fig. 14b / Fig. 15 beam search (optional, paged only, not run), Fig. 16 shared prefix, Fig. 17 chatbot, Fig. 19a microbenchmark (cut); OPT-66B/175B panels of Fig. 12 (do not fit on one GPU). Fig. 14 (parallel sampling) is now partially reproduced (see above).

## Where each file comes from

| File | How it was made | Jobs (logs in `logs/<folder>/`, local only) | Commit |
| --- | --- | --- | --- |
| `fig02_memory_waste/e0_sharegpt.*` | `python benchmarks/simulate_memory.py --trace sharegpt --csv … --plot …` (no GPU) | — | — |
| `fig02_memory_waste/e0_alpaca.*` | same, `--trace alpaca --num-requests 2000` | — | — |
| `fig12_latency_vs_rate/e1_opt13b_sharegpt.csv` | rebuilt from logs | 6919_0–3, + paged 4.0/5.0 req/s plateau 7693 (`logs/fig12_opt13b_sharegpt/`) | 88e0a17, c151a7c |
| `fig12_latency_vs_rate/e1_opt13b_alpaca.csv` | rebuilt from logs | 7692 array (4 systems, 2–20 req/s), 7732 array (paged/Oracle 24–40) (`logs/fig12_opt13b_alpaca/`) | c151a7c |
| `fig12_latency_vs_rate/opt1.3b/e1_opt1.3b_*.csv` | rebuilt from logs (pilot) | 6902_0–3, 6903_0–3 (`logs/fig12_opt1.3b_pilot/`) | 88e0a17 |
| `fig13_batch_size/e2_*.png` | summary charts from the E1 CSVs | (same as Fig. 12) | — |
| `fig14_parallel_sampling/alpaca_<system>_n<N>_bs16.csv` | real cluster CSVs (`run_serving.py --n 2\|4\|6`, systems paged / orca-oracle / orca-max), copied from `~/pa_data/results/e3_parallel/` | 7759_0–8 (`logs/fig14_parallel_sampling/`) | 7d2ed30 |
| `fig14_parallel_sampling/e3_parallel.png`, `e3_mem_saving.png` | `python "Claude outputs/report_src/plot_e3.py" .` from those CSVs (hollow marker = not all requests finished) | 7759 | 7d2ed30 |
| `fig18a_kernel/e6_kernel_graph.csv`, `e6_kernel_eager.csv` | real cluster CSVs from `kernel_bench.py` (CUDA-graph and eager timing; SDPA flash/efficient/auto; Triton num_warps default/1/2/8) | 7750 (`logs/fig18a_kernel/`) | 7d2ed30 |
| `fig18a_kernel/e6_kernel.csv` | rebuilt from log (older eager-timing run, superseded by `e6_kernel_graph.csv`) | 6894 (`logs/fig18a_kernel/`) | b163d12 |
| `fig18a_kernel/e6_vs_fig18a.png` | side-by-side chart: left panel is the paper's Fig. 18a (screenshot, Kwon et al., SOSP 2023), right panel is ours from `e6_kernel_graph.csv` (CUDA-graph timing, flash SDPA, default num_warps, context 64–256); `python "Claude outputs/report_src/plot_e6_vs_fig18a.py" .` | 7750 | 7d2ed30 |
| `fig18a_kernel/e6_vs_fig18a_old_eager.png` | the earlier version of the side-by-side chart, from `e6_kernel.csv` (eager timing, +29–108% at batch 8); kept for comparison | 6894 | b163d12 |
| `fig18b_block_size/e7_e8b_opt1.3b_kv12_sharegpt.csv` | rebuilt from logs; block 1–256 × recompute/swap, 12 GB KV (also the first E8b sweep) | 7694–7711 (`logs/fig18b_19b_kv12/`) | c151a7c |
| `fig18b_block_size/e7_mem_{sharegpt,alpaca}.csv`, `e7_block_size_mem.png` | rebuilt from logs by `python "Claude outputs/report_src/plot_e7_e8b.py" .`; OPT-1.3B, block 1–256, recompute; ShareGPT 1.5 GB KV at 6/8 req/s, Alpaca 0.75 GB KV at 32/48 req/s; 150 s traces. Checked against the cluster CSVs in `_cluster_csv/e7_mem/` (identical) | 7751_0–1, 7755_2–17 (`logs/fig18b_e7_mem/`) | 7d2ed30 |
| `fig19b_swap_vs_recompute/e8b_opt1.3b_kv1.5_sharegpt.csv` | rebuilt from logs; block 4/16/64/256 × recompute/swap, 1.5 GB KV | 7724–7731 (`logs/fig19b_kv1.5/`) | c151a7c |

Smoke tests, dry runs and the cancelled duplicate 13B sweep (6893, 6895, 6896, 6898, 6915) and the E3 smoke run on OPT-125m (7758) are kept in `logs/_smoke_failed_superseded/` and not used. The GPU test-suite run 7749 (13 passed, commit 7d2ed30) is in `logs/_gpu_tests/`.

### Real cluster CSVs (`_cluster_csv/`)

[`_cluster_csv/`](_cluster_csv) is a copy of the cluster's `~/pa_data/results/` as of 22 Nov (same folder structure: `e1_pilot/`, `e1_dry/`, `e1_13b/`, `e1_13b_hi/`, `e3_parallel/`, `e3_smoke/`, `e7_e8b/`, `e7_mem/`, `e8b_kv1.5/`, `fig18a_kernel/`, `smoke/`, `e6_kernel.csv`). These are the CSVs `run_serving.py` / `kernel_bench.py` wrote, with the columns the logs lack (p50/p90 normalized latency, mean latency, TTFT, token throughput, `num_gpu_blocks`, memory-saving counters). The per-figure CSVs marked "rebuilt from logs" in the table above were made before these were available; where both exist, prefer `_cluster_csv/`.

### About the "rebuilt from logs" CSVs

The original CSVs are on the cluster under `~/pa_data/results/` (`e1_pilot/`, `e1_13b/`, `e1_13b_hi/`, `e7_e8b/`, `e8b_kv1.5/`, `e6_kernel.csv`).
The CSVs here were rebuilt from the per-rate lines `run_serving.py` / `kernel_bench.py` print into the SLURM logs:

- Columns match `run_serving.py` where the log has the value; p50/p90 latency, TTFT and token throughput are not in the logs.
- `kv_cache_gb`, `job`, `git_commit` and `source` columns say how each row was run and which log it came from.
- `e1_*.png` made with `benchmarks/plot.py --ymax 1.0`; `e2_*`, `e6_kernel`, `e7_block_size` and `e8b_swap_vs_recompute` are summary charts (generators kept outside the repo).
- The cluster CSVs are now in `_cluster_csv/` (22 Nov); the rebuilt files are kept so the existing charts stay reproducible. Plot scripts under `Claude outputs/report_src/` are local only (not in the repo).

New SLURM logs still land flat in `logs/` on the cluster; sort them into a `logs/<figure>/` folder when copying them over.
