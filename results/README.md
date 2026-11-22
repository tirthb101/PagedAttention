# Results

One folder per paper figure. All runs: 1× NVIDIA L40S 48 GB, KV pool capped at 12 GB (unless stated), 10-minute traces.
Per-figure setup and the full results table are in [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).

## Paper figures reproduced

5 clean matches (Fig. 2, Fig. 12 ShareGPT, Fig. 12 Alpaca, Fig. 13, Fig. 19b) and 2 partial (Fig. 18a, Fig. 18b).

| Paper figure | Folder | Ours | Paper | Verdict |
| --- | --- | --- | --- | --- |
| **Fig. 2** memory waste | [`fig02_memory_waste/`](fig02_memory_waste) | real tokens: paged 97.0%, Orca 19.9–38.2% | ~96% vs 20.4–38.2% | ✅ Match |
| **Fig. 12** latency vs rate, OPT-13B ShareGPT | [`fig12_latency_vs_rate/`](fig12_latency_vs_rate) | 1.75× Oracle, 3.11× Max | 1.7–2.7×, 2.7–8× | ✅ Match (no FasterTransformer line: paged vs Orca Oracle/Pow2/Max only) |
| **Fig. 12** latency vs rate, OPT-13B Alpaca | [`fig12_latency_vs_rate/`](fig12_latency_vs_rate) | 1.50× Oracle, 10.9× Max | same trend | ✅ Match (smaller gain vs Oracle: short sequences; no FasterTransformer line) |
| **Fig. 13** batch size | [`fig13_batch_size/`](fig13_batch_size) | 2.43× Oracle, 5.04× Max (ShareGPT) | 2.2×, 4.3× | ✅ Match |
| **Fig. 19b** swap vs recompute | [`fig19b_swap_vs_recompute/`](fig19b_swap_vs_recompute) | recompute wins at block 4–64, swap at 256 | recompute small, swap large | ✅ Match |
| Fig. 18a kernel overhead | [`fig18a_kernel/`](fig18a_kernel) | context 64–256: +3–9% at batch 32, +29–108% at batch 8 (batch-8 kernel takes only 12–40 µs, fixed launch overhead dominates); baseline is PyTorch SDPA, not FT. Side by side with the paper: [`e6_vs_fig18a.png`](fig18a_kernel/e6_vs_fig18a.png) | +20–26% vs FasterTransformer (batch 8 and 32) | ☑️ Partial: matches at batch 32 (smaller overhead), not at batch 8 |
| Fig. 18b block size | [`fig18b_block_size/`](fig18b_block_size) | blocks 1–2 very slow, flat from 32 | best 16–128 | ☑️ Mostly; ShareGPT only (no Alpaca sweep) |

Not reproduced: Fig. 14 parallel sampling (cut 21 Nov), Fig. 15 beam memory saving (optional), Fig. 16 shared prefix, Fig. 17 chatbot, Fig. 19a microbenchmark (cut); OPT-66B/175B panels of Fig. 12 (do not fit on one GPU).

## Where each file comes from

| File | How it was made | Jobs (logs in `logs/<folder>/`, local only) | Commit |
| --- | --- | --- | --- |
| `fig02_memory_waste/e0_sharegpt.*` | `python benchmarks/simulate_memory.py --trace sharegpt --csv … --plot …` (no GPU) | — | — |
| `fig02_memory_waste/e0_alpaca.*` | same, `--trace alpaca --num-requests 2000` | — | — |
| `fig12_latency_vs_rate/e1_opt13b_sharegpt.csv` | rebuilt from logs | 6919_0–3, + paged 4.0/5.0 req/s plateau 7693 (`logs/fig12_opt13b_sharegpt/`) | 88e0a17, c151a7c |
| `fig12_latency_vs_rate/e1_opt13b_alpaca.csv` | rebuilt from logs | 7692 array (4 systems, 2–20 req/s), 7732 array (paged/Oracle 24–40) (`logs/fig12_opt13b_alpaca/`) | c151a7c |
| `fig12_latency_vs_rate/opt1.3b/e1_opt1.3b_*.csv` | rebuilt from logs (pilot) | 6902_0–3, 6903_0–3 (`logs/fig12_opt1.3b_pilot/`) | 88e0a17 |
| `fig13_batch_size/e2_*.png` | summary charts from the E1 CSVs | (same as Fig. 12) | — |
| `fig18a_kernel/e6_kernel.csv` | rebuilt from log | 6894 (`logs/fig18a_kernel/`) | b163d12 |
| `fig18a_kernel/e6_vs_fig18a.png` | side-by-side chart: left panel is the paper's Fig. 18a (screenshot, Kwon et al., SOSP 2023), right panel is ours from `e6_kernel.csv` (context 64–256, same axes and encoding) | 6894 | b163d12 |
| `fig18b_block_size/e7_e8b_opt1.3b_kv12_sharegpt.csv` | rebuilt from logs; block 1–256 × recompute/swap, 12 GB KV (also the first E8b sweep) | 7694–7711 (`logs/fig18b_19b_kv12/`) | c151a7c |
| `fig19b_swap_vs_recompute/e8b_opt1.3b_kv1.5_sharegpt.csv` | rebuilt from logs; block 4/16/64/256 × recompute/swap, 1.5 GB KV | 7724–7731 (`logs/fig19b_kv1.5/`) | c151a7c |

Smoke tests, dry runs and the cancelled duplicate 13B sweep (6893, 6895, 6896, 6898, 6915) are kept in `logs/_smoke_failed_superseded/` and not used.

### About the "rebuilt from logs" CSVs

The original CSVs are on the cluster under `~/pa_data/results/` (`e1_pilot/`, `e1_13b/`, `e1_13b_hi/`, `e7_e8b/`, `e8b_kv1.5/`, `e6_kernel.csv`).
The CSVs here were rebuilt from the per-rate lines `run_serving.py` / `kernel_bench.py` print into the SLURM logs:

- Columns match `run_serving.py` where the log has the value; p50/p90 latency, TTFT and token throughput are not in the logs.
- `kv_cache_gb`, `job`, `git_commit` and `source` columns say how each row was run and which log it came from.
- `e1_*.png` made with `benchmarks/plot.py --ymax 1.0`; `e2_*`, `e6_kernel`, `e7_block_size` and `e8b_swap_vs_recompute` are summary charts (generators kept outside the repo).
- Once the cluster CSVs are copied over, prefer them and replace these files in place.

New SLURM logs still land flat in `logs/` on the cluster; sort them into a `logs/<figure>/` folder when copying them over.
