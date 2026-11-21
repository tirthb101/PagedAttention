# Results rebuilt from cluster logs

These CSVs were **rebuilt from the SLURM logs in `logs/`** (the per-rate lines `run_serving.py` / `kernel_bench.py` print),
because the original CSVs are on the cluster under `~/pa_data/results/` (`e1_pilot/`, `e1_13b/`, `e1_13b_hi/`, `e7_e8b/`, `e8b_kv1.5/`, `e6_kernel.csv`).

- Columns match `run_serving.py` where the log has the value; p50/p90 latency, TTFT and token throughput are not in the logs.
- `kv_cache_gb`, `job`, `git_commit` and `source` columns say how each row was run and which log it came from.
- Plots: `e1_*.png` made with `benchmarks/plot.py --ymax 1.0`; `e2_*`, `e6_kernel`, `e7_block_size` and `e8b_swap_vs_recompute` are summary charts
  (generators kept outside the repo in `Claude outputs/report_src/`).
- Once the cluster CSVs are copied over, prefer them and delete this folder.

| File | Jobs | Commit |
| --- | --- | --- |
| e1_opt1.3b_sharegpt.csv | 6902_0–3 | 88e0a17 |
| e1_opt1.3b_alpaca.csv | 6903_0–3 | 88e0a17 |
| e1_opt13b_sharegpt.csv | 6919_0–3, + paged 4.0/5.0 req/s plateau (7693) | 88e0a17, c151a7c |
| e1_opt13b_alpaca.csv | 7692 array (4 systems, 2–20 req/s), + paged/Oracle 24–40 req/s (7732 array) | c151a7c |
| e7_e8b_opt1.3b_kv12_sharegpt.csv | 7694–7711 (block 1–256 × recompute/swap, 12 GB KV) | c151a7c |
| e8b_opt1.3b_kv1.5_sharegpt.csv | 7724–7731 (block 4/16/64/256 × recompute/swap, 1.5 GB KV) | c151a7c |
| e6_kernel.csv | 6894 | b163d12 |
