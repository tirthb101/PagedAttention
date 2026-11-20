# Results rebuilt from cluster logs

These CSVs were **rebuilt from the SLURM logs in `logs/`** (the per-rate lines `run_serving.py` / `kernel_bench.py` print),
because the original CSVs are on the cluster under `~/pa_data/results/` (`e1_pilot/`, `e1_13b/`, `e6_kernel.csv`).

- Columns match `run_serving.py` where the log has the value; p50/p90 latency, TTFT and token throughput are not in the logs.
- `job`, `git_commit` and `source` columns say which log each row came from.
- Plots: `e1_*.png` made with `benchmarks/plot.py --ymax 1.0`; `e2_*.png` and `e6_kernel.png` are summary charts.
- Once the cluster CSVs are copied over, prefer them and delete this folder.

| File | Jobs | Commit |
| --- | --- | --- |
| e1_opt1.3b_sharegpt.csv | 6902_0–3 | 88e0a17 |
| e1_opt1.3b_alpaca.csv | 6903_0–3 | 88e0a17 |
| e1_opt13b_sharegpt.csv | 6919_0–3 | 88e0a17 |
| e6_kernel.csv | 6894 | b163d12 |
