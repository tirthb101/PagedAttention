"""The GPU-free simulator: every system must finish the trace, and paging must fit more requests."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.simulate_memory import simulate  # noqa: E402
from benchmarks.trace import synthetic_trace  # noqa: E402
from minivllm.config import SchedulerConfig  # noqa: E402


def test_trace_statistics_and_filtering():
    reqs = synthetic_trace("sharegpt", 500, rate=2.0, seed=1)
    assert all(4 <= r.prompt_len < 1024 and r.prompt_len + r.output_len <= 2048 for r in reqs)
    times = [r.arrival_time for r in reqs]
    assert times == sorted(times) and times[0] > 0
    mean_rate = len(reqs) / times[-1]
    assert 1.6 < mean_rate < 2.4
    alpaca = synthetic_trace("alpaca", 500, seed=1)
    assert sum(r.prompt_len for r in reqs) / 500 > 5 * sum(r.prompt_len for r in alpaca) / 500


def test_paged_beats_every_orca_variant_and_all_finish():
    reqs = synthetic_trace("sharegpt", 120, rate=math.inf, seed=0)
    cfg = SchedulerConfig(max_num_seqs=256, max_num_batched_tokens=2048, max_model_len=2048)
    results = {s: simulate(s, reqs, 16, 256, 256, cfg, check=True)
               for s in ("paged", "orca-oracle", "orca-pow2", "orca-max")}
    total_out = sum(r.output_len for r in reqs)
    for r in results.values():
        assert r.total_output_tokens >= total_out
    assert results["paged"].mean_batch > results["orca-oracle"].mean_batch > results["orca-max"].mean_batch
    assert results["paged"].frac["token_states"] > 0.9
    assert results["orca-oracle"].frac["token_states"] < 0.6
