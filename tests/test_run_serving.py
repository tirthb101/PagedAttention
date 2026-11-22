"""run_serving.py end to end on a tiny random model (seconds, CPU ok): every request finishes
and the CSV row has sane metrics for paged and an Orca baseline."""

import argparse
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.run_serving import load_model, run_one_rate  # noqa: E402


@pytest.mark.parametrize("manager,beam", [("paged", 0), ("orca-oracle", 0), ("paged", 3)])
def test_run_serving_smoke(manager, beam):
    args = argparse.Namespace(
        tiny=True, model="tiny", trace="synthetic-alpaca", duration=2.0, num_requests=12, seed=0,
        max_model_len=2048, max_num_seqs=64, max_num_batched_tokens=2048, preemption="recompute",
        block_size=16, kv_cache_gb=0.01, swap_space_gb=0.01, num_gpu_blocks=256, memory_manager=manager,
        attention_backend="ref", n=1, beam=beam, max_wall_factor=10.0,
    )
    model, tok = load_model(args)
    row = run_one_rate(model, tok, args, rate=8.0)
    assert row["finished"] == row["num_requests"] == 12 and row["timed_out"] == 0
    assert row["norm_latency_mean"] > 0 and not math.isnan(row["norm_latency_p90"])
    assert row["throughput_tok_s"] > 0 and row["mean_batch"] >= 1


@pytest.mark.parametrize("extra", [
    dict(sys_prompt_len=64, prefix_caching=True),
    dict(sys_prompt_len=64, register_prefix=True),
    dict(sys_prompt_len=64, num_sys_prompts=3),
    dict(chat_turns=3, think_time=0.05, prefix_caching=True, trace="synthetic-sharegpt"),
])
def test_run_serving_e9_workloads(extra):
    """E9 (X1): shared system prompt (auto cache / paper's predefined prefix / none) and
    closed-loop multi-turn chat."""
    base = dict(
        tiny=True, model="tiny", trace="synthetic-alpaca", duration=2.0, num_requests=8, seed=0,
        max_model_len=2048, max_num_seqs=64, max_num_batched_tokens=2048, preemption="recompute",
        block_size=16, kv_cache_gb=0.01, swap_space_gb=0.01, num_gpu_blocks=512, memory_manager="paged",
        attention_backend="ref", n=1, beam=0, max_wall_factor=20.0, prefix_caching=False,
        register_prefix=False, sys_prompt_len=0, num_sys_prompts=1, chat_turns=0, think_time=0.0,
    )
    base.update(extra)
    args = argparse.Namespace(**base)
    model, tok = load_model(args)
    row = run_one_rate(model, tok, args, rate=8.0)
    assert row["finished"] == row["num_requests"] and row["timed_out"] == 0
    if args.prefix_caching:
        assert row["cache_hit_rate_pct"] > 0
    if args.chat_turns:
        assert row["workload"] == "chat" and row["num_requests"] > 8   # follow-up turns were sent
    else:
        assert row["prompt_tokens_mean"] > 64
