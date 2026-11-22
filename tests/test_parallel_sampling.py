"""Parallel sampling (§4.4, Fig. 14): n samples of one prompt share the prompt's blocks.

Block-manager and scheduler tests run anywhere (no torch). The engine tests at the end
use the tiny float64 OPT from test_e2e_equivalence.py and are skipped without torch.
"""

import argparse
import math
import random

import pytest

from minivllm.block_manager import AllocStatus, BlockSpaceManager
from minivllm.config import SchedulerConfig
from minivllm.reservation_manager import ReservationBlockManager
from minivllm.scheduler import Scheduler
from minivllm.sequence import SequenceStatus

from conftest import make_group

B = 4  # block size in tokens


def bm_with(gpu=64, cpu=64):
    return BlockSpaceManager(B, gpu, cpu, watermark=0.0)


def admit(bm, group):
    bm.allocate(group)
    for s in group.get_seqs(SequenceStatus.WAITING):
        s.status = SequenceStatus.RUNNING
    return group.get_seqs(SequenceStatus.RUNNING)


def decode(bm, seqs, tokens):
    """Append one token to each sequence and reserve its slot; returns the COW copies."""
    copies = []
    for seq, tok in zip(seqs, tokens):
        seq.num_computed_tokens = seq.get_len()
        seq.append_token(tok)
        cow = bm.append_slot(seq)
        if cow is not None:
            copies.append(cow)
    return copies


# ------------------------------------------------------------------ block manager


@pytest.mark.parametrize("n", [2, 4, 6])
def test_prompt_blocks_are_shared_by_all_samples(n):
    bm = bm_with(gpu=64)
    seqs = admit(bm, make_group(prompt_len=10, block_size=B, n=n))  # 2 full blocks + 2 tokens
    tables = [bm.block_tables[s.seq_id] for s in seqs]
    assert all(len(t) == 3 for t in tables)
    for i in range(3):
        assert all(t[i] is tables[0][i] for t in tables), "prompt block not shared"
        assert tables[0][i].ref_count == n
    assert bm.get_num_free_gpu_blocks() == 64 - 3  # not 3 * n
    assert all(s.num_computed_tokens == 0 for s in seqs)
    bm.check_invariants(seqs)


@pytest.mark.parametrize("n", [2, 4, 6])
def test_copy_on_write_on_the_shared_partial_block(n):
    bm = bm_with(gpu=64)
    seqs = admit(bm, make_group(prompt_len=10, block_size=B, n=n))
    shared_last = bm.block_tables[seqs[0].seq_id][2]
    copies = decode(bm, seqs, range(n))  # every sample writes its own token 11 into block 2

    # The first n-1 writers copy the block; the last one is then its only owner and writes in place.
    assert len(copies) == n - 1
    assert all(src == shared_last.block_number for src, _ in copies)
    assert len({dst for _, dst in copies}) == n - 1
    lasts = [bm.block_tables[s.seq_id][2] for s in seqs]
    assert len({id(b) for b in lasts}) == n and all(b.ref_count == 1 for b in lasts)
    assert lasts[-1] is shared_last
    # Full prompt blocks stay shared.
    for i in range(2):
        assert all(bm.block_tables[s.seq_id][i].ref_count == n for s in seqs)
    assert bm.get_num_free_gpu_blocks() == 64 - (2 + n)

    # Later writes go to private blocks: no more copies.
    assert decode(bm, seqs, range(n)) == []
    bm.check_invariants(seqs)


@pytest.mark.parametrize("n", [2, 4, 6])
def test_full_prompt_needs_no_copy(n):
    bm = bm_with(gpu=64)
    seqs = admit(bm, make_group(prompt_len=8, block_size=B, n=n))  # exactly 2 full blocks
    assert decode(bm, seqs, range(n)) == []  # each sample gets a fresh private block
    assert bm.get_num_free_gpu_blocks() == 64 - (2 + n)
    bm.check_invariants(seqs)


@pytest.mark.parametrize("n", [2, 4, 6])
def test_freeing_samples_one_by_one(n):
    bm = bm_with(gpu=64)
    group = make_group(prompt_len=10, block_size=B, n=n)
    seqs = admit(bm, group)
    decode(bm, seqs, range(n))
    for k, seq in enumerate(seqs):
        seq.status = SequenceStatus.FINISHED_LENGTH
        bm.free(seq)
        bm.check_invariants(seqs)
        remaining = n - k - 1
        expected_used = (2 + remaining) if remaining else 0  # shared blocks go with the last sample
        assert bm.get_num_free_gpu_blocks() == 64 - expected_used
        assert group.is_finished() == (remaining == 0)
    assert not bm.block_tables


@pytest.mark.parametrize("n", [2, 4, 6])
def test_swap_round_trip_keeps_sharing(n):
    bm = bm_with(gpu=64, cpu=64)
    group = make_group(prompt_len=10, block_size=B, n=n)
    seqs = admit(bm, group)
    decode(bm, seqs, range(n))
    out_map = bm.swap_out(group)
    for s in seqs:
        s.status = SequenceStatus.SWAPPED
    assert len(out_map) == 2 + n  # shared blocks copied once
    bm.check_invariants(seqs)
    bm.swap_in(group)
    for s in seqs:
        s.status = SequenceStatus.RUNNING
    t0 = bm.block_tables[seqs[0].seq_id]
    assert all(bm.block_tables[s.seq_id][0] is t0[0] and t0[0].ref_count == n for s in seqs)
    bm.check_invariants(seqs)


def test_recompute_of_diverged_samples_reshares_common_full_blocks():
    """After recompute-preemption the samples differ: only full blocks of the common
    prefix are shared again; the first sample recomputes them, the others skip them."""
    bm = bm_with(gpu=64)
    group = make_group(prompt_len=6, block_size=B, n=3)
    a, b, c = admit(bm, group)
    decode(bm, [a, b, c], [1, 1, 1])
    decode(bm, [a, b, c], [2, 2, 7])  # c diverges at token 8 (index 7)
    decode(bm, [a, b, c], [3, 9, 3])  # b diverges at token 9 (index 8)
    for s in (a, b, c):  # preempt by recomputation
        bm.free(s)
        s.reset_for_recompute()
    assert bm.get_num_free_gpu_blocks() == 64

    # Common prefix = 7 tokens -> 1 full shared block; each sample (9 tokens, 3 blocks) has 2 private.
    assert bm.can_allocate(group) == AllocStatus.OK
    admit(bm, group)
    ta, tb, tc = (bm.block_tables[s.seq_id] for s in (a, b, c))
    assert ta[0] is tb[0] is tc[0] and ta[0].ref_count == 3
    assert len({id(t[i]) for t in (ta, tb, tc) for i in (1, 2)}) == 6
    assert bm.get_num_free_gpu_blocks() == 64 - 7
    assert (a.num_computed_tokens, b.num_computed_tokens, c.num_computed_tokens) == (0, 4, 4)
    bm.check_invariants([a, b, c])


def test_recompute_of_identical_samples_shares_everything():
    bm = bm_with(gpu=64)
    group = make_group(prompt_len=6, block_size=B, n=2)
    a, b = admit(bm, group)
    decode(bm, [a, b], [5, 5])
    for s in (a, b):
        bm.free(s)
        s.reset_for_recompute()
    admit(bm, group)
    assert bm.block_tables[a.seq_id] == bm.block_tables[b.seq_id]
    assert bm.get_num_free_gpu_blocks() == 64 - 2
    assert a.num_computed_tokens == b.num_computed_tokens == 0


# ---------------------------------------------------------------------- scheduler


def make_scheduler(gpu_blocks=16, cpu_blocks=16, mode="recompute", max_tokens=256, max_len=128, bm=None):
    bm = bm or BlockSpaceManager(B, gpu_blocks, cpu_blocks, watermark=0.0)
    cfg = SchedulerConfig(max_num_seqs=min(64, max_tokens), max_num_batched_tokens=max_tokens, max_model_len=max_len,
                          preemption_mode=mode)
    return Scheduler(cfg, bm)


def fill_then_decode(sched, groups):
    for g in groups:
        sched.add_seq_group(g)
    sched.schedule()
    for g in groups:
        for s in g.seqs:
            s.num_computed_tokens = s.get_len()
            s.append_token(3 + s.seq_id % 5)
    return sched.schedule()


def test_group_is_swapped_as_a_gang():
    sched = make_scheduler(gpu_blocks=6, cpu_blocks=16)
    old = make_group(8, B, arrival=0, output_len=50)
    par = make_group(8, B, n=3, arrival=1, output_len=50)
    out = fill_then_decode(sched, [old, par])  # needs 1 + 3 new blocks, only 2 free
    assert out.preempted == [par] and sched.swapped == [par]
    assert all(s.status == SequenceStatus.SWAPPED for s in par.seqs)
    assert len(out.blocks_to_swap_out) == 2  # the 2 shared prompt blocks, copied once
    sched.block_manager.check_invariants(old.seqs + par.seqs)


def test_group_falls_back_to_recompute_without_swap_space():
    sched = make_scheduler(gpu_blocks=6, cpu_blocks=0, mode="swap")
    old = make_group(8, B, arrival=0, output_len=50)
    par = make_group(8, B, n=3, arrival=1, output_len=50)
    out = fill_then_decode(sched, [old, par])
    assert out.preempted == [par] and not sched.swapped
    assert sched.waiting[0] is par and sched.num_swap_fallbacks == 1
    assert all(s.status == SequenceStatus.WAITING and s.num_computed_tokens == 0 for s in par.seqs)
    assert all(s.seq_id not in sched.block_manager.block_tables for s in par.seqs)
    sched.block_manager.check_invariants(old.seqs + par.seqs)


def test_request_finishes_only_when_all_samples_finish():
    sched = make_scheduler(gpu_blocks=32)
    g = make_group(5, B, n=3, output_len=4)
    sched.add_seq_group(g)
    sched.schedule()
    a, b, c = g.seqs
    for s in g.seqs:
        s.num_computed_tokens = s.get_len()
        s.append_token(1)
    a.status = SequenceStatus.FINISHED_STOPPED
    sched.free_seq(a)
    assert sched.free_finished_seq_groups() == [] and sched.running == [g]
    out = sched.schedule()
    assert out.decode_groups == [g] and g.num_seqs(SequenceStatus.RUNNING) == 2
    for s in (b, c):
        s.status = SequenceStatus.FINISHED_LENGTH
        sched.free_seq(s)
    assert sched.free_finished_seq_groups() == [g]
    assert sched.block_manager.get_num_free_gpu_blocks() == 32


@pytest.mark.parametrize("mode,cpu", [("recompute", 64), ("swap", 64), ("swap", 6)])
@pytest.mark.parametrize("seed", range(4))
def test_parallel_workload_finishes_without_leaks(mode, cpu, seed):
    """n in {1,2,4,6} through a tight pool, with swap, recompute and swap->recompute fallback."""
    rng = random.Random(seed)
    sched = make_scheduler(gpu_blocks=64, cpu_blocks=cpu, mode=mode, max_tokens=128, max_len=96)
    groups = [make_group(rng.randint(1, 30), B, n=rng.choice([1, 2, 4, 6]), arrival=float(i),
                         output_len=rng.randint(1, 30)) for i in range(25)]
    pending = list(groups)
    bm = sched.block_manager
    for step in range(5000):
        while pending and pending[0].arrival_time <= step:
            sched.add_seq_group(pending.pop(0))
        out = sched.schedule(now=step)
        for g in out.scheduled_groups:
            for s in g.get_seqs(SequenceStatus.RUNNING):
                assert len(bm.block_tables[s.seq_id]) * B >= s.get_len()
                s.num_computed_tokens = s.get_len()
                s.append_token(rng.randrange(50))  # samples diverge
                if s.get_output_len() >= g.true_output_len:
                    s.status = SequenceStatus.FINISHED_LENGTH
                    sched.free_seq(s)
        sched.free_finished_seq_groups()
        live = [s for g in list(sched.waiting) + sched.running + sched.swapped for s in g.seqs]
        bm.check_invariants(live)
        assert 0.0 <= bm.sharing_stats()["saving"] < 1.0
        if not pending and not sched.has_unfinished():
            break
    else:
        pytest.fail("workload did not finish")
    for g in groups:
        assert all(s.status == SequenceStatus.FINISHED_LENGTH and s.get_output_len() == g.true_output_len
                   for s in g.seqs)
    assert bm.get_num_free_gpu_blocks() == 64 and bm.get_num_free_cpu_blocks() == cpu
    assert sched.num_preemptions > 0
    if cpu == 6:
        assert sched.num_swap_fallbacks > 0


# ------------------------------------------------------------------ Orca baseline


@pytest.mark.parametrize("n", [2, 4, 6])
def test_orca_gives_each_sample_a_private_reservation(n):
    bm = ReservationBlockManager(B, 128, policy="orca-oracle", max_model_len=64)
    group = make_group(10, B, n=n, output_len=6)  # 16 tokens -> 4 blocks per sample
    assert bm.can_allocate(group) == AllocStatus.OK
    seqs = admit(bm, group)
    starts = {bm.reservations[s.seq_id][0] for s in seqs}
    assert len(starts) == n
    assert bm.get_num_free_gpu_blocks() == 128 - 4 * n
    st = bm.sharing_stats()
    assert st["logical_blocks"] == st["physical_blocks"] == 4 * n and st["saving"] == 0.0
    for s in seqs:
        bm.free(s)
    assert bm.get_num_free_gpu_blocks() == 128
    bm.check_invariants()


def test_orca_prefill_budget_counts_every_sample():
    """Orca computes the prompt once per sample, so admission must budget n x prompt tokens."""
    bm = ReservationBlockManager(B, 256, policy="orca-oracle", max_model_len=32)
    sched = make_scheduler(bm=bm, max_tokens=32, max_len=32)
    a = make_group(10, B, arrival=0, output_len=4)
    b = make_group(10, B, n=3, arrival=1, output_len=4)  # 30 prefill tokens
    sched.add_seq_group(a)
    sched.add_seq_group(b)
    out = sched.schedule()
    assert out.prefill_groups == [a] and out.num_batched_tokens == 10
    assert list(sched.waiting) == [b]

    # A group bigger than the whole budget is still admitted into an otherwise empty batch.
    sched = make_scheduler(bm=ReservationBlockManager(B, 256, policy="orca-oracle", max_model_len=32),
                           max_tokens=32, max_len=32)
    big = make_group(20, B, n=3, output_len=4)
    sched.add_seq_group(big)
    out = sched.schedule()
    assert out.prefill_groups == [big] and out.num_batched_tokens == 60


def test_paged_prefill_budget_counts_shared_prompt_once():
    sched = make_scheduler(gpu_blocks=64, max_tokens=32, max_len=32)
    a = make_group(10, B, arrival=0)
    b = make_group(10, B, n=3, arrival=1)
    sched.add_seq_group(a)
    sched.add_seq_group(b)
    out = sched.schedule()
    assert out.prefill_groups == [a, b] and out.num_batched_tokens == 20


# ------------------------------------------------------------------- engine (torch)

try:
    import torch
    import transformers
except ImportError:  # pragma: no cover
    torch = transformers = None
needs_torch = pytest.mark.skipif(torch is None or transformers is None, reason="needs torch + transformers")

VOCAB, MAX_TOKENS = 300, 12


@pytest.fixture(scope="module")
def tiny_opt():
    if torch is None or transformers is None:
        pytest.skip("needs torch + transformers")
    torch.manual_seed(0)
    cfg = transformers.OPTConfig(
        vocab_size=VOCAB, hidden_size=64, num_hidden_layers=2, ffn_dim=128, num_attention_heads=4,
        max_position_embeddings=256, word_embed_proj_dim=64, dropout=0.0, attention_dropout=0.0,
        init_std=0.3,
    )
    return transformers.OPTForCausalLM(cfg).double().eval()


@pytest.fixture(scope="module")
def prompts():
    rng = random.Random(0)
    return [[rng.randrange(3, VOCAB) for _ in range(n)] for n in (1, 5, 16, 17, 33, 50)]


def make_engine(model, gpu_blocks=512, cpu_blocks=512, mode="recompute", memory_manager="paged"):
    from minivllm.engine import LLMEngine
    sched = SchedulerConfig(max_num_seqs=64, max_num_batched_tokens=256, max_model_len=256, preemption_mode=mode)
    return LLMEngine(model, block_size=B, num_gpu_blocks=gpu_blocks, num_cpu_blocks=cpu_blocks,
                     scheduler_config=sched, watermark=0.0, check_invariants=True, memory_manager=memory_manager)


def sampled(n, seed=7):
    from minivllm.sequence import SamplingParams
    return SamplingParams(n=n, temperature=1.0, max_tokens=MAX_TOKENS, ignore_eos=True, seed=seed)


@needs_torch
@pytest.mark.parametrize("n", [2, 4, 6])
def test_engine_samples_are_independent_and_reproducible(tiny_opt, prompts, n):
    outs = make_engine(tiny_opt).generate(prompts, sampled(n))
    for o in outs:
        assert len(o.output_token_ids) == n and all(len(s) == MAX_TOKENS for s in o.output_token_ids)
    # Random draws: samples differ (if every request's n samples were identical, they would
    # all be drawing from one shared RNG state or sampling greedily).
    assert any(len({tuple(s) for s in o.output_token_ids}) > 1 for o in outs)
    again = make_engine(tiny_opt).generate(prompts, sampled(n))
    assert [o.output_token_ids for o in outs] == [o.output_token_ids for o in again]


@needs_torch
@pytest.mark.parametrize("mode,gpu,cpu", [("recompute", 40, 512), ("swap", 40, 512), ("swap", 40, 6)])
def test_engine_samples_unchanged_by_preemption_and_memory_manager(tiny_opt, prompts, mode, gpu, cpu):
    """Each sample has its own RNG, so its tokens may not depend on memory pressure (swap,
    recompute of diverged samples, swap->recompute fallback) or on the memory manager."""
    ref = make_engine(tiny_opt).generate(prompts, sampled(4))
    engine = make_engine(tiny_opt, gpu_blocks=gpu, cpu_blocks=cpu, mode=mode)
    outs = engine.generate(prompts, sampled(4))
    assert engine.scheduler.num_preemptions > 0, "test must actually preempt"
    if cpu == 6:
        assert engine.scheduler.num_swap_fallbacks > 0
    for o, r in zip(outs, ref):
        assert o.output_token_ids == r.output_token_ids
        assert o.extra["logprobs"] == pytest.approx(r.extra["logprobs"], abs=1e-9)
    assert engine.block_manager.get_num_free_gpu_blocks() == gpu

    orca = make_engine(tiny_opt, gpu_blocks=512, memory_manager="orca-oracle").generate(prompts, sampled(4))
    assert [o.output_token_ids for o in orca] == [o.output_token_ids for o in ref]


@needs_torch
@pytest.mark.parametrize("manager", ["paged", "orca-oracle", "orca-max"])
def test_run_serving_parallel_sampling(manager):
    from benchmarks.run_serving import load_model, run_one_rate
    args = argparse.Namespace(
        tiny=True, model="tiny", trace="synthetic-alpaca", duration=2.0, num_requests=8, seed=0,
        max_model_len=2048, max_num_seqs=64, max_num_batched_tokens=2048, preemption="recompute",
        block_size=16, kv_cache_gb=0.01, swap_space_gb=0.01, num_gpu_blocks=512, memory_manager=manager,
        attention_backend="ref", n=4, beam=0, temperature=None, max_wall_factor=10.0,
    )
    model, tok = load_model(args)
    row = run_one_rate(model, tok, args, rate=8.0)
    assert row["finished"] == row["num_requests"] == 8 and row["timed_out"] == 0
    assert row["temperature"] == 1.0 and row["n"] == 4
    if manager == "paged":
        assert 0 < row["mem_saving_avg_pct"] <= row["mem_saving_max_pct"] < 100
    else:
        assert row["mem_saving_avg_pct"] == row["mem_saving_max_pct"] == 0
    assert not math.isnan(row["norm_latency_mean"])
