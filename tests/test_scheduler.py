import random

import pytest

from minivllm.block_manager import BlockSpaceManager
from minivllm.config import SchedulerConfig
from minivllm.scheduler import Scheduler
from minivllm.sequence import SequenceStatus

from conftest import make_group


def make_scheduler(gpu_blocks=16, cpu_blocks=16, block_size=4, mode="recompute", max_seqs=64, max_tokens=256, max_len=128):
    bm = BlockSpaceManager(block_size, gpu_blocks, cpu_blocks, watermark=0.0)
    cfg = SchedulerConfig(max_num_seqs=max_seqs, max_num_batched_tokens=max_tokens, max_model_len=max_len, preemption_mode=mode)
    return Scheduler(cfg, bm)


def fake_model_step(sched: Scheduler, out, rng: random.Random, new_seq_id, beam_fork_prob: float = 0.0):
    """Stand-in for the model: checks the scheduler gave every token a slot, then 'samples' one token."""
    bm = sched.block_manager
    for group in out.scheduled_groups:
        running = group.get_seqs(SequenceStatus.RUNNING)
        for seq in running:
            table = bm.block_tables[seq.seq_id]
            assert len(table) * bm.block_size >= seq.get_len(), "no KV slot for a token about to be computed"
            assert all(b.device.name == "GPU" for b in table)
            if group in out.decode_groups:
                assert seq.get_num_uncomputed_tokens() == 1, "decode step must process exactly one token"
            seq.num_computed_tokens = seq.get_len()
        for seq in running:
            seq.append_token(rng.randrange(50_000))
            if seq.get_output_len() >= group.true_output_len:
                seq.status = SequenceStatus.FINISHED_LENGTH
                sched.free_seq(seq)
        # Beam-search-like reshuffle: drop one candidate, fork another in its place.
        alive = group.get_seqs(SequenceStatus.RUNNING)
        if len(alive) >= 2 and rng.random() < beam_fork_prob:
            keep, drop = rng.sample(alive, 2)
            drop.status = SequenceStatus.FINISHED_STOPPED
            sched.free_seq(drop)
            group.remove(drop.seq_id)
            child = keep.fork(new_seq_id())
            bm.fork(keep, child)
            group.add(child)
    sched.free_finished_seq_groups()


def all_live_seqs(sched):
    groups = list(sched.waiting) + sched.running + sched.swapped
    return [s for g in groups for s in g.seqs]


# --------------------------------------------------------------------- unit tests


def test_fcfs_admission_and_batching():
    sched = make_scheduler(gpu_blocks=16)
    groups = [make_group(8, 4, arrival=t) for t in range(3)]
    for g in groups:
        sched.add_seq_group(g)
    out = sched.schedule()
    assert out.prefill_groups == groups
    assert out.num_batched_tokens == 24
    assert not out.decode_groups


def test_strict_fcfs_no_overtaking():
    sched = make_scheduler(gpu_blocks=4)
    big = make_group(16, 4, arrival=0)       # needs all 4 blocks
    first = make_group(4, 4, arrival=-1)     # arrives earlier, 1 block
    sched.add_seq_group(first)
    sched.add_seq_group(big)
    small = make_group(4, 4, arrival=1)
    sched.add_seq_group(small)
    out = sched.schedule()
    assert out.prefill_groups == [first]     # big doesn't fit -> small must wait behind it


def test_too_long_prompt_is_ignored():
    sched = make_scheduler(gpu_blocks=4, max_len=128)
    g = make_group(200, 4)
    sched.add_seq_group(g)
    out = sched.schedule()
    assert out.ignored == [g]
    assert all(s.status == SequenceStatus.FINISHED_ABORTED for s in g.seqs)


def _fill_and_decode(mode):
    # 2 requests of 8 tokens with block size 4 fill all 4 blocks; the next token needs a 5th.
    sched = make_scheduler(gpu_blocks=4, mode=mode)
    old, young = make_group(8, 4, arrival=0, output_len=50), make_group(8, 4, arrival=1, output_len=50)
    sched.add_seq_group(old)
    sched.add_seq_group(young)
    out = sched.schedule()
    assert out.prefill_groups == [old, young]
    for g in (old, young):
        (s,) = g.seqs
        s.num_computed_tokens = s.get_len()
        s.append_token(7)
    out = sched.schedule()
    return sched, out, old, young


def test_preempt_latest_by_recompute():
    sched, out, old, young = _fill_and_decode("recompute")
    assert out.preempted == [young]
    assert out.decode_groups == [old]
    assert sched.waiting[0] is young
    (s,) = young.seqs
    assert s.status == SequenceStatus.WAITING and s.num_computed_tokens == 0
    assert s.get_output_len() == 1  # generated tokens are kept and will be re-prefilled
    sched.block_manager.check_invariants(all_live_seqs(sched))


def test_preempt_latest_by_swap_blocks_admission():
    sched, out, old, young = _fill_and_decode("swap")
    assert out.preempted == [young]
    assert len(out.blocks_to_swap_out) == 2
    assert sched.swapped == [young]
    sched.add_seq_group(make_group(4, 4, arrival=5))
    out2 = sched.schedule()
    assert not out2.prefill_groups  # nothing admitted while a request is swapped out
    sched.block_manager.check_invariants(all_live_seqs(sched))


def test_multi_sequence_groups_are_swapped_even_in_recompute_mode():
    sched = make_scheduler(gpu_blocks=6, cpu_blocks=16, mode="recompute")
    old = make_group(8, 4, arrival=0, output_len=50)
    beams = make_group(8, 4, n=2, arrival=1, output_len=50)
    sched.add_seq_group(old)
    sched.add_seq_group(beams)
    sched.schedule()
    for g in (old, beams):
        for s in g.seqs:
            s.num_computed_tokens = s.get_len()
            s.append_token(3)
    out = sched.schedule()  # needs 1 + 2 new blocks, only 2 free
    assert out.preempted == [beams]
    assert sched.swapped == [beams]


# ----------------------------------------------------------------- simulation fuzz


@pytest.mark.parametrize("gpu_blocks", [24, 48])
@pytest.mark.parametrize("mode", ["recompute", "swap"])
@pytest.mark.parametrize("seed", range(5))
def test_random_workload_finishes_without_leaks(mode, seed, gpu_blocks, new_seq_id):
    """Random requests through a tight memory pool: every request must finish, invariants
    must hold after every step, and no block may leak.

    With 48 GPU blocks the largest possible request (3 seqs x 15 blocks) always fits, so
    nothing may be aborted. With 24 blocks a heavily forked request can outgrow the whole
    pool; it must then be aborted cleanly instead of deadlocking the queue.
    """
    rng = random.Random(seed)
    sched = make_scheduler(gpu_blocks=gpu_blocks, cpu_blocks=96, block_size=4, mode=mode, max_seqs=8, max_tokens=128, max_len=96)
    groups = []
    for i in range(30):
        n = rng.choice([1, 1, 1, 2, 3])
        g = make_group(rng.randint(1, 30), 4, n=n, arrival=float(i), output_len=rng.randint(1, 30))
        groups.append(g)

    pending = list(groups)
    for step in range(5000):
        while pending and pending[0].arrival_time <= step:
            sched.add_seq_group(pending.pop(0))
        out = sched.schedule(now=step)
        fake_model_step(sched, out, rng, new_seq_id, beam_fork_prob=0.3)
        sched.block_manager.check_invariants(all_live_seqs(sched))
        if not pending and not sched.has_unfinished():
            break
    else:
        pytest.fail("workload did not finish")

    aborted = 0
    for g in groups:
        assert g.is_finished()
        aborted += any(s.status == SequenceStatus.FINISHED_ABORTED for s in g.seqs)
        for s in g.seqs:
            if s.status == SequenceStatus.FINISHED_LENGTH:
                assert s.get_output_len() == g.true_output_len
    if gpu_blocks == 48:
        assert aborted == 0
    bm = sched.block_manager
    assert bm.get_num_free_gpu_blocks() == gpu_blocks
    assert bm.get_num_free_cpu_blocks() == 96
    assert not bm.block_tables
    if mode == "swap":
        assert sched.num_preemptions > 0, "test should exercise preemption"
