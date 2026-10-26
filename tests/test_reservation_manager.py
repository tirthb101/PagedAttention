"""Orca baselines: reservation sizes, buddy allocator, and that they plug into the real scheduler."""

import pytest

from minivllm.block_manager import AllocStatus
from minivllm.reservation_manager import BuddyAllocator, ReservationBlockManager, next_pow2
from minivllm.sequence import SequenceStatus

from conftest import make_group


def test_next_pow2():
    assert [next_pow2(x) for x in (1, 2, 3, 5, 16, 17, 25)] == [1, 2, 4, 8, 16, 32, 32]


@pytest.mark.parametrize("policy,expected_tokens", [("orca-oracle", 10 + 25), ("orca-pow2", 10 + 32), ("orca-max", 2048)])
def test_reservation_sizes_match_paper(policy, expected_tokens):
    # Paper §6.1: true output 25 -> Pow2 reserves 32; Max reserves the 2048-token context.
    bm = ReservationBlockManager(16, 1024, policy=policy, max_model_len=2048)
    assert bm.reserved_tokens(make_group(10, 16, output_len=25)) == expected_tokens


def test_buddy_split_merge_and_alignment():
    b = BuddyAllocator(16)
    a = b.alloc(2)          # 4 blocks
    c = b.alloc(2)
    d = b.alloc(3)          # 8 blocks
    assert {a, c, d} == {0, 4, 8} and all(s % 4 == 0 for s in (a, c))
    assert b.num_free_blocks() == 0 and b.alloc(0) is None
    b.free(a)
    b.free(c)               # 0..3 and 4..7 merge back into one 8-block range
    assert b.max_free_order() == 3
    b.free(d)
    assert b.max_free_order() == 4 and b.num_free_blocks() == 16


def test_buddy_non_power_of_two_pool():
    b = BuddyAllocator(13)  # 8 + 4 + 1
    assert b.num_free_blocks() == 13 and b.max_free_order() == 3
    assert b.alloc(4) is None


def test_external_fragmentation_is_detected():
    # Pool of 8 blocks; two 2-block reservations sit at 0-1 and 4-5 after frees, so 4 blocks are
    # free but no aligned 4-block range exists.
    bm = ReservationBlockManager(4, 8, policy="orca-oracle", max_model_len=64)
    groups = [make_group(4, 4, output_len=4) for _ in range(4)]  # 8 tokens = 2 blocks each
    for g in groups:
        assert bm.can_allocate(g) == AllocStatus.OK
        bm.allocate(g)
    bm.free(groups[1].seqs[0])
    bm.free(groups[3].seqs[0])
    assert bm.get_num_free_gpu_blocks() == 4
    big = make_group(12, 4, output_len=4)  # 16 tokens = 4 blocks
    assert bm.can_allocate(big) == AllocStatus.LATER
    assert bm.num_external_frag_events == 1
    bm.check_invariants()


def test_parallel_samples_get_private_reservations():
    bm = ReservationBlockManager(4, 64, policy="orca-oracle", max_model_len=64)
    g = make_group(6, 4, n=3, output_len=6)   # 12 tokens -> 3 blocks -> rounded to 4 each
    bm.allocate(g)
    tables = [bm.get_block_table(s) for s in g.seqs]
    assert len({tuple(t) for t in tables}) == 3          # no sharing
    assert bm.get_num_free_gpu_blocks() == 64 - 3 * 4
    for s in g.seqs:
        bm.free(s)
    assert bm.get_num_free_gpu_blocks() == 64
    bm.check_invariants()


def test_waste_breakdown_oracle_vs_max():
    for policy, internal_expected in [("orca-oracle", False), ("orca-max", True)]:
        bm = ReservationBlockManager(16, 1024, policy=policy, max_model_len=2048)
        g = make_group(100, 16, output_len=100)
        bm.allocate(g)
        g.seqs[0].status = SequenceStatus.RUNNING
        st = bm.memory_stats(g.seqs)
        assert st["used_slots"] == 100 and st["reserved_future_slots"] == 100
        internal = st["allocated_slots"] - 200
        assert (internal > 1000) == internal_expected  # Max wastes most of 2048 slots
