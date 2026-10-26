from minivllm.block_manager import AllocStatus, BlockSpaceManager
from minivllm.sequence import SequenceStatus

from conftest import make_group


def bm_with(gpu=16, cpu=16, block_size=4):
    return BlockSpaceManager(block_size, gpu, cpu, watermark=0.0)


def admit(bm, group):
    bm.allocate(group)
    for s in group.get_seqs(SequenceStatus.WAITING):
        s.status = SequenceStatus.RUNNING
    return group.seqs


def test_paper_example_on_demand_allocation():
    """§4.3: 'Four score and seven years ago our' = 7 tokens, block size 4."""
    bm = bm_with(block_size=4)
    (seq,) = admit(bm, make_group(prompt_len=7, block_size=4))
    assert len(bm.block_tables[seq.seq_id]) == 2  # blocks hold 4 + 3 tokens

    seq.append_token(1)  # "fathers" fills slot 4 of block 1
    assert bm.append_slot(seq) is None
    assert len(bm.block_tables[seq.seq_id]) == 2  # no new block needed

    seq.append_token(2)  # "brought": block 1 full -> new block
    assert bm.append_slot(seq) is None
    assert len(bm.block_tables[seq.seq_id]) == 3
    bm.check_invariants([seq])

    bm.free(seq)
    assert bm.get_num_free_gpu_blocks() == 16
    bm.check_invariants()


def test_waste_bounded_by_one_block():
    bm = bm_with(gpu=64, block_size=16)
    (seq,) = admit(bm, make_group(prompt_len=37, block_size=16))
    stats = bm.memory_stats([seq])
    assert stats["used_slots"] == 37
    assert stats["allocated_slots"] - stats["used_slots"] < 16


def test_parallel_sampling_shares_prompt_blocks():
    bm = bm_with(gpu=16, block_size=4)
    seqs = admit(bm, make_group(prompt_len=6, block_size=4, n=3))
    tables = [bm.get_block_table(s) for s in seqs]
    assert tables[0] == tables[1] == tables[2]
    assert bm.get_num_free_gpu_blocks() == 16 - 2  # prompt stored once, not three times
    assert all(b.ref_count == 3 for b in bm.block_tables[seqs[0].seq_id])
    bm.check_invariants(seqs)


def test_copy_on_write_when_writing_to_shared_block():
    bm = bm_with(gpu=16, block_size=4)
    a, b = admit(bm, make_group(prompt_len=6, block_size=4, n=2))  # last block has 2 free slots, shared
    shared_last = bm.block_tables[a.seq_id][-1]

    a.append_token(1)
    cow = bm.append_slot(a)
    assert cow is not None
    src, dst = cow
    assert src == shared_last.block_number and dst != src
    assert shared_last.ref_count == 1  # now only b uses it

    b.append_token(2)
    assert bm.append_slot(b) is None  # b is now the sole owner: write in place
    assert bm.block_tables[a.seq_id][0] is bm.block_tables[b.seq_id][0]  # first block still shared
    bm.check_invariants([a, b])


def test_fork_shares_then_free_releases_everything(new_seq_id):
    bm = bm_with(gpu=16, block_size=4)
    (parent,) = admit(bm, make_group(prompt_len=9, block_size=4))
    child = parent.fork(new_seq_id())
    bm.fork(parent, child)
    assert bm.get_block_table(child) == bm.get_block_table(parent)
    assert all(b.ref_count == 2 for b in bm.block_tables[parent.seq_id])
    used = 16 - bm.get_num_free_gpu_blocks()
    assert used == 3  # fork allocated nothing

    bm.free(parent)
    assert bm.get_num_free_gpu_blocks() == 16 - 3  # child still holds them
    bm.free(child)
    assert bm.get_num_free_gpu_blocks() == 16
    bm.check_invariants()


def test_swap_out_and_in_preserves_sharing():
    bm = bm_with(gpu=16, cpu=16, block_size=4)
    group = make_group(prompt_len=8, block_size=4, n=2)
    a, b = admit(bm, group)
    a.append_token(1)
    b.append_token(2)
    bm.append_slot(a)
    bm.append_slot(b)  # each now has a private third block
    gpu_before = 16 - bm.get_num_free_gpu_blocks()
    assert gpu_before == 4  # 2 shared + 2 private

    mapping = bm.swap_out(group)
    for s in group.seqs:
        s.status = SequenceStatus.SWAPPED
    assert len(mapping) == 4  # shared blocks copied once
    assert bm.get_num_free_gpu_blocks() == 16
    assert 16 - bm.get_num_free_cpu_blocks() == 4
    bm.check_invariants(group.seqs)

    assert bm.can_swap_in(group)
    back = bm.swap_in(group)
    for s in group.seqs:
        s.status = SequenceStatus.RUNNING
    assert set(back) == set(mapping.values())
    assert bm.block_tables[a.seq_id][0] is bm.block_tables[b.seq_id][0]  # still shared
    assert bm.get_num_free_cpu_blocks() == 16
    bm.check_invariants(group.seqs)


def test_can_allocate_ok_later_never():
    bm = bm_with(gpu=4, block_size=4)
    assert bm.can_allocate(make_group(prompt_len=16, block_size=4)) == AllocStatus.OK
    assert bm.can_allocate(make_group(prompt_len=17, block_size=4)) == AllocStatus.NEVER
    admit(bm, make_group(prompt_len=12, block_size=4))  # uses 3 of 4 blocks
    assert bm.can_allocate(make_group(prompt_len=8, block_size=4)) == AllocStatus.LATER


def test_shared_prefix_blocks_mapped_not_copied():
    bm = bm_with(gpu=32, block_size=4)
    prefix_tokens = list(range(10))  # 2 full blocks shared, 2 tokens not shareable
    prefix = bm.register_prefix("fewshot", prefix_tokens)
    assert len(prefix.blocks) == 2
    assert bm.get_num_free_gpu_blocks() == 30

    g1 = make_group(prompt_len=14, block_size=4, prefix_tokens=prefix_tokens, prefix_id="fewshot")
    (s1,) = admit(bm, g1)
    assert s1.num_computed_tokens == 0  # first user computes the prefix
    bm.mark_prefix_computed("fewshot")

    g2 = make_group(prompt_len=13, block_size=4, prefix_tokens=prefix_tokens, prefix_id="fewshot")
    (s2,) = admit(bm, g2)
    assert s2.num_computed_tokens == 8  # 2 cached blocks skipped in prefill
    assert bm.block_tables[s1.seq_id][:2] == bm.block_tables[s2.seq_id][:2]
    assert bm.block_tables[s1.seq_id][2] is not bm.block_tables[s2.seq_id][2]
    bm.check_invariants([s1, s2])

    bm.free(s1)
    bm.free(s2)
    assert bm.get_num_free_gpu_blocks() == 30  # prefix stays pinned
    bm.check_invariants()
