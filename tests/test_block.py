import pytest

from minivllm.block import BlockAllocator, Device, DoubleFreeError, OutOfBlocksError


def test_allocate_all_then_free_all():
    alloc = BlockAllocator(Device.GPU, block_size=16, num_blocks=4)
    blocks = [alloc.allocate() for _ in range(4)]
    assert [b.block_number for b in blocks] == [0, 1, 2, 3]
    assert len({id(b) for b in blocks}) == 4
    assert all(b.ref_count == 1 for b in blocks)
    assert alloc.get_num_free_blocks() == 0
    with pytest.raises(OutOfBlocksError):
        alloc.allocate()
    for b in blocks:
        alloc.free(b)
    assert alloc.get_num_free_blocks() == 4


def test_shared_block_returns_to_free_list_only_at_zero_refs():
    alloc = BlockAllocator(Device.GPU, 16, 2)
    b = alloc.allocate()
    b.ref_count += 1  # a second block-table entry points here
    alloc.free(b)
    assert alloc.get_num_free_blocks() == 1  # still in use
    alloc.free(b)
    assert alloc.get_num_free_blocks() == 2


def test_double_free_detected():
    alloc = BlockAllocator(Device.CPU, 16, 1)
    b = alloc.allocate()
    alloc.free(b)
    with pytest.raises(DoubleFreeError):
        alloc.free(b)


def test_wrong_device_rejected():
    gpu = BlockAllocator(Device.GPU, 16, 1)
    cpu = BlockAllocator(Device.CPU, 16, 1)
    with pytest.raises(ValueError):
        cpu.free(gpu.allocate())
