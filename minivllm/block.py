"""Physical KV blocks and the free-list allocator (§4.2).

A physical block is just an index into a big preallocated KV tensor pool (on the GPU
or in CPU memory) plus a reference count. Every block has the same size, so the
allocator is a plain free list: no placement decisions and no external fragmentation
(compare with the explicit free lists of CSAPP §9.9.13).
"""

from __future__ import annotations

import enum


class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class OutOfBlocksError(RuntimeError):
    """Raised when allocating from an empty pool. The scheduler should prevent this."""


class DoubleFreeError(RuntimeError):
    """Raised when freeing a block whose reference count is already zero."""


class PhysicalBlock:
    __slots__ = ("block_number", "device", "block_size", "ref_count")

    def __init__(self, block_number: int, device: Device, block_size: int) -> None:
        self.block_number = block_number
        self.device = device
        self.block_size = block_size
        # Number of block-table entries (across all sequences) that point here.
        # > 1 means the block is shared, so writing to it needs copy-on-write (§4.4).
        self.ref_count = 0

    def __repr__(self) -> str:
        return f"PhysicalBlock({self.device.name}#{self.block_number}, ref={self.ref_count})"


class BlockAllocator:
    """Hands out free blocks from one device's pool."""

    def __init__(self, device: Device, block_size: int, num_blocks: int) -> None:
        if num_blocks < 0:
            raise ValueError("num_blocks must be >= 0")
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.blocks = [PhysicalBlock(i, device, block_size) for i in range(num_blocks)]
        # LIFO free list. Reversed so block 0 is handed out first (easier to read in tests).
        self.free_blocks: list[PhysicalBlock] = list(reversed(self.blocks))

    def allocate(self) -> PhysicalBlock:
        if not self.free_blocks:
            raise OutOfBlocksError(f"no free {self.device.name} blocks")
        block = self.free_blocks.pop()
        assert block.ref_count == 0, f"{block} on the free list with live references"
        block.ref_count = 1
        return block

    def free(self, block: PhysicalBlock) -> None:
        """Drop one reference; the block returns to the free list when none remain."""
        if block.device != self.device:
            raise ValueError(f"{block} does not belong to the {self.device.name} allocator")
        if block.ref_count == 0:
            raise DoubleFreeError(f"double free of {block}")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def get_num_used_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)
