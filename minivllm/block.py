"""Physical KV blocks and the free-list allocator (§4.2).

A physical block is just an index into a big preallocated KV tensor pool (on the GPU
or in CPU memory) plus a reference count. Every block has the same size, so the
allocator is a plain free list: no placement decisions and no external fragmentation
(compare with the explicit free lists of CSAPP §9.9.13).
"""

from __future__ import annotations

import enum
from collections import OrderedDict
from typing import Optional


class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class OutOfBlocksError(RuntimeError):
    """Raised when allocating from an empty pool. The scheduler should prevent this."""


class DoubleFreeError(RuntimeError):
    """Raised when freeing a block whose reference count is already zero."""


class PhysicalBlock:
    __slots__ = ("block_number", "device", "block_size", "ref_count", "content_hash")

    def __init__(self, block_number: int, device: Device, block_size: int) -> None:
        self.block_number = block_number
        self.device = device
        self.block_size = block_size
        # Number of block-table entries (across all sequences) that point here.
        # > 1 means the block is shared, so writing to it needs copy-on-write (§4.4).
        self.ref_count = 0
        # Extension X1 (prefix caching): hash of (parent block's hash, this block's tokens)
        # once the block is full and registered in the cache; None otherwise.
        self.content_hash: Optional[int] = None

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


class CachingBlockAllocator(BlockAllocator):
    """GPU allocator with an automatic prefix cache (extension X1).

    Like the OS page cache: a full block whose contents are known is registered under
    its content hash. When its last user frees it, the block is NOT forgotten; it goes
    to an LRU list of *evictable* blocks that still hold valid KV. A later request whose
    prompt has the same prefix maps the cached block back into its block table
    (ref_count++) instead of recomputing it.

    Allocation takes a truly free block first and only then evicts the least recently
    freed cached block. Evictable blocks count as free, so admission control is unchanged.
    """

    def __init__(self, device: Device, block_size: int, num_blocks: int) -> None:
        super().__init__(device, block_size, num_blocks)
        self.cached: dict[int, PhysicalBlock] = {}               # content hash -> block
        self.evictor: OrderedDict[int, PhysicalBlock] = OrderedDict()  # block number -> block, LRU first
        self.num_evictions = 0

    def allocate(self) -> PhysicalBlock:
        if self.free_blocks:
            return super().allocate()
        if not self.evictor:
            raise OutOfBlocksError(f"no free {self.device.name} blocks")
        _, block = self.evictor.popitem(last=False)  # least recently freed
        assert block.ref_count == 0
        del self.cached[block.content_hash]
        block.content_hash = None
        block.ref_count = 1
        self.num_evictions += 1
        return block

    def free(self, block: PhysicalBlock) -> None:
        if block.device != self.device:
            raise ValueError(f"{block} does not belong to the {self.device.name} allocator")
        if block.ref_count == 0:
            raise DoubleFreeError(f"double free of {block}")
        block.ref_count -= 1
        if block.ref_count == 0:
            if block.content_hash is not None:
                self.evictor[block.block_number] = block  # keep the KV, evict later if needed
            else:
                self.free_blocks.append(block)

    # ---- cache operations ----------------------------------------------------------
    def lookup(self, content_hash: int) -> Optional[PhysicalBlock]:
        return self.cached.get(content_hash)

    def acquire(self, block: PhysicalBlock, refs: int = 1) -> None:
        """Map a cached block into `refs` more block-table entries (a cache hit)."""
        assert self.cached.get(block.content_hash) is block, f"{block} is not in the cache"
        if block.ref_count == 0:
            del self.evictor[block.block_number]
        block.ref_count += refs

    def register(self, block: PhysicalBlock, content_hash: int) -> bool:
        """Publish a full, computed block under its hash. Returns False if another block
        already holds the same content (that one stays the cached copy)."""
        if block.content_hash is not None:
            assert block.content_hash == content_hash, f"{block} re-registered with a different hash"
            return False
        if content_hash in self.cached:
            return False
        assert block.ref_count > 0, "only live blocks are registered"
        block.content_hash = content_hash
        self.cached[content_hash] = block
        return True

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks) + len(self.evictor)

    def get_num_used_blocks(self) -> int:
        return self.num_blocks - self.get_num_free_blocks()

    def get_num_evictable_blocks(self) -> int:
        return len(self.evictor)
