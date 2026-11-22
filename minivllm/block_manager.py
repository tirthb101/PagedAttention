"""The KV cache manager (§4.2–4.5): block tables, sharing, copy-on-write and swapping.

Analogy with OS virtual memory (CSAPP Ch 9):

    page            -> KV block (block_size tokens)
    page table      -> block table: logical block i of a sequence -> PhysicalBlock
    demand paging   -> append_slot() allocates a block only when the last one is full
    fork() + COW    -> fork() shares blocks; append_slot() copies a shared block on write
    swap to disk    -> swap_out()/swap_in() move a request's blocks to/from CPU memory
    page cache      -> (extension X1) full blocks are registered by content hash; freed
                       ones stay cached on an LRU list and are reused by later requests
                       with the same prefix (enable_prefix_caching=True)

This class only does bookkeeping with block numbers. It never touches tensors; the
CacheEngine performs the copies it asks for (blocks_to_copy / swap mappings).
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from minivllm.block import BlockAllocator, CachingBlockAllocator, Device, PhysicalBlock
from minivllm.sequence import Sequence, SequenceGroup, SequenceStatus

BlockTable = list[PhysicalBlock]


class AllocStatus(enum.Enum):
    OK = enum.auto()     # fits now
    LATER = enum.auto()  # fits once other requests free memory
    NEVER = enum.auto()  # larger than the whole GPU pool: must be rejected


@dataclass
class Prefix:
    """A shared prompt prefix whose full blocks are computed once and mapped into many requests (§4.4)."""

    prefix_id: str
    token_ids: list[int]
    blocks: BlockTable = field(default_factory=list)
    computed: bool = False  # set by the engine after the prefix's KV has been written once

    @property
    def num_shared_tokens(self) -> int:
        # Only full blocks are shared; a partial last block would be written by every request.
        return len(self.blocks) * (self.blocks[0].block_size if self.blocks else 0)


class BlockSpaceManager:
    shares_prompt = True  # sequences of one request share the prompt's blocks (§4.4)

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        enable_prefix_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.enable_prefix_caching = enable_prefix_caching
        gpu_cls = CachingBlockAllocator if enable_prefix_caching else BlockAllocator
        self.gpu_allocator = gpu_cls(Device.GPU, block_size, num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size, num_cpu_blocks)
        self.block_tables: dict[int, BlockTable] = {}
        self.prefixes: dict[str, Prefix] = {}
        # X1: per-sequence chain of full-block hashes (a pure function of the tokens; tokens
        # are only ever appended, so the list is extended incrementally).
        self._seq_hashes: dict[int, list[int]] = {}
        self.reset_cache_stats()

    # ------------------------------------------------------------------ helpers
    def _allocator(self, block: PhysicalBlock) -> BlockAllocator:
        return self.gpu_allocator if block.device == Device.GPU else self.cpu_allocator

    def _prefix_blocks_for(self, group: SequenceGroup) -> BlockTable:
        if group.prefix_id is None:
            return []
        return self.prefixes[group.prefix_id].blocks

    # ------------------------------------------------------- prefix caching (X1)
    def reset_cache_stats(self) -> None:
        self.cache_query_tokens = 0   # prompt tokens of admitted requests (incl. re-admissions)
        self.cache_hit_tokens = 0     # of those, tokens whose KV was found in the cache
        if self.enable_prefix_caching:
            self.gpu_allocator.num_evictions = 0

    def _block_hashes(self, seq: Sequence, num_blocks: int) -> list[int]:
        """Hashes of the first `num_blocks` full blocks: h_i = hash(h_{i-1}, tokens of block i).

        Chaining the parent hash makes equal hashes mean equal *prefixes*, not just equal
        16-token chunks, so a hit at block i implies the whole history up to i matches.
        """
        memo = self._seq_hashes.setdefault(seq.seq_id, [])
        if len(memo) < num_blocks:
            tokens = seq.get_token_ids()
            B = self.block_size
            while len(memo) < num_blocks:
                i = len(memo)
                parent = memo[-1] if memo else 0
                memo.append(hash((parent, tuple(tokens[i * B:(i + 1) * B]))))
        return memo[:num_blocks]

    def _cache_hits(self, group: SequenceGroup, seqs: list[Sequence]) -> BlockTable:
        """Cached blocks matching the leading full blocks of a waiting request's tokens.

        Only blocks entirely inside the first len-1 tokens are looked up: the last token
        is always recomputed (its logits are needed), and it must never be written into a
        shared cached block. Requests using a predefined prefix (§4.4) and diverged
        groups (recompute-preempted parallel samples) are not looked up.
        """
        if not self.enable_prefix_caching or group.prefix_id is not None:
            return []
        first = seqs[0]
        tokens = first.get_token_ids()
        if any(s.get_token_ids() != tokens for s in seqs[1:]):
            return []
        hits: BlockTable = []
        for h in self._block_hashes(first, (first.get_len() - 1) // self.block_size):
            block = self.gpu_allocator.lookup(h)
            if block is None:
                break
            hits.append(block)
        return hits

    def _register_full_blocks(self, seq: Sequence, table: BlockTable, num_full: int) -> None:
        if not self.enable_prefix_caching:
            return
        for block, h in zip(table[:num_full], self._block_hashes(seq, num_full)):
            if block.device == Device.GPU:
                self.gpu_allocator.register(block, h)

    def cache_stats(self) -> dict[str, float]:
        if not self.enable_prefix_caching:
            return {"cache_hit_tokens": 0, "cache_query_tokens": 0, "cache_hit_rate": 0.0,
                    "cache_evictions": 0, "cached_blocks": 0, "evictable_blocks": 0}
        q = self.cache_query_tokens
        return {
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_query_tokens": q,
            "cache_hit_rate": self.cache_hit_tokens / q if q else 0.0,
            "cache_evictions": self.gpu_allocator.num_evictions,
            "cached_blocks": len(self.gpu_allocator.cached),
            "evictable_blocks": self.gpu_allocator.get_num_evictable_blocks(),
        }

    # ---------------------------------------------------------------- admission
    def _num_shared_blocks(self, group: SequenceGroup, seqs: list[Sequence]) -> int:
        """How many leading blocks the waiting sequences of a group can share.

        New requests (and recomputed groups whose sequences are still identical) share
        every block, including a partial last block (copy-on-write later). Sequences that
        already diverged (a parallel-sampling or beam-search group preempted by
        recomputation, §4.5) share only the full blocks of their common token prefix,
        and never a block holding the last token of the shortest sequence (each sequence
        must recompute at least its last token to get logits to sample from).
        """
        first = seqs[0].get_token_ids()
        others = [s.get_token_ids() for s in seqs[1:]]
        if all(o == first for o in others):
            return seqs[0].num_logical_blocks
        common = 0
        for i, tok in enumerate(first):
            if any(i >= len(o) or o[i] != tok for o in others):
                break
            common += 1
        min_len = min(s.get_len() for s in seqs)
        num_shared = min(common, min_len - 1) // self.block_size
        return max(num_shared, len(self._prefix_blocks_for(group)))

    def _num_required_blocks(self, group: SequenceGroup, seqs: list[Sequence]) -> int:
        num_shared = self._num_shared_blocks(group, seqs)
        num_private = sum(s.num_logical_blocks - num_shared for s in seqs)
        return num_shared + num_private - len(self._prefix_blocks_for(group))

    def can_allocate(self, group: SequenceGroup) -> AllocStatus:
        seqs = group.get_seqs(SequenceStatus.WAITING)
        num_required = self._num_required_blocks(group, seqs)
        # X1: cache hits on blocks other requests are using cost no memory. (Hits on
        # evictable blocks do: those blocks are counted as free.)
        num_required -= sum(1 for b in self._cache_hits(group, seqs) if b.ref_count > 0)
        if self.num_total_gpu_blocks - num_required < self.watermark_blocks:
            return AllocStatus.NEVER
        if self.gpu_allocator.get_num_free_blocks() - num_required >= self.watermark_blocks:
            return AllocStatus.OK
        return AllocStatus.LATER

    def allocate(self, group: SequenceGroup) -> None:
        """Allocate blocks for a waiting request's tokens.

        A new request's sequences have identical tokens (the prompt), so ONE set of
        physical blocks is allocated and shared by all of them (§4.4 parallel sampling /
        beam search); the first write to the shared partial last block copies it.

        After recompute-preemption the sequences may have diverged: the full blocks of
        their common prefix are shared again and only the rest is private. The first
        sequence recomputes the shared blocks; the others start after them (their KV is
        written by the first sequence earlier in the same model step, see LLMEngine.step).
        """
        seqs = group.get_seqs(SequenceStatus.WAITING)
        first = seqs[0]
        num_shared = self._num_shared_blocks(group, seqs)
        identical = num_shared == first.num_logical_blocks and all(
            s.get_token_ids() == first.get_token_ids() for s in seqs[1:])

        shared: BlockTable = []
        num_cached_tokens = 0
        if group.prefix_id is not None:
            prefix = self.prefixes[group.prefix_id]
            assert first.get_token_ids()[: len(prefix.token_ids)] == prefix.token_ids, "prompt does not start with prefix"
            shared.extend(prefix.blocks)
            if prefix.computed:
                num_cached_tokens = prefix.num_shared_tokens
            for block in prefix.blocks:
                block.ref_count += len(seqs)

        # X1: map cached blocks first (before any allocation can evict them).
        hits = self._cache_hits(group, seqs)
        for block in hits:
            self.gpu_allocator.acquire(block, len(seqs))
            shared.append(block)
        num_cached_tokens = max(num_cached_tokens, len(hits) * self.block_size)
        if self.enable_prefix_caching:
            self.cache_query_tokens += first.get_len()
            self.cache_hit_tokens += len(hits) * self.block_size

        for _ in range(num_shared - len(shared)):
            block = self.gpu_allocator.allocate()  # ref_count = 1
            block.ref_count = len(seqs)
            shared.append(block)

        for i, seq in enumerate(seqs):
            table = list(shared)
            for _ in range(seq.num_logical_blocks - num_shared):
                table.append(self.gpu_allocator.allocate())
            self.block_tables[seq.seq_id] = table
            cached = num_cached_tokens
            if i > 0 and not identical:
                cached = max(cached, num_shared * self.block_size)
            # Always recompute at least the last token: its logits are needed to sample.
            seq.num_computed_tokens = min(cached, seq.get_len() - 1)
            # X1: every full block is computed by the end of this step, so publish it now.
            # A request admitted later in this same step may hit it: each layer writes the
            # KV of all batch tokens before attending (see LLMEngine.step).
            self._register_full_blocks(seq, table, seq.get_len() // self.block_size)

    # ----------------------------------------------------------------- decoding
    def can_append_slots(self, group: SequenceGroup) -> bool:
        """Conservative: each running sequence may need one new block (§4.3)."""
        return self.gpu_allocator.get_num_free_blocks() >= group.num_seqs(SequenceStatus.RUNNING)

    def append_slot(self, seq: Sequence) -> Optional[tuple[int, int]]:
        """Make sure there is a slot for the sequence's newest token before a decode step.

        Returns (src_block, dst_block) if a copy-on-write copy must be performed by the
        cache engine before the step, else None.
        """
        table = self.block_tables[seq.seq_id]
        if len(table) < seq.num_logical_blocks:
            # Last block is full: demand-allocate a new one.
            assert len(table) == seq.num_logical_blocks - 1, "a sequence grows by at most one block per step"
            # X1: the previous last block just became full (all its tokens are computed):
            # publish it, so e.g. the next turn of a chat can reuse the generated answer.
            if self.enable_prefix_caching and table[-1].device == Device.GPU:
                self.gpu_allocator.register(table[-1], self._block_hashes(seq, len(table))[-1])
            table.append(self.gpu_allocator.allocate())
            return None

        last = table[-1]
        assert last.device == Device.GPU
        if last.ref_count == 1:
            return None  # private block: write in place

        # Shared block (forked sequences): copy-on-write at block granularity (§4.4).
        new_block = self.gpu_allocator.allocate()
        table[-1] = new_block
        self.gpu_allocator.free(last)  # drop our reference to the shared block
        return last.block_number, new_block.block_number

    def fork(self, parent: Sequence, child: Sequence) -> None:
        """Child shares every block of the parent (like fork(): copy the page table, not the pages)."""
        table = self.block_tables[parent.seq_id]
        for block in table:
            block.ref_count += 1
        self.block_tables[child.seq_id] = list(table)

    def free(self, seq: Sequence) -> None:
        table = self.block_tables.pop(seq.seq_id, None)
        self._seq_hashes.pop(seq.seq_id, None)
        if table is None:
            return  # never allocated, or already freed
        # X1: free the tail first, so it is evicted before the prefix it depends on
        # (a cached block is useless once one of its ancestors is gone).
        for block in (reversed(table) if self.enable_prefix_caching else table):
            self._allocator(block).free(block)

    # ----------------------------------------------------------------- swapping
    def _unique_blocks(self, group: SequenceGroup, status: SequenceStatus) -> list[PhysicalBlock]:
        seen: dict[int, PhysicalBlock] = {}
        for seq in group.get_seqs(status):
            for block in self.block_tables[seq.seq_id]:
                seen[id(block)] = block
        return list(seen.values())

    def can_swap_in(self, group: SequenceGroup) -> bool:
        blocks = self._unique_blocks(group, SequenceStatus.SWAPPED)
        # +1 per sequence so each can also append a token right after swapping in.
        num_required = len(blocks) + group.num_seqs(SequenceStatus.SWAPPED)
        return self.gpu_allocator.get_num_free_blocks() - num_required >= self.watermark_blocks

    def can_swap_out(self, group: SequenceGroup) -> bool:
        blocks = self._unique_blocks(group, SequenceStatus.RUNNING)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def _move(self, group: SequenceGroup, status: SequenceStatus, src: BlockAllocator, dst: BlockAllocator) -> dict[int, int]:
        """Move all blocks of `group`'s sequences with `status` from src to dst, preserving sharing."""
        mapping: dict[int, PhysicalBlock] = {}  # id(src block) -> dst block
        number_map: dict[int, int] = {}
        for seq in group.get_seqs(status):
            new_table: BlockTable = []
            for block in self.block_tables[seq.seq_id]:
                if block.device != src.device:
                    raise RuntimeError(f"{seq} has a {block.device.name} block while moving from {src.device.name}")
                if id(block) in mapping:
                    new_block = mapping[id(block)]
                    new_block.ref_count += 1
                else:
                    new_block = dst.allocate()
                    mapping[id(block)] = new_block
                    number_map[block.block_number] = new_block.block_number
                src.free(block)
                new_table.append(new_block)
            self.block_tables[seq.seq_id] = new_table
        return number_map

    def swap_out(self, group: SequenceGroup) -> dict[int, int]:
        """GPU -> CPU. Returns {gpu_block_number: cpu_block_number} for the cache engine.

        Note: blocks of a shared prefix are copied too (the CPU copy is private); the
        prefix's GPU blocks stay pinned by the registry.
        """
        return self._move(group, SequenceStatus.RUNNING, self.gpu_allocator, self.cpu_allocator)

    def swap_in(self, group: SequenceGroup) -> dict[int, int]:
        """CPU -> GPU. Returns {cpu_block_number: gpu_block_number} for the cache engine."""
        return self._move(group, SequenceStatus.SWAPPED, self.cpu_allocator, self.gpu_allocator)

    # ------------------------------------------------------------ shared prefix
    def register_prefix(self, prefix_id: str, token_ids: list[int]) -> Prefix:
        """Reserve GPU blocks for the full blocks of a predefined prefix (§4.4).

        The registry holds one reference per block, so the blocks stay resident even
        when no request is using them. The engine computes their KV with the first
        request that uses the prefix, then calls mark_prefix_computed().
        """
        if prefix_id in self.prefixes:
            raise ValueError(f"prefix {prefix_id!r} already registered")
        num_full_blocks = len(token_ids) // self.block_size
        blocks = [self.gpu_allocator.allocate() for _ in range(num_full_blocks)]
        prefix = Prefix(prefix_id, list(token_ids), blocks)
        self.prefixes[prefix_id] = prefix
        return prefix

    def mark_prefix_computed(self, prefix_id: str) -> None:
        self.prefixes[prefix_id].computed = True

    # ---------------------------------------------------------------- queries
    def get_block_table(self, seq: Sequence) -> list[int]:
        return [b.block_number for b in self.block_tables[seq.seq_id]]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()

    def memory_stats(self, seqs: Iterable[Sequence]) -> dict[str, int]:
        """Slots allocated vs slots holding real tokens (paper Fig. 2 metric).

        A shared block counts once, filled up to the longest sequence using it.
        """
        filled: dict[int, int] = {}
        for seq in seqs:
            table = self.block_tables.get(seq.seq_id)
            if table is None:
                continue
            n = seq.get_len()
            for i, block in enumerate(table):
                if block.device != Device.GPU:
                    break
                used = max(0, min(self.block_size, n - i * self.block_size))
                filled[id(block)] = max(filled.get(id(block), 0), used)
        used_blocks = self.gpu_allocator.get_num_used_blocks()
        return {
            "allocated_slots": used_blocks * self.block_size,
            "used_slots": sum(filled.values()),
            "used_blocks": used_blocks,
            "free_blocks": self.gpu_allocator.get_num_free_blocks(),
        }

    def sharing_stats(self) -> dict[str, float]:
        """Memory saved by block sharing right now (paper Fig. 14/15 "memory saving").

        logical  = sum over sequences holding GPU blocks of their block-table length
                   (the blocks they would need if nothing were shared)
        physical = distinct GPU blocks those tables point to
        saving   = (logical - physical) / logical
        Swapped-out sequences (CPU tables) and blocks pinned only by the prefix registry
        are not counted.
        """
        logical = 0
        physical: set[int] = set()
        for table in self.block_tables.values():
            if not table or table[0].device != Device.GPU:
                continue
            logical += len(table)
            physical.update(id(b) for b in table)
        saving = (logical - len(physical)) / logical if logical else 0.0
        return {"logical_blocks": logical, "physical_blocks": len(physical), "saving": saving}

    def check_invariants(self, seqs: Optional[Iterable[Sequence]] = None) -> None:
        """Debug check run after every step in tests: catches leaks, double counts and bad tables."""
        refs: Counter[int] = Counter()
        by_id: dict[int, PhysicalBlock] = {}
        for seq_id, table in self.block_tables.items():
            devices = {b.device for b in table}
            assert len(devices) <= 1, f"seq {seq_id} has blocks on several devices"
            for block in table:
                refs[id(block)] += 1
                by_id[id(block)] = block
        for prefix in self.prefixes.values():
            for block in prefix.blocks:
                refs[id(block)] += 1  # the registry's pin
                by_id[id(block)] = block

        for alloc in (self.gpu_allocator, self.cpu_allocator):
            free_ids = [id(b) for b in alloc.free_blocks]
            assert len(free_ids) == len(set(free_ids)), f"{alloc.device.name} free list has duplicates"
            evictable = {id(b) for b in getattr(alloc, "evictor", {}).values()}
            free_set = set(free_ids)
            assert not (free_set & evictable), "a block is both free and evictable"
            for block in alloc.blocks:
                expected = refs.get(id(block), 0)
                assert block.ref_count == expected, f"{block}: ref_count {block.ref_count} != {expected} table references"
                on_free_list = id(block) in free_set or id(block) in evictable
                assert on_free_list == (expected == 0), f"{block}: free-list membership wrong"
                if id(block) in free_set:
                    assert block.content_hash is None, f"{block} is free but still cached"
                if id(block) in evictable:
                    assert block.content_hash is not None, f"{block} is evictable but not cached"
            for h, block in getattr(alloc, "cached", {}).items():
                assert block.content_hash == h, f"{block}: cache table hash mismatch"
        assert set(by_id) <= {id(b) for a in (self.gpu_allocator, self.cpu_allocator) for b in a.blocks}

        if seqs is not None:
            for seq in seqs:
                if seq.status in (SequenceStatus.RUNNING, SequenceStatus.SWAPPED):
                    table = self.block_tables.get(seq.seq_id)
                    assert table is not None, f"{seq} has no block table"
                    # A running sequence may hold one extra block reserved for its next token.
                    assert len(table) >= seq.num_logical_blocks - 1, f"{seq}: table too short ({len(table)} blocks)"
