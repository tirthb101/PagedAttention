"""Orca baselines (§6.1): reserve each request's whole KV cache up front, contiguously.

This is the memory scheme PagedAttention replaces. The paper re-implemented Orca in
three variants that differ only in how much they reserve per sequence:

    orca-oracle : prompt + true output length   (impossible in practice: knows the future)
    orca-pow2   : prompt + next_pow2(output)    (over-reserves the output by at most 2x)
    orca-max    : max_model_len (2048)          (what a real system without the oracle must do)

and assumed a buddy allocator places the reservations in memory. We do the same: the
KV pool is the same array of blocks as for paged attention, but each sequence gets one
contiguous, power-of-two-sized, aligned range of blocks. This produces all three kinds
of waste from §3.1:

    reserved slots          - space for tokens not generated yet
    internal fragmentation  - space the sequence will never use (over-reservation, buddy rounding)
    external fragmentation  - free blocks that can't be used because no free range is big enough

The class has the same interface as BlockSpaceManager, so the Scheduler, Engine and
simulator run unchanged; only the memory policy differs. There is no sharing: forked
sequences (parallel sampling) each get a full private reservation and recompute the prompt.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

from minivllm.block import Device, PhysicalBlock
from minivllm.block_manager import AllocStatus
from minivllm.sequence import Sequence, SequenceGroup, SequenceStatus

POLICIES = ("orca-oracle", "orca-pow2", "orca-max")


def next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


class BuddyAllocator:
    """Classic binary buddy allocator over block indices [0, num_blocks)."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.free_lists: dict[int, set[int]] = {}   # order -> start indices of free ranges of size 2**order
        self.allocated: dict[int, int] = {}          # start -> order
        # Carve the pool into maximal aligned power-of-two ranges.
        start = 0
        while start < num_blocks:
            order = (num_blocks - start).bit_length() - 1
            while start % (1 << order):
                order -= 1
            self.free_lists.setdefault(order, set()).add(start)
            start += 1 << order

    def max_free_order(self) -> int:
        orders = [o for o, s in self.free_lists.items() if s]
        return max(orders) if orders else -1

    def max_order(self) -> int:
        """Largest range that could ever exist in this pool (when it is empty)."""
        return (self.num_blocks.bit_length() - 1) if self.num_blocks else -1

    def alloc(self, order: int) -> Optional[int]:
        candidates = [o for o, s in self.free_lists.items() if s and o >= order]
        if not candidates:
            return None
        o = min(candidates)
        start = min(self.free_lists[o])  # lowest address first: deterministic
        self.free_lists[o].discard(start)
        while o > order:                  # split, keep the lower half
            o -= 1
            self.free_lists.setdefault(o, set()).add(start + (1 << o))
        self.allocated[start] = order
        return start

    def free(self, start: int) -> None:
        order = self.allocated.pop(start)
        while True:
            buddy = start ^ (1 << order)
            if buddy + (1 << order) > self.num_blocks or buddy not in self.free_lists.get(order, set()):
                break
            self.free_lists[order].discard(buddy)   # merge with the free buddy
            start = min(start, buddy)
            order += 1
        self.free_lists.setdefault(order, set()).add(start)

    def num_free_blocks(self) -> int:
        return sum(len(s) << o for o, s in self.free_lists.items())


class ReservationBlockManager:
    shares_prompt = False   # each sequence holds its own copy of the prompt's KV cache

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int = 0,
        policy: str = "orca-oracle",
        max_model_len: int = 2048,
        watermark: float = 0.0,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.policy = policy
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        self.buddy = BuddyAllocator(num_gpu_blocks)
        self.blocks = [PhysicalBlock(i, Device.GPU, block_size) for i in range(num_gpu_blocks)]
        self.block_tables: dict[int, list[PhysicalBlock]] = {}
        # seq_id -> (start block, n_blocks, final_len): final_len = tokens the sequence will really reach
        self.reservations: dict[int, tuple[int, int, int]] = {}
        self.prefixes: dict = {}
        self.num_external_frag_events = 0  # admissions refused although enough blocks were free

    # ------------------------------------------------------------ reservation
    def reserved_tokens(self, group: SequenceGroup) -> int:
        seq = group.get_unfinished_seqs()[0]
        prompt = seq.get_prompt_len()
        if self.policy == "orca-max":
            tokens = self.max_model_len
        else:
            if group.true_output_len is None:
                raise ValueError(f"{self.policy} needs the request's true output length (the oracle)")
            out = group.true_output_len
            tokens = prompt + (out if self.policy == "orca-oracle" else next_pow2(out))
        return min(max(tokens, seq.get_len()), self.max_model_len)

    def _order_for(self, group: SequenceGroup) -> int:
        n_blocks = math.ceil(self.reserved_tokens(group) / self.block_size)
        return (next_pow2(n_blocks)).bit_length() - 1

    # --------------------------------------------------------------- admission
    def can_allocate(self, group: SequenceGroup) -> AllocStatus:
        order = self._order_for(group)
        n = group.num_seqs(SequenceStatus.WAITING)
        if order > self.buddy.max_order() or n * (1 << order) > self.num_total_gpu_blocks:
            return AllocStatus.NEVER
        # Try for real and roll back: the buddy layout decides, not just the free count.
        starts = []
        for _ in range(n):
            s = self.buddy.alloc(order)
            if s is None:
                break
            starts.append(s)
        ok = len(starts) == n
        for s in starts:
            self.buddy.free(s)
        if ok:
            return AllocStatus.OK
        if self.buddy.num_free_blocks() >= n * (1 << order):
            self.num_external_frag_events += 1
        return AllocStatus.LATER

    def allocate(self, group: SequenceGroup) -> None:
        order = self._order_for(group)
        reserved = self.reserved_tokens(group)
        for seq in group.get_seqs(SequenceStatus.WAITING):
            # How far this sequence will actually grow (for the waste breakdown only).
            final_len = reserved
            if group.true_output_len is not None:
                final_len = min(reserved, seq.get_prompt_len() + group.true_output_len)
            start = self.buddy.alloc(order)
            assert start is not None, "allocate() called without a successful can_allocate()"
            table = self.blocks[start : start + (1 << order)]
            for b in table:
                b.ref_count = 1
            self.block_tables[seq.seq_id] = table
            self.reservations[seq.seq_id] = (start, 1 << order, final_len)
            seq.num_computed_tokens = 0

    # ---------------------------------------------------------------- decoding
    def can_append_slots(self, group: SequenceGroup) -> bool:
        return True  # the space was reserved up front; growing never needs new memory

    def append_slot(self, seq: Sequence) -> Optional[tuple[int, int]]:
        _, n_blocks, _ = self.reservations[seq.seq_id]
        if seq.get_len() > n_blocks * self.block_size:
            raise RuntimeError(f"{seq} outgrew its {self.policy} reservation")
        return None

    def fork(self, parent: Sequence, child: Sequence) -> None:
        raise NotImplementedError("Orca baselines do not share memory between sequences")

    def free(self, seq: Sequence) -> None:
        table = self.block_tables.pop(seq.seq_id, None)
        if table is None:
            return
        start, _, _ = self.reservations.pop(seq.seq_id)
        for b in table:
            b.ref_count = 0
        self.buddy.free(start)

    # Orca never preempts (reservations cover the whole sequence), so swapping is unused.
    def can_swap_in(self, group: SequenceGroup) -> bool:
        return False

    def can_swap_out(self, group: SequenceGroup) -> bool:
        return False

    def swap_in(self, group: SequenceGroup) -> dict[int, int]:
        raise NotImplementedError

    def swap_out(self, group: SequenceGroup) -> dict[int, int]:
        raise NotImplementedError

    def register_prefix(self, prefix_id: str, token_ids: list[int]):
        raise NotImplementedError("Orca baselines have no prefix sharing")

    # ------------------------------------------------------------------ queries
    def get_block_table(self, seq: Sequence) -> list[int]:
        return [b.block_number for b in self.block_tables[seq.seq_id]]

    def get_num_free_gpu_blocks(self) -> int:
        return self.buddy.num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.num_total_cpu_blocks

    def memory_stats(self, seqs: Iterable[Sequence]) -> dict[str, int]:
        used = future = allocated = 0
        for seq in seqs:
            if seq.seq_id not in self.reservations:
                continue
            _, n_blocks, final_len = self.reservations[seq.seq_id]
            allocated += n_blocks * self.block_size
            used += seq.get_len()
            future += max(0, final_len - seq.get_len())  # will be filled later; the rest is internal fragmentation
        used_blocks = self.num_total_gpu_blocks - self.buddy.num_free_blocks()
        return {
            "allocated_slots": used_blocks * self.block_size,
            "used_slots": used,
            "reserved_future_slots": future,
            "used_blocks": used_blocks,
            "free_blocks": self.buddy.num_free_blocks(),
        }

    def check_invariants(self, seqs: Optional[Iterable[Sequence]] = None) -> None:
        covered = [0] * self.num_total_gpu_blocks
        for o, starts in self.buddy.free_lists.items():
            for s in starts:
                assert s % (1 << o) == 0, "misaligned free range"
                for i in range(s, s + (1 << o)):
                    covered[i] += 1
        for s, o in self.buddy.allocated.items():
            for i in range(s, s + (1 << o)):
                covered[i] += 1
        assert all(c == 1 for c in covered), "buddy ranges overlap or leave holes"
        assert set(self.buddy.allocated) == {r[0] for r in self.reservations.values()}
