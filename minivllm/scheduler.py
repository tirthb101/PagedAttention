"""Iteration-level scheduler with preemption (§4.5).

Policy, as in the paper:
  * first-come-first-serve: older requests have priority;
  * when the GPU runs out of blocks, preempt the *latest-arrived* running requests;
  * all-or-nothing: a preempted request gives up all of its blocks;
  * gang scheduling: all sequences of a request (beams, parallel samples) move together;
  * while any request is swapped out, no new requests are admitted.

Each call to schedule() decides one model step. Our batches mix prefill (new or
recomputed requests) and decode (one token per running sequence), as in §4.3.
The scheduler only moves integers around; the engine performs the actual work.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from minivllm.block_manager import AllocStatus, BlockSpaceManager
from minivllm.config import PreemptionMode, SchedulerConfig
from minivllm.sequence import SequenceGroup, SequenceStatus


@dataclass
class SchedulerOutputs:
    prefill_groups: list[SequenceGroup] = field(default_factory=list)  # process all uncomputed tokens
    decode_groups: list[SequenceGroup] = field(default_factory=list)   # process one token per running seq
    num_batched_tokens: int = 0
    blocks_to_swap_in: dict[int, int] = field(default_factory=dict)    # cpu block -> gpu block
    blocks_to_swap_out: dict[int, int] = field(default_factory=dict)   # gpu block -> cpu block
    blocks_to_copy: list[tuple[int, int]] = field(default_factory=list)  # copy-on-write (src, dst)
    preempted: list[SequenceGroup] = field(default_factory=list)
    ignored: list[SequenceGroup] = field(default_factory=list)         # rejected (too long)

    def is_empty(self) -> bool:
        return not (self.prefill_groups or self.decode_groups or self.blocks_to_swap_in or self.blocks_to_swap_out)

    @property
    def scheduled_groups(self) -> list[SequenceGroup]:
        return self.prefill_groups + self.decode_groups


class Scheduler:
    def __init__(self, config: SchedulerConfig, block_manager: BlockSpaceManager) -> None:
        self.config = config
        self.block_manager = block_manager
        self.waiting: deque[SequenceGroup] = deque()
        self.running: list[SequenceGroup] = []
        self.swapped: list[SequenceGroup] = []
        self.num_preemptions = 0

    # --------------------------------------------------------------- requests
    def add_seq_group(self, group: SequenceGroup) -> None:
        self.waiting.append(group)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running or self.swapped)

    def get_num_unfinished(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    # --------------------------------------------------------------- schedule
    def schedule(self, now: float = 0.0) -> SchedulerOutputs:
        out = SchedulerOutputs()
        bm = self.block_manager

        # 1. Reserve a slot for every running sequence's next token, oldest request first.
        #    If memory runs out, preempt from the youngest end.
        queue = sorted(self.running, key=lambda g: g.arrival_time)
        still_running: list[SequenceGroup] = []
        while queue:
            group = queue.pop(0)
            while not bm.can_append_slots(group):
                victim = queue.pop() if queue else group
                self._preempt(victim, out)
                if victim is group:
                    break
            else:
                for seq in group.get_seqs(SequenceStatus.RUNNING):
                    cow = bm.append_slot(seq)
                    if cow is not None:
                        out.blocks_to_copy.append(cow)
                still_running.append(group)
        self.running = still_running

        num_seqs = sum(g.get_max_num_running_seqs() for g in self.running)
        num_tokens = sum(g.num_seqs(SequenceStatus.RUNNING) for g in self.running)

        # 2. Swap preempted requests back in (oldest first), but not in a step that just preempted.
        if not out.preempted:
            self.swapped.sort(key=lambda g: g.arrival_time)
            while self.swapped:
                group = self.swapped[0]
                n = group.get_max_num_running_seqs()
                if num_seqs + n > self.config.max_num_seqs or num_tokens + n > self.config.max_num_batched_tokens:
                    break
                if not bm.can_swap_in(group):
                    if not self.running:
                        # Nothing else holds GPU memory, so this request can never come back
                        # (its forked sequences outgrew the whole pool). Abort it instead of
                        # deadlocking the queue.
                        self.swapped.pop(0)
                        self._abort_swapped(group, out)
                        continue
                    break
                self.swapped.pop(0)
                out.blocks_to_swap_in.update(bm.swap_in(group))
                for seq in group.get_seqs(SequenceStatus.SWAPPED):
                    seq.status = SequenceStatus.RUNNING
                    cow = bm.append_slot(seq)
                    if cow is not None:
                        out.blocks_to_copy.append(cow)
                self.running.append(group)
                num_seqs += n
                num_tokens += n

        out.decode_groups = list(self.running)

        # 3. Admit new (or recompute-preempted) requests in strict FCFS order.
        #    Not while anything is swapped out (paper rule), and not in a step that
        #    preempted (re-admitting a victim immediately would just thrash).
        if not self.swapped and not out.preempted:
            while self.waiting:
                group = self.waiting[0]
                seq = group.get_seqs(SequenceStatus.WAITING)[0]
                if seq.get_len() > self.config.max_model_len:
                    self._ignore(group, out)
                    continue
                status = bm.can_allocate(group)
                if status == AllocStatus.NEVER:
                    self._ignore(group, out)
                    continue
                n = group.get_max_num_running_seqs()
                # Prefill tokens: the prompt is computed once even if n sequences share it.
                new_tokens = seq.get_num_uncomputed_tokens()
                if num_tokens + new_tokens > self.config.max_num_batched_tokens:
                    break
                if num_seqs + n > self.config.max_num_seqs:
                    break
                if status == AllocStatus.LATER:
                    break  # strict FCFS: later requests may not overtake this one
                self.waiting.popleft()
                bm.allocate(group)
                for s in group.get_seqs(SequenceStatus.WAITING):
                    s.status = SequenceStatus.RUNNING
                if group.first_scheduled_time is None:
                    group.first_scheduled_time = now
                self.running.append(group)
                out.prefill_groups.append(group)
                num_seqs += n
                num_tokens += seq.get_num_uncomputed_tokens()  # after allocate: prefix may be cached

        out.num_batched_tokens = num_tokens
        return out

    # ------------------------------------------------------------- preemption
    def _preempt(self, group: SequenceGroup, out: SchedulerOutputs) -> None:
        mode = self.config.preemption_mode
        # Groups with several sequences share blocks; recomputing them would lose the
        # sharing, so they are always swapped (vLLM makes the same choice).
        if group.get_max_num_running_seqs() > 1:
            mode = PreemptionMode.SWAP

        if mode == PreemptionMode.RECOMPUTE:
            for seq in group.get_seqs(SequenceStatus.RUNNING):
                self.block_manager.free(seq)
                seq.reset_for_recompute()
            self.waiting.appendleft(group)  # oldest victims end up at the very front
        else:
            if not self.block_manager.can_swap_out(group):
                raise RuntimeError(
                    "CPU swap space exhausted; increase swap_space_gb or use recompute preemption"
                )
            out.blocks_to_swap_out.update(self.block_manager.swap_out(group))
            for seq in group.get_seqs(SequenceStatus.RUNNING):
                seq.status = SequenceStatus.SWAPPED
            self.swapped.append(group)

        group.num_preemptions += 1
        self.num_preemptions += 1
        out.preempted.append(group)

    def _abort_swapped(self, group: SequenceGroup, out: SchedulerOutputs) -> None:
        for seq in group.get_unfinished_seqs():
            self.block_manager.free(seq)
            seq.status = SequenceStatus.FINISHED_ABORTED
        out.ignored.append(group)

    def _ignore(self, group: SequenceGroup, out: SchedulerOutputs) -> None:
        self.waiting.popleft()
        for seq in group.get_seqs():
            seq.status = SequenceStatus.FINISHED_ABORTED
        out.ignored.append(group)

    # ---------------------------------------------------------------- cleanup
    def free_seq(self, seq) -> None:
        """Release a finished (or pruned) sequence's blocks."""
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> list[SequenceGroup]:
        finished = [g for g in self.running if g.is_finished()]
        self.running = [g for g in self.running if not g.is_finished()]
        return finished
