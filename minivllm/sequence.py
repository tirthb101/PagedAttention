"""Requests and sequences.

A *request* (SequenceGroup) holds one or more *sequences* that share the same prompt:
one for normal sampling, n for parallel sampling or beam search (§4.4). Sequences only
hold token ids; where their KV cache lives is the BlockSpaceManager's business.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Optional


class SequenceStatus(enum.Enum):
    WAITING = enum.auto()          # not yet admitted, or preempted by recomputation
    RUNNING = enum.auto()          # has its KV cache on the GPU
    SWAPPED = enum.auto()          # preempted; KV cache in CPU memory
    FINISHED_STOPPED = enum.auto() # produced EOS
    FINISHED_LENGTH = enum.auto()  # hit max_tokens or the model's context limit
    FINISHED_ABORTED = enum.auto() # rejected or cancelled (e.g. prompt too long)

    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status in (
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH,
            SequenceStatus.FINISHED_ABORTED,
        )


@dataclass
class SamplingParams:
    n: int = 1                     # parallel samples, or beam width when use_beam_search
    use_beam_search: bool = False
    temperature: float = 1.0       # 0 = greedy
    top_p: float = 1.0
    max_tokens: int = 128          # max generated tokens per sequence
    ignore_eos: bool = False       # benchmarks force exact output lengths from the trace
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")


class Sequence:
    """One stream of tokens: the shared prompt plus this sequence's own outputs."""

    def __init__(self, seq_id: int, prompt_token_ids: list[int], block_size: int) -> None:
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        self.seq_id = seq_id
        self.block_size = block_size
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: list[int] = []
        self.output_logprobs: list[float] = []
        self.cumulative_logprob = 0.0
        self.status = SequenceStatus.WAITING
        # Tokens whose K/V are already in the cache. The next model step must process
        # tokens [num_computed_tokens, len). Prefill: the whole prompt; decode: 1 token;
        # after recompute-preemption: everything again (§4.5).
        self.num_computed_tokens = 0

    # ---- token data -------------------------------------------------------------
    def get_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def get_prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    def get_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_last_token_id(self) -> int:
        return self.output_token_ids[-1] if self.output_token_ids else self.prompt_token_ids[-1]

    def append_token(self, token_id: int, logprob: float = 0.0) -> None:
        self.output_token_ids.append(token_id)
        self.output_logprobs.append(logprob)
        self.cumulative_logprob += logprob

    def get_num_uncomputed_tokens(self) -> int:
        return self.get_len() - self.num_computed_tokens

    # ---- blocks -----------------------------------------------------------------
    @property
    def num_logical_blocks(self) -> int:
        """Blocks needed to hold every token of this sequence: ceil(len / B)."""
        return math.ceil(self.get_len() / self.block_size)

    # ---- lifecycle --------------------------------------------------------------
    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)

    def reset_for_recompute(self) -> None:
        """Called when preempted by recomputation: the KV cache is gone, the tokens are kept.

        On re-admission, prompt + generated tokens are prefilled together as one prompt.
        """
        self.status = SequenceStatus.WAITING
        self.num_computed_tokens = 0

    def fork(self, new_seq_id: int) -> "Sequence":
        """Copy the token history (not the blocks — BlockSpaceManager.fork shares those)."""
        child = Sequence.__new__(Sequence)
        child.seq_id = new_seq_id
        child.block_size = self.block_size
        child.prompt_token_ids = self.prompt_token_ids  # never mutated, safe to share
        child.output_token_ids = list(self.output_token_ids)
        child.output_logprobs = list(self.output_logprobs)
        child.cumulative_logprob = self.cumulative_logprob
        child.status = self.status
        child.num_computed_tokens = self.num_computed_tokens
        return child

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, status={self.status.name}, "
            f"len={self.get_len()}, computed={self.num_computed_tokens})"
        )


class SequenceGroup:
    """One user request. All its sequences start from the same prompt."""

    def __init__(
        self,
        request_id: str,
        seqs: list[Sequence],
        sampling_params: SamplingParams,
        arrival_time: float,
        true_output_len: Optional[int] = None,
        prefix_id: Optional[str] = None,
    ) -> None:
        if not seqs:
            raise ValueError("a SequenceGroup needs at least one sequence")
        self.request_id = request_id
        self.seqs = list(seqs)
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time
        # Only the Orca (Oracle) baseline may look at this: it "knows" the output length.
        self.true_output_len = true_output_len
        # Optional shared prefix registered with the block manager (§4.4).
        self.prefix_id = prefix_id
        # Timestamps for latency metrics.
        self.first_scheduled_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.finish_time: Optional[float] = None
        self.num_preemptions = 0

    def get_seqs(self, status: Optional[SequenceStatus] = None) -> list[Sequence]:
        if status is None:
            return list(self.seqs)
        return [s for s in self.seqs if s.status == status]

    def get_unfinished_seqs(self) -> list[Sequence]:
        return [s for s in self.seqs if not s.is_finished()]

    def num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        return len(self.get_seqs(status))

    def get_max_num_running_seqs(self) -> int:
        """How many sequences this group will run at once (for batch-size limits)."""
        return len(self.get_unfinished_seqs())

    def find(self, seq_id: int) -> Sequence:
        for s in self.seqs:
            if s.seq_id == seq_id:
                return s
        raise KeyError(seq_id)

    def add(self, seq: Sequence) -> None:
        self.seqs.append(seq)

    def remove(self, seq_id: int) -> None:
        self.seqs = [s for s in self.seqs if s.seq_id != seq_id]

    def is_finished(self) -> bool:
        return all(s.is_finished() for s in self.seqs)

    def __repr__(self) -> str:
        return f"SequenceGroup(id={self.request_id}, seqs={self.seqs})"
