"""LLMEngine: the step loop that ties everything together (§4.3).

Each step():
  1. Scheduler decides which requests run, and which blocks to swap / copy (integers only).
  2. CacheEngine performs the data movement: swap-out first (a freed block may be reused
     in the same step), then swap-in, then copy-on-write copies.
  3. The batch is flattened: prefill requests contribute all their uncomputed tokens,
     decode sequences contribute one token each. Each token gets a KV slot from its
     block table.
  4. The model runs once over the whole batch (paged attention inside).
  5. Each sequence samples its next token; finished sequences free their blocks.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Optional, Union

import torch

from minivllm.memory import make_block_manager
from minivllm.cache_engine import CacheEngine
from minivllm.config import CacheConfig, ModelSpec, SchedulerConfig
from minivllm.model.opt import InputMetadata, PagedOPT
from minivllm.sampler import sample
from minivllm.scheduler import Scheduler
from minivllm.sequence import SamplingParams, Sequence, SequenceGroup, SequenceStatus


@dataclass
class RequestOutput:
    request_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[list[int]]          # one list per sequence (n for parallel sampling)
    texts: Optional[list[str]] = None
    aborted: bool = False
    arrival_time: float = 0.0
    first_scheduled_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    num_preemptions: int = 0
    extra: dict = field(default_factory=dict)


class LLMEngine:
    def __init__(
        self,
        hf_model: torch.nn.Module,
        tokenizer=None,
        *,
        block_size: int = 16,
        kv_cache_gb: float = 1.0,
        swap_space_gb: float = 1.0,
        num_gpu_blocks: Optional[int] = None,   # overrides kv_cache_gb (tests use tiny pools)
        num_cpu_blocks: Optional[int] = None,   # overrides swap_space_gb
        scheduler_config: Optional[SchedulerConfig] = None,
        watermark: float = 0.01,
        memory_manager: str = "paged",          # "paged" | "orca-oracle" | "orca-pow2" | "orca-max"
        seed: int = 0,
        check_invariants: bool = False,
    ) -> None:
        self.model = PagedOPT(hf_model).eval()
        self.tokenizer = tokenizer
        param = next(hf_model.parameters())
        self.device, self.dtype = param.device, param.dtype
        cfg = hf_model.config

        self.spec = ModelSpec(
            name=getattr(cfg, "_name_or_path", "opt"),
            num_layers=self.model.num_layers,
            num_kv_heads=self.model.num_kv_heads,
            head_dim=self.model.head_dim,
            max_model_len=cfg.max_position_embeddings,
            dtype_bytes=param.element_size(),
        )
        cache_cfg = CacheConfig(block_size=block_size, kv_cache_gb=kv_cache_gb, swap_space_gb=swap_space_gb)
        n_gpu = num_gpu_blocks if num_gpu_blocks is not None else cache_cfg.num_gpu_blocks(self.spec)
        n_cpu = num_cpu_blocks if num_cpu_blocks is not None else cache_cfg.num_cpu_blocks(self.spec)
        self.block_size = block_size

        if scheduler_config is None:
            max_len = cfg.max_position_embeddings
            scheduler_config = SchedulerConfig(max_model_len=max_len, max_num_batched_tokens=max(2048, max_len))
        self.memory_manager = memory_manager
        self.block_manager = make_block_manager(memory_manager, block_size, n_gpu, n_cpu, watermark,
                                                scheduler_config.max_model_len)
        self.scheduler = Scheduler(scheduler_config, self.block_manager)
        self.cache = CacheEngine(
            self.spec.num_layers, self.spec.num_kv_heads, self.spec.head_dim,
            block_size, n_gpu, n_cpu, self.dtype, self.device,
        )
        self.eos_token_id = tokenizer.eos_token_id if tokenizer is not None else cfg.eos_token_id
        self.seed = seed
        self.check_invariants = check_invariants
        self._seq_counter = itertools.count()
        self._generators: dict[str, torch.Generator] = {}
        self._groups: dict[str, SequenceGroup] = {}
        self._t0 = time.perf_counter()
        self.num_steps = 0
        self.num_prefill_tokens = 0  # tokens computed in prefill (shows prefix reuse / recompute cost)

    # ---------------------------------------------------------------- requests
    def now(self) -> float:
        return time.perf_counter() - self._t0

    def add_request(
        self,
        request_id: str,
        prompt: Union[str, list[int]],
        params: SamplingParams,
        arrival_time: Optional[float] = None,
        true_output_len: Optional[int] = None,
        prefix_id: Optional[str] = None,
    ) -> None:
        if params.use_beam_search:
            raise NotImplementedError("beam search is not implemented yet (planned, §4.4)")
        tokens = self._encode(prompt)
        seqs = [Sequence(next(self._seq_counter), tokens, self.block_size) for _ in range(params.n)]
        group = SequenceGroup(
            request_id, seqs, params,
            arrival_time=self.now() if arrival_time is None else arrival_time,
            true_output_len=true_output_len, prefix_id=prefix_id,
        )
        gen = torch.Generator().manual_seed(self.seed if params.seed is None else params.seed)
        self._generators[request_id] = gen
        self._groups[request_id] = group
        self.scheduler.add_seq_group(group)

    def register_prefix(self, prefix_id: str, prefix: Union[str, list[int]]) -> list[int]:
        """Register a shared prefix (§4.4). Prompts using it must start with exactly these tokens."""
        tokens = self._encode(prefix)
        self.block_manager.register_prefix(prefix_id, tokens)
        return tokens

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    # ---------------------------------------------------------------- one step
    def step(self, now: Optional[float] = None) -> list[RequestOutput]:
        now = self.now() if now is None else now
        bm, B = self.block_manager, self.block_size
        out = self.scheduler.schedule(now)
        finished: list[RequestOutput] = [self._finish(g, now, aborted=True) for g in out.ignored]

        # Data movement. Order matters: blocks freed by swap-out can be reused this step.
        self.cache.swap_out(out.blocks_to_swap_out)
        self.cache.swap_in(out.blocks_to_swap_in)
        self.cache.copy(out.blocks_to_copy)

        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        tables: list[list[int]] = []
        ctx_lens: list[int] = []
        q_lens: list[int] = []
        targets: list[tuple[SequenceGroup, list[Sequence]]] = []

        def add_entry(seq: Sequence, start: int, sample_for: list[Sequence], group: SequenceGroup) -> None:
            table = bm.get_block_table(seq)
            end = seq.get_len()
            input_ids.extend(seq.get_token_ids()[start:end])
            for p in range(start, end):
                positions.append(p)
                slots.append(table[p // B] * B + p % B)
            tables.append(table)
            ctx_lens.append(end)
            q_lens.append(end - start)
            targets.append((group, sample_for))

        for group in out.prefill_groups:
            running = group.get_seqs(SequenceStatus.RUNNING)
            if bm.shares_prompt:
                # All sequences of a new group share the prompt blocks: compute the prompt once,
                # then sample one token per sequence from the same logits (parallel sampling).
                first = running[0]
                add_entry(first, first.num_computed_tokens, running, group)
                self.num_prefill_tokens += q_lens[-1]
            else:
                # Orca: every sequence has a private copy, so the prompt is computed n times.
                for seq in running:
                    add_entry(seq, seq.num_computed_tokens, [seq], group)
                    self.num_prefill_tokens += q_lens[-1]
        for group in out.decode_groups:
            for seq in group.get_seqs(SequenceStatus.RUNNING):
                assert seq.get_num_uncomputed_tokens() == 1, f"{seq} is not ready to decode"
                add_entry(seq, seq.get_len() - 1, [seq], group)

        if targets:
            dev = self.device
            meta = InputMetadata(
                positions=torch.tensor(positions, device=dev),
                slot_mapping=torch.tensor(slots, device=dev),
                block_tables=tables,
                context_lens=ctx_lens,
                query_lens=q_lens,
            )
            logits = self.model(torch.tensor(input_ids, device=dev), self.cache.gpu_cache, meta)
            self._process(targets, logits, now)
            self.num_steps += 1

        for group in self.scheduler.free_finished_seq_groups():
            finished.append(self._finish(group, now))
        if self.check_invariants:
            bm.check_invariants(s for g in self._groups.values() for s in g.seqs)
        return finished

    def _process(self, targets, logits: torch.Tensor, now: float) -> None:
        max_len = self.scheduler.config.max_model_len
        for (group, seqs), row in zip(targets, logits):
            params = group.sampling_params
            gen = self._generators[group.request_id]
            for seq in seqs:
                seq.num_computed_tokens = seq.get_len()
                token, logprob = sample(row, params, gen)
                seq.append_token(token, logprob)
                if not params.ignore_eos and token == self.eos_token_id:
                    seq.status = SequenceStatus.FINISHED_STOPPED
                elif seq.get_output_len() >= params.max_tokens or seq.get_len() >= max_len:
                    seq.status = SequenceStatus.FINISHED_LENGTH
                if seq.is_finished():
                    self.scheduler.free_seq(seq)
            if group.first_token_time is None:
                group.first_token_time = now
            if group.prefix_id is not None:
                prefix = self.block_manager.prefixes[group.prefix_id]
                if not prefix.computed:
                    self.block_manager.mark_prefix_computed(group.prefix_id)

    def _finish(self, group: SequenceGroup, now: float, aborted: bool = False) -> RequestOutput:
        group.finish_time = now
        self._groups.pop(group.request_id, None)
        self._generators.pop(group.request_id, None)
        outs = [s.output_token_ids for s in group.seqs]
        texts = None
        if self.tokenizer is not None:
            texts = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outs]
        return RequestOutput(
            request_id=group.request_id,
            prompt_token_ids=group.seqs[0].prompt_token_ids,
            output_token_ids=outs,
            texts=texts,
            aborted=aborted,
            arrival_time=group.arrival_time,
            first_scheduled_time=group.first_scheduled_time,
            first_token_time=group.first_token_time,
            finish_time=now,
            num_preemptions=group.num_preemptions,
            extra={"logprobs": [s.output_logprobs for s in group.seqs]},
        )

    # ---------------------------------------------------------------- helpers
    def generate(self, prompts: list[Union[str, list[int]]], params: SamplingParams) -> list[RequestOutput]:
        """Offline convenience: run all prompts to completion, return outputs in prompt order."""
        ids = [f"gen-{next(self._seq_counter)}" for _ in prompts]
        for rid, prompt in zip(ids, prompts):
            # With ignore_eos every output is exactly max_tokens long (what Orca-Oracle 'knows').
            self.add_request(rid, prompt, params, true_output_len=params.max_tokens)
        done: dict[str, RequestOutput] = {}
        while self.has_unfinished():
            for o in self.step():
                done[o.request_id] = o
        return [done[r] for r in ids]

    def memory_stats(self) -> dict[str, int]:
        live = [s for g in self._groups.values() for s in g.seqs]
        return self.block_manager.memory_stats(live)

    def _encode(self, prompt: Union[str, list[int]]) -> list[int]:
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("pass token ids, or give the engine a tokenizer")
            return self.tokenizer(prompt).input_ids
        return list(prompt)
