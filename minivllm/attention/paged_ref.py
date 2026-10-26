"""PagedAttention in plain PyTorch (§4.1) — the correctness reference.

KV cache layout, per layer:   key_cache[num_blocks, block_size, num_kv_heads, head_dim]
                              value_cache: same shape
A token at position i of a sequence lives in physical block  table[i // B]  at offset  i % B,
i.e. at flat slot  table[i // B] * B + i % B.

Two paths:
  * prefill sequences (many new tokens): gather the sequence's blocks into a contiguous
    history and run causal attention, one sequence at a time;
  * decode sequences (one new token each): gather ALL of them at once into a padded
    [num_seqs, max_len] history and run a single batched attention call.

The block lookups for a step are computed once (plan_paged) and reused by every layer.
This is still a gather-then-attend reference: it copies K/V out of the blocks. The Triton
kernel (§5) reads the blocks in place instead and is checked against this code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def write_kv(
    key: torch.Tensor,          # [num_tokens, num_kv_heads, head_dim]
    value: torch.Tensor,        # [num_tokens, num_kv_heads, head_dim]
    key_cache: torch.Tensor,    # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor, # [num_tokens] flat slot index for each token
) -> None:
    """Scatter new tokens' K/V into their blocks (the paper's fused reshape-and-write)."""
    num_blocks, block_size, h, d = key_cache.shape
    key_cache.view(num_blocks * block_size, h, d).index_copy_(0, slot_mapping, key)
    value_cache.view(num_blocks * block_size, h, d).index_copy_(0, slot_mapping, value)


@dataclass
class PagedPlan:
    """Per-step lookup tables, shared by all layers."""

    prefill: list[tuple[int, int, int, torch.Tensor, torch.Tensor]]  # (offset, qlen, ctx, block idx, causal mask)
    decode_offsets: Optional[torch.Tensor]   # [S] positions of decode tokens in the flat batch
    decode_blocks: Optional[torch.Tensor]    # [S, max_blocks] padded block tables
    decode_mask: Optional[torch.Tensor]      # [S, 1, 1, max_blocks * block_size] valid keys
    num_tokens: int


def plan_paged(block_tables: list[list[int]], context_lens: list[int], query_lens: list[int],
               block_size: int, device: torch.device) -> PagedPlan:
    prefill, dec_off, dec_tables, dec_ctx = [], [], [], []
    offset = 0
    for table, ctx, qlen in zip(block_tables, context_lens, query_lens):
        n_blocks = (ctx + block_size - 1) // block_size
        if qlen == 1:
            dec_off.append(offset)
            dec_tables.append(table[:n_blocks])
            dec_ctx.append(ctx)
        else:
            idx = torch.tensor(table[:n_blocks], device=device, dtype=torch.long)
            q_pos = torch.arange(ctx - qlen, ctx, device=device)
            mask = torch.arange(ctx, device=device)[None, :] <= q_pos[:, None]   # [qlen, ctx]
            prefill.append((offset, qlen, ctx, idx, mask))
        offset += qlen

    if not dec_off:
        return PagedPlan(prefill, None, None, None, offset)
    max_blocks = max(len(t) for t in dec_tables)
    padded = [t + [t[0]] * (max_blocks - len(t)) for t in dec_tables]  # pad with a valid block; masked out
    blocks = torch.tensor(padded, device=device, dtype=torch.long)
    ctx_t = torch.tensor(dec_ctx, device=device)
    mask = torch.arange(max_blocks * block_size, device=device)[None, :] < ctx_t[:, None]
    return PagedPlan(prefill, torch.tensor(dec_off, device=device), blocks, mask[:, None, None, :], offset)


def paged_attention(
    query: torch.Tensor,             # [num_tokens, num_heads, head_dim], all sequences concatenated
    key_cache: torch.Tensor,         # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,
    block_tables: list[list[int]],   # per sequence: physical block numbers
    context_lens: list[int],         # per sequence: tokens to attend to (including the new ones)
    query_lens: list[int],           # per sequence: new tokens in this step (prefill: many, decode: 1)
    scale: float,
    plan: Optional[PagedPlan] = None,
) -> torch.Tensor:
    """Causal attention where each sequence reads its K/V through its block table."""
    block_size, num_kv_heads, head_dim = key_cache.shape[1], key_cache.shape[2], key_cache.shape[3]
    num_heads = query.shape[1]
    rep = num_heads // num_kv_heads
    if plan is None:
        plan = plan_paged(block_tables, context_lens, query_lens, block_size, query.device)
    assert plan.num_tokens == query.shape[0], "query_lens must cover every token"
    out = torch.empty_like(query)

    for offset, qlen, ctx, idx, mask in plan.prefill:
        k = key_cache[idx].reshape(-1, num_kv_heads, head_dim)[:ctx]
        v = value_cache[idx].reshape(-1, num_kv_heads, head_dim)[:ctx]
        if rep > 1:  # grouped-query attention: share each KV head across several query heads
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        q = query[offset : offset + qlen]
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask, scale=scale)
        out[offset : offset + qlen] = o.transpose(0, 1)

    if plan.decode_offsets is not None:
        S = plan.decode_blocks.shape[0]
        k = key_cache[plan.decode_blocks].reshape(S, -1, num_kv_heads, head_dim)   # [S, L, Hkv, D]
        v = value_cache[plan.decode_blocks].reshape(S, -1, num_kv_heads, head_dim)
        if rep > 1:
            k, v = k.repeat_interleave(rep, dim=2), v.repeat_interleave(rep, dim=2)
        q = query[plan.decode_offsets]                                              # [S, H, D]
        o = F.scaled_dot_product_attention(
            q[:, :, None, :], k.transpose(1, 2), v.transpose(1, 2), attn_mask=plan.decode_mask, scale=scale)
        out[plan.decode_offsets] = o[:, :, 0, :]
    return out
