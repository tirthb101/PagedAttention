"""PagedAttention in plain PyTorch (§4.1) — the correctness reference.

KV cache layout, per layer:   key_cache[num_blocks, block_size, num_kv_heads, head_dim]
                              value_cache: same shape
A token at position i of a sequence lives in physical block  table[i // B]  at offset  i % B,
i.e. at flat slot  table[i // B] * B + i % B.

This version gathers each sequence's blocks with tensor indexing and calls PyTorch's
built-in attention. It is slow (a Python loop over sequences) but obviously correct,
which is exactly what we want from a reference. The fast Triton kernel (§5) is checked
against it.
"""

from __future__ import annotations

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


def paged_attention(
    query: torch.Tensor,             # [num_tokens, num_heads, head_dim], all sequences concatenated
    key_cache: torch.Tensor,         # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,
    block_tables: list[list[int]],   # per sequence: physical block numbers
    context_lens: list[int],         # per sequence: tokens to attend to (including the new ones)
    query_lens: list[int],           # per sequence: new tokens in this step (prefill: many, decode: 1)
    scale: float,
) -> torch.Tensor:
    """Causal attention where each sequence reads its K/V through its block table."""
    block_size = key_cache.shape[1]
    num_heads, num_kv_heads = query.shape[1], key_cache.shape[2]
    out = torch.empty_like(query)
    offset = 0
    for table, ctx, qlen in zip(block_tables, context_lens, query_lens):
        n_blocks = (ctx + block_size - 1) // block_size
        idx = torch.as_tensor(table[:n_blocks], device=key_cache.device, dtype=torch.long)
        # Gather this sequence's blocks and flatten them back into one contiguous history.
        k = key_cache[idx].reshape(-1, num_kv_heads, key_cache.shape[3])[:ctx]
        v = value_cache[idx].reshape(-1, num_kv_heads, value_cache.shape[3])[:ctx]
        if num_kv_heads != num_heads:  # grouped-query attention: share each KV head across queries
            k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
            v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
        q = query[offset : offset + qlen]

        # Query j sits at absolute position ctx - qlen + j and may see keys 0..that position.
        q_pos = torch.arange(ctx - qlen, ctx, device=query.device)
        k_pos = torch.arange(ctx, device=query.device)
        mask = k_pos[None, :] <= q_pos[:, None]          # [qlen, ctx]

        o = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),  # [heads, len, dim]
            attn_mask=mask, scale=scale,
        )
        out[offset : offset + qlen] = o.transpose(0, 1)
        offset += qlen
    assert offset == query.shape[0], "query_lens must cover every token"
    return out
