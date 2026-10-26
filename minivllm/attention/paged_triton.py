"""PagedAttention decode kernel in Triton (§4.1, §5 "fused block read and attention").

One program instance handles one (sequence, head) pair. It walks the sequence's block
table, loads one KV block at a time straight from the paged pool (no gather copy), and
keeps a running ("online") softmax so the whole history never has to sit in memory at
once — the same structure as vLLM's CUDA kernel, which assigns a warp per block.

Decode only (one new token per sequence): that is the step that dominates serving time
and the one the paper's kernel targets. Prefill keeps using the PyTorch reference path.

Testing without a GPU: set TRITON_INTERPRET=1 before importing this module and the
kernel runs (slowly) on the CPU through Triton's interpreter.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    Q, K, V, BLOCK_TABLES, CONTEXT_LENS, OUT,
    scale,
    stride_qs, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_vb, stride_vt, stride_vh,
    stride_bts,
    stride_os, stride_oh,
    REP: tl.constexpr,          # query heads per KV head (1 = multi-head, >1 = grouped-query)
    BLOCK_SIZE: tl.constexpr,   # tokens per KV block
    HEAD_DIM: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // REP

    ctx = tl.load(CONTEXT_LENS + seq)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_t = tl.arange(0, BLOCK_SIZE)

    q = tl.load(Q + seq * stride_qs + head * stride_qh + offs_d).to(tl.float32) * scale

    m_i = tl.zeros([], dtype=tl.float32) - float("inf")   # running max of scores
    l_i = tl.zeros([], dtype=tl.float32)                   # running sum of exp(scores - m_i)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)           # running weighted sum of values

    num_blocks = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(0, num_blocks):
        block = tl.load(BLOCK_TABLES + seq * stride_bts + i).to(tl.int64)   # page-table lookup
        pos = i * BLOCK_SIZE + offs_t
        valid = pos < ctx                                                   # last block may be partial

        k_ptrs = K + block * stride_kb + offs_t[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :]
        k = tl.load(k_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)  # [BLOCK_SIZE, HEAD_DIM]
        scores = tl.sum(k * q[None, :], axis=1)                             # [BLOCK_SIZE]
        scores = tl.where(valid, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        v_ptrs = V + block * stride_vb + offs_t[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :]
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(OUT + seq * stride_os + head * stride_oh + offs_d, out.to(OUT.dtype.element_ty))


def paged_decode_attention(
    query: torch.Tensor,          # [num_seqs, num_heads, head_dim], one new token per sequence
    key_cache: torch.Tensor,      # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,   # [num_seqs, max_blocks] (int), padded
    context_lens: torch.Tensor,   # [num_seqs] (int)
    scale: float,
) -> torch.Tensor:
    num_seqs, num_heads, head_dim = query.shape
    _, block_size, num_kv_heads, _ = key_cache.shape
    if head_dim & (head_dim - 1) or block_size & (block_size - 1):
        raise ValueError("Triton kernel needs power-of-two head_dim and block_size")
    query = query.contiguous()
    block_tables = block_tables.to(torch.int32).contiguous()
    context_lens = context_lens.to(torch.int32).contiguous()
    out = torch.empty_like(query)
    _paged_decode_kernel[(num_seqs, num_heads)](
        query, key_cache, value_cache, block_tables, context_lens, out,
        scale,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        block_tables.stride(0),
        out.stride(0), out.stride(1),
        REP=num_heads // num_kv_heads,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
    )
    return out
