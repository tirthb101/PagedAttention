"""PagedAttention (reference) must equal ordinary attention, wherever the blocks are placed."""

import random

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from minivllm.attention import paged_attention, write_kv  # noqa: E402


def dense_attention(q, k, v, scale):
    """q: [qlen, H, D] are the LAST qlen positions of a ctx-long sequence with keys k, v: [ctx, H, D]."""
    qlen, ctx = q.shape[0], k.shape[0]
    q_pos = torch.arange(ctx - qlen, ctx)
    mask = torch.arange(ctx)[None, :] <= q_pos[:, None]
    o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask, scale=scale)
    return o.transpose(0, 1)


@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 4), (8, 2)])
def test_paged_equals_dense_with_shuffled_blocks(block_size, num_heads, num_kv_heads):
    torch.manual_seed(0)
    rng = random.Random(0)
    D, num_blocks = 16, 64
    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, D, dtype=torch.float64)
    v_cache = torch.zeros_like(k_cache)
    free = list(range(num_blocks))
    rng.shuffle(free)  # blocks scattered anywhere in the pool

    # (context length, new tokens): two prefills (one partial, one full) and two decodes
    cases = [(13, 13), (7, 3), (20, 1), (block_size * 2, 1)]
    queries, tables, ctx_lens, q_lens, expected = [], [], [], [], []
    for ctx, qlen in cases:
        k = torch.randn(ctx, num_kv_heads, D, dtype=torch.float64)
        v = torch.randn(ctx, num_kv_heads, D, dtype=torch.float64)
        q = torch.randn(qlen, num_heads, D, dtype=torch.float64)
        n_blocks = -(-ctx // block_size)
        table = [free.pop() for _ in range(n_blocks)]
        slots = torch.tensor([table[p // block_size] * block_size + p % block_size for p in range(ctx)])
        write_kv(k, v, k_cache, v_cache, slots)
        rep = num_heads // num_kv_heads
        expected.append(dense_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), scale=D**-0.5))
        queries.append(q)
        tables.append(table)
        ctx_lens.append(ctx)
        q_lens.append(qlen)

    out = paged_attention(torch.cat(queries), k_cache, v_cache, tables, ctx_lens, q_lens, scale=D**-0.5)
    torch.testing.assert_close(out, torch.cat(expected))
