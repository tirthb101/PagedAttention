"""Triton paged decode kernel must equal the PyTorch reference.

On a machine without CUDA the kernel runs through Triton's CPU interpreter (slow but exact),
so this test also runs on a laptop. On the cluster/Colab it runs on the GPU.
"""

import os
import random

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")  # must be set before triton is imported
pytest.importorskip("triton")

from minivllm.attention.paged_ref import paged_attention, write_kv  # noqa: E402
from minivllm.attention.paged_triton import paged_decode_attention  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("block_size", [1, 4, 16])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 4), (8, 2)])
def test_triton_decode_matches_reference(block_size, num_heads, num_kv_heads):
    torch.manual_seed(0)
    rng = random.Random(0)
    D, num_blocks = 32, 96
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, D, dtype=dtype, device=DEVICE)
    v_cache = torch.zeros_like(k_cache)
    free = list(range(num_blocks))
    rng.shuffle(free)

    ctx_lens = [1, 5, block_size, block_size + 1, 37]
    tables = []
    for ctx in ctx_lens:
        n = -(-ctx // block_size)
        table = [free.pop() for _ in range(n)]
        slots = torch.tensor([table[p // block_size] * block_size + p % block_size for p in range(ctx)], device=DEVICE)
        k = torch.randn(ctx, num_kv_heads, D, dtype=dtype, device=DEVICE)
        v = torch.randn(ctx, num_kv_heads, D, dtype=dtype, device=DEVICE)
        write_kv(k, v, k_cache, v_cache, slots)
        tables.append(table)
    q = torch.randn(len(ctx_lens), num_heads, D, dtype=dtype, device=DEVICE)

    ref = paged_attention(q, k_cache, v_cache, tables, ctx_lens, [1] * len(ctx_lens), scale=D**-0.5)
    max_blocks = max(len(t) for t in tables)
    bt = torch.tensor([t + [0] * (max_blocks - len(t)) for t in tables], device=DEVICE)
    out = paged_decode_attention(q, k_cache, v_cache, bt, torch.tensor(ctx_lens, device=DEVICE), D**-0.5)
    tol = 2e-3 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("memory_manager", ["paged", "orca-oracle"])
def test_engine_with_triton_backend_matches_reference_backend(memory_manager):
    """Whole engine (preemption included) gives the same tokens with the Triton kernel."""
    transformers = pytest.importorskip("transformers")
    from minivllm.config import SchedulerConfig
    from minivllm.engine import LLMEngine
    from minivllm.sequence import SamplingParams

    torch.manual_seed(0)
    cfg = transformers.OPTConfig(vocab_size=300, hidden_size=64, num_hidden_layers=2, ffn_dim=128,
                                 num_attention_heads=4, max_position_embeddings=256, word_embed_proj_dim=64,
                                 init_std=0.3)
    model = transformers.OPTForCausalLM(cfg).eval().to(DEVICE)
    rng = random.Random(0)
    prompts = [[rng.randrange(3, 300) for _ in range(n)] for n in (3, 9, 17, 30)]
    params = SamplingParams(temperature=0.0, max_tokens=10, ignore_eos=True)
    outs = {}
    for backend in ("ref", "triton"):
        sched = SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=256, max_model_len=256)
        engine = LLMEngine(model, block_size=4, num_gpu_blocks=12 if memory_manager == "paged" else 64,
                           num_cpu_blocks=64, scheduler_config=sched, watermark=0.0, check_invariants=True,
                           memory_manager=memory_manager, attention_backend=backend)
        outs[backend] = [o.output_token_ids for o in engine.generate(prompts, params)]
        if memory_manager == "paged":
            assert engine.scheduler.num_preemptions > 0
    assert outs["triton"] == outs["ref"]
