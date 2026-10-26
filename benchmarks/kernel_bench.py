"""Attention kernel microbenchmark (paper §7.1, Fig. 18a). GPU only.

Decode attention latency for batch sizes 8 and 32 and context lengths 64-256 (plus longer),
comparing:
    contiguous   PyTorch scaled_dot_product_attention on a normal contiguous KV cache (no paging)
    paged-ref    our PyTorch reference: gather blocks, then attention
    paged-triton our Triton kernel: reads blocks in place through the block table
The paper's PagedAttention kernel was 20-26% slower than FasterTransformer's contiguous one.

    python benchmarks/kernel_bench.py --model-dims opt-13b --out results/e6_kernel.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from minivllm.attention.paged_ref import paged_attention, plan_paged  # noqa: E402
from minivllm.attention.paged_triton import paged_decode_attention  # noqa: E402

DIMS = {"opt-125m": (12, 64), "opt-1.3b": (32, 64), "opt-13b": (40, 128)}  # (heads, head_dim)


def bench(fn, iters: int = 100) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # microseconds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dims", choices=DIMS, default="opt-13b")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--batches", default="8,32")
    ap.add_argument("--contexts", default="64,128,256,512,1024")
    ap.add_argument("--out", type=Path, default=Path("results/e6_kernel.csv"))
    args = ap.parse_args()
    assert torch.cuda.is_available(), "kernel_bench needs a GPU"

    H, D = DIMS[args.model_dims]
    B = args.block_size
    dt = torch.float16
    rows = []
    for batch in map(int, args.batches.split(",")):
        for ctx in map(int, args.contexts.split(",")):
            n_blocks = -(-ctx // B)
            pool = batch * n_blocks
            k_cache = torch.randn(pool, B, H, D, dtype=dt, device="cuda")
            v_cache = torch.randn_like(k_cache)
            perm = list(range(pool))
            random.Random(0).shuffle(perm)                      # blocks scattered in memory
            tables = [perm[i * n_blocks:(i + 1) * n_blocks] for i in range(batch)]
            q = torch.randn(batch, H, D, dtype=dt, device="cuda")
            bt = torch.tensor(tables, device="cuda")
            cl = torch.full((batch,), ctx, device="cuda")
            plan = plan_paged(tables, [ctx] * batch, [1] * batch, B, q.device)
            k_contig = torch.randn(batch, H, ctx, D, dtype=dt, device="cuda")
            v_contig = torch.randn_like(k_contig)
            scale = D**-0.5

            t_contig = bench(lambda: F.scaled_dot_product_attention(q[:, :, None], k_contig, v_contig, scale=scale))
            t_ref = bench(lambda: paged_attention(q, k_cache, v_cache, tables, [ctx] * batch, [1] * batch, scale, plan=plan))
            t_tri = bench(lambda: paged_decode_attention(q, k_cache, v_cache, bt, cl, scale))
            rows.append({"batch": batch, "context": ctx, "contiguous_us": round(t_contig, 1),
                         "paged_ref_us": round(t_ref, 1), "paged_triton_us": round(t_tri, 1),
                         "triton_vs_contiguous": round(t_tri / t_contig, 2),
                         "gpu": torch.cuda.get_device_name(0)})
            print(f"batch {batch:>3} ctx {ctx:>5}: contiguous {t_contig:8.1f} us | paged-ref {t_ref:8.1f} us | "
                  f"paged-triton {t_tri:8.1f} us  ({t_tri / t_contig:.2f}x contiguous)", flush=True)
            del k_cache, v_cache, k_contig, v_contig

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
