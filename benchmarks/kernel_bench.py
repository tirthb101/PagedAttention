"""Attention kernel microbenchmark (paper §7.1, Fig. 18a). GPU only.

Decode attention latency for batch sizes 8 and 32 and context lengths 64-256 (plus longer),
comparing:
    contiguous   PyTorch scaled_dot_product_attention on a normal contiguous KV cache (no paging),
                 with the SDPA backend pinned (--sdpa-backends; the first one fills contiguous_us)
    paged-ref    our PyTorch reference: gather blocks, then attention (SDPA backend left on auto,
                 because it needs a mask that the flash backend does not take)
    paged-triton our Triton kernel: reads blocks in place through the block table
The paper's PagedAttention kernel was 20-26% slower than FasterTransformer's contiguous one.

Timing modes (--timing):
    graph  (default) each call is captured --graph-inner times into a torch.cuda.CUDAGraph and the
           graph is replayed; CUDA events around the replays give GPU time per call. This removes
           host-side cost (Python, Triton's launcher, PyTorch dispatch) from BOTH sides, so it
           measures the kernels, as the paper's C++-launched kernels effectively were.
    eager  plain back-to-back Python calls timed with CUDA events (the old method). When a call's
           host cost is larger than its GPU time, this measures the host cost instead.

Every call reuses the same inputs, so on large-L2 GPUs (L40S: 96 MB) the small configs run
partly out of L2; this applies equally to both sides.

    python benchmarks/kernel_bench.py --model-dims opt-13b --out results/e6_kernel.csv
    python benchmarks/kernel_bench.py --timing eager --out results/e6_kernel_eager.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from minivllm.attention.paged_ref import paged_attention, plan_paged  # noqa: E402
from minivllm.attention.paged_triton import paged_decode_attention  # noqa: E402

DIMS = {"opt-125m": (12, 64), "opt-1.3b": (32, 64), "opt-13b": (40, 128)}  # (heads, head_dim)


def sdpa_context(name: str):
    """Context manager that restricts SDPA to one backend ('auto' = PyTorch's own choice)."""
    if name == "auto":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel  # torch >= 2.3

    attr = {"flash": "FLASH_ATTENTION", "efficient": "EFFICIENT_ATTENTION", "math": "MATH",
            "cudnn": "CUDNN_ATTENTION"}[name]
    if not hasattr(SDPBackend, attr):
        raise RuntimeError(f"SDPA backend {name} not in this PyTorch build")
    return sdpa_kernel(getattr(SDPBackend, attr))


def _event_time_us(run, iters: int, calls_per_run: int, repeats: int) -> float:
    """Median over `repeats` of GPU time per call, measured with CUDA events."""
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (iters * calls_per_run))
    return statistics.median(samples)


def bench_eager(fn, iters: int, repeats: int) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    return _event_time_us(fn, iters, 1, repeats)


def bench_graph(fn, iters: int, inner: int, repeats: int) -> float:
    # Warm up on a side stream (compiles Triton, lets SDPA pick its kernel) before capture,
    # as torch.cuda.graphs requires.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    t = _event_time_us(graph.replay, iters, inner, repeats)
    del graph
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dims", choices=DIMS, default="opt-13b")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--batches", default="8,32")
    ap.add_argument("--contexts", default="64,128,256,512,1024", help="paper's Fig. 18a range is 64-256")
    ap.add_argument("--timing", choices=["graph", "eager"], default="graph")
    ap.add_argument("--iters", type=int, default=100, help="timed eager calls / graph replays per repeat")
    ap.add_argument("--graph-inner", type=int, default=20, help="calls captured in one CUDA graph")
    ap.add_argument("--repeats", type=int, default=5, help="report the median of this many timings")
    ap.add_argument("--sdpa-backends", default="flash,efficient",
                    help="comma list of flash/efficient/math/cudnn/auto; the first one is contiguous_us")
    ap.add_argument("--triton-warps", default="default",
                    help="comma list of num_warps for the Triton kernel, 'default' = Triton's default (4); "
                         "the first one is paged_triton_us, the others get their own columns")
    ap.add_argument("--no-ref", action="store_true", help="skip the PyTorch gather reference")
    ap.add_argument("--out", type=Path, default=Path("results/e6_kernel.csv"))
    args = ap.parse_args()
    assert torch.cuda.is_available(), "kernel_bench needs a GPU"

    H, D = DIMS[args.model_dims]
    B = args.block_size
    dt = torch.float16
    backends = args.sdpa_backends.split(",")
    warps = [None if w == "default" else int(w) for w in args.triton_warps.split(",")]
    wname = lambda w: "default" if w is None else str(w)  # noqa: E731
    if args.timing == "graph":
        timer = lambda fn: bench_graph(fn, args.iters, args.graph_inner, args.repeats)  # noqa: E731
    else:
        timer = lambda fn: bench_eager(fn, args.iters, args.repeats)  # noqa: E731
    print(f"torch {torch.__version__}, timing={args.timing}, sdpa backends={backends}, "
          f"triton warps={[wname(w) for w in warps]}, gpu={torch.cuda.get_device_name(0)}", flush=True)

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
            q4 = q[:, :, None]                                  # [batch, H, 1, D] view for SDPA
            # Built once, already in the kernel's dtype: no per-call casts in the timed region.
            bt = torch.tensor(tables, device="cuda", dtype=torch.int32)
            cl = torch.full((batch,), ctx, device="cuda", dtype=torch.int32)
            out = torch.empty_like(q)
            plan = plan_paged(tables, [ctx] * batch, [1] * batch, B, q.device)
            k_contig = torch.randn(batch, H, ctx, D, dtype=dt, device="cuda")
            v_contig = torch.randn_like(k_contig)
            scale = D**-0.5

            row = {"timing": args.timing, "batch": batch, "context": ctx}
            # SDPA, contiguous KV, one column per pinned backend.
            t_sdpa = {}
            for name in backends:
                try:
                    with sdpa_context(name):
                        t_sdpa[name] = timer(lambda: F.scaled_dot_product_attention(q4, k_contig, v_contig, scale=scale))
                except RuntimeError as exc:                     # backend not available for these inputs
                    print(f"  sdpa {name} unavailable: {str(exc).splitlines()[0]}", flush=True)
                    t_sdpa[name] = None
            # Triton paged kernel, one column per num_warps setting.
            t_tri = {}
            for w in warps:
                t_tri[w] = timer(lambda: paged_decode_attention(q, k_cache, v_cache, bt, cl, scale, out=out, num_warps=w))
            t_ref = None if args.no_ref else timer(
                lambda: paged_attention(q, k_cache, v_cache, tables, [ctx] * batch, [1] * batch, scale, plan=plan))

            # Numerics check: the kernel must agree with the reference on the same paged cache.
            ref_out = paged_attention(q, k_cache, v_cache, tables, [ctx] * batch, [1] * batch, scale, plan=plan)
            tri_out = paged_decode_attention(q, k_cache, v_cache, bt, cl, scale, num_warps=warps[0])
            max_err = (tri_out.float() - ref_out.float()).abs().max().item()

            primary_b, primary_w = backends[0], warps[0]
            t_c, t_t = t_sdpa[primary_b], t_tri[primary_w]
            r1 = lambda x: None if x is None else round(x, 2)  # noqa: E731
            row.update({
                "sdpa_backend": primary_b,
                "contiguous_us": r1(t_c),
                "paged_ref_us": r1(t_ref),
                "triton_warps": wname(primary_w),
                "paged_triton_us": r1(t_t),
                "triton_vs_contiguous": None if t_c is None else round(t_t / t_c, 3),
            })
            for name in backends[1:]:
                row[f"contiguous_{name}_us"] = r1(t_sdpa[name])
            for w in warps[1:]:
                row[f"paged_triton_w{wname(w)}_us"] = r1(t_tri[w])
            best_w = min(warps, key=lambda w: t_tri[w])
            row.update({"paged_triton_best_us": r1(t_tri[best_w]), "triton_best_warps": wname(best_w),
                        "triton_max_abs_err": f"{max_err:.2e}",
                        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
            rows.append(row)
            extra = " | ".join(f"sdpa-{n} {t_sdpa[n]:.1f}" for n in backends[1:] if t_sdpa[n] is not None)
            print(f"[{args.timing}] batch {batch:>3} ctx {ctx:>5}: contiguous({primary_b}) "
                  f"{t_c if t_c is not None else float('nan'):8.1f} us | "
                  f"paged-ref {t_ref if t_ref is not None else float('nan'):8.1f} us | "
                  f"paged-triton {t_t:8.1f} us (best {t_tri[best_w]:.1f} @ warps={wname(best_w)})"
                  + (f"  ({t_t / t_c:.2f}x contiguous)" if t_c else "")
                  + (f" | {extra}" if extra else "") + f" | err {max_err:.1e}", flush=True)
            del k_cache, v_cache, k_contig, v_contig

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
