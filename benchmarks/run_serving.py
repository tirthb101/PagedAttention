"""Online serving benchmark: latency vs request rate (paper §6, Fig. 12–17).

For each request rate, requests from a trace arrive by a Poisson process in REAL time
and are served by our engine with one memory manager (paged = vLLM, or an Orca
baseline). We record, per request, arrival / first token / finish times, and report the
paper's metric:

    normalized latency = mean over requests of (end-to-end latency / output length)  [s/token]

plus throughput, time-to-first-token, mean batch size and preemptions. One CSV row is
appended per rate, so a run that is killed keeps the rates it finished.

    # smoke test anywhere (no download, CPU ok)
    python benchmarks/run_serving.py --tiny --trace synthetic-sharegpt --rates 2 --duration 10 --out results/smoke.csv
    # real run (cluster / Colab GPU)
    python benchmarks/run_serving.py --model facebook/opt-1.3b --trace sharegpt --rates 1,2,4 \\
        --duration 600 --memory-manager paged --kv-cache-gb 4 --out results/e1/sharegpt_paged.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from benchmarks.trace import Request, load_trace, summarize  # noqa: E402
from minivllm.config import SchedulerConfig  # noqa: E402
from minivllm.engine import LLMEngine  # noqa: E402
from minivllm.memory import MEMORY_MANAGERS  # noqa: E402
from minivllm.sequence import SamplingParams  # noqa: E402

FIELDS = [
    "system", "model", "trace", "rate", "num_requests", "finished", "timed_out",
    "norm_latency_mean", "norm_latency_p50", "norm_latency_p90",
    "latency_mean", "ttft_mean", "throughput_req_s", "throughput_tok_s",
    "mean_batch", "preemptions", "wall_s", "block_size", "kv_cache_gb", "num_gpu_blocks",
    "preemption_mode", "attention_backend", "n", "gpu", "seed",
]


def load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer, OPTConfig, OPTForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if args.tiny:
        torch.manual_seed(0)
        cfg = OPTConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2, ffn_dim=128,
                        num_attention_heads=4, max_position_embeddings=2048, word_embed_proj_dim=64)
        return OPTForCausalLM(cfg).to(device, dtype).eval(), None
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    return model, tok


def run_one_rate(model, tok, args, rate: float) -> dict:
    num = args.num_requests or max(10, int(round(rate * args.duration)))
    reqs: list[Request] = load_trace(args.trace, tok, num, rate, args.seed, args.max_model_len)
    vocab = model.config.vocab_size

    sched = SchedulerConfig(max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
                            max_model_len=args.max_model_len, preemption_mode=args.preemption)
    engine = LLMEngine(model, tok, block_size=args.block_size, kv_cache_gb=args.kv_cache_gb,
                       swap_space_gb=args.swap_space_gb, num_gpu_blocks=args.num_gpu_blocks,
                       scheduler_config=sched, memory_manager=args.memory_manager,
                       attention_backend=args.attention_backend, seed=args.seed)

    # Warm-up (CUDA context, Triton compilation) outside the timed region.
    engine.generate([[5, 6, 7, 8]] * 2, SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
    engine.num_running_sum = engine.num_steps = 0
    engine.scheduler.num_preemptions = 0

    pending = list(reqs)
    outputs = []
    deadline = args.duration * args.max_wall_factor + 120  # stop a hopelessly overloaded run
    engine.reset_clock()
    while pending or engine.has_unfinished():
        now = engine.now()
        if now > deadline:
            break
        while pending and pending[0].arrival_time <= now:
            r = pending.pop(0)
            params = SamplingParams(n=args.n, temperature=0.0, max_tokens=r.output_len, ignore_eos=True)
            engine.add_request(r.request_id, r.token_ids(vocab, args.seed), params,
                               arrival_time=r.arrival_time, true_output_len=r.output_len)
        if not engine.has_unfinished():
            time.sleep(max(0.0, min(pending[0].arrival_time - engine.now(), 0.05)))
            continue
        outputs.extend(engine.step())
    wall = engine.now()

    done = [o for o in outputs if not o.aborted]
    norm = [(o.finish_time - o.arrival_time) / max(1, len(o.output_token_ids[0])) for o in done]
    lat = [o.finish_time - o.arrival_time for o in done]
    ttft = [o.first_token_time - o.arrival_time for o in done if o.first_token_time is not None]
    out_tokens = sum(len(s) for o in done for s in o.output_token_ids)
    q = lambda xs, p: statistics.quantiles(xs, n=100)[p - 1] if len(xs) >= 2 else (xs[0] if xs else math.nan)
    row = {
        "system": args.memory_manager, "model": "tiny-random-opt" if args.tiny else args.model,
        "trace": args.trace, "rate": rate, "num_requests": len(reqs), "finished": len(done),
        "timed_out": int(len(done) < len(reqs)),
        "norm_latency_mean": statistics.fmean(norm) if norm else math.nan,
        "norm_latency_p50": q(norm, 50), "norm_latency_p90": q(norm, 90),
        "latency_mean": statistics.fmean(lat) if lat else math.nan,
        "ttft_mean": statistics.fmean(ttft) if ttft else math.nan,
        "throughput_req_s": len(done) / wall, "throughput_tok_s": out_tokens / wall,
        "mean_batch": engine.num_running_sum / max(1, engine.num_steps),
        "preemptions": engine.scheduler.num_preemptions, "wall_s": wall,
        "block_size": args.block_size, "kv_cache_gb": args.kv_cache_gb,
        "num_gpu_blocks": engine.block_manager.num_total_gpu_blocks,
        "preemption_mode": args.preemption, "attention_backend": args.attention_backend, "n": args.n,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": args.seed,
    }
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--tiny", action="store_true", help="random tiny OPT, no download (smoke test)")
    ap.add_argument("--trace", default="synthetic-sharegpt",
                    help="sharegpt | alpaca (real, from HF cache) | synthetic-sharegpt | synthetic-alpaca")
    ap.add_argument("--rates", default="1", help="comma-separated request rates (req/s)")
    ap.add_argument("--duration", type=float, default=60, help="seconds of arrivals per rate (paper: 3600)")
    ap.add_argument("--num-requests", type=int, default=None, help="override rate*duration")
    ap.add_argument("--memory-manager", choices=MEMORY_MANAGERS, default="paged")
    ap.add_argument("--attention-backend", choices=["ref", "triton"],
                    default="triton" if torch.cuda.is_available() else "ref")
    ap.add_argument("--kv-cache-gb", type=float, default=2.0)
    ap.add_argument("--num-gpu-blocks", type=int, default=None, help="override --kv-cache-gb")
    ap.add_argument("--swap-space-gb", type=float, default=4.0)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--preemption", choices=["recompute", "swap"], default="recompute")
    ap.add_argument("--n", type=int, default=1, help="parallel samples per request (E3)")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-wall-factor", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model, tok = load_model(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    for rate in [float(r) for r in args.rates.split(",")]:
        print(f"[{time.strftime('%H:%M:%S')}] {args.memory_manager} rate={rate} ...", flush=True)
        row = run_one_rate(model, tok, args, rate)
        with args.out.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
                new_file = False
            w.writerow(row)
        print(f"   finished {row['finished']}/{row['num_requests']}  norm latency {row['norm_latency_mean']:.4f} s/tok  "
              f"throughput {row['throughput_req_s']:.2f} req/s  batch {row['mean_batch']:.1f}  "
              f"preempt {row['preemptions']}  wall {row['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
