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
    # E9 (extension X1): shared system prompt / multi-turn chat, with and without prefix caching
    python benchmarks/run_serving.py --tiny --trace synthetic-alpaca --sys-prompt-len 512 --prefix-caching --rates 4 --duration 10 --out results/smoke_e9.csv
    python benchmarks/run_serving.py --tiny --trace synthetic-sharegpt --chat-turns 4 --prefix-caching --rates 1 --duration 10 --out results/smoke_e9.csv
    # real run (cluster / Colab GPU)
    python benchmarks/run_serving.py --model facebook/opt-1.3b --trace sharegpt --rates 1,2,4 \\
        --duration 600 --memory-manager paged --kv-cache-gb 4 --out results/e1/sharegpt_paged.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import heapq
import itertools
import random
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from dataclasses import dataclass  # noqa: E402
from typing import Optional  # noqa: E402

from benchmarks.trace import Conversation, Request, load_conversations, load_trace, summarize, system_prompts  # noqa: E402
from minivllm.config import SchedulerConfig  # noqa: E402
from minivllm.engine import LLMEngine  # noqa: E402
from minivllm.memory import MEMORY_MANAGERS  # noqa: E402
from minivllm.sequence import SamplingParams  # noqa: E402

FIELDS = [
    "system", "model", "trace", "rate", "num_requests", "finished", "timed_out",
    "norm_latency_mean", "norm_latency_p50", "norm_latency_p90",
    "latency_mean", "ttft_mean", "throughput_req_s", "throughput_tok_s",
    "mean_batch", "preemptions", "wall_s", "block_size", "kv_cache_gb", "num_gpu_blocks",
    "preemption_mode", "attention_backend", "n", "gpu", "seed", "beam",
    "temperature", "mem_saving_avg_pct", "mem_saving_max_pct", "mem_saving_blocks_pct", "swap_fallbacks",
    # E9 / extension X1 (prefix caching)
    "workload", "prefix_caching", "register_prefix", "sys_prompt_len", "num_sys_prompts", "chat_turns",
    "think_time", "ttft_p50", "ttft_p90", "prompt_tokens_mean", "prefill_tokens",
    "cache_hit_rate_pct", "cache_hit_tokens", "cache_query_tokens", "cache_evictions",
]


@dataclass
class Submit:
    """One request to hand to the engine at `arrival_time`."""

    arrival_time: float
    request_id: str
    prompt: list[int]
    output_len: int
    prefix_id: Optional[str] = None


class OpenLoop:
    """Requests from a trace at Poisson arrival times (E1-E8; E9 workload B adds a shared
    system prompt in front of every prompt)."""

    def __init__(self, args, tok, rate: float, vocab: int, engine) -> None:
        num = args.num_requests or max(10, int(round(rate * args.duration)))
        L, K = getattr(args, "sys_prompt_len", 0), max(1, getattr(args, "num_sys_prompts", 1))
        reqs: list[Request] = load_trace(args.trace, tok, num, rate, args.seed, args.max_model_len - L)
        sys_prompts = system_prompts(K, L, vocab, args.seed) if L else []
        bos = getattr(tok, "bos_token_id", None) if tok is not None else None
        prefix_ids: list[Optional[str]] = [None] * K
        if L and getattr(args, "register_prefix", False):
            # Paper §4.4 baseline: the operator declares the system prompts as shared prefixes.
            for k, sp in enumerate(sys_prompts):
                prefix_ids[k] = f"sys{k}"
                engine.register_prefix(prefix_ids[k], ([bos] if bos is not None else []) + sp)
        self.items: list[Submit] = []
        for r in reqs:
            prompt = list(r.token_ids(vocab, args.seed))
            k = 0
            if L:
                k = random.Random(f"{args.seed}-{r.request_id}").randrange(K)
                if bos is not None and prompt and prompt[0] == bos:
                    prompt = [bos] + sys_prompts[k] + prompt[1:]  # keep BOS first
                else:
                    prompt = ([bos] if bos is not None else []) + sys_prompts[k] + prompt
            self.items.append(Submit(r.arrival_time, r.request_id, prompt, r.output_len,
                                     prefix_ids[k] if L else None))
        self.num_requests = len(self.items)
        self.extra_wall = 0.0

    def initial(self) -> list[Submit]:
        return self.items

    def on_finish(self, out, now: float) -> Optional[Submit]:
        return None


class ChatLoop:
    """E9 workload A: closed-loop multi-turn chat. Conversations start at Poisson times
    (`rate` = new conversations per second). When turn t finishes, the user "thinks" for
    an exponential time (mean --think-time s) and sends turn t+1, whose prompt is the
    whole history: previous prompt + the model's actual answer + the new message."""

    def __init__(self, args, tok, rate: float, vocab: int, engine) -> None:
        num = args.num_requests or max(4, int(round(rate * args.duration)))
        self.convs: list[Conversation] = load_conversations(
            args.trace, tok, num, rate, args.seed, args.max_model_len, args.chat_turns, vocab)
        self.by_id = {c.conv_id: c for c in self.convs}
        self.prompts: dict[str, list[int]] = {}  # request id -> its prompt (to build the next turn)
        self.think = args.think_time
        self.rng = random.Random(f"think-{args.seed}")
        self.num_requests = sum(c.num_turns for c in self.convs)
        self.extra_wall = args.chat_turns * (args.think_time + 30.0)

    def _submit(self, conv: Conversation, turn: int, history: list[int], at: float) -> Submit:
        rid = f"{conv.conv_id}-t{turn}"
        prompt = history + conv.user_turns[turn]
        self.prompts[rid] = prompt
        return Submit(at, rid, prompt, conv.output_lens[turn])

    def initial(self) -> list[Submit]:
        return [self._submit(c, 0, [], c.arrival_time) for c in self.convs]

    def on_finish(self, out, now: float) -> Optional[Submit]:
        prompt = self.prompts.pop(out.request_id, None)
        if prompt is None or out.aborted:
            return None
        conv_id, turn = out.request_id.rsplit("-t", 1)
        conv, turn = self.by_id[conv_id], int(turn) + 1
        if turn >= conv.num_turns:
            return None
        think = self.rng.expovariate(1.0 / self.think) if self.think > 0 else 0.0
        return self._submit(conv, turn, prompt + out.output_token_ids[0], now + think)



def sampling_temperature(args) -> float:
    """Greedy for n=1 (as before); parallel samples must be random draws (paper §4.4)."""
    t = getattr(args, "temperature", None)
    if t is not None:
        return t
    return 1.0 if getattr(args, "n", 1) > 1 and not getattr(args, "beam", 0) else 0.0


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
    vocab = model.config.vocab_size
    sched = SchedulerConfig(max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
                            max_model_len=args.max_model_len, preemption_mode=args.preemption)
    engine = LLMEngine(model, tok, block_size=args.block_size, kv_cache_gb=args.kv_cache_gb,
                       swap_space_gb=args.swap_space_gb, num_gpu_blocks=args.num_gpu_blocks,
                       scheduler_config=sched, memory_manager=args.memory_manager,
                       attention_backend=args.attention_backend, seed=args.seed,
                       enable_prefix_caching=getattr(args, "prefix_caching", False))

    # Warm-up (CUDA context, Triton compilation) outside the timed region.
    engine.generate([[5, 6, 7, 8]] * 2, SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
    engine.reset_stats()

    chat = getattr(args, "chat_turns", 0) > 0
    workload = (ChatLoop if chat else OpenLoop)(args, tok, rate, vocab, engine)
    temperature = sampling_temperature(args)
    beam = getattr(args, "beam", 0)
    tie = itertools.count()
    pending = [(s.arrival_time, next(tie), s) for s in workload.initial()]
    heapq.heapify(pending)
    outputs = []
    prompt_lens: dict[str, int] = {}
    # Stop a hopelessly overloaded run. Chat runs also wait for the later turns.
    deadline = args.duration * args.max_wall_factor + 120 + workload.extra_wall
    engine.reset_clock()
    while pending or engine.has_unfinished():
        now = engine.now()
        if now > deadline:
            break
        while pending and pending[0][0] <= now:
            sub: Submit = heapq.heappop(pending)[2]
            params = SamplingParams(n=beam or args.n, use_beam_search=beam > 0,
                                    temperature=0.0 if beam else temperature,
                                    max_tokens=sub.output_len, ignore_eos=True)
            engine.add_request(sub.request_id, sub.prompt, params, arrival_time=sub.arrival_time,
                               true_output_len=sub.output_len, prefix_id=sub.prefix_id)
            prompt_lens[sub.request_id] = len(sub.prompt)
        if not engine.has_unfinished():
            time.sleep(max(0.0, min(pending[0][0] - engine.now(), 0.05)))
            continue
        for o in engine.step():
            outputs.append(o)
            nxt = workload.on_finish(o, engine.now())
            if nxt is not None:
                heapq.heappush(pending, (nxt.arrival_time, next(tie), nxt))
    wall = engine.now()
    stats = engine.stats()

    done = [o for o in outputs if not o.aborted]
    norm = [(o.finish_time - o.arrival_time) / max(1, len(o.output_token_ids[0])) for o in done]
    lat = [o.finish_time - o.arrival_time for o in done]
    ttft = [o.first_token_time - o.arrival_time for o in done if o.first_token_time is not None]
    out_tokens = sum(len(s) for o in done for s in o.output_token_ids)
    q = lambda xs, p: statistics.quantiles(xs, n=100)[p - 1] if len(xs) >= 2 else (xs[0] if xs else math.nan)
    num_requests = workload.num_requests
    workload_name = "chat" if chat else ("sysprompt" if getattr(args, "sys_prompt_len", 0) else "single")
    row = {
        "system": args.memory_manager, "model": "tiny-random-opt" if args.tiny else args.model,
        "trace": args.trace, "rate": rate, "num_requests": num_requests, "finished": len(done),
        "timed_out": int(len(done) < num_requests),
        "norm_latency_mean": statistics.fmean(norm) if norm else math.nan,
        "norm_latency_p50": q(norm, 50), "norm_latency_p90": q(norm, 90),
        "latency_mean": statistics.fmean(lat) if lat else math.nan,
        "ttft_mean": statistics.fmean(ttft) if ttft else math.nan,
        "throughput_req_s": len(done) / wall, "throughput_tok_s": out_tokens / wall,
        "mean_batch": stats["mean_batch"],
        "preemptions": stats["preemptions"], "wall_s": wall,
        "block_size": args.block_size, "kv_cache_gb": args.kv_cache_gb,
        "num_gpu_blocks": engine.block_manager.num_total_gpu_blocks,
        "preemption_mode": args.preemption, "attention_backend": args.attention_backend, "n": args.n,
        "beam": getattr(args, "beam", 0), "temperature": temperature,
        # Memory saved by block sharing (paper Fig. 14/15); always 0 for the Orca baselines.
        "mem_saving_avg_pct": stats["mem_saving_avg_pct"], "mem_saving_max_pct": stats["mem_saving_max_pct"],
        "mem_saving_blocks_pct": stats["mem_saving_blocks_pct"], "swap_fallbacks": stats["swap_fallbacks"],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": args.seed,
        # E9 / X1
        "workload": workload_name, "prefix_caching": int(getattr(args, "prefix_caching", False)),
        "register_prefix": int(getattr(args, "register_prefix", False)),
        "sys_prompt_len": getattr(args, "sys_prompt_len", 0), "num_sys_prompts": getattr(args, "num_sys_prompts", 1),
        "chat_turns": getattr(args, "chat_turns", 0), "think_time": getattr(args, "think_time", 0.0),
        "ttft_p50": q(ttft, 50), "ttft_p90": q(ttft, 90),
        "prompt_tokens_mean": statistics.fmean(prompt_lens[o.request_id] for o in done) if done else math.nan,
        "prefill_tokens": stats["num_prefill_tokens"],
        "cache_hit_rate_pct": stats["cache_hit_rate_pct"], "cache_hit_tokens": stats["cache_hit_tokens"],
        "cache_query_tokens": stats["cache_query_tokens"], "cache_evictions": stats["cache_evictions"],
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
    ap.add_argument("--beam", type=int, default=0, help="beam width per request (E3 beam search); paged only")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature (default: 1.0 when --n > 1, else 0 = greedy)")
    # E9 / extension X1 (automatic prefix caching)
    ap.add_argument("--prefix-caching", action="store_true", help="X1: automatic prefix caching (paged only)")
    ap.add_argument("--sys-prompt-len", type=int, default=0,
                    help="E9 workload B: prepend a shared system prompt of this many tokens to every request")
    ap.add_argument("--num-sys-prompts", type=int, default=1, help="workload B: distinct system prompts")
    ap.add_argument("--register-prefix", action="store_true",
                    help="workload B baseline: declare the system prompts as §4.4 shared prefixes (paper)")
    ap.add_argument("--chat-turns", type=int, default=0,
                    help="E9 workload A: closed-loop multi-turn chat with up to this many turns "
                         "(rate = new conversations/s; needs a sharegpt trace)")
    ap.add_argument("--think-time", type=float, default=2.0, help="workload A: mean user think time (s)")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-wall-factor", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.beam and args.n != 1:
        ap.error("use either --n (parallel sampling) or --beam (beam search), not both")
    if args.beam and args.memory_manager != "paged":
        ap.error("--beam is implemented for the paged memory manager only")
    if (args.prefix_caching or args.register_prefix) and args.memory_manager != "paged":
        ap.error("--prefix-caching / --register-prefix need the paged memory manager")
    if args.register_prefix and not args.sys_prompt_len:
        ap.error("--register-prefix needs --sys-prompt-len")
    if args.chat_turns and args.sys_prompt_len:
        ap.error("use either --chat-turns (workload A) or --sys-prompt-len (workload B)")

    model, tok = load_model(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    # Appending to a CSV written by an older version: keep its header (columns it lacks are dropped).
    fields = FIELDS if new_file else next(csv.reader(args.out.open())) or FIELDS
    for rate in [float(r) for r in args.rates.split(",")]:
        print(f"[{time.strftime('%H:%M:%S')}] {args.memory_manager} rate={rate} ...", flush=True)
        row = run_one_rate(model, tok, args, rate)
        with args.out.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if new_file:
                w.writeheader()
                new_file = False
            w.writerow(row)
        print(f"   finished {row['finished']}/{row['num_requests']}  norm latency {row['norm_latency_mean']:.4f} s/tok  "
              f"throughput {row['throughput_req_s']:.2f} req/s  batch {row['mean_batch']:.1f}  "
              f"preempt {row['preemptions']}  wall {row['wall_s']:.0f}s  "
              f"mem saving avg {row['mem_saving_avg_pct']:.1f}% max {row['mem_saving_max_pct']:.1f}%  "
              f"ttft {row['ttft_mean']:.3f}s  cache hit {row['cache_hit_rate_pct']:.1f}%  "
              f"prefill tok {row['prefill_tokens']}", flush=True)


if __name__ == "__main__":
    main()
