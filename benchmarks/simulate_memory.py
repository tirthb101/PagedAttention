"""GPU-free memory simulator (experiment E0, paper Fig. 2 and Fig. 13).

Replays a request trace through the REAL scheduler and the REAL memory managers
(paged = PagedAttention, orca-* = baselines). Only the model is fake: each step every
running sequence "generates" one token until it reaches its true output length.

For every step it records how the KV cache memory is used:
    token states            slots holding real K/V
    reserved                slots reserved for tokens that will be generated later (Orca)
    internal fragmentation  allocated slots that will never be used
    free                    unallocated slots (when requests wait anyway, this is external fragmentation)
and how many requests run concurrently (batch size). With all requests arriving at once
(offline mode, the default) memory is the bottleneck, so this directly shows why paging
fits more requests.

    python benchmarks/simulate_memory.py                       # OPT-13B, 12 GB KV, synthetic ShareGPT
    python benchmarks/simulate_memory.py --trace alpaca --num-requests 2000
    python benchmarks/simulate_memory.py --plot results/e0_memory.png --csv results/e0_memory.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.trace import Request, summarize, synthetic_trace  # noqa: E402
from minivllm.config import CacheConfig, SchedulerConfig, get_model_spec  # noqa: E402
from minivllm.memory import MEMORY_MANAGERS, make_block_manager  # noqa: E402
from minivllm.scheduler import Scheduler  # noqa: E402
from minivllm.sequence import SamplingParams, Sequence, SequenceGroup, SequenceStatus  # noqa: E402


@dataclass
class SimResult:
    system: str
    steps: int = 0
    total_output_tokens: int = 0
    preemptions: int = 0
    batch_sizes: list[int] = field(default_factory=list)
    # time-averaged fractions of the KV pool (only over steps where requests were waiting = memory-bound)
    frac: dict[str, float] = field(default_factory=dict)

    @property
    def mean_batch(self) -> float:
        return sum(self.batch_sizes) / max(1, len(self.batch_sizes))

    @property
    def tokens_per_step(self) -> float:
        return self.total_output_tokens / max(1, self.steps)


def simulate(system: str, reqs: list[Request], block_size: int, num_gpu_blocks: int,
             num_cpu_blocks: int, sched_cfg: SchedulerConfig, check: bool = False) -> SimResult:
    bm = make_block_manager(system, block_size, num_gpu_blocks, num_cpu_blocks,
                            watermark=0.01, max_model_len=sched_cfg.max_model_len)
    sched = Scheduler(sched_cfg, bm)
    pending = sorted(reqs, key=lambda r: r.arrival_time)
    groups: list[SequenceGroup] = []
    res = SimResult(system)
    sums = {"token_states": 0.0, "reserved": 0.0, "internal_frag": 0.0, "free": 0.0}
    n_bound = 0
    total_slots = num_gpu_blocks * block_size
    seq_ids = iter(range(10**9))
    now = 0.0

    while pending or sched.has_unfinished():
        if not sched.has_unfinished() and pending:
            now = max(now, pending[0].arrival_time)  # idle: jump to the next arrival
        while pending and pending[0].arrival_time <= now:
            r = pending.pop(0)
            seq = Sequence(next(seq_ids), [1] * r.prompt_len, block_size)
            g = SequenceGroup(r.request_id, [seq], SamplingParams(max_tokens=r.output_len, ignore_eos=True),
                              arrival_time=r.arrival_time, true_output_len=r.output_len)
            sched.add_seq_group(g)
            groups.append(g)

        out = sched.schedule(now)
        running = [s for g in out.scheduled_groups for s in g.get_seqs(SequenceStatus.RUNNING)]
        # Fake model step: every scheduled sequence computes its pending tokens and emits one.
        for seq in running:
            seq.num_computed_tokens = seq.get_len()
            seq.append_token(7)
            res.total_output_tokens += 1
        for g in out.scheduled_groups:
            for seq in g.get_seqs(SequenceStatus.RUNNING):
                if seq.get_output_len() >= g.true_output_len:
                    seq.status = SequenceStatus.FINISHED_LENGTH
                    sched.free_seq(seq)
        sched.free_finished_seq_groups()

        # Memory snapshot (after this step's tokens are written).
        live = [s for g in sched.running for s in g.get_seqs(SequenceStatus.RUNNING)]
        stats = bm.memory_stats(live)
        if sched.waiting:  # memory-bound moment: someone is waiting for space
            used = stats["used_slots"]
            future = stats.get("reserved_future_slots", 0)
            internal = stats["allocated_slots"] - used - future
            sums["token_states"] += used / total_slots
            sums["reserved"] += future / total_slots
            sums["internal_frag"] += internal / total_slots
            sums["free"] += stats["free_blocks"] * block_size / total_slots
            n_bound += 1
        res.batch_sizes.append(len(sched.running))
        if check:
            bm.check_invariants(live)
        res.steps += 1
        now += 1.0  # one "step" of simulated time; arrival times are in steps for --rate runs

    res.preemptions = sched.num_preemptions
    res.frac = {k: v / max(1, n_bound) for k, v in sums.items()}
    assert all(g.is_finished() for g in groups)
    return res


def plot(results: list[SimResult], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r.system for r in results]
    parts = [("token_states", "Token states", "#2b6cb0"), ("reserved", "Reserved (future tokens)", "#90cdf4"),
             ("internal_frag", "Internal fragmentation", "#ed8936"), ("free", "Free / external frag.", "#cbd5e0")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    left = [0.0] * len(results)
    for key, label, color in parts:
        vals = [100 * r.frac[key] for r in results]
        ax1.barh(names, vals, left=left, color=color, label=label)
        for i, (v, l) in enumerate(zip(vals, left)):
            if key == "token_states":
                ax1.text(l + v / 2, i, f"{v:.0f}%", va="center", ha="center", color="white", fontsize=9)
        left = [l + v for l, v in zip(left, vals)]
    ax1.set_xlim(0, 100)
    ax1.invert_yaxis()
    ax1.set_xlabel("% of KV cache memory (while requests wait for memory)")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False, fontsize=8)
    ax1.set_title("How KV cache memory is used")

    ax2.barh(names, [r.mean_batch for r in results], color="#2b6cb0")
    for i, r in enumerate(results):
        ax2.text(r.mean_batch, i, f" {r.mean_batch:.1f}", va="center", fontsize=9)
    ax2.set_xlim(0, max(r.mean_batch for r in results) * 1.18)
    ax2.invert_yaxis()
    ax2.set_yticklabels([])
    ax2.set_xlabel("mean requests running per step")
    ax2.set_title("Batch size")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="facebook/opt-13b")
    ap.add_argument("--kv-gb", type=float, default=12.0)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--trace", choices=["sharegpt", "alpaca"], default="sharegpt")
    ap.add_argument("--num-requests", type=int, default=1000)
    ap.add_argument("--rate", type=float, default=math.inf, help="requests per simulated step (default: all at once)")
    ap.add_argument("--systems", default=",".join(MEMORY_MANAGERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--plot", type=Path)
    ap.add_argument("--check", action="store_true", help="run memory invariant checks every step (slow)")
    args = ap.parse_args()

    spec = get_model_spec(args.model)
    cache = CacheConfig(block_size=args.block_size, kv_cache_gb=args.kv_gb, swap_space_gb=args.kv_gb)
    n_gpu, n_cpu = cache.num_gpu_blocks(spec), cache.num_cpu_blocks(spec)
    sched_cfg = SchedulerConfig(max_num_seqs=256, max_num_batched_tokens=2048, max_model_len=spec.max_model_len)
    reqs = synthetic_trace(args.trace, args.num_requests, args.rate, args.seed, spec.max_model_len)

    print(f"Model {spec.name}: {spec.kv_bytes_per_token / 1024:.0f} KB KV per token; "
          f"{args.kv_gb} GB pool = {n_gpu} blocks x {args.block_size} = {n_gpu * args.block_size} token slots")
    print(f"Trace: synthetic {args.trace} (approximate stand-in for the real dataset): {summarize(reqs)}\n")
    header = f"{'system':<12} {'tokens%':>8} {'reserved%':>10} {'int.frag%':>10} {'free%':>7} {'batch':>7} {'tok/step':>9} {'steps':>7} {'preempt':>8}"
    print(header)
    print("-" * len(header))
    results = []
    for system in args.systems.split(","):
        r = simulate(system, reqs, args.block_size, n_gpu, n_cpu, sched_cfg, check=args.check)
        results.append(r)
        f = r.frac
        print(f"{system:<12} {100 * f['token_states']:>7.1f}% {100 * f['reserved']:>9.1f}% "
              f"{100 * f['internal_frag']:>9.1f}% {100 * f['free']:>6.1f}% {r.mean_batch:>7.1f} "
              f"{r.tokens_per_step:>9.1f} {r.steps:>7} {r.preemptions:>8}")

    base = {r.system: r for r in results}
    if "paged" in base:
        print()
        for r in results:
            if r.system != "paged":
                print(f"paged vs {r.system:<12}: {base['paged'].mean_batch / r.mean_batch:.2f}x batch size, "
                      f"{r.steps / base['paged'].steps:.2f}x fewer steps to finish the trace")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["system", "token_states", "reserved", "internal_frag", "free", "mean_batch",
                        "tokens_per_step", "steps", "preemptions"])
            for r in results:
                w.writerow([r.system, *(round(r.frac[k], 4) for k in ("token_states", "reserved", "internal_frag", "free")),
                            round(r.mean_batch, 2), round(r.tokens_per_step, 2), r.steps, r.preemptions])
        print(f"\nWrote {args.csv}")
    if args.plot:
        plot(results, args.plot, f"{spec.name}, {args.kv_gb:g} GB KV cache, block size {args.block_size}, "
                                 f"synthetic {args.trace} ({len(reqs)} requests, all arriving at once)")
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
