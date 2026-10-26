"""End-to-end prototype demo: real OPT model, real text, paged KV cache.

    python examples/demo.py                          # facebook/opt-125m, CPU or GPU
    python examples/demo.py --model facebook/opt-1.3b --kv-blocks 64   # tiny pool -> preemption
    python examples/demo.py --tiny --kv-blocks 12    # no download: random tiny OPT, token ids only

Prints each completion, then what the memory manager did: blocks used, waste,
preemptions, and a check against HuggingFace's own generate().
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402  (OPT only for now)

from minivllm.config import SchedulerConfig  # noqa: E402
from minivllm.engine import LLMEngine  # noqa: E402
from minivllm.sequence import SamplingParams  # noqa: E402

PROMPTS = [
    "Four score and seven years ago our",
    "The capital of France is",
    "In computer systems, virtual memory is",
    "The best way to learn programming is",
    "Once upon a time, in a small village,",
    "A cache miss happens when",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--kv-blocks", type=int, default=None, help="GPU KV blocks (small number forces preemption)")
    ap.add_argument("--preemption", choices=["recompute", "swap"], default="recompute")
    ap.add_argument("--n", type=int, default=1, help="parallel samples per prompt (shares prompt blocks)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-check", action="store_true", help="skip the HuggingFace comparison")
    ap.add_argument("--tiny", action="store_true", help="offline smoke test: random tiny OPT, no download")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if args.tiny:
        from transformers import OPTConfig, OPTForCausalLM

        torch.manual_seed(0)
        cfg = OPTConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2, ffn_dim=128, num_attention_heads=4,
                        max_position_embeddings=2048, word_embed_proj_dim=64, init_std=0.3)
        model = OPTForCausalLM(cfg).to(device, dtype).eval()
        tok = None
        prompts = [[(31 * i + 7 * j) % 997 + 3 for j in range(len(p.split()) * 2)] for i, p in enumerate(PROMPTS)]
        print(f"Tiny random OPT on {device} (offline smoke test; outputs are token ids, not text)")
    else:
        print(f"Loading {args.model} on {device} ({dtype}) ...")
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
        prompts = PROMPTS

    sched = SchedulerConfig(max_num_seqs=64, max_num_batched_tokens=2048, max_model_len=2048, preemption_mode=args.preemption)
    engine = LLMEngine(
        model, tok, block_size=args.block_size,
        num_gpu_blocks=args.kv_blocks if args.kv_blocks else 2048, num_cpu_blocks=2048,
        scheduler_config=sched, check_invariants=True,
    )
    params = SamplingParams(n=args.n, temperature=args.temperature, max_tokens=args.max_tokens, ignore_eos=True, seed=0)

    t = time.perf_counter()
    outs = engine.generate(prompts, params)
    elapsed = time.perf_counter() - t

    for prompt, out in zip(prompts, outs):
        print("\n" + "=" * 72)
        print(f"PROMPT: {prompt}")
        shown = out.texts if out.texts is not None else [str(o) for o in out.output_token_ids]
        for i, text in enumerate(shown):
            label = f"[{i}] " if args.n > 1 else ""
            print(f"{label}OUTPUT: {text.strip()}")

    total_tokens = sum(len(o) for out in outs for o in out.output_token_ids)
    print("\n" + "=" * 72)
    print(f"Generated {total_tokens} tokens in {elapsed:.2f}s over {engine.num_steps} engine steps")
    print(f"KV blocks: {engine.block_manager.num_total_gpu_blocks} x {args.block_size} tokens; "
          f"preemptions: {engine.scheduler.num_preemptions} ({args.preemption})")
    print(f"All blocks returned to the free list: "
          f"{engine.block_manager.get_num_free_gpu_blocks() == engine.block_manager.num_total_gpu_blocks}")

    if not args.no_check and args.temperature == 0.0:
        same = 0
        # Reference: HuggingFace forward pass over the whole sequence each step, with no KV
        # cache at all, then argmax. (generate() is not used: it forbids EOS under
        # min_new_tokens, while our benchmarks ignore EOS, so they could legitimately differ.)
        for prompt, out in zip(prompts, outs):
            ids = list(prompt) if tok is None else tok(prompt).input_ids
            n = len(ids)
            with torch.no_grad():
                for _ in range(args.max_tokens):
                    logits = model(torch.tensor([ids], device=device)).logits[0, -1]
                    ids.append(int(torch.argmax(logits)))
            same += ids[n:] == out.output_token_ids[0]
        print(f"Matches HuggingFace (no-cache reference): {same}/{len(prompts)} prompts"
              + ("" if device == "cpu" else "  (FP16 on GPU: rare near-ties can flip a token)"))


if __name__ == "__main__":
    main()
