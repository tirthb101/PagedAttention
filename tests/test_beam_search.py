"""Beam search (§4.4) on the paged KV cache.

Beams fork every step (sharing all blocks), get copy-on-write copies when they write to a
shared block, and free their blocks when pruned. If any of that bookkeeping touched the
wrong block, the KV data would be corrupted and the beams would diverge from:

  * HuggingFace `generate(num_beams=...)` (when its result does not depend on version
    details: no EOS, or EOS with length_penalty=0), and
  * a plain no-cache beam search written below (all length penalties), which re-runs the
    full forward pass for every beam at every step.

Same tiny float64 OPT as test_e2e_equivalence.py; runs on a laptop CPU.
"""

import random
from collections import Counter

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from minivllm.config import SchedulerConfig  # noqa: E402
from minivllm.engine import LLMEngine  # noqa: E402
from minivllm.sequence import SamplingParams  # noqa: E402

VOCAB, MAX_TOKENS, WIDTH = 300, 12, 4


@pytest.fixture(scope="module")
def tiny_opt():
    torch.manual_seed(0)
    cfg = transformers.OPTConfig(
        vocab_size=VOCAB, hidden_size=64, num_hidden_layers=2, ffn_dim=128, num_attention_heads=4,
        max_position_embeddings=256, word_embed_proj_dim=64, dropout=0.0, attention_dropout=0.0,
        init_std=0.3,
    )
    return transformers.OPTForCausalLM(cfg).double().eval()


@pytest.fixture(scope="module")
def prompts():
    rng = random.Random(0)
    return [[rng.randrange(3, VOCAB) for _ in range(n)] for n in (1, 5, 16, 17, 33, 50)]


@pytest.fixture(scope="module")
def eos_token(tiny_opt, prompts):
    """A token the random model actually produces, so that beams really do finish early."""
    outs = make_engine(tiny_opt).generate(prompts, beam_params(ignore_eos=True))
    return Counter(t for o in outs for s in o.output_token_ids for t in s).most_common(1)[0][0]


def beam_params(width=WIDTH, **kw):
    return SamplingParams(n=width, use_beam_search=True, temperature=0.0, max_tokens=MAX_TOKENS, **kw)


def make_engine(model, block_size=4, gpu_blocks=512, mode="recompute", cpu_blocks=512, eos=None):
    sched = SchedulerConfig(max_num_seqs=64, max_num_batched_tokens=256, max_model_len=256, preemption_mode=mode)
    engine = LLMEngine(model, block_size=block_size, num_gpu_blocks=gpu_blocks, num_cpu_blocks=cpu_blocks,
                       scheduler_config=sched, watermark=0.0, check_invariants=True)
    if eos is not None:
        engine.eos_token_id = eos
    return engine


# ------------------------------------------------------------------ references
def hf_beam(model, prompt, width, eos=None, length_penalty=1.0):
    """HuggingFace beam search (with its own contiguous KV cache), best hypothesis first."""
    kw = dict(eos_token_id=None, min_new_tokens=MAX_TOKENS) if eos is None else dict(eos_token_id=eos)
    out = model.generate(
        torch.tensor([prompt]), attention_mask=torch.ones(1, len(prompt), dtype=torch.long),
        num_beams=width, num_return_sequences=width, do_sample=False, max_new_tokens=MAX_TOKENS,
        early_stopping=True, length_penalty=length_penalty, pad_token_id=1, **kw,
    )
    seqs = [s[len(prompt):].tolist() for s in out]
    return [s[: s.index(eos) + 1] if eos is not None and eos in s else s for s in seqs]


def reference_beam(model, prompt, width, eos=None, length_penalty=1.0):
    """Textbook beam search with no KV cache at all: every beam is re-run from scratch.

    Rules (same as HF's BeamSearchScorer with early_stopping=True): rank the best 2*width
    (beam, token) continuations; EOS among the top `width` becomes a finished hypothesis;
    the best `width` non-EOS continuations stay live; stop once `width` hypotheses have
    finished, or at max_tokens (then live beams compete too). Score = cum_logprob /
    out_len**length_penalty. Returns (tokens, cumulative logprob) best first.
    """
    live = [([], 0.0)]
    finished = []
    for step in range(MAX_TOKENS):
        cands = []
        with torch.no_grad():
            for toks, cum in live:
                logits = model(torch.tensor([prompt + toks])).logits[0, -1]
                lp = torch.log_softmax(logits, -1)
                for t in range(VOCAB):
                    cands.append((cum + float(lp[t]), toks, t))
        cands.sort(key=lambda c: c[0], reverse=True)
        new_live = []
        for rank, (score, toks, t) in enumerate(cands[: 2 * width]):
            if eos is not None and t == eos:
                if rank < width:
                    finished.append((toks + [t], score))
                continue
            new_live.append((toks + [t], score))
            if len(new_live) == width:
                break
        live = new_live
        last = step == MAX_TOKENS - 1
        if last:
            finished += live
        if len(finished) >= width or last:
            break
    finished.sort(key=lambda h: h[1] / len(h[0]) ** length_penalty, reverse=True)
    return finished[:width]


def assert_matches(out, ref):
    assert out.output_token_ids == [toks for toks, _ in ref]
    assert out.extra["cumulative_logprobs"] == pytest.approx([c for _, c in ref], abs=1e-9)


# ------------------------------------------------------------------ tests
@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_beam_search_matches_huggingface(tiny_opt, prompts, block_size):
    engine = make_engine(tiny_opt, block_size=block_size)
    outs = engine.generate(prompts, beam_params(ignore_eos=True))
    for prompt, out in zip(prompts, outs):
        assert out.output_token_ids == hf_beam(tiny_opt, prompt, WIDTH)
    assert engine.block_manager.get_num_free_gpu_blocks() == 512  # pruned and finished beams all freed


def test_beam_search_with_eos_matches_huggingface(tiny_opt, prompts, eos_token):
    engine = make_engine(tiny_opt, eos=eos_token)
    outs = engine.generate(prompts, beam_params(length_penalty=0.0))
    assert any(eos_token in s for o in outs for s in o.output_token_ids), "test must hit EOS"
    for prompt, out in zip(prompts, outs):
        assert out.output_token_ids == hf_beam(tiny_opt, prompt, WIDTH, eos=eos_token, length_penalty=0.0)
    assert engine.block_manager.get_num_free_gpu_blocks() == 512


@pytest.mark.parametrize("length_penalty", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("width", [1, 2, 4])
def test_beam_search_matches_no_cache_reference(tiny_opt, prompts, eos_token, width, length_penalty):
    engine = make_engine(tiny_opt, eos=eos_token)
    outs = engine.generate(prompts[:4], beam_params(width, length_penalty=length_penalty))
    for prompt, out in zip(prompts[:4], outs):
        assert_matches(out, reference_beam(tiny_opt, prompt, width, eos=eos_token, length_penalty=length_penalty))


def test_beam_search_survives_swap_preemption(tiny_opt, prompts):
    """6 requests x 4 beams in 40 blocks of 4 tokens: beam groups must be swapped out and
    back in (always swap: recomputing would lose the sharing) without changing the result."""
    engine = make_engine(tiny_opt, gpu_blocks=40, mode="swap")
    outs = engine.generate(prompts, beam_params(ignore_eos=True))
    assert engine.scheduler.num_preemptions > 0, "test must actually preempt"
    for prompt, out in zip(prompts, outs):
        assert out.output_token_ids == hf_beam(tiny_opt, prompt, WIDTH)
    assert engine.block_manager.get_num_free_gpu_blocks() == 40


def test_beams_share_blocks(tiny_opt, prompts):
    """The point of §4.4: beams share their common history instead of each holding a copy."""
    engine = make_engine(tiny_opt, block_size=4)
    engine.add_request("r", prompts[-1], beam_params(ignore_eos=True))   # 50-token prompt = 13 blocks
    saw_sharing = False
    while engine.has_unfinished():
        engine.step()
        live = [s for g in engine._groups.values() for s in g.seqs if not s.is_finished()]
        if len(live) == WIDTH:
            private = sum(s.num_logical_blocks for s in live)          # what 4 separate copies would need
            used = engine.memory_stats()["used_blocks"]
            assert used < private
            saw_sharing |= used <= private // 2
    assert saw_sharing
    assert engine.block_manager.get_num_free_gpu_blocks() == 512


def test_beam_search_needs_paged_manager_and_greedy_params(tiny_opt):
    sched = SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=256, max_model_len=256)
    orca = LLMEngine(tiny_opt, block_size=4, num_gpu_blocks=128, num_cpu_blocks=0, scheduler_config=sched,
                     memory_manager="orca-oracle")
    with pytest.raises(ValueError, match="paged"):
        orca.add_request("r", [5, 6, 7], beam_params())
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(n=2, use_beam_search=True, temperature=0.8)
