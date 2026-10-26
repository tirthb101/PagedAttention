"""Milestone M1: our engine must produce exactly the same tokens as HuggingFace.

Uses a tiny randomly initialised OPT model in float64 (no download, runs on a laptop
CPU in seconds). The reference is the HuggingFace model run *without any KV cache*:
full forward pass over the whole sequence every step, then argmax. If our block tables,
slot mapping, paged attention, swapping or copy-on-write were wrong anywhere, the
tokens would diverge.
"""

import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from minivllm.config import SchedulerConfig  # noqa: E402
from minivllm.engine import LLMEngine  # noqa: E402
from minivllm.sequence import SamplingParams  # noqa: E402

VOCAB, MAX_TOKENS = 300, 12


@pytest.fixture(scope="module")
def tiny_opt():
    torch.manual_seed(0)
    cfg = transformers.OPTConfig(
        vocab_size=VOCAB, hidden_size=64, num_hidden_layers=2, ffn_dim=128, num_attention_heads=4,
        max_position_embeddings=256, word_embed_proj_dim=64, dropout=0.0, attention_dropout=0.0,
        init_std=0.3,  # large weights so attention strongly affects the output (sensitive test)
    )
    return transformers.OPTForCausalLM(cfg).double().eval()


@pytest.fixture(scope="module")
def prompts():
    rng = random.Random(0)
    return [[rng.randrange(3, VOCAB) for _ in range(n)] for n in (1, 5, 16, 17, 33, 50)]


@pytest.fixture(scope="module")
def reference(tiny_opt, prompts):
    """Greedy continuation with plain HuggingFace forward passes, no cache at all.

    Returns (tokens, logprobs) per prompt. Comparing logprobs too makes the test sensitive:
    corrupted KV data changes the probabilities even when the argmax token happens to survive.
    """
    return [hf_greedy(tiny_opt, p) for p in prompts]


def hf_greedy(model, prompt):
    ids, logprobs = list(prompt), []
    with torch.no_grad():
        for _ in range(MAX_TOKENS):
            logits = model(torch.tensor([ids])).logits[0, -1]
            token = int(torch.argmax(logits))
            logprobs.append(float(torch.log_softmax(logits, -1)[token]))
            ids.append(token)
    return ids[len(prompt):], logprobs


def assert_same(engine_outputs, refs, seq_index=0):
    for out, (tokens, logprobs) in zip(engine_outputs, refs):
        assert out.output_token_ids[seq_index] == tokens
        assert out.extra["logprobs"][seq_index] == pytest.approx(logprobs, abs=1e-9)


GREEDY = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)


def make_engine(model, block_size, gpu_blocks, mode="recompute", cpu_blocks=256, memory_manager="paged"):
    sched = SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=256, max_model_len=256, preemption_mode=mode)
    return LLMEngine(model, block_size=block_size, num_gpu_blocks=gpu_blocks, num_cpu_blocks=cpu_blocks,
                     scheduler_config=sched, watermark=0.0, check_invariants=True, memory_manager=memory_manager)


@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_matches_huggingface(tiny_opt, prompts, reference, block_size):
    engine = make_engine(tiny_opt, block_size, gpu_blocks=512)
    outs = engine.generate(prompts, GREEDY)
    assert_same(outs, reference)
    assert engine.block_manager.get_num_free_gpu_blocks() == 512  # everything freed


@pytest.mark.parametrize("mode", ["recompute", "swap"])
def test_matches_huggingface_under_preemption(tiny_opt, prompts, reference, mode):
    # 6 requests need ~50 blocks of 4 tokens at peak; give them 20 to force preemption.
    engine = make_engine(tiny_opt, block_size=4, gpu_blocks=20, mode=mode)
    outs = engine.generate(prompts, GREEDY)
    assert engine.scheduler.num_preemptions > 0, "test must actually preempt"
    assert_same(outs, reference)


def test_parallel_sampling_shares_prompt_and_stays_correct(tiny_opt, prompts, reference):
    """n=3 greedy samples share the prompt blocks (copy-on-write on the first write);
    all three must still equal the single greedy answer."""
    engine = make_engine(tiny_opt, block_size=4, gpu_blocks=512)
    params = SamplingParams(n=3, temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)
    outs = engine.generate(prompts, params)
    for i in range(3):
        assert_same(outs, reference, seq_index=i)


def test_shared_prefix_is_reused_and_correct(tiny_opt):
    rng = random.Random(1)
    prefix = [rng.randrange(3, VOCAB) for _ in range(20)]       # 5 full blocks of 4
    suffixes = [[rng.randrange(3, VOCAB) for _ in range(n)] for n in (3, 9)]
    engine = make_engine(tiny_opt, block_size=4, gpu_blocks=512)
    engine.register_prefix("p", prefix)
    expected = [hf_greedy(tiny_opt, prefix + s) for s in suffixes]

    # First request computes the prefix; the second must reuse it (20 tokens skipped).
    engine.add_request("a", prefix + suffixes[0], GREEDY, prefix_id="p")
    outs = {}
    while engine.has_unfinished():
        for o in engine.step():
            outs[o.request_id] = o
    before = engine.num_prefill_tokens
    engine.add_request("b", prefix + suffixes[1], GREEDY, prefix_id="p")
    while engine.has_unfinished():
        for o in engine.step():
            outs[o.request_id] = o
    assert_same([outs["a"], outs["b"]], expected)
    # b's prefill skipped the 20 cached prefix tokens: only its suffix was computed.
    assert engine.num_prefill_tokens - before == len(suffixes[1])


def test_sampling_is_reproducible_with_seed(tiny_opt, prompts):
    params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=MAX_TOKENS, ignore_eos=True, seed=123)
    a = make_engine(tiny_opt, 4, 512).generate(prompts[:3], params)
    b = make_engine(tiny_opt, 4, 512).generate(prompts[:3], params)
    assert [o.output_token_ids for o in a] == [o.output_token_ids for o in b]


@pytest.mark.parametrize("manager", ["orca-oracle", "orca-pow2", "orca-max"])
def test_orca_baselines_give_identical_output(tiny_opt, prompts, reference, manager):
    """The baselines change only memory management, so the text must be identical.
    A small pool makes requests wait for memory (Orca never preempts)."""
    engine = make_engine(tiny_opt, block_size=4, gpu_blocks=128, memory_manager=manager)
    outs = engine.generate(prompts, GREEDY)
    assert_same(outs, reference)
    assert engine.block_manager.get_num_free_gpu_blocks() == 128


def test_orca_parallel_sampling_recomputes_prompt_per_sequence(tiny_opt, prompts, reference):
    engine = make_engine(tiny_opt, block_size=4, gpu_blocks=512, memory_manager="orca-oracle")
    params = SamplingParams(n=2, temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)
    outs = engine.generate(prompts, params)
    for i in range(2):
        assert_same(outs, reference, seq_index=i)
    # No sharing: every prompt token was computed twice.
    assert engine.num_prefill_tokens == 2 * sum(len(p) for p in prompts)
