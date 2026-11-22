"""Extension X1: automatic prefix caching (content-hashed blocks + LRU page cache).

Block-manager tests run without torch; the end-to-end tests check, on a tiny float64 OPT,
that caching changes only how much is computed, never the generated tokens.
"""

import random

import pytest

from minivllm.block_manager import AllocStatus, BlockSpaceManager
from minivllm.sequence import SamplingParams, Sequence, SequenceGroup, SequenceStatus

from conftest import make_group

B = 4


def cbm(gpu=16, cpu=16, caching=True):
    return BlockSpaceManager(B, gpu, cpu, watermark=0.0, enable_prefix_caching=caching)


def group_of(tokens, n=1, rid=None, ids=iter(range(10_000, 10**9))):
    seqs = [Sequence(next(ids), tokens, B) for _ in range(n)]
    return SequenceGroup(rid or f"g{seqs[0].seq_id}", seqs, SamplingParams(n=n, max_tokens=8, ignore_eos=True), 0.0)


def admit(bm, group):
    assert bm.can_allocate(group) == AllocStatus.OK
    bm.allocate(group)
    for s in group.get_seqs(SequenceStatus.WAITING):
        s.status = SequenceStatus.RUNNING
    bm.check_invariants(group.seqs)
    return group.seqs


def finish(bm, seq):
    seq.status = SequenceStatus.FINISHED_LENGTH
    bm.free(seq)
    bm.check_invariants()


# ---------------------------------------------------------------- block manager
def test_identical_prompt_hits_full_blocks_of_live_request():
    bm = cbm()
    prompt = list(range(100, 110))                  # 10 tokens: blocks 4 + 4 + 2
    (a,) = admit(bm, group_of(prompt))
    assert a.num_computed_tokens == 0
    (b,) = admit(bm, group_of(prompt))
    ta, tb = bm.block_tables[a.seq_id], bm.block_tables[b.seq_id]
    assert tb[:2] == ta[:2] and tb[2] is not ta[2]   # 2 full blocks shared, partial one private
    assert ta[0].ref_count == 2
    assert b.num_computed_tokens == 8
    assert bm.cache_stats()["cache_hit_tokens"] == 8
    assert bm.get_num_free_gpu_blocks() == 16 - 4


def test_last_full_block_is_not_hit():
    """With len % B == 0 the last block holds the token that must be recomputed (for its
    logits); writing it into a shared cached block is not allowed, so that block is private."""
    bm = cbm()
    prompt = list(range(100, 108))                  # exactly 2 full blocks
    (a,) = admit(bm, group_of(prompt))
    (b,) = admit(bm, group_of(prompt))
    ta, tb = bm.block_tables[a.seq_id], bm.block_tables[b.seq_id]
    assert tb[0] is ta[0] and tb[1] is not ta[1]
    assert b.num_computed_tokens == 4


def test_freed_blocks_stay_cached_and_are_reused():
    bm = cbm()
    prompt = list(range(100, 113))                  # 3 full blocks + 1 token
    (a,) = admit(bm, group_of(prompt))
    blocks = list(bm.block_tables[a.seq_id])
    finish(bm, a)
    assert bm.get_num_free_gpu_blocks() == 16       # evictable blocks count as free
    assert bm.gpu_allocator.get_num_evictable_blocks() == 3   # the partial block is not cached
    (b,) = admit(bm, group_of(prompt))
    assert bm.block_tables[b.seq_id][:3] == blocks[:3]
    assert b.num_computed_tokens == 12
    assert bm.gpu_allocator.get_num_evictable_blocks() == 0


def test_different_history_same_chunk_is_a_miss():
    """Hashes are chained: an equal 4-token chunk after a different prefix must not hit."""
    bm = cbm()
    (a,) = admit(bm, group_of([1, 2, 3, 4, 9, 9, 9, 9, 5]))
    (b,) = admit(bm, group_of([7, 7, 7, 7, 9, 9, 9, 9, 5]))
    assert b.num_computed_tokens == 0
    assert not set(map(id, bm.block_tables[a.seq_id])) & set(map(id, bm.block_tables[b.seq_id]))


def test_hit_stops_at_first_miss():
    bm = cbm()
    (a,) = admit(bm, group_of(list(range(100, 117))))         # 4 full blocks + 1
    other = list(range(100, 104)) + [0, 0, 0, 0] + list(range(108, 117))
    (b,) = admit(bm, group_of(other))
    assert b.num_computed_tokens == 4                          # only block 0 matches


def test_lru_evicts_oldest_and_tail_first():
    bm = cbm(gpu=6)
    p1, p2 = list(range(100, 109)), list(range(200, 209))    # 2 full blocks + 1 each
    (a,) = admit(bm, group_of(p1))
    finish(bm, a)                                              # p1 blocks cached (oldest)
    (b,) = admit(bm, group_of(p2))
    finish(bm, b)                                              # p2 blocks cached
    assert bm.gpu_allocator.get_num_evictable_blocks() == 4
    # A new 9-token prompt needs 3 blocks: 2 truly free, then 1 eviction from p1's tail.
    (c,) = admit(bm, group_of(list(range(300, 309))))
    assert bm.gpu_allocator.num_evictions == 1

    def num_hits(prompt):
        g = group_of(prompt)
        return len(bm._cache_hits(g, g.seqs))

    assert num_hits(p1) == 1                                   # p1's tail (block 1) went first
    assert num_hits(p2) == 2                                   # p2 (more recent) fully cached


def test_can_allocate_counts_live_hits_as_free():
    bm = cbm(gpu=4)
    prompt = list(range(100, 113))                  # needs 4 blocks
    admit(bm, group_of(prompt))
    assert bm.get_num_free_gpu_blocks() == 0
    # Without caching this would wait (LATER); with caching only 1 new block is needed... still 0 free.
    assert bm.can_allocate(group_of(prompt)) == AllocStatus.LATER
    bm2 = cbm(gpu=5)
    admit(bm2, group_of(prompt))
    assert bm2.can_allocate(group_of(prompt)) == AllocStatus.OK   # 3 hits + 1 new block


def test_decode_filled_blocks_are_registered():
    """A block filled by generated tokens is cached when the next block is allocated, so a
    follow-up chat turn (prompt = history + answer + new message) reuses the answer too."""
    bm = cbm()
    (a,) = admit(bm, group_of(list(range(100, 106))))           # 6 tokens
    for t in range(500, 504):                                   # grow to 10 tokens
        a.append_token(t)
        a.num_computed_tokens = a.get_len() - 1
        bm.append_slot(a)
    a.num_computed_tokens = a.get_len()
    history = a.get_token_ids()
    finish(bm, a)
    (b,) = admit(bm, group_of(history + [7, 7, 7]))
    assert b.num_computed_tokens == 8                           # both full blocks hit


def test_parallel_sampling_group_shares_hits():
    bm = cbm()
    prompt = list(range(100, 110))
    admit(bm, group_of(prompt))
    s1, s2, s3 = admit(bm, group_of(prompt, n=3))
    t = bm.block_tables
    assert t[s1.seq_id][:2] == t[s2.seq_id][:2] == t[s3.seq_id][:2]
    assert t[s1.seq_id][0].ref_count == 4
    assert all(s.num_computed_tokens == 8 for s in (s1, s2, s3))


def test_swap_round_trip_with_caching():
    bm = cbm(gpu=8, cpu=8)
    g = group_of(list(range(100, 110)))
    (a,) = admit(bm, g)
    bm.swap_out(g)
    a.status = SequenceStatus.SWAPPED
    bm.check_invariants([a])
    assert bm.gpu_allocator.get_num_evictable_blocks() == 2     # GPU copies stay cached
    (b,) = admit(bm, group_of(list(range(100, 110))))           # can still hit them
    assert b.num_computed_tokens == 8
    bm.swap_in(g)
    a.status = SequenceStatus.RUNNING
    bm.check_invariants([a, b])


def test_disabled_by_default():
    bm = BlockSpaceManager(B, 16, 16, watermark=0.0)
    prompt = list(range(100, 110))
    admit(bm, group_of(prompt))
    (b,) = admit(bm, group_of(prompt))
    assert b.num_computed_tokens == 0
    assert bm.cache_stats()["cache_hit_tokens"] == 0


def test_predefined_prefix_groups_are_not_looked_up():
    bm = cbm()
    prefix = list(range(10))
    bm.register_prefix("p", prefix)
    g = make_group(prompt_len=14, block_size=B, prefix_tokens=prefix, prefix_id="p")
    admit(bm, g)
    bm.mark_prefix_computed("p")
    (s,) = admit(bm, make_group(prompt_len=14, block_size=B, prefix_tokens=prefix, prefix_id="p"))
    assert s.num_computed_tokens == 8


# ---------------------------------------------------------------- end to end (tiny model)
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from minivllm.config import SchedulerConfig  # noqa: E402
from minivllm.engine import LLMEngine  # noqa: E402

VOCAB, MAX_TOKENS = 300, 12
GREEDY = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)


@pytest.fixture(scope="module")
def tiny_opt():
    torch.manual_seed(0)
    cfg = transformers.OPTConfig(
        vocab_size=VOCAB, hidden_size=64, num_hidden_layers=2, ffn_dim=128, num_attention_heads=4,
        max_position_embeddings=256, word_embed_proj_dim=64, dropout=0.0, attention_dropout=0.0, init_std=0.3)
    return transformers.OPTForCausalLM(cfg).double().eval()


def hf_greedy(model, prompt):
    ids, logprobs = list(prompt), []
    with torch.no_grad():
        for _ in range(MAX_TOKENS):
            logits = model(torch.tensor([ids])).logits[0, -1]
            token = int(torch.argmax(logits))
            logprobs.append(float(torch.log_softmax(logits, -1)[token]))
            ids.append(token)
    return ids[len(prompt):], logprobs


def make_engine(model, gpu_blocks, caching=True, mode="recompute", block_size=4):
    sched = SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=256, max_model_len=256, preemption_mode=mode)
    return LLMEngine(model, block_size=block_size, num_gpu_blocks=gpu_blocks, num_cpu_blocks=256,
                     scheduler_config=sched, watermark=0.0, check_invariants=True,
                     enable_prefix_caching=caching)


def run_all(engine):
    outs = {}
    while engine.has_unfinished():
        for o in engine.step():
            outs[o.request_id] = o
    return outs


def check(out, ref):
    tokens, logprobs = ref
    assert out.output_token_ids[0] == tokens
    assert out.extra["logprobs"][0] == pytest.approx(logprobs, abs=1e-9)


@pytest.fixture(scope="module")
def shared_prompts():
    rng = random.Random(3)
    system = [rng.randrange(3, VOCAB) for _ in range(23)]
    return [system + [rng.randrange(3, VOCAB) for _ in range(n)] for n in (1, 4, 9, 13, 2, 7)]


@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_same_tokens_with_and_without_caching(tiny_opt, shared_prompts, block_size):
    refs = [hf_greedy(tiny_opt, p) for p in shared_prompts]
    on = make_engine(tiny_opt, 512, caching=True, block_size=block_size)
    off = make_engine(tiny_opt, 512, caching=False, block_size=block_size)
    for engine in (on, off):
        # first request alone, then the rest together (some hit in the same step)
        engine.add_request("r0", shared_prompts[0], GREEDY)
        outs = run_all(engine)
        for i, p in enumerate(shared_prompts[1:], 1):
            engine.add_request(f"r{i}", p, GREEDY)
        outs.update(run_all(engine))
        for i, ref in enumerate(refs):
            check(outs[f"r{i}"], ref)
    assert on.num_prefill_tokens < off.num_prefill_tokens
    assert on.stats()["cache_hit_rate_pct"] > 40  # 23-token prefix: >= 16 of ~25 tokens at block 16


def test_same_step_hits_are_correct(tiny_opt, shared_prompts):
    """All requests admitted in ONE step: later ones hit blocks the first is writing in
    that same step (every layer writes all batch K/V before attending)."""
    refs = [hf_greedy(tiny_opt, p) for p in shared_prompts]
    engine = make_engine(tiny_opt, 512)
    for i, p in enumerate(shared_prompts):
        engine.add_request(f"r{i}", p, GREEDY)
    outs = run_all(engine)
    for i, ref in enumerate(refs):
        check(outs[f"r{i}"], ref)
    assert engine.stats()["cache_hit_tokens"] > 0


def test_multi_turn_chat_reuses_history_and_answer(tiny_opt):
    rng = random.Random(5)
    engine = make_engine(tiny_opt, 512)
    history = [rng.randrange(3, VOCAB) for _ in range(10)]
    for turn in range(3):
        before = engine.num_prefill_tokens
        engine.add_request(f"t{turn}", history, GREEDY)
        out = run_all(engine)[f"t{turn}"]
        check(out, hf_greedy(tiny_opt, history))
        if turn > 0:
            # Everything in full blocks of the previous turn (incl. its answer) was reused.
            assert engine.num_prefill_tokens - before <= len(history) - ((prev_len - 1) // 4) * 4
        prev_len = len(history) + MAX_TOKENS
        history = history + out.output_token_ids[0] + [rng.randrange(3, VOCAB) for _ in range(5)]


@pytest.mark.parametrize("mode", ["recompute", "swap"])
def test_correct_under_preemption_and_eviction(tiny_opt, shared_prompts, mode):
    refs = [hf_greedy(tiny_opt, p) for p in shared_prompts]
    engine = make_engine(tiny_opt, 24, mode=mode)  # tight pool: preemption + LRU eviction
    outs = engine.generate(shared_prompts, GREEDY)
    for out, ref in zip(outs, refs):
        check(out, ref)
    assert engine.scheduler.num_preemptions > 0, "test must actually preempt"
    assert engine.block_manager.get_num_free_gpu_blocks() == 24


def test_orca_rejects_caching(tiny_opt):
    with pytest.raises(ValueError):
        LLMEngine(tiny_opt, block_size=4, num_gpu_blocks=64, num_cpu_blocks=0,
                  memory_manager="orca-oracle", enable_prefix_caching=True)
