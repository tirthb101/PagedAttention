"""Memory-saving counter (paper Fig. 14/15): (logical blocks - physical blocks) / logical blocks.

logical  = blocks the live sequences would need if nothing were shared (sum of block-table lengths)
physical = distinct GPU blocks they actually use
"""

import pytest

from minivllm.block_manager import BlockSpaceManager
from minivllm.reservation_manager import ReservationBlockManager
from minivllm.sequence import SamplingParams, SequenceStatus

from conftest import make_group

B = 4


def admit(bm, group):
    bm.allocate(group)
    for s in group.get_seqs(SequenceStatus.WAITING):
        s.status = SequenceStatus.RUNNING
    return group.get_seqs(SequenceStatus.RUNNING)


def grow(bm, seqs):
    for i, s in enumerate(seqs):
        s.num_computed_tokens = s.get_len()
        s.append_token(i)
        bm.append_slot(s)


def test_empty_manager_saves_nothing():
    bm = BlockSpaceManager(B, 32, 32, watermark=0.0)
    assert bm.sharing_stats() == {"logical_blocks": 0, "physical_blocks": 0, "saving": 0.0}


def test_hand_built_parallel_sampling_case():
    """Prompt of 4 full blocks shared by n=4 samples, then each sample adds 1 private block."""
    bm = BlockSpaceManager(B, 64, 64, watermark=0.0)
    seqs = admit(bm, make_group(prompt_len=16, block_size=B, n=4))
    assert bm.sharing_stats() == {"logical_blocks": 16, "physical_blocks": 4, "saving": 0.75}

    grow(bm, seqs)  # token 17 of each sample starts a new private block
    st = bm.sharing_stats()
    assert (st["logical_blocks"], st["physical_blocks"]) == (20, 8)
    assert st["saving"] == pytest.approx(0.60)

    bm.free(seqs[0])  # a finished sample no longer counts
    st = bm.sharing_stats()
    assert (st["logical_blocks"], st["physical_blocks"]) == (15, 7)


def test_unshared_requests_save_nothing():
    bm = BlockSpaceManager(B, 64, 64, watermark=0.0)
    admit(bm, make_group(prompt_len=10, block_size=B))
    admit(bm, make_group(prompt_len=5, block_size=B))
    assert bm.sharing_stats() == {"logical_blocks": 5, "physical_blocks": 5, "saving": 0.0}


def test_beam_style_fork_is_counted():
    """Beam search shares through fork(): a child copies the table, not the blocks."""
    bm = BlockSpaceManager(B, 64, 64, watermark=0.0)
    group = make_group(prompt_len=12, block_size=B)  # 3 blocks
    (parent,) = admit(bm, group)
    for _ in range(2):
        child = parent.fork(10_000 + len(group.seqs))
        bm.fork(parent, child)
        group.add(child)
    st = bm.sharing_stats()
    assert (st["logical_blocks"], st["physical_blocks"]) == (9, 3)
    assert st["saving"] == pytest.approx(2 / 3)


def test_swapped_out_sequences_are_not_counted():
    bm = BlockSpaceManager(B, 64, 64, watermark=0.0)
    g1 = make_group(prompt_len=16, block_size=B, n=4)
    admit(bm, g1)
    admit(bm, make_group(prompt_len=8, block_size=B))
    bm.swap_out(g1)
    for s in g1.seqs:
        s.status = SequenceStatus.SWAPPED
    assert bm.sharing_stats() == {"logical_blocks": 2, "physical_blocks": 2, "saving": 0.0}


def test_orca_never_saves():
    bm = ReservationBlockManager(B, 128, policy="orca-oracle", max_model_len=64)
    admit(bm, make_group(prompt_len=16, block_size=B, n=4, output_len=4))  # 5 blocks -> 8 (buddy) each
    st = bm.sharing_stats()
    assert st["logical_blocks"] == st["physical_blocks"] == 32 and st["saving"] == 0.0


# ------------------------------------------------------------- engine time average

try:
    import torch  # noqa: F811
    import transformers
except ImportError:  # pragma: no cover
    torch = transformers = None


@pytest.mark.skipif(torch is None or transformers is None, reason="needs torch + transformers")
@pytest.mark.parametrize("manager", ["paged", "orca-oracle"])
def test_engine_averages_saving_over_steps(manager):
    """n=4, 16-token prompt, 4 output tokens, block size 4. Per model step (tables at step time):
        step 1 prefill   : 16 logical / 4 physical  -> 75 %
        steps 2-4 decode : 20 logical / 8 physical  -> 60 %   (tokens 17-19 in a private 5th block)
    avg = (75 + 3*60) / 4 = 63.75 %, max = 75 %, block-weighted = (76 - 28) / 76."""
    from minivllm.config import SchedulerConfig
    from minivllm.engine import LLMEngine

    torch.manual_seed(0)
    cfg = transformers.OPTConfig(vocab_size=300, hidden_size=32, num_hidden_layers=1, ffn_dim=64,
                                 num_attention_heads=2, max_position_embeddings=64, word_embed_proj_dim=32)
    model = transformers.OPTForCausalLM(cfg).double().eval()
    sched = SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=64, max_model_len=64)
    engine = LLMEngine(model, block_size=B, num_gpu_blocks=64, num_cpu_blocks=64, scheduler_config=sched,
                       watermark=0.0, check_invariants=True, memory_manager=manager)
    params = SamplingParams(n=4, temperature=1.0, max_tokens=4, ignore_eos=True, seed=0)
    engine.generate([list(range(10, 26))], params)
    st = engine.stats()
    assert st["num_steps"] == 4
    if manager == "paged":
        assert st["mem_saving_avg_pct"] == pytest.approx(63.75)
        assert st["mem_saving_max_pct"] == pytest.approx(75.0)
        assert st["mem_saving_blocks_pct"] == pytest.approx(100 * 48 / 76)
    else:
        assert st["mem_saving_avg_pct"] == st["mem_saving_max_pct"] == st["mem_saving_blocks_pct"] == 0.0
    engine.reset_stats()
    assert engine.stats()["mem_saving_max_pct"] == 0.0 and engine.stats()["num_steps"] == 0
