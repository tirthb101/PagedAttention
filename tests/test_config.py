import pytest

from minivllm.config import CacheConfig, SchedulerConfig, get_model_spec


def test_opt13b_kv_bytes_matches_paper():
    # Paper §3: 2 (K,V) x 5120 hidden x 40 layers x 2 bytes = 800 KB per token.
    spec = get_model_spec("facebook/opt-13b")
    assert spec.kv_bytes_per_token == 2 * 5120 * 40 * 2 == 819_200


def test_opt13b_12gb_pool_matches_table1():
    # Paper Table 1: 13B model, 12 GB KV cache -> max 15.7K token slots.
    spec = get_model_spec("opt-13b")
    cfg = CacheConfig(block_size=16, kv_cache_gb=12)
    slots = cfg.num_gpu_blocks(spec) * cfg.block_size
    assert 15_600 <= slots <= 15_800


def test_gqa_model_has_much_smaller_kv():
    llama = get_model_spec("Meta-Llama-3-8B")
    opt = get_model_spec("opt-13b")
    assert llama.kv_bytes_per_token == 131_072
    assert opt.kv_bytes_per_token / llama.kv_bytes_per_token > 6


def test_unknown_model():
    with pytest.raises(KeyError):
        get_model_spec("gpt-5")


def test_scheduler_config_validation():
    with pytest.raises(ValueError):
        SchedulerConfig(max_num_seqs=512, max_num_batched_tokens=256)
    cfg = SchedulerConfig(preemption_mode="swap")
    assert cfg.preemption_mode.value == "swap"
