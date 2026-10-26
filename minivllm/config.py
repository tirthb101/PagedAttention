"""Configuration objects.

Kept free of torch/transformers imports so the block manager, scheduler and the
GPU-free memory simulator can run anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

GiB = 2**30


@dataclass(frozen=True)
class ModelSpec:
    """The only model facts the memory manager needs: the size of one token's KV cache."""

    name: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_model_len: int
    dtype_bytes: int = 2  # FP16 / BF16

    @property
    def kv_bytes_per_token(self) -> int:
        # 2 (key and value) x layers x kv heads x head dim x bytes per element.
        # OPT-13B: 2 * 40 * 40 * 128 * 2 = 819,200 B, the paper's "800 KB" (§3).
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes


# Presets so the simulator does not need to download model configs.
MODEL_SPECS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in [
        ModelSpec("facebook/opt-125m", num_layers=12, num_kv_heads=12, head_dim=64, max_model_len=2048),
        ModelSpec("facebook/opt-1.3b", num_layers=24, num_kv_heads=32, head_dim=64, max_model_len=2048),
        ModelSpec("facebook/opt-6.7b", num_layers=32, num_kv_heads=32, head_dim=128, max_model_len=2048),
        ModelSpec("facebook/opt-13b", num_layers=40, num_kv_heads=40, head_dim=128, max_model_len=2048),
        # Grouped-query attention model for the extension experiments (8 KV heads shared by 32 query heads).
        ModelSpec("meta-llama/Meta-Llama-3-8B", num_layers=32, num_kv_heads=8, head_dim=128, max_model_len=8192),
    ]
}


def get_model_spec(name: str) -> ModelSpec:
    short = {s.split("/")[-1]: s for s in MODEL_SPECS}
    key = name if name in MODEL_SPECS else short.get(name)
    if key is None:
        raise KeyError(f"Unknown model {name!r}; known: {sorted(MODEL_SPECS)}")
    return MODEL_SPECS[key]


@dataclass
class CacheConfig:
    """How the KV cache memory is carved into blocks (§4.2)."""

    block_size: int = 16          # tokens per KV block; the paper's default
    kv_cache_gb: float = 12.0     # GPU KV pool; 12 GB = paper's OPT-13B on A100-40GB (Table 1)
    swap_space_gb: float = 12.0   # CPU pool; the paper bounds it by the GPU KV size (§4.5)
    watermark: float = 0.01       # keep ~1% of GPU blocks free on admission to avoid thrashing

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if not 0.0 <= self.watermark < 1.0:
            raise ValueError("watermark must be in [0, 1)")

    def block_bytes(self, model: ModelSpec) -> int:
        return self.block_size * model.kv_bytes_per_token

    def num_gpu_blocks(self, model: ModelSpec) -> int:
        return int(self.kv_cache_gb * GiB) // self.block_bytes(model)

    def num_cpu_blocks(self, model: ModelSpec) -> int:
        return int(self.swap_space_gb * GiB) // self.block_bytes(model)


class PreemptionMode(str, Enum):
    RECOMPUTE = "recompute"
    SWAP = "swap"


@dataclass
class SchedulerConfig:
    """Batching limits and the preemption policy (§4.5)."""

    max_num_seqs: int = 256              # sequences in one batch
    max_num_batched_tokens: int = 2048   # tokens processed in one step (prefill + decode)
    max_model_len: int = 2048            # longest sequence the model supports
    preemption_mode: PreemptionMode = PreemptionMode.RECOMPUTE

    def __post_init__(self) -> None:
        self.preemption_mode = PreemptionMode(self.preemption_mode)
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError("max_num_batched_tokens must be >= max_num_seqs (one decode token per sequence)")
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("max_num_batched_tokens must be >= max_model_len so any prompt can be prefilled")
