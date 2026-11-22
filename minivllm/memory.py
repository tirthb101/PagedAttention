"""Pick a KV-cache memory manager by name (torch-free, shared by the engine and the simulator)."""

from __future__ import annotations

from minivllm.block_manager import BlockSpaceManager
from minivllm.reservation_manager import POLICIES, ReservationBlockManager

MEMORY_MANAGERS = ("paged",) + POLICIES


def make_block_manager(kind: str, block_size: int, num_gpu_blocks: int, num_cpu_blocks: int,
                       watermark: float = 0.01, max_model_len: int = 2048, enable_prefix_caching: bool = False):
    """'paged' = PagedAttention (vLLM); 'orca-oracle' / 'orca-pow2' / 'orca-max' = the paper's baselines.

    enable_prefix_caching turns on extension X1 (automatic prefix caching); paged only.
    """
    if kind == "paged":
        return BlockSpaceManager(block_size, num_gpu_blocks, num_cpu_blocks, watermark=watermark,
                                 enable_prefix_caching=enable_prefix_caching)
    if enable_prefix_caching:
        raise ValueError("prefix caching (X1) needs the paged memory manager")
    if kind in POLICIES:
        return ReservationBlockManager(block_size, num_gpu_blocks, num_cpu_blocks, policy=kind,
                                       max_model_len=max_model_len)
    raise ValueError(f"unknown memory manager {kind!r}; choose from {MEMORY_MANAGERS}")
