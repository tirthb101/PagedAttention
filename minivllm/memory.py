"""Pick a KV-cache memory manager by name (torch-free, shared by the engine and the simulator)."""

from __future__ import annotations

from minivllm.block_manager import BlockSpaceManager
from minivllm.reservation_manager import POLICIES, ReservationBlockManager

MEMORY_MANAGERS = ("paged",) + POLICIES


def make_block_manager(kind: str, block_size: int, num_gpu_blocks: int, num_cpu_blocks: int,
                       watermark: float = 0.01, max_model_len: int = 2048):
    """'paged' = PagedAttention (vLLM); 'orca-oracle' / 'orca-pow2' / 'orca-max' = the paper's baselines."""
    if kind == "paged":
        return BlockSpaceManager(block_size, num_gpu_blocks, num_cpu_blocks, watermark=watermark)
    if kind in POLICIES:
        return ReservationBlockManager(block_size, num_gpu_blocks, num_cpu_blocks, policy=kind,
                                       max_model_len=max_model_len)
    raise ValueError(f"unknown memory manager {kind!r}; choose from {MEMORY_MANAGERS}")
