"""The KV cache tensors and the data movement the scheduler asks for (§4.2, §4.5).

The block manager decides *which* blocks; this class owns the actual memory:
one key and one value tensor per layer on the GPU, and the same in CPU memory for
swapping. It performs swap-out, swap-in and copy-on-write copies with batched
index operations (the paper's "fused block copy").
"""

from __future__ import annotations

import torch


class CacheEngine:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.device = torch.device(device)
        shape_gpu = (num_gpu_blocks, block_size, num_kv_heads, head_dim)
        shape_cpu = (num_cpu_blocks, block_size, num_kv_heads, head_dim)
        pin = self.device.type == "cuda"
        self.gpu_cache = [
            (torch.zeros(shape_gpu, dtype=dtype, device=self.device),
             torch.zeros(shape_gpu, dtype=dtype, device=self.device))
            for _ in range(num_layers)
        ]
        self.cpu_cache = [
            (torch.zeros(shape_cpu, dtype=dtype, pin_memory=pin),
             torch.zeros(shape_cpu, dtype=dtype, pin_memory=pin))
            for _ in range(num_layers)
        ]

    @staticmethod
    def _move(src_cache, dst_cache, mapping: dict[int, int], device_src, device_dst) -> None:
        if not mapping:
            return
        src_idx = torch.tensor(list(mapping.keys()), dtype=torch.long)
        dst_idx = torch.tensor(list(mapping.values()), dtype=torch.long)
        for (src_k, src_v), (dst_k, dst_v) in zip(src_cache, dst_cache):
            s = src_idx.to(src_k.device)
            d = dst_idx.to(dst_k.device)
            dst_k[d] = src_k[s].to(dst_k.device, non_blocking=True)
            dst_v[d] = src_v[s].to(dst_v.device, non_blocking=True)

    def swap_out(self, mapping: dict[int, int]) -> None:
        """GPU block -> CPU block."""
        self._move(self.gpu_cache, self.cpu_cache, mapping, self.device, "cpu")

    def swap_in(self, mapping: dict[int, int]) -> None:
        """CPU block -> GPU block."""
        self._move(self.cpu_cache, self.gpu_cache, mapping, "cpu", self.device)

    def copy(self, pairs: list[tuple[int, int]]) -> None:
        """Copy-on-write copies inside the GPU pool, applied in order."""
        for src, dst in pairs:  # in order: a block freed earlier this step may be reused as a later dst
            for k, v in self.gpu_cache:
                k[dst] = k[src]
                v[dst] = v[src]
