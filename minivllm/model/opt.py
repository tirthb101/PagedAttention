"""OPT decoder whose attention uses our paged KV cache.

We reuse the HuggingFace model's layers (embeddings, linear layers, layer norms) so the
weights are loaded exactly as HuggingFace loads them, and rewrite only the forward pass:
every attention layer writes its new K/V into the block pool (write_kv) and attends
through the block tables (paged_attention). The KV cache therefore belongs entirely to us.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from minivllm.attention import paged_attention, write_kv


@dataclass
class InputMetadata:
    """Everything the model needs for one step, built by the engine from the block tables."""

    positions: torch.Tensor          # [num_tokens] absolute position of each token
    slot_mapping: torch.Tensor       # [num_tokens] flat KV slot for each token
    block_tables: list[list[int]]    # per attention sequence
    context_lens: list[int]          # per attention sequence
    query_lens: list[int]            # per attention sequence (new tokens this step)

    @property
    def last_token_indices(self) -> torch.Tensor:
        """Index of each sequence's last new token in the flat batch (where logits are taken)."""
        ends = torch.tensor(self.query_lens, device=self.positions.device).cumsum(0)
        return ends - 1


class PagedOPT(nn.Module):
    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        cfg = hf_model.config
        if cfg.model_type != "opt":
            raise ValueError(f"PagedOPT supports OPT models, got {cfg.model_type}")
        self.config = cfg
        self.decoder = hf_model.model.decoder
        self.lm_head = hf_model.lm_head
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_attention_heads  # OPT has full multi-head attention
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.pos_offset = getattr(self.decoder.embed_positions, "offset", 2)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,                          # [num_tokens]
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        meta: InputMetadata,
    ) -> torch.Tensor:
        """Returns logits for the last new token of every attention sequence: [num_seqs, vocab]."""
        dec, cfg = self.decoder, self.config
        h = dec.embed_tokens(input_ids)
        if getattr(dec, "project_in", None) is not None:
            h = dec.project_in(h)
        h = h + dec.embed_positions.weight[meta.positions + self.pos_offset]

        num_tokens = input_ids.shape[0]
        for layer, (k_cache, v_cache) in zip(dec.layers, kv_caches):
            attn = layer.self_attn
            residual = h
            if cfg.do_layer_norm_before:
                h = layer.self_attn_layer_norm(h)
            q = attn.q_proj(h).view(num_tokens, self.num_heads, self.head_dim)
            k = attn.k_proj(h).view(num_tokens, self.num_kv_heads, self.head_dim)
            v = attn.v_proj(h).view(num_tokens, self.num_kv_heads, self.head_dim)
            write_kv(k, v, k_cache, v_cache, meta.slot_mapping)
            o = paged_attention(
                q, k_cache, v_cache, meta.block_tables, meta.context_lens, meta.query_lens,
                scale=attn.scaling,
            )
            h = residual + attn.out_proj(o.reshape(num_tokens, -1))
            if not cfg.do_layer_norm_before:
                h = layer.self_attn_layer_norm(h)

            residual = h
            if cfg.do_layer_norm_before:
                h = layer.final_layer_norm(h)
            h = layer.fc2(layer.activation_fn(layer.fc1(h)))
            h = residual + h
            if not cfg.do_layer_norm_before:
                h = layer.final_layer_norm(h)

        if getattr(dec, "final_layer_norm", None) is not None:
            h = dec.final_layer_norm(h)
        if getattr(dec, "project_out", None) is not None:
            h = dec.project_out(h)
        return self.lm_head(h[meta.last_token_indices])
