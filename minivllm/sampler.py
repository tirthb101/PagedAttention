"""Turn logits into next tokens: greedy, temperature and top-p sampling.

Beam search (§4.4) needs scores across a whole group and fork/free calls on the block
manager; it is added later in engine.py. Parallel sampling (n > 1) just samples n times
from the same prompt logits, giving each sequence its own token.
"""

from __future__ import annotations

import torch

from minivllm.sequence import SamplingParams


def sample(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator | None) -> tuple[int, float]:
    """Sample one token from a [vocab] logits row. Returns (token_id, logprob)."""
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    if params.temperature == 0.0:
        logprobs = torch.log_softmax(logits, dim=-1)
        token = int(torch.argmax(logits))
        return token, float(logprobs[token])

    logits = logits / params.temperature
    probs = torch.softmax(logits, dim=-1)
    if params.top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = sorted_probs.cumsum(-1)
        keep = cumulative - sorted_probs < params.top_p  # always keeps the top token
        filtered = torch.zeros_like(probs)
        filtered[sorted_idx[keep]] = sorted_probs[keep]
        probs = filtered / filtered.sum()
    token = int(torch.multinomial(probs.cpu(), 1, generator=generator))
    return token, float(torch.log(probs[token]))
