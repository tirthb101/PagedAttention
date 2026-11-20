"""Turn logits into next tokens: greedy, temperature and top-p sampling, and beam search.

Parallel sampling (n > 1) just samples n times from the same prompt logits, giving each
sequence its own token. Beam search (§4.4) scores candidates across a whole request;
`beam_candidates` ranks them here, and the engine turns the winners into fork/free calls
on the block manager.
"""

from __future__ import annotations

import torch

from minivllm.sequence import SamplingParams, Sequence


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


def beam_candidates(
    parent_rows: list[tuple[Sequence, torch.Tensor]], num_candidates: int
) -> list[tuple[float, Sequence, int, float]]:
    """Rank every (beam, next token) continuation of one request by cumulative logprob.

    parent_rows: one [vocab] logits row per live beam. Returns the best `num_candidates`
    as (score, parent, token_id, token_logprob), best first. The engine asks for 2*width
    candidates so that up to `width` of them may end in EOS and there are still `width`
    live beams left (the same rule as HuggingFace's BeamSearchScorer).
    """
    logprobs = torch.stack([torch.log_softmax(row.float() if row.dtype in (torch.float16, torch.bfloat16)
                                              else row, dim=-1) for _, row in parent_rows])
    base = torch.tensor([seq.cumulative_logprob for seq, _ in parent_rows],
                        dtype=logprobs.dtype, device=logprobs.device)
    scores = (logprobs + base[:, None]).flatten()
    k = min(num_candidates, scores.numel())
    top_scores, top_idx = torch.topk(scores, k)
    vocab = logprobs.shape[1]
    out = []
    for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
        b, token = divmod(idx, vocab)
        out.append((score, parent_rows[b][0], token, float(logprobs[b, token])))
    return out
