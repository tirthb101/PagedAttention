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


def sample_batch(
    logits: torch.Tensor,
    params: list[SamplingParams],
    generators: list[torch.Generator | None],
) -> list[tuple[int, float]]:
    """Sample one token for every row of a [m, vocab] logits matrix, all on the logits device.

    Same distribution as `sample`, but with a single host sync per step instead of one per
    sequence. Random rows use the exponential-race form of categorical sampling:
    argmax(p / E) with E ~ Exp(1) drawn from that row's own generator, so each sequence
    consumes exactly `vocab` draws per step and its stream does not depend on batching.
    Generators must live on the same device as `logits`.
    """
    m = logits.shape[0]
    if m == 0:
        return []
    logits = logits.float()
    temps = torch.tensor([p.temperature for p in params], device=logits.device, dtype=logits.dtype)
    greedy = temps == 0.0
    scaled = logits / torch.where(greedy, torch.ones_like(temps), temps)[:, None]
    logprobs = torch.log_softmax(scaled, dim=-1)

    top_ps = [p.top_p for p in params]
    if any(tp < 1.0 for tp in top_ps):
        probs = logprobs.exp()
        top_p = torch.tensor(top_ps, device=logits.device, dtype=logits.dtype)
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        keep_sorted = (sorted_probs.cumsum(-1) - sorted_probs) < top_p[:, None]  # always keeps the top token
        keep = torch.zeros_like(keep_sorted).scatter(-1, sorted_idx, keep_sorted)
        keep |= greedy[:, None]
        logprobs = torch.log_softmax(scaled.masked_fill(~keep, float("-inf")), dim=-1)

    scores = logprobs.clone()
    rand_rows = (~greedy).nonzero().flatten().tolist() if not bool(greedy.all()) else []
    if rand_rows:
        noise = torch.empty((len(rand_rows), logits.shape[1]), device=logits.device, dtype=logits.dtype)
        for j, i in enumerate(rand_rows):
            noise[j].exponential_(generator=generators[i])
        idx = torch.tensor(rand_rows, device=logits.device)
        scores[idx] = logprobs[idx] - noise.log()  # argmax(log p - log E) == argmax(p / E)
    tokens = scores.argmax(dim=-1)
    token_lp = logprobs.gather(1, tokens[:, None]).squeeze(1)
    out = torch.stack([tokens.to(logprobs.dtype), token_lp]).cpu()  # the one host sync
    return [(int(t), float(lp)) for t, lp in zip(out[0].tolist(), out[1].tolist())]


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
