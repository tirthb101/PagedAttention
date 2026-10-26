"""Request traces for the experiments (§6.1).

A trace is a list of requests, each with a prompt length, an output length and an
arrival time. As in the paper:

  * lengths come from ShareGPT (long, highly varied conversations) or Alpaca (short
    instructions); the paper reports ShareGPT inputs 8.4x and outputs 5.8x longer;
  * arrival times follow a Poisson process at a chosen request rate (the datasets
    have no timestamps);
  * requests are filtered like vLLM's benchmark: prompt < 1024 tokens, prompt + output
    <= 2048 (OPT's context), and very short prompts/outputs dropped.

Two sources:
  1. the real datasets (needs the JSON file and a tokenizer) -> use for final results;
  2. a synthetic stand-in with similar statistics (lognormal lengths, no download) ->
     use for development, tests and the laptop simulator. Always label which one a
     plot used.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Approximate mean token lengths of the two datasets as used in the paper (Fig. 11).
# Treat as approximate; the real-dataset loader is the source of truth.
SYNTHETIC_STATS = {
    #            mean input, mean output, lognormal sigma
    "sharegpt": (161.0, 338.0, 1.0),
    "alpaca": (19.3, 58.5, 0.7),
}


@dataclass
class Request:
    request_id: str
    prompt_len: int
    output_len: int
    arrival_time: float = 0.0
    prompt_token_ids: Optional[list[int]] = None

    def token_ids(self, vocab_size: int = 50_000, seed: int = 0) -> list[int]:
        """Real tokens if loaded from a dataset, otherwise deterministic dummy ids."""
        if self.prompt_token_ids is not None:
            return self.prompt_token_ids
        rng = random.Random(f"{seed}-{self.request_id}")
        return [rng.randrange(4, vocab_size) for _ in range(self.prompt_len)]


def keep(prompt_len: int, output_len: int, max_model_len: int = 2048) -> bool:
    return 4 <= prompt_len < 1024 and output_len >= 4 and prompt_len + output_len <= max_model_len


def poisson_arrivals(n: int, rate: float, rng: random.Random) -> list[float]:
    """Arrival times for n requests at `rate` requests/second. rate=inf -> all at t=0 (offline)."""
    if math.isinf(rate):
        return [0.0] * n
    t, times = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        times.append(t)
    return times


def _lognormal_int(rng: random.Random, mean: float, sigma: float) -> int:
    mu = math.log(mean) - sigma**2 / 2  # so that E[X] = mean
    return max(1, int(round(rng.lognormvariate(mu, sigma))))


def synthetic_trace(name: str, num_requests: int, rate: float = math.inf, seed: int = 0,
                    max_model_len: int = 2048) -> list[Request]:
    if name not in SYNTHETIC_STATS:
        raise ValueError(f"unknown synthetic trace {name!r}; choose from {sorted(SYNTHETIC_STATS)}")
    mean_in, mean_out, sigma = SYNTHETIC_STATS[name]
    rng = random.Random(seed)
    lens: list[tuple[int, int]] = []
    while len(lens) < num_requests:
        p, o = _lognormal_int(rng, mean_in, sigma), _lognormal_int(rng, mean_out, sigma)
        if keep(p, o, max_model_len):
            lens.append((p, o))
    times = poisson_arrivals(num_requests, rate, rng)
    return [Request(f"{name}-{i}", p, o, t) for i, ((p, o), t) in enumerate(zip(lens, times))]


def load_sharegpt(path: str | Path, tokenizer, num_requests: int, rate: float = math.inf, seed: int = 0,
                  max_model_len: int = 2048) -> list[Request]:
    """ShareGPT_V3_unfiltered_cleaned_split.json: first human turn = prompt, first reply = output."""
    data = json.loads(Path(path).read_text())
    rng = random.Random(seed)
    rng.shuffle(data)
    reqs: list[Request] = []
    for conv in data:
        turns = conv.get("conversations", [])
        if len(turns) < 2:
            continue
        prompt_ids = tokenizer(turns[0]["value"]).input_ids
        output_len = len(tokenizer(turns[1]["value"]).input_ids)
        if keep(len(prompt_ids), output_len, max_model_len):
            reqs.append(Request(f"sharegpt-{len(reqs)}", len(prompt_ids), output_len, 0.0, prompt_ids))
        if len(reqs) == num_requests:
            break
    for r, t in zip(reqs, poisson_arrivals(len(reqs), rate, rng)):
        r.arrival_time = t
    return reqs


def load_alpaca(path: str | Path, tokenizer, num_requests: int, rate: float = math.inf, seed: int = 0,
                max_model_len: int = 2048) -> list[Request]:
    """Alpaca as JSON (alpaca_data.json) or the HF parquet: instruction (+input) = prompt, output = output."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd

        data = pd.read_parquet(path).to_dict("records")
    else:
        data = json.loads(path.read_text())
    rng = random.Random(seed)
    rng.shuffle(data)
    reqs: list[Request] = []
    for ex in data:
        prompt = ex["instruction"] + (("\n" + ex["input"]) if ex.get("input") else "")
        prompt_ids = tokenizer(prompt).input_ids
        output_len = len(tokenizer(ex["output"]).input_ids)
        if keep(len(prompt_ids), output_len, max_model_len):
            reqs.append(Request(f"alpaca-{len(reqs)}", len(prompt_ids), output_len, 0.0, prompt_ids))
        if len(reqs) == num_requests:
            break
    for r, t in zip(reqs, poisson_arrivals(len(reqs), rate, rng)):
        r.arrival_time = t
    return reqs


def summarize(reqs: list[Request]) -> str:
    n = len(reqs)
    mi = sum(r.prompt_len for r in reqs) / n
    mo = sum(r.output_len for r in reqs) / n
    return f"{n} requests, mean input {mi:.1f} tokens, mean output {mo:.1f} tokens"


SHAREGPT_REPO, SHAREGPT_FILE = "anon8231489123/ShareGPT_Vicuna_unfiltered", "ShareGPT_V3_unfiltered_cleaned_split.json"
ALPACA_REPO = "tatsu-lab/alpaca"


def load_trace(name: str, tokenizer, num_requests: int, rate: float = math.inf, seed: int = 0,
               max_model_len: int = 2048) -> list[Request]:
    """'sharegpt' / 'alpaca' = real data from the Hugging Face cache (downloaded by slurm/setup_env.sh;
    works offline once cached). 'synthetic-sharegpt' / 'synthetic-alpaca' = no download."""
    if name.startswith("synthetic-"):
        return synthetic_trace(name.removeprefix("synthetic-"), num_requests, rate, seed, max_model_len)
    from huggingface_hub import hf_hub_download, snapshot_download

    if name == "sharegpt":
        path = hf_hub_download(SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset")
        return load_sharegpt(path, tokenizer, num_requests, rate, seed, max_model_len)
    if name == "alpaca":
        root = Path(snapshot_download(ALPACA_REPO, repo_type="dataset", allow_patterns=["*.parquet"]))
        return load_alpaca(next(root.rglob("*.parquet")), tokenizer, num_requests, rate, seed, max_model_len)
    raise ValueError(f"unknown trace {name!r}")
