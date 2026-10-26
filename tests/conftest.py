import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minivllm.sequence import SamplingParams, Sequence, SequenceGroup  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU (run on the cluster with slurm/test_gpu.sbatch)")


_ids = itertools.count()


def make_group(
    prompt_len: int,
    block_size: int,
    n: int = 1,
    arrival: float = 0.0,
    output_len: int = 8,
    prefix_tokens: list[int] | None = None,
    prefix_id: str | None = None,
    request_id: str | None = None,
) -> SequenceGroup:
    """A request whose prompt is `prompt_len` tokens (optionally starting with a shared prefix)."""
    prompt = list(prefix_tokens or [])
    prompt += [1000 + i for i in range(prompt_len - len(prompt))]
    seqs = [Sequence(next(_ids), prompt, block_size) for _ in range(n)]
    return SequenceGroup(
        request_id or f"req{next(_ids)}",
        seqs,
        SamplingParams(n=n, max_tokens=output_len, ignore_eos=True),
        arrival_time=arrival,
        true_output_len=output_len,
        prefix_id=prefix_id,
    )


@pytest.fixture
def new_seq_id():
    return lambda: next(_ids)
