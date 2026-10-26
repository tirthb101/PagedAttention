# mini-vLLM: Reimplementing PagedAttention

CS612 Computer Systems — Project 2 (Part B: implement a paper end to end).

**Paper:** Woosuk Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023. [arXiv 2309.06180](https://arxiv.org/abs/2309.06180)

**One-line summary:** LLM servers waste 60–80% of the GPU memory reserved for the KV cache because each request gets one contiguous, max-sized chunk. The paper stores the KV cache in fixed-size *blocks* addressed through a per-request *block table* — the same idea as pages and page tables in OS virtual memory — and gets 2–4× higher serving throughput.

## Goals

1. Build a small LLM serving engine that implements every single-GPU idea in the paper (§4.1–4.5, §5).
2. Reproduce the paper's experiments (§6–7) at single-GPU scale on an NVIDIA L40S (48 GB).
3. Add one thing the authors did not do (instructor requirement) — see [docs/EXTENSIONS.md](docs/EXTENSIONS.md).
4. Present it in 30 minutes and answer any question about the paper or the code.

## Scope

| Paper part | Status in this project |
| --- | --- |
| §4.1 PagedAttention | In scope — PyTorch reference first, Triton kernel second |
| §4.2 KV cache manager (block tables, allocator) | In scope — core of the project |
| §4.3 Decoding with on-demand block allocation | In scope |
| §4.4 Parallel sampling, beam search, shared prefix (ref counts + copy-on-write) | In scope (beam search is the first thing cut if late) |
| §4.5 Scheduling: FCFS, all-or-nothing preemption, swap and recompute | In scope |
| §4.6 Distributed (tensor parallel, multi-GPU) | Stretch goal — the cluster allows 2 GPUs per job, so a 2-way version is possible if time remains (EXTENSIONS X6); otherwise explained in the talk |
| §5 FastAPI / OpenAI-compatible frontend | Out of scope — a trace-replay script replaces it |
| §6 Orca (Oracle / Pow2 / Max) baselines | In scope — reimplemented inside our engine, as the authors did |
| §6 FasterTransformer baseline | Optional — approximated by static batching |
| §6 OPT-66B / OPT-175B results | Out of scope — do not fit on one GPU |

## Documentation

| File | What it is for |
| --- | --- |
| [docs/PAPER_NOTES.md](docs/PAPER_NOTES.md) | The paper explained, key numbers, and a Q&A bank for the talk |
| [docs/DESIGN.md](docs/DESIGN.md) | Architecture of our implementation, data structures, interfaces, paper → code map |
| [docs/PLAN.md](docs/PLAN.md) | Five-week plan to the 28 Nov deadline: milestones, weekly checklists, owners, cut order |
| [docs/CLUSTER.md](docs/CLUSTER.md) | How we run on the college SLURM cluster: setup, job arrays, storage, habits |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Every figure we reproduce: setup, metric, script, expected result |
| [docs/EXTENSIONS.md](docs/EXTENSIONS.md) | Candidate "beyond the paper" additions and the one we pick |

## Planned repository layout

```
PagedAttention/
├── README.md
├── docs/                      # this documentation
├── minivllm/                  # the engine
│   ├── config.py              # EngineConfig: block size, KV budget, policies
│   ├── sequence.py            # Sequence, SequenceGroup, SamplingParams
│   ├── block.py               # PhysicalBlock, BlockAllocator (GPU + CPU pools)
│   ├── block_manager.py       # §4.2/4.4: block tables, ref counts, COW, swap
│   ├── reservation_manager.py # Orca baselines: Oracle / Pow2 / Max
│   ├── scheduler.py           # §4.5: FCFS, preemption (swap / recompute)
│   ├── cache_engine.py        # KV tensors on GPU/CPU, swap + copy ops
│   ├── attention/
│   │   ├── paged_ref.py       # PagedAttention in plain PyTorch (correctness oracle)
│   │   └── paged_triton.py    # fast Triton kernel (§5 kernel optimisations)
│   ├── model/opt.py           # OPT decoder wired to paged attention
│   ├── sampler.py             # greedy / temperature / top-p / beam
│   └── engine.py              # LLMEngine: add_request(), step()
├── benchmarks/
│   ├── trace.py               # ShareGPT / Alpaca lengths, Poisson arrivals
│   ├── simulate_memory.py     # GPU-free simulator: memory waste, batch size
│   ├── run_serving.py         # latency vs request rate (Fig. 12–17)
│   ├── kernel_bench.py        # Fig. 18a
│   └── plot.py
├── tests/                     # unit tests + paged-vs-HuggingFace equivalence
├── slurm/                     # cluster.env.example, setup_env.sh, *.sbatch job scripts
├── logs/                      # SLURM job logs
├── requirements.txt
└── results/                   # CSVs and plots copied back from the cluster
```

## Hardware and models

- **GPU:** NVIDIA L40S, 48 GB GDDR6, 864 GB/s, on the college SLURM cluster (batch jobs via `sbatch`; see [docs/CLUSTER.md](docs/CLUSTER.md)). No local GPU: all non-GPU work and unit tests run on a laptop CPU.
- **Development model:** `facebook/opt-125m` (runs on a laptop CPU for tests).
- **Experiment models:** `facebook/opt-1.3b` for fast sweeps, `facebook/opt-13b` (≈26 GB FP16) for the headline result — the same model as the paper's single-GPU setup.
- **Extension model:** a grouped-query-attention model such as Llama-3-8B (see EXTENSIONS).

We use OPT because the paper uses OPT, so per-token KV sizes and memory pressure match the paper directly.

## Status

Deadline: **28 Nov 2026** — see [docs/PLAN.md](docs/PLAN.md).

| Part | State |
| --- | --- |
| Docs, plan, SLURM scripts | Done (22 Oct) |
| Memory core: `config`, `sequence`, `block`, `block_manager`, `scheduler` | Done (22 Oct), pure Python |
| **Working prototype:** paged attention (PyTorch reference), cache engine, OPT model, sampler, engine, demo | Done (26 Oct) — output matches HuggingFace token-for-token, including under preemption and parallel sampling |
| Orca baselines (buddy allocator), traces, GPU-free memory simulator | Done (26 Oct) — reproduces Fig. 2: Oracle 38.2% vs paged 97.0% of memory used by real tokens |
| Triton paged decode kernel (§5) + kernel benchmark (Fig. 18a) | Done (26 Oct) — matches the reference (checked in Triton's CPU interpreter); speed to be measured on GPU |
| Serving benchmark `run_serving.py` + `plot.py` (Fig. 12–17) | Done (26 Oct) — smoke-tested on CPU; real runs on the cluster next |
| Beam search, E3 memory-saving counter, WMT16 loader | Next |

**82 tests pass** (≈1 min on a laptop CPU). The engine tests were checked by deliberately breaking the code (wrong slot, skipped swap or copy-on-write copy, no causal mask): each break makes a test fail.

### Run the prototype

```bash
pip install torch transformers pytest
python -m pytest                                   # all tests (tiny random model, no download)
python examples/demo.py --tiny --kv-blocks 12 --block-size 4   # offline smoke test, forces preemption
python examples/demo.py                            # real OPT-125m (downloads ~250 MB), real text
python examples/demo.py --kv-blocks 8 --preemption swap        # tiny KV pool: preemption by swapping
python examples/demo.py --n 3 --temperature 0.8 --no-check     # parallel sampling, shared prompt blocks
```

On Google Colab (T4 GPU): open `colab/run_prototype.ipynb` and run the cells — tests, demo, preemption, parallel sampling, Triton kernel benchmark, and a small paged-vs-Orca serving run.

### Serving benchmark (latency vs request rate)

```bash
# smoke test anywhere
python benchmarks/run_serving.py --tiny --trace synthetic-sharegpt --rates 2,8 --duration 10 --out results/smoke.csv
# on a GPU: one CSV per system, then plot them together
for mm in paged orca-oracle orca-pow2 orca-max; do
  python benchmarks/run_serving.py --model facebook/opt-1.3b --trace sharegpt --rates 1,2,4,8 \
      --duration 600 --kv-cache-gb 4 --memory-manager $mm --out results/e1/sharegpt_$mm.csv
done
python benchmarks/plot.py results/e1/*.csv --out results/e1_sharegpt.png
python benchmarks/kernel_bench.py --model-dims opt-13b      # Fig. 18a
```

On the cluster, `slurm/serve_sweep.sbatch` runs exactly this as a job array (docs/CLUSTER.md).

### Memory simulator (no GPU, no download, ~10 s)

```bash
pip install matplotlib
python benchmarks/simulate_memory.py --plot results/e0_sharegpt.png   # paper Fig. 2 + Fig. 13 style chart
python benchmarks/simulate_memory.py --trace alpaca --num-requests 2000
```

## Team

- Member A: _name_ — memory management + scheduler
- Member B: _name_ — model, attention kernels, benchmarks

AI assistance is permitted for this project (instructor's Project 2 rules). Every module must still be explainable by both members.
