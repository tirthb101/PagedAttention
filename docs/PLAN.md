# Plan — 5 weeks to 28 November

**Deadline:** Sat 28 Nov 2026 (presentation). Work starts Fri 23 Oct.
**Load:** the first ~2 weeks overlap ~10 quizzes and 2 other paper presentations, so weeks 1–2 are deliberately light (~1–1.5 h/day each); weeks 3–5 carry the GPU work.
**GPU:** college SLURM cluster (L40S) via `sbatch` — see [CLUSTER.md](CLUSTER.md). Everything before Milestone 1 runs on a laptop CPU.
**Rule:** AI writes most code; we read, test, and must be able to explain every line.

Owners: **A** = memory management + scheduler, **B** = model + attention + benchmarks + cluster. If solo, do A's list first each week, then B's.

---

## Day 0 — Thu 22 Oct (today)

- [x] Pick paper: PagedAttention (SOSP 2023)
- [x] Documentation, plan, SLURM scripts (this folder)
- [ ] **Submit project preference on the Google Classroom post** (it was due around the middle of this week)
- [ ] Ask instructor: is §4.6 multi-GPU required? (We can do a 2-GPU version as a stretch goal — cluster allows 2 GPUs per job)
- [ ] Create private git repo from this folder; commit everything
- [ ] Cluster: check the "still to check" table in CLUSTER.md §1 (GPU model, internet on compute nodes, RAM, home quota)

## Milestones

| # | Date | What is true |
| --- | --- | --- |
| M0 | Thu 29 Oct | Cluster ready: conda env + models downloaded; `check_gpu.sbatch` passes. All memory-management code passes CPU tests |
| M1 | Thu 5 Nov | **Correctness:** our engine's greedy output == HuggingFace on OPT-125m, with forced preemption (swap and recompute) |
| M2 | Thu 12 Nov | **First real result:** paged vs Orca latency-vs-rate on OPT-1.3B from the cluster; sharing (parallel sampling, shared prefix, beam search) works |
| M3 | Thu 19 Nov | **All paper figures reproduced** on OPT-13B, Triton kernel in use |
| M4 | Tue 24 Nov | **Extension done**, experiments frozen, slides drafted |
| — | Sat 28 Nov | Present |

## Week 1 — Fri 23 → Thu 29 Oct: memory management (CPU only, light week)

> 26 Oct: **working prototype** — paged attention, cache engine, OPT model, sampler, engine and demo done; output matches HuggingFace exactly (57 tests). Parallel sampling and shared prefix work; memory-saving counter and beam search still to do.
>
> 22 Oct: A's week-1 and week-2 items written early: `minivllm/` config, sequence, block, block_manager (incl. shared prefix), scheduler; 43 tests pass (`python -m pytest tests`). **Both members: read these files and the tests this week.**

**A**
- [x] `config.py`, `sequence.py`, `block.py` (allocator) + tests
- [x] `block_manager.py`: allocate / append_slot / free + tests
- [x] `fork`, ref counts, copy-on-write, swap maps + tests
- [x] Invariant checker (ref counts == table entries; free + used == total)

**B**
- [ ] Cluster setup: `cluster.env`, `setup_env.sh` (start the OPT-13B + dataset downloads early — they are large)
- [ ] `sbatch slurm/check_gpu.sbatch` passes → **M0**
- [x] `model/opt.py` forward with a contiguous cache; matches HF logits on CPU (OPT-125m)

## Week 2 — Fri 30 Oct → Thu 5 Nov: paged attention + engine

**A**
- [x] `scheduler.py`: FCFS admission, latest-first preemption, recompute path + tests
- [x] Swap path (all-or-nothing, gang scheduling, no admission while swapped) + tests
- [x] `engine.py` step loop (with B)

**B**
- [x] `attention/paged_ref.py`: `write_kv` + gather attention; test vs plain attention with shuffled block placement
- [x] `cache_engine.py`: GPU/CPU pools, swap in/out, block copy
- [x] `sampler.py`: greedy / temperature / top-p
- [x] `tests/test_e2e_equivalence.py` passes on CPU → **M1 reached 26 Oct** (10 days early)
- [ ] Same tests + `examples/demo.py` with real OPT-125m on a GPU (Colab notebook or `test_gpu.sbatch`)

## Week 3 — Fri 6 → Thu 12 Nov: baselines, traces, sharing, first GPU curves

**A**
- [x] `reservation_manager.py`: Orca Oracle / Pow2 / Max (+ buddy rounding) + tests
- [x] Parallel sampling (n = 2, 4, 6) with COW; memory-saving counter
- [x] Shared prefix (`register_prefix`)
- [ ] Beam search (fork/free every step)

**B**
- [x] `benchmarks/trace.py`: synthetic ShareGPT / Alpaca, real ShareGPT / Alpaca loaders, Poisson arrivals, fixed seeds
- [ ] Run the simulator on the real ShareGPT / Alpaca files (needs the download + tokenizer); WMT16 loader for E4
- [x] `benchmarks/simulate_memory.py` (E0, laptop) — first chart for the slides
- [x] `benchmarks/run_serving.py` + `plot.py` (26 Oct; smoke-tested on CPU)
- [ ] Smoke job on OPT-125m, then E1 dry run on OPT-1.3B (find the request-rate range) → **M2**

## Week 4 — Fri 13 → Thu 19 Nov: full reproduction on OPT-13B

**A**
- [ ] E3 parallel sampling / beam search runs (Fig. 14–15)
- [ ] E4 shared prefix runs (Fig. 16)
- [ ] E7 block-size sweep, E8 swap vs recompute (Fig. 18b, 19)

**B**
- [x] Triton decode kernel; test vs reference (CPU interpreter); engine `attention_backend="triton"` (26 Oct)
- [ ] Triton kernel speed on GPU: `kernel_bench.py` on Colab T4, then cluster
- [ ] E6 kernel microbenchmark (Fig. 18a) — script `benchmarks/kernel_bench.py` written
- [ ] **E1–E2 headline runs on OPT-13B**: quick pass (10-min traces) first, then the final 1-hour-trace pass (= paper) submitted by Sat 14 Nov; it takes ~1.5 days on 2 GPUs
- [ ] E5 chatbot (Fig. 17)
- [ ] Fill the results table in EXPERIMENTS.md → **M3**

## Week 5 — Fri 20 → Sat 28 Nov: extension, slides, rehearsal

- [ ] (A+B) Extension X1 — automatic prefix caching — implement + tests (Fri 20 – Sun 22)
- [ ] (B) E9 extension runs on the cluster (Sun 22 – Mon 23)
- [ ] (A) Backup / second extension X2 — adaptive swap-vs-recompute — if X1 is done
- [ ] (Stretch) X6 — 2-GPU tensor parallelism (§4.6), only if everything else is done by 22 Nov
- [ ] **Tue 24 Nov: experiments frozen → M4**
- [ ] Slides (Tue 24 – Wed 25): problem → VM analogy → design → correctness → results vs paper → extension → limitations
- [ ] Rehearsals with timer + mock Q&A from PAPER_NOTES (Thu 26 – Fri 27)
- [ ] Record a short demo video as backup in case the cluster is down on the day
- [ ] Sat 28 Nov: present

## Cut order if behind (drop from the top)

1. 2-GPU tensor parallelism (X6) and the second extension (X2)
2. Chatbot experiment (Fig. 17)
3. Beam search (keep parallel sampling — it already demonstrates COW)
4. Triton kernel (report PyTorch reference; explain the paper's kernel fusions instead)
5. OPT-13B runs (use OPT-1.3B/6.7B with a smaller KV cap to create the same memory pressure)

**Never cut:** block manager, scheduler with both preemption modes, Orca baselines, the main latency-vs-rate experiment, one extension, the e2e correctness test.

## Risks

| Risk | Mitigation |
| --- | --- |
| Cluster queue waits near the deadline (everyone's projects) | Headline runs submitted by 14 Nov; OPT-1.3B results from week 3 as fallback |
| Final 1-hour-trace runs take ~1.5 days | Submit by 14 Nov; jobs may run up to 5 days; 10-min results are the fallback |
| Compute nodes have no internet | Everything pre-downloaded by `setup_env.sh` on the login node |
| Bookkeeping bugs (leaks, double frees) | Invariant checker after every step in debug mode |
| Triton kernel numerics | Always compare to the PyTorch reference (atol 1e-2 in FP16) |
| Quiz/presentation crunch in weeks 1–2 | Weeks 1–2 are CPU-only and light; slack in week 5 |
| Can't explain AI-written code | Nothing is merged until the other member can explain it in 2 minutes |

## Weekly habit

- Commit often with clear messages (the instructor watches commit history).
- End of each week: tick boxes here, update the results table, note deviations from DESIGN.md.
- Add Q&A items you learned to PAPER_NOTES.md.
