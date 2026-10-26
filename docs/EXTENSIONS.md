# Extensions — going beyond the paper

The instructor asks us to find something the authors could have done (an extra experiment, a new feature, or an issue) and do it. Below are the candidates, scored for our 2-week window, then our pick.

## Candidates

| # | Extension | Type | Effort | Builds on | Why it is interesting |
| --- | --- | --- | --- | --- | --- |
| X1 | **Automatic prefix caching** across requests | New feature | Medium (2 days) | block manager, ref counts | Paper only shares *predefined* prefixes. Real chat traffic repeats system prompts and conversation history — reuse it automatically, like an OS page cache |
| X2 | **Adaptive swap-vs-recompute** per request | Fix a limitation | Low (1 day) | scheduler | Paper picks one mode globally, yet its own Fig. 19 shows the best choice depends on block size and sequence length |
| X3 | **Grouped-query-attention models** (e.g. Llama-3-8B) | New experiment | Medium (2 days, new model file) | model | 2023 OPT models store K/V for every head; modern GQA models store far less. Does paging still give 2–4×? |
| X4 | Chunked prefill (Sarathi-Serve, OSDI 2024) | New feature | Medium | scheduler | Long prompts stall running decodes; paper does not measure per-token latency jitter |
| X5 | FP8 KV cache (Ada supports FP8) | New feature | Medium–High | kernels | Halves KV bytes → bigger batches; interacts with paging |
| X6 | 2-way tensor parallelism (paper §4.6) on 2 GPUs | Paper feature, stretch | High | model, cache engine | Cluster allows 2 GPUs per job: one centralized block manager, each GPU stores KV for half the heads. Would cover the last unimplemented paper section |

## Our pick: X1 — Automatic prefix caching (backup: X2)

### Idea

Today, when a request finishes, its blocks go straight back to the free list and their contents are forgotten. Instead:

1. Give every **full** block a content hash: `hash(parent_block_hash, token_ids_in_this_block)`. Chaining the parent hash means equal hashes imply equal *prefixes*, not just equal 16-token chunks.
2. Keep a table `hash → physical block`.
3. On `allocate`, walk the new prompt block by block; for each full block whose hash is in the table, map that physical block into the block table (ref_count++) instead of allocating and computing it. Only the uncached suffix is prefilled.
4. When a block's ref count reaches 0, don't erase it — put it on an **LRU list of evictable cached blocks**. Allocation takes from truly free blocks first, then evicts the least-recently-used cached block.

**CSAPP link:** this is the OS **page cache** with LRU replacement (Ch 6 replacement policies + Ch 9 sharing), layered on top of the paper's paging.

### Code changes

- `block_manager.py`: `content_hash` on `PhysicalBlock`; `cached_blocks: dict[hash, PhysicalBlock]`; `evictor: OrderedDict` (LRU); `allocate()` returns how many prompt tokens are already cached.
- `scheduler.py` / `engine.py`: prefill only `prompt[num_cached:]`; positions and slot mapping start at `num_cached`.
- `attention/paged_ref.py`: prefill attention must attend to cached blocks too (context_len = full prompt).
- Tests: second identical request does zero prefill for full blocks; LRU evicts oldest; outputs identical with/without caching.

### Experiment (E9)

- **Workload A — multi-turn chat (ShareGPT):** send each conversation turn as a new request whose prompt includes the full history. Without caching, every turn re-prefills the whole history.
- **Workload B — shared system prompt:** 1,000-token system prompt + short user questions (Alpaca) — like a real deployed assistant.
- **Metrics:** prefix-cache hit rate; time-to-first-token (TTFT); normalized latency vs request rate; capacity gain over the paper's vLLM (paged, no caching).
- **Hypothesis:** TTFT drops sharply and capacity rises on both workloads; the gain grows with conversation length/prefix length; under heavy memory pressure the hit rate falls because cached blocks are evicted — an interesting trade-off to plot (hit rate vs KV pool size).

### How to present it

"The paper shares prefixes only when the operator declares them in advance. We made sharing automatic with content-hashed blocks and an LRU page cache — the same thing operating systems do with file pages. vLLM itself later added this feature, which confirms it was a real gap."

## Backup: X2 — Adaptive swap vs recompute

- **Idea:** at preemption time, estimate `swap_cost = 2 × num_blocks × block_bytes / PCIe_bandwidth` and `recompute_cost ≈ prefill_time(seq_len)` (from a small profiled table); pick the cheaper per request.
- **Code:** ~50 lines in `scheduler.py` plus a one-time profiling script.
- **Experiment:** E8 sweep with `--preemption adaptive`; expect it to match the better of the two at every block size.
- **Use if:** X1 runs into trouble by Day 12, or as a second, smaller extension if time remains.

## Also worth one slide (no code): issues we noticed

- FCFS + "latest-first" preemption can starve long requests under sustained overload — no fairness study in the paper.
- Swapping blocks all new admissions — head-of-line blocking at high load.
- The strongest baseline, Orca (Oracle), assumes output lengths are known in advance, which no real system knows. Realistic Orca sits between Pow2 and Max, so the paper's headline comparison against Oracle is conservative.
