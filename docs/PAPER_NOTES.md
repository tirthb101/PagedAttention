# Paper notes — PagedAttention / vLLM (SOSP 2023)

Read this before the talk. Section numbers refer to the paper ([arXiv 2309.06180](https://arxiv.org/abs/2309.06180)).
Numbers marked *(verify)* were read off a figure; check them against the PDF before putting them on a slide.

---

## 1. Background you need (§2)

- **Autoregressive generation.** A model produces one token at a time; each new token depends on all previous ones.
- **Two phases per request.**
  - *Prompt (prefill) phase:* all prompt tokens are processed in one parallel pass. Compute-bound.
  - *Generation (decode) phase:* one token per step. Memory-bound — the GPU spends most time reading weights and KV cache.
- **KV cache.** In every attention layer, each token produces a key vector and a value vector. Later tokens attend to all earlier keys/values, so they are cached instead of recomputed.
- **Batching.** Because decode is memory-bound, serving many requests together raises throughput almost for free. Batch size is limited by **how many KV caches fit in memory**. That is the whole game.
- **Iteration-level (continuous) batching** (from Orca, OSDI 2022): requests join and leave the batch at every step instead of waiting for the whole batch to finish. vLLM keeps this.

## 2. The problem (§1, §3)

**The KV cache is big and its size is unknown in advance.**

- OPT-13B: one token's KV cache = 2 (K and V) × 5120 (hidden size) × 40 (layers) × 2 bytes (FP16) = **800 KB**.
- A 2048-token request can need **1.6 GB**.
- On an A100 40 GB serving a 13B model: ~**65%** of memory is weights, ~**30%** is KV cache, the rest activations.

**Existing systems (Orca, FasterTransformer) allocate each request one contiguous chunk sized for the maximum possible length.** This wastes memory three ways (§3.1):

1. **Reserved slots** — space for tokens not generated yet (it will be used eventually, but blocks other requests now).
2. **Internal fragmentation** — the request ends before reaching the reserved length; the rest is never used.
3. **External fragmentation** — gaps between chunks left by the allocator (e.g., buddy allocator).

Result (Fig. 2): only **20.4%–38.2%** of KV cache memory holds real token states in existing systems. vLLM's bar is ≈**96%** *(verify)*.

**Memory sharing is also impossible** with contiguous chunks: parallel sampling and beam search share the prompt, but each copy gets its own chunk.

**CSAPP mapping:** this is exactly the contiguous-allocation problem that virtual memory solves (Ch 9), plus internal/external fragmentation from `malloc` (9.9).

## 3. The idea: PagedAttention (§4.1)

- Split each sequence's KV cache into **KV blocks** of a fixed number of tokens *B* (default **16**).
- Blocks can live **anywhere** in GPU memory — not contiguous.
- The attention kernel is rewritten to fetch keys/values **block by block** using a lookup table.
- Mathematically identical to normal attention, so **no accuracy change**.

## 4. The KV cache manager (§4.2) — the virtual memory analogy

| OS concept | vLLM |
| --- | --- |
| Page | KV block |
| Byte | Token |
| Process | Sequence (request) |
| Virtual page → physical frame | Logical block → physical block |
| Page table | **Block table** (also stores *# filled slots* per block) |
| Demand paging | Allocate a new physical block only when the last one is full |
| Frames anywhere in RAM | Blocks anywhere in a preallocated GPU pool |
| Swap to disk | Swap blocks to **CPU** RAM |
| `fork()` + copy-on-write | Sequence fork + block-level copy-on-write |

- A **block engine** preallocates one big GPU region and divides it into physical blocks; the same on CPU for swapping.
- **Waste is bounded:** at most one partly-filled block per sequence (internal fragmentation < B tokens), no external fragmentation (all blocks equal size), no reservation.

## 5. Decoding walkthrough (§4.3) — use this example on a slide

Prompt "Four score and seven years ago our" = 7 tokens, block size 4.

1. Prefill: logical block 0 ← tokens 1–4 → physical block 7; logical block 1 ← tokens 5–7 → physical block 1 (3 of 4 slots filled).
2. Decode step 1: new token "fathers" goes into the free slot of physical block 1 (now 4/4). No allocation.
3. Decode step 2: block 1 is full → allocate a new physical block (say 3), map logical block 2 → 3, write "brought".

Two requests interleave in memory freely; their block tables keep them separate.

## 6. Other decoding scenarios (§4.4)

All built from three primitives (§5): **`fork`** (new sequence sharing the parent's blocks), **`append`** (add a token), **`free`** (release).

- **Parallel sampling** (n outputs per prompt): all samples share the prompt's blocks; each block has a **reference count**. When a sample writes into a shared block (ref count > 1), **copy-on-write** at block granularity: allocate new block, copy, decrement old count.
- **Beam search:** candidates share blocks with each other, and sharing changes every step as beams are pruned/forked. Blocks whose ref count drops to 0 are freed. Replaces the expensive full-cache copies in earlier systems.
- **Shared prefix:** a common system prompt / few-shot examples are computed once; their physical blocks are reserved and mapped into each request's block table (like a shared library mapped into many processes, CSAPP 7.10).
- **Mixed decoding:** because everything goes through block tables, requests with different decoding methods can be batched together; the kernel just sees block tables.

## 7. Scheduling and preemption (§4.5)

- Policy: **first-come-first-serve**; when memory runs out, **preempt the latest-arrived** requests first.
- **All-or-nothing eviction:** evict *all* blocks of a sequence, because all of them are read on every step anyway.
- **Gang scheduling:** sequences of one group (e.g. beams of one request) are preempted/rescheduled together, because they may share blocks.
- Two ways to bring a preempted request back:
  1. **Swapping** — copy its blocks to CPU RAM; while anything is swapped out, no new requests are admitted. CPU swap space is bounded by the GPU KV size.
  2. **Recomputation** — drop the blocks; later, treat prompt + generated tokens as one new prompt and rebuild the KV cache in a single prefill pass (cheap because prefill is parallel).
- Ablation (§7.3): recompute wins at small block sizes (swapping does many tiny PCIe copies); swap wins at large; similar for 16–64.

## 8. Distributed execution (§4.6) — not implemented, but explain it

- Megatron-LM tensor parallelism: each GPU holds a slice of attention heads.
- **One centralized KV cache manager** in the scheduler; all workers share the same block tables (each worker stores only its heads' KV for each block).
- Each step the scheduler broadcasts token IDs + block tables; workers all-reduce internally without the scheduler.

## 9. Implementation (§5)

- ~8.5K lines Python + 2K lines C++/CUDA. Scheduler and block manager in Python; kernels in CUDA.
- FastAPI frontend extending the OpenAI API.
- Kernel optimisations:
  1. **Fused reshape and block write** — new K/V written straight into their blocks in one kernel.
  2. **Fused block read and attention** — one GPU warp per block, handles variable sequence lengths.
  3. **Fused block copy** — batch all copy-on-write copies into one kernel launch.

## 10. Evaluation (§6) — headline numbers

Setup: A100 GPUs; OPT-13B (1 GPU), OPT-66B (4), OPT-175B (8), LLaMA-13B. Workloads: **ShareGPT** (long, varied) and **Alpaca** (short). Poisson arrivals. Metric: **normalized latency** = mean(end-to-end latency / output length) vs request rate.

Baselines: **FasterTransformer** (static max-batch scheduler); **Orca** reimplemented with a buddy allocator in 3 variants — **Oracle** (knows true output length), **Pow2** (over-reserves up to 2×), **Max** (reserves full 2048).

| Experiment | Result |
| --- | --- |
| Basic sampling, ShareGPT (Fig. 12) | 1.7–2.7× request rate vs Orca (Oracle), 2.7–8× vs Orca (Max), up to 22× vs FasterTransformer |
| Batched requests, OPT-13B ShareGPT (Fig. 13) | 2.2× more concurrent requests than Orca (Oracle), 4.3× vs Orca (Max) |
| Alpaca, OPT-175B | Smaller gain — memory is plentiful and sequences short, so compute-bound |
| Parallel sampling / beam search (Fig. 14–15) | Gain over Oracle grows from 1.3× to 2.3× at beam width 6; memory saved 6.1–9.8% (parallel) and 37.6–55.2% (beam) on Alpaca |
| Shared prefix, LLaMA-13B, WMT16 (Fig. 16) | 1.67× (1-shot, 80-token prefix), 3.58× (5-shot, 341-token prefix) |
| Chatbot (Fig. 17) | 2× request rate vs all three Orca variants |

## 11. Ablations (§7)

- **Kernel (Fig. 18a):** PagedAttention kernel is 20–26% slower than FasterTransformer's — the cost of indirection. Only affects attention, so end-to-end still wins.
- **Block size (Fig. 18b):** too small → poor GPU parallelism; too large → internal fragmentation and less sharing. 16–128 good on ShareGPT, 16–32 on Alpaca. Default 16.
- **Recompute vs swap (Fig. 19):** see §7 above.

## 12. Limitations / discussion (§8 and our own)

- Paging helps when memory is **dynamically allocated and the workload is memory-bound** — not every GPU workload (e.g. DNN training with static shapes).
- Indirection makes the attention kernel slower (20–26%).
- Recomputation/swap heuristics are simple (FCFS, latest-first eviction); no fairness or priority.
- Evaluated on 2023 models with full multi-head attention. Newer models (grouped-query attention, sliding windows) have much smaller KV caches — does the gain shrink? (→ our extension candidate.)
- No prefix caching **across** requests (only predefined shared prefixes) — later added by SGLang (RadixAttention) and vLLM itself.

---

## Q&A bank (practise answering out loud)

1. **Why not just use `malloc`/`cudaMalloc` per token?** Per-token allocation is too fine-grained: huge tables, scattered reads that kill memory bandwidth. Blocks amortise lookups over 16 tokens and let a warp read contiguous memory.
2. **Why is there no external fragmentation?** All blocks are the same size, so any free block fits any request.
3. **What is the worst-case waste per sequence?** B − 1 token slots in the last block.
4. **Why default block size 16?** Large enough to use GPU parallelism efficiently, small enough to keep internal fragmentation low; best on both workloads in Fig. 18b.
5. **Why does recomputation work at all?** Prefill is parallel, so rebuilding N tokens' KV costs about one forward pass, not N.
6. **When is swapping better?** Large blocks (fewer, bigger PCIe transfers) and long sequences.
7. **Why all-or-nothing eviction?** Every decode step reads every block of a sequence; keeping half is useless.
8. **Why stop admitting new requests while something is swapped out?** To guarantee preempted requests can come back (no starvation) and bound CPU swap space.
9. **How does copy-on-write work here?** Ref count > 1 on the block being written → allocate a new block, copy the old one, write into the copy, decrement the old count.
10. **Is the output identical to normal attention?** Yes — same math, only the memory layout differs.
11. **Why is the kernel slower but the system faster?** Throughput is limited by batch size (memory), not attention speed.
12. **Why does OPT-175B on Alpaca gain less?** Plenty of KV memory (264 GB) and short sequences → compute-bound, not memory-bound.
13. **How is this different from OS paging?** No TLB/hardware; the GPU kernel does the translation in software. Eviction is per sequence (all-or-nothing), and recomputation is an option an OS doesn't have.
14. **What would change with grouped-query attention models?** KV per token shrinks (e.g. 8 KV heads instead of 32), so more requests fit even without paging — paging's relative benefit may shrink. (Our extension tests this.)
15. **How did you validate correctness?** Paged engine output compared token-by-token with HuggingFace greedy decoding (see DESIGN.md §Testing).
16. **What did you not implement and why?** §4.6 multi-GPU and the 66B/175B runs — one L40S. The OpenAI-API frontend — replaced by trace replay; it does not affect results.
