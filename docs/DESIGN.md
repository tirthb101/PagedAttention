# Design — mini-vLLM

How our implementation is structured, what each module does, and which paper section it implements.
This is the spec we code against; update it when the code diverges.

---

## 1. Architecture

```
            trace.py (requests + arrival times)
                         │
                         ▼
┌────────────────────── LLMEngine (engine.py) ──────────────────────┐
│                                                                   │
│  add_request() ──► Scheduler (scheduler.py)                       │
│                     │  waiting / running / swapped queues         │
│                     │  asks BlockSpaceManager "can I fit this?"   │
│                     ▼                                             │
│               BlockSpaceManager (block_manager.py)                │
│                 block tables, ref counts, COW, swap maps          │
│                 GPU BlockAllocator + CPU BlockAllocator           │
│                     │                                             │
│  step(): SchedulerOutputs ─► CacheEngine (cache_engine.py)        │
│           (swap in/out, copy-on-write copies)                     │
│                     │                                             │
│                     ▼                                             │
│           Model forward (model/opt.py)                            │
│             per layer: QKV → write_kv(slot_mapping)               │
│                         → paged_attention(block_tables)           │
│                     │                                             │
│                     ▼                                             │
│           Sampler (sampler.py) → new tokens → update sequences    │
└───────────────────────────────────────────────────────────────────┘
```

**Key separation (same as the paper):** the *scheduler + block manager* only do bookkeeping with integers (block numbers). They never touch tensors. The *cache engine + model* own the tensors and just follow the integer instructions. This lets us test all memory-management logic on a CPU with no model at all, and lets the GPU-free simulator (`benchmarks/simulate_memory.py`) reuse the real block manager.

## 2. Paper → code map

| Paper | Module | Main functions |
| --- | --- | --- |
| §4.1 PagedAttention | `attention/paged_ref.py`, `attention/paged_triton.py` | `write_kv()`, `paged_attention()` |
| §4.2 KV cache manager | `block.py`, `block_manager.py` | `BlockAllocator`, `BlockSpaceManager.allocate/append_slot/free` |
| §4.3 On-demand allocation | `block_manager.py` | `append_slot()` |
| §4.4 Sharing, COW | `block_manager.py`, `sequence.py` | `fork()`, ref counts, `append_slot()` returns COW pairs |
| §4.4 Beam search | `sampler.py`, `engine.py` | beam step → fork/free |
| §4.4 Shared prefix | `block_manager.py` | `register_prefix()`, `allocate(prefix=…)` |
| §4.5 Scheduling | `scheduler.py` | `schedule()`, `_preempt()` |
| §4.5 Swapping | `block_manager.py`, `cache_engine.py` | `swap_out/swap_in()`, `CacheEngine.swap_*()` |
| §4.5 Recomputation | `scheduler.py` | `_preempt_by_recompute()` |
| §5 Kernel fusions | `attention/paged_triton.py`, `cache_engine.py` | fused write, fused read+attention, batched copy |
| §6 Orca baselines | `reservation_manager.py` | same interface as `BlockSpaceManager` |

## 3. Data structures

### 3.1 Config (`config.py`)

```python
@dataclass
class EngineConfig:
    model: str = "facebook/opt-125m"
    dtype: str = "float16"
    block_size: int = 16                 # tokens per KV block (paper default)
    kv_cache_gb: float = 12.0            # GPU KV pool size; 12 GB matches paper's OPT-13B/A100-40GB setup
    swap_space_gb: float = 12.0          # CPU pool; paper bounds it by GPU KV size
    max_num_seqs: int = 256              # max sequences in a batch
    max_num_batched_tokens: int = 2048   # token budget per step (bounds prefill size)
    max_model_len: int = 2048            # OPT context window
    preemption_mode: str = "recompute"   # "recompute" | "swap"
    memory_manager: str = "paged"        # "paged" | "orca-oracle" | "orca-pow2" | "orca-max"
    attention_backend: str = "ref"       # "ref" (PyTorch) | "triton"
```

Number of GPU blocks = `kv_cache_gb * 2**30 // (block_size * kv_bytes_per_token)`, where
`kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes`.
OPT-13B: 2 × 40 × 40 × 128 × 2 = 819,200 B (≈800 KB, matches paper §3).

**Why cap the KV pool at 12 GB on a 48 GB L40S?** With OPT-13B the L40S would leave ~18 GB for KV, more than the paper's A100-40GB. Capping to 12 GB reproduces the paper's memory pressure; we also run uncapped and report both.

### 3.2 Sequences (`sequence.py`)

```python
class SequenceStatus(Enum):
    WAITING, RUNNING, SWAPPED, FINISHED_STOPPED, FINISHED_LENGTH, FINISHED_ABORTED

@dataclass
class SamplingParams:
    n: int = 1                    # parallel samples (§4.4)
    use_beam_search: bool = False # beam width = n when True
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 128
    ignore_eos: bool = False      # benchmarks force exact output lengths from the trace

class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    status: SequenceStatus
    cumulative_logprob: float     # for beam search
    def get_len(self) -> int                  # prompt + output
    def num_logical_blocks(self, B) -> int    # ceil(len / B)
    def fork(self, new_id) -> "Sequence"      # copies token lists, NOT blocks

class SequenceGroup:                # one user request; all seqs share the prompt
    request_id: str
    seqs: list[Sequence]
    sampling_params: SamplingParams
    arrival_time: float
    first_scheduled_time, finish_time: float | None   # for latency metrics
    true_output_len: int | None     # from the trace; used only by Orca-Oracle
```

### 3.3 Physical blocks and allocator (`block.py`)

```python
class Device(Enum): GPU, CPU

@dataclass
class PhysicalBlock:
    block_number: int     # index into the KV tensor pool
    device: Device
    ref_count: int = 0

class BlockAllocator:
    def __init__(self, device, num_blocks): free list = all blocks
    def allocate(self) -> PhysicalBlock      # pop free list, ref_count = 1; raise if empty
    def free(self, block)                    # ref_count -= 1; back to free list at 0
    def get_num_free_blocks(self) -> int
```

A free list (LIFO stack) is enough: all blocks are the same size, so there is no placement decision and no external fragmentation — compare with the explicit free lists of CSAPP 9.9.13.

### 3.4 Block tables (`block_manager.py`)

```python
BlockTable = list[PhysicalBlock]      # index = logical block number

class BlockSpaceManager:
    block_tables: dict[int, BlockTable]   # seq_id -> table
    gpu_allocator, cpu_allocator: BlockAllocator
    watermark_blocks: int                 # keep ~1% free to avoid thrashing

    # admission
    def can_allocate(self, group) -> AllocStatus   # OK | LATER | NEVER (prompt > pool)
    def allocate(self, group)                       # prompt blocks; shared by all seqs in group (ref_count = n)

    # decoding (§4.3, §4.4)
    def can_append_slots(self, group) -> bool       # ≤ 1 new block per running seq
    def append_slot(self, seq) -> tuple[int, int] | None
        # if last block is full → allocate new block
        # elif last block ref_count > 1 → copy-on-write: allocate new, return (src, dst) to copy
        # else write in place → None
    def fork(self, parent, child)                   # child table = parent table; ref_count++ each block
    def free(self, seq)                             # ref_count-- each block

    # swapping (§4.5)
    def can_swap_in(self, group) -> bool
    def swap_out(self, group) -> dict[int, int]     # gpu_block -> cpu_block (shared blocks mapped once)
    def swap_in(self, group) -> dict[int, int]      # cpu_block -> gpu_block

    # shared prefix (§4.4)
    def register_prefix(self, prefix_id, token_ids) # allocate + pin blocks once
    # allocate(group, prefix_id=...) maps the pinned full blocks into the new table (ref_count++)

    # metrics for the experiments
    def num_used_slots(self) -> int                 # tokens actually stored
    def num_allocated_slots(self) -> int            # blocks in use × block_size
```

**Slot mapping.** A token at position `i` of sequence `s` lives at
`slot = table[i // B].block_number * B + (i % B)`. The model gets one `slot_mapping` entry per new token and writes K/V there.

### 3.5 Orca baselines (`reservation_manager.py`)

Same interface, different policy: on `allocate`, reserve **all** blocks the request may ever need, up front:

| Variant | Reserved tokens |
| --- | --- |
| `orca-oracle` | prompt_len + true_output_len (from trace) |
| `orca-pow2` | next power of two ≥ prompt_len + true_output_len (≤ 2× over-reservation) |
| `orca-max` | max_model_len (2048) |

`append_slot` never allocates (space is already reserved). No sharing: `fork` copies the whole reservation. The paper models Orca's address placement with a **buddy allocator**; we round reservations to power-of-two block counts and place them with a simple buddy allocator in the simulator so external fragmentation is counted too. Storage in the real engine can still use our paged tensors — what matters for throughput is how much memory is *reserved*, which is what limits batch size.

## 4. Scheduler (`scheduler.py`, §4.5)

Three queues: `waiting` (FIFO by arrival), `running`, `swapped`.

```
schedule():
  # 1. Make room for running sequences to grow by one token each.
  running.sort(by arrival_time)                   # FCFS priority
  for group in running (oldest first):
      while not block_manager.can_append_slots(group):
          victim = running.pop_latest()           # latest-arrived first
          preempt(victim)                         # recompute or swap (all-or-nothing, whole group)
          if victim is group: break
      else:
          for seq in group: record append_slot(seq) → collect COW (src,dst) pairs

  # 2. Bring back swapped groups (only if nothing was preempted this step).
  if no preemption:
      for group in swapped (oldest first):
          if can_swap_in(group) and budget allows: swap_in → running
          elif nothing is running: abort group   # it outgrew the whole GPU pool; avoids deadlock

  # 3. Admit new requests (only if nothing is swapped out — paper rule —
  #    and not in a step that preempted, to avoid re-admitting a victim immediately).
  if not swapped and no preemption:
      for group in waiting (oldest first):
          if token budget and seq budget allow and can_allocate(group) == OK:
              allocate(group) → running (prefill this step)
          elif NEVER: abort group
          else: break                              # strict FCFS: no skipping

  return SchedulerOutputs(prefill_groups, decode_groups,
                          blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

preempt(group):
  if mode == "recompute" and group has 1 sequence:
      free all blocks; status = WAITING; output tokens kept;
      put at FRONT of waiting (its prompt is now prompt + outputs)
  else:                                           # swap (always for multi-seq groups, they share blocks)
      swap_out → status = SWAPPED; append to swapped
```

Multi-sequence groups (parallel sampling, beams) are always swapped, never recomputed: recomputing them would lose their block sharing. The open-source vLLM code makes the same choice.

## 5. Model execution

### 5.1 Batch layout

Each step flattens all scheduled tokens into one 1-D batch:

- prefill sequences contribute all their (prompt + any recomputed) tokens,
- decode sequences contribute exactly one token.

`InputMetadata` carries, per step:

```python
positions: Tensor[num_tokens]
slot_mapping: Tensor[num_tokens]          # where to write each token's K/V
prompt_lens: list[int]                    # per prefill sequence
block_tables: Tensor[num_seqs, max_blocks]# padded with -1
context_lens: Tensor[num_seqs]            # tokens to attend to (incl. current)
```

### 5.2 Attention (`attention/`)

1. `write_kv(k, v, key_cache, value_cache, slot_mapping)` — scatter new K/V into blocks (§5 fused reshape-and-write).
2. `paged_attention(q, key_cache, value_cache, block_tables, context_lens)`:
   - **Reference (PyTorch):** for each sequence, gather its blocks `key_cache[block_table[:n_blocks]]`, reshape to `[ctx_len, heads, dim]`, run `scaled_dot_product_attention` (causal mask for prefill). Slow but obviously correct — this is our oracle.
   - **Triton (decode only):** one program per (sequence, head); loop over the block table, load one block of K, compute scores, online softmax, accumulate V. Prefill can keep using the reference/SDPA path — the paper's kernel work is about decode too.

KV cache tensor layout per layer: `key_cache[num_blocks, block_size, num_heads, head_dim]` (same for values). Simple to index; the Triton kernel may later use vLLM's transposed layout for coalescing (an ablation we can report).

### 5.3 Model (`model/opt.py`)

Our own OPT decoder (embeddings + learned positions, N decoder layers, final layer norm, LM head) whose attention calls the functions above. Weights loaded from the HuggingFace checkpoint (`safetensors`). We write our own instead of patching `transformers` so that the KV cache is entirely ours.

### 5.4 Sampler (`sampler.py`)

Greedy, temperature, top-p, and beam search. Beam search per group each step: score all `beam × vocab` continuations, keep top `beam` → for each kept candidate either reuse its parent sequence or `fork` it; free dropped parents. This is where ref counts and COW get exercised hardest.

## 6. Engine loop (`engine.py`)

```python
engine = LLMEngine(config)
engine.add_request(request_id, prompt_token_ids, sampling_params, arrival_time)
while engine.has_unfinished():
    outputs = engine.step()   # schedule → cache ops → forward → sample → update → free finished
```

`step()` also records per-request timestamps so benchmarks can compute normalized latency.

## 7. Testing strategy

| Test | What it proves |
| --- | --- |
| `test_block.py` | allocator never double-allocates; free list returns all blocks |
| `test_block_manager.py` | block tables grow on demand; fork shares; COW triggers exactly when ref_count > 1; swap round-trips; no block leaks after all seqs freed |
| `test_scheduler.py` | FCFS order; latest-first preemption; no admission while swapped; recompute re-queues at front |
| `test_paged_attention.py` | paged reference == plain attention on random tensors with shuffled block placement; Triton == reference (GPU only) |
| `test_e2e_equivalence.py` | greedy output of our engine == HuggingFace `generate` on `opt-125m`, for 8 prompts of different lengths, with block_size ∈ {1, 4, 16} and forced preemption |
| `test_orca.py` | reservation sizes for Oracle / Pow2 / Max |

**Invariant check (debug mode):** after every step, sum of ref counts == number of block-table entries across all live sequences, and free + used == total. Catches most bookkeeping bugs immediately.

## 8. Deviations from the paper (state these in the talk)

| Paper | Ours | Why |
| --- | --- | --- |
| (not specified) | A swapped request that can never fit back on the GPU (forked sequences outgrew the pool) is aborted | Found by our randomized scheduler test: otherwise the queue deadlocks |
| (not specified) | No new admissions in a step that preempted | Prevents immediately re-admitting the victim (thrashing) |
| Own model implementation | `PagedOPT` reuses HuggingFace's OPT layers (weights) and rewrites only the forward pass | Weights load exactly as in HuggingFace; the KV cache is still entirely ours |
| Separate prompt and generation iterations in vLLM's code | One step mixes prefill and decode tokens | Simpler batching; same as the paper's §4.3 description |
| CUDA kernels | Triton (+ PyTorch reference) | Faster to write and debug; same algorithm |
| A100 40/80 GB, up to 8 GPUs | 1× L40S 48 GB | Available hardware |
| OPT-66B/175B, tensor parallel | Not run | Do not fit on one GPU |
| FastAPI server | Offline trace replay | Does not affect measured throughput |
| Orca as a separate system | Reservation policies inside our engine | Same as the authors' own reimplementation |
| 1-hour traces | 10-min traces for development and ablations; 1-hour traces for the final headline runs | GPU budget; the cluster allows 5-day jobs, so the headline matches the paper |
