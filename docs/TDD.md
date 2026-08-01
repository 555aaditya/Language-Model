# Optimized GPT-style Language Model Engine Built From Scratch

**Technical Design Document (TDD)**
**Status:** Draft v1.1
**Audience:** Senior ML Engineers, Systems Engineers, Infrastructure Engineers

> This document specifies the architecture, interfaces, algorithms, and engineering
> strategy for a complete decoder-only transformer language model engine built from
> scratch. It prioritizes performance, scalability, memory efficiency, clean
> architecture, and explicit engineering tradeoffs. It is a design/reference
> document, not a tutorial.

---

## 0. Executive Summary

We design a **from-scratch**, decoder-only GPT-style transformer language model
system with a production-grade engineering posture. No pretrained weights. No
HuggingFace abstractions. All core components are implemented manually on PyTorch.

The system is decomposed into seven cooperating modules plus an overarching
benchmarking and testing strategy:

1. **Custom Tokenizer** — byte-level BPE.
2. **Efficient Dataset Engine** — memory-mapped, streaming, multiparallel feeder.
3. **Transformer Architecture** — decoder-only blocks (embedding, causal
   multi-head attention, MLP, output head).
4. **Attention Optimization** — standard → block/flash-inspired tiled attention.
5. **Training Engine** — AdamW, LR schedule, mixed precision, checkpointing.
6. **Inference Engine** — autoregressive generation + KV cache + batching + sampling.
7. **Model Optimization Layer** — quantization, memory pooling/allocation.

Cutting across all modules: a **benchmarking system** and a **TDD test strategy**.

The project follows a four-phase roadmap: Foundation → Functional Model →
Optimization → Production Engine. Every major decision is recorded as a Technical
Decision Record (TDR) in Section 13.

---

## 1. Custom Tokenizer

### 1.1 Scope

A subword tokenizer using **Byte Pair Encoding (BPE)** at the **byte level**
(GPT-2 style). It must convert arbitrary UTF-8 text to a compact sequence of
integer token ids and back losslessly.

### 1.2 Requirements

- Vocabulary creation / learning
- Token merging algorithm
- Encoding pipeline (text → ids)
- Decoding pipeline (ids → text)
- Special token handling
- Vocabulary serialization (save/load)

### 1.3 Architecture

```
Input text
   │
   ▼ encode utf-8 → byte sequence
Base byte vocabulary (256 tokens: byte 0..255)
   │  apply learned merges greedily
   ▼
BPE merges (ranked) → token ids
```

**Internal state:**
- `encoder: Dict[str, int]` — rendered-token → id.
- `decoder: Dict[int, str]` — id → rendered-token.
- `merges: List[tuple]` — rank-ordered merge rules.
- `special_tokens: List[str]` — atomic tokens, reserved ids 0..k-1.

**Rendering table:** A GPT-2-style `bytes_to_unicode` map renders the 256 byte
values to one-to-one printable unicode characters so that every byte sequence is a
valid string key (avoids collisions with special-token text).

### 1.4 Algorithm

**Training / merge learning (greedy BPE):**

1. Initialize each token sequence as its constituent UTF-8 bytes (id = `k + byte`,
   where `k` = number of special tokens).
2. Count frequencies of all adjacent pairs.
3. Merge the most frequent pair into a new token id, record its rank.
4. Replace all occurrences of that pair; repeat until the target `vocab_size` is
   reached or no pairs remain.

**Encoding (greedy):**

1. Detect and reserve special-token substrings (atomic; never merged across).
2. Map remaining bytes to base token ids.
3. Repeatedly merge the adjacent pair with the **lowest learned merge rank** until
   no merge applies.

**Decoding:**

1. For each id, emit either the special token's literal text or the original bytes
   recovered by inverting the rendering table.
2. Concatenate bytes and decode UTF-8 (loss-tolerant with `errors="replace"`).

### 1.5 Memory Considerations

- `encoder`/`decoder` dicts are the dominant memory cost; both grow with
  `vocab_size`. Each entry is `O(1)`; total is `O(vocab_size)`.
- The byte-decomposition cache (`token_id → bytes`) lets us render merged tokens
  without re-deriving byte sequences; its cost is `O(vocab_size × avg_bytes)`.
- Training counts pairs over the corpus each iteration; for very large corpora
  this must be streaming / sampled rather than fully resident.

### 1.6 Performance Optimizations

- **Single-pass tokenization:** base byte mapping via the precomputed rendering
  table (a `bytes.translate`-style lookup), not a per-byte Python loop.
- **Rank-driven greedy merge:** iterate merges in rank order over a slice/list to
  avoid an `O(n²)` scan per merge; use a heap of candidate pairs.
- Special-token detection front-loaded before byte merging so the hot merge loop
  never branches on specials.

### 1.7 Design Tradeoffs

| Approach | Pros | Cons |
|----------|------|------|
| Byte-level BPE | No OOV, handles all Unicode/emoji, robust subwords | Slightly more tokens for ASCII than word-level |
| Word-level BPE / SentencePiece-style | Cleaner for Latin scripts | OOV handling and Unicode complexity |

---

## 2. Efficient Dataset Engine

### 2.1 Scope

A high-performance data pipeline that feeds training batches to the model without
loading the entire corpus into RAM and without stalling the compute device.

### 2.2 Requirements

- Memory-mapped dataset loading
- Streaming dataset support
- Multiprocessing workers
- Asynchronous prefetching
- Pinned-memory transfer (GPU)
- Efficient batching
- Configurable shuffling

### 2.3 Data Flow Architecture

```
Dataset Storage (binary token files)
        │  open (os page cache)
        ▼
Memory Mapping (mmap) — virtual address space, no bulk RAM copy
        │  __getitem__(idx)
        ▼
Worker Processes (num_workers=N, fork/spawn)
        │  decode + batch construction
        ▼ (async)
Prefetch Queue (DataLoader `prefetch_factor`)
        │  .to(device, non_blocking=True) on pinned tensors
        ▼
GPU Training Batch (x, y)
```

### 2.4 Components

- **Memory-Mapped Reader:** token ids stored as a flat array of `uint16`
  (`numpy.memmap` or `torch.from_file`). Only pages actually read are brought into
  memory by the OS.
- **TokenDataset:** `__len__` = number of usable `block_size` windows; `__getitem__`
  returns a contiguous slice `(T,)`; a `collate` fn offsets by one to produce
  `(x, y)` pairs where `y = x shifted left by one`.
- **Multiprocessing workers:** each worker owns an independent memmap handle,
  avoiding GIL contention.
- **Prefetching:** `DataLoader(num_workers, prefetch_factor, pin_memory,
  non_blocking=True)` keeps the GPU fed.
- **Shuffling:** window-index shuffling (shuffle the starting offsets) rather than
  shuffling live tensors; for streaming, reshuffle epoch boundaries.

### 2.5 CPU ↔ GPU Interaction & Memory Management

- Pinned host buffers allow `H2D` copies without intermediate staging.
- Prefetching overlaps I/O, tokenization, and device transfer with compute.
- The memmap uses OS page cache, so frequently accessed regions benefit from
  kernel caching (re-warm across epochs cheaply).

### 2.6 Bottlenecks & Optimization Opportunities

| Bottleneck | Mitigation |
|------------|-----------|
| Disk latency | mmap (page cache), OS-level prefetch, SSDs |
| Python decode/encode | vectorized byte mapping, C-accelerated ops |
| GIL in workers | multiprocessing over threads |
| Device idle | prefetch + async pinned H2D |
| Memory spikes | mmap (no bulk load), fixed `block_size` |

---

## 3. Transformer Architecture

A standard decoder-only transformer. All tensors are `(B, T, C)` for batch,
sequence, and channel dims.

### 3.1 High-Level Stack

```
tok ids (B, T)
   │
Token Embedding (V → C)
Positional Embedding (T → C)
   │
N × TransformerBlock:
   ├─ LayerNorm → MultiHeadCausalAttention → Residual
   ├─ LayerNorm → MLP (C→4C→C, GELU, dropout) → Residual
   │
Final LayerNorm
   │
LM Head (C → V)  → logits (B, T, V)
```

### 3.2 Embedding Layer

- **Token embeddings:** `nn.Embedding(V, C)`.
- **Positional embeddings:** learnable absolute table `nn.Embedding(block_size, C)`,
  added per position. Chosen for Phase 1 simplicity; Rotary/ALiBi flagged as future
  improvements for longer-context extrapolation (see §13).

### 3.3 Self-Attention

- **Q/K/V projections:** a single fused `nn.Linear(C, 3C)` for efficiency, then
  split into Q, K, V.
- **Multi-head:** head size `d_head = C // n_head`; heads computed independently.
- **Causal masking:** upper-triangular mask (future tokens set to `-inf`) applied to
  the scaled `QKᵀ` scores.
- **Scaling:** `scores /= sqrt(d_head)` to keep softmax in a well-conditioned range.
- **Softmax + value aggregation:** `Attention(Q,K,V) = softmax(QKᵀ/sqrt(d)) V`.
- **Output projection:** `nn.Linear(C, C)`.

### 3.4 Feed-Forward Network (MLP)

- `Linear(C, 4C) → GELU → Linear(4C, C) → Dropout(p)`.
- Expansion factor `4×` is a standard compute/memory tradeoff.

### 3.5 Transformer Block

- **Pre-LayerNorm:** LayerNorm applied before each sublayer (Pre-LN), improving deep
  network trainability and reducing warmup sensitivity.
- **Residual connections** around both sublayers to carry gradients.

### 3.6 Output Layer

- `nn.Linear(C, V)` produces logits.
- **Training:** cross-entropy over vocab at each position.
- **Inference:** softmax over the last position → sampling distribution.

### 3.7 Config Dataclass (`ModelConfig`)

`vocab_size`, `block_size`, `n_layer`, `n_head`, `n_embd`, `dropout`, `bias`,
`weight_tying` (token embedding reused as head).

A concrete starter "nano" config (Phase 1/2 smoke testing) and a "base" config are
defined under `configs/`; sizes target ~1–15M parameters so early iterations fit
comfortably on CPU and a single consumer GPU.

---

## 4. Attention Optimization

### 4.1 Standard Attention

```
Attention(Q,K,V) = softmax( Q Kᵀ / sqrt(d) ) V
```

- **Compute complexity:** `O(T² · d)` per head, summed over heads.
- **Memory complexity:** `O(T² · n_head)` for the full score tensor.
- **Limitation:** at long contexts, the `T²` score matrix dominates memory and
  bandwidth, becoming the primary scalability bottleneck.

### 4.2 Optimized / Flash-Attention-Inspired Attention (Tiled)

Algorithm: process the sequence in fixed-size **blocks**; maintain online softmax
running max `m` and running sum `l`; rescale the accumulated output when a new max
is found; never materialize the full `T²` matrix.

- **Memory:** `O(T)` (only output + per-block buffers), regardless of sequence
  length; only the current `(block_b, block_k)` tile is materialized.
- **Performance:** improved cache locality (IO-aware), lower bandwidth pressure.
- **Numerical:** online softmax with running max keeps values stable.

**Implementation challenges:**
- Online rescaling adds complexity (must carry `m`, `l` accumulators).
- Backward pass requires recomputing scores or caching `P = softmax(QKᵀ)`
  per block.
- Block granule tuning (`BLOCK_M`, `BLOCK_N`) and boundary padding.

**Tradeoff matrix:**

| Scheme | Memory | Complexity | Notes |
|--------|--------|-----------|-------|
| Standard | O(T²) | naive | simple |
| Block/tiled (flash-inspired) | O(T) | tiled, online softmax | exact, memory-scalable |

---

## 5. Training Engine

### 5.1 Training Loop

```
for step:
    x, y = next(batch)                       # dataset + collate
    with autocast(dtype): logits, loss = model(x, y)
    loss_scaler.scale(loss).backward()
    grad_norm = clip_grad_norm_(max_norm)
    loss_scaler.step(optimizer); update; zero_grad()
```

### 5.2 Optimization

- **Custom AdamW:** decoupled weight decay, bias correction, `eps`.
- **LR schedule:** cosine decay with linear warmup.
- **Gradient clipping:** global norm clipping for stability.
- **Warmup:** prevents early instability from large initial steps.

### 5.3 Training Improvements

- **Mixed precision:** `torch.autocast` + `GradScaler` (fp16/bf16). CPU reduces to
  fp32 (no-op path preserved so GPU code is portable).
- **Gradient accumulation:** accumulate over `K` micro-batches to emulate a larger
  effective batch on limited memory.
- **Checkpointing:** periodic save of model, optimizer, scaler, step, rng state.
- **Experiment tracking:** metrics reported to console/CSV/optional tracker.

### 5.4 Stability, Performance, Scalability

- Pre-LN + warmup + gradient clipping together stabilize early training.
- Mixed precision + pinned prefetch both raise throughput.
- Gradient accumulation + batching let batch size scale independent of device memory.
- Target: profile each stage to confirm where time is spent (forward/backward,
  dataloader, optimizer, H2D transfer); set throughput targets relative to a
  measured CPU baseline before claiming GPU/AMP gains.

---

## 6. Inference Engine

### 6.1 Text Generation

Autoregressive loop: encode prompt → loop `max_new_tokens` → sample next token →
append → decode.

**Sampling strategies:**
- **Greedy:** `argmax`.
- **Temperature:** `logits / temperature` before softmax.
- **Top-k:** restrict to top-k logits.
- **Top-p (nucleus):** smallest set with cumulative prob ≥ `p`.

### 6.2 KV Cache

```
Without cache: every generated token recomputes attention over all prior tokens → O(T²) cumulative
With cache:     keys/values from prior tokens are reused → O(T) total
```

- **Prefill:** encode prompt, compute & store all K/V.
- **Decode:** per new token compute only `q` and new `k/v`, append to cache,
  attend over full cached `K`, `V`.

**Memory tradeoff:** cache grows `2 * n_layer * n_head * d_head * T` floats per
request (linear in context length); mitigates with cache clearing/paging (future).

**Latency improvement:** transforms decoding from compute-bound into
memory/cache-bandwidth-bound, a large constant-factor speedup.

### 6.3 Batching

- Stack multiple requests; each request maintains its own KV cache in the batch dim.
- Improves device utilization and aggregate throughput.

### 6.4 Measured Metrics

First-token latency, per-token latency, tokens/sec, throughput, memory footprint.

---

## 7. Model Optimization Layer

### 7.1 Quantization

Compare **FP32** (baseline), **FP16**, and **INT8**.

| Precision | Memory | Speed (mixed HW) | Accuracy |
|-----------|--------|-------------------|----------|
| FP32 | 4 bytes/param | baseline | reference |
| FP16/BF16 | 2 bytes/param | faster w/ mixed HW | near-lossless |
| INT8 | 1 byte/param | fastest | some loss, needs calibration |

> **Hardware caveat:** On CPU-only validation (our primary portable path), fp16/int8
> may **not** be faster than fp32. These entries assume mixed-precision-capable
> hardware (e.g., GPU with Tensor Cores). The benchmarking layer measures both to
> validate claims per-device rather than assume a global winner.

**Approach:** FP16 via `.half()`/autocast; INT8 via post-training quantization with
per-tensor/per-channel scales and calibration on a reference split.

### 7.2 Memory Optimization

- **Tensor reuse:** avoid temporaries (in-place where safe), fused projections.
- **Memory pooling:** preallocated key/value caches, buffer reuse across blocks.
- **Allocation strategy:** modeled on CUDA/PyTorch caching allocator ideas —
  arena-based pooling so repeated same-shape allocations reuse blocks instead of
  hitting `malloc`/`cudaMalloc`.

---

## 8. Low-Level System Design

### 8.1 High-Level Architecture Diagram

```
Tokenizer
    │ (token ids)
    ▼
Dataset Engine        ── training batches ──►  Training Engine
    │                                              │ (weights)
    ▼                                              ▼
Transformer Model ◄────────────────────── Optimization Layer
    │                                              │
    ▼                                              ▼
Inference Engine ◄────────────────────────────── (optimized weights)
    │
    ▼
API Layer  (programmatic; optional serving interface later)
```

### 8.2 Module Responsibilities

| Module | Responsibility | Inputs | Outputs | Depends on |
|--------|----------------|--------|---------|------------|
| tokenizer | text ↔ token ids | str / list[int] | list[int] / str | config |
| dataset | training batch production | binary token file path | (x, y) batch tensors | tokenizer |
| model | forward, logits, loss | (B,T) token ids | (B,T,V) logits / loss | attention, config |
| attention | causal attention compute | q, k, v | attended output | — |
| training | weight updates | batches, model | weights, metrics | model, optimizer |
| inference | generation | prompt, params | generated text | model, KV cache |
| optimization | reduce footprint/latency | model | quantized/pooled model | model |
| benchmark | measure performance | model, data | metric report | all |

### 8.3 Class Design

```
Tokenizer.BPE            Dataset.TokenDataset, Dataset.DataEngine
Model.MultiHeadAttention, Model.TransformerBlock, Model.GPT, Model.ModelConfig
Attention.standard_attention, Attention.flash_attention
Training.AdamW, Training.Scheduler, Training.Trainer, Training.Checkpoint
Inference.KVCache, Inference.Generator, Inference.Sampler
Optimization.quantize_fp16, Optimization.quantize_int8, Optimization.MemoryPool
Benchmark.TrainingBenchmark, Benchmark.InferenceBenchmark
```

### 8.4 Internal API Design

Core contracts:

```python
tokenizer.BPE.encode(text) -> list[int]
tokenizer.BPE.decode(ids)  -> str
dataset.DataEngine.next_batch(device) -> (x, y)
model.GPT.forward(x, y)    -> (logits, loss)
training.Trainer.step()    -> metrics
inference.Generator.generate(prompt, **sampling) -> str
optimization.quantize(model, precision) -> model
```

---

## 9. Benchmarking System

Structure under `benchmark/`:

```
benchmark/
├── training_benchmark.py   # tokens/sec, throughput, GPU/CPU util, memory
├── inference_benchmark.py  # first-token latency, gen latency, tokens/sec, mem
├── memory_benchmark.py     # footprint FP32 vs FP16 vs INT8, KV cache size
└── latency_test.py         # per-token + end-to-end latency distributions
```

Metrics (training): tokens/sec, throughput, GPU utilization, CPU utilization,
peak memory (via `torch.cuda.max_memory_allocated` or `tracemalloc`).

Metrics (inference): first-token latency, generation latency, tokens/sec,
aggregate throughput, memory footprint (model + KV cache).

---

## 10. Test Driven Development Strategy

### 10.1 Unit Tests

- **Tokenizer:** encoding correctness, decoding correctness, round-trip
  consistency, vocabulary consistency (encoder/decoder inverse), special tokens,
  serialization round-trip.
- **Attention:** tensor shape `(B, H, T, d_head)`, causal-mask correctness,
  gradient flow (finite-difference check or `loss.backward()` presence).
- **Dataset:** batch shapes, window slicing, memory loading, offset consistency.
- **AdamW/optimizer:** step reduces loss, clip bounds norm.

### 10.2 Integration Tests

- tokenizer + dataset (tokens flow correctly)
- dataset + model (batch shapes through forward)
- model + training loop (loss decreases on a tiny dataset)
- inference pipeline (generate reproduces a deterministic greedy prefix)

### 10.3 Performance Tests

- Validate throughput/memory/latency meet targets and that optimizations improve
  baselines (gated thresholds).

### 10.4 Regression Tests

- Optimization (flash attention, quantization, KV cache) does **not** degrade
  final loss/accuracy beyond a tolerance vs the reference implementation.
- Map to module first, then entire suite re-run.

**Workflow:** every module: write failing test → implement → refactor. See
`~/vault/roadmap/` for per-phase checklists.

---

## 11. Repository Structure

```
language-model-engine/
├── tokenizer/      # custom BPE tokenizer
├── dataset/        # memory-mapped, streaming data engine
├── model/          # transformer architecture (embedding, blocks, head)
├── attention/      # standard + flash-inspired attention
├── training/       # trainer, AdamW, schedulers, checkpointing
├── inference/      # generator, KV cache, sampling, batching
├── optimization/   # quantization, memory pooling
├── benchmark/      # training/inference/memory/latency benchmarks
├── tests/          # pytest suite (unit + integration)
├── configs/        # runnable JSON/configuration files
├── checkpoints/    # serialized model/optimizer state
├── docs/           # TDD, README, module docs
├── .github/        # GitHub Actions CI workflows (lint, type, test, smoke)
└── train.py        # command-line entry point
```

Purpose of each directory summarized in the tree above; `tests/` mirrors the
module tree for maintainability. `.github/workflows/` holds the CI definitions
(see §16.3).

---

## 12. Development Roadmap

### Phase 1 — Foundation
Tokenizer (train/encode/decode/serialize) → dataset engine (mmap, workers,
prefetch, batching) → simple transformer forward pass (embedding, causal attention,
MLP, head). *Exit:* all unit tests passing; forward shapes correct.

### Phase 2 — Functional Model
Training engine (loop, AdamW, scheduler, warmup, grad-clip, AMP, checkpointing) →
basic autoregressive generation. *Exit:* loss converges on a smoke dataset;
generation produces coherent continuations.

### Phase 3 — Optimization
KV cache → quantization (fp16/int8) → flash-inspired attention → memory
optimization → inference batching. *Exit:* quantitative speedup/memory reductions
measured vs Phase 2 baseline.

### Phase 4 — Production Engine
Benchmarking framework → API layer → documentation → full integration + regression
suite. *Exit:* reproducible benchmark report; documentation complete.

---

## 13. Technical Decision Records (TDRs)

### TDR-001: Decoder-Only Architecture
- **Decision:** Decoder-only (GPT-style) causal transformer.
- **Reason:** Native fit for autoregressive next-token generation; modern SOTA
  architecture class.
- **Alternatives:** encoder-only (BERT), encoder-decoder (T5).
- **Tradeoffs:** Simpler unified stack vs no bidirectional context; sequential
  decoding (mitigated by KV cache).

### TDR-002: Byte-Level BPE Tokenizer
- **Decision:** GPT-2-style byte-level BPE with a `bytes_to_unicode` rendering table.
- **Reason:** Zero OOV for arbitrary UTF-8; robust subwords; well-understood.
- **Alternatives:** word-level BPE, SentencePiece unigram, WordPiece.
- **Tradeoffs:** Slightly larger token counts on Latin text vs OOV/Unicode ease.

### TDR-003: PyTorch Manual Implementation
- **Decision:** Implement all layers/training manually on PyTorch; no HF/pretrained.
- **Reason:** Autograd and device management provided while we own all ML logic.
- **Alternatives:** NumPy (hard autograd), JAX.
- **Tradeoffs:** Development speed vs authenticity/control; explicitly mandated.

### TDR-004: Memory-Mapped Dataset Pipeline
- **Decision:** mmap + workers + prefetch for training data.
- **Reason:** Avoids RAM blowup on large corpora; overlaps I/O with compute.
- **Alternatives:** full in-memory load, TFRecord-style shards.
- **Tradeoffs:** OS page cache dependency vs low memory ceiling and streaming.

### TDR-005: Pre-LayerNorm Transformer
- **Decision:** Pre-LN (normalize-before-sublayer).
- **Reason:** More stable deep training, less warmup sensitivity.
- **Alternatives:** Post-LN, original transformer.
- **Tradeoffs:** Pre-LN may require weighting for very deep stacks; simpler anyway.

### TDR-006: Learnable Absolute Positional Embeddings
- **Decision:** Learnable absolute position table for Phase 1.
- **Reason:** Simplest correct baseline; acceptable within the fixed `block_size`.
- **Alternatives:** sinusoidal, Rotary, ALiBi.
- **Tradeoffs:** No extrapolation beyond `block_size`; Rotary/ALiBi flagged as
  future optimizations (`~/vault/roadmap/future_improvements.md`).

### TDR-007: Flash-Attention-Inspired Tiled Attention
- **Decision:** Block-tiled attention with online softmax.
- **Reason:** Breaks the O(T²) memory wall for long contexts; IO-aware.
- **Alternatives:** Standard attention, true FlashAttention kernel.
- **Tradeoffs:** More complex bookkeeping vs large, exact memory savings.

### TDR-008: KV Cache for Inference
- **Decision:** Per-layer K/V cache with prefill-then-decode.
- **Reason:** Eliminates O(T²) recomputation; large decode speedup.
- **Alternatives:** full recompute, custom kernels.
- **Tradeoffs:** Linear memory growth vs large latency/throughput win.

### TDR-009: Custom AdamW + Cosine Schedule with Warmup
- **Decision:** AdamW with decoupled weight decay; cosine decay + linear warmup;
  gradient clipping.
- **Reason:** Standard, robust, well-calibrated optimizer recipe.
- **Alternatives:** SGD + momentum, LAMB/LARS, fixed LR.
- **Tradeoffs:** Slightly more hyperparams vs strong convergence/stability defaults.

### TDR-010: LayerNorm → FP16/BF16 Mixed Precision with Loss Scaling
- **Decision:** autocast + `GradScaler` for fp16/bf16; CPU degrades to fp32.
- **Reason:** Memory/speed gains on mixed-precision HW with near-lossless fp16.
- **Alternatives:** pure FP32, full bf16.
- **Tradeoffs:** FP16 needs scaled loss (grad underflow) and master weights on some
  setups.

### TDR-011: Global Knowledge Vault
- **Decision:** Maintain a global project knowledge vault at `~/vault/` (outside
  the repo), as single source of truth for decisions, research, benchmarks.
- **Reason:** Long-lived engineering memory independent of a single codebase;
  reusable across projects.
- **Alternatives:** in-repo `vault/`, wiki, issue docs.
- **Tradeoffs:** Slight indirection (not in repo history) vs global, durable,
  cross-project knowledge.

### TDR-012: Benchmarking Gated by Baselines
- **Decision:** Every optimization must demonstrate measurable improvement vs a
  stated baseline before acceptance.
- **Reason:** Prevents regressions and validates engineering claims.
- **Alternatives:** ad-hoc manual measurement.
- **Tradeoffs:** Slight overhead vs accountable, quantifiable engineering.

### TDR-013: Locked `main` + PR-Gated Git Workflow
- **Decision:** `main` is locked; all work on feature branches, merged only via
  reviewed PRs. Issues tracked in GitHub Issues.
- **Reason:** Keeps `main` production-stable, enables review, auditable history.
- **Alternatives:** direct-to-main, full git-flow.
- **Tradeoffs:** Slight overhead (branches/PRs) vs strong safety and review gate.
  See §16 and `~/vault/decisions/architecture_decisions.md` (ADR-003).

### TDR-014: Free-Tier CI (Lint + Type + Tests) Gate
- **Decision:** GitHub Actions free-tier CI enforcing lint, type check, tests, and
  import smoke on every PR and push to `main`; CI green is a merge requirement.
- **Reason:** Automated, enforced quality gate at zero cost on a public repo.
- **Alternatives:** local-only checks, paid/self-hosted CI, pre-commit alone.
- **Tradeoffs:** Runner resource limits vs encrypted, reproducible enforcement.
  See §16.3 and `~/vault/decisions/architecture_decisions.md` (ADR-004).

---

## 14. Project Knowledge Vault

Per the project requirements, all long-lived decisions, architecture, research,
experiments, roadmap, and implementation notes are stored in a **global**, modular
knowledge vault at **`~/vault/`** (available across projects):

```
~/vault/
├── architecture/    system_design, model_architecture, component_design, data_flow
├── decisions/       architecture_decisions, optimization_decisions, tradeoffs
├── research/        transformer_notes, attention_optimization, inference_optimization, papers
├── implementation/  coding_guidelines, module_interfaces, development_notes
├── experiments/     benchmarks, training_runs, optimization_results
├── roadmap/         milestones, completed_features, future_improvements
└── project_context.md   (overview, repo path, vault location, status)
```

**Rules:** (1) review vault before significant tasks; (2) update vault after
significant tasks; (3) capture new ideas in research/design docs; (4) keep it
current, marking stale info deprecated; (5) base recommendations on prior
decisions, noting alignment/conflict.

---

## 15. Deliverables Summary

- Complete from-scratch, decoder-only GPT engine on PyTorch.
- Custom BPE tokenizer; mmap multiparallel dataset engine.
- Standard + flash-inspired tiled attention.
- Training engine (AdamW, scheduler, AMP, checkpointing).
- Inference engine (KV cache, sampling, batching).
- Optimization layer (fp16/int8 quantization, memory pooling).
- Benchmarking framework and TDD test suite.
- Global knowledge vault at `~/vault/`.

---

## 16. Engineering Process: Git Workflow, CI, and Issue Tracking

We operate this project as a **production-grade, real-world codebase**. The
following engineering process governs how work is delivered.

### 16.1 Branch Strategy & Locked `main`

- **`main` is locked.** It is never committed to directly.
- All work happens on **feature branches** (e.g., `feat/tokenizer-bpe`,
  `fix/dataset-mmap`, `opt/flash-attention`).
- A branch is merged to `main` **only via a Pull Request (PR)** and **only when the
  work is complete and reviewed**.
- **Rule of thumb:** temporary/in-progress work lives on branches; `main` always
  holds final, merged, stable state.

### 16.2 Issue Tracking (GitHub Issues)

- Every task/feature/bug/optimization is tracked as a **GitHub Issue**.
- Feature branches and PRs reference the issue they address; PRs close issues with
  `Closes #N`.
- Work items map back to the vault roadmap (`~/vault/roadmap/milestones.md`) for
  traceability between design, implementation, and delivery.

### 16.3 CI/CD (Free-Tier GitHub Actions)

Continuous Integration runs automatically on **every PR** and **every push to
`main`**, consisting of free-tier GitHub-hosted jobs:

| Job | Tool | Purpose |
|-----|------|---------|
| Lint | `ruff` (and/or `flake8`/`black`) | enforce style & catch basic problems |
| Type check | `mypy` or `pyright` | enforce type correctness |
| Tests | `pytest` (unit + integration) | verify behavior, catch regressions |
| Import/build smoke | `python -c "import <pkg>"` | ensure modules import cleanly |

- **PR merge requirement:** all CI jobs must pass before a PR can be merged.
- **Reproducibility:** pin Python version and core dependencies
  (`pyproject.toml` + lockfile) so CI and local runs match.

### 16.4 Definition of "Complete" (Merge Criterion)

A PR is eligible for merge to `main` only when **all** of these hold:

1. Feature/optimization implemented per the design in this TDD.
2. Unit + integration tests written/passing (TDD) and CI green (lint + type + test).
3. A regression/performance gate (TDR-012) passes where optimization applies.
4. Associated GitHub Issue updated/resolved.
5. Vault updated to reflect the change (decisions, milestones, results).

This keeps `main` production-safe and makes every state of the repository auditable
and revertible.

---

## 17. Updated Deliverables & Engineering Goals

In addition to the functional deliverables in §15, the project delivers:

- A locked-`main`, PR-gated **branch workflow** with **GitHub Issue tracking**.
- **Free-tier CI** (lint + type check + tests + import smoke) enforced on PRs.
- A **global knowledge vault** (`~/vault/`) kept current with decisions,
  milestones, and results, acting as the single source of truth.

---

*End of TDD (rev 1.1).*
