# Language-Model

A production-grade, decoder-only language model built **from scratch** in PyTorch —
custom byte-level BPE tokenizer, grouped-query causal attention with RoPE and RMSNorm,
a cosine-scheduled training loop, and a modular inference/serving stack.

The full engineering rationale lives in the Obsidian vault (`~/vault/`) and
[`docs/TDD.md`](docs/TDD.md).

## Architecture (7 modules)

```
raw text ──► tokenizer ──► dataset ──► training ──► checkpoint ──► inference
                  │            │           │                         │
                  └────────────┴──── model ┴── optimization ─────────┘
                                         │
                                     benchmark  (measures everything)
```

| Module | Responsibility |
|--------|----------------|
| `tokenizer/` | Byte-level BPE (GPT-2 style rendered-unicode keys) |
| `dataset/` | Data ingestion, cleaning, packing, DataLoaders |
| `model/` | Transformer blocks (RoPE, RMSNorm, GQA, SwiGLU) |
| `attention/` | Attention mechanisms + KV cache |
| `training/` | Optimizer, scheduler, AMP, distributed training loop |
| `inference/` | Sampling, decoding strategies, generation engine |
| `optimization/` | Quantization, pruning, kernel/graph optimization |
| `benchmark/` | Metrics harness (accuracy, latency, throughput, memory) |

## Quick start

```bash
pip install -e .[dev]
pytest                                  # run the test suite
python -m training.train --config configs/default.yaml
```

## Status

- ✅ Tokenizer (byte-level BPE) — implemented & tested
- ⬜ Dataset · Model · Attention · Training · Inference · Optimization · Benchmark — planned

See `docs/TDD.md` and the vault for the complete technical design.
