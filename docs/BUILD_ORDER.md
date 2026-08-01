# Build Order & Module Interfaces

Dependency-aware implementation plan for the remaining modules. Each stage is
independently testable and unblocks the next. The architecture is the
vault/TDD design; this document is the *sequencing* and the *contracts* that
keep the modules decoupled.

## Dependency graph

```
tokenizer ──► dataset ──► training ──► checkpoints ──► inference
                 ▲            ▲                              ▲
                 │            │                              │
              (token ids)  model ◄── attention          optimization
                            ▲                              ▲
                            └──────── benchmark ───────────┘
```

- `attention` and `dataset` only depend on `tokenizer`/config → build them early, in parallel.
- `model` composes `attention`.
- `training` needs `dataset` + `model`.
- `inference` needs a trained `model` checkpoint.
- `optimization` transforms a trained model → feeds back into `inference`.
- `benchmark` instruments everything; build a thin harness early, grow it last.

## Stages (in order)

| # | Module | Depends on | Deliverable | Exit test |
|---|--------|-----------|-------------|-----------|
| 0 | **tokenizer** ✅ | — | `BPE` with train/encode/decode/save/load | 14/14 unit tests |
| 1 | **dataset** | tokenizer | streaming dataset → `(input_ids, labels)` batches | shapes/dtype/shift-correctness test |
| 2 | **attention** | config | causal GQA + RoPE + KV cache | causal-mask & KV-cache equivalence test |
| 3 | **model** | attention | `CausalLM` (embed → blocks → RMSNorm → LM head) | forward shape + param-count test |
| 4 | **training** | dataset + model | AdamW + cosine + AMP loop in `training/train.py` | overfit-one-batch loss↓ test |
| 5 | **inference** | model ckpt | greedy / top-k / top-p sampling | deterministic greedy test |
| 6 | **optimization** | model ckpt | int8/int4 quant + optional kernels | perplexity-within-tolerance test |
| 7 | **benchmark** | all | latency / throughput / memory / perplexity harness | reproduces a recorded baseline |

> Build 1 and 2 in parallel. Keep 6 optional/feature-flagged so it never blocks 5.

## Contracts (stable interfaces — change only via ADR)

```python
# dataset
class TokenDataset(IterableDataset):
    def __iter__(
        self,
    ) -> Iterator[dict]: ...  # {"input_ids": LongTensor[T], "labels": LongTensor[T]}


# attention
class CausalAttention(nn.Module):
    def forward(self, x, *, kv_cache=None, use_cache=False): ...  # -> (out, new_kv_cache)


# model
class CausalLM(nn.Module):
    def forward(self, input_ids, *, kv_cache=None, use_cache=False): ...  # -> logits [B, T, V]
    @classmethod
    def from_config(cls, cfg: dict) -> "CausalLM": ...


# inference
def sample(logits, *, temperature=1.0, top_k=0, top_p=1.0) -> int: ...
def generate(model, tokenizer, prompt: str, *, max_new_tokens: int, **sampling) -> str: ...


# optimization
def quantize(model, *, bits: int = 8, scheme: str = "int8_weight") -> nn.Module: ...
```

## Conventions

- **Config-driven**: every module reads from the loaded YAML dict (see `configs/default.yaml`); no magic constants.
- **tokenizer representation** is GPT-2 style rendered-unicode (see ADR in vault). Raw bytes only at the UTF-8 boundary via `token_bytes()`.
- **KV cache** is the canonical inference path; training never uses it.
- **TDD**: write the exit test in `tests/unit/` *before* implementing each stage.
- One module = one public API (exported in its `__init__.py`).
