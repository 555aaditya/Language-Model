# Build Order & Module Interfaces

Dependency-aware implementation plan for the remaining modules. Each stage is
independently testable and unblocks the next. The architecture is specified in
[`TDD.md`](TDD.md); this document is the *sequencing* and the *contracts* that
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
| 0 | **tokenizer** ✅ | — | `BPE` with train/encode/decode/save/load | 14 unit tests |
| 1 | **dataset** ✅ | tokenizer | mmap + streaming datasets → dict batches | 29 unit tests |
| 2 | **attention** ✅ | config | causal GQA + RoPE + KV cache, 3 kernels | 39 unit tests |
| 3 | **model** ✅ | attention | `CausalLM` (embed → blocks → RMSNorm → LM head) | 25 unit tests |
| 4 | **training** ✅ | dataset + model | AdamW + cosine + AMP loop | 58 unit tests (incl. overfit-one-batch) |
| 5 | **inference** ✅ | model ckpt | greedy / top-k / top-p sampling | 29 unit tests |
| 6 | **optimization** | model ckpt | int8/int4 quant + optional kernels | perplexity-within-tolerance test |
| 7 | **benchmark** | all | latency / throughput / memory / perplexity harness | reproduces a recorded baseline |

> Stages 0–5 are implemented (204 tests). Keep 6 optional/feature-flagged so it never blocks 5.

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
    def new_cache(self) -> list[KVCache]: ...  # one entry per layer
    @classmethod
    def from_config(cls, cfg: dict) -> "CausalLM": ...

# forward returns logits only -- cross-entropy lives in the trainer, so the
# model stays pure for inference. Because of that there is nowhere to return a
# freshly built cache, so the cache is CALLER-OWNED: allocate once with
# new_cache() and pass the same list every step; entries mutate in place.
# use_cache=True without a cache raises rather than silently discarding it.


# inference
def sample(logits, *, temperature=1.0, top_k=0, top_p=1.0) -> int: ...
def generate(model, tokenizer, prompt: str, *, max_new_tokens: int, **sampling) -> str: ...


# optimization
def quantize(model, *, bits: int = 8, scheme: str = "int8_weight") -> nn.Module: ...
```

## Conventions

- **Config-driven**: every module reads from the loaded YAML dict (see `configs/default.yaml`); no magic constants.
- **tokenizer representation** is GPT-2 style rendered-unicode (TDR-002). Raw bytes only at the UTF-8 boundary via `token_bytes()`.
- **KV cache** is the canonical inference path; training never uses it.
- **TDD**: write the exit test in `tests/unit/` *before* implementing each stage.
- One module = one public API (exported in its `__init__.py`).
