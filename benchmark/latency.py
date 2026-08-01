"""Inference latency and throughput (TDD §6.4, §9).

Generation has two distinct phases and averaging them together hides the thing
you actually care about:

- **Prefill / first token** — the whole prompt in one parallel pass. Compute-bound,
  scales with prompt length.
- **Decode / per token** — one token at a time against the cache. Memory-bandwidth
  bound, roughly flat per step.

A single "tokens/sec" number mixes the two and moves around with prompt length
for reasons that have nothing to do with decode speed, so both are reported.
"""

from __future__ import annotations

from typing import Any

import torch

from benchmark.harness import synchronize, time_it
from inference import generate_ids
from model import CausalLM


def benchmark_prefill(
    model: CausalLM,
    *,
    prompt_len: int = 64,
    batch_size: int = 1,
    warmup: int = 2,
    repeats: int = 10,
) -> dict[str, Any]:
    """Time to process a prompt and produce the first next-token logits."""
    device = next(model.parameters()).device
    prompt = torch.randint(0, model.vocab_size, (batch_size, prompt_len), device=device)

    def run() -> None:
        cache = model.new_cache()
        with torch.no_grad():
            model(prompt, kv_cache=cache, use_cache=True)

    timing = time_it(run, warmup=warmup, repeats=repeats, device=device)
    tokens = batch_size * prompt_len
    return {
        "phase": "prefill",
        "prompt_len": prompt_len,
        "batch_size": batch_size,
        "timing": timing.as_dict(),
        "tokens_per_sec": round(tokens / (timing.median_ms / 1000.0), 2),
    }


def benchmark_decode(
    model: CausalLM,
    *,
    prompt_len: int = 16,
    max_new_tokens: int = 32,
    use_cache: bool = True,
    warmup: int = 1,
    repeats: int = 5,
) -> dict[str, Any]:
    """Full generation, reported as per-token cost."""
    device = next(model.parameters()).device
    prompt = torch.randint(0, model.vocab_size, (1, prompt_len), device=device)

    def run() -> None:
        generate_ids(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            use_cache=use_cache,
        )

    timing = time_it(run, warmup=warmup, repeats=repeats, device=device)
    per_token_ms = timing.median_ms / max_new_tokens
    return {
        "phase": "decode",
        "use_cache": use_cache,
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "timing": timing.as_dict(),
        "per_token_ms": round(per_token_ms, 4),
        "tokens_per_sec": round(1000.0 / per_token_ms, 2),
    }


def kv_cache_speedup(
    model: CausalLM, *, prompt_len: int = 16, max_new_tokens: int = 32, repeats: int = 3
) -> dict[str, Any]:
    """Cached vs uncached decoding — the measurement TDR-008 is claimed on.

    TDR-012 requires an optimization to demonstrate a measured gain rather than
    assert one, so the ratio is reported whichever way it falls. At small model
    sizes and short sequences the cache can genuinely lose: the O(T²) recompute
    it removes is cheap, while the per-step Python and allocation overhead it
    adds is not. That crossover is a real property worth surfacing, not a bug.
    """
    common = {"prompt_len": prompt_len, "max_new_tokens": max_new_tokens, "repeats": repeats}
    cached = benchmark_decode(model, use_cache=True, **common)
    uncached = benchmark_decode(model, use_cache=False, **common)
    ratio = uncached["timing"]["median_ms"] / cached["timing"]["median_ms"]
    return {
        "cached_ms": cached["timing"]["median_ms"],
        "uncached_ms": uncached["timing"]["median_ms"],
        "speedup": round(ratio, 3),
        "cache_wins": ratio > 1.0,
        "max_new_tokens": max_new_tokens,
    }


def benchmark_attention_impls(
    model_cfg: dict[str, Any], *, seq_len: int = 128, repeats: int = 5
) -> dict[str, Any]:
    """Compare the three attention kernels on one forward pass (TDD §4.3)."""
    from attention import VALID_IMPLS

    results = {}
    for impl in VALID_IMPLS:
        cfg = {**model_cfg, "attention": {"impl": impl}}
        torch.manual_seed(0)
        model = CausalLM.from_config(cfg).eval()
        device = next(model.parameters()).device
        ids = torch.randint(0, model.vocab_size, (1, seq_len), device=device)

        def run(m: CausalLM = model, x: torch.Tensor = ids) -> None:
            with torch.no_grad():
                m(x)

        synchronize(device)
        results[impl] = time_it(run, warmup=2, repeats=repeats, device=device).as_dict()

    return {"seq_len": seq_len, "impls": results}
