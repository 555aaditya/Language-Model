"""Autoregressive generation over the KV cache (TDD §6).

Two entry points:

- ``generate_ids`` — tensor in, token ids out. The real engine; testable
  without a tokenizer.
- ``generate`` — the string-level contract from BUILD_ORDER, a thin wrapper.

``use_cache=False`` exists purely as a correctness oracle: it re-runs the whole
prefix through the model on every step, which is the O(T²) behaviour the cache
removes. The two paths must produce identical greedy output, which is what
``test_cached_generation_matches_uncached`` checks — the cheapest way to catch a
cache that has quietly stopped being equivalent to a full forward.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch

from inference.sampling import sample
from model import CausalLM


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


@torch.no_grad()
def generate_ids(
    model: CausalLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    stop_id: int | None = None,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Generate a continuation for a single ``[1, T]`` prompt. Returns new ids only."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"expected a single [1, T] prompt, got {tuple(input_ids.shape)}")
    if input_ids.shape[1] == 0:
        raise ValueError("cannot generate from an empty prompt")

    was_training = model.training
    model.eval()
    try:
        max_seq_len = model.max_seq_len
        device = next(model.parameters()).device
        context = input_ids.to(device)

        cache = model.new_cache() if use_cache else None
        logits = model(context, kv_cache=cache, use_cache=use_cache)

        produced: list[int] = []
        for _ in range(max_new_tokens):
            # Only the final position predicts the next token; the rest of the
            # prefill logits are thrown away.
            next_id = sample(
                logits[0, -1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )
            produced.append(next_id)
            if stop_id is not None and next_id == stop_id:
                break

            step = torch.tensor([[next_id]], dtype=torch.long, device=device)
            if use_cache:
                if len(cache[0]) + 1 > max_seq_len:  # type: ignore[index]
                    break
                logits = model(step, kv_cache=cache, use_cache=True)
            else:
                context = torch.cat([context, step], dim=1)
                if context.shape[1] > max_seq_len:
                    break
                logits = model(context)

        return produced
    finally:
        model.train(was_training)


def generate(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    return_prompt: bool = True,
    **sampling: Any,
) -> str:
    """String-level generation (docs/BUILD_ORDER.md contract)."""
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt encoded to zero tokens")

    device = next(model.parameters()).device
    prompt_ids = torch.tensor([ids], dtype=torch.long, device=device)
    new_ids = generate_ids(model, prompt_ids, max_new_tokens=max_new_tokens, **sampling)

    return tokenizer.decode((ids + new_ids) if return_prompt else new_ids)
