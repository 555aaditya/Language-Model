"""Batched generation (TDD §6.3).

Single-stream decoding leaves the device almost idle: a decode step is one token
wide, so the matmuls are tall-and-thin and bound by weight bandwidth rather than
arithmetic. Stacking B requests reuses each weight read across B rows, which is
close to free — the win is throughput, not per-request latency.

Two things make batching more than a reshape:

- **Left padding, not right.** Prompts differ in length, and the next token is
  always predicted from the *last* position. Right-padding would put pad tokens
  there, so every request would be continuing from `<pad>`. Left-padding keeps
  every sequence's real final token at index -1.
- **Per-row completion.** Requests finish at different times. A finished row
  keeps being fed (the KV cache is one contiguous block; you cannot cheaply
  remove a row mid-flight) but its output is masked to the stop token, so
  nothing after the stop is reported.
"""

from __future__ import annotations

import torch

from inference.sampling import apply_temperature, apply_top_k, apply_top_p
from model import CausalLM


def sample_batch(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one token per row from ``[B, vocab]`` logits. Returns ``[B]`` ids.

    The single-row ``sample()`` returns an ``int`` by contract, which cannot
    express a batch, so this is a separate entry point rather than a widening of
    that signature.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected [batch, vocab] logits, got {tuple(logits.shape)}")

    logits = logits.detach().float()
    if temperature == 0:
        return torch.argmax(logits, dim=-1)

    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)

    probs = torch.softmax(logits, dim=-1)
    if generator is not None and generator.device != probs.device:
        probs = probs.to(generator.device)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def left_pad(sequences: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad ragged prompts to ``[B, T]``. Returns ``(ids, attention_mask)``.

    Padding goes on the **left** so every row's real final token sits at index
    -1, which is the position the next token is predicted from.
    """
    if not sequences:
        raise ValueError("no sequences to pad")
    width = max(len(s) for s in sequences)
    if width == 0:
        raise ValueError("cannot generate from empty prompts")

    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    for row, seq in enumerate(sequences):
        if seq:
            ids[row, width - len(seq) :] = torch.tensor(seq, dtype=torch.long)
            mask[row, width - len(seq) :] = True
    return ids, mask


@torch.no_grad()
def generate_batch(
    model: CausalLM,
    prompts: list[list[int]],
    *,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    stop_id: int | None = None,
    pad_id: int = 0,
    generator: torch.Generator | None = None,
) -> list[list[int]]:
    """Generate continuations for several prompts at once.

    Returns one list of new ids per prompt, truncated at ``stop_id`` where hit.

    Note the honest limitation: left-padded rows carry `pad_id` in their KV
    cache, and this model has no attention mask on the padded positions, so a
    short prompt in a wide batch attends to a few pad tokens. Greedy output for
    equal-length prompts is therefore identical to single-stream decoding, but
    ragged batches are approximate. Masking padded keys is the fix and is not
    implemented — recorded rather than hidden.
    """
    if not prompts:
        return []

    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        ids, _ = left_pad(prompts, pad_id)
        ids = ids.to(device)
        batch = ids.shape[0]

        cache = model.new_cache()
        logits = model(ids, kv_cache=cache, use_cache=True)

        produced: list[list[int]] = [[] for _ in range(batch)]
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            nxt = sample_batch(
                logits[:, -1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            ).to(device)

            for row in range(batch):
                if not finished[row]:
                    produced[row].append(int(nxt[row]))

            if stop_id is not None:
                finished |= nxt == stop_id
                if bool(finished.all()):
                    break

            if len(cache[0]) + 1 > model.max_seq_len:
                break
            logits = model(nxt.view(batch, 1), kv_cache=cache, use_cache=True)

        return produced
    finally:
        model.train(was_training)
