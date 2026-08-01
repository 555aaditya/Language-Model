"""Token sampling strategies (TDD §6.1).

Filters compose in the order temperature → top-k → top-p, which is the
convention every reference implementation uses. Order matters: temperature
rescales the logits and therefore changes the probability mass that top-p is
measuring, so applying nucleus filtering first would make ``top_p`` mean
something different at every temperature.
"""

from __future__ import annotations

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0 (use 0 for greedy), got {temperature}")
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep the ``top_k`` highest logits, mask the rest."""
    if top_k <= 0:
        return logits
    k = min(top_k, logits.shape[-1])
    threshold = torch.topk(logits, k).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering: keep the *smallest* set whose mass reaches ``top_p``.

    The token that carries the cumulative sum across the threshold is kept, not
    dropped — otherwise the retained mass is strictly below ``top_p``, and with
    a confident distribution (first token already above ``p``) the whole
    vocabulary would be masked and softmax would return NaN.
    """
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if top_p == 1.0:
        return logits

    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    remove = cumulative > top_p
    # Shift right so the crossing token survives, and never drop the argmax.
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    return logits.masked_fill(remove.scatter(-1, sorted_idx, remove), float("-inf"))


def sample(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> int:
    """Draw one token id from a ``[vocab]`` (or ``[1, vocab]``) logit vector.

    ``temperature=0`` means greedy: the limit of the softmax as temperature
    approaches zero is the argmax, but computing it would divide by zero, so it
    is special-cased rather than approximated with a small epsilon.
    """
    logits = logits.detach().float()
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 1:
        raise ValueError(f"expected a single [vocab] logit vector, got {tuple(logits.shape)}")

    if temperature == 0:
        return int(torch.argmax(logits).item())

    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)

    probs = torch.softmax(logits, dim=-1)

    if generator is not None and generator.device != probs.device:
        # torch.multinomial requires the generator and the tensor to share a
        # device, but a seeded CPU generator is the natural way to ask for
        # reproducible sampling — so a caller doing the obvious thing would hit
        # a RuntimeError as soon as the model moved to MPS or CUDA. Draw on the
        # generator's device instead: this is one vocabulary-sized vector, so
        # the copy is negligible beside the forward pass that produced it, and
        # it makes a given seed yield identical text on every device.
        probs = probs.to(generator.device)

    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())
