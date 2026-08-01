"""Per-layer key/value cache for autoregressive decoding (TDR-008).

Without a cache, generating token *n* recomputes attention over all *n-1* prior
tokens, so a full generation costs O(T²). With one, each step computes a single
new K/V pair and reads the rest, making it O(T) overall.

**Current implementation is concatenation-based**, i.e. each append allocates a
new tensor and copies the old one, which is O(T) per step and O(T²) copied bytes
across a generation. That is deliberate for now: it is obviously correct and the
copy is not the bottleneck at our context lengths. The preallocated ring buffer
that removes it is scoped as memory pooling in TDD §7.2 / build stage 6, where
it can be justified against a measured baseline rather than assumed (TDR-012).
"""

from __future__ import annotations

import torch


class KVCache:
    """Growable K/V store for one attention layer.

    Holds ``[B, n_kv_heads, T, head_dim]`` tensors — note ``n_kv_heads``, not
    ``n_heads``: the cache stores the *unexpanded* grouped-query heads, which is
    where GQA's memory saving actually lands (TDR-016). Expansion to full head
    count happens after the read, on values that are never stored.
    """

    def __init__(self) -> None:
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    def update(self, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this step's K/V and return the full cached history."""
        if self.keys is None or self.values is None:
            self.keys, self.values = keys, values
        else:
            if keys.shape[0] != self.keys.shape[0]:
                raise ValueError(
                    f"batch size changed mid-generation: cache holds "
                    f"{self.keys.shape[0]}, got {keys.shape[0]}"
                )
            self.keys = torch.cat((self.keys, keys), dim=2)
            self.values = torch.cat((self.values, values), dim=2)
        return self.keys, self.values

    def __len__(self) -> int:
        """Number of positions currently cached."""
        return 0 if self.keys is None else int(self.keys.shape[2])

    @property
    def nbytes(self) -> int:
        """Memory held by this cache — the quantity GQA is trying to shrink."""
        if self.keys is None or self.values is None:
            return 0
        return (
            self.keys.numel() * self.keys.element_size()
            + self.values.numel() * self.values.element_size()
        )

    def reset(self) -> None:
        self.keys = None
        self.values = None
