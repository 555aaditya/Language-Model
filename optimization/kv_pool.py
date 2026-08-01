"""Preallocated KV cache — the memory pooling deferred in TDD §7.2.

`attention.KVCache` grows by concatenation: every decode step allocates a new
`[B, H, T+1, D]` tensor and copies the old one into it. That is O(T) bytes
copied per step and O(T²) across a generation, plus a fresh allocation each
time for the allocator to serve and later reclaim.

This version allocates the full `[B, H, max_seq_len, D]` arena once and writes
each step into a slice, so the steady-state cost of a decode step is one write
of the new position and zero allocations. It is a strict drop-in: it subclasses
`KVCache`, so the type checker and every `CausalAttention` call site accept it
unchanged.

The tradeoff is honest and worth stating: peak memory is now `max_seq_len`
from the first token rather than growing with the sequence. For long-context
single-stream decoding that is a win; for many short concurrent sequences it
reserves far more than it uses. `nbytes` therefore reports the *reserved*
figure, with `nbytes_used` alongside it, so a benchmark cannot accidentally
quote the flattering one.
"""

from __future__ import annotations

import torch

from attention.kv_cache import KVCache
from model import CausalLM


class PreallocatedKVCache(KVCache):
    """Fixed-capacity KV arena written in place."""

    def __init__(
        self,
        batch_size: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        shape = (batch_size, n_kv_heads, max_seq_len, head_dim)
        self.keys = torch.zeros(shape, dtype=dtype, device=device)
        self.values = torch.zeros(shape, dtype=dtype, device=device)
        self.max_seq_len = max_seq_len
        self._length = 0

    def update(self, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.keys is not None and self.values is not None
        n = keys.shape[2]
        if self._length + n > self.max_seq_len:
            raise ValueError(
                f"KV arena full: holding {self._length} of {self.max_seq_len} "
                f"positions, cannot add {n} more"
            )
        if keys.shape[0] != self.keys.shape[0]:
            raise ValueError(
                f"batch size changed mid-generation: arena holds "
                f"{self.keys.shape[0]}, got {keys.shape[0]}"
            )

        end = self._length + n
        # Cast on write: under autocast the incoming K/V may be bf16 while the
        # arena is fp32. Upcasting is lossless; the reverse would not be, so the
        # arena dtype should match what the run actually produces.
        self.keys[:, :, self._length : end] = keys.to(self.keys.dtype)
        self.values[:, :, self._length : end] = values.to(self.values.dtype)
        self._length = end

        # Return views of the filled prefix, never the whole arena -- attention
        # must not see the zeroed tail as real keys.
        return self.keys[:, :, :end], self.values[:, :, :end]

    def __len__(self) -> int:
        return self._length

    def reset(self) -> None:
        """Rewind to empty while keeping the arena — that is the whole point."""
        self._length = 0

    @property
    def nbytes(self) -> int:
        """Bytes *reserved*. See ``nbytes_used`` for the occupied portion."""
        assert self.keys is not None and self.values is not None
        return (
            self.keys.numel() * self.keys.element_size()
            + self.values.numel() * self.values.element_size()
        )

    @property
    def nbytes_used(self) -> int:
        assert self.keys is not None
        per_position = self.keys[:, :, :1].numel() * self.keys.element_size()
        return 2 * per_position * self._length


def preallocated_cache(
    model: CausalLM,
    *,
    batch_size: int = 1,
    max_seq_len: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> list[PreallocatedKVCache]:
    """One arena per layer, sized from the model's own config.

    Drop-in for ``model.new_cache()``: pass the result straight to
    ``forward(..., kv_cache=..., use_cache=True)``.
    """
    attn = model.blocks[0].attn
    if device is None:
        device = next(model.parameters()).device
    return [
        PreallocatedKVCache(
            batch_size,
            attn.n_kv_heads,
            max_seq_len if max_seq_len is not None else attn.max_seq_len,
            attn.head_dim,
            dtype=dtype,
            device=device,
        )
        for _ in range(len(model.blocks))
    ]
