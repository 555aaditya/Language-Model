"""Rotary Position Embeddings (TDR-015).

RoPE rotates each consecutive pair of channels in Q and K by an angle
proportional to the token's absolute position. Because a dot product between
two rotated vectors depends only on the *difference* of their angles, the
resulting attention score is a function of relative distance — absolute
position never appears in the score, only in the rotation applied on the way in.

Two details that are easy to get wrong:

- **V is never rotated.** RoPE encodes *where to look*, not *what is there*.
  Rotating V corrupts the values being aggregated.
- **Under KV caching the offset is the cache length, not the slice index.**
  A decode step passes a 1-token slice; rotating it at position 0 instead of
  position ``cache_len`` makes every generated token believe it is the start of
  the sequence. Prefill still looks perfect, so this bug survives most tests —
  hence ``test_cached_decode_matches_full_forward``.
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin rotation tables for a given head dimension."""

    # Declared so the type checker knows these are Tensors; `register_buffer`
    # alone leaves them typed as `Tensor | Module`.
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even to rotate in pairs, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Channel pair i rotates at frequency theta^(-2i/head_dim): low channels
        # turn slowly (they encode long-range position), high channels quickly.
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        # Duplicated halves so the table lines up with rotate_half's [-x2, x1] layout.
        emb = torch.cat((angles, angles), dim=-1)

        # persistent=False: these are pure functions of (head_dim, max_seq_len,
        # theta) and would otherwise be dead weight in every checkpoint.
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` of shape ``[seq_len, head_dim]`` for positions
        ``offset .. offset + seq_len``."""
        end = offset + seq_len
        if end > self.max_seq_len:
            raise ValueError(
                f"position {end} exceeds max_seq_len={self.max_seq_len}; "
                f"rebuild the rotary table or shorten the context"
            )
        return self.cos_cached[offset:end], self.sin_cached[offset:end]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map ``[x1, x2] -> [-x2, x1]`` over the split halves of the last dim."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``[B, H, T, head_dim]`` by the given cos/sin ``[T, head_dim]``."""
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return x * cos + rotate_half(x) * sin
