"""Causal grouped-query attention with RoPE and an optional KV cache.

Contract (docs/BUILD_ORDER.md):

    CausalAttention.forward(x, *, kv_cache=None, use_cache=False)
        -> (out, new_kv_cache)

The three masking regimes, which are the subtle part:

| q_len | k_len       | situation           | masking                          |
|-------|-------------|---------------------|----------------------------------|
| T     | T           | training / prefill  | standard causal triangle         |
| 1     | cache + 1   | decode step         | none — one query sees everything |
| n     | cache + n   | chunked prefill     | explicit bottom-right triangle   |

The third row is why we cannot simply pass ``is_causal=True`` everywhere: torch
aligns its implicit triangle to the top-left, which is only equivalent to ours
when ``q_len == k_len``.
"""

from __future__ import annotations

import torch
from torch import nn

from attention.kernels import (
    causal_block_mask,
    flash_attention,
    manual_attention,
    repeat_kv,
    sdpa_attention,
)
from attention.kv_cache import KVCache
from attention.rope import RotaryEmbedding, apply_rotary_emb

VALID_IMPLS = ("manual", "sdpa", "flash")


class CausalAttention(nn.Module):
    """Multi-head causal attention with grouped KV heads and rotary positions."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        *,
        max_seq_len: int = 1024,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        impl: str = "sdpa",
        bias: bool = False,
        block_q: int = 64,
        block_k: int = 64,
    ) -> None:
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads} "
                f"so each KV head serves an equal-sized query group"
            )
        if impl not in VALID_IMPLS:
            raise ValueError(f"unknown attention impl {impl!r}; expected one of {VALID_IMPLS}")
        if impl == "flash" and dropout > 0.0:
            # The tiled path never materialises normalised probabilities, so
            # there is nothing to drop out of; silently ignoring the setting
            # would make `flash` train differently from `manual`.
            raise ValueError("attention dropout is not supported by impl='flash'")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.impl = impl
        self.block_q = block_q
        self.block_k = block_k

        # Q is full width; K/V are narrower by the group ratio. This asymmetry
        # is why the fused Linear(C, 3C) of a classic MHA block does not apply.
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, theta=rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """``key_padding_mask`` is ``[batch, k_len]``, True where a key is real.

        Needed for batched generation over ragged prompts: short prompts are
        left-padded, and without masking those pad positions every short request
        attends to filler as though it were context.
        """
        batch, q_len, _ = x.shape
        offset = len(kv_cache) if kv_cache is not None else 0

        if offset + q_len > self.max_seq_len:
            raise ValueError(
                f"context length {offset + q_len} exceeds max_seq_len={self.max_seq_len}"
            )
        if kv_cache is not None and kv_cache.keys is not None:
            if kv_cache.keys.shape[0] != batch:
                raise ValueError(
                    f"batch size changed mid-generation: cache holds "
                    f"{kv_cache.keys.shape[0]}, got {batch}"
                )

        # [B, T, C] -> [B, H, T, head_dim]
        q = self.q_proj(x).view(batch, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Rotate at absolute positions. `offset` is the cache length, so a decode
        # step rotates its single token at its true position rather than at 0.
        cos, sin = self.rope(q_len, offset=offset)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Cache the *pre-expansion* KV heads: storing the repeated copies would
        # throw away the entire GQA memory saving.
        if use_cache:
            kv_cache = kv_cache if kv_cache is not None else KVCache()
            k, v = kv_cache.update(k, v)
        elif kv_cache is not None:
            k, v = kv_cache.update(k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        out = self._attend(q, k, v, offset=offset, key_padding_mask=key_padding_mask)

        out = out.transpose(1, 2).reshape(batch, q_len, self.n_heads * self.head_dim)
        out = self.resid_dropout(self.o_proj(out))
        return out, (kv_cache if use_cache else None)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        offset: int,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_len, k_len = q.shape[2], k.shape[2]

        # A single query attends over the whole cache: nothing to mask -- unless
        # some of that cache is padding.
        no_mask_needed = q_len == 1 and key_padding_mask is None
        # Square case only: torch's implicit triangle is top-left aligned.
        plain_causal = q_len == k_len and key_padding_mask is None

        if self.impl == "flash":
            if key_padding_mask is not None:
                # The tiled kernel builds its mask per tile from positions alone.
                # Threading a per-batch padding mask through it is real work, and
                # a silent fallback to another kernel would hide which code path
                # actually ran -- so refuse explicitly.
                raise ValueError(
                    "impl='flash' does not support key_padding_mask; use 'sdpa' or "
                    "'manual' for batched generation over ragged prompts"
                )
            return flash_attention(
                q,
                k,
                v,
                causal=not no_mask_needed,
                offset=offset,
                block_q=self.block_q,
                block_k=self.block_k,
            )

        if no_mask_needed:
            mask = None
        elif plain_causal and self.impl == "sdpa":
            mask = None  # handled by is_causal below, which skips materialising it
        else:
            mask = causal_block_mask(q_len, k_len, offset, q.device)

        if key_padding_mask is not None:
            if key_padding_mask.shape != (q.shape[0], k_len):
                raise ValueError(
                    f"key_padding_mask must be [batch, k_len] = "
                    f"{(q.shape[0], k_len)}, got {tuple(key_padding_mask.shape)}"
                )
            # Broadcast [B, k_len] -> [B, 1, 1, k_len] and union with the causal
            # triangle. A key is blocked if it is in the future OR is padding.
            padded = ~key_padding_mask[:, None, None, :]
            mask = padded if mask is None else (mask | padded)

        if self.impl == "sdpa":
            return sdpa_attention(
                q,
                k,
                v,
                mask=mask,
                is_causal=plain_causal and not no_mask_needed,
                dropout_p=self.dropout,
                training=self.training,
            )
        return manual_attention(q, k, v, mask=mask, dropout_p=self.dropout, training=self.training)
