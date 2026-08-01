"""Pre-norm transformer block (TDR-005, TDR-017)."""

from __future__ import annotations

import torch
from torch import nn

from attention import CausalAttention, KVCache
from model.ffn import SwiGLU
from model.norm import RMSNorm


class TransformerBlock(nn.Module):
    """``x + attn(norm(x))`` then ``x + ffn(norm(x))``.

    Pre-norm, not post-norm: the residual stream is never normalised, so there
    is an unobstructed identity path from input to output through the whole
    stack. That is what makes deep pre-norm models trainable without a delicate
    warmup schedule. It also means a block whose sublayers output zero is
    exactly the identity, which is how ``test_block_is_pre_norm_with_live_residuals``
    distinguishes this from a post-norm arrangement.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        *,
        max_seq_len: int = 1024,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        impl: str = "sdpa",
        bias: bool = False,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = CausalAttention(
            d_model,
            n_heads,
            n_kv_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            dropout=dropout,
            impl=impl,
            bias=bias,
        )
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLU(d_model, d_ff, bias=bias, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_out, new_cache = self.attn(
            self.attn_norm(x),
            kv_cache=kv_cache,
            use_cache=use_cache,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache
