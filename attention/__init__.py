"""Causal grouped-query attention with RoPE, a KV cache, and three kernels.

See docs/TDD.md §3.3 / §4 and TDR-007, TDR-008, TDR-015, TDR-016.
"""

from attention.causal import VALID_IMPLS, CausalAttention
from attention.kernels import (
    causal_block_mask,
    flash_attention,
    manual_attention,
    repeat_kv,
    sdpa_attention,
)
from attention.kv_cache import KVCache
from attention.rope import RotaryEmbedding, apply_rotary_emb, rotate_half

__all__ = [
    "VALID_IMPLS",
    "CausalAttention",
    "KVCache",
    "RotaryEmbedding",
    "apply_rotary_emb",
    "causal_block_mask",
    "flash_attention",
    "manual_attention",
    "repeat_kv",
    "rotate_half",
    "sdpa_attention",
]
