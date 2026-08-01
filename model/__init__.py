"""Decoder-only transformer: RMSNorm + SwiGLU blocks over grouped-query attention.

See docs/TDD.md §3 and TDR-005, TDR-017.
"""

from model.block import TransformerBlock
from model.causal_lm import CausalLM
from model.ffn import SwiGLU
from model.norm import RMSNorm

__all__ = ["CausalLM", "RMSNorm", "SwiGLU", "TransformerBlock"]
