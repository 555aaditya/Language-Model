"""Sampling and autoregressive generation over the KV cache.

See docs/TDD.md §6 and TDR-008.
"""

from inference.generate import generate, generate_ids
from inference.sampling import apply_temperature, apply_top_k, apply_top_p, sample

__all__ = [
    "apply_temperature",
    "apply_top_k",
    "apply_top_p",
    "generate",
    "generate_ids",
    "sample",
]
