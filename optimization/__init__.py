"""Post-training optimization: weight quantization and KV cache pooling.

See docs/TDD.md §7 and TDR-012.
"""

from optimization.kv_pool import PreallocatedKVCache, preallocated_cache
from optimization.quantize import (
    VALID_BITS,
    VALID_SCHEMES,
    QuantizedLinear,
    dequantize_int4,
    dequantize_int8,
    model_nbytes,
    quantize,
    quantize_int4,
    quantize_int8,
)

__all__ = [
    "VALID_BITS",
    "VALID_SCHEMES",
    "PreallocatedKVCache",
    "QuantizedLinear",
    "dequantize_int4",
    "dequantize_int8",
    "model_nbytes",
    "preallocated_cache",
    "quantize",
    "quantize_int4",
    "quantize_int8",
]
