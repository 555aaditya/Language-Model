"""Memory metrics (TDD §9).

Unlike latency, these numbers are **exact and machine-independent** — they are
computed from tensor shapes and element sizes, not measured. That makes them the
only part of the benchmark suite that can be asserted against a recorded
baseline in CI, which is what `test_memory_report_reproduces_the_baseline` does.

Deliberately *not* using peak-allocator readings here: `torch.cuda.max_memory_allocated`
has no MPS equivalent (TDR-019), so a peak-memory number would be CUDA-only and
not comparable across the devices this project actually runs on.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from optimization.quantize import model_nbytes, quantize

MIB = 1024 * 1024


def model_memory_report(model: nn.Module) -> dict[str, Any]:
    """Bytes held by the model under each precision, plus compression ratios."""
    fp32 = model_nbytes(model)
    variants = {
        "fp32": fp32,
        "fp16": model_nbytes(quantize(model, scheme="fp16")),
        "int8": model_nbytes(quantize(model, bits=8)),
        "int4": model_nbytes(quantize(model, bits=4)),
    }
    return {
        "bytes": variants,
        "mib": {k: round(v / MIB, 4) for k, v in variants.items()},
        "compression_vs_fp32": {k: round(fp32 / v, 4) for k, v in variants.items()},
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def kv_cache_memory(
    *,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    bytes_per_element: int = 4,
) -> int:
    """Bytes a full KV cache occupies: 2 (K and V) x layers x heads x T x dim."""
    return 2 * n_layers * batch_size * n_kv_heads * seq_len * head_dim * bytes_per_element


def gqa_saving_report(
    *,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
) -> dict[str, Any]:
    """What grouped-query attention actually saves on the cache (TDR-016).

    Quoted per *config*, not per measurement: the saving is exactly the head
    ratio, so reporting it as an empirical result would dress up arithmetic as
    an experiment.
    """
    common = {
        "n_layers": n_layers,
        "head_dim": head_dim,
        "seq_len": seq_len,
        "batch_size": batch_size,
    }
    mha = kv_cache_memory(n_kv_heads=n_heads, **common)
    gqa = kv_cache_memory(n_kv_heads=n_kv_heads, **common)
    return {
        "mha_bytes": mha,
        "gqa_bytes": gqa,
        "mha_mib": round(mha / MIB, 4),
        "gqa_mib": round(gqa / MIB, 4),
        "saving_ratio": round(mha / gqa, 4),
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
    }


def activation_memory_estimate(
    *, batch_size: int, n_heads: int, seq_len: int, bytes_per_element: int = 4
) -> dict[str, Any]:
    """Score-matrix footprint for standard vs tiled attention (TDR-007).

    Standard attention materialises ``[B, H, T, T]``; the tiled kernel never
    holds more than one ``[block_q, block_k]`` tile, so its score memory is
    independent of ``T``. This is the O(T²) → O(T) claim, stated in bytes.
    """
    standard = batch_size * n_heads * seq_len * seq_len * bytes_per_element
    return {
        "standard_score_bytes": standard,
        "standard_score_mib": round(standard / MIB, 4),
        "seq_len": seq_len,
        "scales_as": "O(T^2)",
    }
