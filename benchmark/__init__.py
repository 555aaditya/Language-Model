"""Benchmark harness: latency, throughput and memory (TDD §9, TDR-012).

Metrics split two ways. ``memory`` is exact arithmetic over tensor shapes and is
identical on every machine, so it can be asserted against a recorded baseline.
``latency`` and ``throughput`` are wall-clock and are reproducible only in
shape — always read them next to ``report.environment()``.
"""

from benchmark.harness import Timing, synchronize, time_it
from benchmark.latency import (
    benchmark_attention_impls,
    benchmark_decode,
    benchmark_prefill,
    kv_cache_speedup,
)
from benchmark.memory import (
    activation_memory_estimate,
    gqa_saving_report,
    kv_cache_memory,
    model_memory_report,
)
from benchmark.report import build_report, deterministic_report, environment
from benchmark.throughput import benchmark_training_step

__all__ = [
    "Timing",
    "activation_memory_estimate",
    "benchmark_attention_impls",
    "benchmark_decode",
    "benchmark_prefill",
    "benchmark_training_step",
    "build_report",
    "deterministic_report",
    "environment",
    "gqa_saving_report",
    "kv_cache_memory",
    "kv_cache_speedup",
    "model_memory_report",
    "synchronize",
    "time_it",
]
