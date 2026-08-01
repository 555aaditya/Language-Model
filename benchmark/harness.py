"""Timing primitives (TDD §9).

The one thing a GPU benchmark must get right is **synchronisation**. CUDA and
MPS both queue work asynchronously, so a naive ``perf_counter`` around a forward
pass measures how long it took to *submit* the kernels, not to run them — which
reliably produces impossibly fast numbers that then get quoted. Every timed
region here syncs the device on both sides.

Warmup matters for the same reason: the first call on any device pays for
kernel compilation, lazy module init and allocator growth. Those costs are real
but they are not per-step costs, so folding them into the average
misrepresents steady state.

Latency is reported as **median and percentiles, not just mean**. Step times are
right-skewed — an allocator growth or a page fault produces occasional large
outliers — so a mean quietly reports a number that no individual step achieved.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch


def synchronize(device: torch.device | str | None) -> None:
    """Block until queued work on ``device`` has actually finished."""
    if device is None:
        return
    kind = torch.device(device).type
    if kind == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif kind == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    # CPU is synchronous already.


@dataclass(frozen=True)
class Timing:
    """Wall-clock statistics for a repeated operation, in milliseconds."""

    runs: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_samples(cls, samples_s: list[float]) -> Timing:
        ms = sorted(s * 1000.0 for s in samples_s)
        return cls(
            runs=len(ms),
            mean_ms=statistics.fmean(ms),
            median_ms=statistics.median(ms),
            p90_ms=ms[min(len(ms) - 1, int(0.90 * len(ms)))],
            p99_ms=ms[min(len(ms) - 1, int(0.99 * len(ms)))],
            min_ms=ms[0],
            max_ms=ms[-1],
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def time_it(
    fn: Callable[[], Any],
    *,
    warmup: int = 3,
    repeats: int = 10,
    device: torch.device | str | None = None,
) -> Timing:
    """Run ``fn`` ``repeats`` times after ``warmup`` untimed calls."""
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    for _ in range(warmup):
        fn()
    synchronize(device)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        synchronize(device)  # inside the timed region, or this measures nothing
        samples.append(time.perf_counter() - start)

    return Timing.from_samples(samples)
