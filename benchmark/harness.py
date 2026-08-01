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

Latency is reported as **p50/p90/p95/p99, not just mean**. Step times are
right-skewed — an allocator growth or a page fault produces occasional large
outliers — so a mean quietly reports a number that no individual step achieved.

A percentile is only as good as the sample count behind it: ``n`` runs resolve
tail probabilities no finer than ``1/n``, so p99 needs ~100 runs to mean
anything a p90 doesn't already say. ``Timing.resolvable_percentile`` reports
that ceiling next to the numbers, because an undersampled p99 looks exactly
like a well-sampled one.
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
    stdev_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    @property
    def median_ms(self) -> float:
        """Alias for ``p50_ms`` — the same statistic under its readable name."""
        return self.p50_ms

    @property
    def resolvable_percentile(self) -> float:
        """Highest percentile this sample count can distinguish from ``max``.

        With ``n`` samples the smallest resolvable tail probability is ``1/n``,
        so 10 runs cannot say anything about p99 that it doesn't also say about
        p90 — both interpolate against the same top-most observation. Reported
        alongside the percentiles so an undersampled p99 is visibly
        undersampled rather than quietly meaningless.
        """
        return round(100.0 * (1.0 - 1.0 / self.runs), 2)

    @classmethod
    def from_samples(cls, samples_s: list[float]) -> Timing:
        if not samples_s:
            raise ValueError("need at least one sample")
        ms = sorted(s * 1000.0 for s in samples_s)

        # Linear interpolation between order statistics (the standard
        # "inclusive" estimator, matching numpy's default). The previous
        # `ms[int(q * n)]` indexing collapsed every high percentile onto the
        # maximum at small sample counts: with n=10, int(0.90*10) == int(0.99*10) == 9.
        cuts = statistics.quantiles(ms, n=100, method="inclusive") if len(ms) > 1 else None

        def pct(q: int) -> float:
            return ms[0] if cuts is None else cuts[q - 1]

        return cls(
            runs=len(ms),
            mean_ms=statistics.fmean(ms),
            stdev_ms=statistics.stdev(ms) if len(ms) > 1 else 0.0,
            p50_ms=pct(50),
            p90_ms=pct(90),
            p95_ms=pct(95),
            p99_ms=pct(99),
            min_ms=ms[0],
            max_ms=ms[-1],
        )

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "resolvable_percentile": self.resolvable_percentile}


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
