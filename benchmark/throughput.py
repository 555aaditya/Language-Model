"""Training throughput (TDD §9).

Reports tokens/sec for a full optimizer step — forward, backward, clip and
update — because that is the number that determines how long a run takes.
Timing the forward pass alone overstates throughput by roughly 3x, since the
backward pass costs about twice the forward.
"""

from __future__ import annotations

from typing import Any

from benchmark.harness import synchronize
from training.trainer import Trainer


def benchmark_training_step(
    trainer: Trainer, *, warmup: int = 3, repeats: int = 10
) -> dict[str, Any]:
    """Measure steady-state step time on an already-constructed Trainer.

    Warmup steps are real optimizer steps, so this mutates the model — it is a
    throughput probe, not something to run against weights you care about.
    """
    for _ in range(warmup):
        trainer.step()
    synchronize(trainer.device)

    samples_ms = []
    tokens = 0
    for _ in range(repeats):
        metrics = trainer.step()
        synchronize(trainer.device)
        # Trainer already measures its own step wall-clock; reuse it rather than
        # double-counting with an outer timer.
        samples_ms.append(metrics["tokens"] / metrics["tokens_per_sec"] * 1000.0)
        tokens += int(metrics["tokens"])

    samples_ms.sort()
    median = samples_ms[len(samples_ms) // 2]
    return {
        "steps": repeats,
        "median_step_ms": round(median, 4),
        "tokens_per_step": tokens // repeats,
        "tokens_per_sec": round((tokens // repeats) / (median / 1000.0), 2),
        "device": str(trainer.device),
        "amp_dtype": str(trainer.amp_dtype),
        "grad_accum_steps": trainer.grad_accum_steps,
    }
