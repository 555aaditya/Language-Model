"""Cosine learning-rate decay with linear warmup (TDR-009).

    step < warmup :  base_lr * (step + 1) / warmup          — linear ramp
    otherwise     :  min_lr + ½(base_lr − min_lr)(1 + cos(π · progress))

Warmup uses ``(step + 1) / warmup`` rather than ``step / warmup`` so the very
first step actually moves; a zero learning rate on step 0 is a wasted step.

The two pieces meet exactly at ``step == warmup``: the ramp's last value and the
cosine's value at ``progress = 0`` are both ``base_lr``, so there is no
discontinuity to show up as a loss spike.
"""

from __future__ import annotations

import math

import torch


class CosineScheduler:
    """Stateless schedule — the LR is a pure function of the step number.

    Stateless on purpose: resuming from a checkpoint only needs the step count,
    with no scheduler state to save, restore, or get out of sync with the
    optimizer.
    """

    def __init__(self, base_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> None:
        if min_lr > base_lr:
            raise ValueError(f"min_lr ({min_lr}) must not exceed base_lr ({base_lr})")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if total_steps < warmup_steps:
            raise ValueError(
                f"total_steps ({total_steps}) must be >= warmup_steps ({warmup_steps})"
            )
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps

        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def apply(self, optimizer: torch.optim.Optimizer, step: int) -> float:
        """Write the rate for ``step`` onto every param group. Returns it."""
        lr = self.lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    @classmethod
    def from_config(cls, cfg: dict) -> CosineScheduler:
        t = cfg["training"]
        return cls(
            base_lr=float(t["lr"]),
            min_lr=float(t.get("min_lr", float(t["lr"]) * 0.1)),
            warmup_steps=int(t.get("warmup_steps", 0)),
            total_steps=int(t["steps"]),
        )
