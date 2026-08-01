"""The training loop (TDD §5).

One ``step()`` is one *optimizer* step, which may span several micro-batches:

    for _ in range(grad_accum_steps):
        loss = cross_entropy(model(input_ids), labels) / grad_accum_steps
        loss.backward()
    grad_norm = clip_grad_norm_(params, grad_clip)
    scheduler.apply(optimizer, step)
    optimizer.step(); optimizer.zero_grad()

Dividing the micro-batch loss by ``grad_accum_steps`` is what makes accumulation
equivalent to a single larger batch rather than to a larger *learning rate* —
gradients sum across the backward passes, so without the division the effective
step is K times too big.

Mixed precision follows the device (TDR-019): ``GradScaler`` exists only for
CUDA fp16. bf16 has fp32's exponent range and needs no loss scaling, and MPS has
no scaler at all, so on those paths autocast runs bare.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import nn

from training.checkpoint import load_checkpoint, save_checkpoint
from training.device import autocast_dtype, resolve_device, supports_grad_scaler
from training.optimizer import AdamW, build_param_groups
from training.scheduler import CosineScheduler


class BatchSource(Protocol):
    """Anything that can hand over the next batch — DataEngine satisfies this."""

    def next_batch(self, device: torch.device | str | None = ...) -> dict[str, torch.Tensor]: ...


class Trainer:
    """Owns the model, optimizer, schedule and step counter for one run."""

    def __init__(
        self,
        model: nn.Module,
        engine: BatchSource,
        cfg: dict[str, Any],
        *,
        device: torch.device | None = None,
    ) -> None:
        t = cfg.get("training", {})
        self.cfg = cfg
        self.model = model
        self.engine = engine
        self.device = device if device is not None else resolve_device(cfg.get("device"))
        self.model.to(self.device)

        self.grad_accum_steps = max(1, int(t.get("grad_accum_steps", 1)))
        self.grad_clip = float(t.get("grad_clip", 0.0))
        self.total_steps = int(t.get("steps", 0))
        self.log_every = int(t.get("log_every", 50))
        self.step_count = 0
        # Best-validation tracking, so a run can keep its best weights rather
        # than only its most recent ones.
        self.best_val_loss = float("inf")
        self.best_step = -1
        self.evals_since_improvement = 0

        decay, no_decay = build_param_groups(model, float(t.get("weight_decay", 0.0)))
        self.optimizer = AdamW(
            [decay, no_decay],
            lr=float(t.get("lr", 3e-4)),
            betas=(float(t.get("beta1", 0.9)), float(t.get("beta2", 0.95))),
            eps=float(t.get("eps", 1e-8)),
        )
        self.scheduler = CosineScheduler(
            base_lr=float(t.get("lr", 3e-4)),
            min_lr=float(t.get("min_lr", float(t.get("lr", 3e-4)) * 0.1)),
            warmup_steps=int(t.get("warmup_steps", 0)),
            total_steps=max(1, self.total_steps),
        )

        self.amp_dtype = (
            autocast_dtype(self.device, str(t.get("amp_dtype", "bfloat16")))
            if t.get("amp", True)
            else None
        )
        use_scaler = supports_grad_scaler(self.device) and self.amp_dtype is torch.float16
        self.scaler = torch.amp.GradScaler(enabled=use_scaler)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _autocast(self) -> Any:
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def loss_on(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Next-token cross-entropy. Lives here, not in the model (BUILD_ORDER)."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        with self._autocast():
            logits = self.model(input_ids)
            # float() before cross-entropy: under bf16 autocast the softmax
            # denominator is summed over the whole vocabulary, where reduced
            # mantissa shows up directly in the loss.
            return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))

    # ------------------------------------------------------------------
    # One optimizer step
    # ------------------------------------------------------------------
    def step(self) -> dict[str, float]:
        self.model.train()
        started = time.perf_counter()
        total_loss = 0.0
        tokens = 0

        for _ in range(self.grad_accum_steps):
            batch = self.engine.next_batch(self.device)
            loss = self.loss_on(batch) / self.grad_accum_steps
            self.scaler.scale(loss).backward()
            total_loss += loss.item()
            tokens += batch["input_ids"].numel()

        if self.grad_clip > 0:
            # Unscale first, or the clip threshold is applied to scaled
            # gradients and effectively does nothing.
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            )
        else:
            grad_norm = float("nan")

        lr = self.scheduler.apply(self.optimizer, self.step_count)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.step_count += 1

        elapsed = max(time.perf_counter() - started, 1e-9)
        return {
            "step": float(self.step_count),
            "loss": total_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "tokens": float(tokens),
            "tokens_per_sec": tokens / elapsed,
        }

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def train(
        self,
        steps: int | None = None,
        *,
        on_log: Callable[[dict[str, float]], None] | None = None,
    ) -> list[dict[str, float]]:
        history = []
        for _ in range(steps if steps is not None else self.total_steps):
            metrics = self.step()
            history.append(metrics)
            if on_log and self.step_count % max(1, self.log_every) == 0:
                on_log(metrics)
        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, engine: BatchSource, *, batches: int = 20) -> dict[str, float]:
        """Mean loss and perplexity over ``batches`` from a held-out source.

        Averages the *loss* and exponentiates once at the end. Averaging
        perplexities instead would be wrong — perplexity is exp of a mean, and
        the mean of exponentials is not the exponential of the mean (Jensen), so
        per-batch averaging reports a number that is always too high.

        Also tracks the best validation loss seen and how many evaluations have
        passed without improvement, which is what makes best-checkpoint saving
        and early stopping possible. Without it a run happily trains past its own
        optimum and keeps only the *worst* model: the first TinyShakespeare run
        bottomed out at validation perplexity 98.9 around step 500 and finished at
        304.6, and because checkpoints were periodic rather than
        best-tracked, the good weights were never written to disk.

        Restores the previous train/eval mode so this can be called mid-loop
        without silently leaving dropout off for the rest of training.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            total = 0.0
            for _ in range(batches):
                total += self.loss_on(engine.next_batch(self.device)).item()
            mean = total / max(1, batches)

            improved = mean < self.best_val_loss
            if improved:
                self.best_val_loss = mean
                self.best_step = self.step_count
                self.evals_since_improvement = 0
            else:
                self.evals_since_improvement += 1

            return {
                "val_loss": mean,
                "val_perplexity": math.exp(min(mean, 80.0)),
                "val_improved": float(improved),
                "best_val_loss": self.best_val_loss,
                "best_step": float(self.best_step),
                "evals_since_improvement": float(self.evals_since_improvement),
            }
        finally:
            self.model.train(was_training)

    def should_stop_early(self, patience: int) -> bool:
        """True once ``patience`` consecutive evaluations have failed to improve.

        ``patience <= 0`` disables the check. Patience is counted in
        *evaluations*, not steps, so it means the same thing regardless of how
        often the loop evaluates.
        """
        return patience > 0 and self.evals_since_improvement >= patience

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            step=self.step_count,
            cfg=self.cfg,
        )

    def load(self, path: str) -> None:
        payload = load_checkpoint(
            path, model=self.model, optimizer=self.optimizer, map_location=self.device
        )
        self.step_count = int(payload.get("step", 0))
