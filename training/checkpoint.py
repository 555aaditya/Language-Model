"""Checkpoint save/load.

What is saved: model weights, optimizer state, step counter, the config the run
was launched with, and RNG state.

**What is deliberately not saved: the data loader's position.** Resuming
therefore restarts the corpus from the beginning of a shuffled epoch rather
than the exact window the run died on. For a shuffled multi-epoch run that is
harmless; for a single-pass run over a large corpus it means some windows are
seen twice and others not at all. Recording it here rather than implying
resumption is exact.

Optimizer state matters more than it looks. Dropping it resets Adam's first and
second moments to zero, so the first step after a resume takes a full
bias-corrected stride in whatever direction the current gradient points —
usually a visible loss spike. ``test_resume_restores_optimizer_moments`` pins it.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    cfg: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "cfg": cfg,
        "rng": {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        },
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore into ``model`` (and ``optimizer``). Returns the payload."""
    # weights_only=False: the payload carries RNG state and the config dict, not
    # just tensors. Only ever load checkpoints this project wrote.
    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)

    # strict=True on purpose: a silently partial load produces a model that is
    # half-trained and half-random, which trains without ever looking broken.
    model.load_state_dict(payload["model"], strict=True)

    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])

    if restore_rng and payload.get("rng"):
        rng = payload["rng"]
        torch.set_rng_state(rng["torch"])
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])

    return payload
