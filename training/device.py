"""Device selection (TDR-019).

Preference order is ``mps -> cuda -> cpu``: Apple Silicon is the hardware this
project is developed and measured on, CUDA is supported but untested here, and
CPU is the always-correct fallback.

MPS is not a drop-in CUDA. Two differences that leak into the layers above:

- There is no ``torch.cuda.max_memory_allocated`` equivalent, so peak-memory
  metrics have to come from ``tracemalloc``/RSS instead and are not directly
  comparable to CUDA numbers.
- ``GradScaler`` is CUDA-only, and bf16 does not need loss scaling anyway. AMP
  on MPS therefore means ``autocast(bfloat16)`` with no scaler.
"""

from __future__ import annotations

import torch


def resolve_device(preferred: str | None = None) -> torch.device:
    """Return the device to run on. ``preferred`` overrides auto-selection."""
    if preferred and preferred != "auto":
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_dtype(device: torch.device, requested: str = "bfloat16") -> torch.dtype | None:
    """AMP dtype for ``device``, or ``None`` when mixed precision should be off.

    fp16 on CPU is emulated and slower than fp32, so CPU always returns ``None``
    rather than pretending AMP helps.
    """
    if device.type == "cpu":
        return None
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(requested)


def supports_grad_scaler(device: torch.device) -> bool:
    """``GradScaler`` is a CUDA fp16 mechanism; bf16 and MPS do not use it."""
    return device.type == "cuda"
