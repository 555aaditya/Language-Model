"""Root-mean-square layer normalisation (TDR-017)."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """LayerNorm with the mean-centring and the bias removed.

    ``x * rsqrt(mean(x²) + eps) * weight``

    Two things follow from dropping the mean subtraction: one fewer reduction
    pass over the last dimension, and a constant offset in the input survives
    normalisation instead of being annihilated. The second is what
    ``test_rmsnorm_does_not_centre_the_mean`` checks — swapping a LayerNorm in
    here would keep every shape valid and every other test passing.

    The reduction runs in fp32 regardless of input dtype. Under autocast the
    activations arrive as bf16/fp16, and squaring a large half-precision
    activation overflows to ``inf`` well before the mean is taken.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(dtype) * self.weight

    def extra_repr(self) -> str:
        return f"dim={tuple(self.weight.shape)[0]}, eps={self.eps}"
