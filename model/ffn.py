"""SwiGLU feed-forward network (TDR-017)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    """Gated feed-forward block: ``down(silu(gate(x)) * up(x))``.

    The ``*`` is a genuine multiplicative gate, not a sum: when ``gate(x)`` is
    strongly negative, ``silu`` drives it to zero and the whole branch is
    suppressed regardless of what ``up(x)`` computed. That is the property a
    plain GELU MLP lacks, and it costs a third weight matrix.

    Because of that third matrix, a parameter-neutral comparison against a
    ``4·d_model`` GELU MLP would need ``d_ff ≈ 8/3·d_model``. The shipped config
    uses ``d_ff = 4·d_model``, so the FFN carries ~1.5× the parameters of the
    GPT-2 baseline. That is a deliberate choice at this scale, recorded rather
    than presented as parity.
    """

    def __init__(
        self, d_model: int, d_ff: int, *, bias: bool = False, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        out: torch.Tensor = self.dropout(self.down_proj(gated))
        return out
