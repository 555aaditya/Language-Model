"""Hand-written AdamW (TDR-009).

Implemented directly rather than imported, per the project's no-external-
implementations rule. It is written to be *bit-comparable* with
``torch.optim.AdamW`` given the same hyperparameters — that equivalence is the
only thing that makes "custom optimizer" a defensible claim, and
``test_matches_torch_adamw_step_for_step`` enforces it.

The update, in order:

    p  <- p * (1 - lr * weight_decay)          # decoupled, applied to the weight
    m  <- b1*m + (1-b1)*g
    v  <- b2*v + (1-b2)*g²
    p  <- p - (lr / (1-b1^t)) * m / (sqrt(v)/sqrt(1-b2^t) + eps)

The decay is applied to the *parameter*, not folded into the gradient. Folding
it in (plain L2) means Adam's per-parameter normalisation rescales the decay
too, so the effective decay ends up depending on the gradient magnitude — which
is the whole reason AdamW exists.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


class AdamW(torch.optim.Optimizer):
    """Adam with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        super().__init__(
            params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t

                denom = (exp_avg_sq.sqrt() / bias_correction2**0.5).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

        return loss


def build_param_groups(
    module: nn.Module, weight_decay: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split parameters into decayed (matrices) and undecayed (1-D) groups.

    Weight decay is a prior toward small *weights*. Applying it to a 1-D
    RMSNorm gain or a bias pulls a learned scale toward zero for no
    regularisation benefit, so those are excluded — standard practice, and
    cheap to get wrong silently.

    Tied parameters appear under more than one name but are one tensor;
    de-duplicating by identity keeps them from being decayed twice per step.
    """
    seen: set[int] = set()
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    for param in module.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        (decay if param.ndim >= 2 else no_decay).append(param)

    return (
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    )
