"""Optimizer and LR schedule tests (TDD - written before implementation).

The AdamW here is hand-written (TDR-009 / no-external-implementations rule), so
the tests treat ``torch.optim.AdamW`` purely as a numerical oracle: given the
same parameters, gradients and hyperparameters, the two must agree step for
step. Anything less and "custom AdamW" means "an optimizer that roughly
descends", which is not a claim worth making.
"""

import math

import pytest
import torch
from torch import nn

from training.optimizer import AdamW, build_param_groups
from training.scheduler import CosineScheduler

# ---------------------------------------------------------------------------
# AdamW numerics
# ---------------------------------------------------------------------------


def _twin_params(seed=0, shape=(6, 4)):
    """Two independent parameters initialised identically."""
    torch.manual_seed(seed)
    base = torch.randn(*shape)
    return nn.Parameter(base.clone()), nn.Parameter(base.clone())


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("betas", [(0.9, 0.999), (0.9, 0.95)])
def test_matches_torch_adamw_step_for_step(weight_decay, betas):
    mine_p, torch_p = _twin_params()
    mine = AdamW([mine_p], lr=1e-2, betas=betas, eps=1e-8, weight_decay=weight_decay)
    theirs = torch.optim.AdamW([torch_p], lr=1e-2, betas=betas, eps=1e-8, weight_decay=weight_decay)

    torch.manual_seed(1)
    for _ in range(25):
        grad = torch.randn_like(mine_p)
        mine_p.grad, torch_p.grad = grad.clone(), grad.clone()
        mine.step()
        theirs.step()
        mine.zero_grad()
        theirs.zero_grad()

    torch.testing.assert_close(mine_p, torch_p, rtol=1e-6, atol=1e-7)


def test_bias_correction_makes_the_first_step_full_size():
    """Without bias correction the first step is ~(1-beta1)x too small."""
    param = nn.Parameter(torch.zeros(1))
    opt = AdamW([param], lr=0.1, weight_decay=0.0)
    param.grad = torch.ones(1)
    opt.step()
    # first step should be almost exactly -lr, not -lr * 0.1
    torch.testing.assert_close(param.detach(), torch.tensor([-0.1]), rtol=1e-4, atol=1e-5)


def test_weight_decay_is_decoupled_from_the_gradient():
    """A zero-gradient parameter must still decay by exactly (1 - lr*wd).

    Coupled (L2) decay would add wd*p into the gradient, where Adam's
    normalisation would rescale it to roughly a fixed step size -- so this
    exact factor is what distinguishes AdamW from Adam+L2.
    """
    param = nn.Parameter(torch.full((3,), 2.0))
    opt = AdamW([param], lr=0.1, weight_decay=0.5)
    param.grad = torch.zeros(3)
    opt.step()
    torch.testing.assert_close(param.detach(), torch.full((3,), 2.0 * (1 - 0.1 * 0.5)))


def test_params_without_gradients_are_skipped():
    used, unused = nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(2))
    opt = AdamW([used, unused], lr=0.1, weight_decay=0.0)
    used.grad = torch.ones(2)
    opt.step()
    assert not torch.equal(used.detach(), torch.ones(2))
    torch.testing.assert_close(unused.detach(), torch.ones(2))


def test_rejects_nonsense_hyperparameters():
    param = nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="lr"):
        AdamW([param], lr=-1.0)
    with pytest.raises(ValueError, match="beta"):
        AdamW([param], betas=(1.5, 0.999))
    with pytest.raises(ValueError, match="eps"):
        AdamW([param], eps=-1e-8)


# ---------------------------------------------------------------------------
# Parameter grouping
# ---------------------------------------------------------------------------


def test_norms_and_biases_are_excluded_from_weight_decay():
    """Decaying a 1-D gain or bias toward zero degrades the model for nothing."""
    net = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    decay, no_decay = build_param_groups(net, weight_decay=0.1)

    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"])
    assert all(p.ndim < 2 for p in no_decay["params"])
    # every parameter must land in exactly one group
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(net.parameters()))


def test_tied_weights_are_not_decayed_twice():
    """A tied embedding/head appears under two names but is one tensor."""
    net = nn.Module()
    net.embed = nn.Embedding(8, 4)
    net.head = nn.Linear(4, 8, bias=False)
    net.head.weight = net.embed.weight

    decay, no_decay = build_param_groups(net, weight_decay=0.1)
    ids = [id(p) for p in decay["params"] + no_decay["params"]]
    assert len(ids) == len(set(ids)), "the same tensor was added to a group twice"


# ---------------------------------------------------------------------------
# Cosine schedule with linear warmup
# ---------------------------------------------------------------------------


@pytest.fixture
def sched():
    return CosineScheduler(base_lr=1.0, min_lr=0.1, warmup_steps=10, total_steps=110)


def test_warmup_rises_linearly_to_the_peak(sched):
    lrs = [sched.lr_at(s) for s in range(10)]
    assert lrs == sorted(lrs)
    assert lrs[0] == pytest.approx(0.1)  # 1/10 of base, never exactly zero
    assert lrs[-1] == pytest.approx(1.0)


def test_first_step_is_not_zero(sched):
    """A zero-LR first step is a wasted step; warmup uses (step+1)/warmup."""
    assert sched.lr_at(0) > 0


def test_peak_is_reached_exactly_at_warmup_end(sched):
    assert sched.lr_at(10) == pytest.approx(1.0)


def test_decays_to_min_lr_at_the_final_step(sched):
    assert sched.lr_at(110) == pytest.approx(0.1)


def test_never_drops_below_min_lr_after_the_end(sched):
    assert sched.lr_at(500) == pytest.approx(0.1)


def test_decay_is_monotonic_after_warmup(sched):
    lrs = [sched.lr_at(s) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(lrs[:-1], lrs[1:], strict=True))


def test_midpoint_is_the_cosine_halfway_value(sched):
    """Halfway through decay a cosine sits at the arithmetic mean, not 0.5x."""
    assert sched.lr_at(60) == pytest.approx(0.1 + 0.5 * (1.0 - 0.1))


def test_zero_warmup_starts_at_the_peak():
    assert CosineScheduler(1.0, 0.1, 0, 100).lr_at(0) == pytest.approx(1.0)


def test_scheduler_writes_the_rate_onto_the_optimizer():
    param = nn.Parameter(torch.ones(2))
    opt = AdamW([param], lr=999.0)
    lr = CosineScheduler(1.0, 0.1, 10, 110).apply(opt, step=0)
    assert lr == pytest.approx(0.1)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)


def test_schedule_shape_matches_the_closed_form():
    s = CosineScheduler(base_lr=3e-4, min_lr=3e-5, warmup_steps=5, total_steps=25)
    for step in range(5, 26):
        progress = (step - 5) / 20
        expected = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * progress))
        assert s.lr_at(step) == pytest.approx(expected)
