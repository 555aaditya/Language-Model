"""Trainer and checkpoint tests (TDD - written before implementation).

Exit criterion for build stage 4 (docs/BUILD_ORDER.md): loss falls when
overfitting a single batch. That test is the one that proves the whole stack
composes -- a broken causal mask, a mis-shifted label, a dead gradient path or
a scheduler stuck at zero all show up as a flat loss curve here.
"""

import math

import pytest
import torch

from model import CausalLM
from training.checkpoint import load_checkpoint, save_checkpoint
from training.trainer import Trainer

VOCAB, SEQ, BATCH = 32, 8, 4


class FixedEngine:
    """Replays a fixed batch list so tests are deterministic.

    Satisfies the same duck-typed `next_batch` interface as DataEngine, which
    is all Trainer depends on.
    """

    def __init__(self, batches):
        self.batches = batches
        self.i = 0
        self.served = 0

    def next_batch(self, device=None):
        batch = self.batches[self.i % len(self.batches)]
        self.i += 1
        self.served += 1
        if device is None:
            return batch
        return {k: v.to(device) for k, v in batch.items()}


def make_batch(seed=0, batch=BATCH, seq=SEQ):
    torch.manual_seed(seed)
    ids = torch.randint(0, VOCAB, (batch, seq + 1))
    return {"input_ids": ids[:, :-1], "labels": ids[:, 1:]}


def make_cfg(**overrides):
    training = {
        "steps": 100,
        "lr": 1e-3,
        "min_lr": 1e-4,
        "warmup_steps": 0,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "grad_accum_steps": 1,
        "amp": False,
        "log_every": 10,
    }
    training.update(overrides)
    return {
        "model": {
            "vocab_size": VOCAB,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "d_ff": 64,
            "max_seq_len": 32,
        },
        "attention": {"impl": "manual"},
        "training": training,
        "device": "cpu",
    }


def make_trainer(batches=None, **cfg_overrides):
    cfg = make_cfg(**cfg_overrides)
    torch.manual_seed(0)
    model = CausalLM.from_config(cfg)
    engine = FixedEngine(batches or [make_batch()])
    return Trainer(model, engine, cfg, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# A single step
# ---------------------------------------------------------------------------


def test_step_reports_the_expected_metrics():
    metrics = make_trainer().step()
    assert set(metrics) == {"step", "loss", "lr", "grad_norm", "tokens", "tokens_per_sec"}
    assert metrics["step"] == 1
    assert metrics["tokens"] == BATCH * SEQ
    assert metrics["tokens_per_sec"] > 0


def test_first_loss_is_near_uniform_entropy():
    loss = make_trainer().step()["loss"]
    assert abs(loss - math.log(VOCAB)) < 0.3


def test_step_actually_updates_the_weights():
    trainer = make_trainer()
    before = trainer.model.embed_tokens.weight.detach().clone()
    trainer.step()
    assert not torch.equal(before, trainer.model.embed_tokens.weight)


def test_gradients_are_cleared_between_steps():
    """A missing zero_grad accumulates forever and silently inflates the step."""
    trainer = make_trainer()
    trainer.step()
    grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
    assert all(g.count_nonzero() == 0 for g in grads) or not grads


# ---------------------------------------------------------------------------
# The exit test
# ---------------------------------------------------------------------------


def test_overfits_a_single_batch():
    """Repeat one batch until the model memorises it. Loss must collapse.

    This is the stage-4 exit criterion: a flat curve here means something
    upstream (mask, label shift, gradient path, schedule) is broken.
    """
    trainer = make_trainer(steps=200, lr=3e-3, warmup_steps=10)
    history = trainer.train(200)

    first, last = history[0]["loss"], history[-1]["loss"]
    assert first > 3.0, f"expected to start near ln(32)={math.log(32):.2f}, got {first:.2f}"
    assert last < 0.2, f"failed to memorise one batch: {first:.2f} -> {last:.2f}"


def test_loss_is_monotonic_enough_to_be_real():
    """Averaged over windows the curve must fall, not just end low by luck."""
    history = make_trainer(steps=120, lr=3e-3, warmup_steps=5).train(120)
    losses = [h["loss"] for h in history]
    early = sum(losses[:20]) / 20
    late = sum(losses[-20:]) / 20
    assert late < early / 2


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------


def test_accumulating_k_micro_batches_matches_one_big_batch():
    """Two micro-batches of 4 must give the same gradient as one batch of 8.

    If the per-micro-batch loss is not divided by grad_accum_steps, the summed
    gradient is K times too large -- which looks like training with a K-times
    larger learning rate and is easy to mistake for a tuning problem.
    """
    a, b = make_batch(1), make_batch(2)
    big = {k: torch.cat([a[k], b[k]], dim=0) for k in a}

    accum = make_trainer([a, b], grad_accum_steps=2, grad_clip=0.0)
    single = make_trainer([big], grad_accum_steps=1, grad_clip=0.0)

    accum.model.load_state_dict(single.model.state_dict())

    for trainer in (accum, single):
        trainer.model.train()
        for _ in range(trainer.grad_accum_steps):
            batch = trainer.engine.next_batch(trainer.device)
            (trainer.loss_on(batch) / trainer.grad_accum_steps).backward()

    for (name, p_accum), (_, p_single) in zip(
        accum.model.named_parameters(), single.model.named_parameters(), strict=True
    ):
        torch.testing.assert_close(
            p_accum.grad, p_single.grad, rtol=1e-4, atol=1e-6, msg=f"gradient differs: {name}"
        )


def test_accumulation_consumes_one_batch_per_micro_step():
    trainer = make_trainer([make_batch(1), make_batch(2), make_batch(3)], grad_accum_steps=3)
    trainer.step()
    assert trainer.engine.served == 3


# ---------------------------------------------------------------------------
# Gradient clipping and schedule
# ---------------------------------------------------------------------------


def test_gradient_clipping_bounds_the_update():
    """With a tiny threshold the post-clip norm must sit at the threshold."""
    trainer = make_trainer(grad_clip=1e-4, lr=1e-2)
    trainer.model.train()
    batch = trainer.engine.next_batch(trainer.device)
    trainer.loss_on(batch).backward()
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1e-4)

    total = torch.norm(
        torch.stack([p.grad.norm() for p in trainer.model.parameters() if p.grad is not None])
    )
    assert total <= 1e-4 * 1.01


def test_reported_grad_norm_is_the_pre_clip_value():
    """Reporting the post-clip norm would pin it at the threshold and hide spikes."""
    metrics = make_trainer(grad_clip=1e-6).step()
    assert metrics["grad_norm"] > 1e-6


def test_learning_rate_follows_the_warmup_then_decay_schedule():
    trainer = make_trainer(steps=50, lr=1e-2, min_lr=1e-3, warmup_steps=10)
    lrs = [trainer.step()["lr"] for _ in range(50)]
    assert lrs[0] == pytest.approx(1e-3)  # first warmup step
    assert lrs[9] == pytest.approx(1e-2)  # peak at end of warmup
    assert lrs[-1] < lrs[9]  # decaying afterwards


def test_amp_is_disabled_on_cpu():
    """fp16 on CPU is emulated and slower than fp32 -- don't pretend otherwise."""
    assert make_trainer(amp=True).amp_dtype is None


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def test_checkpoint_round_trips_the_weights(tmp_path):
    trainer = make_trainer()
    trainer.train(3)
    path = tmp_path / "ckpt.pt"
    trainer.save(str(path))

    fresh = make_trainer()
    assert not torch.equal(fresh.model.embed_tokens.weight, trainer.model.embed_tokens.weight)
    fresh.load(str(path))
    for (name, a), (_, b) in zip(
        fresh.model.named_parameters(), trainer.model.named_parameters(), strict=True
    ):
        torch.testing.assert_close(a, b, msg=f"parameter differs after resume: {name}")


def test_resume_restores_the_step_counter(tmp_path):
    trainer = make_trainer()
    trainer.train(5)
    path = tmp_path / "ckpt.pt"
    trainer.save(str(path))

    fresh = make_trainer()
    fresh.load(str(path))
    assert fresh.step_count == 5


def test_resume_restores_optimizer_moments(tmp_path):
    """Dropping Adam's moments causes a visible loss spike on the next step."""
    trainer = make_trainer()
    trainer.train(5)
    path = tmp_path / "ckpt.pt"
    trainer.save(str(path))

    fresh = make_trainer()
    fresh.load(str(path))

    original = [
        trainer.optimizer.state[p]["exp_avg"]
        for g in trainer.optimizer.param_groups
        for p in g["params"]
        if p in trainer.optimizer.state
    ]
    restored = [
        fresh.optimizer.state[p]["exp_avg"]
        for g in fresh.optimizer.param_groups
        for p in g["params"]
        if p in fresh.optimizer.state
    ]
    assert original and len(original) == len(restored)
    for a, b in zip(original, restored, strict=True):
        torch.testing.assert_close(a, b)


def test_resumed_run_continues_identically(tmp_path):
    """The step after a resume must match the step that would have happened."""
    uninterrupted = make_trainer()
    uninterrupted.train(4)
    path = tmp_path / "ckpt.pt"
    uninterrupted.save(str(path))
    expected = uninterrupted.step()["loss"]

    resumed = make_trainer()
    resumed.load(str(path))
    assert resumed.step()["loss"] == pytest.approx(expected, rel=1e-5)


def test_loading_into_a_different_architecture_is_rejected(tmp_path):
    """A partial load yields a half-random model that still trains."""
    trainer = make_trainer()
    path = tmp_path / "ckpt.pt"
    trainer.save(str(path))

    cfg = make_cfg()
    cfg["model"]["n_layers"] = 4
    mismatched = CausalLM.from_config(cfg)
    with pytest.raises(RuntimeError):
        load_checkpoint(str(path), model=mismatched)


def test_save_checkpoint_creates_missing_directories(tmp_path):
    trainer = make_trainer()
    path = tmp_path / "nested" / "deeper" / "ckpt.pt"
    save_checkpoint(path, model=trainer.model, step=0)
    assert path.exists()
