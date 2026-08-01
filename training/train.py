"""Training entry point.

Usage:
    python -m training.train --config configs/default.yaml
    python -m training.train --config configs/default.yaml training.lr=1e-4 model.n_layers=12

This is a thin orchestration layer: it loads the config, wires together the
dataset / model / optimizer, and runs the training loop. The heavy lifting is
implemented in the respective modules as they are built out (see docs/TDD.md
and docs/BUILD_ORDER.md for the dependency-aware plan).
"""

import argparse
import os
import random
from pathlib import Path
from typing import Any

import torch
import yaml

from dataset import DataEngine
from model import CausalLM
from training.device import autocast_dtype, resolve_device
from training.trainer import Trainer


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dict(data) if data else {}


def set_deep(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def coerce(raw: str) -> Any:
    """Parse an override value with the same rules that parsed the YAML file.

    Reusing the YAML scalar parser (rather than a hand-rolled int/float/bool
    ladder) means ``lr=3e-4`` is a float, ``amp=true`` is a bool, and
    ``out_dir=checkpoints`` is a string, all exactly as they would be if written
    into the config file -- so an override can never produce a differently typed
    value than the key it replaces.
    """
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw

    # PyYAML implements YAML 1.1, whose float grammar demands a decimal point
    # in scientific notation: "3.0e-4" parses, "3e-4" comes back as a string.
    # Learning rates are almost always written the short way, so rescue it --
    # otherwise `training.lr=3e-4` sets the LR to the *string* "3e-4".
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} is not of the form dotted.key=value")
        set_deep(cfg, key, coerce(raw))
    return cfg


def build_val_engine(cfg: dict[str, Any]) -> DataEngine | None:
    """A second engine over ``dataset.val_path``, or None if none is configured.

    Held-out evaluation is opt-in rather than automatic: the synthetic source has
    no meaningful validation split (it is uniform noise by construction), and
    silently evaluating against training data would report a number that looks
    like generalisation and is not.
    """
    val_path = cfg.get("dataset", {}).get("val_path")
    if not val_path:
        return None
    val_cfg = {
        **cfg,
        "dataset": {**cfg["dataset"], "source": "file", "path": val_path, "shuffle": False},
    }
    return DataEngine.from_config(val_cfg, vocab_size=cfg["model"]["vocab_size"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the language model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="echo the full config")
    parser.add_argument("overrides", nargs="*", help="dotted.key=value overrides")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    device = resolve_device(cfg.get("device"))
    amp_dtype = autocast_dtype(device, cfg["training"].get("amp_dtype", "bfloat16"))

    out_dir = Path(cfg["training"]["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    if args.verbose:
        print("Loaded config from", args.config)
        print(yaml.safe_dump(cfg, sort_keys=False))

    model = CausalLM.from_config(cfg)
    engine = DataEngine.from_config(cfg, vocab_size=cfg["model"]["vocab_size"])
    trainer = Trainer(model, engine, cfg, device=device)
    val_engine = build_val_engine(cfg)

    steps = int(cfg["training"]["steps"])
    log_every = max(1, int(cfg["training"].get("log_every", 50)))
    ckpt_every = int(cfg["training"].get("ckpt_every", 0))
    eval_every = int(cfg["training"].get("eval_every", 0))
    eval_batches = int(cfg["training"].get("eval_batches", 20))

    print(
        f"device {device} | amp {amp_dtype or 'off (fp32)'} | "
        f"{model.num_parameters() / 1e6:.2f}M params | {steps} steps"
        + ("" if val_engine else " | no validation split configured")
    )

    last_evaluated = -1

    def report_validation(step: int) -> None:
        nonlocal last_evaluated
        # Guard against the final report duplicating an eval_every hit that just
        # landed on the same step.
        if val_engine is None or step == last_evaluated:
            return
        last_evaluated = step
        val = trainer.evaluate(val_engine, batches=eval_batches)
        print(
            f"step {step:>6}/{steps}  val_loss {val['val_loss']:.4f}  "
            f"val_ppl {val['val_perplexity']:.2f}"
        )

    for _ in range(steps):
        metrics = trainer.step()
        n = trainer.step_count
        if n % log_every == 0 or n == steps:
            print(
                f"step {n:>6}/{steps}  loss {metrics['loss']:.4f}  "
                f"lr {metrics['lr']:.2e}  grad_norm {metrics['grad_norm']:.3f}  "
                f"{metrics['tokens_per_sec']:,.0f} tok/s"
            )
        if eval_every and n % eval_every == 0:
            report_validation(n)
        if ckpt_every and n % ckpt_every == 0:
            trainer.save(str(out_dir / f"step_{n}.pt"))

    report_validation(trainer.step_count)
    final = out_dir / "final.pt"
    trainer.save(str(final))
    print(f"saved {final}")


if __name__ == "__main__":
    main()
