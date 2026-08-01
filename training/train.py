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
from typing import Any

import torch
import yaml

from dataset import DataEngine
from training.device import autocast_dtype, resolve_device


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the language model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("overrides", nargs="*", help="dotted.key=value overrides")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    device = resolve_device(cfg.get("device"))
    amp_dtype = autocast_dtype(device, cfg["training"].get("amp_dtype", "bfloat16"))

    os.makedirs(cfg["training"]["out_dir"], exist_ok=True)

    print("Loaded config from", args.config)
    print(yaml.safe_dump(cfg, sort_keys=False))
    print(f"device: {device} | amp dtype: {amp_dtype or 'disabled (fp32)'}")

    # Stages 0-2 are implemented; the loop below stays a stub until the model
    # exists, because a "training" run with no model would be theatre.
    engine = DataEngine.from_config(cfg, vocab_size=cfg["model"]["vocab_size"])
    batch = engine.next_batch(device)
    print(
        f"dataset engine OK: input_ids {tuple(batch['input_ids'].shape)}, "
        f"labels {tuple(batch['labels'].shape)} on {batch['input_ids'].device}"
    )

    print(
        "Implemented: tokenizer, dataset, attention.\n"
        "Remaining, in dependency order:\n"
        "  3. model        (RMSNorm + SwiGLU blocks -> CausalLM head)\n"
        "  4. training     (AdamW, cosine schedule, AMP loop)\n"
        "  5. inference    (sampling / decoding)\n"
        "  6. optimization (quantization / kernels)\n"
        "  7. benchmark    (metrics harness)"
    )


if __name__ == "__main__":
    main()
