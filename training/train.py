"""Training entry point.

Usage:
    python -m training.train --config configs/default.yaml
    python -m training.train --config configs/default.yaml training.lr=1e-4 model.n_layers=12

This is a thin orchestration layer: it loads the config, wires together the
dataset / model / optimizer, and runs the training loop. The heavy lifting is
implemented in the respective modules as they are built out (see docs/TDD.md
and the vault's implementation/build_order.md for the dependency-aware plan).
"""

import argparse
import os
import random

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dict(data) if data else {}


def set_deep(cfg: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides:
        key, _, raw = item.partition("=")
        # Best-effort type coercion (int/float/bool/str).
        for cast in (int, float):
            try:
                value = cast(raw)
                break
            except ValueError:
                continue
        else:
            value = (
                raw.lower() in ("true", "1", "yes")
                if raw.lower() in ("true", "false", "1", "0", "yes", "no")
                else raw
            )
        set_deep(cfg, key, value)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the language model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("overrides", nargs="*", help="dotted.key=value overrides")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # torch optional until model/training are implemented
        torch = None

    os.makedirs(cfg["training"]["out_dir"], exist_ok=True)

    print("Loaded config from", args.config)
    print(yaml.safe_dump(cfg, sort_keys=False))
    print(
        "Scaffolding in place. Modules to implement next, in dependency order:\n"
        "  1. dataset      (DataLoaders feeding batches of token ids)\n"
        "  2. attention    (causal GQA + RoPE + KV cache)\n"
        "  3. model        (transformer blocks -> full LM head)\n"
        "  4. training     (optimizer, scheduler, AMP loop)\n"
        "  5. inference    (sampling / decoding)\n"
        "  6. optimization (quantization / kernels)\n"
        "  7. benchmark    (metrics harness)"
    )


if __name__ == "__main__":
    main()
