"""Assemble and emit a benchmark report (TDD §9).

    python -m benchmark.report --config configs/default.yaml --out report.json

The report separates two kinds of number, because they deserve very different
levels of trust:

- ``deterministic`` — memory footprints, compression ratios, cache sizes. Exact
  arithmetic over shapes, identical on every machine. These are the ones a test
  can assert against a recorded baseline.
- ``measured`` — latency and throughput. Machine-, thermal- and load-dependent.
  Reproducible only in shape, never in value, so nothing asserts them.

Keeping them in one flat blob would invite exactly the mistake TDR-012 exists to
prevent: quoting a timing from one machine as though it were a property of the
code.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch

from benchmark.latency import benchmark_decode, benchmark_prefill, kv_cache_speedup
from benchmark.memory import (
    activation_memory_estimate,
    gqa_saving_report,
    model_memory_report,
)
from model import CausalLM
from training.device import resolve_device


def environment() -> dict[str, Any]:
    """Everything needed to interpret a measured number later."""
    device = resolve_device(None)
    return {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }


def deterministic_report(cfg: dict[str, Any]) -> dict[str, Any]:
    """Exact, machine-independent metrics. Safe to assert against a baseline."""
    m = cfg["model"]
    torch.manual_seed(0)
    model = CausalLM.from_config(cfg)
    head_dim = int(m["d_model"]) // int(m["n_heads"])
    seq_len = int(m.get("max_seq_len", 1024))

    return {
        "config": {k: m[k] for k in sorted(m)},
        "model_memory": model_memory_report(model),
        "gqa_kv_cache": gqa_saving_report(
            n_layers=int(m["n_layers"]),
            n_heads=int(m["n_heads"]),
            n_kv_heads=int(m.get("n_kv_heads", m["n_heads"])),
            head_dim=head_dim,
            seq_len=seq_len,
        ),
        "attention_scores": activation_memory_estimate(
            batch_size=1, n_heads=int(m["n_heads"]), seq_len=seq_len
        ),
    }


def measured_report(
    cfg: dict[str, Any], *, prompt_len: int = 32, max_new_tokens: int = 32
) -> dict[str, Any]:
    """Wall-clock metrics. Interpret only alongside ``environment()``.

    Sample counts are set by what each phase costs, not by a single default.
    A percentile resolves tail probabilities no finer than ``1/runs``, so the
    5-repeat default would report a p99 that is just the maximum under another
    name. Prefill is ~3 ms, so 100 runs is nearly free; decode is ~80 ms, so 20
    runs is the affordable compromise — and ``resolvable_percentile`` in the
    output states the ceiling either way.
    """
    torch.manual_seed(0)
    model = CausalLM.from_config(cfg).eval().to(resolve_device(cfg.get("device")))
    return {
        "prefill": benchmark_prefill(model, prompt_len=prompt_len, warmup=5, repeats=100),
        "decode_cached": benchmark_decode(
            model,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            warmup=2,
            repeats=20,
        ),
        "kv_cache_speedup": kv_cache_speedup(
            model, prompt_len=prompt_len, max_new_tokens=max_new_tokens, repeats=10
        ),
    }


def build_report(cfg: dict[str, Any], *, measured: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {
        "environment": environment(),
        "deterministic": deterministic_report(cfg),
    }
    if measured:
        report["measured"] = measured_report(cfg)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark suite")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default=None, help="write JSON here")
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="skip wall-clock measurement (exact metrics only)",
    )
    args = parser.parse_args()

    from training.train import load_config

    cfg = load_config(args.config)
    report = build_report(cfg, measured=not args.deterministic_only)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
