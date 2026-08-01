"""A single screenshot-ready summary of the engine.

    python -m benchmark.showcase --config configs/tinyshakespeare.yaml \
        --checkpoint checkpoints/tinyshakespeare/best.pt \
        --vocab data/tinyshakespeare/vocab.json

Prints four blocks: the architecture as configured, the exact memory figures, the
measured latency, and live completions from the checkpoint. Everything here is
computed at run time — nothing is a stored string — so a screenshot of the output
is a screenshot of the code actually working.

Deliberately prints the held-out perplexity next to the completions. A generation
sample on its own invites the reader to judge fluency, which at this scale is
flattering and meaningless; the perplexity is the number that says what the model
is really worth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from benchmark.memory import gqa_saving_report, model_memory_report
from model import CausalLM
from tokenizer import BPE
from training.checkpoint import load_checkpoint
from training.device import resolve_device

RULE = "─" * 74

DEFAULT_PROMPTS = (
    "ROMEO:",
    "To be, or not to be",
    "My lord, the queen",
    "What say you to",
)


def header(title: str) -> None:
    print(f"\n{RULE}\n  {title}\n{RULE}")


def show_architecture(cfg: dict, model: CausalLM) -> None:
    m = cfg["model"]
    header("ARCHITECTURE — decoder-only, built from scratch")
    rows = [
        ("parameters", f"{model.num_parameters():,}"),
        ("layers x d_model", f"{m['n_layers']} x {m['d_model']}"),
        ("attention", f"{m['n_heads']} query heads / {m['n_kv_heads']} KV heads (GQA)"),
        ("head dim", str(m["d_model"] // m["n_heads"])),
        ("feed-forward", f"SwiGLU, d_ff {m['d_ff']}"),
        ("normalisation", "RMSNorm, pre-norm"),
        ("positions", f"RoPE, theta {m.get('rope_theta', 10000.0):.0f}"),
        ("context", f"{m['max_seq_len']} tokens"),
        ("vocabulary", f"{m['vocab_size']} (byte-level BPE, no OOV)"),
        ("kernel", f"{cfg.get('attention', {}).get('impl', 'sdpa')} (manual | sdpa | flash)"),
    ]
    for key, value in rows:
        print(f"  {key:<20} {value}")


def show_memory(model: CausalLM, cfg: dict) -> None:
    m = cfg["model"]
    header("MEMORY — exact arithmetic, identical on every machine")
    report = model_memory_report(model)
    print(f"  {'precision':<12}{'size':>12}{'compression':>14}")
    for name in ("fp32", "fp16", "int8", "int4"):
        mib = report["mib"][name]
        ratio = report["compression_vs_fp32"][name]
        print(f"  {name:<12}{mib:>9.2f} MiB{ratio:>13.2f}x")

    gqa = gqa_saving_report(
        n_layers=int(m["n_layers"]),
        n_heads=int(m["n_heads"]),
        n_kv_heads=int(m.get("n_kv_heads", m["n_heads"])),
        head_dim=int(m["d_model"]) // int(m["n_heads"]),
        seq_len=int(m["max_seq_len"]),
    )
    print(
        f"\n  KV cache @ {gqa['seq_len']} tokens:  "
        f"MHA {gqa['mha_mib']:.2f} MiB -> GQA {gqa['gqa_mib']:.2f} MiB "
        f"({gqa['saving_ratio']:.2f}x smaller)"
    )
    print("  int8 is not 4x: the tied embedding stays fp32 and sets the floor.")


def show_latency(model: CausalLM) -> None:
    from benchmark.latency import benchmark_decode, benchmark_prefill, kv_cache_speedup

    header("LATENCY — wall clock, this machine only")
    prefill = benchmark_prefill(model, prompt_len=32, warmup=3, repeats=50)
    decode = benchmark_decode(model, prompt_len=32, max_new_tokens=32, warmup=2, repeats=10)

    for label, report in (("prefill", prefill), ("decode", decode)):
        t = report["timing"]
        print(
            f"  {label:<8} p50 {t['p50_ms']:7.2f} ms   p90 {t['p90_ms']:7.2f}   "
            f"p95 {t['p95_ms']:7.2f}   p99 {t['p99_ms']:7.2f}   "
            f"({t['runs']} runs, resolvable <= p{t['resolvable_percentile']})"
        )
    print(f"\n  prefill throughput  {prefill['tokens_per_sec']:>10,.0f} tok/s")
    print(f"  decode  throughput  {decode['tokens_per_sec']:>10,.0f} tok/s")

    # Reported across several lengths, because a single figure hides the point:
    # the cache removes O(T^2) recompute, so its advantage *grows* with the
    # generation. Quoting only a short run makes a real asymptotic win look
    # like noise.
    print("\n  KV cache — measured, not assumed (TDR-012):")
    print(f"    {'new tokens':>11}{'uncached':>11}{'cached':>10}{'speedup':>10}")
    for length in (32, 128, 256):
        result = kv_cache_speedup(model, prompt_len=16, max_new_tokens=length, repeats=3)
        print(
            f"    {length:>11}{result['uncached_ms']:>8.0f} ms{result['cached_ms']:>7.0f} ms"
            f"{result['speedup']:>9.2f}x"
        )


def measure_perplexity(model: CausalLM, cfg: dict, val_path: str, batches: int) -> float:
    """Evaluate held-out perplexity here and now, rather than accept a number.

    A figure typed in on the command line is indistinguishable from one that was
    invented. Measuring it in the same process that prints it means the whole
    output is generated.
    """
    from dataset import DataEngine
    from training.trainer import Trainer

    eval_cfg = {
        **cfg,
        "dataset": {**cfg["dataset"], "source": "file", "path": val_path, "shuffle": False},
        "training": {**cfg.get("training", {}), "amp": False},
    }
    engine = DataEngine.from_config(eval_cfg, vocab_size=int(cfg["model"]["vocab_size"]))
    trainer = Trainer(model, engine, eval_cfg, device=next(model.parameters()).device)
    return float(trainer.evaluate(engine, batches=batches)["val_perplexity"])


def show_completions(
    model: CausalLM, tokenizer: BPE, prompts: tuple[str, ...], val_ppl: float | None
) -> None:
    from inference import generate

    header("AUTOCOMPLETION — greedy-free sampling from the trained checkpoint")
    if val_ppl is not None:
        print(
            f"  held-out perplexity {val_ppl:.1f} (measured now) — read the samples against this\n"
        )

    for prompt in prompts:
        text = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=48,
            temperature=0.8,
            top_k=40,
            generator=torch.Generator().manual_seed(0),
        )
        continuation = text[len(prompt) :].replace("<|endoftext|>", " ⏎ ").strip()
        print(f"  > {prompt}")
        for line in continuation.splitlines()[:4]:
            if line.strip():
                print(f"      {line.strip()}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a showcase summary")
    parser.add_argument("--config", default="configs/tinyshakespeare.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--vocab", default=None)
    parser.add_argument(
        "--val-path",
        default=None,
        help="held-out .bin; perplexity is measured here rather than passed in",
    )
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    from training.train import load_config

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device"))
    torch.manual_seed(int(cfg.get("seed", 0)))

    model = CausalLM.from_config(cfg)
    step = None
    if args.checkpoint and Path(args.checkpoint).exists():
        payload = load_checkpoint(args.checkpoint, model=model)
        step = payload.get("step")
    model = model.to(device).eval()

    print(f"\n  Language Model Engine — {device}, torch {torch.__version__}")
    if step is not None:
        print(f"  checkpoint: {args.checkpoint} (step {step})")

    show_architecture(cfg, model)
    show_memory(model, cfg)
    if not args.skip_latency:
        show_latency(model)

    val_ppl = None
    if args.val_path and Path(args.val_path).exists():
        val_ppl = measure_perplexity(model, cfg, args.val_path, args.eval_batches)

    if args.vocab and Path(args.vocab).exists():
        tokenizer = BPE(vocab_size=int(cfg["model"]["vocab_size"]))
        tokenizer.load(args.vocab)
        show_completions(model, tokenizer, DEFAULT_PROMPTS, val_ppl)
    print(RULE)


if __name__ == "__main__":
    main()
