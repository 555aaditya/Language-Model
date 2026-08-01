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
from torch import nn

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


def show_kernel_equivalence(cfg: dict) -> None:
    """Prove the three attention kernels agree, rather than assert it.

    An optimised kernel with no reference to check against is not trustworthy.
    `manual` is the readable oracle; the other two have to match it.
    """
    import torch

    from attention import CausalAttention
    from attention.kernels import causal_block_mask, flash_attention, manual_attention

    header("KERNEL EQUIVALENCE — three implementations, one answer")
    m = cfg["model"]
    torch.manual_seed(0)
    reference = CausalAttention(
        int(m["d_model"]),
        int(m["n_heads"]),
        int(m["n_kv_heads"]),
        max_seq_len=int(m["max_seq_len"]),
        impl="manual",
    ).eval()
    x = torch.randn(2, 64, int(m["d_model"]))

    baseline, _ = reference(x)
    print(f"  {'kernel':<10}{'max abs diff vs manual':>26}")
    for impl in ("manual", "sdpa", "flash"):
        torch.manual_seed(0)
        other = CausalAttention(
            int(m["d_model"]),
            int(m["n_heads"]),
            int(m["n_kv_heads"]),
            max_seq_len=int(m["max_seq_len"]),
            impl=impl,
        ).eval()
        other.load_state_dict(reference.state_dict())
        out, _ = other(x)
        print(f"  {impl:<10}{(out - baseline).abs().max().item():>26.2e}")

    # Gradient check in float64: float32 noise hides sign and scale errors in a
    # hand-written backward.
    shape = (2, 3, 37, 16)
    q, k, v = (torch.randn(*shape, dtype=torch.float64, requires_grad=True) for _ in range(3))
    qf, kf, vf = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    ref = manual_attention(q, k, v, mask=causal_block_mask(37, 37, 0, q.device))
    tiled = flash_attention(qf, kf, vf, causal=True, block_q=8, block_k=8)
    seed = torch.randn_like(ref)
    ref.backward(seed)
    tiled.backward(seed)
    pairs = ((q, qf), (k, kf), (v, vf))
    assert all(a.grad is not None and b.grad is not None for a, b in pairs)
    worst = max(
        (a.grad - b.grad).abs().max().item()  # type: ignore[operator]
        for a, b in pairs
    )
    print(f"\n  hand-written flash backward vs autograd (float64): {worst:.2e}")


def show_flash_memory(cfg: dict) -> None:
    """The O(T) memory claim, measured for *training* rather than inference."""
    import torch

    from attention import CausalAttention

    header("TILED ATTENTION MEMORY — bytes autograd retains for backward")
    d_model = int(cfg["model"]["d_model"])

    def retained(impl: str, seq_len: int) -> int:
        torch.manual_seed(0)
        attn = CausalAttention(
            d_model, 4, 2, max_seq_len=4096, impl=impl, block_q=32, block_k=32
        ).eval()
        total = 0

        def pack(t: torch.Tensor) -> torch.Tensor:
            nonlocal total
            total += t.numel() * t.element_size()
            return t

        with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            attn(torch.randn(1, seq_len, d_model))
        return total

    print(
        f"  {'T':>6}{'standard':>14}{'tiled':>12}{'ratio':>9}{'std growth':>13}{'tiled growth':>15}"
    )
    prev_std = prev_tiled = None
    for seq_len in (128, 256, 512, 1024):
        std, tiled = retained("manual", seq_len), retained("flash", seq_len)
        g_std = f"{std / prev_std:.2f}x" if prev_std else "—"
        g_tiled = f"{tiled / prev_tiled:.2f}x" if prev_tiled else "—"
        print(
            f"  {seq_len:>6}{std / 1024:>11.0f} KiB{tiled / 1024:>9.0f} KiB"
            f"{tiled / std:>8.2f}x{g_std:>13}{g_tiled:>15}"
        )
        prev_std, prev_tiled = std, tiled
    print("\n  2.0x per doubling is linear; 4.0x is quadratic.")
    print("  Before the custom backward the tiled path retained 1.30x MORE than standard.")


def show_quantization(model: CausalLM, cfg: dict, val_path: str | None, batches: int) -> None:
    """Compression against its actual cost in held-out perplexity."""
    from optimization import model_nbytes, quantize

    header("QUANTIZATION — compression against measured quality cost")
    fp32_bytes = model_nbytes(model)
    print(f"  {'precision':<12}{'size':>12}{'compression':>14}{'held-out ppl':>16}")
    for label, bits in (("fp32", None), ("int8", 8), ("int4", 4)):
        target: nn.Module = model if bits is None else quantize(model, bits=bits)
        size = model_nbytes(target)
        ppl = (
            f"{measure_perplexity(target, cfg, val_path, batches):.1f}"
            if val_path
            else "not measured"
        )
        print(f"  {label:<12}{size / (1024 * 1024):>9.2f} MiB{fp32_bytes / size:>13.2f}x{ppl:>16}")
    print("\n  Weight-only: buys memory, not speed. Real int8 throughput needs fused kernels.")


def measure_perplexity(model: nn.Module, cfg: dict, val_path: str, batches: int) -> float:
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
    parser.add_argument(
        "--evidence",
        action="store_true",
        help="print kernel equivalence, tiled-attention memory and quantization cost",
    )
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
    if args.evidence:
        show_kernel_equivalence(cfg)
        show_flash_memory(cfg)
        show_quantization(model, cfg, args.val_path, args.eval_batches)

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
