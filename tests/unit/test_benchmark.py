"""Benchmark harness tests (TDD - written before implementation).

Exit criterion for build stage 7 (docs/BUILD_ORDER.md): reproduces a recorded
baseline.

Only the *deterministic* half of the report can honour that literally. Memory
figures are exact arithmetic over tensor shapes, so they are asserted against a
checked-in baseline byte for byte. Latency is machine-, thermal- and
load-dependent; asserting a wall-clock number would produce a test that fails on
a busy laptop and passes on an idle one, which is worse than no test. So timings
are asserted only for *shape* -- present, finite, ordered percentiles.
"""

import json
from pathlib import Path

import pytest
import torch

from benchmark import (
    activation_memory_estimate,
    benchmark_decode,
    benchmark_prefill,
    build_report,
    deterministic_report,
    environment,
    gqa_saving_report,
    kv_cache_memory,
    kv_cache_speedup,
    model_memory_report,
    synchronize,
    time_it,
)
from benchmark.harness import Timing
from model import CausalLM

BASELINE = Path(__file__).parent.parent / "baselines" / "nano.json"

CFG = {
    "model": {
        "vocab_size": 64,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "d_ff": 64,
        "max_seq_len": 32,
        "rope_theta": 10000.0,
        "dropout": 0.0,
        "tie_weights": True,
    },
    "attention": {"impl": "manual"},
    "device": "cpu",
}


def make_model():
    torch.manual_seed(0)
    return CausalLM.from_config(CFG).eval()


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------


def test_synchronize_is_safe_on_every_device():
    for device in ("cpu", None):
        synchronize(device)


def test_time_it_runs_exactly_the_requested_repeats():
    calls = []
    time_it(lambda: calls.append(1), warmup=3, repeats=7, device="cpu")
    assert len(calls) == 10  # 3 warmup + 7 timed


def test_warmup_calls_are_not_timed():
    """Warmup exists to keep first-call costs out of the average."""
    timing = time_it(lambda: None, warmup=2, repeats=5, device="cpu")
    assert timing.runs == 5


def test_percentiles_are_ordered():
    timing = time_it(lambda: sum(range(500)), warmup=1, repeats=20, device="cpu")
    assert timing.min_ms <= timing.median_ms <= timing.p90_ms <= timing.p99_ms
    assert timing.p99_ms <= timing.max_ms
    assert timing.mean_ms > 0


def test_timing_reports_median_alongside_mean():
    """Step times are right-skewed; a mean alone misrepresents steady state."""
    timing = Timing.from_samples([0.001] * 9 + [1.0])
    assert timing.median_ms == pytest.approx(1.0)
    assert timing.mean_ms > timing.median_ms * 50


def test_zero_repeats_is_rejected():
    with pytest.raises(ValueError, match="repeats"):
        time_it(lambda: None, repeats=0)


# ---------------------------------------------------------------------------
# Memory arithmetic
# ---------------------------------------------------------------------------


def test_kv_cache_memory_matches_the_closed_form():
    # 2 (K and V) x layers x batch x heads x T x dim x bytes
    assert (
        kv_cache_memory(n_layers=4, n_kv_heads=2, head_dim=64, seq_len=128, bytes_per_element=4)
        == 2 * 4 * 1 * 2 * 128 * 64 * 4
    )


def test_gqa_saving_is_exactly_the_head_ratio():
    report = gqa_saving_report(n_layers=8, n_heads=8, n_kv_heads=2, head_dim=64, seq_len=512)
    assert report["saving_ratio"] == pytest.approx(4.0)
    assert report["gqa_bytes"] * 4 == report["mha_bytes"]


def test_attention_score_memory_is_quadratic_in_sequence_length():
    small = activation_memory_estimate(batch_size=1, n_heads=8, seq_len=128)
    large = activation_memory_estimate(batch_size=1, n_heads=8, seq_len=256)
    assert large["standard_score_bytes"] == small["standard_score_bytes"] * 4


def test_model_memory_report_orders_the_precisions():
    report = model_memory_report(make_model())
    b = report["bytes"]
    assert b["fp32"] > b["fp16"] > b["int8"] > b["int4"]
    assert report["compression_vs_fp32"]["fp16"] == pytest.approx(2.0, rel=0.01)


def test_memory_numbers_are_identical_across_runs():
    """Exact arithmetic -- two runs must agree bit for bit, or it is measured."""
    assert model_memory_report(make_model()) == model_memory_report(make_model())


# ---------------------------------------------------------------------------
# The exit test: reproduce a recorded baseline
# ---------------------------------------------------------------------------


def test_deterministic_report_reproduces_the_baseline():
    """Stage 7 exit criterion, over the half of the report that can be exact.

    Regenerate deliberately with:
        python -m benchmark.report --config <cfg> --deterministic-only
    A diff here means the architecture's memory profile changed -- which is
    either the point of your change, or a bug.
    """
    assert BASELINE.exists(), f"missing baseline: {BASELINE}"
    recorded = json.loads(BASELINE.read_text())
    assert deterministic_report(CFG) == recorded


def test_the_baseline_holds_no_wall_clock_numbers():
    """A timing in a checked-in baseline would fail on any other machine."""
    text = BASELINE.read_text().lower()
    for banned in ("_ms", "tokens_per_sec", "timing", "elapsed"):
        assert banned not in text, f"baseline contains a measured field: {banned}"


# ---------------------------------------------------------------------------
# Measured metrics -- shape only
# ---------------------------------------------------------------------------


def test_prefill_reports_positive_finite_throughput():
    report = benchmark_prefill(make_model(), prompt_len=16, repeats=3, warmup=1)
    assert report["tokens_per_sec"] > 0
    assert report["timing"]["median_ms"] > 0


def test_decode_reports_per_token_cost():
    report = benchmark_decode(make_model(), prompt_len=8, max_new_tokens=4, repeats=2)
    assert report["per_token_ms"] > 0
    assert report["timing"]["median_ms"] >= report["per_token_ms"]


def test_kv_cache_speedup_reports_the_direction_it_measured():
    """TDR-012 wants the number reported, not assumed -- including when it loses."""
    report = kv_cache_speedup(make_model(), prompt_len=8, max_new_tokens=8, repeats=2)
    assert report["speedup"] > 0
    assert isinstance(report["cache_wins"], bool)
    assert report["cache_wins"] == (report["speedup"] > 1.0)


def test_environment_records_what_is_needed_to_read_a_timing():
    env = environment()
    for key in ("torch", "python", "platform", "machine", "device"):
        assert env[key], f"environment is missing {key}"


def test_full_report_separates_exact_from_measured():
    report = build_report(CFG, measured=False)
    assert set(report) == {"environment", "deterministic"}
    assert "measured" not in report
