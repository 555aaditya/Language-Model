"""Quantization tests (TDD - written before implementation).

Exit criterion for build stage 6 (docs/BUILD_ORDER.md): perplexity within
tolerance of the fp32 reference.

Quantization is unusually easy to fake. A `quantize()` that returns the model
untouched passes every shape and perplexity check trivially, so the tests here
assert the *memory actually shrank* by the expected factor -- measured in bytes
off the real tensors, not inferred from the dtype name.
"""

import math

import pytest
import torch
from torch import nn

from model import CausalLM
from optimization import (
    QuantizedLinear,
    dequantize_int4,
    dequantize_int8,
    model_nbytes,
    quantize,
    quantize_int4,
    quantize_int8,
)

VOCAB = 64


def make_model(**overrides):
    cfg = {
        "model": {
            "vocab_size": VOCAB,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "d_ff": 64,
            "max_seq_len": 32,
            **overrides,
        },
        "attention": {"impl": "manual"},
    }
    torch.manual_seed(0)
    return CausalLM.from_config(cfg).eval()


# ---------------------------------------------------------------------------
# int8 tensor round trip
# ---------------------------------------------------------------------------


def test_int8_round_trip_stays_close_to_the_original():
    torch.manual_seed(0)
    w = torch.randn(16, 32)
    restored = dequantize_int8(*quantize_int8(w))
    # Symmetric per-channel int8 has 127 levels per side, so the worst-case
    # error is half a step: max|w_row| / 254.
    tolerance = w.abs().amax(dim=1, keepdim=True) / 254
    assert (restored - w).abs().le(tolerance * 1.01).all()


def test_int8_scales_are_per_output_channel_not_per_tensor():
    """One row 1000x larger than the rest must not destroy the small rows.

    A single per-tensor scale would quantise the small rows to nearly all
    zeros; per-channel scales keep each row's own dynamic range.
    """
    w = torch.randn(4, 32)
    w[0] *= 1000.0
    q, scales = quantize_int8(w)
    assert scales.shape == (4, 1)

    restored = dequantize_int8(q, scales)
    small_rows_error = (restored[1:] - w[1:]).abs().max()
    assert small_rows_error < w[1:].abs().max() * 0.02


def test_int8_values_stay_in_range():
    q, _ = quantize_int8(torch.randn(8, 16) * 50)
    assert q.dtype == torch.int8
    assert q.min() >= -127 and q.max() <= 127


def test_int8_handles_an_all_zero_row_without_dividing_by_zero():
    w = torch.randn(4, 8)
    w[2] = 0.0
    restored = dequantize_int8(*quantize_int8(w))
    assert torch.isfinite(restored).all()
    torch.testing.assert_close(restored[2], torch.zeros(8))


# ---------------------------------------------------------------------------
# int4 packing
# ---------------------------------------------------------------------------


def test_int4_packs_two_values_per_byte():
    """The memory claim is the point -- assert the byte count, not the dtype."""
    w = torch.randn(8, 32)
    packed, scales, in_features = quantize_int4(w)
    assert in_features == 32
    assert packed.dtype == torch.uint8
    assert packed.shape == (8, 16)
    assert packed.numel() * packed.element_size() == w.numel() // 2


def test_int4_round_trip_is_coarser_than_int8_but_bounded():
    torch.manual_seed(0)
    w = torch.randn(16, 32)
    err8 = (dequantize_int8(*quantize_int8(w)) - w).abs().mean()
    err4 = (dequantize_int4(*quantize_int4(w)) - w).abs().mean()
    assert err8 < err4, "int4 should be coarser than int8"
    tolerance = w.abs().amax(dim=1, keepdim=True) / 14
    assert (dequantize_int4(*quantize_int4(w)) - w).abs().le(tolerance * 1.01).all()


def test_int4_pads_an_odd_number_of_columns():
    """Two values share a byte, so an odd width needs a pad column."""
    w = torch.randn(4, 7)
    packed, scales, in_features = quantize_int4(w)
    assert in_features == 7
    assert packed.shape == (4, 4)
    assert dequantize_int4(packed, scales, in_features).shape == (4, 7)


# ---------------------------------------------------------------------------
# QuantizedLinear
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_linear_tracks_the_float_layer(bits):
    torch.manual_seed(0)
    linear = nn.Linear(32, 16, bias=False)
    q = QuantizedLinear.from_linear(linear, bits=bits)

    x = torch.randn(4, 32)
    expected, actual = linear(x), q(x)
    assert actual.shape == expected.shape
    relative = (actual - expected).norm() / expected.norm()
    assert relative < (0.02 if bits == 8 else 0.2), f"relative error {relative:.4f}"


def test_quantized_linear_preserves_the_bias_in_full_precision():
    """Bias is one value per output -- quantising it saves nothing and costs accuracy."""
    linear = nn.Linear(8, 4, bias=True)
    q = QuantizedLinear.from_linear(linear, bits=8)
    torch.testing.assert_close(q.bias, linear.bias)


def test_quantized_linear_stores_no_float_weight():
    q = QuantizedLinear.from_linear(nn.Linear(32, 16, bias=False), bits=8)
    assert not any(p.dtype.is_floating_point and p.numel() > 16 for p in q.buffers())
    assert not list(q.parameters()), "quantised weights must not stay trainable"


# ---------------------------------------------------------------------------
# Whole-model quantization
# ---------------------------------------------------------------------------


def test_quantize_shrinks_the_model():
    model = make_model()
    before = model_nbytes(model)
    after = model_nbytes(quantize(make_model(), bits=8))
    assert after < before, f"{before} -> {after} bytes: no saving at all"


def test_int4_is_smaller_than_int8():
    eight = model_nbytes(quantize(make_model(), bits=8))
    four = model_nbytes(quantize(make_model(), bits=4))
    assert four < eight


def test_quantize_replaces_the_linear_layers():
    model = quantize(make_model(), bits=8)
    quantised = [m for m in model.modules() if isinstance(m, QuantizedLinear)]
    assert quantised, "no layer was actually replaced"
    # every attention/FFN projection in a 2-layer model: 4 attn + 3 ffn
    assert len(quantised) == 2 * 7


def test_the_tied_lm_head_is_left_alone_by_default():
    """Quantising a tied head silently breaks the tie with the embedding.

    The head would become an int8 copy while the embedding stayed fp32, so the
    two matrices -- which are supposed to be one tensor -- would diverge.
    """
    model = quantize(make_model(tie_weights=True), bits=8)
    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.weight is model.embed_tokens.weight


def test_quantized_model_still_produces_valid_logits():
    model = quantize(make_model(), bits=8)
    logits = model(torch.randint(0, VOCAB, (2, 8)))
    assert logits.shape == (2, 8, VOCAB)
    assert torch.isfinite(logits).all()


def test_quantized_model_still_generates_with_a_cache():
    from inference import generate_ids

    model = quantize(make_model(), bits=8)
    out = generate_ids(model, torch.tensor([[1, 2, 3]]), max_new_tokens=5, temperature=0.0)
    assert len(out) == 5


def test_fp16_halves_the_model():
    model = make_model()
    before = model_nbytes(model)
    after = model_nbytes(quantize(make_model(), scheme="fp16"))
    assert after == pytest.approx(before / 2, rel=0.01)


def test_unknown_scheme_and_bit_width_are_rejected():
    with pytest.raises(ValueError, match="scheme"):
        quantize(make_model(), scheme="int3_magic")
    with pytest.raises(ValueError, match="bits"):
        quantize(make_model(), bits=7)


# ---------------------------------------------------------------------------
# The exit test: perplexity within tolerance
# ---------------------------------------------------------------------------


def _perplexity(model, ids):
    with torch.no_grad():
        logits = model(ids[:, :-1])
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
    return math.exp(loss.item())


@pytest.fixture(scope="module")
def trained():
    """A briefly-trained model, so quantization error has real structure to damage."""
    from training.trainer import Trainer

    cfg = {
        "model": {
            "vocab_size": VOCAB,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "d_ff": 128,
            "max_seq_len": 64,
        },
        "attention": {"impl": "manual"},
        "training": {
            "steps": 150,
            "lr": 3e-3,
            "min_lr": 3e-4,
            "warmup_steps": 10,
            "grad_clip": 1.0,
            "amp": False,
            "weight_decay": 0.0,
        },
    }
    torch.manual_seed(0)
    # A learnable pattern: token t+1 is always (t * 7 + 3) mod VOCAB.
    seq = torch.tensor([[(i * 7 + 3) % VOCAB for i in range(33)]]).repeat(8, 1)
    batch = {"input_ids": seq[:, :-1], "labels": seq[:, 1:]}

    class OneBatch:
        def next_batch(self, device=None):
            return batch

    model = CausalLM.from_config(cfg)
    Trainer(model, OneBatch(), cfg, device=torch.device("cpu")).train(150)
    return model.eval(), seq


def test_int8_perplexity_stays_within_tolerance(trained):
    """The stage 6 exit criterion."""
    model, seq = trained
    reference = _perplexity(model, seq)
    quantised = _perplexity(quantize(model, bits=8), seq)
    assert quantised < reference * 1.10, f"ppl {reference:.3f} -> {quantised:.3f} (>10% worse)"


def test_int4_degrades_more_than_int8(trained):
    """A sanity check on the tradeoff -- fewer bits must cost something."""
    model, seq = trained
    eight = _perplexity(quantize(model, bits=8), seq)
    four = _perplexity(quantize(model, bits=4), seq)
    assert eight <= four
