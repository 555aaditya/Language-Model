"""Model unit tests (TDD - written before implementation).

Exit criteria for build stage 3 (docs/BUILD_ORDER.md): forward shape and
parameter count.

The parameter-count test is not bookkeeping. It is the only cheap way to catch
a silently-wrong architecture: a SwiGLU built with two matrices instead of
three, weight tying that did not actually tie, or GQA projections sized at full
width all produce correct shapes and train fine -- they just are not the model
the config describes.
"""

import math

import pytest
import torch
from torch import nn

from attention import KVCache
from model import RMSNorm, SwiGLU, TransformerBlock
from model.causal_lm import CausalLM

VOCAB, D_MODEL, N_LAYERS, N_HEADS, N_KV_HEADS, D_FF = 64, 32, 2, 4, 2, 64
BATCH, SEQ = 2, 6


def make_model(**overrides):
    cfg = dict(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        d_ff=D_FF,
        max_seq_len=64,
        dropout=0.0,
        impl="manual",
    )
    cfg.update(overrides)
    torch.manual_seed(0)
    return CausalLM(**cfg).eval()


@pytest.fixture
def ids():
    torch.manual_seed(0)
    return torch.randint(0, VOCAB, (BATCH, SEQ))


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


def test_rmsnorm_scales_to_unit_rms():
    norm = RMSNorm(8)
    x = torch.randn(4, 8) * 17.0  # deliberately far from unit scale
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones(4), rtol=1e-3, atol=1e-3)


def test_rmsnorm_does_not_centre_the_mean():
    """The defining difference from LayerNorm: no mean subtraction.

    A constant-offset input keeps its offset under RMSNorm; LayerNorm would
    annihilate it. If this passes with a LayerNorm underneath, it is wrong.
    """
    norm = RMSNorm(8)
    out = norm(torch.full((1, 8), 3.0))
    assert out.mean() > 0.9, "input mean was centred away -- this is LayerNorm, not RMSNorm"


def test_rmsnorm_weight_starts_as_identity_gain():
    norm = RMSNorm(8)
    assert torch.equal(norm.weight, torch.ones(8))
    assert not hasattr(norm, "bias") or norm.bias is None


def test_rmsnorm_applies_its_learned_gain():
    norm = RMSNorm(4)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.randn(1, 4)
    plain = RMSNorm(4)(x)
    torch.testing.assert_close(norm(x), plain * torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_rmsnorm_survives_half_precision_input():
    """Normalisation must accumulate in fp32 or large activations overflow.

    Squaring a half-precision activation of ~200 overflows to inf before the
    mean is ever taken (fp16 tops out at 65504), so a naive implementation
    returns nan here. The result promotes back to fp32 because the gain is an
    fp32 parameter -- that matches the reference Llama implementation, and
    under autocast the next Linear casts it straight back down.
    """
    norm = RMSNorm(8)
    out = norm((torch.randn(4, 8) * 200).half())
    assert torch.isfinite(out).all(), "fp16 squares overflowed -- reduce in fp32"
    rms = out.float().pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones(4), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------


def test_swiglu_output_shape():
    assert SwiGLU(D_MODEL, D_FF)(torch.randn(BATCH, SEQ, D_MODEL)).shape == (
        BATCH,
        SEQ,
        D_MODEL,
    )


def test_swiglu_has_three_projections_not_two():
    """The whole point of a gated FFN -- two matrices means it is a plain MLP."""
    ffn = SwiGLU(D_MODEL, D_FF)
    linears = [m for m in ffn.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 3
    assert ffn.gate_proj.out_features == ffn.up_proj.out_features == D_FF
    assert ffn.down_proj.in_features == D_FF


def test_swiglu_parameter_count_is_three_matrices():
    ffn = SwiGLU(D_MODEL, D_FF, bias=False)
    assert sum(p.numel() for p in ffn.parameters()) == 3 * D_MODEL * D_FF


def test_swiglu_gate_can_suppress_the_signal():
    """A strongly negative gate drives silu -> 0, so the block output vanishes.

    This distinguishes a real multiplicative gate from a summed one.
    """
    ffn = SwiGLU(4, 8, bias=False)
    with torch.no_grad():
        ffn.gate_proj.weight.fill_(-100.0)
        ffn.up_proj.weight.fill_(1.0)
        ffn.down_proj.weight.fill_(1.0)
    out = ffn(torch.ones(1, 1, 4))
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------


def test_block_preserves_shape():
    block = TransformerBlock(
        d_model=D_MODEL, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, d_ff=D_FF, max_seq_len=64
    ).eval()
    out, cache = block(torch.randn(BATCH, SEQ, D_MODEL))
    assert out.shape == (BATCH, SEQ, D_MODEL)
    assert cache is None


def test_block_is_pre_norm_with_live_residuals():
    """With both sublayers zeroed the block must be the identity.

    That is only true for pre-norm `x + f(norm(x))`. A post-norm block would
    return `norm(x)` instead, and a block missing its residual would return 0.
    """
    block = TransformerBlock(
        d_model=D_MODEL, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, d_ff=D_FF, max_seq_len=64
    ).eval()
    with torch.no_grad():
        block.attn.o_proj.weight.zero_()
        block.ffn.down_proj.weight.zero_()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out, _ = block(x)
    torch.testing.assert_close(out, x, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# CausalLM -- shapes and parameter count
# ---------------------------------------------------------------------------


def test_forward_returns_logits_over_the_vocabulary(ids):
    logits = make_model()(ids)
    assert logits.shape == (BATCH, SEQ, VOCAB)
    assert torch.isfinite(logits).all()


def test_forward_returns_logits_only_not_a_loss(ids):
    """Loss belongs to the trainer; the model stays pure for inference."""
    assert isinstance(make_model()(ids), torch.Tensor)


def test_parameter_count_matches_the_architecture():
    model = make_model(tie_weights=True)
    head_dim = D_MODEL // N_HEADS
    attn = (
        D_MODEL * N_HEADS * head_dim  # q
        + 2 * D_MODEL * N_KV_HEADS * head_dim  # k, v (narrow under GQA)
        + N_HEADS * head_dim * D_MODEL  # o
    )
    ffn = 3 * D_MODEL * D_FF  # gate, up, down
    per_layer = attn + ffn + 2 * D_MODEL  # + two RMSNorm gains
    expected = VOCAB * D_MODEL + N_LAYERS * per_layer + D_MODEL  # + final norm
    assert sum(p.numel() for p in model.parameters()) == expected


def test_weight_tying_shares_one_tensor_not_a_copy():
    tied = make_model(tie_weights=True)
    assert tied.lm_head.weight is tied.embed_tokens.weight

    untied = make_model(tie_weights=False)
    assert untied.lm_head.weight is not untied.embed_tokens.weight
    delta = sum(p.numel() for p in untied.parameters()) - sum(p.numel() for p in tied.parameters())
    assert delta == VOCAB * D_MODEL


def test_from_config_reads_the_yaml_shape():
    model = CausalLM.from_config(
        {
            "model": {
                "vocab_size": VOCAB,
                "d_model": D_MODEL,
                "n_layers": N_LAYERS,
                "n_heads": N_HEADS,
                "n_kv_heads": N_KV_HEADS,
                "d_ff": D_FF,
                "max_seq_len": 64,
            },
            "attention": {"impl": "manual"},
        }
    )
    assert len(model.layers) == N_LAYERS
    assert model.layers[0].attn.impl == "manual"


def test_shipped_default_config_builds():
    """configs/default.yaml must describe a model that actually constructs."""
    from training.train import load_config

    model = CausalLM.from_config(load_config("configs/default.yaml"))
    n = sum(p.numel() for p in model.parameters())
    assert 10e6 < n < 40e6, f"{n / 1e6:.1f}M params is outside the designed range"


def test_sequences_past_max_seq_len_are_rejected():
    model = make_model(max_seq_len=8)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.randint(0, VOCAB, (1, 9)))


# ---------------------------------------------------------------------------
# Causality and caching at whole-model level
# ---------------------------------------------------------------------------


def test_future_tokens_cannot_change_earlier_logits():
    """Causality must survive composition through every layer."""
    model = make_model()
    torch.manual_seed(1)
    a = torch.randint(0, VOCAB, (1, SEQ))
    b = a.clone()
    cut = 3
    b[:, cut:] = torch.randint(0, VOCAB, (1, SEQ - cut))

    torch.testing.assert_close(model(a)[:, :cut], model(b)[:, :cut], rtol=1e-4, atol=1e-5)


def test_cached_decode_matches_full_forward():
    """Token-by-token generation must reproduce one parallel forward exactly.

    At model level this additionally pins that per-layer caches are kept
    separate -- a single shared cache across layers still runs and still
    produces plausible text.
    """
    model = make_model()
    torch.manual_seed(2)
    ids = torch.randint(0, VOCAB, (1, SEQ))

    full = model(ids)

    cache = model.new_cache()
    steps = [model(ids[:, t : t + 1], kv_cache=cache, use_cache=True) for t in range(SEQ)]
    torch.testing.assert_close(full, torch.cat(steps, dim=1), rtol=1e-4, atol=1e-5)


def test_new_cache_has_one_entry_per_layer():
    model = make_model()
    cache = model.new_cache()
    assert len(cache) == N_LAYERS
    assert all(isinstance(c, KVCache) for c in cache)
    assert all(len(c) == 0 for c in cache)


def test_use_cache_without_a_cache_is_rejected():
    """Silently discarding the cache would make generation quadratic again."""
    model = make_model()
    with pytest.raises(ValueError, match="new_cache"):
        model(torch.randint(0, VOCAB, (1, SEQ)), use_cache=True)


# ---------------------------------------------------------------------------
# Initialisation and gradients
# ---------------------------------------------------------------------------


def test_untrained_loss_is_near_uniform_entropy():
    """A correctly initialised LM starts at ln(vocab_size) on random tokens.

    Materially below means the head is leaking; materially above means the
    initialisation scale is wrong.

    Note the shift: the loss must compare ``logits[:, :-1]`` against
    ``ids[:, 1:]``. Scoring a position against *its own* token instead is not a
    weaker test, it is a broken one -- with tied weights the residual stream
    still carries the token's own embedding and the head is that same matrix,
    so self-prediction scores far better than chance and the assertion fails on
    a perfectly correct model.
    """
    model = make_model()
    torch.manual_seed(3)
    ids = torch.randint(0, VOCAB, (8, 32))  # enough tokens to keep sampling noise small
    logits = model(ids)
    loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
    assert abs(loss.item() - math.log(VOCAB)) < 0.2, (
        f"untrained loss {loss.item():.3f} vs ln(V)={math.log(VOCAB):.3f}"
    )


def test_residual_projections_get_depth_scaled_init():
    """Output projections are scaled by 1/sqrt(2 * n_layers) to keep the
    residual stream variance from growing with depth."""
    model = make_model(n_layers=8)
    deep = model.layers[0].attn.o_proj.weight.std().item()
    shallow = model.layers[0].attn.q_proj.weight.std().item()
    assert deep < shallow * 0.6


def test_gradients_reach_every_parameter(ids):
    model = make_model()
    model.train()
    logits = model(ids)
    nn.functional.cross_entropy(logits.reshape(-1, VOCAB), ids.reshape(-1)).backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient has nan/inf"
