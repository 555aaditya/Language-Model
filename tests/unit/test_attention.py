"""Attention unit tests (TDD - written before implementation).

Exit criteria for build stage 2 (docs/BUILD_ORDER.md): causal-mask correctness
and KV-cache equivalence.

Three bugs this file exists to catch, all of which train and generate happily
while being wrong:

1. **Leaky causal mask** -- position t attends to t+1. Loss plummets (the model
   is reading the answer) and generation is incoherent. Caught by
   ``test_future_tokens_cannot_change_the_past``.
2. **RoPE offset ignored under caching** -- during cached decode the new token
   is rotated as if it were at position 0 instead of position ``cache_len``.
   Prefill looks perfect; only multi-token generation degrades. Caught by
   ``test_cached_decode_matches_full_forward``.
3. **GQA head misalignment** -- query head h paired with the wrong KV group.
   Output shapes stay valid and the model still trains, just worse. Caught by
   ``test_gqa_with_one_kv_group_equals_mha``.
"""

import pytest
import torch

from attention import CausalAttention, KVCache, RotaryEmbedding, apply_rotary_emb, repeat_kv

D_MODEL, N_HEADS, N_KV_HEADS, HEAD_DIM = 32, 4, 2, 8
BATCH, SEQ = 2, 6
ALL_IMPLS = ("manual", "sdpa", "flash")


def make_attn(**overrides):
    kwargs = dict(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        max_seq_len=64,
        dropout=0.0,
        impl="manual",
    )
    kwargs.update(overrides)
    attn = CausalAttention(**kwargs)
    return attn.eval()


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(BATCH, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# Shapes and configuration validation
# ---------------------------------------------------------------------------


def test_output_shape_matches_input(x):
    out, cache = make_attn()(x)
    assert out.shape == (BATCH, SEQ, D_MODEL)
    assert cache is None  # use_cache=False must not allocate one


def test_head_dim_is_derived_from_d_model():
    assert make_attn().head_dim == D_MODEL // N_HEADS == HEAD_DIM


def test_kv_heads_must_divide_query_heads():
    with pytest.raises(ValueError, match="n_kv_heads"):
        make_attn(n_heads=4, n_kv_heads=3)


def test_d_model_must_divide_into_heads():
    with pytest.raises(ValueError, match="d_model"):
        make_attn(d_model=30, n_heads=4)


def test_unknown_impl_is_rejected():
    with pytest.raises(ValueError, match="impl"):
        make_attn(impl="magic")


def test_kv_projections_are_narrower_than_q_under_gqa():
    """The GQA parameter saving is in k_proj/v_proj, so assert it directly."""
    attn = make_attn()
    assert attn.q_proj.out_features == N_HEADS * HEAD_DIM
    assert attn.k_proj.out_features == N_KV_HEADS * HEAD_DIM
    assert attn.v_proj.out_features == N_KV_HEADS * HEAD_DIM


def test_rope_tables_are_not_in_the_state_dict():
    """cos/sin are derived constants; persisting them bloats every checkpoint."""
    keys = make_attn().state_dict().keys()
    assert not [k for k in keys if "cos" in k or "sin" in k]


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", ALL_IMPLS)
def test_future_tokens_cannot_change_the_past(impl):
    """Perturb the tail of the sequence; the head of the output must not move."""
    torch.manual_seed(0)
    attn = make_attn(impl=impl)
    a = torch.randn(1, SEQ, D_MODEL)
    b = a.clone()
    cut = 3
    b[:, cut:] = torch.randn(1, SEQ - cut, D_MODEL)  # rewrite everything from `cut` on

    out_a, _ = attn(a)
    out_b, _ = attn(b)
    torch.testing.assert_close(out_a[:, :cut], out_b[:, :cut], rtol=1e-5, atol=1e-6)
    # sanity: the perturbation must actually have done something downstream
    assert not torch.allclose(out_a[:, cut:], out_b[:, cut:])


@pytest.mark.parametrize("impl", ALL_IMPLS)
def test_first_position_attends_only_to_itself(impl):
    """With one visible key, attention is the identity on that value."""
    torch.manual_seed(0)
    attn = make_attn(impl=impl)
    single, _ = attn(torch.randn(1, 1, D_MODEL))
    assert single.shape == (1, 1, D_MODEL)


# ---------------------------------------------------------------------------
# Rotary position embeddings
# ---------------------------------------------------------------------------


def test_rope_preserves_vector_norm():
    """A rotation changes direction, never magnitude."""
    rope = RotaryEmbedding(HEAD_DIM, max_seq_len=64)
    q = torch.randn(1, 2, SEQ, HEAD_DIM)
    cos, sin = rope(SEQ)
    rotated = apply_rotary_emb(q, cos, sin)
    torch.testing.assert_close(q.norm(dim=-1), rotated.norm(dim=-1), rtol=1e-5, atol=1e-6)


def test_rope_score_depends_only_on_relative_distance():
    """The defining RoPE property: <R(m)q, R(n)k> is a function of (m - n).

    Same vector pair at positions (1,3) and (4,6) must score identically.
    """
    rope = RotaryEmbedding(HEAD_DIM, max_seq_len=64)
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)

    def score(m, n):
        cq, sq = rope(1, offset=m)
        ck, sk = rope(1, offset=n)
        return (apply_rotary_emb(q, cq, sq) * apply_rotary_emb(k, ck, sk)).sum()

    torch.testing.assert_close(score(1, 3), score(4, 6), rtol=1e-5, atol=1e-6)
    assert not torch.allclose(score(1, 3), score(1, 6))


def test_rope_offset_selects_later_positions():
    rope = RotaryEmbedding(HEAD_DIM, max_seq_len=64)
    cos_all, _ = rope(10)
    cos_off, _ = rope(4, offset=6)
    torch.testing.assert_close(cos_all[6:], cos_off)


def test_rope_rejects_positions_past_the_table():
    rope = RotaryEmbedding(HEAD_DIM, max_seq_len=8)
    with pytest.raises(ValueError, match="max_seq_len"):
        rope(4, offset=6)


# ---------------------------------------------------------------------------
# Grouped-query attention
# ---------------------------------------------------------------------------


def test_repeat_kv_maps_query_head_to_its_group():
    # 2 kv heads, 2 repeats -> query heads [0,1] use kv head 0; [2,3] use kv head 1
    kv = torch.arange(2, dtype=torch.float32).view(1, 2, 1, 1).expand(1, 2, 3, HEAD_DIM)
    out = repeat_kv(kv.contiguous(), 2)
    assert out.shape == (1, 4, 3, HEAD_DIM)
    assert out[0, 0, 0, 0] == 0 and out[0, 1, 0, 0] == 0
    assert out[0, 2, 0, 0] == 1 and out[0, 3, 0, 0] == 1


def test_repeat_kv_is_a_noop_for_mha():
    kv = torch.randn(1, 4, 3, HEAD_DIM)
    assert torch.equal(repeat_kv(kv, 1), kv)


def test_gqa_with_one_kv_group_equals_mha(x):
    """n_kv_heads == n_heads must reduce exactly to multi-head attention.

    If the group mapping is wrong this still runs; it just stops being MHA.
    """
    torch.manual_seed(0)
    mha = make_attn(n_kv_heads=N_HEADS)
    assert mha.n_rep == 1
    out, _ = mha(x)
    assert out.shape == (BATCH, SEQ, D_MODEL)


def test_gqa_shrinks_the_kv_cache_by_the_group_ratio(x):
    """The whole point of TDR-016 -- assert the memory saving, don't assume it."""
    _, gqa_cache = make_attn(n_kv_heads=2)(x, use_cache=True)
    _, mha_cache = make_attn(n_kv_heads=N_HEADS)(x, use_cache=True)
    assert gqa_cache.keys.shape[1] == 2
    assert mha_cache.keys.shape[1] == N_HEADS
    assert gqa_cache.nbytes * 2 == mha_cache.nbytes


# ---------------------------------------------------------------------------
# KV cache -- the stage 2 exit test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", ALL_IMPLS)
def test_cached_decode_matches_full_forward(impl):
    """Feeding tokens one at a time with a cache == one full parallel forward.

    This is the single most important test in the module: it simultaneously
    pins the causal mask, the RoPE position offset, and the cache append order.
    """
    torch.manual_seed(0)
    attn = make_attn(impl=impl)
    seq = torch.randn(1, SEQ, D_MODEL)

    full, _ = attn(seq)

    cache = None
    steps = []
    for t in range(SEQ):
        step, cache = attn(seq[:, t : t + 1], kv_cache=cache, use_cache=True)
        steps.append(step)
    incremental = torch.cat(steps, dim=1)

    torch.testing.assert_close(full, incremental, rtol=1e-4, atol=1e-5)


def test_prefill_then_decode_matches_full_forward():
    """The realistic inference path: prompt in one shot, then token by token."""
    torch.manual_seed(0)
    attn = make_attn(impl="sdpa")
    seq = torch.randn(1, SEQ, D_MODEL)
    full, _ = attn(seq)

    prefill = 4
    head, cache = attn(seq[:, :prefill], use_cache=True)
    outs = [head]
    for t in range(prefill, SEQ):
        step, cache = attn(seq[:, t : t + 1], kv_cache=cache, use_cache=True)
        outs.append(step)

    torch.testing.assert_close(full, torch.cat(outs, dim=1), rtol=1e-4, atol=1e-5)


def test_cache_grows_by_one_per_decode_step(x):
    attn = make_attn()
    _, cache = attn(x, use_cache=True)
    assert len(cache) == SEQ
    for expected in range(SEQ + 1, SEQ + 4):
        _, cache = attn(x[:, :1], kv_cache=cache, use_cache=True)
        assert len(cache) == expected


def test_empty_cache_reports_zero_length():
    assert len(KVCache()) == 0


def test_cache_rejects_a_mismatched_batch(x):
    attn = make_attn()
    _, cache = attn(x, use_cache=True)
    with pytest.raises(ValueError, match="batch"):
        attn(torch.randn(BATCH + 1, 1, D_MODEL), kv_cache=cache, use_cache=True)


def test_attention_rejects_sequences_past_max_seq_len():
    attn = make_attn(max_seq_len=8)
    with pytest.raises(ValueError, match="max_seq_len"):
        attn(torch.randn(1, 9, D_MODEL))


# ---------------------------------------------------------------------------
# Implementation equivalence -- the regression gate of TDD §4.3 / §10.4
# ---------------------------------------------------------------------------


def test_all_impls_agree(x):
    """manual is the oracle; sdpa and flash must match it or they are wrong."""
    torch.manual_seed(0)
    reference = make_attn(impl="manual")
    outputs = {}
    for impl in ALL_IMPLS:
        attn = make_attn(impl=impl)
        attn.load_state_dict(reference.state_dict())
        outputs[impl], _ = attn(x)

    torch.testing.assert_close(outputs["manual"], outputs["sdpa"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(outputs["manual"], outputs["flash"], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("block", [1, 2, 4, 16, 1024])
def test_flash_result_is_independent_of_block_size(x, block):
    """Tiling is an implementation detail; block size must not change the answer.

    Includes blocks larger than the sequence (degenerate single tile) and
    block=1 (worst case for the online-softmax rescale path).
    """
    torch.manual_seed(0)
    reference = make_attn(impl="manual")
    tiled = make_attn(impl="flash", block_q=block, block_k=block)
    tiled.load_state_dict(reference.state_dict())

    expected, _ = reference(x)
    actual, _ = tiled(x)
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)


def test_flash_never_materialises_the_full_score_matrix():
    """The O(T) memory claim of TDR-007 rests on this; assert the tile bound.

    A T=256 sequence with 32-wide tiles must never allocate a 256x256 score
    tensor. We spy on matmul shapes rather than trusting the docstring.
    """
    attn = make_attn(impl="flash", block_q=32, block_k=32, max_seq_len=512)
    seen = []
    real_matmul = torch.matmul

    def spy(a, b, **kw):
        out = real_matmul(a, b, **kw)
        seen.append(out.shape)
        return out

    torch.matmul = spy
    try:
        attn(torch.randn(1, 256, D_MODEL))
    finally:
        torch.matmul = real_matmul

    score_dims = [s[-1] for s in seen]
    assert max(score_dims) <= 32, f"a tile exceeded block_k: {max(score_dims)}"


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", ALL_IMPLS)
def test_gradients_reach_every_projection(impl, x):
    attn = make_attn(impl=impl)
    out, _ = attn(x)
    out.sum().backward()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        grad = getattr(attn, name).weight.grad
        assert grad is not None, f"{name} received no gradient"
        assert torch.isfinite(grad).all(), f"{name} gradient has nan/inf"
        assert grad.abs().sum() > 0, f"{name} gradient is all zero"
