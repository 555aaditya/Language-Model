"""Preallocated KV cache tests (TDD - written before implementation).

The arena has to be a *strict* substitute for the concatenating cache: same
interface, same numbers out. `test_pooled_generation_matches_the_concat_cache`
is the one that matters -- an arena that returns the whole buffer instead of the
filled prefix, or forgets to advance its write head, still generates fluent text
from a quietly different model.
"""

import pytest
import torch

from attention import CausalAttention, KVCache
from inference import generate_ids
from model import CausalLM
from optimization import PreallocatedKVCache, preallocated_cache

VOCAB, D_MODEL, N_HEADS, N_KV_HEADS, HEAD_DIM = 64, 32, 4, 2, 8


def make_model():
    cfg = {
        "model": {
            "vocab_size": VOCAB,
            "d_model": D_MODEL,
            "n_layers": 2,
            "n_heads": N_HEADS,
            "n_kv_heads": N_KV_HEADS,
            "d_ff": 64,
            "max_seq_len": 64,
        },
        "attention": {"impl": "manual"},
    }
    torch.manual_seed(0)
    return CausalLM.from_config(cfg).eval()


def make_arena(batch=1, max_seq_len=16):
    return PreallocatedKVCache(batch, N_KV_HEADS, max_seq_len, HEAD_DIM)


def kv(batch=1, n=1):
    return torch.randn(batch, N_KV_HEADS, n, HEAD_DIM)


# ---------------------------------------------------------------------------
# Drop-in compatibility
# ---------------------------------------------------------------------------


def test_it_is_a_kv_cache():
    """Subclassing is what lets CausalAttention accept it without a cast."""
    assert isinstance(make_arena(), KVCache)


def test_starts_empty_despite_holding_a_full_buffer():
    arena = make_arena(max_seq_len=16)
    assert len(arena) == 0
    assert arena.keys.shape[2] == 16


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_update_returns_only_the_filled_prefix():
    """Returning the whole arena would feed attention a tail of zeroed keys."""
    arena = make_arena(max_seq_len=16)
    keys, values = arena.update(kv(n=3), kv(n=3))
    assert keys.shape[2] == 3 and values.shape[2] == 3
    assert len(arena) == 3


def test_successive_writes_advance_the_head():
    arena = make_arena(max_seq_len=16)
    arena.update(kv(n=4), kv(n=4))
    keys, _ = arena.update(kv(n=1), kv(n=1))
    assert keys.shape[2] == 5
    assert len(arena) == 5


def test_written_values_are_preserved_in_order():
    arena = make_arena(max_seq_len=8)
    first, second = kv(n=2), kv(n=1)
    arena.update(first, first)
    keys, _ = arena.update(second, second)
    torch.testing.assert_close(keys[:, :, :2], first)
    torch.testing.assert_close(keys[:, :, 2:3], second)


def test_the_buffer_is_never_reallocated():
    """The entire point: steady-state decoding must not allocate."""
    arena = make_arena(max_seq_len=32)
    identity = id(arena.keys)
    pointer = arena.keys.data_ptr()
    for _ in range(20):
        arena.update(kv(), kv())
    assert id(arena.keys) == identity
    assert arena.keys.data_ptr() == pointer


def test_overflowing_the_arena_raises():
    arena = make_arena(max_seq_len=4)
    arena.update(kv(n=4), kv(n=4))
    with pytest.raises(ValueError, match="full"):
        arena.update(kv(n=1), kv(n=1))


def test_batch_mismatch_raises():
    arena = make_arena(batch=2, max_seq_len=8)
    with pytest.raises(ValueError, match="batch"):
        arena.update(kv(batch=3), kv(batch=3))


def test_reset_rewinds_without_freeing_the_arena():
    arena = make_arena(max_seq_len=8)
    pointer = arena.keys.data_ptr()
    arena.update(kv(n=3), kv(n=3))
    arena.reset()
    assert len(arena) == 0
    assert arena.keys.data_ptr() == pointer, "reset freed the arena it exists to keep"


def test_bf16_input_is_cast_into_an_fp32_arena():
    arena = make_arena(max_seq_len=8)
    keys, _ = arena.update(kv().bfloat16(), kv().bfloat16())
    assert keys.dtype == torch.float32
    assert torch.isfinite(keys).all()


# ---------------------------------------------------------------------------
# Memory accounting
# ---------------------------------------------------------------------------


def test_nbytes_reports_the_reserved_arena_not_the_used_part():
    """Quoting the used figure would hide the cost this design actually pays."""
    arena = make_arena(max_seq_len=64)
    reserved = arena.nbytes
    arena.update(kv(n=1), kv(n=1))
    assert arena.nbytes == reserved
    assert arena.nbytes_used < reserved


def test_used_bytes_grow_with_the_fill():
    arena = make_arena(max_seq_len=64)
    arena.update(kv(n=1), kv(n=1))
    one = arena.nbytes_used
    arena.update(kv(n=3), kv(n=3))
    assert arena.nbytes_used == pytest.approx(one * 4)


def test_an_arena_costs_more_than_a_concat_cache_at_short_lengths():
    """State the tradeoff rather than pretending preallocation is free."""
    arena = make_arena(max_seq_len=64)
    concat = KVCache()
    arena.update(kv(n=2), kv(n=2))
    concat.update(kv(n=2), kv(n=2))
    assert arena.nbytes > concat.nbytes


# ---------------------------------------------------------------------------
# Equivalence -- the load-bearing tests
# ---------------------------------------------------------------------------


def test_attention_gives_the_same_output_with_either_cache():
    torch.manual_seed(0)
    attn = CausalAttention(D_MODEL, N_HEADS, N_KV_HEADS, max_seq_len=64, impl="manual").eval()
    seq = torch.randn(1, 6, D_MODEL)

    concat: KVCache | None = None
    arena = PreallocatedKVCache(1, N_KV_HEADS, 64, HEAD_DIM)

    for t in range(6):
        step = seq[:, t : t + 1]
        expected, concat = attn(step, kv_cache=concat, use_cache=True)
        actual, _ = attn(step, kv_cache=arena, use_cache=True)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_pooled_generation_matches_the_concat_cache():
    """Greedy decoding must produce identical tokens through either cache."""
    model = make_model()
    prompt = torch.tensor([[3, 9, 14]])

    baseline = generate_ids(model, prompt, max_new_tokens=12, temperature=0.0)

    arenas = preallocated_cache(model, batch_size=1, max_seq_len=64)
    logits = model(prompt, kv_cache=arenas, use_cache=True)
    pooled = []
    for _ in range(12):
        nxt = int(logits[0, -1].argmax())
        pooled.append(nxt)
        logits = model(torch.tensor([[nxt]]), kv_cache=arenas, use_cache=True)

    assert pooled == baseline


def test_preallocated_cache_builds_one_arena_per_layer():
    model = make_model()
    arenas = preallocated_cache(model, batch_size=2, max_seq_len=32)
    assert len(arenas) == len(model.blocks)
    assert all(isinstance(a, PreallocatedKVCache) for a in arenas)
    assert arenas[0].keys.shape == (2, N_KV_HEADS, 32, HEAD_DIM)


def test_arena_defaults_to_the_models_own_limits():
    model = make_model()
    arena = preallocated_cache(model)[0]
    assert arena.max_seq_len == model.max_seq_len
    assert arena.keys.device == next(model.parameters()).device
