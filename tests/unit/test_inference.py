"""Sampling and generation tests (TDD - written before implementation).

Exit criterion for build stage 5 (docs/BUILD_ORDER.md): deterministic greedy
decoding.

The load-bearing test is ``test_cached_generation_matches_uncached``. A KV cache
that has drifted out of equivalence with a full forward still generates fluent
text -- it is just generating from a subtly different model than the one that
was trained. Comparing the two paths under greedy decoding is the only cheap
way to notice.
"""

import math

import pytest
import torch

from inference import apply_top_k, apply_top_p, generate, generate_ids, sample
from model import CausalLM

VOCAB = 32


def make_model(**overrides):
    cfg = {
        "model": {
            "vocab_size": VOCAB,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "d_ff": 64,
            "max_seq_len": 64,
            **overrides,
        },
        "attention": {"impl": "manual"},
    }
    torch.manual_seed(0)
    return CausalLM.from_config(cfg).eval()


def logits_from_probs(probs):
    return torch.log(torch.tensor(probs, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Greedy
# ---------------------------------------------------------------------------


def test_temperature_zero_is_argmax():
    logits = torch.tensor([0.1, 5.0, 0.3, 2.0])
    assert sample(logits, temperature=0.0) == 1


def test_greedy_is_deterministic():
    logits = torch.randn(VOCAB)
    assert len({sample(logits, temperature=0.0) for _ in range(20)}) == 1


def test_top_k_one_equals_greedy():
    logits = torch.randn(VOCAB)
    assert sample(logits, top_k=1) == int(logits.argmax())


def test_accepts_a_leading_batch_dimension():
    logits = torch.tensor([[0.1, 5.0, 0.3]])
    assert sample(logits, temperature=0.0) == 1


def test_rejects_a_real_batch():
    with pytest.raises(ValueError, match="single"):
        sample(torch.randn(4, VOCAB), temperature=0.0)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_top_k_masks_everything_outside_the_k_best():
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
    kept = apply_top_k(logits, 2)
    assert torch.isinf(kept[[0, 2, 3]]).all()
    assert kept[1] == 5.0 and kept[4] == 4.0


def test_top_k_larger_than_the_vocabulary_is_a_noop():
    logits = torch.randn(5)
    torch.testing.assert_close(apply_top_k(logits, 99), logits)


def test_sampling_never_leaves_the_top_k_support():
    torch.manual_seed(0)
    logits = torch.randn(VOCAB)
    allowed = set(torch.topk(logits, 3).indices.tolist())
    drawn = {sample(logits, top_k=3) for _ in range(200)}
    assert drawn <= allowed


def test_top_p_keeps_the_minimal_set_reaching_the_threshold():
    """probs [.5, .3, .15, .05], p=0.7 -> keep {0, 1} (cumulative 0.8).

    Dropping the crossing token would retain only 0.5, i.e. less than p.
    """
    logits = logits_from_probs([0.5, 0.3, 0.15, 0.05])
    kept = apply_top_p(logits, 0.7)
    assert torch.isfinite(kept[[0, 1]]).all()
    assert torch.isinf(kept[[2, 3]]).all()


def test_top_p_always_keeps_the_argmax():
    """A confident distribution must not mask its own top token into NaN."""
    logits = logits_from_probs([0.9, 0.06, 0.03, 0.01])
    kept = apply_top_p(logits, 0.5)
    assert torch.isfinite(kept[0])
    assert torch.isfinite(torch.softmax(kept, -1)).all()


def test_top_p_of_one_is_a_noop():
    logits = torch.randn(8)
    torch.testing.assert_close(apply_top_p(logits, 1.0), logits)


def test_tiny_top_p_collapses_to_greedy():
    logits = torch.randn(VOCAB)
    assert {sample(logits, top_p=1e-6) for _ in range(20)} == {int(logits.argmax())}


def test_temperature_flattens_the_distribution():
    """High temperature must broaden the support actually drawn from."""
    torch.manual_seed(0)
    logits = torch.randn(VOCAB) * 5
    cold = {sample(logits, temperature=0.1) for _ in range(200)}
    hot = {sample(logits, temperature=10.0) for _ in range(200)}
    assert len(hot) > len(cold)


def test_invalid_sampling_parameters_are_rejected():
    logits = torch.randn(8)
    with pytest.raises(ValueError, match="temperature"):
        sample(logits, temperature=-1.0)
    with pytest.raises(ValueError, match="top_p"):
        sample(logits, top_p=0.0)
    with pytest.raises(ValueError, match="top_p"):
        sample(logits, top_p=1.5)


def test_sampling_is_reproducible_with_a_seeded_generator():
    logits = torch.randn(VOCAB)
    a = [sample(logits, generator=torch.Generator().manual_seed(7)) for _ in range(5)]
    b = [sample(logits, generator=torch.Generator().manual_seed(7)) for _ in range(5)]
    assert a == b


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="requires Apple MPS"
            ),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
        ),
    ],
)
def test_a_cpu_generator_works_against_non_cpu_logits(device):
    """torch.multinomial demands generator and tensor share a device.

    A seeded CPU generator is the obvious way to ask for reproducible sampling,
    so it must keep working once the model moves off CPU -- otherwise every
    accelerator run with a seed raises. Regression test: this only reproduces
    on a real accelerator, so it is skipped rather than faked on CPU-only CI.
    """
    logits = torch.randn(VOCAB, device=device)
    token = sample(logits, temperature=0.9, generator=torch.Generator().manual_seed(3))
    assert 0 <= token < VOCAB


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_a_seed_gives_the_same_token_on_every_device():
    """Drawing on the generator's device makes seeded output device-independent."""
    logits = torch.randn(VOCAB)
    on_cpu = sample(logits, temperature=0.9, generator=torch.Generator().manual_seed(11))
    on_mps = sample(logits.to("mps"), temperature=0.9, generator=torch.Generator().manual_seed(11))
    assert on_cpu == on_mps


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_generates_exactly_the_requested_number_of_tokens():
    model = make_model()
    out = generate_ids(model, torch.tensor([[1, 2, 3]]), max_new_tokens=7, temperature=0.0)
    assert len(out) == 7
    assert all(0 <= t < VOCAB for t in out)


def test_greedy_generation_is_deterministic():
    """The stage 5 exit test."""
    model = make_model()
    prompt = torch.tensor([[1, 2, 3]])
    runs = [
        tuple(generate_ids(model, prompt, max_new_tokens=10, temperature=0.0)) for _ in range(3)
    ]
    assert len(set(runs)) == 1


def test_cached_generation_matches_uncached():
    """The cache must be exactly equivalent to recomputing the whole prefix."""
    model = make_model()
    prompt = torch.tensor([[5, 9, 2, 7]])
    cached = generate_ids(model, prompt, max_new_tokens=12, temperature=0.0, use_cache=True)
    plain = generate_ids(model, prompt, max_new_tokens=12, temperature=0.0, use_cache=False)
    assert cached == plain


def test_stop_token_ends_generation_early():
    """Generation must halt at the *first* occurrence of the stop id.

    An untrained model often emits the same token repeatedly, so indexing the
    expected stop position rather than assuming greedy[k] is its first
    appearance is what keeps this test honest.
    """
    model = make_model()
    prompt = torch.tensor([[1, 2, 3]])
    greedy = generate_ids(model, prompt, max_new_tokens=10, temperature=0.0)

    stop = greedy[2]
    first = greedy.index(stop)
    out = generate_ids(model, prompt, max_new_tokens=10, temperature=0.0, stop_id=stop)
    assert out == greedy[: first + 1]
    assert out[-1] == stop


def test_generation_stops_at_max_seq_len():
    """Running past the rotary table would raise; generation must stop first."""
    model = make_model(max_seq_len=16)
    out = generate_ids(model, torch.tensor([[1, 2, 3]]), max_new_tokens=1000, temperature=0.0)
    assert 0 < len(out) <= 16


def test_generation_leaves_the_model_in_its_original_mode():
    model = make_model()
    model.train()
    generate_ids(model, torch.tensor([[1, 2]]), max_new_tokens=2, temperature=0.0)
    assert model.training, "generate() must restore train mode it did not own"


def test_generation_allocates_no_gradients():
    model = make_model()
    generate_ids(model, torch.tensor([[1, 2]]), max_new_tokens=3, temperature=0.0)
    assert all(p.grad is None for p in model.parameters())


def test_empty_prompt_is_rejected():
    with pytest.raises(ValueError, match="empty prompt"):
        generate_ids(make_model(), torch.zeros(1, 0, dtype=torch.long), max_new_tokens=1)


# ---------------------------------------------------------------------------
# String-level contract
# ---------------------------------------------------------------------------


def test_generate_round_trips_through_a_tokenizer():
    from tokenizer import BPE

    # A byte-level BPE cannot go below 256 (one token per byte value), so the
    # model's vocabulary has to be at least that to pair with a real tokenizer.
    tok = BPE(vocab_size=256)
    model = make_model(vocab_size=256)

    text = generate(model, tok, "ab", max_new_tokens=5, temperature=0.0)
    assert text.startswith("ab")

    only_new = generate(model, tok, "ab", max_new_tokens=5, temperature=0.0, return_prompt=False)
    assert text == "ab" + only_new


def test_untrained_generation_is_roughly_uniform():
    """A fresh model should not favour one token -- that would signal a leak."""
    model = make_model()
    torch.manual_seed(0)
    out = generate_ids(model, torch.tensor([[1, 2, 3]]), max_new_tokens=200)
    assert len(set(out)) > math.sqrt(VOCAB)
