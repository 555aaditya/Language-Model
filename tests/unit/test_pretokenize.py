"""Pre-tokenization tests (TDR-020).

The load-bearing test is ``test_pattern_tiles_every_input``. If the pattern has a
gap, ``encode()`` silently drops characters — and a round-trip test only notices
if it happens to use a dropped character. So coverage is asserted directly
against a spread of inputs rather than inferred.
"""

import time

import pytest

from tokenizer import BPE, GPT2_PATTERN, pretokenize, tiles_exactly

SAMPLES = [
    "hello world",
    "The quick brown fox.",
    "it's don't we've I'm you'll he'd",
    "snake_case and CamelCase and kebab-case",
    "numbers 123 4567 0.5 -3",
    "punctuation!!! ??? ...---",
    "  leading and trailing   ",
    "tabs\tand\nnewlines\r\n",
    "unicode: héllo Ω 你好 日本語",
    "emoji 🎉 mixed ☕ with text",
    "",
    " ",
    "\n\n\n",
    "a",
    "_",
    "_leading_underscore",
    "url https://example.com/a?b=c#d",
    "math ∑ ≈ ∞ × ÷",
    "quotes \"double\" 'single' `back`",
    "mixed123abc456",
]


# ---------------------------------------------------------------------------
# Coverage -- the pattern must lose nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", SAMPLES)
def test_pattern_tiles_every_input(text):
    """Concatenating the pieces must reproduce the input exactly."""
    assert tiles_exactly(text), f"pattern dropped characters from {text!r}"


def test_pattern_tiles_a_large_real_document():
    from pathlib import Path

    text = Path("docs/TDD.md").read_text(encoding="utf-8")
    assert tiles_exactly(text)


@pytest.mark.parametrize("codepoint", [0x5F, 0x7F, 0xA0, 0x2603, 0x1F600])
def test_awkward_codepoints_are_not_dropped(codepoint):
    """Underscore, DEL, NBSP, snowman, emoji -- each has bitten a naive pattern."""
    text = f"a{chr(codepoint)}b"
    assert tiles_exactly(text)


# ---------------------------------------------------------------------------
# Splitting behaviour
# ---------------------------------------------------------------------------


def test_a_leading_space_attaches_to_its_word():
    """GPT-2 convention: " world" is one piece, so spacing is learned not guessed."""
    assert pretokenize("hello world") == ["hello", " world"]


def test_letters_digits_and_symbols_split_apart():
    assert pretokenize("abc123!!") == ["abc", "123", "!!"]


def test_contractions_stay_whole():
    assert "'s" in pretokenize("it's")


def test_pattern_is_reachable_from_the_tokenizer():
    assert BPE(vocab_size=256).pattern is GPT2_PATTERN


# ---------------------------------------------------------------------------
# Merges must not cross piece boundaries
# ---------------------------------------------------------------------------


def test_no_merge_spans_a_piece_boundary():
    """ "the cat" repeated cannot produce a token containing the space-joined pair.

    Before TDR-020 the merge loop ran over whole documents, so "e c" was a
    learnable token. Restricting merges to pieces is what makes the encode cache
    valid -- a cached piece would otherwise depend on its neighbours.
    """
    tok = BPE(vocab_size=400)
    tok.train(["the cat sat on the mat "] * 50)

    for idx in tok.decoder:
        raw = tok.token_bytes(idx)
        # A token may *start* with a space (GPT-2 style " the"), but must never
        # contain an interior space -- that would be a cross-piece merge.
        assert b" " not in raw[1:], f"token {raw!r} spans a piece boundary"


def test_pieces_encode_independently_of_context():
    """The cache is only sound if a piece's ids never depend on what surrounds it."""
    tok = BPE(vocab_size=400)
    tok.train(["the cat sat on the mat "] * 50)

    alone = tok.encode(" cat")
    in_context = tok.encode("the cat sat")
    assert alone == in_context[len(tok.encode("the")) :][: len(alone)]


# ---------------------------------------------------------------------------
# Caching and complexity
# ---------------------------------------------------------------------------


def test_each_unique_piece_is_merged_only_once():
    """Repetition must cost a dict lookup, not a re-merge.

    Deterministic stand-in for the throughput claim: 200 copies of the same
    sentence must invoke the merge routine only once per distinct piece.
    """
    tok = BPE(vocab_size=400)
    tok.train(["the cat sat on the mat "] * 20)
    tok._piece_cache.clear()

    calls = []
    original = tok._encode_piece
    tok._encode_piece = lambda piece: calls.append(piece) or original(piece)  # type: ignore[method-assign]

    text = "the cat sat on the mat " * 200
    tok.encode(text)

    assert len(calls) == len(set(calls)), "a piece was merged more than once"
    assert len(calls) < 10, f"expected a handful of distinct pieces, got {len(calls)}"


def test_encode_scales_far_better_than_quadratically():
    """Regression for the O(n^2) whole-document merge loop.

    The old implementation measured 4.0x time per 2x input. A 16x input increase
    would therefore have cost ~256x. The bound here is deliberately loose -- it
    only needs to fail for a quadratic implementation, not to pin a throughput.
    """
    from pathlib import Path

    text = Path("docs/TDD.md").read_text(encoding="utf-8") * 4
    tok = BPE(vocab_size=1024)
    tok.train([text[:100_000]])

    def encode_time(n):
        tok._piece_cache.clear()
        chunk = text[:n]
        start = time.perf_counter()
        tok.encode(chunk)
        return time.perf_counter() - start

    small = encode_time(8_192)
    large = encode_time(131_072)  # 16x the input
    # Quadratic would be ~256x. The bound is loose on purpose: this runs on
    # shared CI runners where a 2x timing margin flakes, and the test only needs
    # to separate linear-ish from quadratic.
    assert large < small * 64, f"16x input cost {large / small:.0f}x time -- looks quadratic"


def test_the_cache_does_not_change_the_answer():
    tok = BPE(vocab_size=400)
    tok.train(["the cat sat on the mat "] * 20)
    text = "the cat sat on the mat and the dog ran"

    warm = tok.encode(text)
    tok._piece_cache.clear()
    cold = tok.encode(text)
    assert warm == cold


def test_training_clears_a_stale_cache():
    """Cached ids from a previous vocabulary would be silently wrong."""
    tok = BPE(vocab_size=400)
    tok.encode("hello")
    assert tok._piece_cache
    tok.train(["hello world "] * 20)
    assert not tok._piece_cache


# ---------------------------------------------------------------------------
# Special tokens under the new splitter
# ---------------------------------------------------------------------------


def test_special_tokens_survive_pretokenization():
    tok = BPE(vocab_size=400, special_tokens=["<|endoftext|>", "<|pad|>"])
    tok.train(["hello world "] * 20)
    eot = tok.encoder["<|endoftext|>"]

    ids = tok.encode("hello<|endoftext|>world")
    assert eot in ids
    assert tok.decode(ids) == "hello<|endoftext|>world"


def test_adjacent_special_tokens_are_separate_ids():
    tok = BPE(vocab_size=400, special_tokens=["<|endoftext|>"])
    tok.train(["hello "] * 20)
    eot = tok.encoder["<|endoftext|>"]
    assert tok.encode("<|endoftext|><|endoftext|>") == [eot, eot]


def test_a_longer_special_token_wins_over_its_prefix():
    """Alternation is ordered longest-first, or "<|end|>" would shadow "<|endoftext|>"."""
    tok = BPE(vocab_size=400, special_tokens=["<|end|>", "<|endoftext|>"])
    tok.train(["hello "] * 20)
    assert tok.encode("<|endoftext|>") == [tok.encoder["<|endoftext|>"]]


def test_special_token_at_each_boundary():
    tok = BPE(vocab_size=400, special_tokens=["<|endoftext|>"])
    tok.train(["hello world "] * 20)
    for text in ("<|endoftext|>hello", "hello<|endoftext|>", "<|endoftext|>"):
        assert tok.decode(tok.encode(text)) == text
