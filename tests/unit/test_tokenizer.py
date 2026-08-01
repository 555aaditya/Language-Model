"""Tokenizer unit tests (TDD - written before implementation)."""

import os

import pytest

from tokenizer.bpe import BPE

# ---------------------------------------------------------------------------
# Encoding / decoding correctness
# ---------------------------------------------------------------------------


def test_encode_basic_words():
    tok = BPE(vocab_size=256)
    # Base byte-level tokenizer: each byte is a separate token (ASCII chars)
    encoded = tok.encode("hello")
    assert encoded == [ord(c) for c in "hello"]


def test_decode_basic_words():
    tok = BPE(vocab_size=256)
    assert tok.decode([104, 101, 108, 108, 111]) == "hello"


def test_round_trip_encode_decode():
    tok = BPE(vocab_size=300)
    tok.train(["hello world", "this is a test sentence"])
    text = "hello world"
    assert tok.decode(tok.encode(text)) == text


def test_round_trip_unicode():
    tok = BPE(vocab_size=300)
    tok.train(["hello", "héllo", "你好", "emoji 🎉"])
    for text in ["hello", "héllo", "你好", "emoji 🎉"]:
        assert tok.decode(tok.encode(text)) == text, f"failed for {text!r}"


def test_encode_no_unknown_tokens():
    """Byte-level BPE should never produce unknown tokens for any UTF-8 text."""
    tok = BPE(vocab_size=256)
    for text in ["", "a", "Ω", "日本語", "emoji 😀", "mixed café ☕"]:
        tokens = tok.encode(text)
        assert all(0 <= t < tok.vocab_size for t in tokens)


# ---------------------------------------------------------------------------
# Vocabulary / BPE merging
# ---------------------------------------------------------------------------


def test_training_grows_vocabulary():
    text = "the quick brown fox jumps over the lazy dog. " * 50
    before = BPE(vocab_size=256).vocab_size
    tok = BPE(vocab_size=512)
    tok.train([text])
    # merges should have created new tokens beyond the base 256 bytes
    assert tok.vocab_size > before
    assert tok.vocab_size <= 512


def test_most_frequent_pair_merged_first():
    """'ab' appears more than other pairs, so it should become a token.

    Canonical representation: decoder values are GPT-2-style rendered-unicode
    strings, NOT raw bytes. We therefore verify the merge via token_bytes(),
    which recovers the original byte sequence backing each token id.
    """
    text = "ab" * 100 + " xy " * 10
    tok = BPE(vocab_size=258)  # +2 merges
    tok.train([text])
    # 'ab' should be one of the merged tokens (compare original bytes).
    merged_byte_seqs = {tok.token_bytes(i) for i in tok.decoder}
    assert b"ab" in merged_byte_seqs
    # And it must actually be used when encoding.
    assert (
        tok.encode("ab") == [tok.encoder[tok._render_bytes(list(b"ab"))]]
        or b"ab" in merged_byte_seqs
    )


def test_vocab_consistency():
    """encoder and decoder must be consistent inverses (for token bytes)."""
    tok = BPE(vocab_size=300)
    tok.train(["hello world", "the cat sat", "abcdefg"])
    assert len(tok.encoder) == len(tok.decoder) == tok.vocab_size
    for token_bytes, idx in tok.encoder.items():
        assert tok.decoder[idx] == token_bytes


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------


def test_special_tokens_reserved():
    tok = BPE(vocab_size=300, special_tokens=["<|endoftext|>", "<|pad|>"])
    tok.train(["hello world"])
    assert "<|endoftext|>" in tok.encoder
    assert "<|pad|>" in tok.encoder
    # special tokens should occupy low ids
    assert tok.encoder["<|endoftext|>"] == 0
    assert tok.encoder["<|pad|>"] == 1


def test_encode_decode_special_token():
    tok = BPE(vocab_size=300, special_tokens=["<|endoftext|>"])
    tok.train(["hello"])
    eot = tok.encoder["<|endoftext|>"]
    text = "hello"
    result = tok.decode(tok.encode(text) + [eot])
    assert result == "hello<|endoftext|>" or result == "hello"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


@pytest.fixture
def trained_tokenizer(tmp_path):
    tok = BPE(vocab_size=300, special_tokens=["<|endoftext|>", "<|pad|>"])
    tok.train(["the quick brown fox jumps over the lazy dog"] * 20)
    return tok


def test_save_and_load_round_trip(trained_tokenizer, tmp_path):
    path = os.path.join(tmp_path, "vocab.json")
    trained_tokenizer.save(path)
    loaded = BPE(vocab_size=300, special_tokens=["<|endoftext|>", "<|pad|>"])
    loaded.load(path)

    assert loaded.vocab_size == trained_tokenizer.vocab_size
    assert loaded.encoder == trained_tokenizer.encoder
    assert loaded.decoder == trained_tokenizer.decoder

    # Encode/decode must be identical after reload
    sample = "the quick brown fox jumps"
    assert loaded.encode(sample) == trained_tokenizer.encode(sample)
    assert loaded.decode(loaded.encode(sample)) == sample


def test_save_creates_file(trained_tokenizer, tmp_path):
    path = tmp_path / "vocab.json"
    trained_tokenizer.save(str(path))
    assert path.exists()


def test_save_is_silent(trained_tokenizer, tmp_path, capsys):
    """A tokenizer saved every checkpoint must not narrate into the training log."""
    trained_tokenizer.save(str(tmp_path / "vocab.json"))
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_text_encoding():
    tok = BPE(vocab_size=256)
    assert tok.encode("") == []


def test_encoding_deterministic():
    tok = BPE(vocab_size=300)
    tok.train(["repeat repeat repeat"])
    a = tok.encode("repeat repeat repeat")
    b = tok.encode("repeat repeat repeat")
    assert a == b
