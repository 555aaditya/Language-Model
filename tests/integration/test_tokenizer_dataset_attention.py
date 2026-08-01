"""Integration: raw text → tokenizer → .bin corpus → batches → attention.

Unit tests pin each module against its own contract; this pins the *seams*
between them, which is where the contracts actually get violated:

- token ids the tokenizer emits must fit the dataset's ``uint16`` format
- batch tensors must be the dtype/shape an ``nn.Embedding`` will accept
- the next-token shift must survive the round trip through disk
"""

import torch
from torch import nn

from attention import CausalAttention
from dataset import DataEngine, count_tokens_bin, encode_texts_to_bin
from tokenizer import BPE

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog sleeps while the quick fox runs",
    "a fox and a dog met under the brown oak tree",
] * 20


def build_tokenizer():
    tok = BPE(vocab_size=400, special_tokens=["<|endoftext|>"])
    tok.train(CORPUS)
    return tok


def test_tokenizer_output_fits_the_on_disk_token_format(tmp_path):
    tok = build_tokenizer()
    path = tmp_path / "corpus.bin"
    n = encode_texts_to_bin(tok, CORPUS, path, eot_token="<|endoftext|>")
    assert n == count_tokens_bin(path) > 0
    # uint16 caps the vocab at 65535; a tokenizer that outgrew it must fail loudly
    # at write time, not corrupt ids by wrapping.
    assert tok.vocab_size <= 65536


def test_corpus_round_trips_through_disk(tmp_path):
    """Decoding the .bin must reproduce the original text, EOT markers and all."""
    tok = build_tokenizer()
    path = tmp_path / "corpus.bin"
    encode_texts_to_bin(tok, CORPUS[:2], path, eot_token="<|endoftext|>")

    from dataset import read_tokens_bin

    decoded = tok.decode(read_tokens_bin(path).tolist())
    assert decoded == "".join(text + "<|endoftext|>" for text in CORPUS[:2])


def test_batches_feed_an_embedding_and_then_attention(tmp_path):
    """The real training path, minus the loss: ids → embedding → attention."""
    tok = build_tokenizer()
    path = tmp_path / "corpus.bin"
    encode_texts_to_bin(tok, CORPUS, path, eot_token="<|endoftext|>")

    d_model, seq_len, batch_size = 32, 16, 4
    engine = DataEngine.from_config(
        {
            "dataset": {
                "source": "file",
                "path": str(path),
                "seq_len": seq_len,
                "batch_size": batch_size,
            },
            "seed": 0,
        },
        vocab_size=tok.vocab_size,
    )
    batch = engine.next_batch()

    # Contract with nn.Embedding: long dtype, every id inside the vocabulary.
    assert batch["input_ids"].dtype == torch.long
    assert int(batch["input_ids"].max()) < tok.vocab_size

    embed = nn.Embedding(tok.vocab_size, d_model)
    attn = CausalAttention(
        d_model=d_model, n_heads=4, n_kv_heads=2, max_seq_len=seq_len, impl="sdpa"
    ).eval()

    out, _ = attn(embed(batch["input_ids"]))
    assert out.shape == (batch_size, seq_len, d_model)
    assert torch.isfinite(out).all()


def test_shift_invariant_survives_the_whole_pipeline(tmp_path):
    """labels[t] == input_ids[t+1] must still hold after tokenize → disk → batch."""
    tok = build_tokenizer()
    path = tmp_path / "corpus.bin"
    encode_texts_to_bin(tok, CORPUS, path, eot_token="<|endoftext|>")

    engine = DataEngine.from_config(
        {"dataset": {"source": "file", "path": str(path), "seq_len": 16, "batch_size": 4}},
        vocab_size=tok.vocab_size,
    )
    batch = engine.next_batch()
    assert torch.equal(batch["labels"][:, :-1], batch["input_ids"][:, 1:])


def test_generation_step_shapes_hold_end_to_end(tmp_path):
    """Prefill a real prompt, then decode one token with the cache."""
    tok = build_tokenizer()
    d_model = 32
    embed = nn.Embedding(tok.vocab_size, d_model)
    attn = CausalAttention(
        d_model=d_model, n_heads=4, n_kv_heads=2, max_seq_len=128, impl="sdpa"
    ).eval()

    ids = torch.tensor([tok.encode("the quick brown fox")], dtype=torch.long)
    prompt_len = ids.shape[1]

    out, cache = attn(embed(ids), use_cache=True)
    assert out.shape == (1, prompt_len, d_model)
    assert len(cache) == prompt_len

    next_id = torch.tensor([[tok.encode(" dog")[0]]], dtype=torch.long)
    step, cache = attn(embed(next_id), kv_cache=cache, use_cache=True)
    assert step.shape == (1, 1, d_model)
    assert len(cache) == prompt_len + 1
