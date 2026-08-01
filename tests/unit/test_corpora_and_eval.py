"""Corpus splitting, validation perplexity, and batched generation tests.

The document-split tests exist for one reason: a token-level split leaks. Windows
are contiguous slices of a flat array, so cutting a tokenised stream in half puts
validation tokens inside training windows and the reported perplexity comes out
quietly too good. Splitting documents first makes that structurally impossible,
and `test_no_document_appears_in_both_splits` pins it.
"""

import math

import pytest
import torch

from dataset import CORPORA, read_documents, split_documents
from inference import generate_batch, generate_ids, left_pad, sample_batch
from model import CausalLM
from training.trainer import Trainer

VOCAB = 64
DOCS = [f"document number {i} with some words in it" for i in range(200)]


def make_cfg(**training):
    base = {
        "steps": 50,
        "lr": 1e-3,
        "min_lr": 1e-4,
        "warmup_steps": 0,
        "grad_clip": 1.0,
        "amp": False,
        "weight_decay": 0.0,
    }
    base.update(training)
    return {
        "model": {
            "vocab_size": VOCAB,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "d_ff": 64,
            "max_seq_len": 64,
        },
        "attention": {"impl": "manual"},
        "training": base,
        "device": "cpu",
    }


def make_model():
    torch.manual_seed(0)
    return CausalLM.from_config(make_cfg()).eval()


class ConstantEngine:
    def __init__(self, loss_seed=0):
        torch.manual_seed(loss_seed)
        ids = torch.randint(0, VOCAB, (4, 17))
        self.batch = {"input_ids": ids[:, :-1], "labels": ids[:, 1:]}

    def next_batch(self, device=None):
        return self.batch


# ---------------------------------------------------------------------------
# Document splitting
# ---------------------------------------------------------------------------


def test_split_is_disjoint_and_complete():
    train, val = split_documents(DOCS, val_fraction=0.2)
    assert set(train).isdisjoint(val)
    assert len(train) + len(val) == len(DOCS)


def test_no_document_appears_in_both_splits():
    train, val = split_documents(DOCS, val_fraction=0.1)
    assert not (set(train) & set(val))


def test_split_is_reproducible_without_a_stored_seed():
    """Content-hashed, so the same corpus always splits the same way."""
    assert split_documents(DOCS, val_fraction=0.1) == split_documents(DOCS, val_fraction=0.1)


def test_reordering_the_corpus_does_not_move_documents():
    """A shuffle-based split would reassign sides and leak across a resume."""
    train_a, val_a = split_documents(DOCS, val_fraction=0.2)
    train_b, val_b = split_documents(list(reversed(DOCS)), val_fraction=0.2)
    assert set(train_a) == set(train_b)
    assert set(val_a) == set(val_b)


def test_adding_documents_leaves_existing_sides_untouched():
    train_a, val_a = split_documents(DOCS, val_fraction=0.2)
    extended = DOCS + [f"brand new document {i}" for i in range(50)]
    train_b, val_b = split_documents(extended, val_fraction=0.2)
    assert set(train_a) <= set(train_b)
    assert set(val_a) <= set(val_b)


def test_split_fraction_is_roughly_honoured():
    _, val = split_documents(DOCS, val_fraction=0.25)
    assert 0.15 < len(val) / len(DOCS) < 0.35


def test_a_seed_change_produces_a_different_split():
    _, val_a = split_documents(DOCS, val_fraction=0.2, seed=0)
    _, val_b = split_documents(DOCS, val_fraction=0.2, seed=1)
    assert set(val_a) != set(val_b)


def test_degenerate_fractions_are_rejected():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="val_fraction"):
            split_documents(DOCS, val_fraction=bad)


def test_a_corpus_too_small_to_split_raises():
    with pytest.raises(ValueError, match="empty side"):
        split_documents(["only one document"], val_fraction=0.5)


def test_read_documents_drops_blanks(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("one\n\n\n\ntwo\n\nthree\n", encoding="utf-8")
    assert read_documents(path) == ["one", "two", "three"]


def test_out_dir_accepts_a_plain_string():
    """The natural call passes a str; `str / "train.bin"` is a TypeError."""
    import tempfile

    from dataset import prepare_documents
    from tokenizer import BPE

    tok = BPE(vocab_size=300)
    tok.train(DOCS)
    with tempfile.TemporaryDirectory() as tmp:
        info = prepare_documents(DOCS, tok, out_dir=tmp)  # str, not Path
        assert info["train_tokens"] > 0


def test_every_corpus_declares_a_licence():
    """ "Where did the training data come from" is unpleasant to answer late."""
    for name, corpus in CORPORA.items():
        assert corpus.licence, f"{name} has no licence recorded"
        assert corpus.url.startswith("https://"), f"{name} is not fetched over TLS"


# ---------------------------------------------------------------------------
# Validation perplexity
# ---------------------------------------------------------------------------


def test_evaluate_reports_loss_and_perplexity():
    cfg = make_cfg()
    trainer = Trainer(CausalLM.from_config(cfg), ConstantEngine(), cfg, device=torch.device("cpu"))
    metrics = trainer.evaluate(ConstantEngine(1), batches=3)
    assert set(metrics) == {"val_loss", "val_perplexity"}
    assert metrics["val_perplexity"] == pytest.approx(math.exp(metrics["val_loss"]), rel=1e-6)


def test_untrained_validation_perplexity_is_near_the_vocabulary_size():
    cfg = make_cfg()
    trainer = Trainer(CausalLM.from_config(cfg), ConstantEngine(), cfg, device=torch.device("cpu"))
    ppl = trainer.evaluate(ConstantEngine(1), batches=5)["val_perplexity"]
    assert VOCAB * 0.6 < ppl < VOCAB * 1.6, f"expected ~{VOCAB}, got {ppl:.1f}"


def test_evaluate_does_not_leave_the_model_in_eval_mode():
    """Called mid-loop, this must not silently disable dropout for the rest of training."""
    cfg = make_cfg()
    trainer = Trainer(CausalLM.from_config(cfg), ConstantEngine(), cfg, device=torch.device("cpu"))
    trainer.model.train()
    trainer.evaluate(ConstantEngine(1), batches=2)
    assert trainer.model.training


def test_evaluate_allocates_no_gradients():
    cfg = make_cfg()
    trainer = Trainer(CausalLM.from_config(cfg), ConstantEngine(), cfg, device=torch.device("cpu"))
    trainer.evaluate(ConstantEngine(1), batches=2)
    assert all(p.grad is None for p in trainer.model.parameters())


def test_training_lowers_validation_loss_on_the_same_distribution():
    cfg = make_cfg(steps=150, lr=3e-3, warmup_steps=10)
    engine = ConstantEngine()
    trainer = Trainer(CausalLM.from_config(cfg), engine, cfg, device=torch.device("cpu"))
    before = trainer.evaluate(engine, batches=2)["val_loss"]
    trainer.train(150)
    after = trainer.evaluate(engine, batches=2)["val_loss"]
    assert after < before / 2, f"{before:.3f} -> {after:.3f}"


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------


def test_left_padding_puts_real_tokens_last():
    """The next token is predicted from index -1; right padding would break that."""
    ids, mask = left_pad([[1, 2, 3], [9]], pad_id=0)
    assert ids.tolist() == [[1, 2, 3], [0, 0, 9]]
    assert mask.tolist() == [[True, True, True], [False, False, True]]


def test_left_pad_rejects_empty_input():
    with pytest.raises(ValueError, match="no sequences"):
        left_pad([], pad_id=0)
    with pytest.raises(ValueError, match="empty prompts"):
        left_pad([[], []], pad_id=0)


def test_sample_batch_returns_one_token_per_row():
    logits = torch.randn(5, VOCAB)
    out = sample_batch(logits, temperature=0.0)
    assert out.shape == (5,)
    assert torch.equal(out, logits.argmax(dim=-1))


def test_sample_batch_rejects_a_single_row_vector():
    with pytest.raises(ValueError, match=r"\[batch, vocab\]"):
        sample_batch(torch.randn(VOCAB))


def test_batched_greedy_matches_single_stream_for_equal_lengths():
    """With no padding involved the two paths must agree exactly."""
    model = make_model()
    prompts = [[3, 9, 14], [7, 2, 55]]
    batched = generate_batch(model, prompts, max_new_tokens=8, temperature=0.0)
    single = [
        generate_ids(model, torch.tensor([p]), max_new_tokens=8, temperature=0.0) for p in prompts
    ]
    assert batched == single


def test_batched_generation_returns_one_continuation_per_prompt():
    out = generate_batch(make_model(), [[1, 2], [3], [4, 5, 6]], max_new_tokens=5, temperature=0.0)
    assert len(out) == 3
    assert all(len(c) == 5 for c in out)


def test_a_finished_row_stops_growing():
    model = make_model()
    greedy = generate_batch(model, [[3, 9, 14]], max_new_tokens=10, temperature=0.0)[0]
    stop = greedy[2]
    out = generate_batch(model, [[3, 9, 14]], max_new_tokens=10, temperature=0.0, stop_id=stop)[0]
    assert out == greedy[: greedy.index(stop) + 1]


def test_empty_prompt_list_returns_empty():
    assert generate_batch(make_model(), [], max_new_tokens=4) == []


def test_batched_generation_respects_max_seq_len():
    model = make_model()
    out = generate_batch(model, [[1, 2, 3]], max_new_tokens=1000, temperature=0.0)
    assert 0 < len(out[0]) <= model.max_seq_len
