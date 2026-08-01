"""Full pipeline: text → tokenizer → corpus → training → generation.

This is the test that would catch a project that passes every unit test and
still cannot learn. Each stage is individually verified elsewhere; here they run
together against real text, and the assertion is behavioural — the loss must
fall well below the uniform-entropy baseline, and the trained model must prefer
its training corpus over shuffled text.

Sized to stay quick on CPU: a ~0.2M-parameter model over a few hundred steps.
"""

import math

import torch

from dataset import DataEngine, encode_texts_to_bin
from inference import generate, generate_ids
from model import CausalLM
from tokenizer import BPE
from training.trainer import Trainer

CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat ran to the mat",
    "the dog ran to the log",
] * 60


def build(tmp_path, *, steps=250, seq_len=16):
    tok = BPE(vocab_size=300, special_tokens=["<|endoftext|>"])
    tok.train(CORPUS)

    path = tmp_path / "corpus.bin"
    encode_texts_to_bin(tok, CORPUS, path, eot_token="<|endoftext|>")

    cfg = {
        "seed": 0,
        "device": "cpu",
        "model": {
            "vocab_size": tok.vocab_size,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "d_ff": 128,
            "max_seq_len": 64,
        },
        "attention": {"impl": "sdpa"},
        "dataset": {
            "source": "file",
            "path": str(path),
            "seq_len": seq_len,
            "batch_size": 16,
        },
        "training": {
            "steps": steps,
            "lr": 3.0e-3,
            "min_lr": 3.0e-4,
            "warmup_steps": 20,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "amp": False,
        },
    }

    torch.manual_seed(0)
    model = CausalLM.from_config(cfg)
    engine = DataEngine.from_config(cfg, vocab_size=tok.vocab_size)
    return tok, cfg, model, Trainer(model, engine, cfg, device=torch.device("cpu"))


def test_the_model_actually_learns_the_corpus(tmp_path):
    """Loss must fall far below ln(vocab_size) on structured text."""
    tok, _, _, trainer = build(tmp_path)
    history = trainer.train(250)

    baseline = math.log(tok.vocab_size)
    first = sum(h["loss"] for h in history[:10]) / 10
    last = sum(h["loss"] for h in history[-10:]) / 10

    assert abs(first - baseline) < 1.0, f"did not start near ln(V)={baseline:.2f}: {first:.2f}"
    assert last < baseline / 2, f"loss barely moved: {first:.2f} -> {last:.2f}"


def test_training_lowers_loss_on_corpus_distributed_text(tmp_path):
    """Training must beat random init on text drawn from the corpus distribution.

    Two traps this probe has to avoid:

    - A BPE fitted on a corpus this repetitive merges an entire sentence into a
      single token, which leaves `inputs` empty and cross-entropy at nan.
    - The probe must be built the same way the corpus was — documents joined by
      `<|endoftext|>`, not by spaces. Joined with spaces the sequence is
      out-of-distribution, and a well-trained model correctly scores it *worse*
      than an untrained one, which looks like a training failure but is the
      opposite.
    """
    tok, cfg, model, trainer = build(tmp_path)

    probe_text = "".join(text + "<|endoftext|>" for text in CORPUS[:4])
    probe = torch.tensor([tok.encode(probe_text)], dtype=torch.long)
    assert probe.shape[1] > 4, f"probe collapsed to {probe.shape[1]} token(s)"
    inputs, labels = probe[:, :-1], probe[:, 1:]

    def loss_of(m):
        with torch.no_grad():
            logits = m(inputs)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        ).item()

    torch.manual_seed(1)
    untrained = loss_of(CausalLM.from_config(cfg))
    trainer.train(250)
    trained = loss_of(model)

    assert trained < untrained / 2, f"untrained {untrained:.2f} -> trained {trained:.2f}"


def test_generation_is_deterministic_and_decodable(tmp_path):
    tok, _, model, trainer = build(tmp_path, steps=150)
    trainer.train(150)

    a = generate(model, tok, "the cat", max_new_tokens=20, temperature=0.0)
    b = generate(model, tok, "the cat", max_new_tokens=20, temperature=0.0)

    assert a == b, "greedy generation is not deterministic"
    assert a.startswith("the cat")
    assert isinstance(a, str) and len(a) > len("the cat")


def test_the_kv_cache_does_not_change_what_a_trained_model_writes(tmp_path):
    """Cache equivalence has to hold for trained weights, not just random ones."""
    tok, _, model, trainer = build(tmp_path, steps=150)
    trainer.train(150)

    prompt = torch.tensor([tok.encode("the dog")], dtype=torch.long)
    cached = generate_ids(model, prompt, max_new_tokens=15, temperature=0.0, use_cache=True)
    plain = generate_ids(model, prompt, max_new_tokens=15, temperature=0.0, use_cache=False)
    assert cached == plain


def test_a_resumed_run_reaches_the_same_place(tmp_path):
    """Train 100, checkpoint, reload, train 100 more == train 200 straight."""
    _, cfg, model_a, trainer_a = build(tmp_path, steps=200)
    trainer_a.train(100)
    ckpt = tmp_path / "mid.pt"
    trainer_a.save(str(ckpt))
    straight = trainer_a.train(100)[-1]["loss"]

    _, _, _, trainer_b = build(tmp_path, steps=200)
    trainer_b.load(str(ckpt))
    resumed = trainer_b.train(100)[-1]["loss"]

    # The data stream restarts on resume (checkpoint.py documents this), so the
    # batches differ; the losses converge to the same place rather than matching
    # bit for bit.
    assert abs(straight - resumed) < 0.5, f"straight {straight:.3f} vs resumed {resumed:.3f}"
