"""Dataset engine unit tests (TDD - written before implementation).

Exit criteria for build stage 1 (docs/BUILD_ORDER.md): shapes, dtype, and
shift-correctness. The shift test is the one that actually matters -- an
off-by-one in the label window trains the model to predict the *current* token,
which still produces a falling loss curve and looks fine on a dashboard.
"""

import numpy as np
import pytest
import torch

from dataset import (
    TOKEN_DTYPE,
    DataEngine,
    MemmapTokenDataset,
    StreamingTokenDataset,
    read_tokens_bin,
    write_tokens_bin,
)

# ---------------------------------------------------------------------------
# Binary token file
# ---------------------------------------------------------------------------


def test_write_then_read_round_trip(tmp_path):
    ids = [0, 1, 2, 65535, 300, 42]
    path = tmp_path / "toks.bin"
    n = write_tokens_bin(ids, path)
    assert n == len(ids)
    assert read_tokens_bin(path).tolist() == ids


def test_bin_file_is_uint16_on_disk(tmp_path):
    path = tmp_path / "toks.bin"
    write_tokens_bin(range(100), path)
    # 2 bytes per token, no header -- the format is a bare flat array so any
    # tool (np.memmap, torch.from_file, xxd) can read it.
    assert path.stat().st_size == 100 * np.dtype(TOKEN_DTYPE).itemsize
    assert np.dtype(TOKEN_DTYPE).itemsize == 2


def test_token_id_too_large_for_uint16_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="uint16"):
        write_tokens_bin([1, 2, 65536], tmp_path / "bad.bin")


def test_negative_token_id_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="uint16"):
        write_tokens_bin([1, -1], tmp_path / "bad.bin")


# ---------------------------------------------------------------------------
# Memory-mapped map-style dataset
# ---------------------------------------------------------------------------


@pytest.fixture
def corpus_bin(tmp_path):
    """A 100-token corpus whose value equals its index -- makes shifts visible."""
    path = tmp_path / "corpus.bin"
    write_tokens_bin(range(100), path)
    return path


def test_memmap_window_count(corpus_bin):
    # Each window consumes seq_len+1 ids (T inputs + the final label), so with
    # 100 tokens and seq_len 10 there are floor((100-1)/10) = 9 whole windows.
    ds = MemmapTokenDataset(corpus_bin, seq_len=10)
    assert len(ds) == 9


def test_memmap_item_shapes_and_dtype(corpus_bin):
    ds = MemmapTokenDataset(corpus_bin, seq_len=10)
    item = ds[0]
    assert set(item) == {"input_ids", "labels"}
    assert item["input_ids"].shape == (10,)
    assert item["labels"].shape == (10,)
    assert item["input_ids"].dtype == torch.long
    assert item["labels"].dtype == torch.long


def test_memmap_labels_are_inputs_shifted_by_one(corpus_bin):
    ds = MemmapTokenDataset(corpus_bin, seq_len=10)
    for i in (0, 1, len(ds) - 1):
        item = ds[i]
        # labels[t] must be the token that *follows* input_ids[t]
        assert torch.equal(item["labels"][:-1], item["input_ids"][1:])
        # and it must be the real next token from the corpus, not padding
        assert item["labels"][-1].item() == item["input_ids"][-1].item() + 1


def test_memmap_windows_are_contiguous_and_non_overlapping(corpus_bin):
    ds = MemmapTokenDataset(corpus_bin, seq_len=10)
    starts = [ds[i]["input_ids"][0].item() for i in range(len(ds))]
    assert starts == [0, 10, 20, 30, 40, 50, 60, 70, 80]


def test_memmap_last_window_stays_in_bounds(corpus_bin):
    ds = MemmapTokenDataset(corpus_bin, seq_len=10)
    last = ds[len(ds) - 1]
    # Window 8 reads tokens[80:91], so the final label is token 90. A read past
    # the end would come back as 0 from the memmap rather than raising, so the
    # assertion is on the exact value, not just "no exception".
    assert last["input_ids"][0].item() == 80
    assert last["labels"][-1].item() == 90
    assert last["labels"].max().item() < ds.n_tokens


def test_memmap_discards_the_trailing_partial_window(corpus_bin):
    """100 tokens / seq_len 10 uses 91 and drops 9 -- intentional, not a bug.

    Windows are non-overlapping, so a remainder shorter than seq_len+1 has no
    home. Pinned here so a future change to window striding has to say so.
    """
    ds = MemmapTokenDataset(corpus_bin, seq_len=10)
    consumed = len(ds) * ds.seq_len + 1
    assert consumed == 91
    assert ds.n_tokens - consumed == 9


def test_memmap_negative_index_works_like_a_sequence(corpus_bin):
    ds = MemmapTokenDataset(corpus_bin, seq_len=10)
    assert torch.equal(ds[-1]["input_ids"], ds[len(ds) - 1]["input_ids"])


def test_memmap_out_of_range_index_raises(corpus_bin):
    ds = MemmapTokenDataset(corpus_bin, seq_len=10)
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_memmap_rejects_corpus_too_small_for_one_window(tmp_path):
    path = tmp_path / "tiny.bin"
    write_tokens_bin(range(5), path)
    with pytest.raises(ValueError, match="too small"):
        MemmapTokenDataset(path, seq_len=10)


def test_memmap_does_not_load_corpus_into_memory(corpus_bin):
    """The whole point of TDR-004: the reader is a view, not a copy."""
    ds = MemmapTokenDataset(corpus_bin, seq_len=10)
    assert isinstance(ds.tokens, np.memmap)


# ---------------------------------------------------------------------------
# Streaming iterable dataset
# ---------------------------------------------------------------------------


def test_streaming_yields_same_windows_as_memmap(corpus_bin):
    mm = MemmapTokenDataset(corpus_bin, seq_len=10)
    st = StreamingTokenDataset(corpus_bin, seq_len=10)
    streamed = list(st)
    assert len(streamed) == len(mm)
    for a, b in zip(streamed, (mm[i] for i in range(len(mm))), strict=True):
        assert torch.equal(a["input_ids"], b["input_ids"])
        assert torch.equal(a["labels"], b["labels"])


def test_streaming_labels_are_shifted(corpus_bin):
    for item in StreamingTokenDataset(corpus_bin, seq_len=10):
        assert torch.equal(item["labels"][:-1], item["input_ids"][1:])


def test_streaming_is_re_iterable(corpus_bin):
    """A fresh __iter__ must restart -- a generator-as-self bug shows up here."""
    st = StreamingTokenDataset(corpus_bin, seq_len=10)
    assert len(list(st)) == len(list(st)) == 9


def test_streaming_shuffle_changes_order_but_not_content(corpus_bin):
    plain = [x["input_ids"][0].item() for x in StreamingTokenDataset(corpus_bin, seq_len=10)]
    shuf = [
        x["input_ids"][0].item()
        for x in StreamingTokenDataset(corpus_bin, seq_len=10, shuffle=True, seed=0)
    ]
    assert sorted(shuf) == sorted(plain)
    assert shuf != plain


def test_streaming_shuffle_is_seed_deterministic(corpus_bin):
    def starts(seed):
        ds = StreamingTokenDataset(corpus_bin, seq_len=10, shuffle=True, seed=seed)
        return [x["input_ids"][0].item() for x in ds]

    assert starts(7) == starts(7)
    assert starts(7) != starts(8)


def test_streaming_workers_shard_without_duplicates(corpus_bin):
    """Each worker must own a disjoint slice, or the model sees duplicate data."""
    from torch.utils.data import DataLoader

    ds = StreamingTokenDataset(corpus_bin, seq_len=10)
    loader = DataLoader(ds, batch_size=1, num_workers=2)
    starts = [b["input_ids"][0, 0].item() for b in loader]
    assert sorted(starts) == [0, 10, 20, 30, 40, 50, 60, 70, 80]


# ---------------------------------------------------------------------------
# Synthetic source (no corpus file needed)
# ---------------------------------------------------------------------------


def test_synthetic_engine_needs_no_file():
    eng = DataEngine.from_config(
        {"dataset": {"source": "synthetic", "seq_len": 8, "batch_size": 4}},
        vocab_size=50,
    )
    batch = eng.next_batch()
    assert batch["input_ids"].shape == (4, 8)
    assert batch["labels"].shape == (4, 8)


def test_synthetic_tokens_are_inside_the_vocab():
    eng = DataEngine.from_config(
        {"dataset": {"source": "synthetic", "seq_len": 8, "batch_size": 4}, "seed": 1},
        vocab_size=50,
    )
    for _ in range(5):
        batch = eng.next_batch()
        assert batch["input_ids"].min() >= 0
        assert batch["input_ids"].max() < 50
        assert batch["labels"].max() < 50


def test_synthetic_is_seed_deterministic():
    def first(seed):
        eng = DataEngine.from_config(
            {"dataset": {"source": "synthetic", "seq_len": 8, "batch_size": 4}, "seed": seed},
            vocab_size=50,
        )
        return eng.next_batch()["input_ids"]

    assert torch.equal(first(3), first(3))
    assert not torch.equal(first(3), first(4))


# ---------------------------------------------------------------------------
# DataEngine
# ---------------------------------------------------------------------------


def test_engine_batches_from_a_file(corpus_bin):
    eng = DataEngine.from_config(
        {"dataset": {"source": "file", "path": str(corpus_bin), "seq_len": 10, "batch_size": 3}},
        vocab_size=100,
    )
    batch = eng.next_batch()
    assert batch["input_ids"].shape == (3, 10)
    assert torch.equal(batch["labels"][:, :-1], batch["input_ids"][:, 1:])


def test_engine_wraps_around_forever(corpus_bin):
    """next_batch() is an infinite stream -- training loops count steps, not epochs."""
    eng = DataEngine.from_config(
        {"dataset": {"source": "file", "path": str(corpus_bin), "seq_len": 10, "batch_size": 4}},
        vocab_size=100,
    )
    # 9 windows / batch 4 = 2 batches per epoch; ask for well past that
    for _ in range(10):
        assert eng.next_batch()["input_ids"].shape == (4, 10)


def test_engine_moves_batch_to_device(corpus_bin):
    eng = DataEngine.from_config(
        {"dataset": {"source": "file", "path": str(corpus_bin), "seq_len": 10, "batch_size": 2}},
        vocab_size=100,
    )
    batch = eng.next_batch(device="cpu")
    assert batch["input_ids"].device.type == "cpu"


def test_a_split_too_small_for_one_batch_fails_clearly(corpus_bin):
    """drop_last turns "fewer windows than batch_size" into *zero* batches.

    Without a guard this surfaces as a bare StopIteration from inside torch's
    sampler, which says nothing about the cause. Most often hit by a small
    validation split.
    """
    eng = DataEngine.from_config(
        {"dataset": {"source": "file", "path": str(corpus_bin), "seq_len": 10, "batch_size": 99}},
        vocab_size=100,
    )
    with pytest.raises(RuntimeError, match="no batches"):
        eng.next_batch()


def test_drop_last_false_allows_a_short_final_batch(corpus_bin):
    eng = DataEngine.from_config(
        {
            "dataset": {
                "source": "file",
                "path": str(corpus_bin),
                "seq_len": 10,
                "batch_size": 99,
                "drop_last": False,
            }
        },
        vocab_size=100,
    )
    assert eng.next_batch()["input_ids"].shape[0] == 9  # all available windows


def test_engine_rejects_unknown_source():
    with pytest.raises(ValueError, match="source"):
        DataEngine.from_config({"dataset": {"source": "nope", "seq_len": 8}}, vocab_size=50)


def test_engine_requires_path_for_file_source():
    with pytest.raises(ValueError, match="path"):
        DataEngine.from_config({"dataset": {"source": "file", "seq_len": 8}}, vocab_size=50)


def test_engine_rejects_prefetch_without_workers(corpus_bin):
    """torch raises on prefetch_factor with num_workers=0; we must not pass it."""
    eng = DataEngine.from_config(
        {
            "dataset": {
                "source": "file",
                "path": str(corpus_bin),
                "seq_len": 10,
                "batch_size": 2,
                "num_workers": 0,
                "prefetch_factor": 4,
            }
        },
        vocab_size=100,
    )
    assert eng.next_batch()["input_ids"].shape == (2, 10)
