"""Window readers over a ``uint16`` token corpus.

Two datasets, one item contract. Both yield
``{"input_ids": LongTensor[T], "labels": LongTensor[T]}`` where ``labels`` is
``input_ids`` shifted one position left.

The shift is done at *window* level: a window reads ``seq_len + 1`` ids and
splits them, rather than reading ``seq_len`` and shifting inside. That costs one
extra token of overlap per window and buys a real label for the final position
instead of padding -- with a 512-token window, shifting inside would silently
discard 1/512 of the training signal and leave one position learning from a fake
target.

Which class to use:

- ``MemmapTokenDataset`` (map-style) -- a fixed corpus on disk. Supports
  ``len()`` and random access, so ``DataLoader(shuffle=True)`` can shuffle
  window order for free. The default (TDR-004).
- ``StreamingTokenDataset`` (iterable) -- sequential passes and worker-sharded
  reads, for corpora where random access is undesirable.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from dataset.binfile import TOKEN_DTYPE, count_tokens_bin

Item = dict[str, torch.Tensor]


def _n_windows(n_tokens: int, seq_len: int) -> int:
    """Whole windows available. Each consumes seq_len+1 ids (T inputs + 1 label)."""
    return max(0, (n_tokens - 1) // seq_len)


class _MemmapReader:
    """Shared lazy-memmap machinery for both dataset classes.

    The memmap handle is opened on first use and *dropped when pickled*. That
    matters: ``DataLoader(num_workers>0)`` uses ``spawn`` on macOS, which pickles
    the dataset into each worker. A live ``np.memmap`` pickles as a plain
    ``ndarray`` -- i.e. it would serialise the entire corpus through a pipe into
    every worker, which is precisely the bulk copy mmap exists to avoid. Opening
    lazily per process gives each worker its own independent handle, as intended
    by TDR-004.
    """

    def __init__(self, path: str | os.PathLike[str], seq_len: int) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        self.path = Path(path)
        self.seq_len = int(seq_len)
        self.n_tokens = count_tokens_bin(self.path)
        self.n_windows = _n_windows(self.n_tokens, self.seq_len)
        if self.n_windows < 1:
            raise ValueError(
                f"corpus too small: {self.path} holds {self.n_tokens} tokens but a "
                f"window of seq_len={self.seq_len} needs {self.seq_len + 1}"
            )
        self._tokens: np.memmap | None = None

    @property
    def tokens(self) -> np.memmap:
        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        return self._tokens

    def window(self, index: int) -> Item:
        """Read window ``index`` and split it into inputs and shifted labels."""
        start = index * self.seq_len
        # Copy out of the mmap: the returned tensors outlive this call and must
        # not alias page-cache memory that a later window may remap.
        chunk = np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64)
        return {
            "input_ids": torch.from_numpy(chunk[:-1].copy()),
            "labels": torch.from_numpy(chunk[1:].copy()),
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_tokens"] = None  # never ship an open handle across a process boundary
        return state


class MemmapTokenDataset(_MemmapReader, Dataset[Item]):
    """Map-style, memory-mapped view over a token corpus."""

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, index: int) -> Item:
        if index < 0:
            index += self.n_windows
        if not 0 <= index < self.n_windows:
            raise IndexError(f"window index out of range: {index} (have {self.n_windows})")
        return self.window(index)


class StreamingTokenDataset(_MemmapReader, IterableDataset[Item]):
    """Iterable, worker-sharded view over a token corpus.

    Sharding is strided (``order[worker_id::num_workers]``) so every window is
    emitted by exactly one worker -- duplicated windows would quietly inflate the
    effective epoch and bias training toward whatever the overlap contains.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        seq_len: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(path, seq_len)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[Item]:
        order = np.arange(self.n_windows)
        if self.shuffle:
            # Reseeded per epoch so successive passes differ, but reproducibly.
            np.random.default_rng([self.seed, self.epoch]).shuffle(order)
        self.epoch += 1

        info = get_worker_info()
        if info is not None:
            order = order[info.id :: info.num_workers]

        for index in order:
            yield self.window(int(index))


class SyntheticTokenDataset(IterableDataset[Item]):
    """Endless uniform-random token stream -- a corpus-free smoke-test source.

    There is no signal to learn here, so loss should sit near ``ln(vocab_size)``
    and stay there. That makes it a useful control: a model that "improves" on
    synthetic data has a leak (labels in the inputs, or a shifted mask).
    """

    def __init__(self, vocab_size: int, seq_len: int, *, seed: int = 0) -> None:
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self) -> Iterator[Item]:
        info = get_worker_info()
        # Offset the seed per worker, or every worker emits the identical stream.
        generator = torch.Generator().manual_seed(self.seed + (info.id if info else 0))
        while True:
            chunk = torch.randint(
                0, self.vocab_size, (self.seq_len + 1,), generator=generator, dtype=torch.long
            )
            yield {"input_ids": chunk[:-1], "labels": chunk[1:]}
