"""``DataEngine`` -- config in, endless device-ready batches out.

The training loop counts *steps*, not epochs, so ``next_batch()`` never raises
``StopIteration``: a finite dataset simply wraps around. Everything else here is
DataLoader plumbing whose defaults are chosen to avoid three specific traps:

- ``prefetch_factor`` is illegal when ``num_workers == 0`` (torch raises), so it
  is only passed when workers exist.
- ``pin_memory`` allocates CUDA page-locked host memory. With no CUDA device it
  is at best a no-op and at worst a warning on every batch, so it defaults to
  ``torch.cuda.is_available()`` rather than ``True``. Apple MPS reads from
  unified memory and gains nothing from pinning (TDR-019).
- ``shuffle`` is rejected outright by DataLoader for an ``IterableDataset``;
  streaming sources shuffle internally instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from dataset.token_dataset import (
    Item,
    MemmapTokenDataset,
    StreamingTokenDataset,
    SyntheticTokenDataset,
)

Batch = dict[str, torch.Tensor]

VALID_SOURCES = ("synthetic", "file", "stream")


class DataEngine:
    """Wraps a dataset in a DataLoader and serves an endless stream of batches."""

    def __init__(
        self,
        dataset: Dataset[Item],
        *,
        batch_size: int = 8,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        pin_memory: bool | None = None,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        # An IterableDataset owns its own ordering; DataLoader raises if asked
        # to shuffle one.
        self.shuffle = shuffle and not isinstance(dataset, IterableDataset)
        self.drop_last = drop_last
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
        self.prefetch_factor = prefetch_factor

        self._loader = self._build_loader()
        self._iter: Iterator[Batch] = iter(self._loader)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict[str, Any], vocab_size: int) -> DataEngine:
        """Build from the ``dataset`` block of a loaded YAML config."""
        dcfg = cfg.get("dataset", {})
        source = dcfg.get("source", "synthetic")
        seq_len = int(dcfg.get("seq_len", 512))
        seed = int(cfg.get("seed", 0))

        dataset: Dataset[Item]
        if source == "synthetic":
            dataset = SyntheticTokenDataset(vocab_size, seq_len, seed=seed)
        elif source in ("file", "stream"):
            path = dcfg.get("path")
            if not path:
                raise ValueError(f"dataset.source={source!r} requires dataset.path")
            if source == "file":
                dataset = MemmapTokenDataset(path, seq_len)
            else:
                dataset = StreamingTokenDataset(
                    path, seq_len, shuffle=dcfg.get("shuffle", True), seed=seed
                )
        else:
            raise ValueError(f"unknown dataset.source {source!r}; expected one of {VALID_SOURCES}")

        return cls(
            dataset,
            batch_size=int(dcfg.get("batch_size", 8)),
            num_workers=int(dcfg.get("num_workers", 0)),
            prefetch_factor=dcfg.get("prefetch_factor"),
            pin_memory=dcfg.get("pin_memory"),
            shuffle=bool(dcfg.get("shuffle", True)),
            drop_last=bool(dcfg.get("drop_last", True)),
        )

    def _build_loader(self) -> DataLoader[Item]:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
        }
        if not isinstance(self.dataset, IterableDataset):
            kwargs["shuffle"] = self.shuffle
        if self.num_workers > 0:
            # Both of these are invalid with num_workers=0.
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.prefetch_factor)
            kwargs["persistent_workers"] = True
        return DataLoader(self.dataset, **kwargs)

    # ------------------------------------------------------------------
    # Batch production
    # ------------------------------------------------------------------
    def next_batch(self, device: torch.device | str | None = None) -> Batch:
        """Next batch, wrapping around at the end of a finite dataset."""
        try:
            batch = next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            batch = next(self._iter)
        if device is None:
            return batch
        # non_blocking pairs with pinned memory to overlap H2D with compute; it
        # degrades to a synchronous copy elsewhere.
        return {k: v.to(device, non_blocking=self.pin_memory) for k, v in batch.items()}

    def __iter__(self) -> Iterator[Batch]:
        while True:
            yield self.next_batch()
