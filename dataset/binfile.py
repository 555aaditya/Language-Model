"""On-disk token format: a bare flat array of little-endian ``uint16``.

No header, no index, no framing. That is deliberate -- the file is exactly
``n_tokens * 2`` bytes, so the window count is derivable from ``stat()`` alone
(no need to open or parse anything), and any tool can read it: ``np.memmap``,
``torch.from_file``, or ``xxd``. Metadata that would otherwise live in a header
(vocab size, tokenizer version) belongs with the tokenizer's ``vocab.json``,
which must be kept alongside the ``.bin`` -- token ids are meaningless without
the vocabulary that produced them.

``uint16`` caps the vocabulary at 65,536, which is comfortably above the 4,096
we train with and matches GPT-2's 50,257. Going past that means ``uint32`` and
double the disk/page-cache footprint, so we validate rather than widen.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

# The single source of truth for the on-disk dtype. Import this rather than
# writing np.uint16 at call sites, so a future widening is one edit.
TOKEN_DTYPE = np.uint16

_MAX_TOKEN_ID = np.iinfo(TOKEN_DTYPE).max


def write_tokens_bin(ids: Iterable[int], path: str | os.PathLike[str]) -> int:
    """Write token ids to ``path`` as a flat ``uint16`` array. Returns the count.

    Raises ``ValueError`` on any id outside ``[0, 65535]``. We check explicitly
    because numpy would otherwise wrap silently (``65536 -> 0``), corrupting the
    corpus in a way that surfaces much later as inexplicable training loss.
    """
    arr = np.fromiter(ids, dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() > _MAX_TOKEN_ID):
        raise ValueError(
            f"token ids must fit uint16 [0, {_MAX_TOKEN_ID}]; got range [{arr.min()}, {arr.max()}]"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.astype(TOKEN_DTYPE).tofile(path)
    return int(arr.size)


def read_tokens_bin(path: str | os.PathLike[str]) -> np.memmap:
    """Open a token file read-only as a memory map (no bulk copy into RAM)."""
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


def count_tokens_bin(path: str | os.PathLike[str]) -> int:
    """Number of tokens in a token file, from its size alone -- no open, no read."""
    return Path(path).stat().st_size // np.dtype(TOKEN_DTYPE).itemsize


def encode_texts_to_bin(
    tokenizer: object,
    texts: Iterable[str],
    path: str | os.PathLike[str],
    *,
    eot_token: str | None = None,
) -> int:
    """Tokenize ``texts`` into a single ``.bin`` corpus. Returns the token count.

    ``eot_token`` (e.g. ``"<|endoftext|>"``) is appended after each text so the
    model learns a document boundary; without it, packed windows silently splice
    the end of one document onto the start of the next.
    """
    encode = tokenizer.encode  # type: ignore[attr-defined]
    eot_ids: list[int] = encode(eot_token) if eot_token else []

    def stream() -> Iterator[int]:
        for text in texts:
            yield from encode(text)
            yield from eot_ids

    return write_tokens_bin(stream(), path)
