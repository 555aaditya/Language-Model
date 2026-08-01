"""Efficient dataset engine: memory-mapped, streaming, worker-sharded.

See docs/TDD.md §2 and TDR-004.
"""

from dataset.binfile import (
    TOKEN_DTYPE,
    count_tokens_bin,
    encode_texts_to_bin,
    read_tokens_bin,
    write_tokens_bin,
)
from dataset.engine import DataEngine
from dataset.token_dataset import (
    MemmapTokenDataset,
    StreamingTokenDataset,
    SyntheticTokenDataset,
)

__all__ = [
    "TOKEN_DTYPE",
    "DataEngine",
    "MemmapTokenDataset",
    "StreamingTokenDataset",
    "SyntheticTokenDataset",
    "count_tokens_bin",
    "encode_texts_to_bin",
    "read_tokens_bin",
    "write_tokens_bin",
]
