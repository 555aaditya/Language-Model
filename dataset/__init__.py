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
from dataset.corpora import (
    CORPORA,
    Corpus,
    download,
    prepare,
    prepare_documents,
    read_documents,
    split_documents,
    use_system_trust_store,
)
from dataset.engine import DataEngine
from dataset.token_dataset import (
    MemmapTokenDataset,
    StreamingTokenDataset,
    SyntheticTokenDataset,
)

__all__ = [
    "CORPORA",
    "TOKEN_DTYPE",
    "Corpus",
    "DataEngine",
    "MemmapTokenDataset",
    "StreamingTokenDataset",
    "SyntheticTokenDataset",
    "count_tokens_bin",
    "download",
    "encode_texts_to_bin",
    "prepare",
    "prepare_documents",
    "read_documents",
    "read_tokens_bin",
    "split_documents",
    "use_system_trust_store",
    "write_tokens_bin",
]
