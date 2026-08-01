"""Corpus acquisition and document-level splitting (TDD §2).

Corpora are downloaded on demand and never committed — they are large and
licence-encumbered, and `.gitignore` already excludes `data/`. Each entry names
its licence, because "where did the training data come from" is a question that
gets asked later and is unpleasant to answer retroactively.

**The split is by document, never by token.** Windows are contiguous slices of
one flat token array, so splitting a *tokenised* stream puts validation tokens
inside training windows: the model trains on text it is then evaluated on, and
the reported perplexity is quietly too good. Splitting the document list first
makes that impossible by construction.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data")


@dataclass(frozen=True)
class Corpus:
    """A downloadable text corpus and the terms it comes under."""

    name: str
    url: str
    licence: str
    description: str
    doc_separator: str = "\n\n"

    @property
    def raw_path(self) -> Path:
        return DATA_DIR / f"{self.name}.txt"


# Small, permissively licensed, and plain-text — the constraints that matter for
# a from-scratch project with no data-engineering budget.
CORPORA: dict[str, Corpus] = {
    "tinyshakespeare": Corpus(
        name="tinyshakespeare",
        url=(
            "https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt"
        ),
        licence="public domain (Shakespeare)",
        description="~1.1 MB of Shakespeare. Small enough to tokenise in seconds.",
    ),
}


def use_system_trust_store() -> bool:
    """Route TLS verification through the OS trust store. True if it took effect.

    Needed on any machine behind a TLS-inspecting corporate proxy (Zscaler,
    Netskope, Blue Coat). The proxy re-signs every certificate with a corporate
    root CA that lives in the OS keychain, so `curl` succeeds — but `requests`
    verifies against certifi's bundle, which has never heard of it, and fails
    with `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`.

    Never disable verification to work around this: that turns a working
    corporate MITM into an accepted arbitrary one. Point at the real CA instead,
    either via the OS store here or `REQUESTS_CA_BUNDLE` (which requests honours
    natively).
    """
    try:
        import truststore
    except ImportError:
        return False
    truststore.inject_into_ssl()
    return True


def download(corpus: Corpus | str, *, force: bool = False, timeout: int = 60) -> Path:
    """Fetch a corpus to ``data/``. Returns the local path; skips if present."""
    entry = CORPORA[corpus] if isinstance(corpus, str) else corpus
    path = entry.raw_path
    if path.exists() and not force:
        return path

    import requests  # imported lazily so the package works offline

    used_system_trust = use_system_trust_store()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(entry.url, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.SSLError as exc:
        raise RuntimeError(
            f"TLS verification failed fetching {entry.url}.\n"
            f"System trust store in use: {used_system_trust}.\n"
            "On a TLS-inspecting corporate network the certificate is re-signed "
            "by a corporate root CA that certifi does not carry. Fix with either:\n"
            "  pip install truststore          # use the OS keychain\n"
            "  export REQUESTS_CA_BUNDLE=/path/to/corporate-ca.pem\n"
            "Do not disable verification."
        ) from exc

    path.write_text(response.text, encoding="utf-8")
    return path


def read_documents(path: str | os.PathLike[str], separator: str = "\n\n") -> list[str]:
    """Split a raw text file into documents, dropping blanks."""
    text = Path(path).read_text(encoding="utf-8")
    return [doc.strip() for doc in text.split(separator) if doc.strip()]


def split_documents(
    documents: list[str], *, val_fraction: float = 0.05, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Partition documents into (train, val) with a stable, content-derived hash.

    Hashing the document text rather than shuffling an index means the split is
    reproducible without carrying a random seed alongside the data, and a
    document keeps its side even if the corpus is re-ordered or extended. Adding
    new documents therefore never reshuffles the existing ones into the other
    split, which would leak across a resumed run.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    threshold = int(val_fraction * (1 << 32))
    train: list[str] = []
    val: list[str] = []
    for doc in documents:
        digest = hashlib.blake2b(f"{seed}:{doc}".encode(), digest_size=4).digest()
        (val if int.from_bytes(digest, "big") < threshold else train).append(doc)

    if not train or not val:
        raise ValueError(
            f"split produced an empty side ({len(train)} train, {len(val)} val); "
            f"corpus has only {len(documents)} documents"
        )
    return train, val


def prepare_documents(
    documents: list[str],
    tokenizer: object,
    *,
    name: str = "custom",
    licence: str = "unspecified",
    val_fraction: float = 0.05,
    seed: int = 0,
    eot_token: str | None = "<|endoftext|>",
    out_dir: Path | None = None,
) -> dict[str, object]:
    """Split → encode → write ``train.bin`` / ``val.bin`` from documents in memory.

    The entry point for any corpus that is not a plain URL download — a local
    export, a scrape, a mailbox.

    **On personal data.** A language model memorises: this project measured a
    32.5M-parameter model reaching loss 0.017 on a 10,677-token corpus, i.e.
    reproducing it. Any corpus fed here ends up recoverable from the weights, so
    a checkpoint trained on mail, chat logs or customer records *is* a copy of
    that data and inherits its handling obligations. Scrub before this call, not
    after — and note ``licence`` so the provenance travels with the artefacts.
    """
    from dataset.binfile import encode_texts_to_bin

    train_docs, val_docs = split_documents(documents, val_fraction=val_fraction, seed=seed)

    destination = out_dir or (DATA_DIR / name)
    train_bin = destination / "train.bin"
    val_bin = destination / "val.bin"
    n_train = encode_texts_to_bin(tokenizer, train_docs, train_bin, eot_token=eot_token)
    n_val = encode_texts_to_bin(tokenizer, val_docs, val_bin, eot_token=eot_token)

    return {
        "corpus": name,
        "licence": licence,
        "documents": len(documents),
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "train_tokens": n_train,
        "val_tokens": n_val,
        "train_bin": str(train_bin),
        "val_bin": str(val_bin),
    }


def prepare(
    corpus: Corpus | str,
    tokenizer: object,
    *,
    val_fraction: float = 0.05,
    seed: int = 0,
    eot_token: str | None = "<|endoftext|>",
    out_dir: Path | None = None,
) -> dict[str, object]:
    """Download → split by document → encode → write ``train.bin`` / ``val.bin``."""
    entry = CORPORA[corpus] if isinstance(corpus, str) else corpus
    documents = read_documents(download(entry), entry.doc_separator)
    return prepare_documents(
        documents,
        tokenizer,
        name=entry.name,
        licence=entry.licence,
        val_fraction=val_fraction,
        seed=seed,
        eot_token=eot_token,
        out_dir=out_dir,
    )
