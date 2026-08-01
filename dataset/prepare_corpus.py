"""Corpus preparation CLI.

    python -m dataset.prepare_corpus --corpus tinyshakespeare --vocab-size 8192

Downloads, trains a tokenizer, splits by document, and writes
``train.bin`` / ``val.bin`` plus the ``vocab.json`` they were encoded with.

The vocabulary is written **beside the token files on purpose**. Token ids are
meaningless without the vocabulary that produced them, and a `.bin` paired with
the wrong `vocab.json` decodes to plausible-looking nonsense rather than failing,
so the two travel together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset.corpora import CORPORA, prepare, read_documents
from tokenizer import BPE


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare a corpus")
    parser.add_argument("--corpus", default="tinyshakespeare", choices=sorted(CORPORA))
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tokenizer-sample-mb",
        type=float,
        default=64.0,
        help="cap the text used to fit the vocabulary (TDD §1.5)",
    )
    args = parser.parse_args()

    entry = CORPORA[args.corpus]
    from dataset.corpora import download

    path = download(entry)
    documents = read_documents(path, entry.doc_separator)
    print(f"{entry.name}: {len(documents)} documents from {path} ({entry.licence})")

    # Fit the vocabulary on a bounded sample rather than the whole corpus: merge
    # frequencies converge long before the corpus is exhausted, and training cost
    # scales with the text fed in.
    budget = int(args.tokenizer_sample_mb * 1024 * 1024)
    sample: list[str] = []
    used = 0
    for doc in documents:
        if used >= budget:
            break
        sample.append(doc)
        used += len(doc)
    print(f"fitting vocabulary on {used / 1024 / 1024:.2f} MB ({len(sample)} documents)")

    tokenizer = BPE(vocab_size=args.vocab_size, special_tokens=["<|endoftext|>"])
    tokenizer.train(sample)
    print(f"vocabulary: {tokenizer.vocab_size} tokens, {len(tokenizer.merges)} merges")

    info = prepare(entry, tokenizer, val_fraction=args.val_fraction, seed=args.seed)
    vocab_path = Path(str(info["train_bin"])).parent / "vocab.json"
    tokenizer.save(str(vocab_path))
    info["vocab"] = str(vocab_path)

    total = int(info["train_tokens"]) + int(info["val_tokens"])
    info["chars_per_token"] = round(sum(len(d) for d in documents) / max(1, total), 3)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
