"""Byte-level Byte Pair Encoding (BPE) tokenizer built from scratch.

Design notes (see docs/TDD.md §1, TDR-002, TDR-020):

- **Byte-level.** Text is encoded to UTF-8 bytes, so any text — emoji, rare
  scripts — is tokenizable with no "unknown" token.
- **Vocabulary.** Ids 0..k-1 are reserved special tokens; then the 256 base byte
  tokens; then learned merges.
- **Pre-tokenized.** Text is split into pieces (words, digit runs, symbol runs)
  before merging, and merges never cross a piece boundary (TDR-020). This is
  what makes both training and encoding tractable — see the complexity note
  below — and it stops the vocabulary being spent on cross-word artefacts.

Why pre-tokenization is not optional (measured, not assumed):

Merging is quadratic in the length of the sequence it runs over. Run over a whole
document, `encode()` measured 4.0x slower per 2x input — 50 KB/s at 1 KB falling
to 2.9 KB/s at 16 KB — which makes a 500 MB corpus unusable. Restricted to
pieces of ~1-20 bytes, with identical pieces cached, the same corpus encodes at
~11 MB/s. That is the difference between a two-day preprocessing step and a
40-second one.

Training uses the same idea from the other direction: pieces are counted once
into a frequency table, and each merge only revisits the pieces that actually
contained the merged pair, rather than rescanning the corpus.

API contract (see docs/BUILD_ORDER.md "Contracts"):
    BPE(vocab_size, special_tokens)
    .train(texts)
    .encode(text) -> list[int]
    .decode(ids) -> str
    .save(path) / .load(path)
    .vocab_size property
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable

from tokenizer.pretokenize import GPT2_PATTERN, pretokenize

SAVE_FORMAT_VERSION = 2


class BPE:
    """Byte-level Byte Pair Encoding tokenizer built from scratch."""

    def __init__(
        self,
        vocab_size: int = 4096,
        special_tokens: list[str] | None = None,
    ) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (bytes)")

        self.vocab_size = vocab_size
        self.special_tokens = list(special_tokens) if special_tokens else []

        # Predefined byte<->unicode rendering table (GPT-2 style).
        self.bs, self.sb = self._bytes_to_unicode_map()

        # encoder: token_key (str) -> id
        # decoder: id -> token_key (str)
        # For bytes, token_key is the rendered unicode string.
        # For special tokens, token_key is the literal special string itself.
        self.encoder: dict[str, int] = {}
        self.decoder: dict[int, str] = {}

        # Special tokens first, occupying the lowest ids (atomic, never merged).
        for i, sp in enumerate(self.special_tokens):
            self.encoder[sp] = i
            self.decoder[i] = sp
        self.num_special = len(self.special_tokens)

        # Add the 256 base byte tokens.
        next_id = self.num_special
        for b in range(256):
            rendered = self.bs[b]
            self.encoder[rendered] = next_id
            self.decoder[next_id] = rendered
            next_id += 1
        self._base_size = next_id

        # Byte decomposition for every token id (token_id -> list[byte value]).
        self._byte_seqs: dict[int, list[int]] = {self.num_special + b: [b] for b in range(256)}

        # BPE merges: list of ((a, b), merged_bytes, merged_key), rank-ordered.
        self.merges: list[tuple] = []
        self.merge_rank: dict[tuple, int] = {}
        # (a, b) -> merged id. Keeping the id here means the encode hot loop
        # never re-renders a token key or hashes a long string.
        self.merge_to_id: dict[tuple, int] = {}

        self._special_keys = set(self.special_tokens)
        self._special_re: re.Pattern[str] | None = None
        self._rebuild_special_re()
        # piece -> ids. Natural text repeats pieces heavily, so after warmup most
        # of a corpus costs one dict lookup.
        self._piece_cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------
    # bytes <-> unicode rendering (GPT-2 style)
    # ------------------------------------------------------------------
    @staticmethod
    def _bytes_to_unicode_map():
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        b_to_u = {b: chr(c) for b, c in zip(bs, cs, strict=True)}
        u_to_b = {c: b for b, c in b_to_u.items()}
        return b_to_u, u_to_b

    def _render_bytes(self, byte_tokens: list[int]) -> str:
        """Render a list of byte values (0-255) to the unicode token key string."""
        return "".join(self.bs[b] for b in byte_tokens)

    def _rebuild_special_re(self) -> None:
        """Alternation over special tokens, longest first so prefixes lose."""
        if not self.special_tokens:
            self._special_re = None
            return
        ordered = sorted(self.special_tokens, key=len, reverse=True)
        self._special_re = re.compile("(" + "|".join(re.escape(s) for s in ordered) + ")")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, texts: Iterable[str]) -> None:
        """Learn BPE merges from a corpus.

        Counts pre-tokenized pieces into a frequency table, then repeatedly
        merges the most frequent adjacent pair. An index from pair -> pieces
        containing it means each merge only rewrites the pieces it affects,
        instead of rescanning the whole corpus — the difference between
        O(merges x corpus) and O(merges x affected pieces).
        """
        freqs: Counter[str] = Counter()
        for text in texts:
            freqs.update(pretokenize(text))

        words: list[list[int]] = [
            [self.num_special + b for b in piece.encode("utf-8")] for piece in freqs
        ]
        weights: list[int] = list(freqs.values())

        pair_counts: dict[tuple, int] = {}
        pair_where: dict[tuple, set[int]] = {}

        def distinct_pairs(word: list[int]) -> set[tuple]:
            return {(word[j], word[j + 1]) for j in range(len(word) - 1)}

        def add(index: int) -> None:
            word, weight = words[index], weights[index]
            for j in range(len(word) - 1):
                pair = (word[j], word[j + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + weight
            for pair in distinct_pairs(word):
                pair_where.setdefault(pair, set()).add(index)

        def remove(index: int) -> None:
            word, weight = words[index], weights[index]
            for j in range(len(word) - 1):
                pair = (word[j], word[j + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) - weight
            # Discard once per *distinct* pair: a word holding the same pair
            # twice must not be dropped from the index on the first decrement.
            for pair in distinct_pairs(word):
                holders = pair_where.get(pair)
                if holders is not None:
                    holders.discard(index)
                    if not holders:
                        pair_where.pop(pair, None)
                if pair_counts.get(pair, 0) <= 0:
                    pair_counts.pop(pair, None)

        for index in range(len(words)):
            add(index)

        while len(self.encoder) < self.vocab_size and pair_counts:
            # Most frequent pair; tie-break deterministically by tuple value.
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            affected = list(pair_where.get(best, ()))
            if not affected:
                break

            a, b = best
            merged_bytes = self._byte_seqs[a] + self._byte_seqs[b]
            merged_key = self._render_bytes(merged_bytes)
            merged_id = len(self.encoder)

            self.encoder[merged_key] = merged_id
            self.decoder[merged_id] = merged_key
            self._byte_seqs[merged_id] = merged_bytes
            self.merges.append((best, merged_bytes, merged_key))
            self.merge_rank[best] = len(self.merges) - 1
            self.merge_to_id[best] = merged_id

            for index in affected:
                remove(index)
            for index in affected:
                words[index] = self._apply_merge(words[index], best, merged_id)
            for index in affected:
                add(index)

        self.vocab_size = len(self.encoder)
        self._piece_cache.clear()

    @staticmethod
    def _apply_merge(word: list[int], pair: tuple, merged_id: int) -> list[int]:
        """Replace every occurrence of ``pair`` in ``word`` with ``merged_id``."""
        out: list[int] = []
        i = 0
        n = len(word)
        while i < n:
            if i + 1 < n and (word[i], word[i + 1]) == pair:
                out.append(merged_id)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return out

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def _encode_piece(self, piece: str) -> list[int]:
        """Merge within one pre-tokenized piece, applying all occurrences per pass."""
        ids = [self.num_special + b for b in piece.encode("utf-8")]
        while len(ids) >= 2:
            best_pair = None
            best_rank = len(self.merges)
            for i in range(len(ids) - 1):
                rank = self.merge_rank.get((ids[i], ids[i + 1]))
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = (ids[i], ids[i + 1])
            if best_pair is None:
                break
            ids = self._apply_merge(ids, best_pair, self.merge_to_id[best_pair])
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode a UTF-8 string to a list of token ids."""
        if not text:
            return []

        spans = self._special_re.split(text) if self._special_re is not None else [text]

        tokens: list[int] = []
        for span in spans:
            if not span:
                continue
            if span in self._special_keys:
                tokens.append(self.encoder[span])
                continue
            for piece in pretokenize(span):
                cached = self._piece_cache.get(piece)
                if cached is None:
                    cached = self._piece_cache[piece] = self._encode_piece(piece)
                tokens.extend(cached)
        return tokens

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def token_bytes(self, idx: int) -> bytes:
        """Return the raw byte sequence backing a token id."""
        key = self.decoder[idx]
        if key in self._special_keys:
            return key.encode("utf-8")
        return bytes(self.sb[u] for u in key)

    @property
    def pattern(self) -> re.Pattern[str]:
        return GPT2_PATTERN

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back to a UTF-8 string."""
        byte_parts: list[int] = []
        for i in ids:
            key = self.decoder[i]
            if key in self._special_keys:
                byte_parts.extend(key.encode("utf-8"))
            else:
                for u in key:
                    byte_parts.append(self.sb[u])
        return bytes(byte_parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialize vocabulary + merges to a JSON file."""
        data = {
            "version": SAVE_FORMAT_VERSION,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "encoder": dict(self.encoder),
            "merges": [
                {"a": a, "b": b, "bytes": mb, "key": key} for (a, b), mb, key in self.merges
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, sort_keys=True)

    def load(self, path: str) -> None:
        """Load vocabulary + merges from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.special_tokens = list(data["special_tokens"])
        self.encoder = {k: int(v) for k, v in data["encoder"].items()}
        self.decoder = {int(v): k for k, v in data["encoder"].items()}
        self.bs, self.sb = self._bytes_to_unicode_map()
        self.num_special = len(self.special_tokens)
        self._special_keys = set(self.special_tokens)
        self._rebuild_special_re()
        self._piece_cache.clear()

        self._byte_seqs = {self.num_special + b: [b] for b in range(256)}

        self.merges = []
        self.merge_rank = {}
        self.merge_to_id = {}
        for rank, m in enumerate(data["merges"]):
            a, b, merged_bytes, merged_key = m["a"], m["b"], m["bytes"], m["key"]
            merged_id = self.encoder[merged_key]
            self.merges.append(((a, b), merged_bytes, merged_key))
            self._byte_seqs[merged_id] = merged_bytes
            self.merge_rank[(a, b)] = rank
            self.merge_to_id[(a, b)] = merged_id
