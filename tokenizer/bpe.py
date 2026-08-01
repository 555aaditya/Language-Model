"""Byte-level Byte Pair Encoding (BPE) tokenizer built from scratch.

Design notes (see vault/research/transformer_notes.md & decisions/):

- Byte-level: Text is encoded to UTF-8 bytes, so any text (including emojis and
  rare unicode) is tokenizable with no "unknown" tokens. This mirrors GPT-2's
  byte-level BPE strategy.
- Vocabulary: starts with 0..255 byte tokens plus reserved special tokens placed
  at the lowest ids (ids 0..k-1). BPE merges are learned by repeatedly joining
  the most frequent adjacent byte/token pair until vocab_size is reached.
- Memory: encoder is a dict (token_key -> id), decoder is a dict (id -> token_key).
  Merges are stored as rank-ordered lists so we can encode greedily.

API contract (see vault/implementation/module_interfaces.md):
    BPE(vocab_size, special_tokens)
    .train(texts)
    .encode(text) -> list[int]
    .decode(ids) -> str
    .save(path) / .load(path)
    .vocab_size property
"""

import json
from collections import defaultdict
from collections.abc import Iterable


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
        # Base byte tokens decompose to a single byte; merged tokens are added
        # during training (or reconstructed in load()).
        self._byte_seqs: dict[int, list[int]] = {self.num_special + b: [b] for b in range(256)}

        # BPE merges: list of (pair_of_byte_tokens (a,b), merged_token_key).
        # merge_rank: (a, b) [base byte token ids] -> rank (lower = merged first).
        self.merges: list[tuple] = []
        self.merge_rank: dict[tuple, int] = {}

        # Set of special token keys (for decode handling).
        self._special_keys = set(self.special_tokens)

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

    # ------------------------------------------------------------------
    # Token key / byte helpers
    # ------------------------------------------------------------------
    def _render_bytes(self, byte_tokens: list[int]) -> str:
        """Render a list of byte values (0-255) to the unicode token key string."""
        return "".join(self.bs[b] for b in byte_tokens)

    # ------------------------------------------------------------------
    # Training: learn BPE merges
    # ------------------------------------------------------------------
    def train(self, texts: Iterable[str]) -> None:
        """Learn BPE merges by greedily merging the most frequent adjacent pair.

        Repeatedly merges the most frequent adjacent pair until the vocabulary
        reaches the target size (or no more pairs remain). Standard greedy BPE.
        It is O(merges * corpus_len); for very large corpora one would maintain
        an incremental max-heap of pair frequencies.
        """
        # Convert each text to a list of base byte token ids (offset by specials).
        sequences: list[list[int]] = []
        for text in texts:
            sequences.append([self.num_special + b for b in text.encode("utf-8")])

        while len(self.encoder) < self.vocab_size:
            pair_counts: dict[tuple, int] = defaultdict(int)
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1

            if not pair_counts:
                break

            # Most frequent pair; tie-break deterministically by tuple value.
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best_pair] < 1:
                break

            a, b = best_pair
            merged_bytes = self._byte_seqs[a] + self._byte_seqs[b]
            merged_key = self._render_bytes(merged_bytes)
            merged_id = len(self.encoder)

            self.encoder[merged_key] = merged_id
            self.decoder[merged_id] = merged_key
            self._byte_seqs[merged_id] = merged_bytes
            self.merges.append(((a, b), merged_bytes, merged_key))
            self.merge_rank[(a, b)] = len(self.merges) - 1

            # Replace all occurrences of the pair in every sequence.
            new_sequences = []
            for seq in sequences:
                new_seq = []
                i = 0
                n = len(seq)
                while i < n:
                    if i + 1 < n and (seq[i], seq[i + 1]) == best_pair:
                        new_seq.append(merged_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_sequences.append(new_seq)
            sequences = new_sequences

        # Reflect the actual grown vocabulary size.
        self.vocab_size = len(self.encoder)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Encode a UTF-8 string to a list of token ids (greedy BPE)."""
        if not text:
            return []

        tokens: list[int] = []
        remaining = text
        while remaining:
            consumed_special = False
            for sp in self.special_tokens:
                if remaining.startswith(sp):
                    tokens.append(self.encoder[sp])
                    remaining = remaining[len(sp) :]
                    consumed_special = True
                    break
            if consumed_special:
                continue
            # Encode the full UTF-8 byte sequence for this character (1-4 bytes),
            # appending one base token per byte so multi-byte chars never drop data.
            char = remaining[0]
            for byte in char.encode("utf-8"):
                tokens.append(self.num_special + byte)
            remaining = remaining[1:]

        # Apply BPE merges greedily: repeatedly merge the lowest-rank adjacent
        # pair (in merge_rank order). Special-token ids are < num_special, so the
        # byte-only merge_rank lookups never apply across them. We precompute each
        # adjacent pair's rank once per pass (O(n)) instead of an inner dict scan.
        while True:
            best = None
            best_rank = float("inf")
            for i in range(len(tokens) - 1):
                rank = self.merge_rank.get((tokens[i], tokens[i + 1]))
                if rank is not None and rank < best_rank:
                    best = i
                    best_rank = rank
            if best is None:
                break
            a, b = tokens[best], tokens[best + 1]
            merged_bytes = self._byte_seqs[a] + self._byte_seqs[b]
            merged_key = self._render_bytes(merged_bytes)
            tokens[best] = self.encoder[merged_key]
            del tokens[best + 1]

        return tokens

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def token_bytes(self, idx: int) -> bytes:
        """Return the raw byte sequence backing a token id.

        For special tokens this is their literal UTF-8 text; for byte/merged
        tokens it is the original byte sequence recovered from the rendering.
        """
        key = self.decoder[idx]
        if key in self._special_keys:
            return key.encode("utf-8")
        return bytes(self.sb[u] for u in key)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back to a UTF-8 string."""
        byte_parts: list[int] = []
        for i in ids:
            key = self.decoder[i]
            if key in self._special_keys:
                # A special token: emit its literal text bytes.
                byte_parts.extend(key.encode("utf-8"))
            else:
                # A rendered unicode token: map each char back to the original byte.
                for u in key:
                    byte_parts.append(self.sb[u])
        return bytes(byte_parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialize vocabulary + merges to a JSON file."""
        data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "encoder": {k: v for k, v in self.encoder.items()},
            "merges": [
                {"a": a, "b": b, "bytes": mb, "key": key} for (a, b), mb, key in self.merges
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, sort_keys=True)
        print(f"saved tokenizer to {path}")

    def load(self, path: str) -> None:
        """Load vocabulary + merges from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.special_tokens = list(data["special_tokens"])
        self.encoder = {k: int(v) for k, v in data["encoder"].items()}
        # rebuild decoder by inverting encoder
        self.decoder = {int(v): k for k, v in data["encoder"].items()}
        self.bs, self.sb = self._bytes_to_unicode_map()
        self.num_special = len(self.special_tokens)
        self._special_keys = set(self.special_tokens)

        # Rebuild byte decompositions for base byte tokens.
        self._byte_seqs = {}
        for b in range(256):
            self._byte_seqs[self.num_special + b] = [b]

        # Rebuild merges + byte decompositions + merge_rank.
        self.merges = []
        self.merge_rank = {}
        for rank, m in enumerate(data["merges"]):
            a, b, merged_bytes, merged_key = m["a"], m["b"], m["bytes"], m["key"]
            merged_id = self.encoder[merged_key]
            self.merges.append(((a, b), merged_bytes, merged_key))
            self._byte_seqs[merged_id] = merged_bytes
            self.merge_rank[(a, b)] = rank
