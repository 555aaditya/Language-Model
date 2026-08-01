"""Pre-tokenization: split text into pieces before BPE merges (TDR-020).

BPE merges are applied *within* a piece and never across piece boundaries. That
serves two purposes:

1. **Quality.** It stops the tokenizer learning merges that straddle word
   boundaries (`"the c"`), which waste vocabulary on artefacts of word order.
2. **Speed.** Merging is quadratic in the length of the sequence it runs over.
   Applied to a whole document that is fatal; applied to pieces of ~1-20 bytes
   the quadratic term is a small constant, and identical pieces can be cached.

The pattern is GPT-2's, with one deviation: GPT-2 uses `\\p{L}` / `\\p{N}`, which
the standard library's `re` does not support (it needs the third-party `regex`
module). The unicode-aware equivalents here are `[^\\W\\d]` for letters and `\\d`
for digits, which keeps the dependency list honest at the cost of grouping
underscore with letters.

**The pattern must tile the input exactly** — the concatenation of all matches
has to reproduce the input character for character. A pattern with a gap makes
`encode()` silently drop text, which round-trip tests only catch if they happen
to use the dropped character. `test_pattern_tiles_every_input` asserts it
directly instead.
"""

from __future__ import annotations

import re

# Order matters: contractions first, then letter/digit/symbol runs each allowed
# one leading space, then whitespace. The `\s+(?!\S)` branch keeps a trailing
# run of whitespace together while letting a single space attach to the next
# word rather than becoming its own token.
GPT2_PATTERN = re.compile(
    r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def pretokenize(text: str, pattern: re.Pattern[str] = GPT2_PATTERN) -> list[str]:
    """Split ``text`` into pieces that BPE may merge within, but not across."""
    return pattern.findall(text)


def tiles_exactly(text: str, pattern: re.Pattern[str] = GPT2_PATTERN) -> bool:
    """True if the pattern reproduces ``text`` exactly — no dropped characters."""
    return "".join(pattern.findall(text)) == text
