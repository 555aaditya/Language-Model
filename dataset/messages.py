"""Preparing email and chat data as training documents (TDD §2).

This module builds documents from mail/message records. It deliberately ships
with **no connector to any mailbox or chat workspace** — the caller supplies the
records, so pointing it at real data is an explicit act rather than a default.

## Read this before using it on real mail

**A language model memorises its training data.** This project measured a
32.5M-parameter model reaching loss 0.017 on a 10,677-token corpus — i.e.
reproducing it. Anything fed through here is recoverable from the resulting
weights, so **a checkpoint trained on mail is a copy of that mail** and inherits
every handling obligation the mail had: retention, access control, deletion
requests, cross-border transfer.

**`redact()` is best-effort, not a compliance control.** It removes shapes it
recognises (addresses, phone numbers, long digit runs). It cannot recognise a
name in prose, a deal codename, a customer referred to obliquely, or an amount
that is sensitive only in context. Treat it as noise reduction on top of a
decision that the corpus is already acceptable to train on — never as the thing
that makes it acceptable.

**Scale makes it a fine-tuning corpus, not a pretraining one.** A mailbox is
~1-10M tokens against the ~650M a 32.5M-parameter model wants. Trained alone it
memorises rather than generalises; the sound route is pretrain on a large public
corpus, then fine-tune.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Structure markers, so the model can learn where a field begins rather than
# inferring it from punctuation. Register these as tokenizer special tokens.
FROM_TOKEN = "<|from|>"
SUBJECT_TOKEN = "<|subject|>"
BODY_TOKEN = "<|body|>"
MESSAGE_TOKEN = "<|message|>"
STRUCTURE_TOKENS = (FROM_TOKEN, SUBJECT_TOKEN, BODY_TOKEN, MESSAGE_TOKEN)

# Quoted-reply openers. Without stripping these, a thread of N replies contributes
# the same text N times and the model over-weights whatever gets quoted most.
_QUOTE_MARKERS = (
    re.compile(r"^\s*>.*$", re.MULTILINE),
    re.compile(r"^\s*On .{0,80}wrote:\s*$.*", re.MULTILINE | re.DOTALL),
    re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}.*", re.MULTILINE | re.DOTALL),
    re.compile(r"^\s*From:.*$.*", re.MULTILINE | re.DOTALL),
)

# `-- ` on its own line is the RFC 3676 signature delimiter.
_SIGNATURE = re.compile(r"^--\s*$.*", re.MULTILINE | re.DOTALL)

# Boilerplate footers dominate corporate mail by volume. Left in, the model
# learns to emit disclaimers, because that is genuinely the most predictable text
# in the corpus.
_DISCLAIMER = re.compile(
    # "is confidential", "are confidential and intended", "is intended solely" --
    # the copula varies with whether attachments are mentioned, and clauses sit
    # between it and the keyword. Bounded to one sentence ([^.\n]) so ordinary
    # prose that merely uses the word "confidential" is not swallowed.
    r"(this (e-?mail|message)\b[^.\n]{0,80}\b(is|are)\b[^.\n]{0,40}(confidential|intended)"
    r"|if you (are not|have received) th(is|e intended))",
    re.IGNORECASE,
)

# Order matters: the phone pattern accepts separators and would otherwise claim a
# bare 16-digit card or account number. Digit runs with no separators are matched
# first and labelled <|number|>; only separated runs reach <|phone|>.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<|email|>"),
    (re.compile(r"https?://\S+"), "<|url|>"),
    (re.compile(r"\b\d{6,}\b"), "<|number|>"),
    (re.compile(r"\+?\d[\d\s().-]{7,}\d"), "<|phone|>"),
)


@dataclass
class Message:
    """One mail or chat record. ``sender`` may already be a pseudonym."""

    body: str
    sender: str | None = None
    subject: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


def strip_quoted_replies(text: str) -> str:
    """Remove quoted history so a thread is not counted once per reply."""
    for pattern in _QUOTE_MARKERS:
        text = pattern.sub("", text)
    return text


def strip_signature(text: str) -> str:
    return _SIGNATURE.sub("", text)


def looks_like_disclaimer(line: str) -> bool:
    return bool(_DISCLAIMER.search(line))


def strip_disclaimers(text: str) -> str:
    """Drop everything from the first confidentiality-notice line onward."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if looks_like_disclaimer(line):
            return "\n".join(lines[:i])
    return text


def redact(text: str) -> str:
    """Mask recognisable identifier *shapes*. Best-effort — see the module note."""
    for pattern, placeholder in _REDACTIONS:
        text = pattern.sub(placeholder, text)
    return text


def clean_body(text: str, *, apply_redaction: bool = True) -> str:
    """Full cleaning pipeline for one message body."""
    text = strip_quoted_replies(text)
    text = strip_disclaimers(text)
    text = strip_signature(text)
    if apply_redaction:
        text = redact(text)
    # Collapse the runs of blank lines that stripping leaves behind.
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def to_document(
    message: Message, *, include_structure: bool = True, apply_redaction: bool = True
) -> str:
    """Render one message as a training document."""
    body = clean_body(message.body, apply_redaction=apply_redaction)
    if not include_structure:
        return body

    parts = [MESSAGE_TOKEN]
    if message.sender:
        parts += [FROM_TOKEN, redact(message.sender) if apply_redaction else message.sender]
    if message.subject:
        parts += [SUBJECT_TOKEN, redact(message.subject) if apply_redaction else message.subject]
    parts += [BODY_TOKEN, body]
    return "\n".join(parts)


def build_documents(
    messages: list[Message],
    *,
    include_structure: bool = True,
    apply_redaction: bool = True,
    min_chars: int = 40,
) -> list[str]:
    """Clean and render messages, dropping ones too short to carry signal.

    Very short messages ("thanks", "ok, done") are the bulk of a real mailbox by
    count and near-worthless by content; keeping them teaches the model to emit
    acknowledgements. ``min_chars`` applies to the *cleaned* body, so a long
    quoted reply with two new words is correctly dropped.
    """
    documents = []
    for message in messages:
        body = clean_body(message.body, apply_redaction=apply_redaction)
        if len(body) < min_chars:
            continue
        documents.append(
            to_document(
                Message(body=body, sender=message.sender, subject=message.subject),
                include_structure=include_structure,
                apply_redaction=apply_redaction,
            )
        )
    return documents
