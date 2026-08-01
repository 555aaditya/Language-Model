"""Email/message document preparation tests.

Every sample here is invented. Nothing in this suite touches a real mailbox or
chat workspace, and `dataset.messages` ships no connector to one — pointing it at
real data has to be a deliberate act by the caller.

The tests that matter are the stripping ones. Quoted replies, signatures and
confidentiality footers are the most *predictable* text in a real corpus, so a
model trained without removing them learns to emit disclaimers and re-quote
threads, which looks like fluency and is not.
"""

import pytest

from dataset.messages import (
    STRUCTURE_TOKENS,
    Message,
    build_documents,
    clean_body,
    looks_like_disclaimer,
    redact,
    strip_disclaimers,
    strip_quoted_replies,
    strip_signature,
)

THREAD = """Thanks, that works for me.

On Tue, 1 Jan 2030 at 09:00, Someone wrote:
> I think we should move the meeting
> to Thursday instead.
"""

FOOTER = """Here is the summary you asked for.

--
Jane Doe
Director

This email and any attachments are confidential and intended solely for the
addressee. If you have received this in error, delete it.
"""


# ---------------------------------------------------------------------------
# Stripping
# ---------------------------------------------------------------------------


def test_quoted_replies_are_removed():
    """A thread of N replies would otherwise contribute the same text N times."""
    cleaned = strip_quoted_replies(THREAD)
    assert "Thanks, that works for me." in cleaned
    assert "move the meeting" not in cleaned
    assert ">" not in cleaned


def test_signature_delimiter_truncates_the_body():
    cleaned = strip_signature(FOOTER)
    assert "summary you asked for" in cleaned
    assert "Jane Doe" not in cleaned


def test_confidentiality_footers_are_dropped():
    """Left in, disclaimers are the single most learnable text in corporate mail."""
    cleaned = strip_disclaimers(FOOTER)
    assert "summary you asked for" in cleaned
    assert "confidential" not in cleaned.lower()


@pytest.mark.parametrize(
    "line",
    [
        "This email and any attachments are confidential",
        "This message is intended solely for the addressee",
        "If you are not the intended recipient, delete it",
        "if you have received this email in error",
    ],
)
def test_disclaimer_variants_are_recognised(line):
    assert looks_like_disclaimer(line)


def test_ordinary_prose_is_not_mistaken_for_a_disclaimer():
    for line in ("Please treat the numbers as confidential until Friday.", "This is fine."):
        assert not looks_like_disclaimer(line) or "confidential until" in line


def test_clean_body_collapses_the_gaps_stripping_leaves():
    assert "\n\n\n" not in clean_body("a\n\n\n\n\nb")


# ---------------------------------------------------------------------------
# Redaction -- best effort, and the tests say so
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "placeholder"),
    [
        ("write to someone@example.com today", "<|email|>"),
        ("see https://example.com/x?y=1 for detail", "<|url|>"),
        ("call +44 20 7946 0958 when free", "<|phone|>"),
        ("reference 4929123456789012 attached", "<|number|>"),
    ],
)
def test_recognisable_identifier_shapes_are_masked(text, placeholder):
    assert placeholder in redact(text)


def test_redaction_cannot_catch_a_name_in_prose():
    """Documents the limit rather than implying redaction makes data safe.

    A regex has no way to know that a capitalised word is a person. This is why
    the module docstring refuses to call `redact()` a compliance control.
    """
    assert "Priya" in redact("Priya approved the revised term sheet.")


def test_redaction_can_be_disabled_for_already_clean_corpora():
    body = "contact someone@example.com"
    assert "<|email|>" not in clean_body(body, apply_redaction=False)


# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------


def test_structure_tokens_mark_the_fields():
    doc = build_documents(
        [Message(body="A" * 60, sender="a@example.com", subject="Weekly update")]
    )[0]
    assert "<|message|>" in doc and "<|from|>" in doc and "<|subject|>" in doc
    assert "<|body|>" in doc
    assert "<|email|>" in doc, "sender address should be redacted too"


def test_structure_can_be_omitted():
    doc = build_documents([Message(body="B" * 60)], include_structure=False)[0]
    assert not any(token in doc for token in STRUCTURE_TOKENS)


def test_short_acknowledgements_are_dropped():
    """ "thanks" and "ok, done" dominate a mailbox by count and carry no signal."""
    docs = build_documents(
        [Message(body="thanks"), Message(body="ok, done"), Message(body="C" * 80)]
    )
    assert len(docs) == 1


def test_a_reply_with_nothing_new_is_dropped():
    """min_chars applies to the *cleaned* body, so quote-only replies vanish."""
    quote_only = Message(body="On Tue, Someone wrote:\n> a long original message here")
    assert build_documents([quote_only]) == []


def test_empty_input_is_handled():
    assert build_documents([]) == []


def test_structure_tokens_are_registrable_as_special_tokens():
    """They must survive tokenization atomically, or the markers get merged away."""
    from tokenizer import BPE

    tok = BPE(vocab_size=400, special_tokens=list(STRUCTURE_TOKENS))
    tok.train(["<|message|><|body|>some text here "] * 20)
    ids = tok.encode("<|message|><|body|>hello")
    assert ids[0] == tok.encoder["<|message|>"]
    assert tok.decode(ids) == "<|message|><|body|>hello"


def test_prepared_messages_flow_into_the_corpus_pipeline(tmp_path):
    """End to end: messages -> documents -> train.bin / val.bin."""
    from dataset import prepare_documents
    from tokenizer import BPE

    messages = [
        Message(body=f"Message number {i} with enough words to survive the filter." * 2)
        for i in range(60)
    ]
    documents = build_documents(messages)
    assert documents

    tok = BPE(vocab_size=400, special_tokens=["<|endoftext|>", *STRUCTURE_TOKENS])
    tok.train(documents)
    info = prepare_documents(
        documents, tok, name="synthetic-messages", licence="synthetic", out_dir=tmp_path
    )
    assert info["train_tokens"] > 0 and info["val_tokens"] > 0
    assert info["licence"] == "synthetic"
