"""Fixed-source encoder diagnostics, not whole-memory-system evaluation."""

from __future__ import annotations


def lossless_chunks(text, *, token_count, max_tokens, soft_chars=1000):
    """Split exact substrings until each fits, including caller-counted prefixes."""
    if not isinstance(text, str) or type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("text and positive integer token limit required")
    if type(soft_chars) is not int or soft_chars < 1:
        raise ValueError("positive integer soft character target required")
    chunks = []
    offset = 0
    while offset < len(text):
        length = min(soft_chars, len(text) - offset)
        while token_count(text[offset:offset + length]) > max_tokens:
            if length == 1:
                raise ValueError("single character cannot fit encoder context")
            length = max(1, length // 2)
        chunks.append(text[offset:offset + length])
        offset += length
    return chunks


def session_recall(ranked_session_ids, gold_session_ids, *, k):
    """Top-k chunks mapped to unique native sessions; empty gold is unscored."""
    if type(k) is not int or k < 1:
        raise ValueError("positive integer k required")
    gold = set(gold_session_ids)
    if not gold:
        return None
    return len(set(ranked_session_ids[:k]) & gold) / len(gold)
