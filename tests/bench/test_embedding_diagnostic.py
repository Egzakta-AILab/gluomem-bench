"""Diagnostics preserve raw source bytes; gold is used only by the scorer."""


def test_lossless_chunking_checks_encoder_limit_without_truncating():
    from gluomem_bench.bench.embedding_diagnostic import lossless_chunks

    text = "first paragraph\nsecond paragraph with details\n"
    chunks = lossless_chunks(text, token_count=lambda x: len(x) + 2,
                             max_tokens=10, soft_chars=20)
    assert "".join(chunks) == text
    assert all(0 < len(chunk) + 2 <= 10 for chunk in chunks)


def test_session_recall_matches_provenance_not_answer_text():
    from gluomem_bench.bench.embedding_diagnostic import session_recall

    assert session_recall(["s1", "s1", "s3"], ["s1", "s2"], k=2) == 0.5
    assert session_recall(["s1", "s1", "s3"], ["s3"], k=2) == 0.0
    assert session_recall(["s1"], [], k=10) is None
