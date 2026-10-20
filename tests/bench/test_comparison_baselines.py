"""Fair-comparison baseline contracts; no model or external dataset access."""

import pytest

from gluomem_bench.bench import baselines
from gluomem_bench.bench.types import Sample


def test_no_memory_calls_shared_reader_with_only_question(bench_cfg, answer_fn):
    sample = Sample("q", "history", "What do I prefer?", "private gold", 1, "preference")
    assert hasattr(baselines, "NoMemoryRetriever"), "comparison needs a no-history baseline"
    retriever = baselines.NoMemoryRetriever(bench_cfg, answer_fn)
    prediction = retriever.answer(sample)
    assert answer_fn.calls == [(sample.question, "")]
    assert prediction.sample_id == sample.id
    assert prediction.answer == "scripted answer"
    assert prediction.retrieved == ()
    assert retriever.retrieves is False


def test_strict_full_history_refuses_character_overflow_before_reader(
    bench_cfg, fixture_adapter, answer_fn
):
    cfg = bench_cfg.with_overrides({"benchmark.full_context.max_chars": 1})
    conversations = fixture_adapter.conversations()
    retriever = baselines.FullContextRetriever(
        cfg, conversations, answer_fn, allow_truncation=False
    )
    with pytest.raises(ValueError, match="complete history"):
        retriever.answer(fixture_adapter.load()[0])
    assert answer_fn.calls == []


def test_longmemeval_question_date_reaches_every_baseline_reader(bench_cfg, answer_fn):
    import hashlib
    from pathlib import Path

    from gluomem_bench.bench.adapters.longmemeval import LongMemEvalAdapter

    path = Path(__file__).parents[1] / "fixtures" / "longmemeval_tiny.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cfg = bench_cfg.with_overrides({
        "benchmark.datasets.longmemeval_oracle.sha256": digest,
    })
    adapter = LongMemEvalAdapter(cfg, "longmemeval_oracle", path=path)
    sample = adapter.load()[0]
    assert getattr(sample, "question_date", None) == "2023/05/30 (Tue) 23:40"
    expected = "Current Date: 2023/05/30 (Tue) 23:40\nQuestion: " + sample.question
    for retriever in (
        baselines.NoMemoryRetriever(cfg, answer_fn),
        baselines.FullContextRetriever(cfg, adapter.conversations(), answer_fn,
                                       allow_truncation=False),
        baselines.Bm25Retriever(cfg, adapter.conversations(), answer_fn),
    ):
        retriever.answer(sample)
        assert answer_fn.calls[-1][0] == expected
