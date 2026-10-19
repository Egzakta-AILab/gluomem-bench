"""Official-schema adapter contracts using authored parquet, not quality data."""

import hashlib

import pytest


def test_conflict_adapter_shares_identical_history_without_gold_ingestion(tmp_path, bench_cfg):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from gluomem_bench.bench.adapters.memoryagentbench import MemoryAgentBenchConflictAdapter

    context = "A factual document.\nA later correction.\n"
    rows = [{"context": context, "questions": ["Authored question"],
             "answers": [["GOLD_MUST_NOT_BE_INGESTED"]],
             "metadata": {"source": source, "qa_pair_ids": [source + "-q"]}}
            for source in ["factconsolidation_sh_6k", "factconsolidation_mh_6k"]]
    path = tmp_path / "conflict.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    cfg = bench_cfg.with_overrides({"benchmark.datasets.memoryagentbench_conflict": {
        "url": "", "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }})
    adapter = MemoryAgentBenchConflictAdapter(cfg, path=path, chunk_chars=13)
    samples = adapter.load()
    assert len(samples) == 2
    assert samples[0].conv_id == samples[1].conv_id
    convs = adapter.conversations()
    assert len(convs) == 1
    turns = convs[0].sessions[0].turns
    assert "".join(t.text for t in turns) == context
    assert all(t.speaker == "document" for t in turns)
    assert all("GOLD_MUST_NOT_BE_INGESTED" not in t.text for t in turns)
    assert {s.category for s in samples} == {1, 2}
    assert all(s.question == "Authored question" for s in samples)
    assert all("larger serial number" in s.reader_question for s in samples)
    assert all("not real-world facts" in s.reader_question for s in samples)
