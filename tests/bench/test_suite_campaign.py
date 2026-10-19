"""Campaign failure/freshness checks are integration contracts, not QA results."""

import hashlib
import json

import httpx
import pytest


@pytest.fixture
def conflict_adapter(tmp_path, bench_cfg):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from gluomem_bench.bench.adapters.memoryagentbench import MemoryAgentBenchConflictAdapter

    rows = [{"context": "Authored history fact.", "questions": ["Authored question"],
             "answers": [["right"]], "metadata": {"source": source, "qa_pair_ids": [source]}}
            for source in ["factconsolidation_sh_6k", "factconsolidation_mh_6k"]]
    path = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    cfg = bench_cfg.with_overrides({"benchmark.datasets.memoryagentbench_conflict": {
        "url": "", "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }})
    return cfg, MemoryAgentBenchConflictAdapter(cfg, path=path)


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_history_workload_has_exact_coverage_or_fatal_receipt(tmp_path, conflict_adapter, finish):
    from gluomem_bench.bench.rack import FatalRackError, RackProfile
    from gluomem_bench.bench.suite_campaign import run_history, verify_history

    cfg, adapter = conflict_adapter
    out = tmp_path / "run"
    requests = []

    def serve(request):
        requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "example/served-model",
                                                       "max_model_len": 131072}]})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 10})
        return httpx.Response(200, json={"model": "example/served-model", "choices": [
            {"finish_reason": finish, "message": {"content": "right"}}]})

    profile = RackProfile(base_url="http://localhost:8000/v1", model="example/served-model")
    args = dict(cfg=cfg, adapter=adapter, samples=adapter.load(), profile=profile,
                out=out, system="no_memory", transport=httpx.MockTransport(serve))
    if finish == "length":
        with pytest.raises(FatalRackError):
            run_history(**args)
        assert not (out / "manifest.json").exists()
        assert not (out / "scores.json").exists()
        assert json.loads((out / "failure.json").read_text())["generation_calls"] == 1
        assert sum(r.url.path.endswith("/chat/completions") for r in requests) == 1
    else:
        result = run_history(**args)
        assert result["questions"] == 2
        assert verify_history(out) == result
        payloads = [json.loads(r.content) for r in requests
                    if r.url.path.endswith("/chat/completions")]
        assert len(payloads) == 2
        assert all("Authored history fact" not in json.dumps(p) for p in payloads)
        prior = len(requests)
        with pytest.raises(FileExistsError):
            run_history(**args)
        assert len(requests) == prior


def test_document_source_events_and_committed_coverage(
    tmp_path, conflict_adapter, mock_embedder
):
    from gluomem_bench.bench.beam_ingest import _ingest_conversation, _read_rows, _validate_rows
    from gluomem_bench.bench.suite_campaign import make_source_events
    from tests.bench.test_beam_ingest import RecordingExtractors

    cfg, adapter = conflict_adapter
    conv = adapter.conversations()[0]
    events = make_source_events(conv, adapter.name)
    assert "".join(e.content for e in events) == "Authored history fact."
    assert all(e.source_type == "document" and not e.messages for e in events)
    assert all(e.event_time is None and e.kind is None for e in events)
    stores = tmp_path / "stores"
    stores.mkdir()
    _ingest_conversation(cfg, adapter, stores, RecordingExtractors(), mock_embedder,
                         conv, None, source_events=events,
                         coverage_protocol="suite:committed-input-receipts@v1")
    meta = json.loads((stores / f"{conv.conv_id}.ingest.json").read_text())
    receipts, _ = _validate_rows(_read_rows(stores / f"{conv.conv_id}.mind"), events)
    assert meta["committed_receipts"] == receipts
    assert meta["coverage_protocol"] == "suite:committed-input-receipts@v1"


def test_production_workload_reuses_real_ingest_and_hybrid_search(
    tmp_path, conflict_adapter, mock_embedder, monkeypatch
):
    from gluomem_bench.bench.rack import RackProfile
    from gluomem_bench.bench.suite_campaign import run_history, verify_history
    from tests.bench.test_beam_ingest import RecordingExtractors

    cfg, adapter = conflict_adapter
    recorder = RecordingExtractors()
    monkeypatch.setattr("gluomem.embedding.providers.make_embedder", lambda _: mock_embedder)
    monkeypatch.setattr("gluomem.llm.modules.DspyExtractors", lambda *a, **k: recorder)

    def serve(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "example/served-model",
                                                       "max_model_len": 131072}]})
        return httpx.Response(200, json={"count": 10} if request.url.path == "/tokenize" else {
            "model": "example/served-model", "choices": [
                {"finish_reason": "stop", "message": {"content": "unknown"}}]})

    out = tmp_path / "production"
    result = run_history(cfg=cfg, adapter=adapter, samples=adapter.load(),
                         profile=RackProfile(base_url="http://localhost:8000/v1",
                                             model="example/served-model"),
                         out=out, system="gluomem", transport=httpx.MockTransport(serve))
    assert result["questions"] == 2
    assert recorder.extractions == [("Authored history fact.", None)]
    assert verify_history(out) == result
    sentinels = list((out / "stores").glob("*.ingest.json"))
    assert len(sentinels) == 1
    assert json.loads(sentinels[0].read_text())["complete"] is True


def test_offline_verifier_rejects_empty_evidence_and_failed_prediction(tmp_path):
    from gluomem_bench.bench.suite_campaign import PROTOCOL, verify_history

    sample = {"id": "q", "category_name": "task", "gold": '["right"]'}
    row = {"sample": sample, "score": 1.0, "judge_verdict": None,
           "prediction": {"sample_id": "q", "answer": "right", "failure": "timeout"}}
    manifest = {"protocol": PROTOCOL, "complete": True, "artifact_sha256": {},
                "sample_ids": ["q"], "dataset": "memoryagentbench_conflict", "system": "no_memory"}
    scores = {"dataset": "memoryagentbench_conflict", "system": "no_memory", "questions": 1,
              "by_category": {"task": {"n": 1, "mean": 1.0}}}
    for name, value in [("manifest.json", manifest), ("predictions.json", [row]),
                        ("scores.json", scores)]:
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_history(tmp_path)


def test_finalization_error_retains_consumed_calls_without_complete_manifest(
    tmp_path, conflict_adapter, monkeypatch
):
    from gluomem_bench.bench import suite_campaign
    from gluomem_bench.bench.rack import RackProfile

    cfg, adapter = conflict_adapter
    write = suite_campaign._write

    def fail_scores(path, value):
        if path.name == "scores.json":
            raise OSError("injected finalization failure")
        return write(path, value)

    monkeypatch.setattr(suite_campaign, "_write", fail_scores)

    def serve(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "example/served-model",
                                                       "max_model_len": 131072}]})
        return httpx.Response(200, json={"count": 10} if request.url.path == "/tokenize" else {
            "model": "example/served-model", "choices": [
                {"finish_reason": "stop", "message": {"content": "right"}}]})

    out = tmp_path / "finalization"
    with pytest.raises(OSError, match="injected"):
        suite_campaign.run_history(
            cfg=cfg, adapter=adapter, samples=adapter.load(), out=out, system="no_memory",
            profile=RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
            transport=httpx.MockTransport(serve),
        )
    assert not (out / "manifest.json").exists()
    assert (out / "failure.json").is_file()
    assert json.loads((out / "failure.json").read_text())["generation_calls"] == 2


def test_manifest_is_not_published_before_final_verification(
    tmp_path, conflict_adapter, monkeypatch
):
    from gluomem_bench.bench import suite_campaign
    from gluomem_bench.bench.rack import RackProfile

    cfg, adapter = conflict_adapter
    out = tmp_path / "verification"

    def reject(_out, **kwargs):
        assert not (out / "manifest.json").exists()
        assert kwargs["manifest_name"] == "manifest.pending.json"
        raise ValueError("injected verification rejection")

    monkeypatch.setattr(suite_campaign, "verify_history", reject)

    def serve(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "example/served-model",
                                                       "max_model_len": 131072}]})
        return httpx.Response(200, json={"count": 10} if request.url.path == "/tokenize" else {
            "model": "example/served-model", "choices": [
                {"finish_reason": "stop", "message": {"content": "right"}}]})

    with pytest.raises(ValueError, match="injected verification rejection") as failure:
        suite_campaign.run_history(
            cfg=cfg, adapter=adapter, samples=adapter.load(), out=out, system="no_memory",
            profile=RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
            transport=httpx.MockTransport(serve),
        )
    assert failure.value.__dict__["generation_calls"] == 2
    assert not (out / "manifest.json").exists()
    assert not (out / "manifest.pending.json").exists()
    assert not (out / "scores.json").exists()
    assert json.loads((out / "failure.json").read_text())["generation_calls"] == 2
