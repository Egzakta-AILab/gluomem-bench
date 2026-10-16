"""Production SourceEvent ingestion, with offline typed extractors for tests."""

import json
from typing import cast

from gluomem.llm.outputs import ConsolidationDecision, ExtractedClaimSpec, Extractors

from gluomem_bench.bench.adapters.beam import BeamAdapter
from tests.bench.test_beam import beam_data


def test_exhausted_workload_does_not_finalize_a_fake_complete_run(tmp_path):
    import httpx
    import pytest

    from gluomem_bench.bench.beam_eval import run_beam
    from gluomem_bench.bench.rack import RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    cfg, _, _ = beam_data(tmp_path)
    budget = WorkloadBudget(max_calls=1)
    budget.failed = True
    client = RackClient(
        RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
        receipt=tmp_path / "calls.jsonl",
        budget=budget,
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": RackProfile(
                                base_url="http://localhost:8000/v1", model="example/served-model"
                            ).model
                        }
                    ]
                },
            )
        ),
    )
    out = tmp_path / "run"
    try:
        with pytest.raises(RuntimeError, match="workload"):
            run_beam(cfg, client=client, out=out, retriever="bm25", limit=1)
        assert not (out / "manifest.json").exists()
        if (out / "ingest.json").exists():
            assert not json.loads((out / "ingest.json").read_text()).get("complete")
    finally:
        client.close()


def test_beam_ingest_refuses_null_extractors_instead_of_direct_claim_fallback(
    tmp_path, mock_embedder
):
    import pytest
    from gluomem.llm.fake import NullExtractors

    from gluomem_bench.bench.beam_ingest import ingest_beam

    cfg, _, _ = beam_data(tmp_path)
    with pytest.raises(ValueError, match="extract"):
        ingest_beam(
            cfg, db_dir=tmp_path / "stores", extractors=NullExtractors(), embedder=mock_embedder
        )


class RecordingExtractors:
    def __init__(self):
        self.extractions = []
        self.consolidations = []

    def extract_event_claims(self, text, reference_date=None):
        self.extractions.append((text, reference_date))
        return [ExtractedClaimSpec(text="The user prefers blue.", kind="preference")]

    def consolidate_claim(self, text, candidates):
        self.consolidations.append((text, candidates))
        return ConsolidationDecision(action="NOOP", target_frame_id=candidates[0].frame_id)


def test_beam_production_ingest_extracts_consolidates_and_records_verified_coverage(
    tmp_path, mock_embedder
):
    from gluomem_bench.bench.beam_ingest import ingest_beam, verify_beam_ingest

    cfg, path, rows = beam_data(tmp_path)
    rows[0]["chat"].append(
        [{"id": 2, "role": "user", "time_anchor": "March-16-2024", "content": "Still blue."}]
    )
    path.write_text(json.dumps(rows), encoding="utf-8")
    import hashlib

    cfg.benchmark.datasets["beam_128k"].sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    adapter = BeamAdapter(cfg)
    extractors = RecordingExtractors()
    db_dir = tmp_path / "stores"
    reports = ingest_beam(
        cfg, db_dir=db_dir, extractors=cast(Extractors, extractors), embedder=mock_embedder
    )
    assert len(extractors.extractions) == 2
    assert len(extractors.consolidations) == 1
    assert all(
        "Which color?" not in text and "Mentions blue" not in text
        for text, _ in extractors.extractions
    )
    assert reports[0]["path"] == "production_source_event"
    assert reports[0]["messages"] == 3
    assert reports[0]["events"] == 2
    assert verify_beam_ingest(cfg, adapter, db_dir, {"beam-1"}) == reports
    assert (db_dir / "beam-1.ingest.json").is_file()
    (db_dir / "beam-1.ingest.json").unlink()
    import pytest

    with pytest.raises(ValueError, match="incomplete"):
        verify_beam_ingest(cfg, adapter, db_dir, {"beam-1"})


def test_real_production_adapter_length_failure_leaves_no_events_or_sentinel(
    tmp_path, monkeypatch, mock_embedder
):
    import dspy
    import httpx
    import pytest
    from gluomem.llm.modules import DspyExtractors
    from gluomem.storage.db import MindDB

    from gluomem_bench.bench.beam_ingest import ingest_beam
    from gluomem_bench.bench.rack import RackClient, RackProfile, controlled_config
    from gluomem_bench.bench.rack_trace import RackRouter, WorkloadBudget

    _, path, _ = beam_data(tmp_path)
    import hashlib

    cfg = controlled_config(
        RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
        data=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv("GLUOMEM_RACK_API_KEY", "rack-local-no-auth")

    def provider(lm, *args, **kwargs):
        content = '{"claims": [{"text": "Blue", "kind": "preference"}]}'
        lm.history.append(
            {
                "response": {
                    "model": RackProfile(
                        base_url="http://localhost:8000/v1", model="example/served-model"
                    ).model,
                    "choices": [{"finish_reason": "length", "message": {"content": content}}],
                }
            }
        )
        return [content]

    monkeypatch.setattr(dspy.LM, "__call__", provider)
    budget = WorkloadBudget()
    client = RackClient(
        RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
        receipt=tmp_path / "http.jsonl",
        budget=budget,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"count": 10})),
    )
    extractors = DspyExtractors(
        cfg, router=RackRouter(cfg, client=client, trace=tmp_path / "raw.jsonl", budget=budget)
    )
    stores = tmp_path / "stores"
    try:
        with pytest.raises(RuntimeError, match="incomplete"):
            ingest_beam(
                cfg,
                db_dir=stores,
                extractors=extractors,
                embedder=mock_embedder,
                completion_guard=budget.check,
            )
        assert not (stores / "beam-1.ingest.json").exists()
        db = MindDB(stores / "beam-1.mind", dim=384)
        try:
            assert not db.fetchall("SELECT id FROM source_events")
            assert not db.fetchall("SELECT id FROM memory_frames")
        finally:
            db.close()
        raw = [json.loads(x) for x in (tmp_path / "raw.jsonl").read_text().splitlines()]
        assert raw[0]["finish_reasons"] == ["length"]
        assert budget.calls == 1
    finally:
        client.close()


def test_failed_coverage_readback_removes_completion_sentinel(tmp_path, monkeypatch, mock_embedder):
    import pytest

    from gluomem_bench.bench import beam_ingest

    cfg, _, _ = beam_data(tmp_path)
    monkeypatch.setattr(beam_ingest, "ingest_source_event", lambda *a, **k: None)
    stores = tmp_path / "stores"
    with pytest.raises(ValueError, match="source-event coverage"):
        beam_ingest.ingest_beam(
            cfg,
            db_dir=stores,
            extractors=cast(Extractors, RecordingExtractors()),
            embedder=mock_embedder,
        )
    assert not (stores / "beam-1.ingest.json").exists()


def test_ingest_deadline_failure_never_writes_completion_sentinel(tmp_path, mock_embedder):
    import pytest

    from gluomem_bench.bench.beam_ingest import ingest_beam
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    cfg, _, _ = beam_data(tmp_path)
    clock = [0.0]
    budget = WorkloadBudget(max_seconds=10, clock=lambda: clock[0])

    class SlowExtractors(RecordingExtractors):
        def extract_event_claims(self, text, reference_date=None):
            result = super().extract_event_claims(text, reference_date)
            clock[0] = 11
            return result

    db_dir = tmp_path / "stores"
    with pytest.raises(RuntimeError, match="workload"):
        ingest_beam(
            cfg,
            db_dir=db_dir,
            extractors=cast(Extractors, SlowExtractors()),
            embedder=mock_embedder,
            completion_guard=budget.check,
        )
    assert not (db_dir / "beam-1.ingest.json").exists()


def test_beam_ingest_verification_rejects_changed_event_metadata(tmp_path, mock_embedder):
    from gluomem.storage.db import MindDB

    from gluomem_bench.bench.beam_ingest import ingest_beam, verify_beam_ingest

    cfg, _, _ = beam_data(tmp_path)
    db_dir = tmp_path / "stores"
    ingest_beam(
        cfg,
        db_dir=db_dir,
        extractors=cast(Extractors, RecordingExtractors()),
        embedder=mock_embedder,
    )
    db = MindDB(db_dir / "beam-1.mind", dim=cfg.storage.vector_dim)
    try:
        row = db.fetchone("SELECT payload_json FROM source_events")
        assert row is not None
        payload = json.loads(row["payload_json"])
        payload["event_time"] = "2099-01-01T00:00:00"
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        import hashlib

        # A self-consistent payload hash is not proof of dataset coverage.
        with db.transaction():
            db.execute(
                "UPDATE source_events SET payload_json = ?, payload_hash = ?",
                (encoded, hashlib.sha256(encoded.encode()).hexdigest()),
            )
    finally:
        db.close()
    import pytest

    with pytest.raises(ValueError, match="event payload"):
        verify_beam_ingest(cfg, BeamAdapter(cfg), db_dir, {"beam-1"})
