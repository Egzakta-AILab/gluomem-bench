"""Input receipts prove processing, not retention of private plaintext."""

import json
from typing import cast

import pytest
from gluomem.llm.outputs import Extractors
from gluomem.storage.db import MindDB

from gluomem_bench.bench import beam_ingest
from gluomem_bench.bench.adapters.beam import BeamAdapter
from tests.bench.test_beam import beam_data
from tests.bench.test_beam_ingest import RecordingExtractors


@pytest.mark.parametrize("field", ["messages", "events"])
def test_boolean_counts_cannot_impersonate_integer_coverage(tmp_path, mock_embedder, field):
    cfg, _, _ = beam_data(tmp_path)
    stores = tmp_path / "stores"
    beam_ingest.ingest_beam(
        cfg,
        db_dir=stores,
        extractors=cast(Extractors, RecordingExtractors()),
        embedder=mock_embedder,
    )
    sentinel = stores / "beam-1.ingest.json"
    meta = json.loads(sentinel.read_text())
    meta[field] = True if field == "events" else 2.0
    sentinel.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="incompatible"):
        beam_ingest.verify_beam_ingest(cfg, BeamAdapter(cfg), stores, {"beam-1"})


def test_missing_commit_result_is_not_certified_by_a_forged_row(
    tmp_path, mock_embedder, monkeypatch
):
    cfg, _, _ = beam_data(tmp_path)
    original = beam_ingest.ingest_source_event

    def no_commit_ack(*args, **kwargs):
        original(*args, **kwargs)
        return None

    monkeypatch.setattr(beam_ingest, "ingest_source_event", no_commit_ack)
    stores = tmp_path / "stores"
    with pytest.raises(ValueError, match="commit"):
        beam_ingest.ingest_beam(
            cfg,
            db_dir=stores,
            extractors=cast(Extractors, RecordingExtractors()),
            embedder=mock_embedder,
        )
    assert not (stores / "beam-1.ingest.json").exists()
    failure = json.loads((stores / "beam-1.ingest.failure.json").read_text())
    assert failure["complete"] is False
    assert failure["persisted_event_rows"] == 1
    assert failure["seconds"] >= 0
    assert failure["error"] == "production_source_event_failed"
    assert failure["failure_diagnostic"] == {"category": "ValueError", "stage": "commit-coverage"}


def test_redacted_committed_input_is_verified_without_restoring_plaintext(
    tmp_path, mock_embedder, monkeypatch
):
    cfg, _, _ = beam_data(tmp_path)
    original = beam_ingest.ingest_source_event

    def ingest_then_redact(*args, **kwargs):
        result = original(*args, **kwargs)
        db = kwargs["store"]._db
        with db.transaction():
            db.execute(
                "UPDATE source_events SET payload_json = '{}' WHERE id = ?", (result.event_row_id,)
            )
        return result

    monkeypatch.setattr(beam_ingest, "ingest_source_event", ingest_then_redact)
    stores = tmp_path / "stores"
    reports = beam_ingest.ingest_beam(
        cfg,
        db_dir=stores,
        extractors=cast(Extractors, RecordingExtractors()),
        embedder=mock_embedder,
    )
    assert reports[0]["coverage_protocol"] == "beam:committed-input-receipts@v2"
    assert reports[0]["messages"] == 2
    assert reports[0]["retained_plaintext_messages"] == 0
    assert reports[0]["redacted_events"] == 1
    assert reports[0]["committed_receipts"][0]["messages"] == 2
    assert "content" not in json.dumps(reports[0]["committed_receipts"])
    assert beam_ingest.verify_beam_ingest(cfg, BeamAdapter(cfg), stores, {"beam-1"}) == reports
    db = MindDB(stores / "beam-1.mind", dim=cfg.storage.vector_dim)
    try:
        row = db.fetchone("SELECT payload_json FROM source_events")
        assert row is not None
        assert row["payload_json"] == "{}"
    finally:
        db.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "hash",
        "event_id",
        "revision",
        "source_ref",
        "source_type",
        "tenant_id",
        "principal_id",
        "namespace",
        "message",
        "scrubbed_hash",
        "sentinel_count",
        "sentinel_protocol",
        "sentinel_receipt",
        "sentinel_receipt_type",
    ],
)
def test_redacted_coverage_rejects_tampering(tmp_path, mock_embedder, mutation):
    import hashlib

    cfg, _, _ = beam_data(tmp_path)
    stores = tmp_path / "stores"
    beam_ingest.ingest_beam(
        cfg,
        db_dir=stores,
        extractors=cast(Extractors, RecordingExtractors()),
        embedder=mock_embedder,
    )
    sentinel = stores / "beam-1.ingest.json"
    meta = json.loads(sentinel.read_text())
    db = MindDB(stores / "beam-1.mind", dim=cfg.storage.vector_dim)
    try:
        row = db.fetchone("SELECT payload_json FROM source_events")
        assert row is not None
        payload = json.loads(row["payload_json"])
        with db.transaction():
            db.execute("UPDATE source_events SET payload_json = '{}'")
            meta["redacted_events"] = 1
            meta["retained_plaintext_messages"] = 0
            if mutation == "missing":
                db.execute("DELETE FROM source_events")
            elif mutation in (
                "hash",
                "event_id",
                "revision",
                "source_ref",
                "source_type",
                "tenant_id",
                "principal_id",
                "namespace",
            ):
                key = "payload_hash" if mutation == "hash" else mutation
                db.execute(
                    f"UPDATE source_events SET {key} = ?", (2 if key == "revision" else "wrong",)
                )
            elif mutation == "message":
                payload["messages"][0]["content"] = "Unprocessed replacement"
                digest = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()
                db.execute("UPDATE source_events SET payload_hash = ?", (digest,))
                meta["committed_receipts"][0]["payload_hash"] = digest
            elif mutation == "scrubbed_hash":
                digest = hashlib.sha256(b"{}").hexdigest()
                db.execute("UPDATE source_events SET payload_hash = ?", (digest,))
            elif mutation == "sentinel_count":
                meta["messages"] = 1
            elif mutation == "sentinel_protocol":
                meta["coverage_protocol"] = "legacy"
            elif mutation == "sentinel_receipt":
                meta["committed_receipts"][0]["messages"] = 1
            elif mutation == "sentinel_receipt_type":
                meta["committed_receipts"][0]["messages"] = 2.0
    finally:
        db.close()
    sentinel.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        beam_ingest.verify_beam_ingest(cfg, BeamAdapter(cfg), stores, {"beam-1"})


def test_expected_hash_matches_independent_full_scoped_envelope(tmp_path):
    import hashlib

    cfg, _, _ = beam_data(tmp_path)
    event = beam_ingest._events(BeamAdapter(cfg).conversations()[0])[0]
    expected = {
        "event_id": event.event_id,
        "revision": 1,
        "source_type": "chat_turn",
        "source_ref": event.source_ref,
        "scope": {"tenant_id": "local", "principal_id": "default", "namespace": "personal"},
        "messages": [m.model_dump() for m in event.messages],
        "content": None,
        "kind": None,
        "confidence": 1.0,
        "require_review": False,
        "pinned": False,
        "event_time": event.event_time,
    }
    assert (
        beam_ingest._expected_receipt(event)["payload_hash"]
        == hashlib.sha256(
            json.dumps(expected, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
    )
    assert event.messages[1].role == "assistant"


def test_fresh_ingest_refusal_never_deletes_a_prior_completion_receipt(tmp_path, mock_embedder):
    cfg, _, _ = beam_data(tmp_path)
    stores = tmp_path / "stores"
    kwargs = dict(
        cfg=cfg,
        db_dir=stores,
        extractors=cast(Extractors, RecordingExtractors()),
        embedder=mock_embedder,
    )
    beam_ingest.ingest_beam(**kwargs)
    sentinel = stores / "beam-1.ingest.json"
    before = sentinel.read_bytes()
    with pytest.raises(FileExistsError):
        beam_ingest.ingest_beam(**kwargs)
    assert sentinel.read_bytes() == before


def test_failure_receipt_survives_a_redacted_payload(tmp_path, mock_embedder, monkeypatch):
    cfg, _, _ = beam_data(tmp_path)
    original = beam_ingest.ingest_source_event

    def fail_after_redaction(*args, **kwargs):
        original(*args, **kwargs)
        db = kwargs["store"]._db
        with db.transaction():
            db.execute("UPDATE source_events SET payload_json = '{}' ")
        raise RuntimeError("bounded extractor failure after commit")

    monkeypatch.setattr(beam_ingest, "ingest_source_event", fail_after_redaction)
    stores = tmp_path / "stores"
    with pytest.raises(RuntimeError, match="bounded extractor"):
        beam_ingest.ingest_beam(
            cfg,
            db_dir=stores,
            extractors=cast(Extractors, RecordingExtractors()),
            embedder=mock_embedder,
        )
    failure = json.loads((stores / "beam-1.ingest.failure.json").read_text())
    assert failure["complete"] is False
    assert failure["persisted_event_rows"] == 1
    assert failure["retained_plaintext_messages"] == 0
    assert failure["redacted_events"] == 1
    assert failure["seconds"] >= 0
    assert failure["error"] == "production_source_event_failed"
    assert failure["failure_diagnostic"] == {"category": "RuntimeError", "stage": "native-ingest"}
    assert "bounded extractor" not in json.dumps(failure)
    assert not (stores / "beam-1.ingest.json").exists()
