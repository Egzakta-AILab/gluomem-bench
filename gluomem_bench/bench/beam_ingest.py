"""BEAM ingestion through production extraction/consolidation SourceEvent API.

Never set explicit kind (which would bypass extraction), never seed gold, and
never translate assistant roles into user. Core currently excludes assistant
messages from claim extraction; the receipt exposes this limitation explicitly.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from gluomem.embedding.providers import make_embedder
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.storage.db import MindDB
from gluomem.write.events import SourceEvent, SourceEventMessage, ingest_source_event

from gluomem_bench.bench.adapters.beam import BeamAdapter

if TYPE_CHECKING:
    from gluomem.embedding.base import Embedder
    from gluomem.llm.outputs import Extractors

    from gluomem_bench.config import AppConfig
    from gluomem_bench.ingest import Conversation


def _events(conv: Conversation) -> list[SourceEvent]:
    return [
        SourceEvent(
            event_id=f"{conv.conv_id}:session:{session.index}",
            source_ref=f"beam/{conv.conv_id}/{session.index}",
            source_type="chat_turn",
            event_time=session.date_time.isoformat(),
            messages=[
                SourceEventMessage(
                    message_id=str(turn.dia_id),
                    role=turn.speaker,
                    content=turn.text,
                    timestamp=session.date_time.isoformat(),
                )
                for turn in session.turns
            ],
        )
        for session in conv.sessions
    ]


def _config_sha(cfg: AppConfig) -> str:
    return hashlib.sha256(
        json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()
    ).hexdigest()


COVERAGE_PROTOCOL = "beam:committed-input-receipts@v2"


def _expected_receipt(event: SourceEvent) -> dict:
    # FrameStore is deliberately constructed without a caller-provided scope.
    # Trust the production default, never a scope claimed by an artifact/DB row.
    from gluomem.storage.scope import DEFAULT_SCOPE
    from gluomem.write.events import _canonical_event_payload

    _, digest = _canonical_event_payload(event.model_copy(update={"scope": DEFAULT_SCOPE}))
    return {
        **DEFAULT_SCOPE.as_dict(),
        "event_id": event.event_id,
        "revision": event.revision,
        "source_type": event.source_type,
        "source_ref": event.source_ref,
        "payload_hash": digest,
        "messages": len(event.messages),
    }


def _read_rows(path: Path) -> list[dict]:
    # Read-only sees committed WAL data; do not use immutable=1 on a live DB.
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in db.execute("SELECT * FROM source_events ORDER BY id")]
    finally:
        db.close()


def _validate_rows(rows: list[dict], events: list[SourceEvent]) -> tuple[list[dict], dict]:
    if len(rows) != len(events):
        raise ValueError("incomplete source-event coverage")
    receipts = []
    retained = redacted = 0
    for row, event in zip(rows, events, strict=True):
        expected = _expected_receipt(event)
        if any(row.get(k) != v for k, v in expected.items() if k != "messages"):
            raise ValueError("incompatible source-event payload or scoped receipt")
        payload = json.loads(row["payload_json"])
        if payload == {}:
            redacted += 1
        else:
            from gluomem.storage.scope import DEFAULT_SCOPE
            from gluomem.write.events import _canonical_event_payload

            encoded, _ = _canonical_event_payload(event.model_copy(update={"scope": DEFAULT_SCOPE}))
            if payload != json.loads(encoded):
                raise ValueError("incompatible source-event payload")
            retained += len(event.messages)
        receipts.append({"row_id": row["id"], **expected})
    return receipts, {"retained_plaintext_messages": retained, "redacted_events": redacted}


def verify_beam_ingest(
    cfg: AppConfig, adapter: BeamAdapter, db_dir: Path, conv_ids: set[str]
) -> list[dict]:
    reports = []
    adapter.verify()
    conversations = {c.conv_id: c for c in adapter.conversations()}
    if not conv_ids <= conversations.keys():
        raise ValueError("unknown BEAM history selection")
    for cid in sorted(conv_ids):
        sentinel = db_dir / f"{cid}.ingest.json"
        db_path = db_dir / f"{cid}.mind"
        if not sentinel.is_file() or not db_path.is_file():
            raise ValueError(f"incomplete BEAM ingest: {cid}")
        meta = json.loads(sentinel.read_text(encoding="utf-8"))
        events = _events(conversations[cid])
        if (
            meta.get("coverage_protocol") != COVERAGE_PROTOCOL
            or meta.get("dataset_sha256") != adapter.sha
            or meta.get("config_sha256") != _config_sha(cfg)
            or meta.get("conv_id") != cid
            or meta.get("path") != "production_source_event"
            or type(meta.get("events")) is not int
            or type(meta.get("messages")) is not int
            or meta.get("events") != len(events)
            or meta.get("messages") != sum(len(e.messages) for e in events)
            or meta.get("complete") is not True
        ):
            raise ValueError(f"incomplete or incompatible BEAM ingest: {cid}")
        receipts, retention = _validate_rows(_read_rows(db_path), events)
        if json.dumps(meta.get("committed_receipts"), sort_keys=True) != json.dumps(
            receipts, sort_keys=True
        ) or any(type(meta.get(k)) is not int or meta.get(k) != v for k, v in retention.items()):
            raise ValueError(f"incompatible committed receipt coverage: {cid}")
        reports.append(meta)
    return reports


def select_fresh_beam_ingest(
    cfg: AppConfig,
    db_dir: Path,
    conv_ids: set[str] | None,
    *,
    include_shared_traces: bool = False,
) -> tuple[BeamAdapter, list[Conversation]]:
    """Read-only selection-wide refusal, before any run-owned writes or models.

    CLI owns shared traces and checks them before preflight. API callers may
    already own a trace (e.g. a shared whole-run driver), so the API checks the
    selected stores/receipts but does not claim ownership of caller traces.
    """
    adapter = BeamAdapter(cfg)
    adapter.verify()
    conversations = adapter.conversations()
    if conv_ids is not None:
        if not conv_ids <= {c.conv_id for c in conversations}:
            raise ValueError("unknown BEAM history selection")
        conversations = [c for c in conversations if c.conv_id in conv_ids]
    paths = [
        db_dir / f"{conv.conv_id}{suffix}"
        for conv in conversations
        for suffix in (".mind", ".mind-wal", ".mind-shm", ".ingest.json", ".ingest.failure.json")
    ]
    if include_shared_traces:
        paths.extend(
            db_dir / name for name in ("calls.jsonl", "production-raw.jsonl", "failure.json")
        )
    for path in paths:
        if path.exists():
            raise FileExistsError(f"{path}: fresh ingestion required; no partial resume")
    return adapter, conversations


def ingest_beam(
    cfg: AppConfig,
    *,
    db_dir: Path,
    extractors: Extractors,
    embedder: Embedder | None = None,
    conv_ids: set[str] | None = None,
    completion_guard: Callable[[], None] | None = None,
) -> list[dict]:
    from gluomem.llm.fake import NullExtractors

    if extractors is None or isinstance(extractors, NullExtractors):
        raise ValueError("BEAM production ingestion requires active extraction/consolidation")
    adapter, conversations = select_fresh_beam_ingest(cfg, db_dir, conv_ids)
    db_dir.mkdir(parents=True, exist_ok=True)
    if embedder is None:
        embedder = make_embedder(cfg)
    try:
        for conv in conversations:
            _ingest_conversation(cfg, adapter, db_dir, extractors, embedder, conv, completion_guard)
        reports = verify_beam_ingest(cfg, adapter, db_dir, {c.conv_id for c in conversations})
        if completion_guard is not None:
            completion_guard()
        return reports
    except Exception:
        for conv in conversations:
            (db_dir / f"{conv.conv_id}.ingest.json").unlink(missing_ok=True)
        raise


def _write_meta(path: Path, meta: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _failure_diagnostic(error, stage):
    """Bounded static categories only; validation values/messages never persist."""
    from pydantic import ValidationError
    from gluomem_bench.bench.embedding_runtime import safe_failure_diagnostic

    result = {'category': safe_failure_diagnostic(error)['type'], 'stage': stage}
    fields = frozenset(('claims', 'text', 'kind', 'confidence', 'event_date', 'verbatim_span',
        'message_id', 'entities', 'relations', 'entity_type', 'name', 'properties', 'source',
        'target', 'relation_type', 'decision', 'action', 'target_frame_id', 'rationale'))
    types = frozenset(('literal_error', 'missing', 'datetime_from_date_parsing',
        'datetime_parsing', 'datetime_type', 'string_type', 'list_type', 'dict_type',
        'model_type', 'float_parsing', 'float_type', 'int_parsing', 'int_type',
        'greater_than_equal', 'less_than_equal', 'value_error', 'extra_forbidden'))
    pending, seen = [error], set()
    for exc in pending:
        if id(exc) in seen:
            continue
        seen.add(id(exc))
        if len(seen) > 16:
            break
        if type(exc) is ValidationError:
            rows = exc.errors(include_url=False, include_context=False, include_input=False)
            result.update(category='validation-error', validation_errors=[{
                'type': row['type'] if row['type'] in types else 'unknown',
                'loc': [part if (type(part) is str and part in fields)
                        or (type(part) is int and 0 <= part < 2**31) else 'unknown'
                        for part in row['loc'][:8]],
            } for row in rows[:16]])
            break
        pending.extend(v for v in (exc.__cause__, exc.__context__) if v is not None)
    return result


def _ingest_conversation(
    cfg, adapter, db_dir, extractors, embedder, conv, completion_guard,
    *, source_events=None, coverage_protocol=COVERAGE_PROTOCOL,
):
    from gluomem.llm.fake import NullExtractors

    if extractors is None or isinstance(extractors, NullExtractors):
        raise ValueError("production ingestion requires active extraction/consolidation")
    db_path = db_dir / f"{conv.conv_id}.mind"
    sentinel = db_dir / f"{conv.conv_id}.ingest.json"
    if db_path.exists() or sentinel.exists():
        raise FileExistsError(f"{db_path}: fresh ingestion required; no partial resume")
    started = time.perf_counter()
    events = _events(conv) if source_events is None else list(source_events)
    committed = []
    db = MindDB(db_path, dim=cfg.storage.vector_dim)
    stage = 'store-setup'
    try:
        store = FrameStore(db, embedder=embedder, chunking=cfg.chunking, lifecycle=cfg.lifecycle)
        kg = KnowledgeGraph(db)
        for index, event in enumerate(events):
            stage = 'completion-guard'
            if completion_guard is not None:
                completion_guard()
            stage = 'native-ingest'
            result = ingest_source_event(event, store=store, kg=kg, cfg=cfg, extractors=extractors)
            stage = 'commit-coverage'
            observed, _ = _validate_rows(_read_rows(db_path), events[: index + 1])
            if (
                result is None
                or result.deduplicated
                or result.event_row_id != observed[-1]["row_id"]
                or result.event_id != event.event_id
                or result.revision != event.revision
            ):
                raise ValueError("missing or incompatible source-event commit acknowledgement")
            if observed[:-1] != committed:
                raise ValueError("changed prior source-event commit receipt")
            committed.append(observed[-1])
            stage = 'completion-guard'
            if completion_guard is not None:
                completion_guard()
        stage = 'final-coverage'
        db.close()
        receipts, retention = _validate_rows(_read_rows(db_path), events)
        if receipts != committed:
            raise ValueError("changed committed source-event coverage")
        stage = 'metadata'
        usage = getattr(extractors, "usage", None)
        meta = {
            "coverage_protocol": coverage_protocol,
            "committed_receipts": committed,
            **retention,
            "conv_id": conv.conv_id,
            "path": "production_source_event",
            "complete": True,
            "events": len(events),
            "messages": sum(len(e.messages) for e in events),
            "assistant_claim_extraction": "excluded_by_core_policy; original roles preserved",
            "dataset_sha256": adapter.sha,
            "config_sha256": _config_sha(cfg),
            "seconds": time.perf_counter() - started,
            "usage": usage.snapshot() if usage is not None else {},
        }
        stage = 'completion-guard'
        if completion_guard is not None:
            completion_guard()
        stage = 'metadata-write'
        _write_meta(sentinel, meta)
    except Exception as exc:
        sentinel.unlink(missing_ok=True)
        failure = {
            "complete": False,
            "coverage_protocol": coverage_protocol,
            "conv_id": conv.conv_id,
            "error": "production_source_event_failed",
            "failure_diagnostic": _failure_diagnostic(exc, stage),
            "seconds": time.perf_counter() - started,
            "expected_events": len(events),
            "expected_messages": sum(len(e.messages) for e in events),
            "committed_receipts": committed,
        }
        try:
            rows = _read_rows(db_path)
            payloads = [json.loads(r["payload_json"]) for r in rows]
            failure.update(
                persisted_event_rows=len(rows),
                retained_plaintext_messages=sum(len(v.get("messages", [])) for v in payloads),
                redacted_events=sum(v == {} for v in payloads),
            )
        except Exception as diagnostic_exc:
            failure["readback_error"] = "source_event_readback_failed"
            failure["readback_diagnostic"] = _failure_diagnostic(diagnostic_exc, "failure-readback")
        try:
            _write_meta(db_dir / f"{conv.conv_id}.ingest.failure.json", failure)
        except Exception:
            # Failure telemetry must not replace the initiating exception.
            exc.__dict__['ingest_failure_receipt_persisted'] = False
        raise
    finally:
        db.close()
