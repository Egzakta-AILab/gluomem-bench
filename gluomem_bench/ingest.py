"""LoCoMo benchmark dataset parser and file ingestion helper.

Re-exports `Turn`, `Session`, `Conversation`, `IngestReport`, and
`ingest_conversation` from `gluomem.write.conversation` while keeping
LoCoMo-specific file parsing in `gluomem_bench`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from gluomem.embedding.providers import make_embedder
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.storage.db import MindDB
from gluomem.storage.scope import MemoryScope, resolve_scope
from gluomem.write.conversation import (
    Conversation,
    IngestReport,
    Session,
    Turn,
    _admit_observation,
    admit_observation,
    find_base_frame,
    find_edge_conflict_frames,
    find_subject_frames,
    ingest_conversation,
)
from gluomem.write.gate import make_scanner

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig
    from gluomem.embedding.base import Embedder
    from gluomem.llm.outputs import Extractors

__all__ = [
    "Conversation",
    "IngestReport",
    "Session",
    "Turn",
    "_admit_observation",
    "_parse_locomo_datetime",
    "admit_observation",
    "find_base_frame",
    "find_edge_conflict_frames",
    "find_subject_frames",
    "ingest_conversation",
    "ingest_path",
    "load_locomo",
]

_LOCOMO_DT_FORMATS = (
    "%I:%M %p on %d %B, %Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


def _parse_locomo_datetime(raw: str) -> datetime:
    """Parse a LoCoMo session datetime string against known formats."""
    for fmt in _LOCOMO_DT_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(
        f"unrecognized LoCoMo session datetime {raw!r}; known formats: "
        f"{', '.join(_LOCOMO_DT_FORMATS)}"
    )


def load_locomo(path: str | Path) -> list[Conversation]:
    """Parse a LoCoMo-format JSON file into typed conversations."""
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of LoCoMo samples")

    conversations: list[Conversation] = []
    for i, sample in enumerate(data):
        conv_id = sample.get("sample_id")
        conv = sample.get("conversation")
        if not conv_id or not isinstance(conv, dict):
            raise ValueError(f"{path}: sample {i} missing sample_id/conversation")
        speaker_a = conv.get("speaker_a")
        speaker_b = conv.get("speaker_b")
        if not speaker_a or not speaker_b:
            raise ValueError(f"{path}: sample '{conv_id}' missing speaker_a/b")

        sessions: list[Session] = []
        n = 1
        while f"session_{n}" in conv:
            raw_turns = conv[f"session_{n}"]
            raw_dt = conv.get(f"session_{n}_date_time")
            if raw_dt is None:
                raise ValueError(f"{path}: sample '{conv_id}' session_{n} has no date_time")
            turns: list[Turn] = []
            for t in raw_turns:
                text = t.get("text", "")
                caption = t.get("blip_caption")
                if caption:
                    text = f"{text} [shares photo: {caption}]".strip()
                turns.append(
                    Turn(
                        speaker=t.get("speaker", "unknown"),
                        text=text,
                        dia_id=t.get("dia_id"),
                    )
                )
            sessions.append(Session(index=n, date_time=_parse_locomo_datetime(raw_dt), turns=turns))
            n += 1
        if not sessions:
            raise ValueError(f"{path}: sample '{conv_id}' has no sessions")
        conversations.append(
            Conversation(
                conv_id=str(conv_id),
                speakers=(str(speaker_a), str(speaker_b)),
                sessions=sessions,
            )
        )
    return conversations


def ingest_path(
    path: str | Path,
    cfg: AppConfig,
    *,
    facade: Extractors,
    db_path: str | Path | None = None,
    embedder: Embedder | None = None,
    scope: MemoryScope | None = None,
) -> IngestReport:
    """Wire storage and gate, then ingest every conversation in a LoCoMo file."""
    conversations = load_locomo(path)
    db = MindDB(db_path or cfg.storage.path, dim=cfg.storage.vector_dim)
    if embedder is None:
        embedder = make_embedder(cfg)
    s = resolve_scope(scope)
    store = FrameStore(
        db, embedder=embedder, chunking=cfg.chunking, scope=s, lifecycle=cfg.lifecycle
    )
    kg = KnowledgeGraph(db, scope=s)
    scanner = make_scanner(cfg)

    total = IngestReport()
    for conv in conversations:
        report = ingest_conversation(
            conv,
            facade=facade,
            scanner=scanner,
            store=store,
            kg=kg,
            cfg=cfg,
            scope=s,
        )
        for attr in (
            "conversations",
            "sessions",
            "turns",
            "i_frames",
            "p_frames",
            "b_frames",
            "deduped",
            "semantic_deduped",
            "quarantined",
            "rejected",
            "entities",
            "relations",
            "retired_edges",
            "demoted",
        ):
            setattr(total, attr, getattr(total, attr) + getattr(report, attr))
        total.extraction_errors.extend(report.extraction_errors)
    return total
