"""Synchronous online native-experience memory, not async EvidenceBackend.

Drivers own dataset configuration, encoders and evidence serialization.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

from . import arena_campaign as native
from .embedding_runtime import RemoteEmbedder
from .rack_trace import RackRouter

SYSTEMS = ("no_memory", "bm25", "dense", "hybrid", "full_context", "gluomem")


class ArenaProductionRouter(RackRouter):
    """Scoped authless rack placeholder, never read or mutate ambient credentials."""

    def assert_api_key(self, model_name):
        entry = self._registry.resolve(model_name)
        if (
            entry.api_base != self.client.profile.base_url
            or entry.provider_model != "openai/" + self.client.profile.model
        ):
            raise ValueError("production document role model/endpoint differs")

    def lm_for(self, task):
        lm = super().lm_for(task)
        lm.kwargs["api_key"] = "local-endpoint-placeholder"
        return lm


class BudgetedArenaEmbedder(RemoteEmbedder):
    """Query/document embeddings share the same whole-task deadline as generation."""

    def __init__(self, base_url, *, budget):
        super().__init__(base_url)
        self.budget = budget

    def _encode(self, texts, role):
        self.budget.check()
        self.timeout = min(120, self.budget.remaining())
        result = super()._encode(texts, role)
        self.budget.check()
        return result


class RawExperienceMemory:
    """Online native agent experience controls, isolated per task and method."""

    def __init__(self, mode, *, budget, embedder=None, top_k=10):
        import sqlite3

        if mode not in set(SYSTEMS) - {"gluomem"} or type(top_k) is not int or top_k < 1:
            raise ValueError("unsupported raw memory mode")
        if mode in {"dense", "hybrid"} and embedder is None:
            raise ValueError("explicit real shared encoder required")
        self.mode, self.budget, self.entries = mode, budget, []
        self.embedder, self.top_k, self.vectors = embedder, top_k, []
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE VIRTUAL TABLE docs USING fts5(content)")

    def query(self, question):
        self.budget.check()
        if self.mode == "no_memory" or not self.entries:
            return ""
        if self.mode == "full_context":
            return "\n\n".join(e["content"] for e in self.entries)
        import re

        tokens = re.findall(r"\w+", question)
        query = " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)
        bm25 = (
            [
                row[0] - 1
                for row in self.db.execute(
                    "SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs), rowid", (query,)
                )
            ]
            if query
            else []
        )
        ranked = bm25
        if self.mode in {"dense", "hybrid"}:
            import numpy as np

            matrix = np.asarray(self.vectors, dtype=float)
            q = np.asarray(self.embedder.embed_query(question), dtype=float)
            norm = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
            sims = np.divide(matrix @ q, norm, out=np.zeros(len(matrix)), where=norm != 0)
            dense = sorted(range(len(matrix)), key=lambda i: (-sims[i], i))
            if self.mode == "dense":
                ranked = dense
            else:
                scores = {}
                for ranks in (bm25, dense):
                    for rank, i in enumerate(ranks):
                        scores[i] = scores.get(i, 0) + 1 / (60 + rank + 1)
                ranked = sorted(scores, key=lambda i: (-scores[i], i))
        return "\n\n".join(self.entries[i]["content"] for i in ranked[: self.top_k])

    def insert(self, experience):
        self.budget.check()
        self.entries.append(deepcopy(experience))
        self.db.execute("INSERT INTO docs(content) VALUES (?)", (experience["content"],))
        self.db.commit()
        if self.mode in {"dense", "hybrid"}:
            self.vectors.append(self.embedder.embed_batch([experience["content"]])[0])
        readback = self.db.execute(
            "SELECT content FROM docs WHERE rowid=?", (len(self.entries),)
        ).fetchone()
        if readback != (experience["content"],):
            raise ValueError("raw experience readback mismatch")
        return {
            "event_id": experience["source_ref"],
            "content_sha256": hashlib.sha256(experience["content"].encode()).hexdigest(),
            "readback_verified": True,
        }

    def close(self):
        self.db.close()


class GluoExperienceMemory:
    """Production document SourceEvents, DspyExtractors, admission and HybridSearch.

    Native agent authorship is retained as a document, never relabeled as a user.
    The readback includes committed envelope hashes and frame provenance, even
    when extraction legitimately produces no admissible claims.
    """

    def __init__(self, *, cfg, extractors, embedder, budget, db_path, evidence_path, top_k=10):
        from gluomem.frames.store import FrameStore
        from gluomem.kg.graph import KnowledgeGraph
        from gluomem.llm.modules import DspyExtractors
        from gluomem.search.hybrid import HybridSearch
        from gluomem.storage.db import MindDB

        if type(extractors) is not DspyExtractors:
            raise ValueError("real production DspyExtractors required; no Null/Fake extractor")
        self.cfg, self.extractors, self.embedder, self.budget = cfg, extractors, embedder, budget
        self.path, self.evidence_path, self.top_k = Path(db_path), Path(evidence_path), top_k
        if self.path.exists() or self.evidence_path.exists():
            raise FileExistsError("fresh online memory and evidence required")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = MindDB(self.path, dim=cfg.storage.vector_dim)
        self.store = FrameStore(
            self.db, embedder=embedder, chunking=cfg.chunking, lifecycle=cfg.lifecycle
        )
        self.kg = KnowledgeGraph(self.db)
        self.search = HybridSearch(
            self.db, self.store, cfg, embedder=embedder, reranker=None, kg=self.kg
        )
        self.events, self.receipts = [], []

    def query(self, question):
        self.budget.check()
        if not self.events:
            return ""
        result = self.search.search(question, limit=self.top_k)
        self.budget.check()
        return "" if result.abstained else "\n\n".join(hit.frame.content for hit in result.hits)

    def insert(self, experience):
        import sqlite3

        from gluomem.write.events import ingest_source_event

        from .beam_ingest import _read_rows, _validate_rows

        self.budget.check()
        if (
            experience.get("authorship") != "agent"
            or experience.get("feedback_in_memory") is not False
        ):
            raise ValueError("native agent-authored document without evaluator feedback required")
        event = native.document_experience(experience)
        result = ingest_source_event(
            event, store=self.store, kg=self.kg, cfg=self.cfg, extractors=self.extractors
        )
        events = [*self.events, event]
        committed, retention = _validate_rows(_read_rows(self.path), events)
        if (
            result.deduplicated
            or committed[:-1] != self.receipts
            or result.event_row_id != committed[-1]["row_id"]
        ):
            raise ValueError("production source-event commit/readback mismatch")
        reader = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        reader.row_factory = sqlite3.Row
        try:
            provenance = [
                dict(r)
                for r in reader.execute(
                    "SELECT frame_id, source_event_id, source_type, source_ref, "
                    "message_id, verbatim_span "
                    "FROM memory_provenance WHERE source_event_id=? ORDER BY frame_id",
                    (result.event_row_id,),
                )
            ]
            frame_ids = {f.id for f in result.frames}
            if not frame_ids <= {r["frame_id"] for r in provenance} or any(
                r["source_type"] != "document"
                or r["source_ref"] != event.source_ref
                or r["message_id"] is not None
                for r in provenance
            ):
                raise ValueError("production document frame provenance mismatch")
        finally:
            reader.close()
        receipt = {
            **committed[-1],
            "readback_verified": True,
            "provenance_count": len(provenance),
            "provenance": provenance,
            "content_sha256": hashlib.sha256(experience["content"].encode()).hexdigest(),
            "frames": [
                {"id": f.id, "content": f.content, "source": f.source, "kind": f.kind}
                for f in result.frames
            ],
        }
        native._append(
            self.evidence_path, {"event": event.model_dump(mode="json"), "receipt": receipt}
        )
        self.events, self.receipts = events, committed
        self.budget.check()
        return receipt

    def close(self):
        self.db.close()


def production_memory(cfg, *, client, embedder, budget, out):
    """Assemble native document memory; callers own config and evidence serialization."""
    from gluomem.llm.modules import DspyExtractors

    router = ArenaProductionRouter(
        cfg, client=client, trace=out / "production-raw.jsonl", budget=budget
    )
    return GluoExperienceMemory(
        cfg=cfg, extractors=DspyExtractors(cfg, router=router), embedder=embedder, budget=budget,
        db_path=out / "production.mind", evidence_path=out / "memory-events.jsonl", top_k=10,
    )
