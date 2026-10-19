"""Synchronous campaign seam over identical allowed raw-history chunks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

from gluomem_bench.bench.comparison_backends import Evidence, RawEvidenceBackend
from gluomem_bench.bench.types import Prediction, RetrievedDoc


class RawCampaignRetriever:
    """Raw BM25/dense/hybrid; shared reader owns final evidence-token packing."""

    retrieves = True

    def __init__(self, cfg, conv, answer_fn, *, mode, embedder=None, chunk_chars=1000):
        if type(chunk_chars) is not int or chunk_chars < 1:
            raise ValueError("positive integer chunk_chars required")
        self.name = mode
        self.cfg = cfg
        self.answer_fn = answer_fn
        self.embedder = embedder
        self.chunk_chars = chunk_chars
        self.chunks = []
        for session in conv.sessions:
            for turn_index, turn in enumerate(session.turns):
                prefix = f"[{session.date_time.isoformat()}] {turn.speaker}: "
                for offset in range(0, len(turn.text), chunk_chars):
                    source_id = turn.dia_id or (
                        f"{conv.conv_id}/session/{session.index}/{turn_index}")
                    self.chunks.append(Evidence(
                        id=(f"{conv.conv_id}/s{session.index}/t{turn_index}"
                            f"/c{offset // chunk_chars}"),
                        text=prefix + turn.text[offset:offset + chunk_chars],
                        provenance={"source_turn_id": source_id, "session_index": session.index,
                                    "role": turn.speaker, "time": session.date_time.isoformat(),
                                    "offset": offset},
                    ))
        self.backend = RawEvidenceBackend(
            mode, encoder=embedder.embed_batch if mode != "bm25" and embedder else None,
            query_encoder=embedder.embed_query if mode != "bm25" and embedder else None,
        )
        asyncio.run(self.backend.add(self.chunks))

    def answer(self, sample):
        started = time.monotonic()
        hits = asyncio.run(self.backend.search(sample.question,
                                              limit=self.cfg.benchmark.answer.context_k))
        context = "\n".join(hit.text for hit in hits)
        answer = self.answer_fn(sample.reader_question, context)
        return Prediction(
            sample_id=sample.id, answer=answer.text,
            retrieved=tuple(RetrievedDoc(rank=i + 1, text=hit.text, score=hit.score,
                                        dia_id=hit.provenance["source_turn_id"])
                            for i, hit in enumerate(hits)),
            prompt_tokens=answer.prompt_tokens, completion_tokens=answer.completion_tokens,
            latency_ms=(time.monotonic() - started) * 1000,
        )

    def manifest_extra(self):
        materials = [(chunk.id, chunk.text, dict(chunk.provenance)) for chunk in self.chunks]
        return {
            "mode": self.name, "chunk_chars": self.chunk_chars, "overlap": 0,
            "chunks": len(self.chunks), "chunk_sha256": hashlib.sha256(
                json.dumps(materials, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "encoder_fingerprint": self.embedder.fingerprint.as_dict()
            if self.embedder is not None and self.name != "bm25" else None,
            "fusion": "RRF(k=60); full-corpus lanes" if self.name == "hybrid" else None,
        }

    def close(self):
        asyncio.run(self.backend.close())
