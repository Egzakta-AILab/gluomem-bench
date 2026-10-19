"""Synchronous native graph evidence adapter; the common reader alone answers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge, verify_native_evidence
from gluomem_bench.bench.cognee_graph import (
    GRAPH_CONDITION,
    NATIVE_SOURCE_PINS,
    CogneeGraphEvidenceBackend,
)
from gluomem_bench.bench.cognee_native_boundary import clean_diagnostic, native_diagnostic
from gluomem_bench.bench.comparison_backends import COGNEE_WHEEL_SHA256, Evidence
from gluomem_bench.bench.rack_trace import WorkloadBudget
from gluomem_bench.bench.types import Prediction, RetrievedDoc


class NativeWorkloadBudget(WorkloadBudget):
    """Retain conservative outer debits even if coordination persistence raises."""

    def begin_call(self):
        before = self.outer_budget.calls if self.outer_budget is not None else None
        local = self.calls
        try:
            super().begin_call()
        except BaseException:
            if before is not None:
                # One synchronous native operation per job; reader never dispatches concurrently.
                debit = self.outer_budget.calls - before
                self.calls = max(self.calls, local + debit)
            self.failed = True
            raise


def save_json(path, value):
    Path(path).write_bytes(
        (
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n"
        ).encode("utf-8")
    )


def verify_graph_receipt(settings, documents, bridge, pins, *, quiescent=False):
    """Input processing is algorithm-neutral; graph lineage is verified separately."""
    if pins != {
        "version": "1.6.5",
        "wheel_sha256": COGNEE_WHEEL_SHA256,
        "source_sha256": NATIVE_SOURCE_PINS,
    }:
        bridge.fail("native graph source mismatch")
        raise ValueError("native graph source pins differ")
    receipt = verify_native_evidence(
        settings.root,
        dataset=settings.dataset,
        expected=documents,
        bridge=bridge,
        quiescent=quiescent,
    )
    coverage = {
        k: v
        for k, v in receipt["coverage"].items()
        if k not in {"condition", "graph_retrieval_qualified", "graph_retrieval_gap"}
    }
    return {
        "condition": GRAPH_CONDITION,
        "coverage_kind": "native-processed-input-readback",
        "input_processing_qualified": True,
        "coverage": coverage,
        "generation": receipt["generation"],
        "source_pins": pins,
        "complete_campaign_admission": False,
    }


class CogneeRetrieverFactory:
    """Single-history factory for run_history; separate native subledger, SAME outer job."""

    def __init__(self, *, settings, profile, outer_budget, out, transport=None, documents=None):
        self.documents = tuple(documents) if documents is not None else None
        self.settings, self.profile = settings, profile
        self.outer_budget, self.out = outer_budget, Path(out)
        if outer_budget is None:
            raise ValueError("shared outer job budget required")
        if settings.root.resolve() != (self.out / "native").resolve():
            raise ValueError("native root must be evidence output/native")
        self.transport = transport
        self.retriever = None
        self.budget = None

    def __call__(self, cfg, conv, answer_fn):
        if self.budget is not None:
            raise RuntimeError("single-use native retriever factory")
        self.outer_budget.check()
        if self.out.exists() or self.out.is_symlink():
            raise FileExistsError("fresh graph evidence output required")
        self.out.mkdir(parents=True)
        self.budget = NativeWorkloadBudget(
            max_calls=self.profile.max_generation_calls,
            max_seconds=self.profile.max_run_seconds,
            policy=self.profile.profile_version,
            outer_budget=self.outer_budget,
        )
        # Retain the object BEFORE initialization so ingestion failures can be closed/accounted.
        self.retriever = CogneeCampaignRetriever(self, cfg, conv, answer_fn)
        self.retriever.prepare()
        return self.retriever

    def close(self):
        if self.retriever is not None:
            self.retriever.close()


class CogneeCampaignRetriever:
    name = "cognee"
    retrieves = True

    def __init__(self, factory, cfg, conv, answer_fn):
        self.factory, self.cfg, self.answer_fn = factory, cfg, answer_fn
        self.documents = (
            factory.documents if factory.documents is not None else session_documents(conv)
        )
        self.runner = asyncio.Runner()
        self.backend = self.bridge = None
        self.pins = None
        self.contexts = []
        self.closed = self.failed = False
        self.failure = None
        self.started = time.monotonic()

    def _fail(self, exc):
        if self.failure is None:
            try:
                self.failure = (
                    clean_diagnostic(getattr(exc, "native_diagnostic", None))
                    if hasattr(exc, "native_diagnostic")
                    else native_diagnostic(exc)
                )
            except Exception:
                self.failure = clean_diagnostic(None)
        self.failed = True
        self.factory.budget.failed = True
        self.factory.outer_budget.failed = True
        if self.bridge is not None:
            self.bridge.fail("native_failure")
        exc.__dict__["native_diagnostic"] = self.failure
        exc.__dict__["native_generation_calls"] = self.factory.budget.calls

    def _bound_timeout(self):
        assert self.backend is not None
        if self.factory.settings.operation_deadline is not None:
            # Sample time BEFORE remaining(): RPC accounting latency cannot extend the clock.
            now = time.monotonic()
            deadline = min(
                self.backend.settings.operation_deadline, now + self.factory.budget.remaining()
            )
            self.backend.settings = replace(self.backend.settings, operation_deadline=deadline)
            return
        self.backend.settings = replace(
            self.backend.settings,
            timeout_seconds=min(
                self.factory.settings.timeout_seconds, self.factory.budget.remaining()
            ),
        )

    def prepare(self):
        f = self.factory
        try:
            self.bridge = CogneeGenerationBridge(
                f.profile,
                budget=f.budget,
                receipt=f.out / "generation.jsonl",
                transport=f.transport,
                operation_deadline=f.settings.operation_deadline,
                cleanup_reserve_seconds=f.settings.cleanup_reserve_seconds,
            )
            self.bridge.__enter__()
            self.backend = CogneeGraphEvidenceBackend(self.bridge.bind(f.settings))
            self._bound_timeout()
            self.runner.run(self.backend.add(self.documents))
            self._bound_timeout()
            self.pins = self.runner.run(self.backend.source_pins())
            receipt = verify_graph_receipt(f.settings, self.documents, self.bridge, self.pins)
            save_json(f.out / "before-search.json", receipt)
        except BaseException as exc:
            self._fail(exc)
            try:
                self.close()
            except BaseException as cleanup:
                exc.add_note("native cleanup failure: " + type(cleanup).__name__)
            raise

    def answer(self, sample):
        f = self.factory
        started = time.monotonic()
        try:
            if self.closed:
                raise RuntimeError("graph retriever closed")
            self.bridge.check()
            self._bound_timeout()
            hits = self.runner.run(
                self.backend.search(sample.question, top_k=self.cfg.benchmark.answer.context_k)
            )
            for hit in hits:
                p = hit.provenance
                if (
                    p.get("condition") != GRAPH_CONDITION
                    or p.get("dataset_name") != f.settings.dataset
                    or not p.get("dataset_id")
                    or not any(ref.get("kind") == "graph_edge" for ref in p["native_evidence"])
                    or p.get("completion") is not None
                ):
                    raise ValueError("native graph evidence lacks scoped lineage")
            receipt = verify_graph_receipt(f.settings, self.documents, self.bridge, self.pins)
            row = {
                "sample_id": sample.id,
                "query": sample.question,
                "hits": [asdict(hit) for hit in hits],
                "native_calls_after_search": f.budget.calls,
                "native_final_completion_used": False,
            }
            self.contexts.append(row)
            save_json(f.out / "graph-contexts.json", self.contexts)
            receipt["graph_context"] = row
            save_json(f.out / f"before-reader-{len(self.contexts):06d}.json", receipt)
            answer = self.answer_fn(sample.reader_question, "\n".join(hit.text for hit in hits))
            self.bridge.check()
            return Prediction(
                sample_id=sample.id,
                answer=answer.text,
                retrieved=tuple(
                    RetrievedDoc(rank=i + 1, text=hit.text, score=hit.score)
                    for i, hit in enumerate(hits)
                ),
                prompt_tokens=answer.prompt_tokens,
                completion_tokens=answer.completion_tokens,
                latency_ms=(time.monotonic() - started) * 1000,
            )
        except BaseException as exc:
            self._fail(exc)
            raise

    def close(self):
        if self.closed:
            return
        f = self.factory
        error = None
        receipt = None
        try:
            if self.backend is not None:
                self.runner.run(self.backend.close())
        except BaseException as exc:
            error = exc
            self._fail(exc)
        finally:
            try:
                if self.bridge is not None:
                    self.bridge.__exit__(None, None, None)
            except BaseException as exc:
                error = error or exc
                self._fail(exc)
            self.runner.close()
            self.closed = True
        try:
            if not self.failed:
                receipt = verify_graph_receipt(
                    f.settings, self.documents, self.bridge, self.pins, quiescent=True
                )
            ledger = {
                "status": "failed" if self.failed else "closed_verified",
                "condition": GRAPH_CONDITION,
                "dataset": f.settings.dataset,
                "generation_calls": f.budget.calls,
                "wall_seconds": time.monotonic() - self.started,
                "native_worker_closed": self.backend is not None and self.backend.worker_closed,
                "native_diagnostic": self.failure,
                "clocks": {
                    key: getattr(self.backend.settings if self.backend else f.settings, key)
                    for key in (
                        "request_timeout_seconds",
                        "operation_deadline",
                        "timeout_seconds",
                        "native_llm_timeout_seconds",
                        "cleanup_reserve_seconds",
                    )
                },
                "proxy_closed": self.bridge is not None and self.bridge._server is None,
                "graph_context_receipts": len(self.contexts),
                "source_pins": self.pins,
                "source_inputs": [asdict(item) for item in self.documents],
                "verification": receipt,
                "generation": receipt["generation"] if receipt else None,
                "artifact_sha256": {
                    p.relative_to(f.out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(f.out.rglob("*"))
                    if p.is_file() and p.name != "native-ledger.json"
                },
            }
            save_json(f.out / "native-ledger.json", ledger)
        except BaseException as exc:
            error = error or exc
            self._fail(exc)
        if error is not None:
            raise error


def session_documents(conv):
    """Complete COMMON allowed projection per session; opaque provenance only."""
    sessions = sorted(conv.sessions, key=lambda s: (s.date_time, s.index))
    return tuple(
        Evidence(
            f"session-{ordinal:06d}",
            f"[{session.date_time.isoformat()}]\n"
            + "\n".join(f"{turn.speaker}: {turn.text}" for turn in session.turns),
            {"session_ordinal": ordinal, "date": session.date_time.isoformat()},
        )
        for ordinal, session in enumerate(sessions, 1)
    )
