"""Auditable, adapted BEAM rubric evaluation (not native paper headline scores).

Each rubric nugget receives 0/0.5/1; questions average their nuggets, then
questions average within categories. Event-ordering here is rubric compliance,
NOT the upstream report's Kendall tau metric. Never label it QA accuracy.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from gluomem_bench.bench.adapters.beam import CODE_REVISION, HF_REVISION, BeamAdapter
from gluomem_bench.bench.baselines import Bm25Retriever, transcript
from gluomem_bench.bench.rack import AnswerFormatError, RackClient
from gluomem_bench.llm.outputs import AnswerResult

if TYPE_CHECKING:
    from gluomem_bench.config import AppConfig

PROTOCOL = "adapted:beam_128k_rubric@v1"
ARTIFACTS = (
    "predictions.jsonl",
    "judgments.jsonl",
    "evidence.jsonl",
    "ingest.json",
    "usage.json",
    "timing.json",
    "calls.jsonl",
)


def _json(path: Path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _append(path: Path, data):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_beam(parquet: Path, out: Path) -> dict:
    """Export only the SHA-pinned official tier; never auto-download data."""
    from gluomem_bench.bench.adapters.beam import PARQUET_SHA256, PARQUET_URL

    if _hash(parquet) != PARQUET_SHA256:
        raise ValueError("BEAM parquet sha256 mismatch")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("BEAM preparation requires uv sync --extra beam") from exc
    rows = pq.read_table(parquet).to_pylist()
    if len(rows) != 20:
        raise ValueError("official BEAM 128K tier requires 20 histories")
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _json(out, rows)
    manifest = {
        "source_url": PARQUET_URL,
        "source_sha256": PARQUET_SHA256,
        "revision": HF_REVISION,
        "storage_split": "100K",
        "tier": "128K",
        "export_sha256": _hash(out),
        "histories": len(rows),
        "license": "CC-BY-SA-4.0",
    }
    _json(out.with_suffix(".manifest.json"), manifest)
    return manifest


def rescore_beam(out: Path, *, check: bool = False) -> dict:
    """No model, dataset or config access; validate coverage and artifact hashes."""
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("unsupported BEAM scoring protocol")
    for name in ARTIFACTS:
        if _hash(out / name) != manifest["artifact_sha256"][name]:
            raise ValueError(f"artifact hash mismatch: {name}")
    predictions = [
        json.loads(line)
        for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    judgments = [
        json.loads(line)
        for line in (out / "judgments.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if [row["sample"]["id"] for row in predictions] != manifest["sample_ids"]:
        raise ValueError("prediction coverage mismatch")
    expected = [(p["sample"]["id"], i) for p in predictions for i in range(len(p["rubric"]))]
    if [(row["sample_id"], row["nugget_index"]) for row in judgments] != expected:
        raise ValueError("judgment coverage mismatch")
    per_sample = defaultdict(list)
    lookup = {p["sample"]["id"]: p for p in predictions}
    for row in judgments:
        pred = lookup[row["sample_id"]]
        if row["rubric"] != pred["rubric"][row["nugget_index"]]:
            raise ValueError("judgment rubric mismatch")
        if type(row["score"]) not in (int, float) or row["score"] not in (0, 0.5, 1):
            raise ValueError("invalid nugget score")
        per_sample[row["sample_id"]].append(row["score"])
    categories = defaultdict(list)
    for pred in predictions:
        values = per_sample[pred["sample"]["id"]]
        categories[pred["sample"]["category_name"]].append(sum(values) / len(values))
    values = [sum(v) / len(v) for v in per_sample.values()]
    scores = {
        "protocol": PROTOCOL,
        "questions": len(predictions),
        "nuggets": len(judgments),
        "rubric_question_mean": sum(values) / len(values),
        "by_category": {
            cat: {"mean": sum(v) / len(v), "n": len(v)} for cat, v in sorted(categories.items())
        },
        "answer_failures": sum(p["failure"] is not None for p in predictions),
        "judge_failures": sum(j["failure"] is not None for j in judgments),
    }
    report = (
        "# BEAM 128K adapted rubric evaluation\n\n"
        "Not a native BEAM headline score. Event ordering is rubric compliance, not tau.\n\n"
        f"Questions: {scores['questions']}; nuggets: {scores['nuggets']}; "
        f"question-mean rubric compliance: {scores['rubric_question_mean']:.6f}\n"
    )
    if check:
        if json.loads((out / "scores.json").read_text(encoding="utf-8")) != scores:
            raise ValueError("scores differ from offline rescore")
        if (out / "report.md").read_text(encoding="utf-8") != report:
            raise ValueError("report differs from offline rescore")
    else:
        _json(out / "scores.json", scores)
        (out / "report.md").write_text(report, encoding="utf-8")
    return scores


def run_beam(
    cfg: AppConfig,
    *,
    client: RackClient,
    out: Path,
    retriever: str = "full_context",
    limit: int | None = None,
    db_dir: Path | None = None,
    embedder=None,
) -> dict:
    """Fresh run only; fatal failures preserve actual progress without a score."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    try:
        return _run_beam(
            cfg,
            client=client,
            out=out,
            retriever=retriever,
            limit=limit,
            db_dir=db_dir,
            embedder=embedder,
        )
    except Exception as exc:
        out.mkdir(parents=True, exist_ok=True)
        failure = {
            "complete": False,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_seconds": time.perf_counter() - started,
            "generation_calls": client.budget.calls if client.budget is not None else None,
            "budget_failed": client.budget.failed if client.budget is not None else None,
        }
        for name in ("predictions", "judgments", "evidence"):
            path = out / f"{name}.jsonl"
            failure[name] = (
                len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0
            )
        (out / "manifest.json").unlink(missing_ok=True)
        if (out / "ingest.json").exists():
            ingest = json.loads((out / "ingest.json").read_text(encoding="utf-8"))
            ingest["complete"] = False
            _json(out / "ingest.json", ingest)
        _json(out / "failure.json", failure)
        raise


def _run_beam(
    cfg: AppConfig,
    *,
    client: RackClient,
    out: Path,
    retriever: str = "full_context",
    limit: int | None = None,
    db_dir: Path | None = None,
    embedder=None,
) -> dict:
    """Sequential runner; preserves every selected failure in the denominator."""
    started = time.perf_counter()
    client.check()
    adapter = BeamAdapter(cfg)
    adapter.verify()
    samples = adapter.load()
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        samples = samples[:limit]
    if not samples:
        raise ValueError("empty BEAM selection")
    if retriever not in ("full_context", "bm25", "gluomem"):
        raise ValueError("unsupported BEAM retriever")
    selected_ids = {s.conv_id for s in samples}
    conversations = {c.conv_id: c for c in adapter.conversations() if c.conv_id in selected_ids}
    ingest_receipts = None
    if retriever == "gluomem":
        from gluomem_bench.bench.beam_ingest import verify_beam_ingest

        if db_dir is None:
            raise ValueError("gluomem requires --db-dir with complete production ingest")
        ingest_receipts = verify_beam_ingest(cfg, adapter, db_dir, selected_ids)
    context = ""

    def answer_fn(question: str, evidence: str) -> AnswerResult:
        nonlocal context
        context = evidence
        return AnswerResult(
            text=client.complete(
                "answer",
                {"question": question, "context": evidence},
                {"answer": {"type": "string"}},
            )["answer"]
        )

    engine = None
    if retriever == "bm25":
        engine = Bm25Retriever(cfg, list(conversations.values()), answer_fn)
    elif retriever == "gluomem":
        from gluomem_bench.bench.retriever import GluoMemRetriever

        assert db_dir is not None
        engine = GluoMemRetriever(
            cfg,
            db_dir=db_dir,
            answer_fn=answer_fn,
            content_to_dias=adapter.content_to_dias(),
            embedder=embedder,
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for name in ("predictions.jsonl", "judgments.jsonl", "evidence.jsonl"):
        (out / name).touch()
    rubrics = adapter.rubrics()
    _json(
        out / "ingest.json",
        {
            "path": "production_source_event"
            if ingest_receipts is not None
            else "raw_transcript_baseline",
            "complete": True,
            "receipts": ingest_receipts,
            "histories": [
                {
                    "id": cid,
                    "sessions": len(c.sessions),
                    "turns": sum(len(s.turns) for s in c.sessions),
                }
                for cid, c in conversations.items()
            ],
        },
    )

    def close_engine():
        if engine is not None:
            for stack in getattr(engine, "_stacks", {}).values():
                stack._db.close()

    def check_workload():
        client.check()

    try:
        client.preflight()
        for sample in samples:
            check_workload()
            context = ""
            failure = None
            answer = ""
            retrieved = []
            try:
                if engine is None:
                    answer = answer_fn(
                        sample.question, transcript(conversations[sample.conv_id])
                    ).text
                else:
                    prediction = engine.answer(sample)
                    answer = prediction.answer
                    retrieved = [asdict(doc) for doc in prediction.retrieved]
            except AnswerFormatError as exc:
                check_workload()
                failure = f"{type(exc).__name__}: {exc}"
            _append(
                out / "evidence.jsonl",
                {"sample_id": sample.id, "context": context, "retrieved": retrieved},
            )
            _append(
                out / "predictions.jsonl",
                {
                    "sample": asdict(sample),
                    "answer": answer,
                    "rubric": rubrics[sample.id],
                    "failure": failure,
                },
            )
            for index, nugget in enumerate(rubrics[sample.id]):
                judge_failure = failure
                verdict = {"score": 0, "reason": "answer failed"}
                if failure is None:
                    verdict = client.complete(
                        "rubric",
                        {"question": sample.question, "rubric": nugget, "prediction": answer},
                        {
                            "score": {"type": "number", "enum": [0, 0.5, 1]},
                            "reason": {"type": "string"},
                        },
                    )
                _append(
                    out / "judgments.jsonl",
                    {
                        "sample_id": sample.id,
                        "nugget_index": index,
                        "rubric": nugget,
                        **verdict,
                        "failure": judge_failure,
                    },
                )
        check_workload()
    finally:
        close_engine()
    calls = [json.loads(line) for line in client.receipt.read_text(encoding="utf-8").splitlines()]
    usage = defaultdict(int)
    for call in calls:
        for key, value in call.get("response", {}).get("usage", {}).items():
            if isinstance(value, int):
                usage[key] += value
    _json(
        out / "usage.json",
        {"tokens": dict(usage), "requests": len(calls), "cost_usd": None, "commercial_calls": 0},
    )
    _json(
        out / "timing.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "request_seconds": sum(c["elapsed_seconds"] for c in calls),
        },
    )
    _json(
        out / "manifest.json",
        {
            "protocol": PROTOCOL,
            "official_result": False,
            "dataset_sha256": adapter.sha,
            "source": "Mohammadta/BEAM",
            "tier": "128K",
            "storage_split": "100K",
            "hf_revision": HF_REVISION,
            "code_revision": CODE_REVISION,
            "license": "CC-BY-SA-4.0 (HF dataset); MIT (upstream code)",
            "model": client.profile.model_dump(),
            "retriever": retriever,
            "sample_ids": [s.id for s in samples],
            "config": cfg.model_dump(mode="json"),
            "artifact_sha256": {name: _hash(out / name) for name in ARTIFACTS},
        },
    )
    return rescore_beam(out)
