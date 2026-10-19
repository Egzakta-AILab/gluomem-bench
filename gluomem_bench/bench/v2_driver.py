"""Full LongMemEval-V2 native-input text-history execution (separate from frozen V3).

No history pixels are encoded by this declared method. Original trajectory objects
and screenshot references remain intact; every question image goes to the reader.
No imports of upstream clients, commercial SDKs or production memories at import.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import mimetypes
import os
import re
import typing
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from gluomem_bench.bench.error_privacy import safe_error, safe_kind

PROTOCOL = "longmemeval-v2:native-input-text-history@v1"
CODE_REVISION = "2cc8c540bdb87fe6761629b585e727e1c4704520"
DATA_REVISION = "f152293e235517d504809563c833d7190b8c713b"
METHODS = ("no_memory", "bm25", "dense", "hybrid", "full_context", "gluomem", "mem0", "cognee")


def source_documents(data, group, *, chunk_chars=65536):
    """Lossless per-trajectory AX-text projection, matching audited field policy.

    Chunking changes document units, never selects states or shortens histories.
    History screenshot refs stay in originals and out-of-band coverage only.
    """
    if type(chunk_chars) is not int or chunk_chars < 1:
        raise ValueError("positive chunk_chars required")
    docs, ledgers, refs = [], [], []
    states = chars = 0
    for ordinal, trajectory in enumerate(data.histories(group)):
        projected = {
            key: trajectory[key]
            for key in ("domain", "environment", "goal", "outcome", "start_url")
            if key in trajectory
        }
        projected["trajectory_ordinal"] = ordinal
        projected["states"] = [
            {
                key: state[key]
                for key in ("state_index", "step", "url", "action", "thought", "accessibility_tree")
                if key in state
            }
            for state in trajectory["states"]
        ]
        text = json.dumps(projected, ensure_ascii=False, separators=(",", ":"))
        parts = []
        for offset in range(0, len(text), chunk_chars):
            value = text[offset : offset + chunk_chars]
            doc = {
                "id": f"{group.id}/trajectory/{ordinal}/chunk/{offset // chunk_chars}",
                "text": value,
                "trajectory_ordinal": ordinal,
                "offset": offset,
                "sha256": hashlib.sha256(value.encode()).hexdigest(),
            }
            docs.append(doc)
            parts.append(value)
        if "".join(parts) != text:
            raise ValueError("lossless source coverage failed")
        ledgers.append(
            {
                "trajectory_ordinal": ordinal,
                "source_object_sha256": digest(trajectory),
                "projection_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "chars": len(text),
                "states": len(trajectory["states"]),
            }
        )
        refs.extend(s["screenshot"] for s in trajectory["states"] if s.get("screenshot"))
        chars += len(text)
        states += len(trajectory["states"])
    if len(ledgers) != len(group.trajectory_ids):
        raise ValueError("incomplete native haystack")
    return docs, {
        "complete": True,
        "group_id": group.id,
        "trajectories": len(ledgers),
        "states": states,
        "chars": chars,
        "events": len(docs),
        "chunk_chars": chunk_chars,
        "projection": "ordered complete AX-text; incoming/destination actions unchanged",
        "documents_sha256": digest(docs),
        "trajectory_ledger": ledgers,
        "history_screenshot_refs": refs,
        "source_world_time": None,
    }


def document_events(docs):
    from gluomem.write.events import SourceEvent

    return [
        SourceEvent(
            event_id=d["id"],
            source_ref=f"longmemeval-v2/{d['id']}",
            source_type="document",
            content=d["text"],
        )
        for d in docs
    ]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified(root, name, pins):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path escapes root")
    pin = pins.get(name, "")
    if len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
        raise ValueError(f"SHA256 pin required: {name}")
    if file_hash(path) != pin:
        raise ValueError(f"SHA256 mismatch: {name}")
    return path


@dataclass(frozen=True)
class HistoryGroup:
    id: str
    tier: str
    domain: str
    trajectory_ids: tuple[str, ...]
    question_ids: tuple[str, ...]


def write_json(path, value):
    Path(path).write_bytes(
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    )


def qualify_vision(profile, native_root, out, *, transport=None):
    """At most four authored calls/120s; exact native reader payload, no QA.

    Raw endpoint receipts are private operator artifacts. A known failure is a
    blocker, never repaired by guessing an image token allowance or retrying.
    """
    import struct
    import time
    import zlib

    from gluomem_bench.bench.rack import RackClient
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    native = NativeInterface(native_root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(b"\x00" + b"\xff\x00\x00" * 32 for _ in range(32))))
        + chunk(b"IEND", b"")
    )
    image = out / "authored-red-square.png"
    image.write_bytes(png)
    budget = WorkloadBudget(max_calls=4, max_seconds=120)
    client = RackClient(profile, receipt=out / "calls.jsonl", transport=transport, budget=budget)
    receipt = {
        "kind": "authored-native-v2-reader-vision-transport",
        "qualified": False,
        "benchmark_questions_used": 0,
        "max_generation_calls": 4,
        "max_seconds": 120,
        "model": profile.model,
        "profile_sha256": digest(profile.model_dump(mode="json")),
        "native": native.receipt,
        "image_sha256": file_hash(image),
        "cases": [],
    }
    started = time.monotonic()
    try:
        models = client._request("GET", "/models")
        if profile.model not in {m.get("id") for m in models.get("data", [])}:
            raise ValueError("served model identity not present")
        native.use_served_context_counter(client)
        processor_probe = [
            {"type": "text", "value": "Public synthetic red square."},
            {"type": "image", "value": str(image)},
        ]
        _, processor_tokens, _ = native.truncate(processor_probe, max_tokens=4096)
        if type(processor_tokens) is not int or processor_tokens <= 0:
            raise ValueError("actual served image-aware memory processor count absent")
        receipt["memory_context_processor_probe_tokens"] = processor_tokens
        receipt["memory_context_actual_served_processor"] = True
        for domain, context in [
            ("web", []),
            ("enterprise", [{"type": "text", "value": "Authored memory: visual inspection only."}]),
        ]:
            question = (
                "What is the dominant color in the attached square? "
                "Answer with one color in \\boxed{}."
            )
            messages = native.messages(domain, question, str(image), context)
            without_image = native.messages(domain, question, None, context)
            image_count = client.prompt_token_count(messages)
            text_count = client.prompt_token_count(without_image)
            if (
                type(image_count) is not int
                or type(text_count) is not int
                or image_count <= text_count
            ):
                raise ValueError("served tokenizer does not account for actual query image input")
            result = client.generate_messages("answer", messages)
            case = {
                "domain": domain,
                "messages": messages,
                "image_prompt_tokens": image_count,
                "text_prompt_tokens": text_count,
                "result": result,
            }
            receipt["cases"].append(case)
            if result["prompt_tokens"] != image_count:
                raise ValueError(
                    "actual response usage prompt_tokens differs from served image tokenizer count"
                )
            if "red" not in native.ns["extract_boxed_answer"](result["text"]).lower():
                raise ValueError("authored red-square visual assertion failed")
        receipt["qualified"] = True
        receipt["tokenizer_usage_parity"] = True
    except Exception as exc:
        receipt["blocker"] = safe_error(exc)
        receipt["tokenizer_usage_parity"] = False
    finally:
        client.close()
        receipt["generation_calls"] = budget.calls
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["raw_receipts_sha256"] = (
            file_hash(out / "calls.jsonl") if (out / "calls.jsonl").exists() else None
        )
        write_json(out / "qualification.json", receipt)
    return receipt


class ContextNotApplicable(ValueError):
    """Measured complete full-history input cannot fit; not a QA zero."""


def reader_pack(native, client, question, items, *, full_fit_only=False, reader_tokens=4096):
    """Native whole-item truncation followed by exact complete-prompt packing.

    4096 is the complete reader INPUT operating point, not a character budget.
    Query images/system/question count toward it. Output/schema are additionally
    reserved against the served context window. Full history never truncates.
    """
    if full_fit_only:
        kept = list(items)
        original_tokens = native.ns["count_memory_context_tokens"](
            kept, native.ns["load_memory_context_images"](kept)
        )
        native_tokens = original_tokens
    else:
        kept, original_tokens, native_tokens = native.truncate(items, max_tokens=reader_tokens)
    counts = []

    def count(prefix):
        messages = native.messages(
            question["domain"], question["question"], question["image"], prefix
        )
        n = client.prompt_token_count(messages)
        if type(n) is not int or n < 0:
            raise ValueError("actual served image-aware token count required")
        counts.append({"messages_sha256": digest(messages), "items": len(prefix), "tokens": n})
        return messages, n

    messages, tokens = count(kept)
    if not full_fit_only and tokens > reader_tokens:
        lo, hi = 0, len(kept)
        _, overhead = count([])
        if overhead > reader_tokens:
            raise ValueError("question/image overhead exceeds complete reader operating point")
        while lo < hi:
            mid = (lo + hi + 1) // 2
            _, n = count(kept[:mid])
            if n <= reader_tokens:
                lo = mid
            else:
                hi = mid - 1
        kept = kept[:lo]
        messages, tokens = count(kept)
        native_tokens = native.ns["count_memory_context_tokens"](
            kept, native.ns["load_memory_context_images"](kept)
        )
    output = client.profile.output_budget("answer")
    if (
        full_fit_only
        and tokens + output + client.profile.schema_reserve_tokens > client.profile.context_window
    ):
        raise ContextNotApplicable(
            "measured whole haystack + question/image/output does not fit; no truncation"
        )
    client.admit_context(
        tokens, output_budget=output, input_chars=len(json.dumps(messages, ensure_ascii=False))
    )
    return {
        "items": kept,
        "messages": messages,
        "complete_prompt_tokens": tokens,
        "native_original_tokens": original_tokens,
        "native_kept_tokens": native_tokens,
        "original_items": len(items),
        "kept_items": len(kept),
        "counts": counts,
        "output_reserve": output,
        "schema_reserve": client.profile.schema_reserve_tokens,
        "operating_point": "whole-haystack-fit-only"
        if full_fit_only
        else f"{reader_tokens}-complete-reader-prompt-tokens; whole-item-prefix",
    }


class ImmutableVectorCache:
    """Content-addressed singleton vectors; the caller still owns a LOCAL corpus.

    No text or corpus statistics are shared with retrieval. First dispatch is
    shared setup cost; consumers record hits separately, never imaginary calls.
    Exact batch shape is singleton for both first encode and every reuse.
    """

    def __init__(self, inner, *, root, budget):
        self.inner, self.root, self.budget = inner, Path(root), budget
        from dataclasses import asdict, is_dataclass
        self.encoder_identity = asdict(inner.fingerprint) if is_dataclass(inner.fingerprint) else inner.fingerprint
        self.root.mkdir(parents=True, exist_ok=True)
        self.receipt = {'hits': 0, 'misses': 0, 'encoder': self.encoder_identity,
                        'batch_shape': 'singleton; exact text/role/encoder',
                        'shared_setup_http_calls': 0, 'entries': {}}

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _vector(self, text, role):
        import math
        self.budget.check()
        identity = {'encoder': self.encoder_identity, 'role': role,
                    'text': text, 'batch_shape': [1]}
        key = digest(identity)
        path = self.root / (key + '.json')
        if path.exists():
            row = json.loads(path.read_bytes())
            if row.get('key') != key or row.get('sha256') != digest(row.get('vector')):
                raise ValueError('immutable vector cache integrity mismatch')
            self.receipt['hits'] += 1
        else:
            vector = (self.inner.embed_batch([text])[0] if role == 'document'
                      else self.inner.embed_query(text))
            vector = [float(x) for x in vector]
            if not vector or not all(math.isfinite(x) for x in vector):
                raise ValueError('invalid shared vector')
            row = {'key': key, 'vector': vector, 'sha256': digest(vector)}
            with path.open('xb') as stream:
                stream.write(json.dumps(row, allow_nan=False).encode('utf-8'))
            self.receipt['misses'] += 1
            self.receipt['shared_setup_http_calls'] += 1
        self.receipt['entries'][key] = file_hash(path)
        self.budget.check()
        return list(row['vector'])

    def embed_batch(self, texts):
        return [self._vector(text, 'document') for text in texts]

    def embed_query(self, text):
        return self._vector(text, 'query')


class DeadlineEncoder:
    """Same qualified encoder/roles, lossless bounded batches and shared deadline.

    Embeddings are separate HTTP calls, not generations. Server token usage is
    not exposed by RemoteEmbedder: mark it unknown rather than estimating it.
    """

    def __init__(self, inner, *, budget, receipt, batch_size=32):
        self.inner, self.budget, self.receipt, self.batch_size = (
            inner,
            budget,
            Path(receipt),
            batch_size,
        )
        if type(batch_size) is not int or not 1 <= batch_size <= 32:
            raise ValueError("bounded batch_size 1..32 required")

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _call(self, method, value):
        import time

        from gluomem_bench.bench.embedding_runtime import RemoteEmbedder

        self.budget.check()
        bounded = RemoteEmbedder(
            self.inner.base_url,
            recipe=self.inner.recipe,
            timeout=min(self.inner.timeout, self.budget.remaining()),
        )
        row = {
            "method": method,
            "input_sha256": digest(value),
            "timeout": bounded.timeout,
            "items": len(value) if method == "embed_batch" else 1,
            "usage_tokens": None,
            "usage_status": "not exposed by native RemoteEmbedder",
        }
        started = time.monotonic()
        try:
            result = getattr(bounded, method)(value)
            self.budget.check()
            row["complete"] = True
            return result
        except Exception as exc:
            self.budget.failed = True
            row.update(complete=False, error=safe_error(exc))
            raise
        finally:
            row["elapsed_seconds"] = time.monotonic() - started
            with self.receipt.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")

    def embed_batch(self, texts):
        result = []
        for start in range(0, len(texts), self.batch_size):
            result.extend(self._call("embed_batch", texts[start : start + self.batch_size]))
        return result

    def embed(self, text):
        return self._call("embed", text)

    def embed_query(self, text):
        return self._call("embed_query", text)


class ResourceNotAdmitted(ValueError):
    """Complete history cannot be admitted under the declared bound."""


def campaign_budget(*, max_calls, max_seconds):
    """V2 campaign envelope; unchanged per-group V6 bounds remain independent."""
    from gluomem_bench.bench.rack_trace import WorkloadBudget
    if (type(max_calls) is not int or not 1 <= max_calls <= 10**9 or
            type(max_seconds) not in (int, float) or not 1 <= max_seconds <= 10**9):
        raise ValueError('explicit finite V2 campaign bounds required')
    result = WorkloadBudget(max_calls=1, max_seconds=1, policy='role-budgets-v6')
    result.max_calls, result.max_seconds = max_calls, max_seconds
    return result


def remaining_calls(budget):
    local = budget.max_calls - budget.calls
    outer = getattr(budget, 'outer_budget', None)
    return min(local, remaining_calls(outer)) if outer is not None else local


class GluoDocumentMemory:
    """Actual production document SourceEvents + active DSPy extraction/consolidation.

    No direct claims, NullExtractors or reduced source groups. Failed ingestion
    raises before search/reader construction and retains the production failure.
    """

    def __init__(self, *, cfg, group, docs, proof, out, client, budget, embedder, qa_reserve):
        if len(docs) + qa_reserve > remaining_calls(budget):
            raise ResourceNotAdmitted(
                "whole-history extraction lower bound plus QA reserve exceeds budget; "
                "not measured fit"
            )
        if not proof["complete"] or proof["documents_sha256"] != digest(docs):
            raise ValueError("complete source proof required")
        from types import SimpleNamespace

        from gluomem.llm.modules import DspyExtractors

        from gluomem_bench.bench.beam_ingest import _ingest_conversation, _read_rows, _validate_rows
        from gluomem_bench.bench.rack_trace import RackRouter
        from gluomem_bench.bench.retriever import GluoMemRetriever

        self.group, self.budget, self.cfg = group, budget, cfg
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        stores = self.out / "stores"
        stores.mkdir(exist_ok=False)
        self.events = document_events(docs)
        router = RackRouter(
            cfg, client=client, trace=self.out / "production-raw.jsonl", budget=budget
        )
        extractors = DspyExtractors(cfg, router=router)
        budget.check()
        _ingest_conversation(
            cfg,
            SimpleNamespace(sha=proof["documents_sha256"]),
            stores,
            extractors,
            embedder,
            SimpleNamespace(conv_id=group.id),
            budget.check,
            source_events=self.events,
            coverage_protocol=PROTOCOL,
        )
        committed, retention = _validate_rows(_read_rows(stores / f"{group.id}.mind"), self.events)
        sentinel = json.loads((stores / f"{group.id}.ingest.json").read_bytes())
        if (
            sentinel.get("complete") is not True
            or sentinel.get("coverage_protocol") != PROTOCOL
            or sentinel.get("committed_receipts") != committed
            or sentinel.get("events") != len(docs)
            or sentinel.get("dataset_sha256") != proof["documents_sha256"]
        ):
            raise ValueError("production committed full-source coverage mismatch")
        self.coverage = {
            **proof,
            "source_documents": len(docs),
            "processed_documents": len(committed),
            "committed_receipts": committed,
            **retention,
            "production_path": "SourceEvent(document) -> DspyExtractors -> consolidation",
        }
        write_json(self.out / "coverage.json", self.coverage)
        self.retriever = GluoMemRetriever(
            cfg, db_dir=stores, answer_fn=None, content_to_dias={}, embedder=embedder
        )
        budget.check()

    def query(self, text, *, query_image=None):
        self.budget.check()
        result = self.retriever._search_for(self.group.id).search(
            text, limit=self.cfg.benchmark.answer.context_k
        )
        self.budget.check()
        # No created_at-as-historical-time wrapper. Always invoke native reader,
        # including empty/abstained retrieval, so NoRetrieval is genuinely paired.
        return (
            []
            if result.abstained
            else [{"type": "text", "value": hit.frame.content} for hit in result.hits]
        )

    def close(self):
        for stack in self.retriever._stacks.values():
            stack._db.close()
        self.retriever._stacks.clear()


class RawTextMemory:
    """Custom lexical/dense/hybrid baselines, NOT unchanged upstream RAG.

    Uses the same evidence backend as RawCampaignRetriever, without that chat
    adapter's fabricated dated-speaker wrapper. Every allowed document indexed.
    """

    def __init__(self, method, docs, proof, *, budget, embedder=None, top_k=32):
        import asyncio

        from gluomem_bench.bench.comparison_backends import Evidence, RawEvidenceBackend

        if method not in METHODS[:5]:
            raise ValueError("unsupported raw method")
        if not proof["complete"] or proof["documents_sha256"] != digest(docs):
            raise ValueError("incomplete source coverage")
        self.method, self.docs, self.budget, self.top_k = method, docs, budget, top_k
        self.backend = None
        budget.check()
        if method in {"bm25", "dense", "hybrid"}:
            self.backend = RawEvidenceBackend(
                method,
                encoder=embedder.embed_batch if embedder is not None and method != "bm25" else None,
                query_encoder=embedder.embed_query
                if embedder is not None and method != "bm25"
                else None,
            )
            asyncio.run(
                self.backend.add(
                    [Evidence(d["id"], d["text"], {"offset": d["offset"]}) for d in docs]
                )
            )
            if len(self.backend.chunks) != len(docs) or any(
                self.backend.chunks[d["id"]].text != d["text"] for d in docs
            ):
                raise ValueError("raw index source coverage mismatch")
        budget.check()
        self.coverage = {
            **proof,
            "source_documents": len(docs),
            "method": method,
            "processing": "native NoRetrieval ignores complete source; no index"
            if method == "no_memory"
            else "complete AX-text, no generation extraction",
            "history_pixels_encoded": False,
            "query_image": "reader receives independently",
        }

    def query(self, text, *, query_image=None):
        import asyncio

        self.budget.check()
        if self.method == "no_memory":
            result = []
        elif self.method == "full_context":
            result = [{"type": "text", "value": d["text"]} for d in self.docs]
        else:
            result = [
                {"type": "text", "value": hit.text}
                for hit in asyncio.run(self.backend.search(text, limit=self.top_k))
            ]
        self.budget.check()
        return result

    def close(self):
        import asyncio

        if self.backend is not None:
            asyncio.run(self.backend.close())


def generation_usage(out, *, consumed_calls):
    """Wire-only accounting; SDK wrapper/bridge summary rows are NOT dispatches."""
    out = Path(out)
    responses = []
    for name in ('calls.jsonl', 'production-raw.jsonl'):
        path = out / name
        if path.exists():
            for row in map(json.loads, path.read_bytes().splitlines()):
                if row.get('url', '').endswith('/chat/completions') or 'task' in row:
                    responses.append(row.get('response', {}))
    path = out / 'native-cognee/generation.jsonl'
    if path.exists():
        rows = list(map(json.loads, path.read_bytes().splitlines()))
        # Intent is the actual budget debit even on HTTP/response failure.
        intents = [row for row in rows if row.get('kind') == 'dispatch-intent']
        wires = [row for row in rows if row.get('kind') == '/chat/completions']
        for index in range(len(intents)):
            try:
                responses.append(json.loads(base64.b64decode(wires[index]['response_b64'])))
            except (IndexError, KeyError, ValueError):
                responses.append({})
    if len(responses) != consumed_calls:
        raise ValueError('raw request generation accounting mismatch; no invented usage')
    usages = [r.get('usage', {}) for r in responses]
    return {'generation_calls': consumed_calls, **{
        key: sum(u[key] for u in usages) if all(type(u.get(key)) is int for u in usages) else None
        for key in ('prompt_tokens', 'completion_tokens')}}


_EXECUTION_AUTHORITY = object()


def _latch_group_failure(client, exc):
    from gluomem_bench.bench.rack import FatalRackError, SharedRackError
    # Rack/native transports already classify their own request boundary.
    # A scorer/filesystem OSError is not fresh shared transport evidence.
    classified = exc if isinstance(exc, SharedRackError) or getattr(exc, 'shared_failure', False) else (
        FatalRackError(safe_error(exc)))
    client._latch(classified)


def run_group(
    *,
    data,
    group,
    method,
    native,
    client,
    budget,
    out,
    cfg=None,
    embedder=None,
    document_factories=None,
    authored=False,
    _authority=None,
    top_k=32,
    raw_chunk_chars=1000,
    document_chunk_chars=65536,
    vector_cache_root=None,
):
    """Whole-haystack build then every native question occurrence in the group.

    External Mem0/Cognee factories receive documents only, the SAME client/budget,
    and must provide persistent full-source verification before any reader call.
    A factory absence/failure is a group failure/unrun, never fabricated QA 0.
    """
    import time

    if not authored and _authority is not _EXECUTION_AUTHORITY:
        raise ValueError("native QA requires frozen approved execute entry point")
    if method not in METHODS:
        raise ValueError("unknown method")
    out = Path(out)
    rows = [
        {
            "question_id": qid,
            "tier": group.tier,
            "method": method,
            "group_id": group.id,
            "status": "unrun",
        }
        for qid in group.question_ids
    ]
    result = {
        "status": "running",
        "rows": rows,
        "group_id": group.id,
        "method": method,
        "authored_integration_only": authored,
        "clock": "native source time unknown",
        "completion_authority": "root-status.json",
    }
    memory = None
    initial_calls = budget.calls
    started = time.monotonic()
    stage = "whole-source-ingestion"
    try:
        budget.check()
        reserve = sum(
            1
            + int(
                data.questions[qid]["eval_function"].split("|")[0]
                in {"llm_abstention_checker", "llm_gotchas_checker"}
            )
            for qid in group.question_ids
        )
        if reserve > remaining_calls(budget):
            raise ResourceNotAdmitted(
                "whole-group native reader/judge reserve exceeds remaining calls"
            )
        if method == "no_memory":
            docs = []
            proof = {
                "complete": True,
                "documents_sha256": digest(docs),
                "source_documents": 0,
                "trajectories": len(group.trajectory_ids),
                "states": sum(
                    data.trajectory_metadata[tid]["states"] for tid in group.trajectory_ids
                ),
                "ordered_source_objects": [
                    data.trajectory_metadata[tid] for tid in group.trajectory_ids
                ],
                "source_archive_sha256": data.pins["trajectories.jsonl"],
            }
        else:
            docs, proof = source_documents(
                data,
                group,
                chunk_chars=raw_chunk_chars
                if method in {"bm25", "dense", "hybrid"}
                else document_chunk_chars,
            )
        write_json(out / "source-coverage.json", proof)
        if method in {'mem0', 'cognee'}:
            write_json(out / 'source-documents.json', docs)
        if method in {'dense', 'hybrid'} and vector_cache_root is not None:
            embedder = ImmutableVectorCache(embedder, root=vector_cache_root, budget=budget)
        if method in METHODS[:5]:
            memory = RawTextMemory(
                method, docs, proof, budget=budget, embedder=embedder, top_k=top_k
            )
        elif method == "gluomem":
            memory = GluoDocumentMemory(
                cfg=cfg,
                group=group,
                docs=docs,
                proof=proof,
                out=out,
                client=client,
                budget=budget,
                embedder=embedder,
                qa_reserve=reserve,
            )
        else:
            factory = (document_factories or {}).get(method)
            if factory is None:
                raise ResourceNotAdmitted(
                    f"{method} whole-source native document factory not configured"
                )
            memory = factory(
                cfg=cfg,
                group=group,
                documents=docs,
                source_proof=proof,
                out=out,
                client=client,
                budget=budget,
                embedder=embedder,
                qa_reserve=reserve,
            )
            observed = memory.verify_source_coverage(documents=docs, source_proof=proof)
            expected = [(d["id"], d["sha256"]) for d in docs]
            if (
                observed.get("complete") is not True
                or observed.get("ordered_document_hashes") != expected
                or observed.get("documents_sha256") != proof["documents_sha256"]
                or observed.get("shared_budget_identity") != id(budget)
                or observed.get("shared_client_identity") != id(client)
                or not observed.get("persistent_readback")
            ):
                raise ValueError(
                    "native document factory lacks hard persistent full-source/shared-budget proof"
                )
            memory.coverage = observed
        write_json(out / "coverage.json", memory.coverage)
        result["coverage"] = memory.coverage
        stage = "native-qa"
        for row in rows:
            budget.check()
            qid = row["question_id"]
            q = data.reader_question(qid)
            items = memory.query(q["question"], query_image=q["image"])
            try:
                pack = reader_pack(native, client, q, items, full_fit_only=method == "full_context")
            except ContextNotApplicable as exc:
                row.update(status="not_applicable", reason=str(exc))
                write_json(out / "progress.json", result)
                continue
            answer = client.generate_messages("answer", pack["messages"])
            if (
                answer["prompt_tokens"] != pack["complete_prompt_tokens"]
                or type(answer["completion_tokens"]) is not int
                or answer["completion_tokens"] < 0
            ):
                from gluomem_bench.bench.rack import SharedRackError
                raise SharedRackError("reader usage/image-tokenizer parity missing or mismatched")
            score = native.score(data.questions[qid], answer["text"], client=client)
            row.update(status="complete", pack=pack, answer=answer, score=score)
            write_json(out / "progress.json", result)
        budget.check()
        result["status"] = "complete"
    except Exception as exc:
        if not isinstance(exc, ResourceNotAdmitted):
            _latch_group_failure(client, exc)
        result.update(
            status="not_admitted" if isinstance(exc, ResourceNotAdmitted) else "failed",
            error=safe_error(exc),
            failure_stage=stage,
        )
    finally:
        try:
            if memory is not None:
                memory.close()
                if method in {'mem0', 'cognee'} and result['status'] == 'complete':
                    from gluomem_bench.bench.v2_memory_factories import verify_closed_native
                    result['closed_native_verification'] = verify_closed_native(
                        out, method=method, documents=docs, source_proof=proof, profile=client.profile)
        except Exception as exc:
            result.update(status="failed", error=f'cleanup: {safe_kind(exc)}: {safe_error(exc)}')
        try:
            result['usage'] = generation_usage(out, consumed_calls=budget.calls - initial_calls)
        except Exception as exc:
            from gluomem_bench.bench.rack import SharedRackError
            client._latch(SharedRackError(f'generation accounting unavailable: {safe_error(exc)}'))
            result['usage'] = {'generation_calls': budget.calls - initial_calls,
                               'prompt_tokens': None, 'completion_tokens': None}
            result.update(status='failed', accounting_error=safe_error(exc))
        if isinstance(embedder, ImmutableVectorCache):
            write_json(out / 'embedding-cache.json', embedder.receipt)
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(out / "root-status.json", result)
    return result


def scope_matrix(data):
    return [
        {
            "question_id": qid,
            "tier": tier,
            "method": method,
            "group_id": group.id,
            "status": "unrun",
        }
        for method in METHODS
        for group in data.groups()
        for tier in [group.tier]
        for qid in group.question_ids
    ]


def verify_matrix(matrix, data):
    expected = {(row["method"], row["tier"], row["question_id"]) for row in scope_matrix(data)}
    actual = [(row["method"], row["tier"], row["question_id"]) for row in matrix]
    if len(actual) != len(expected) or len(set(actual)) != len(expected) or set(actual) != expected:
        raise ValueError("matrix exact native occurrence coverage mismatch")
    return _matrix_summary(matrix, {t: len(data.haystacks[t]) for t in data.haystacks})


def _matrix_summary(matrix, questions_per_tier):
    """Recompute status/score coverage from saved rows without source access."""
    methods = {method: {} for method in METHODS}
    for method in METHODS:
        for tier in ("small", "medium"):
            rows = [r for r in matrix if r["method"] == method and r["tier"] == tier]
            for row in rows:
                if row["status"] not in {
                    "unrun",
                    "complete",
                    "not_applicable",
                    "failed",
                    "not_admitted",
                }:
                    raise ValueError("invalid matrix status")
                if row["status"] != "complete" and "score" in row:
                    raise ValueError(
                        "unrun/failed/inapplicable cases must not have fabricated quality scores"
                    )
                if (
                    row["status"] == "complete"
                    and type(row.get("score", {}).get("correct")) is not bool
                ):
                    raise ValueError("complete case requires exact native binary score")
            counts = Counter(r["status"] for r in rows)
            counts["planned"] = len(rows)
            counts["correct"] = sum(r.get("score", {}).get("correct", False) for r in rows)
            methods[method][tier] = dict(counts)
    return {
        "expected_questions_per_tier": questions_per_tier,
        "expected_occurrences": len(matrix),
        "methods": methods,
    }


def example_config():
    """Print with `python -m gluomem_bench.bench.v2_driver example`.

    Operator supplies local roots/pins/qualified profile. No private defaults.
    Missing Mem0/Cognee factories remain explicit matrix statuses, not omission.
    """
    return {
        "protocol": PROTOCOL,
        "dataset_root": "PATH_TO_PINNED_DATA",
        "pins_path": "PATH_TO_DATA_AND_29_IMAGE_SHA256_JSON",
        "native_root": "PATH_TO_PINNED_UPSTREAM_CODE",
        "profile_path": "PATH_TO_EXPLICIT_RACK_PROFILE_JSON",
        "qualification_path": "PATH_TO_NATIVE_VISION_QUALIFICATION_JSON",
        "scope": {"tiers": ["small", "medium"], "questions_per_tier": 451, "query_images": 29},
        "methods": list(METHODS),
        "reader_prompt_tokens": 4096,
        "top_k": 32,
        "document_chunk_chars": 65536,
        "raw_chunk_chars": 1000,
        "campaign_max_calls": 16384,
        "campaign_max_seconds": 21600,
        "embedding_url": None,
        "native_processor_path": "served",
        "factory_specs": {},
    }


def execute(spec_path, approval_path, out):
    approval = json.loads(Path(approval_path).read_bytes())
    if (
        approval.get("approved") is not True
        or approval.get("spec_sha256") != file_hash(spec_path)
        or approval.get("driver_sha256") != file_hash(__file__)
    ):
        raise ValueError("explicit frozen spec/driver approval required before native QA")
    return _execute_approved(Path(spec_path), Path(out))


def _read_config(config, *, authored=False):
    from gluomem_bench.bench.rack import RackProfile

    if set(config) != set(example_config()):
        raise ValueError("explicit complete config schema required; unknown/omitted keys forbidden")
    if config["protocol"] != PROTOCOL or config["scope"] != example_config()["scope"]:
        raise ValueError("full published Small/Medium 451-question scope required")
    if (
        not config["methods"]
        or len(set(config["methods"])) != len(config["methods"])
        or not set(config["methods"]) <= set(METHODS)
        or config["reader_prompt_tokens"] != 4096
    ):
        raise ValueError("declared methods/4096 complete reader operating point required")
    for key in ("top_k", "document_chunk_chars", "raw_chunk_chars"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("positive chunk and retrieval bounds required")
    pins = json.loads(Path(config["pins_path"]).read_bytes())
    if not authored:
        published = {
            "questions.jsonl": "0a3ae5ebea938c24d7800e1e0b0828e08ae1646f939a53853b2b8cdc08e292b7",
            "trajectories.jsonl":
                "363cec9a8e87aa8d9101ce4e600aadbf7031d674056ebe4f969e8424abc5f3c6",
            "haystacks/lme_v2_small.json":
                "9b5301defb23a088a5f06e45ff8d5f35e569d78305a66d492046a9fff9b46593",
            "haystacks/lme_v2_medium.json":
                "4756d5126347f0d18f045bb6c47b08cb3b23e9db24386cc48a9b2879e7969b59",
        }
        if any(pins.get(k) != value for k, value in published.items()):
            raise ValueError("published data revision pins differ; no native QA on substitutes")
    data = SourceDataset(config["dataset_root"], pins=pins)
    native = NativeInterface(config["native_root"])
    native.receipt['metric_dispatch_counts'] = native.validate_specs(data.questions.values())
    profile = RackProfile.model_validate(json.loads(Path(config["profile_path"]).read_bytes()))
    if profile.profile_version != "role-budgets-v6" or profile.output_budget("answer") != 4096:
        raise ValueError("explicit V6 role profile with 4096 reader output required")
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    campaign_budget(
        max_calls=config["campaign_max_calls"],
        max_seconds=config["campaign_max_seconds"],
    )
    return data, native, profile


def native_factory_sources(entries):
    """Locate installed native SDK bytes without executing their import hooks."""
    import importlib.util
    import subprocess
    sources = {}
    for method, entry in entries.items():
        if entry.get('module', 'gluomem_bench.bench.v2_memory_factories') != 'gluomem_bench.bench.v2_memory_factories':
            continue  # legacy authored absence test; concrete loader refuses at dispatch
        if method == 'mem0':
            found = importlib.util.find_spec('mem0')
            if found is None or found.origin is None:
                raise ValueError('actual installed native Mem0 required')
            root = Path(found.origin).parent
        elif method == 'cognee':
            completed = subprocess.run([entry['options']['python_executable'], '-I', '-c',
                "import importlib.util; print(importlib.util.find_spec('cognee').origin)"],
                capture_output=True, text=True, check=True, timeout=30)
            root = Path(completed.stdout.strip()).parent
        else:
            raise ValueError('unknown native factory')
        for path in sorted(root.rglob('*.py')):
            sources[method + '/' + path.relative_to(root).as_posix()] = path.resolve()
    return sources


def prepare(config, out, *, authored=False):
    """Read-only admission + recoverable source archive; zero QA/model calls."""
    import zipfile

    import gluomem

    data, native, profile = _read_config(config, authored=authored)
    out = Path(out).resolve()
    bench_root = Path(__file__).resolve().parents[2]
    core_root = Path(gluomem.__file__).resolve().parent.parent
    if any(out.is_relative_to(r) for r in (bench_root, core_root)):
        raise ValueError("artifacts must be outside source repositories")
    out.mkdir(parents=True, exist_ok=False)
    producer_files = {}
    archive = out / "sourceArchive.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipout:
        for label, root, source in [
            ("bench", bench_root, bench_root / "gluomem_bench"),
            ("core", core_root, core_root / "gluomem"),
        ]:
            files = sorted(p for p in source.rglob('*') if p.is_file()
                           and '__pycache__' not in p.parts and p.suffix != '.pyc') + [
                root / name for name in ("pyproject.toml", "uv.lock") if (root / name).exists()
            ]
            for path in files:
                producer_files[str(path)] = file_hash(path)
                zipout.write(path, f"{label}/{path.relative_to(root).as_posix()}")
        for relative, path in native_factory_sources(config['factory_specs']).items():
            producer_files[str(path)] = file_hash(path)
            zipout.write(path, 'native-sdk/' + relative)
        for key in ('profile_path', 'qualification_path', 'pins_path'):
            zipout.write(config[key], 'operator/' + key + '.json')
        for name in native.PINS:
            path = Path(config["native_root"]) / name
            producer_files[str(path.resolve())] = file_hash(path)
            zipout.write(path, "native/" + name)
    for key in ("profile_path", "qualification_path", "pins_path"):
        path = Path(config[key]).resolve()
        producer_files[str(path)] = file_hash(path)
    for entry in config["factory_specs"].values():
        path = Path(entry["source_path"]).resolve()
        if file_hash(path) != entry["sha256"]:
            raise ValueError("document factory source pin mismatch")
        producer_files[str(path)] = entry["sha256"]
    spec = {
        "protocol": PROTOCOL,
        "authored_integration_only": authored,
        "config": config,
        "counts": data.counts(),
        "dataset_receipt": data.receipt,
        "native_receipt": native.receipt,
        "matrix": scope_matrix(data),
        "reader_questions": {qid: data.reader_question(qid) for qid in data.questions},
        "query_image_hashes": {qid: data.pins[q['image']] if q.get('image') else None
                               for qid, q in data.questions.items()},
        "native_system_prompts": native.ns['DOMAIN_SYSTEM_PROMPTS'],
        "producer_files": producer_files,
        "source_archive": {"path": str(archive), "sha256": file_hash(archive)},
        "driver_sha256": file_hash(__file__),
        "profile": profile.model_dump(mode="json"),
        "imported_core": str(Path(gluomem.__file__).resolve()),
    }
    write_json(out / "spec.json", spec)
    return spec


def _verify_frozen(spec):
    if spec["driver_sha256"] != file_hash(__file__):
        raise ValueError("frozen driver hash differs")
    for name, pin in spec["producer_files"].items():
        if file_hash(name) != pin:
            raise ValueError("frozen producer/config/source hash differs")
    archive = spec["source_archive"]
    if file_hash(archive["path"]) != archive["sha256"]:
        raise ValueError("sourceArchive hash differs")
    import gluomem

    if str(Path(gluomem.__file__).resolve()) != spec["imported_core"]:
        raise ValueError("exact imported production module location differs")


def _load_document_factories(spec):
    """Explicit deferred factory protocol; never ambient SDK/provider discovery.

    Callable receives cfg/group/documents/source_proof/out/client/budget/embedder/
    qa_reserve and returns query/close + verify_source_coverage. Pin the concrete
    native factory module source before prepare; no stub or raw fallback exists.
    """
    import importlib

    factories = {}
    for method, entry in spec["config"]["factory_specs"].items():
        if method not in {"mem0", "cognee"}:
            raise ValueError("only native Mem0/Cognee external document factories allowed")
        if file_hash(entry["source_path"]) != entry["sha256"]:
            raise ValueError("native factory pin changed")
        module = importlib.import_module(entry["module"])
        if Path(module.__file__).resolve() != Path(entry["source_path"]).resolve():
            raise ValueError("native factory imported location differs")
        from functools import partial
        if (entry['module'] != 'gluomem_bench.bench.v2_memory_factories'
                or entry['callable'] != method + '_factory'):
            raise ValueError('concrete native factory identity required')
        options = entry.get('options', {})
        if (method == 'mem0' and options) or (method == 'cognee' and
                (set(options) != {'python_executable', 'chunk_size'} or not options['python_executable'])):
            raise ValueError('explicit concrete native factory options required')
        factories[method] = partial(getattr(module, entry['callable']), **options)
    return factories


def _campaign_usage(outer, jobs):
    accounted = sum(job['usage']['generation_calls'] for job in jobs) == outer.calls
    return {'generation_calls': outer.calls, **{
        key: sum(job['usage'][key] for job in jobs)
        if accounted and all(type(job['usage'][key]) is int for job in jobs) else None
        for key in ('prompt_tokens', 'completion_tokens')}}


def _execute_approved(spec_path, out, *, authored=False):
    from gluomem_bench.bench.rack import RackClient, SharedRackError, controlled_config
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    class GroupBudget(WorkloadBudget):
        def fail_shared(self):
            # Shared rack guards call this explicitly. The V2 outer envelope
            # is itself a budget, not the scheduler's JobContext wrapper.
            self.failed = True
            self.outer_budget.failed = True

    spec = json.loads(Path(spec_path).read_bytes())
    if bool(spec.get("authored_integration_only")) != authored:
        raise ValueError("authored fixtures cannot masquerade as published native execution")
    _verify_frozen(spec)
    config = spec["config"]
    data, native, profile = _read_config(config, authored=authored)
    qualification = json.loads(Path(config["qualification_path"]).read_bytes())
    if (
        qualification.get("qualified") is not True
        or qualification.get("model") != profile.model
        or qualification.get("profile_sha256") != digest(profile.model_dump(mode="json"))
    ):
        raise ValueError("exact profile native vision transport qualification required")
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("fresh output root required; no partial resume")
    if out.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("external artifacts required")
    out.mkdir(parents=True)
    (out / "spec.json").write_bytes(Path(spec_path).read_bytes())
    matrix = scope_matrix(data)
    indexed = {(r["method"], r["tier"], r["question_id"]): r for r in matrix}
    outer = campaign_budget(
        max_calls=config["campaign_max_calls"],
        max_seconds=config["campaign_max_seconds"],
    )
    root = {
        "protocol": PROTOCOL,
        "status": "running",
        "matrix": matrix,
        "jobs": [],
        "usage": _campaign_usage(outer, []),
        "authored_integration_only": authored,
        "source_archive": spec["source_archive"],
        "spec_sha256": file_hash(out / "spec.json"),
        "model": profile.model,
        "native_clock": "source timestamp absent; not invented",
        "counts": data.counts(),
    }
    write_json(out / "root-status.json", root)
    embedder = None
    if config["embedding_url"]:
        from gluomem_bench.bench.embedding_runtime import RemoteEmbedder

        embedder = RemoteEmbedder(
            config["embedding_url"], timeout=min(profile.timeout_seconds, outer.remaining())
        )
    cfg = controlled_config(
        profile,
        data=data.paths["questions.jsonl"],
        sha256=data.pins["questions.jsonl"],
        embedder=embedder,
    )
    cfg = cfg.with_overrides({"benchmark.answer.context_k": config["top_k"]})
    try:
        for method in config["methods"]:
            for group in data.groups():
                outer.check()
                _verify_frozen(spec)
                job = out / "jobs" / method / group.id
                job.mkdir(parents=True)
                budget = GroupBudget(
                    max_calls=profile.max_generation_calls,
                    max_seconds=profile.max_run_seconds,
                    policy=profile.profile_version,
                    outer_budget=outer,
                )
                client = RackClient(profile, receipt=job / "calls.jsonl", budget=budget)
                try:
                    if config["native_processor_path"] == "served":
                        native.use_served_context_counter(client)
                    elif config["native_processor_path"] is not None:
                        raise ValueError(
                            "native_processor_path must be served "
                            "(declared actual-processor substitution) or null (native cached 9B)"
                        )
                    try:
                        model_info = client._request("GET", "/models")
                        model = next(m for m in model_info["data"] if m["id"] == profile.model)
                        if model.get("max_model_len") != profile.context_window:
                            raise SharedRackError(
                                "served model/context identity differs from frozen profile"
                            )
                    except Exception as exc:
                        budget.fail_shared()
                        client._latch(exc)
                        raise
                    # Import optional native SDK factories only for the selected
                    # condition, so a missing Mem0 environment never blocks raw.
                    local_spec = {
                        **spec,
                        "config": {
                            **config,
                            "factory_specs": {
                                k: v for k, v in config["factory_specs"].items() if k == method
                            },
                        },
                    }

                    def deferred_factory(_spec=local_spec, _method=method, **kwargs):
                        selected = _load_document_factories(_spec).get(_method)
                        if selected is None:
                            raise ResourceNotAdmitted("native document factory not configured")
                        return selected(**kwargs)

                    factories = (
                        {method: deferred_factory} if method in config["factory_specs"] else {}
                    )
                    result = run_group(
                        data=data,
                        group=group,
                        method=method,
                        native=native,
                        client=client,
                        budget=budget,
                        out=job,
                        cfg=cfg,
                        embedder=DeadlineEncoder(
                            embedder, budget=budget, receipt=job / "embeddings.jsonl"
                        )
                        if embedder is not None
                        else None,
                        document_factories=factories,
                        authored=authored,
                        _authority=_EXECUTION_AUTHORITY,
                        top_k=config["top_k"],
                        raw_chunk_chars=config["raw_chunk_chars"],
                        document_chunk_chars=config["document_chunk_chars"],
                        vector_cache_root=out / 'shared-chunk-vectors',
                    )
                finally:
                    client.close()
                for row in result["rows"]:
                    indexed[(method, group.tier, row["question_id"])].update(row)
                root["jobs"].append(
                    {
                        "method": method,
                        "group_id": group.id,
                        "status": result["status"],
                        "path": str(job),
                        "root_sha256": file_hash(job / "root-status.json"),
                        "usage": result["usage"],
                    }
                )
                root['usage'] = _campaign_usage(outer, root['jobs'])
                root["summary"] = verify_matrix(matrix, data)
                write_json(out / "root-status.json", root)
                # Local generation/schema/judge failures remain latched on
                # their own group. Only an explicit shared latch cancels scope.
                if outer.failed:
                    raise RuntimeError(
                        "shared campaign failure; remaining scope unrun"
                    )
        _verify_frozen(spec)
        final_status = (
            "complete" if all(j["status"] == "complete" for j in root["jobs"]) else "incomplete"
        )
    except Exception as exc:
        root.update(status="failed", error=safe_error(exc))
        final_status = "failed"
    finally:
        # Keep the authoritative path non-complete throughout finalization.
        # The verified candidate is promoted by one same-directory replacement.
        root['status'] = 'failed' if root['status'] == 'failed' else 'incomplete'
        root["summary"] = verify_matrix(matrix, data)
        root['usage'] = _campaign_usage(outer, root['jobs'])
        root["files"] = {
            p.relative_to(out).as_posix(): file_hash(p)
            for p in out.rglob("*")
            if p.is_file() and p.name != "root-status.json"
        }
        write_json(out / "root-status.json", root)
    candidate = out / '.root-status.pending.json'
    try:
        root['status'] = final_status
        write_json(candidate, root)
        candidate_bytes = candidate.read_bytes()
        if json.loads(candidate_bytes) != root:
            raise ValueError('root candidate write/readback differs')
        verify_root(out, _root_path=candidate)
        _verify_frozen(spec)
        if candidate.read_bytes() != candidate_bytes:
            raise ValueError('root candidate changed during verification/readback')
        candidate.replace(out / 'root-status.json')
    except Exception as exc:
        root.update(status='failed', error=f'finalization: {safe_kind(exc)}: {safe_error(exc)}')
        write_json(out / 'root-status.json', root)
        raise
    return root


def verify_root(out, *, _root_path=None):
    """Offline artifact/root/model/clock/modality/exact-occurrence verification."""
    out = Path(out)
    root = json.loads((_root_path or (out / "root-status.json")).read_bytes())
    spec = json.loads((out / "spec.json").read_bytes())
    if root["spec_sha256"] != file_hash(out / "spec.json"):
        raise ValueError("frozen spec hash differs")
    for name, pin in root["files"].items():
        path = (out / name).resolve()
        if not path.is_relative_to(out.resolve()) or file_hash(path) != pin:
            raise ValueError("root artifact hash differs")
    expected = [(r["method"], r["tier"], r["question_id"], r['group_id']) for r in spec["matrix"]]
    actual = [(r["method"], r["tier"], r["question_id"], r['group_id']) for r in root["matrix"]]
    if (
        len(actual) != len(expected)
        or len(set(actual)) != len(expected)
        or set(actual) != set(expected)
    ):
        raise ValueError("exact 451-per-tier occurrence coverage differs")
    if root['counts'] != spec['counts']:
        raise ValueError('root frozen occurrence/image counts differ')
    if root['summary'] != _matrix_summary(
            root['matrix'], {t: spec['counts'][t] for t in ('small', 'medium')}):
        raise ValueError('root matrix status/score summary differs')
    planned_jobs = {(r['method'], r['group_id']) for r in spec['matrix']
                    if r['method'] in spec['config']['methods']}
    job_keys = [(j['method'], j['group_id']) for j in root['jobs']]
    if len(set(job_keys)) != len(job_keys) or not set(job_keys) <= planned_jobs:
        raise ValueError('root duplicate/unplanned job authority')
    if root['status'] == 'complete' and (
            set(job_keys) != planned_jobs or any(j['status'] != 'complete' for j in root['jobs'])):
        raise ValueError('root completion requires every selected verified complete job')
    for row in root['matrix']:
        if (row['method'], row['group_id']) not in set(job_keys) and row['status'] != 'unrun':
            raise ValueError('matrix status without a retained job authority')
    for job in root["jobs"]:
        path = Path(job["path"]).resolve()
        if (
            not path.is_relative_to(out.resolve())
            or file_hash(path / "root-status.json") != job["root_sha256"]
        ):
            raise ValueError("leaf root status hash differs")
        leaf = json.loads((path / "root-status.json").read_bytes())
        parent_rows = [
            r
            for r in root["matrix"]
            if r["method"] == job["method"] and r["group_id"] == job["group_id"]
        ]
        if job['status'] == 'complete' and any(
                row['status'] not in {'complete', 'not_applicable'} for row in parent_rows):
            raise ValueError('complete job contains incomplete question occurrences')
        if (
            parent_rows != leaf["rows"]
            or leaf["status"] != job["status"]
            or leaf["usage"] != job["usage"]
        ):
            raise ValueError("parent/leaf exact row/status/usage binding differs")
        if leaf['usage'] != generation_usage(path, consumed_calls=leaf['usage']['generation_calls']):
            raise ValueError('leaf wire usage accounting differs')
        if leaf['status'] == 'complete' and job['method'] in {'mem0', 'cognee'}:
            from gluomem_bench.bench.v2_memory_factories import verify_closed_native
            from gluomem_bench.bench.rack import RackProfile
            verify_closed_native(path, method=job['method'],
                documents=json.loads((path/'source-documents.json').read_bytes()),
                source_proof=json.loads((path/'source-coverage.json').read_bytes()),
                profile=RackProfile.model_validate(spec['profile']))
            context_path = path / ('native-mem0/query-contexts.json' if job['method'] == 'mem0'
                                   else 'native-cognee/graph-contexts.json')
            contexts = json.loads(context_path.read_bytes())
            if len(contexts) != len(leaf['rows']):
                raise ValueError('native retrieval exact query coverage differs')
            for row, context in zip(leaf['rows'], contexts, strict=True):
                key = 'memory' if job['method'] == 'mem0' else 'text'
                expected_items = [{'type': 'text', 'value': hit[key]} for hit in context['hits']]
                if (context['query'] != spec['reader_questions'][row['question_id']]['question'] or
                        row['pack']['items'] != expected_items[:len(row['pack']['items'])]):
                    raise ValueError('native evidence-only reader binding differs; no raw fallback')
        calls_path = path / 'calls.jsonl'
        wires = list(map(json.loads, calls_path.read_bytes().splitlines())) if calls_path.exists() else []
        unused = {i for i, wire in enumerate(wires)
                  if wire.get('url', '').endswith('/chat/completions')}
        for row in leaf['rows']:
            if row['status'] != 'complete':
                continue
            bound = [(row['pack']['messages'], row['answer']['response'])]
            bound.extend((j['messages'], j['result']['response'])
                         for j in row['score']['judgments'])
            for messages, response in bound:
                matches = [i for i in unused
                           if wires[i].get('request', {}).get('messages') == messages
                           and wires[i].get('response') == response]
                if not matches:
                    raise ValueError('reader/judge wire response/input binding differs')
                unused.remove(matches[0])
    if (
        root["model"] != spec["profile"]["model"]
        or root["native_clock"] != "source timestamp absent; not invented"
    ):
        raise ValueError("model/native clock differs")
    for row in root["matrix"]:
        if row["status"] != "complete":
            if "score" in row:
                raise ValueError("unrun/failed row has fabricated score")
            continue
        messages = row["pack"]["messages"]
        if row["answer"]["response"]["model"] != root["model"]:
            raise ValueError("actual model response identity differs")
        if row["answer"]["prompt_tokens"] != row["pack"]["complete_prompt_tokens"]:
            raise ValueError("actual usage/image-tokenizer parity differs")
        if (
            row["pack"]["operating_point"] != "whole-haystack-fit-only"
            and row["pack"]["complete_prompt_tokens"] > 4096
        ):
            raise ValueError("complete reader operating point exceeded")
        if not isinstance(messages[1]["content"], list):
            raise ValueError("native text+image input structure differs")
        expected_question = spec['reader_questions'][row['question_id']]
        if messages[0] != {'role': 'system', 'content':
                           spec['native_system_prompts'][expected_question['domain']]}:
            raise ValueError('native system text input differs')
        image_parts = [p for p in messages[1]['content'] if p['type'] == 'image_url']
        expected_image_hash = spec['query_image_hashes'][row['question_id']]
        if len(image_parts) != int(expected_image_hash is not None):
            raise ValueError('required native query image input absent or extra history images')
        if image_parts:
            image_bytes = base64.b64decode(
                image_parts[0]['image_url']['url'].split(',', 1)[1], validate=True)
            if hashlib.sha256(image_bytes).hexdigest() != expected_image_hash:
                raise ValueError('query image input SHA256 differs')
        text_parts = [p['text'] for p in messages[1]['content'] if p['type'] == 'text']
        intro = '### Memory context:\n' + ('' if row['pack']['items'] else '(empty)')
        expected_text = [intro] + [i['value'] for i in row['pack']['items']] + [
            '\n\n### Question to answer:\n' + expected_question['question']]
        if text_parts != expected_text or expected_question['question_time'] is not None:
            raise ValueError('native question/clock/complete context text differs')
    if (
        sum(j["usage"]["generation_calls"] for j in root["jobs"])
        != root["usage"]["generation_calls"]
    ):
        raise ValueError("exact generation usage reconciliation differs")
    for key in ('prompt_tokens', 'completion_tokens'):
        values = [job['usage'][key] for job in root['jobs']]
        total = sum(values) if all(type(value) is int for value in values) else None
        if root['usage'][key] != total:
            raise ValueError('root token usage reconciliation differs')
    return {
        "verified": True,
        "status": root["status"],
        "expected_questions_per_tier": {"small": 451, "medium": 451},
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("example")
    freeze = sub.add_parser("prepare")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--out", required=True)
    qualify = sub.add_parser("qualify-vision")
    qualify.add_argument("--profile", required=True)
    qualify.add_argument("--native-root", required=True)
    qualify.add_argument("--out", required=True)
    run = sub.add_parser("run")
    run.add_argument("--spec", required=True)
    run.add_argument("--approval", required=True)
    run.add_argument("--out", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "example":
        print(json.dumps(example_config(), indent=2))
    elif args.command == "prepare":
        result = prepare(json.loads(Path(args.config).read_bytes()), args.out)
        print(json.dumps({"counts": result["counts"], "source_archive": result["source_archive"]}))
    elif args.command == "qualify-vision":
        from gluomem_bench.bench.rack import RackProfile

        profile = RackProfile.model_validate(json.loads(Path(args.profile).read_bytes()))
        result = qualify_vision(profile, args.native_root, args.out)
        print(
            json.dumps(
                {
                    k: result.get(k)
                    for k in ("qualified", "generation_calls", "elapsed_seconds", "blocker")
                }
            )
        )
    elif args.command == "run":
        result = execute(args.spec, args.approval, args.out)
        print(json.dumps({"status": result["status"], "usage": result["usage"]}))
    elif args.command == "verify":
        print(json.dumps(verify_root(args.out)))


class NativeInterface:
    """Execute byte-pinned native functions without importing upstream modules.

    Only the two network functions are substituted, inside an isolated namespace;
    native prompts/spec/parser/objective metrics/UNKNOWN policy stay unmodified.
    """

    PINS = {
        "evaluation/harness.py": "618396982fbdf59a6b33d5aa9cc3b4ee673d8789b822d32b0b75d60867bc3719",
        "evaluation/qa_eval_metrics.py":
            "e95b0fa4ecacdea5adfcb4a3a82539523bc9d102866a1015aca6c29bdcc1805a",
    }

    def __init__(self, root, *, processor=None):
        root = Path(root).resolve()
        paths = {name: verified(root, name, self.PINS) for name in self.PINS}
        ns = {
            **vars(typing),
            "json": json,
            "os": os,
            "re": re,
            "Path": Path,
            "mimetypes": mimetypes,
            "base64": base64,
            "MemoryContextItem": dict,
            "LLM_EVAL_FUNCTIONS": {"llm_abstention_checker", "llm_gotchas_checker"},
        }
        metrics = ast.parse(paths["evaluation/qa_eval_metrics.py"].read_text(encoding="utf-8"))
        nodes = [
            node
            for node in metrics.body
            if isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign))
            and not (
                isinstance(node, ast.FunctionDef)
                and node.name in {"_create_openai_client", "_call_chat_completion"}
            )
        ]
        exec(
            compile(
                ast.Module(body=nodes, type_ignores=[]),
                str(paths["evaluation/qa_eval_metrics.py"]),
                "exec",
            ),
            ns,
        )
        selected = {
            "require",
            "to_data_url",
            "build_messages",
            "score_prediction",
            "load_memory_context_images",
            "count_memory_context_tokens",
            "truncate_memory_context",
        }
        harness = ast.parse(paths["evaluation/harness.py"].read_text(encoding="utf-8"))
        nodes = [
            node
            for node in harness.body
            if (isinstance(node, ast.FunctionDef) and node.name in selected)
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "DOMAIN_SYSTEM_PROMPTS"
                    for t in node.targets
                )
            )
        ]
        exec(
            compile(
                ast.Module(body=nodes, type_ignores=[]), str(paths["evaluation/harness.py"]), "exec"
            ),
            ns,
        )

        def get_processor():
            if processor is not None:
                return processor
            if not hasattr(self, "_processor"):
                from transformers import AutoProcessor

                # No implicit download. An operator must prepare pinned assets.
                self._processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen3.5-9B", local_files_only=True
                )
            return self._processor

        ns["get_memory_context_processor"] = get_processor
        self.ns = ns
        self.receipt = {
            "sources": {name: self.PINS[name] for name in paths},
            "code_revision": CODE_REVISION,
            "eager_upstream_client_imports": False,
            "transport_substitution": "guarded RackClient, no SDK retries",
            "official_judge_model": "gpt-5.2",
            "native_clock": "unknown; not invented",
        }

    def validate_specs(self, questions):
        counts = {'objective': 0, 'llm': 0}
        objective = {'norm_phrase_set_match', 'norm_phrase_set_match_ordered',
                     'mc_choice_match', 'mc_choice_set_match'}
        semantic = {'llm_abstention_checker', 'llm_gotchas_checker'}
        for q in questions:
            self.ns['parse_eval_function_spec'](q['eval_function'])
            name = self.ns['eval_name'](q['eval_function'])
            if name not in objective | semantic:
                raise ValueError('unsupported native metric')
            counts['llm' if name in semantic else 'objective'] += 1
        return counts

    def use_served_context_counter(self, client):
        """Keep native whole-item prefix search, substitute actual served processor.

        This is an explicit operating-point model substitution, not reproduction
        of upstream's fixed Qwen3.5-9B processor. No char/token estimates. Image
        inputs are real data URLs; /tokenize must support them or fail closed.
        """

        def count(items, loaded_images):
            if len(items) != len(loaded_images):
                raise ValueError("native item/image alignment differs")
            if not items:
                return 0
            parts = [
                {"type": "text", "text": item["value"]}
                if item["type"] == "text"
                else {
                    "type": "image_url",
                    "image_url": {"url": self.ns["to_data_url"](item["value"])},
                }
                for item in items
            ]
            result = client._request(
                "POST",
                "/tokenize",
                {
                    "model": client.profile.model,
                    "messages": [{"role": "user", "content": parts}],
                    "add_generation_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            n = result.get("count")
            if type(n) is not int or n < 0:
                raise ValueError("actual served memory context token count missing")
            return n

        self.ns["count_memory_context_tokens"] = count
        self.receipt["memory_processor"] = (
            f"actual served {client.profile.model}; substituted for native Qwen/Qwen3.5-9B"
        )

    def messages(self, domain, question, image, context):
        return self.ns["build_messages"](
            self.ns["DOMAIN_SYSTEM_PROMPTS"][domain], question, image, context
        )[0]

    def truncate(self, items, *, max_tokens=4096, question_id="opaque"):
        return self.ns["truncate_memory_context"](
            items, max_tokens=max_tokens, question_id=question_id
        )

    def score(self, question_item, raw, *, client=None):
        ns = dict(self.ns)
        # Function globals belong to the isolated ns: serialize scoring; no
        # shared mutable upstream module, no eager OpenAI imports or fallback.
        ns = self.ns
        judgments = []

        def create_client(**kwargs):
            if client is None:
                raise ValueError("native LLM scorer requires common guarded client")
            return client

        def call_chat_completion(**kwargs):
            result = client.generate_messages("rubric", kwargs["messages"], top_p=1.0)
            if (type(result['prompt_tokens']) is not int or result['prompt_tokens'] < 0
                    or type(result['completion_tokens']) is not int or result['completion_tokens'] < 0
                    or result['prompt_tokens'] != result.get('admitted_prompt_tokens', result['prompt_tokens'])):
                from gluomem_bench.bench.rack import SharedRackError
                raise SharedRackError('native judge tokenizer/usage accounting mismatch')
            judgments.append({"messages": kwargs["messages"], "result": result})
            return result["text"]

        ns["_create_openai_client"] = create_client
        ns["_call_chat_completion"] = call_chat_completion
        parsed = ns["extract_boxed_answer"](raw)
        name = ns["eval_name"](question_item["eval_function"])
        row = {
            "eval_name": name,
            "question_item": question_item,
            "response_parsed_boxed": parsed,
            "response_raw": raw,
            "eval_function": question_item["eval_function"],
            "answer_gold": question_item["answer"],
            "is_unknown": ns["is_unknown"](parsed),
        }
        config = (
            {}
            if client is None
            else {
                "evaluator_model": client.profile.model,
                "evaluator_base_url": client.profile.base_url,
                "evaluator_api_key": "EMPTY",
                "evaluator_reasoning_effort": None,
                "evaluator_max_completion_tokens": client.profile.output_budget("rubric"),
                "evaluator_timeout_seconds": client.profile.timeout_seconds,
                "evaluator_temperature": 0,
                "evaluator_top_p": 1.0,
            }
        )
        try:
            correct, _, unknown = ns["score_prediction"](row, config)
        except Exception as exc:
            if client is not None:
                _latch_group_failure(client, exc)
            raise
        finally:
            ns["_create_openai_client"] = lambda **kw: (_ for _ in ()).throw(
                RuntimeError("unguarded judge forbidden")
            )
            ns["_call_chat_completion"] = ns["_create_openai_client"]
        return {
            "correct": correct,
            "parsed": parsed,
            "is_unknown": unknown,
            "eval_name": name,
            "judgments": judgments,
            "judge_model_substitution": f"{client.profile.model} (official gpt-5.2)"
            if judgments
            else None,
        }


class SourceDataset:
    """Verify full source bytes; index offsets, not 902 copies of 1.2GB histories.

    expected_questions/images are fixed for published scope. Authored fixtures can
    opt into different expected counts, never via the native execution CLI.
    """

    def __init__(self, root, *, pins, expected_questions=451, expected_images=29):
        self.root = Path(root).resolve()
        self.pins = dict(pins)
        names = ["questions.jsonl", "trajectories.jsonl"] + [
            f"haystacks/lme_v2_{tier}.json" for tier in ("small", "medium")
        ]
        self.paths = {name: verified(self.root, name, pins) for name in names}
        self.questions = {}
        with self.paths["questions.jsonl"].open(encoding="utf-8") as stream:
            for line in stream:
                q = json.loads(line)
                qid = q["id"]
                if not isinstance(qid, str) or not qid or qid in self.questions:
                    raise ValueError("duplicate/invalid question ID")
                if q["domain"] not in {"web", "enterprise"} or not isinstance(q["question"], str):
                    raise ValueError("invalid question")
                self.questions[qid] = q
        if len(self.questions) != expected_questions:
            raise ValueError("entire published question scope required")
        image_refs = [q["image"] for q in self.questions.values() if q.get("image")]
        if len(image_refs) != expected_images or len(set(image_refs)) != expected_images:
            raise ValueError("entire query image scope required")
        self.images = {name: verified(self.root, name, pins) for name in image_refs}
        self.haystacks = {}
        for tier in ("small", "medium"):
            mapping = json.loads(self.paths[f"haystacks/lme_v2_{tier}.json"].read_bytes())
            if set(mapping) != set(self.questions):
                raise ValueError("exact question/haystack occurrence coverage required")
            for ids in mapping.values():
                if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
                    raise ValueError("invalid haystack")
            self.haystacks[tier] = mapping
        self.offsets = {}
        self.trajectory_metadata = {}
        wanted = {
            tid for mapping in self.haystacks.values() for ids in mapping.values() for tid in ids
        }
        with self.paths["trajectories.jsonl"].open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                trajectory = json.loads(line)
                tid = trajectory["id"]
                if not isinstance(tid, str) or not tid or tid in self.offsets:
                    raise ValueError("duplicate/invalid trajectory ID")
                states = trajectory["states"]
                if not states or any(
                    not isinstance(s, dict) or not isinstance(s.get("accessibility_tree"), str)
                    for s in states
                ):
                    raise ValueError("invalid complete source states")
                self.offsets[tid] = (offset, len(line), trajectory["domain"])
                self.trajectory_metadata[tid] = {
                    "sha256": hashlib.sha256(line).hexdigest(),
                    "offset": offset,
                    "bytes": len(line),
                    "states": len(states),
                }
        if not wanted <= self.offsets.keys():
            raise ValueError("missing source trajectory")
        for mapping in self.haystacks.values():
            for qid, ids in mapping.items():
                if any(self.offsets[tid][2] != self.questions[qid]["domain"] for tid in ids):
                    raise ValueError("cross-domain haystack")
        self.receipt = {
            "protocol": PROTOCOL,
            "code_revision": CODE_REVISION,
            "data_revision": DATA_REVISION,
            "root": str(self.root),
            "files": {
                name: {"sha256": pins[name], "bytes": p.stat().st_size}
                for name, p in {**self.paths, **self.images}.items()
            },
            "history_images_required": False,
            "query_images_required": True,
            "clock": "source world timestamp absent; no fabricated event_time",
            "counts": self.counts(),
        }

    def counts(self):
        return {
            "questions": len(self.questions),
            "query_images": len(self.images),
            **{tier: len(mapping) for tier, mapping in self.haystacks.items()},
            "occurrences": sum(len(m) for m in self.haystacks.values()),
        }

    def groups(self):
        result = []
        for tier, mapping in self.haystacks.items():
            grouped = {}
            for qid, ids in mapping.items():
                key = (self.questions[qid]["domain"], tuple(ids))
                grouped.setdefault(key, []).append(qid)
            for (domain, ids), qids in grouped.items():
                result.append(
                    HistoryGroup(
                        f"{tier}-{domain}-{digest(ids)[:16]}", tier, domain, ids, tuple(qids)
                    )
                )
        return result

    def histories(self, group):
        with self.paths["trajectories.jsonl"].open("rb") as stream:
            for tid in group.trajectory_ids:
                offset, length, _ = self.offsets[tid]
                stream.seek(offset)
                yield json.loads(stream.read(length))

    def reader_question(self, qid):
        q = self.questions[qid]
        return {
            "question": q["question"],
            "domain": q["domain"],
            "image": str(self.images[q["image"]]) if q.get("image") else None,
            "question_time": None,
        }


if __name__ == "__main__":
    main()
