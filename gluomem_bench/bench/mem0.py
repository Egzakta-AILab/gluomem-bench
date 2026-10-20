"""mem0 OSS head-to-head baseline (ingest + Retriever)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gluomem_bench.bench.adapters import make_adapter
from gluomem_bench.bench.mem0_input_policy import (
    plan_mem0_inputs,
    verify_mem0_block_receipts,
    verify_mem0_input_plan,
)
from gluomem_bench.bench.types import Prediction, RetrievedDoc, Sample

if TYPE_CHECKING:
    from gluomem_bench.bench.rack import RackProfile
    from gluomem_bench.bench.types import AnswerFn
    from gluomem_bench.config import AppConfig
    from gluomem_bench.ingest import Session

_SENTINEL = "ingest_meta.json"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _import_memory_cls() -> Any:
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    try:
        from mem0 import Memory
    except ImportError as exc:
        raise RuntimeError(
            "the 'mem0' retriever/system needs the optional dependency "
            "group — run `uv sync --extra mem0`"
        ) from exc
    return Memory


def _installed_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in ("mem0ai", "qdrant-client"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def _provider_model_id(slug: str) -> str:
    if slug.startswith("openrouter/"):
        return slug.removeprefix("openrouter/")
    return slug


def build_mem0_config(cfg: AppConfig, *, conv_dir: Path,
                     profile: RackProfile | None = None, embedder=None) -> dict[str, Any]:
    """Build the ``Memory.from_config`` dict for one conversation store."""
    llm_entry = cfg.models[cfg.benchmark.mem0.model]
    embed_entry = cfg.models[cfg.benchmark.mem0.embedder_model]
    llm_cfg: dict[str, Any] = {
        "model": _provider_model_id(llm_entry.provider_model),
        "openrouter_base_url": llm_entry.api_base or _OPENROUTER_BASE_URL,
    }
    if profile is None and llm_entry.api_key_env:
        api_key = os.environ.get(llm_entry.api_key_env)
        if api_key:
            llm_cfg["api_key"] = api_key
    if profile is not None:
        if (llm_entry.api_base != profile.base_url
                or llm_entry.provider_model != "openai/" + profile.model):
            raise ValueError("Mem0 model/endpoint differs from frozen rack profile")
        llm_cfg = {
            "model": profile.model,
            "openai_base_url": profile.base_url,
            "api_key": "rack-transport-placeholder",
            "temperature": profile.temperature,
            "top_p": 1.0,
            "max_tokens": profile.output_budget("event_claims"),
        }
    result = {
        "llm": {
            "provider": "openai",
            "config": llm_cfg,
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": embed_entry.provider_model},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0",
                "path": str(conv_dir / "qdrant"),
                "on_disk": True,
                "embedding_model_dims": embed_entry.dim,
            },
        },
        "history_db_path": str(conv_dir / "history.db"),
    }
    if embedder is not None:
        from gluomem_bench.bench.rack import shared_encoder_evidence

        if profile is None:
            raise ValueError("shared native encoder requires guarded rack profile")
        shared_encoder_evidence(cfg, embedder)
        # Native constructor only: never send text through OpenAI's newline rewriting.
        result["embedder"] = {"provider": "openai", "config": {
            "model": embed_entry.provider_model, "embedding_dims": embed_entry.dim,
            "openai_base_url": "http://127.0.0.1:8766/v1",
            "api_key": "shared-encoder-loopback-placeholder",
        }}
    elif profile is not None:
        result["embedder"]["config"]["model_kwargs"] = {
            "revision": embed_entry.revision,
            "local_files_only": True,
            "device": "cpu",
        }
    return result


class SharedMem0Embedding:
    """Per-memory native embedding protocol; preserve text bytes and input roles."""

    def __init__(self, embedder, config):
        self.encoder = embedder
        self.config = config
        self.fingerprint = embedder.fingerprint

    def _vectors(self, texts, action):
        import numpy as np

        if action not in (None, "add", "update", "search"):
            raise ValueError("unsupported native memory_action")
        if action == "search":
            vectors = [self.encoder.embed_query(text) for text in texts]
        else:
            vectors = self.encoder.embed_batch(texts)
        matrix = np.asarray(vectors, dtype=np.float32)
        if not texts:
            return []
        if (matrix.shape != (len(texts), self.encoder.dim) or not np.isfinite(matrix).all()
                or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4)):
            raise ValueError("shared native embedding invalid vector shape/norm")
        return matrix.tolist()

    def embed(self, text, memory_action=None):
        return self._vectors([text], memory_action)[0]

    def embed_batch(self, texts, memory_action="add"):
        return self._vectors(texts, memory_action)


def install_shared_mem0_encoder(memory, *, cfg, embedder):
    """Replace only this instance's embedding protocol, never native algorithms."""
    from gluomem_bench.bench.rack import shared_encoder_evidence

    evidence = shared_encoder_evidence(cfg, embedder)
    original = memory.embedding_model
    if (original.config.model != embedder.recipe.model
            or original.config.embedding_dims != embedder.dim):
        raise ValueError("native embedding configuration differs from shared recipe")
    original.client.close()
    memory.embedding_model = SharedMem0Embedding(embedder, original.config)
    memory._benchmark_embedding_evidence = evidence
    return evidence


def inspect_native_extraction(text: str) -> dict:
    """Observe raw extraction without repairing it or replacing the native parser."""
    try:
        value = json.loads(text, strict=False)
    except (ValueError, TypeError):
        return {"status": "malformed_json"}
    if (not isinstance(value, dict) or not isinstance(value.get("memory"), list)
            or any(not isinstance(m, dict) or not isinstance(m.get("text"), str)
                   or not m["text"].strip() or m.get("event", "ADD") != "ADD"
                   for m in value["memory"])):
        return {"status": "schema_incompatible"}
    return {"status": "valid" if value["memory"] else "valid_empty",
            "extracted_memories": len(value["memory"])}


class GuardedMem0SDK:
    """Per-instance SDK transport; native Mem0 prompts/algorithm stay unchanged."""

    def __init__(self, client, *, trace: Path, expected_system: str):
        self.rack = client
        self.trace = Path(trace)
        self.expected_system = expected_system
        self.records: list[dict] = []
        self.chat = self
        self.completions = self

    def create(self, **params):
        from openai.types.chat import ChatCompletion

        from gluomem_bench.bench.rack import FatalRackError

        record = {"ordinal": len(self.records) + 1, "task": "event_claims", "request": params}
        try:
            self.rack.check()
            expected_keys = {"model", "temperature", "top_p", "max_tokens", "messages",
                             "response_format"}
            if (set(params) != expected_keys or params["model"] != self.rack.profile.model
                    or params["temperature"] != 0 or params["top_p"] != 1.0
                    or params["max_tokens"] != self.rack.profile.output_budget("event_claims")
                    or params["response_format"] != {"type": "json_object"}
                    or len(params["messages"]) != 2
                    or params["messages"][0] != {"role": "system", "content": self.expected_system}
                    or params["messages"][1]["role"] != "user"):
                raise FatalRackError("unsupported native Mem0 provider request")
            result = self.rack.generate_messages(
                "event_claims", params["messages"], response_format=params["response_format"],
                top_p=params["top_p"],
            )
            record["response"] = result["response"]
            return ChatCompletion.model_validate(result["response"])
        except Exception as exc:
            from gluomem_bench.bench.embedding_runtime import safe_failure_diagnostic

            self.rack._latch(exc)
            record["error"] = {"type": "failure", "message": "diagnostic unavailable"}
            try:
                record["error"] = safe_failure_diagnostic(exc)
            except Exception:
                pass  # Telemetry must not replace the initiating exception.
            raise
        finally:
            try:
                self.records.append(record)
                with self.trace.open("ab") as handle:
                    material = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    handle.write(material.encode())
            except Exception:
                if "error" not in record:
                    raise  # Success still requires durable native request coverage.

    def close(self):
        """The campaign, not Mem0, owns the shared RackClient lifetime."""


def _session_messages(session: Session) -> list[dict[str, str]]:
    iso = session.date_time.isoformat()
    return [
        {
            "role": turn.speaker if turn.speaker in ("user", "assistant") else "user",
            "content": f"[{iso}] {turn.speaker}: {turn.text}",
        }
        for turn in session.turns
    ]


def _safe_mem0_config(value):
    if isinstance(value, dict):
        return {k: "<redacted>" if k in ("api_key", "password", "token")
                else _safe_mem0_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_mem0_config(v) for v in value]
    return value


def _mem0_digest(value):
    material = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(material).hexdigest()


def _capture_native_mem0_log(logger, record):
    """Intercept before *any* child/ancestor handler can format private arguments.

    A parent handler/filter is insufficient: child handlers run first and native
    children may disable propagation. This temporary dispatch hook also covers
    children created after capture starts. Unrelated and sub-warning dispatch is
    delegated unchanged. No original record or exception is modified.
    """
    cls = NativeMem0Diagnostics
    with cls._capture_lock:
        name = logger.name
        native = type(name) is str and (name == "mem0" or name.startswith("mem0."))
        fields = record.__dict__ if type(record) is logging.LogRecord else None
        level = fields.get("levelno") if type(fields) is dict else None
        if cls._captures and native and (type(level) is not int or level >= logging.WARNING):
            for capture in tuple(cls._captures):
                capture.emit(record)
            return  # Never forward raw records to child/root/lastResort handlers.
        original = cls._original_dispatch
    return original(logger, record)


class NativeMem0Diagnostics(logging.Handler):
    """Strict categorical admission evidence; never format native log payloads.

    Overlapping captures conservatively observe all Mem0 warnings (as before).
    The last close restores the exact prior dispatcher; existing handlers,
    filters, levels, propagation and record/exception objects remain untouched.
    """

    _capture_lock = threading.RLock()
    _captures = set()
    _original_dispatch = None

    def __init__(self, path):
        self._capturing = False
        super().__init__(level=logging.WARNING)
        self.path = Path(path)
        self.phase = "initialization"
        self.events = []
        self.path.write_bytes(b"[]\n")
        cls = NativeMem0Diagnostics
        with cls._capture_lock:
            if not cls._captures:
                cls._original_dispatch = logging.Logger.callHandlers
                logging.Logger.callHandlers = _capture_native_mem0_log
            cls._captures.add(self)
            self._capturing = True

    def emit(self, record):
        # Exact built-in types only: no subclass conversion, equality, hashing,
        # getMessage, msg/args, exception class names or traceback inspection.
        fields = record.__dict__ if type(record) is logging.LogRecord else None
        level = fields.get("levelno") if type(fields) is dict else None
        levels = {logging.WARNING: "WARNING", logging.ERROR: "ERROR", logging.CRITICAL: "CRITICAL"}
        phase = self.phase
        phase = phase if type(phase) is str and phase in (
            "initialization", "ingestion", "retrieval") else "unknown"
        event = {"logger": "mem0", "phase": phase,
                 "level": levels.get(level, "UNKNOWN") if type(level) is int else "UNKNOWN",
                 "message": "native Mem0 runtime diagnostic"}
        with NativeMem0Diagnostics._capture_lock:
            if self._capturing:
                self.events.append(event)
                self.path.write_bytes((json.dumps(self.events, ensure_ascii=False) + "\n").encode())

    def check(self):
        from gluomem_bench.bench.rack import FatalRackError

        if self.events:
            raise FatalRackError("native Mem0 runtime warning/error; see retained diagnostics")

    def close(self):
        cls = NativeMem0Diagnostics
        with cls._capture_lock:
            if self._capturing:
                cls._captures.remove(self)
                self._capturing = False
                if not cls._captures and logging.Logger.callHandlers is _capture_native_mem0_log:
                    logging.Logger.callHandlers = cls._original_dispatch
        super().close()


def close_mem0_memory(memory):
    """Native close releases SQLite only; close the owned local Qdrant client too."""
    clients = [memory.vector_store.client]
    entity = getattr(memory, "_entity_store", None)
    if entity is not None and entity.client is not clients[0]:
        clients.append(entity.client)
    try:
        memory.close()
    finally:
        try:
            for client in clients:
                client.close()
        finally:
            diagnostics = getattr(memory, "_benchmark_diagnostics", None)
            if diagnostics is not None:
                diagnostics.close()


def mem0_temporal_contract(clock_mode="native"):
    """Pinned OSS 2.2.1 rejects add.timestamp AND search.reference_date.

    Public signature presence does not establish feature support. No metadata,
    prompt rewrite or global clock patch can make this source-clock equivalent.
    """
    if clock_mode not in ("native", "source"):
        raise ValueError("unknown Mem0 clock_mode")
    return {"protocol": "native:mem0-oss-clock@v1", "clock_mode": clock_mode,
            "status": "admitted_disclosed" if clock_mode == "native" else "unsupported",
            "source_clock_supported": False, "add_timestamp_supported": False,
            "search_reference_date_supported": False,
            "observation_clock": "native execution date; not source session date",
            "search_clock": "native execution date; no reference_date override",
            "external_reader_clock": "unchanged Sample.reader_question question_date"}


def ingest_guarded_mem0(cfg, *, conv, out: Path, client, budget, embedder=None,
                        max_batch_input_tokens=None, count_query_tokens=None,
                        clock_mode="native"):
    """Native ADD-only ingestion, with attempted-input and committed-memory readback.

    The native rolling message buffer is not full source coverage. Actual complete
    extraction request receipts bind each submitted session; no gold is accepted.
    Malformed native output is retained but cannot publish completed QA evidence.
    """
    from gluomem_bench.bench.rack import FatalRackError

    out = Path(out)
    temporal = mem0_temporal_contract(clock_mode)
    (out / "mem0-temporal-contract.json").write_bytes(
        (json.dumps(temporal, sort_keys=True) + "\n").encode())
    if temporal["status"] == "unsupported":
        raise ValueError("Mem0 OSS source clock unsupported for add and search")
    if (os.environ.get("MEM0_TELEMETRY") != "False" or not os.environ.get("MEM0_DIR")
            or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        raise FatalRackError("Mem0 requires isolated import-time home, telemetry and providers")
    project = Path(__file__).resolve().parents[2]
    native_home = Path(os.environ["MEM0_DIR"]).resolve()
    if any(native_home.is_relative_to(p) for p in (project, project.parent / "GluoMem")):
        raise FatalRackError("Mem0 home must be outside source repositories")
    if (conv.conv_id in (".", "..") or Path(conv.conv_id).name != conv.conv_id
            or "/" in conv.conv_id or "\\" in conv.conv_id):
        raise FatalRackError("unsafe Mem0 conversation identity")
    memory_cls = _import_memory_cls()
    from mem0.configs.base import mem0_dir
    from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT
    from mem0.memory.telemetry import MEM0_TELEMETRY
    from mem0.memory.utils import parse_messages
    from mem0.utils.entity_extraction import extract_entities

    if MEM0_TELEMETRY or Path(mem0_dir).resolve() != native_home:
        raise FatalRackError("Mem0 was imported before environment isolation")
    if _installed_versions()["mem0ai"] != "2.2.1":
        raise FatalRackError("unadmitted Mem0 source version")
    conv_dir = out / "stores" / conv.conv_id
    conv_dir.mkdir(parents=True, exist_ok=False)
    config = build_mem0_config(cfg, conv_dir=conv_dir, profile=client.profile, embedder=embedder)

    def write(name, value):
        material = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        (out / name).write_bytes(material.encode())

    write("mem0-config.json", _safe_mem0_config(config))
    write("mem0-source.json", {
        "versions": _installed_versions(), "extractor_system": ADDITIVE_EXTRACTION_PROMPT,
        "clock_policy": "stock OSS observation/current date; source dates in content/metadata only",
        "temporal_contract": temporal,
        "input_protocol": ("native:mem0-session-input@v1" if max_batch_input_tokens is None
                           else "adapted:mem0-token-blocks@v1"),
        "parse_policy": "raw valid memory-list JSON required; no repair; otherwise incomplete",
        "search": {"top_k": cfg.benchmark.answer.context_k, "threshold": 0.1, "rerank": False},
        "native_auxiliary": "spaCy en_core_web_sm and fastembed Qdrant/bm25 required",
    })
    inputs = [{"session_index": s.index, "session_date": s.date_time.isoformat(),
               "messages": _session_messages(s)} for s in conv.sessions]
    original_inputs = inputs
    input_plan = None
    if max_batch_input_tokens is not None:
        input_plan = plan_mem0_inputs(inputs, scope=conv.conv_id,
                                     max_batch_input_tokens=max_batch_input_tokens,
                                     count_query_tokens=count_query_tokens)
        write("mem0-original-inputs.json", original_inputs)
        write("mem0-input-plan.json", input_plan)
        inputs = input_plan["blocks"]
    write("mem0-source-inputs.json", inputs)
    memory = None
    diagnostics = NativeMem0Diagnostics(out / "mem0-diagnostics.json")
    sessions = []
    allowed_messages = Counter()
    try:
        memory = memory_cls.from_config(config)
        memory._benchmark_diagnostics = diagnostics
        if embedder is not None:
            write("mem0-embedding.json", install_shared_mem0_encoder(
                memory, cfg=cfg, embedder=embedder))
        dense_model = cfg.models[cfg.benchmark.mem0.embedder_model].provider_model
        if (memory.llm.config.model != client.profile.model
                or str(memory.llm.client.base_url).rstrip("/") != client.profile.base_url
                or memory.embedding_model.config.model != dense_model
                or not memory.vector_store.is_local):
            raise FatalRackError("instantiated Mem0 providers differ from admitted configuration")
        if (memory.vector_store._get_bm25_encoder() is None
                or not extract_entities("Microsoft Azure in Prague and United Airlines")):
            raise FatalRackError("Mem0 native BM25/NLP features degraded")
        diagnostics.check()
        memory.llm.client.close()
        sdk = GuardedMem0SDK(client, trace=out / "mem0-native.jsonl",
                             expected_system=ADDITIVE_EXTRACTION_PROMPT)
        memory.llm.client = sdk
        diagnostics.phase = "ingestion"
        for source in inputs:
            budget.check()
            before = len(sdk.records)
            result = memory.add(source["messages"], user_id=conv.conv_id,
                                metadata={"session_index": source["session_index"],
                                          "session_date": source["session_date"]}, infer=True)
            if len(sdk.records) != before + 1 or sdk.records[-1].get("error"):
                raise FatalRackError("Mem0 native extraction request coverage differs")
            record = sdk.records[-1]
            expected_segment = ("## New Messages\n" + parse_messages(source["messages"])
                                + "\n\n## Observation Date\n")
            if expected_segment not in record["request"]["messages"][1]["content"]:
                raise FatalRackError("Mem0 request does not contain the complete allowed session")
            raw_text = record["response"]["choices"][0]["message"]["content"]
            extracted = inspect_native_extraction(raw_text)
            if not isinstance(result, dict) or not isinstance(result.get("results"), list):
                raise FatalRackError("Mem0 native ADD acknowledgment missing results list")
            added = result["results"]
            committed = []
            for item in added:
                saved = memory.get(item["id"])
                if (item.get("event") != "ADD" or saved is None
                        or saved.get("user_id") != conv.conv_id
                        or saved.get("memory") != item.get("memory")):
                    raise FatalRackError("Mem0 reported memory lacks committed scoped readback")
                committed.append(saved)
            allowed_messages.update((m["role"], m["content"]) for m in source["messages"])
            buffered = memory.db.connection.execute(
                "SELECT role, content FROM messages ORDER BY rowid").fetchall()
            if len(buffered) > 10 or Counter(tuple(r) for r in buffered) - allowed_messages:
                raise FatalRackError("Mem0 rolling buffer contains unsubmitted source")
            sessions.append({
                "session_index": source["session_index"], "session_date": source["session_date"],
                "messages_sha256": _mem0_digest(source["messages"]),
                "submitted_turns": len(source["messages"]), "ordinal": record["ordinal"],
                "native_request_sha256": _mem0_digest(record["request"]),
                "native_response_sha256": _mem0_digest(record["response"]),
                "extraction": extracted, "added": committed,
                "rolling_buffer": [list(r) for r in buffered],
            })
            if input_plan is not None:
                sessions[-1].update({"source_session_ordinal": source["source_session_ordinal"],
                                     "block_index": source["block_index"],
                                     "source_parts": source["source_parts"],
                                     "native_add_ack": True})
            write("mem0-input-receipts.json", sessions)
            if extracted["status"] not in ("valid", "valid_empty"):
                raise FatalRackError("native Mem0 extraction " + extracted["status"])
            diagnostics.check()
            budget.check()
        snapshot = memory.get_all(filters={"user_id": conv.conv_id}, top_k=10000)["results"]
        if len(snapshot) == 10000 or any(m.get("user_id") != conv.conv_id for m in snapshot):
            raise FatalRackError("Mem0 final snapshot cannot establish complete scoped inventory")
        write("mem0-memories.json", snapshot)
        history = memory.db.connection.execute(
            "SELECT memory_id, new_memory, event FROM history ORDER BY rowid").fetchall()
        if Counter((r[0], r[1]) for r in history if r[2] == "ADD") != Counter(
                (m["id"], m["memory"]) for m in snapshot):
            raise FatalRackError("Mem0 history and vector-store inventory differ")
        write("mem0-history.json", [list(r) for r in history])
        summary = {"complete": True, "sessions": sessions, "memories": len(snapshot),
                   "submitted_turns": sum(s["submitted_turns"] for s in sessions),
                   "source_inputs_sha256": _mem0_digest(inputs), "versions": _installed_versions()}
        if input_plan is not None:
            summary.update({"input_protocol": input_plan["protocol"],
                            "original_inputs_sha256": _mem0_digest(original_inputs),
                            "input_plan_sha256": _mem0_digest(input_plan),
                            "original_submitted_turns": input_plan["original_turns"],
                            "native_add_calls": len(sessions),
                            "native_extraction_calls": len(sdk.records),
                            "temporal_contract": temporal})
        budget.check()
        write("mem0-ingestion.json", summary)
        material = json.dumps(summary, ensure_ascii=False) + "\n"
        (conv_dir / _SENTINEL).write_bytes(material.encode())
        return memory
    except Exception:
        if memory is not None:
            close_mem0_memory(memory)
        else:
            diagnostics.close()
        raise


def mem0_adapted_binding(cfg, original_inputs, *, scope):
    """Derive both digest domains from independently reconstructed allowed input."""
    from gluomem_bench.bench.tokenizer_policy import mem0_token_counter

    counter = mem0_token_counter(cfg)
    if counter is None:
        raise ValueError("explicit adapted Mem0 configuration required")
    plan = plan_mem0_inputs(original_inputs, scope=scope,
        max_batch_input_tokens=cfg.benchmark.mem0.max_batch_input_tokens,
        count_query_tokens=counter)
    return plan, {"mem0_input_protocol": plan["protocol"],
        "mem0_original_source_inputs_sha256": _mem0_digest(original_inputs),
        "mem0_source_inputs_sha256": _mem0_digest(plan["blocks"]),
        "mem0_input_plan_sha256": _mem0_digest(plan),
        "mem0_max_batch_input_tokens": cfg.benchmark.mem0.max_batch_input_tokens,
        "mem0_clock_mode": cfg.benchmark.mem0.clock_mode,
        "mem0_token_counter": counter.metadata}


def verify_mem0_adapted_evidence(out, inventory, protocol, *, cfg, original_inputs,
                                 scope, expected=None):
    """Additional adapted-only guard; legacy native evidence checks stay intact.

    Caller supplies source reconstructed from a pinned adapter/cache, NEVER
    receipts, and independent effective config/spec expectations. No SQL opens.
    """
    if cfg is None or original_inputs is None or not scope:
        raise ValueError("independent Mem0 original source and config required")
    required = {"mem0-original-inputs.json", "mem0-input-plan.json",
                "mem0-temporal-contract.json"}
    if not required <= inventory.keys():
        raise ValueError("adapted Mem0 evidence missing")
    out = Path(out)
    def read(name):
        return json.loads((out / name).read_bytes())
    if read("effective-config.json") != cfg.model_dump(mode="json"):
        raise ValueError("adapted Mem0 effective config differs")
    plan, binding = mem0_adapted_binding(cfg, original_inputs, scope=scope)
    if (_mem0_digest({k: protocol.get(k) for k in binding}) != _mem0_digest(binding)
            or (expected is not None and _mem0_digest(
                {k: expected.get(k) for k in binding}) != _mem0_digest(binding))):
        raise ValueError("adapted Mem0 source/counter binding differs from frozen condition")
    saved = read("mem0-input-plan.json")
    receipts = read("mem0-input-receipts.json")
    from gluomem_bench.bench.tokenizer_policy import mem0_token_counter

    counter = mem0_token_counter(cfg)
    verify_mem0_block_receipts(original_inputs, saved, receipts, scope=scope,
        max_batch_input_tokens=cfg.benchmark.mem0.max_batch_input_tokens,
        count_query_tokens=counter)
    from datetime import date

    clock = protocol.get("mem0_native_clock", {})
    day = clock.get("execution_day_at_start")
    if (type(day) is not str or date.fromisoformat(day).isoformat() != day
            or clock != {"mode": "native", "execution_day_at_start": day,
                         "source_dates_override_native_clock": False}):
        raise ValueError("Mem0 native execution-day disclosure differs")
    summary = read("mem0-ingestion.json")
    temporal = mem0_temporal_contract(cfg.benchmark.mem0.clock_mode)
    wanted = {"input_protocol": plan["protocol"],
        "original_inputs_sha256": _mem0_digest(original_inputs),
        "input_plan_sha256": _mem0_digest(plan), "original_submitted_turns": plan["original_turns"],
        "native_add_calls": len(plan["blocks"]), "native_extraction_calls": len(plan["blocks"]),
        "temporal_contract": temporal}
    if (_mem0_digest(read("mem0-original-inputs.json")) != _mem0_digest(original_inputs)
            or _mem0_digest(read("mem0-source-inputs.json")) != _mem0_digest(plan["blocks"])
            or _mem0_digest({k: summary.get(k) for k in wanted}) != _mem0_digest(wanted)
            or temporal["status"] != "admitted_disclosed"
            or read("mem0-temporal-contract.json") != temporal
            or read("mem0-source.json").get("temporal_contract") != temporal
            or read("mem0-source.json").get("input_protocol") != plan["protocol"]
            or any(row.get("user_id") != scope for row in read("mem0-memories.json"))):
        raise ValueError("adapted Mem0 original/block/summary/clock evidence differs")
    return {**(expected or {}), **binding}


@dataclass(frozen=True)
class Mem0IngestReport:
    conv_id: str
    sessions: int
    turns: int
    memories: int
    seconds: float
    usage: dict[str, dict[str, int]]
    cost_usd: float | None


def _memories_added(add_result: Any) -> int:
    if isinstance(add_result, dict):
        results = add_result.get("results")
        if isinstance(results, list):
            return len(results)
    return 0


def ingest_mem0_per_conversation(
    cfg: AppConfig,
    *,
    dataset: str,
    db_dir: Path,
    skip_existing: bool = False,
    conv_ids: set[str] | None = None,
) -> list[Mem0IngestReport]:
    """Ingest one mem0 store per conversation under ``<db_dir>/<conv_id>/``."""
    memory_cls = _import_memory_cls()
    adapter = make_adapter(dataset, cfg)
    adapter.verify()
    db_dir.mkdir(parents=True, exist_ok=True)
    batch_started = time.perf_counter()

    convs = adapter.conversations()
    if conv_ids is not None:
        convs = [c for c in convs if c.conv_id in conv_ids]

    todo: list[Any] = []
    for conv in convs:
        conv_dir = db_dir / conv.conv_id
        sentinel = conv_dir / _SENTINEL
        if conv_dir.exists():
            if sentinel.is_file():
                if skip_existing:
                    print(f"skipped {conv.conv_id}: {sentinel} exists")
                    continue
                raise RuntimeError(
                    f"{conv_dir} already ingested; pass --skip-existing to "
                    f"resume around it or delete the directory to redo"
                )
            raise RuntimeError(
                f"{conv_dir} exists without {_SENTINEL} — partial ingest; "
                f"delete the directory and rerun"
            )
        todo.append(conv)

    workers = min(cfg.write.conv_parallelism, max(1, len(todo)))
    parallel = workers > 1 and len(todo) > 1

    def _ingest_one(conv: Any) -> Mem0IngestReport:
        conv_dir = db_dir / conv.conv_id
        conv_dir.mkdir(parents=True)
        memory = memory_cls.from_config(build_mem0_config(cfg, conv_dir=conv_dir))
        started = time.perf_counter()
        memories = 0
        turns = 0
        for session in conv.sessions:
            messages = _session_messages(session)
            turns += len(messages)
            result = memory.add(
                messages,
                user_id=conv.conv_id,
                metadata={
                    "session_index": session.index,
                    "session_date": session.date_time.isoformat(),
                },
                infer=cfg.benchmark.mem0.infer,
            )
            memories += _memories_added(result)
        seconds = time.perf_counter() - started
        report = Mem0IngestReport(
            conv_id=conv.conv_id,
            sessions=len(conv.sessions),
            turns=turns,
            memories=memories,
            seconds=seconds,
            usage={},
            cost_usd=None,
        )
        meta = {
            "conv_id": conv.conv_id,
            "sessions": report.sessions,
            "turns": report.turns,
            "memories": report.memories,
            "seconds": round(seconds, 3),
            "usage": {},
            "usage_measured": False,
            "cost_usd": None,
            "versions": _installed_versions(),
            "mem0_config": _safe_mem0_config(build_mem0_config(cfg, conv_dir=conv_dir)),
        }
        sentinel = conv_dir / _SENTINEL
        tmp = sentinel.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        tmp.replace(sentinel)
        print(
            f"ingested {conv.conv_id} (mem0): {report.memories} memories from "
            f"{report.turns} turns / {report.sessions} sessions | {seconds:.0f}s",
            flush=True,
        )
        return report

    reports_by_id: dict[str, Mem0IngestReport] = {}
    failures: list[str] = []
    if parallel:
        from concurrent.futures import ThreadPoolExecutor

        def _run(conv: Any) -> None:
            try:
                reports_by_id[conv.conv_id] = _ingest_one(conv)
            except Exception as exc:  # noqa: BLE001
                failures.append(conv.conv_id)
                print(
                    f"mem0 ingest failed for {conv.conv_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_run, todo))
    else:
        for conv in todo:
            reports_by_id[conv.conv_id] = _ingest_one(conv)

    if failures:
        print(f"mem0 ingest failures: {sorted(failures)}", file=sys.stderr)
    reports = [reports_by_id[c.conv_id] for c in todo if c.conv_id in reports_by_id]
    total = sum(r.memories for r in reports)
    print(
        f"mem0 ingest total: {len(reports)} conversation(s), {total} memories "
        f"in {time.perf_counter() - batch_started:.0f}s"
    )
    return reports


class Mem0Retriever:
    """``Retriever`` protocol implementation over per-conversation mem0 stores."""

    name = "mem0"

    def __init__(self, cfg: AppConfig, *, db_dir: Path, answer_fn: AnswerFn,
                 prepared_memories: dict[str, Any] | None = None,
                 clock_mode="native") -> None:
        self._temporal_contract = mem0_temporal_contract(clock_mode)
        if self._temporal_contract["status"] == "unsupported":
            raise ValueError("Mem0 OSS source clock unsupported for add and search")
        self._memory_cls = _import_memory_cls() if prepared_memories is None else None
        self._prepared_only = prepared_memories is not None
        self._cfg = cfg
        self._db_dir = Path(db_dir)
        self._answer_fn = answer_fn
        self._memories: dict[str, Any] = dict(prepared_memories or {})
        self._lock = threading.Lock()

    @property
    def _top_k(self) -> int:
        return (
            self._cfg.benchmark.mem0.top_k
            if self._cfg.benchmark.mem0.top_k is not None
            else self._cfg.benchmark.answer.context_k
        )

    def _memory_for(self, conv_id: str) -> Any:
        memory = self._memories.get(conv_id)
        if memory is None:
            if self._prepared_only:
                raise FileNotFoundError("no guarded prepared Mem0 instance for this history")
            conv_dir = self._db_dir / conv_id
            if not (conv_dir / _SENTINEL).is_file():
                raise FileNotFoundError(
                    f"no completed mem0 store for conversation '{conv_id}' at "
                    f"{conv_dir} — run `gluomem bench ingest --dataset locomo "
                    f"--system mem0 --db-dir {self._db_dir}` first"
                )
            assert self._memory_cls is not None
            memory = self._memory_cls.from_config(build_mem0_config(self._cfg, conv_dir=conv_dir))
            self._memories[conv_id] = memory
        return memory

    def answer(self, s: Sample) -> Prediction:
        start = time.perf_counter()
        with self._lock:
            memory = self._memory_for(s.conv_id)
            diagnostics = getattr(memory, "_benchmark_diagnostics", None)
            if diagnostics is not None:
                diagnostics.phase = "retrieval"
            response = memory.search(
                s.question,
                filters={"user_id": s.conv_id},
                top_k=self._top_k,
            )
            if diagnostics is not None:
                diagnostics.check()
        results = response.get("results", []) if isinstance(response, dict) else []
        if not results:
            return Prediction(
                sample_id=s.id,
                answer=self._cfg.benchmark.answer.abstain_text,
                abstained=True,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                abstention={"reason": "no_hits", "candidates": 0},
            )
        docs: list[RetrievedDoc] = []
        lines: list[str] = []
        for rank, row in enumerate(results):
            text = str(row.get("memory", ""))
            metadata = row.get("metadata") or {}
            date = metadata.get("session_date") or row.get("created_at") or ""
            docs.append(
                RetrievedDoc(
                    rank=rank,
                    text=text,
                    score=float(row.get("score") or 0.0),
                )
            )
            lines.append(f"[{date}] {text}")
        out = self._answer_fn(s.reader_question, "\n".join(lines))
        return Prediction(
            sample_id=s.id,
            answer=out.text,
            retrieved=tuple(docs),
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )

    def manifest_extra(self) -> dict[str, Any]:
        """Provenance for the run manifest."""
        ingest_meta: dict[str, Any] = {}
        if self._db_dir.is_dir():
            for sentinel in sorted(self._db_dir.glob(f"*/{_SENTINEL}")):
                meta = json.loads(sentinel.read_text(encoding="utf-8"))
                ingest_meta[meta.get("conv_id", sentinel.parent.name)] = meta
        return {
            "system": "mem0-oss",
            "temporal_contract": self._temporal_contract,
            "versions": _installed_versions(),
            "mem0_config": _safe_mem0_config(
                build_mem0_config(self._cfg, conv_dir=self._db_dir / "<conv_id>")),
            "llm_policy": "openai-compatible OpenRouter endpoint",
            "search_params": {
                "top_k": self._top_k,
                "threshold": "mem0-default",
                "rerank": "mem0-default",
            },
            "identity_mapping": (
                "user_id=conv_id; native user/assistant roles preserved; "
                "named speakers role=user; content "
                "'[session_iso] Speaker: text'; session date also in metadata"
            ),
            "evidence_metrics": "not-applicable (distilled memories, no dia_ids)",
            "ingest_meta": ingest_meta,
        }


__all__ = [
    "Mem0IngestReport",
    "Mem0Retriever",
    "build_mem0_config",
    "ingest_mem0_per_conversation",
    "mem0_adapted_binding",
    "verify_mem0_adapted_evidence",
    "verify_mem0_input_plan",
    "verify_mem0_block_receipts",
]
