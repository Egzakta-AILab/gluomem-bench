"""Evidence-only backends; no reader, dataset loader or commercial defaults.

Shared async API: ``add(chunks: Sequence[Evidence]) -> None``,
``search(query: str, *, limit: int | None = None, top_k: int | None = None)
-> list[Evidence]``, ``close() -> None``. Supply only one limit alias (default 10).
Raw conditions ingest identical caller-owned chunks without re-chunking::

    raw = RawEvidenceBackend("hybrid", encoder=remote.embed_batch,
                             query_encoder=remote.embed_query)
    await raw.add([Evidence("message-1", "original text", {"session": "s1"})])
    hits = await raw.search("question", limit=10)
    packed = pack_evidence(hits, token_budget=4096, count_tokens=served_tokenizer)
    await raw.close()

Native: ``CogneeEvidenceBackend(CogneeSettings(...))`` uses the same async API.
All deployment fields are mandatory (see CogneeSettings); ``root`` is a fresh
external Path, and ``python_executable`` points to an isolated cognee==1.6.5 env.
Native signatures inspected in the pinned wheel: add(data, dataset_name,
embedding_config, run_in_background); cognify(datasets, chunk_size,
chunks_per_batch, data_per_batch, embedding_config, extractor, raise_on_error,
run_in_background); search(query_text, query_type, datasets, top_k, verbose,
embedding_config). These are keyword subsets, not replacements for native APIs.
CHUNKS returns native evidence, not graph-completion answers. Native cognify still
builds the graph; it is intentionally not replaced with the raw baseline index.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit


def _diagnostics():
    # Both package and absolute-file isolated workers load this stdlib-only sibling.
    import importlib.util

    name = "_cognee_safe_diagnostics"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("cognee_native_boundary.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    score: float = 0.0


class EvidenceBackend(Protocol):
    async def add(self, chunks: Sequence[Evidence]) -> None: ...
    async def search(
        self, query: str, *, top_k: int | None = None, limit: int | None = None
    ) -> list[Evidence]: ...
    async def close(self) -> None: ...


def _terms(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


Encoder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


@dataclass(frozen=True)
class PackedEvidence:
    items: tuple[Evidence, ...]
    text: str
    tokens: int


def pack_evidence(
    items: Sequence[Evidence], *, token_budget: int, count_tokens: Callable[[str], int]
) -> PackedEvidence:
    """Greedy rank-order whole-chunk packing; count the exact rendered string.

    Skips oversized chunks, includes citation labels/separators in the count.
    This is an evidence allowance, not full reader-prompt/context admission.
    """
    if token_budget < 0:
        raise ValueError("token_budget must be nonnegative")
    accepted: list[Evidence] = []
    text = ""
    tokens = 0
    for item in items:
        candidate = text + ("\n\n" if text else "") + f"[{item.id}]\n{item.text}"
        size = count_tokens(candidate)
        if size < 0:
            raise ValueError("token count must be nonnegative")
        if size <= token_budget:
            accepted.append(item)
            text, tokens = candidate, size
    return PackedEvidence(tuple(accepted), text, tokens)


def _validate_chunks(chunks: Sequence[Evidence], existing: Mapping[str, Any]) -> None:
    seen = set(existing)
    for chunk in chunks:
        if not chunk.id or not chunk.text:
            raise ValueError("evidence id and text must be nonempty")
        if chunk.id in seen:
            raise ValueError("duplicate evidence id")
        seen.add(chunk.id)


def _vectors(encoder: Encoder, texts: Sequence[str], dimension: int | None) -> list[list[float]]:
    vectors = [list(map(float, vector)) for vector in encoder(texts)]
    if len(vectors) != len(texts):
        raise ValueError("encoder returned incorrect vector count")
    for vector in vectors:
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("embedding vectors must be nonempty and finite")
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension:
            raise ValueError("embedding dimension mismatch")
    return vectors


def reciprocal_rank_fusion(lanes: Sequence[Sequence[Evidence]], *, k: int = 60) -> list[Evidence]:
    """Deduplicate each lane before rank assignment; ties use stable evidence IDs."""
    if k < 0:
        raise ValueError("RRF k must be nonnegative")
    scores: dict[str, float] = {}
    chunks: dict[str, Evidence] = {}
    for lane in lanes:
        seen: set[str] = set()
        for chunk in lane:
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            chunks.setdefault(chunk.id, chunk)
            scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (k + len(seen))
    return [
        replace(chunks[key], score=score)
        for key, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


class RawEvidenceBackend:
    """In-memory Okapi BM25 (k1=1.5, b=.75), without re-chunking or extraction."""

    def __init__(
        self,
        mode: str,
        *,
        encoder: Encoder | None = None,
        query_encoder: Callable[[str], Sequence[float]] | None = None,
    ):
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("unsupported retrieval mode")
        if mode != "bm25" and encoder is None:
            raise ValueError("dense/hybrid requires an explicit shared encoder")
        self.mode = mode
        self.encoder = encoder
        self.query_encoder = query_encoder
        self.vectors: dict[str, Sequence[float]] = {}
        self.chunks: dict[str, Evidence] = {}
        self.closed = False

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError("backend is closed")

    async def add(self, chunks: Sequence[Evidence]) -> None:
        self._check_open()
        _validate_chunks(chunks, self.chunks)
        if self.encoder is not None and chunks:
            dimension = len(next(iter(self.vectors.values()))) if self.vectors else None
            vectors = _vectors(self.encoder, [chunk.text for chunk in chunks], dimension)
            self.vectors.update(
                (chunk.id, vector) for chunk, vector in zip(chunks, vectors, strict=True)
            )
        self.chunks.update((chunk.id, chunk) for chunk in chunks)

    async def search(
        self, query: str, *, top_k: int | None = None, limit: int | None = None
    ) -> list[Evidence]:
        self._check_open()
        top_k = _search_limit(top_k, limit)
        if not self.chunks or not query.strip() or top_k == 0:
            return []
        lexical = self._bm25(query)
        if self.mode == "bm25":
            return lexical[:top_k]
        assert self.encoder is not None
        query_encoder = self.query_encoder
        query_batch = (
            (lambda texts: [query_encoder(text) for text in texts])
            if query_encoder is not None
            else self.encoder
        )
        query_vector = _vectors(query_batch, [query], len(next(iter(self.vectors.values()))))[0]
        dense = [
            replace(chunk, score=_cosine(query_vector, self.vectors[key]))
            for key, chunk in self.chunks.items()
        ]
        dense.sort(key=lambda chunk: (-chunk.score, chunk.id))
        if self.mode == "dense":
            return dense[:top_k]
        # Full-corpus lanes avoid changing hybrid behavior with the requested output k.
        return reciprocal_rank_fusion([lexical, dense])[:top_k]

    def _bm25(self, query: str) -> list[Evidence]:
        counts = {key: Counter(_terms(chunk.text)) for key, chunk in self.chunks.items()}
        lengths = {key: sum(counter.values()) for key, counter in counts.items()}
        average = sum(lengths.values()) / len(counts) or 1
        scores = dict.fromkeys(counts, 0.0)
        for term in set(_terms(query)):
            df = sum(term in counter for counter in counts.values())
            idf = math.log(1 + (len(counts) - df + 0.5) / (df + 0.5))
            for key, counter in counts.items():
                tf = counter[term]
                denominator = tf + 1.5 * (0.25 + 0.75 * lengths[key] / average)
                scores[key] += idf * tf * 2.5 / denominator
        return [
            replace(self.chunks[key], score=score)
            for key, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            if score > 0
        ]

    async def close(self) -> None:
        self.closed = True
        self.chunks.clear()
        self.vectors.clear()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    denominator = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / denominator if denominator else 0.0


def _search_limit(top_k: int | None, limit: int | None) -> int:
    if top_k is not None and limit is not None:
        raise ValueError("cannot specify both top_k and limit")
    value = top_k if top_k is not None else (limit if limit is not None else 10)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("search limit must be a nonnegative integer")
    return value


COGNEE_VERSION = "1.6.5"
# PyPI cognee-1.6.5-py3-none-any.whl; native source is shipped in that wheel.
COGNEE_WHEEL_SHA256 = "aaef9fdc02f542b1d125cde21759dde2fac56355d8928e39c734e7b5c6705eca"


@dataclass(frozen=True)
class CogneeSettings:
    python_executable: str
    root: Path
    dataset: str
    llm_url: str = field(repr=False)
    llm_model: str
    llm_api_key: str = field(repr=False)
    embedding_url: str = field(repr=False)
    embedding_model: str
    embedding_dimensions: int
    embedding_api_key: str = field(repr=False)
    max_completion_tokens: int = 4096
    chunk_size: int = 512
    timeout_seconds: float = 300
    # Opt-in v2: transport request allowance is NOT a whole native RPC limit.
    request_timeout_seconds: float | None = None
    operation_deadline: float | None = None  # absolute time.monotonic(), shared history clock
    native_llm_timeout_seconds: float = 90
    cleanup_reserve_seconds: float = 0  # explicit slice WITHIN operation_deadline, never extra time

    def __post_init__(self) -> None:
        for endpoint in (self.llm_url, self.embedding_url):
            parsed = urlsplit(endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("explicit HTTP(S) endpoint required")
        if self.embedding_dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")
        if not self.dataset.strip():
            raise ValueError("dataset must be explicit")
        if not all(
            (
                self.llm_model,
                self.embedding_model,
                self.llm_api_key,
                self.embedding_api_key,
                self.python_executable,
            )
        ):
            raise ValueError("models, API keys (local placeholders permitted), and Python required")
        for value in (
            self.timeout_seconds,
            self.request_timeout_seconds,
            self.operation_deadline,
            self.native_llm_timeout_seconds,
        ):
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError("clocks must be positive finite numbers")
        if min(self.max_completion_tokens, self.chunk_size, self.timeout_seconds) <= 0:
            raise ValueError("budgets must be positive")
        if (
            type(self.cleanup_reserve_seconds) not in (int, float)
            or not math.isfinite(self.cleanup_reserve_seconds)
            or self.cleanup_reserve_seconds < 0
        ):
            raise ValueError("cleanup reserve must be finite and nonnegative")


def cognee_chunk_evidence(results: Sequence[Mapping[str, Any]]) -> list[Evidence]:
    """Parse pinned CHUNKS/verbose output, never replace it with local retrieval.

    Cognee's score is backend cosine distance (lower is better); negate for the
    common higher-is-better interface, retaining the untouched native payload.
    """
    evidence = []
    for result in results:
        if result.get("error"):
            raise RuntimeError("native search failed; inspect external native receipt")
        for payload in result.get("text_result") or []:
            if not isinstance(payload, dict) or not payload.get("id") or not payload.get("text"):
                raise ValueError("native chunk payload lacks id/text provenance")
            distance = float(payload["score"])
            if not math.isfinite(distance):
                raise ValueError("non-finite native distance")
            evidence.append(
                Evidence(
                    str(payload["id"]),
                    payload["text"],
                    {
                        "backend": "cognee",
                        "dataset_id": str(result.get("dataset_id", "")),
                        "dataset_name": result.get("dataset_name"),
                        "native_payload": dict(payload),
                        "score_kind": "negative_distance",
                    },
                    -distance,
                )
            )
    return evidence


class CogneeEvidenceBackend:
    """Pinned native OSS lifecycle in a dedicated, operator-owned Python process.

    Install ``cognee==1.6.5`` in ``settings.python_executable``'s environment.
    ``add`` calls native add then native cognify (LLM extraction, no replacement).
    ``search`` calls native CHUNKS with verbose=True, not an answer generator.
    Cognee controls its own chunking: only the three raw conditions are guaranteed
    identical retrieval units. Caller IDs/provenance are retained in inputs.jsonl;
    returned provenance is native chunk/document/dataset metadata, not fabricated
    attribution to original caller IDs. Close exits the worker to release all DBs.
    Supply a guarded OpenAI-compatible proxy URL to enforce a shared call budget;
    native extraction may issue multiple calls. No budget is silently inferred.
    """

    def __init__(self, settings: CogneeSettings):
        self.settings = settings
        if settings.root.exists() and any(settings.root.iterdir()):
            raise FileExistsError("Cognee requires a fresh external store directory")
        self.closed = False
        self._process: asyncio.subprocess.Process | None = None
        self._log: Any = None
        self._chunks: dict[str, Evidence] = {}
        self._lock = asyncio.Lock()
        self._deadline: float | None = settings.operation_deadline

    async def _rpc(self, operation: str, **payload: Any) -> Any:
        deadline = self.settings.operation_deadline
        if deadline is None:
            deadline = time.monotonic() + self.settings.timeout_seconds
        if self.settings.operation_deadline is not None and self._deadline is not None:
            deadline = min(self._deadline, deadline)
        self._deadline = deadline
        scope = asyncio.timeout(
            max(0, deadline - self.settings.cleanup_reserve_seconds - time.monotonic())
        )
        expired = False
        try:
            work_deadline = deadline - self.settings.cleanup_reserve_seconds
            if work_deadline <= time.monotonic():
                expired = True
                raise TimeoutError()
            # Admission/lock, lazy init and native work share this one clock.
            async with scope:
                async with self._lock:
                    return await self._exchange(operation, payload)
        except BaseException as exc:
            try:
                if not hasattr(exc, "native_diagnostic"):
                    clock = (
                        "operation_deadline"
                        if self.settings.operation_deadline
                        else "legacy_rpc_timeout"
                    )
                    exc.native_diagnostic = _diagnostics().native_diagnostic(
                        exc, "rpc", clock if expired or scope.expired() else "unknown"
                    )
            except Exception:
                pass
            self.closed = True
            try:
                await self._terminate()
            except BaseException:
                pass  # Cleanup must never replace the initiating exception.
            raise

    async def _exchange(self, operation: str, payload: dict) -> Any:
        if self._process is None:
            root = self.settings.root.resolve()
            if root.exists() and any(root.iterdir()):
                raise FileExistsError("Cognee requires a fresh external store directory")
            root.mkdir(parents=True, exist_ok=True)
            # Exclusive ownership prevents two workers acquiring the same store.
            with (root / "owner.lock").open("x", encoding="utf-8") as lock:
                lock.write("native-cognee-worker\n")
            self._log = (root / "native.log").open("w", encoding="utf-8")
            allowed = {
                "PATH",
                "SYSTEMROOT",
                "WINDIR",
                "TEMP",
                "TMP",
                "TMPDIR",
                "HOME",
                "USERPROFILE",
                "LOCALAPPDATA",
                "APPDATA",
                "SSL_CERT_FILE",
                "HF_HOME",
                "HF_HUB_CACHE",
                "HUGGINGFACE_HUB_CACHE",
            }
            env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
            env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            self._process = await asyncio.create_subprocess_exec(
                *[
                    self.settings.python_executable,
                    "-B",
                    "-I",
                    str(Path(__file__).with_name(self._worker_file).resolve()),
                    self._worker_flag,
                ],
                cwd=root,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=self._log,
                limit=sys.maxsize,  # preserve the prior readline protocol; no new payload allowance
            )
            config = asdict(self.settings)
            config["root"] = str(root)
            await self._send({"operation": "init", "settings": config})
        return await self._send({"operation": operation, **payload})

    _worker_file = "comparison_backends.py"
    _worker_flag = "--cognee-worker"

    async def _send(self, request: dict) -> Any:
        assert self._process is not None
        assert self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
        await self._process.stdin.drain()
        line = await self._process.stdout.readline()
        if not line:
            raise RuntimeError("Cognee worker exited; inspect external native.log")
        response = json.loads(line)
        if not response["ok"]:
            error = RuntimeError("Cognee native failure; inspect categorical diagnostic")
            error.native_diagnostic = _diagnostics().clean_diagnostic(response.get("diagnostic"))
            raise error
        return response.get("result")

    async def _terminate(self) -> None:
        try:
            if self._process is not None:
                if self._process.returncode is None:
                    self._process.kill()
                # No executor/readline thread survives cancellation. No fresh cleanup clock.
                remaining = max(0, self._deadline - time.monotonic()) if self._deadline else 0
                if self._process.returncode is None:
                    await asyncio.wait_for(self._process.wait(), remaining)
        finally:
            if self._process is not None:
                self._process._transport.close()
            if self._log:
                self._log.close()

    @property
    def worker_closed(self) -> bool:
        return self._process is None or self._process.returncode is not None

    async def add(self, chunks: Sequence[Evidence]) -> None:
        if self.closed:
            raise RuntimeError("backend is closed")
        _validate_chunks(chunks, self._chunks)
        if chunks:
            await self._rpc("add", chunks=[asdict(chunk) for chunk in chunks])
            self._chunks.update((chunk.id, chunk) for chunk in chunks)

    async def search(
        self, query: str, *, top_k: int | None = None, limit: int | None = None
    ) -> list[Evidence]:
        if self.closed:
            raise RuntimeError("backend is closed")
        top_k = _search_limit(top_k, limit)
        if top_k == 0 or not query.strip():
            return []
        return cognee_chunk_evidence(await self._rpc("search", query=query, top_k=top_k))[:top_k]

    async def close(self) -> None:
        if self.closed:
            await self._terminate()
            return
        try:
            if self._process is not None:
                await self._rpc("close")

        finally:
            self.closed = True
            await self._terminate()


def _native_environment(settings: dict) -> None:
    root = Path(settings["root"])
    empty_env = root / "isolated.env"
    empty_env.write_bytes(b"")
    values = {
        "COGNEE_ENV_FILE": str(empty_env),
        "TELEMETRY_DISABLED": "true",
        "COGNEE_TRACING_ENABLED": "false",
        "LOG_LEVEL": "WARNING",
        "SYSTEM_ROOT_DIRECTORY": str(root / "system"),
        "DATA_ROOT_DIRECTORY": str(root / "data"),
        "CACHE_ROOT_DIRECTORY": str(root / "cache"),
        "COGNEE_LOGS_DIR": str(root / "logs"),
        "COGNEE_REPOS_DIR": str(root / "repos"),
        "LLM_PROVIDER": "openai",
        "LLM_ENDPOINT": settings["llm_url"],
        "LLM_MODEL": settings["llm_model"],
        "LLM_API_KEY": settings["llm_api_key"],
        "LLM_MAX_COMPLETION_TOKENS": str(settings["max_completion_tokens"]),
        "LLM_TEMPERATURE": "0",
        "LLM_ARGS": json.dumps(
            {
                "num_retries": 0,
                "timeout": settings.get("native_llm_timeout_seconds", 90),
                "custom_llm_provider": "openai",
            }
        ),
        "STRUCTURED_OUTPUT_FRAMEWORK": "litellm_native",
        "LLM_RATE_LIMIT_ENABLED": "false",
        "EMBEDDING_PROVIDER": "openai_compatible",
        "EMBEDDING_ENDPOINT": settings["embedding_url"],
        "EMBEDDING_MODEL": settings["embedding_model"],
        "EMBEDDING_API_KEY": settings["embedding_api_key"],
        "EMBEDDING_DIMENSIONS": str(settings["embedding_dimensions"]),
        "EMBEDDING_MAX_COMPLETION_TOKENS": str(settings["chunk_size"]),
        "EMBEDDING_BATCH_SIZE": "1",
        "EMBEDDING_MAX_CONCURRENT_DATA_POINTS": "1",
        "EMBEDDING_ENCODING_FORMAT": "float",
        "DB_PROVIDER": "sqlite",
        "VECTOR_DB_PROVIDER": "lancedb",
        "GRAPH_DATABASE_PROVIDER": "ladybug",
        "ENABLE_BACKEND_ACCESS_CONTROL": "true",
        "GRAPH_DATABASE_SUBPROCESS_ENABLED": "false",
        "VECTOR_DB_SUBPROCESS_ENABLED": "false",
        "KUZU_NUM_THREADS": "2",
        "KUZU_BUFFER_POOL_SIZE": "268435456",
        "KUZU_MAX_DB_SIZE": "1073741824",
    }
    # Route all native task roles explicitly; no ambient/commercial stage fallback.
    for stage in ("EXTRACTION", "SUMMARIZATION", "QUERY"):
        for key in ("PROVIDER", "ENDPOINT", "MODEL", "API_KEY"):
            values[f"LLM_{stage}_{key}"] = values[f"LLM_{key}"]
    os.environ.update(values)


def _embedding_kwargs(settings: dict, role: str) -> dict:
    return {
        "embedding_provider": "openai_compatible",
        "embedding_endpoint": settings["embedding_url"],
        "embedding_model": settings["embedding_model"],
        "embedding_api_key": settings["embedding_api_key"],
        "embedding_dimensions": settings["embedding_dimensions"],
        "embedding_input_type": role,
        "embedding_max_completion_tokens": settings["chunk_size"],
        "embedding_batch_size": 1,
        "embedding_max_concurrent_data_points": 1,
        "embedding_encoding_format": "float",
    }


def _refresh_embedding_role(factory: Any) -> None:
    # Cognee 1.6.5 keys its vector-adapter cache only by DB configuration,
    # capturing the previous role's embedding engine. Its own closing-cache API
    # must evict those handles before a different per-call embedding_config.
    # No native retrieval algorithm, prompt, embedding or stored data is replaced.
    factory.cache_clear()


async def _cognee_worker() -> None:
    """Private line RPC, invoked by absolute file path in the isolated interpreter."""
    import importlib.metadata

    # Preserve one protocol stream; native Python and C-level stdout go to log.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    native = None
    settings = None
    for line in sys.stdin:
        request = json.loads(line)
        operation = request["operation"]
        stage = operation if operation in ("init", "add", "search", "close") else "unknown"
        try:
            result: Any = None
            if operation == "init":
                settings = request["settings"]
                if importlib.metadata.version("cognee") != COGNEE_VERSION:
                    raise RuntimeError(f"requires cognee=={COGNEE_VERSION}")
                _native_environment(settings)
                import cognee

                native = cognee
                result = {"version": COGNEE_VERSION}
            elif operation == "close":
                protocol.write(json.dumps({"ok": True, "result": None}) + "\n")
                break
            else:
                assert native is not None and settings is not None
                from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig

                # Pin role per native call, including inside dataset contexts; a static
                # EMBEDDING_INPUT_TYPE cannot distinguish query/document geometry.
                role = "query" if operation == "search" else "document"
                embedding_config = EmbeddingConfig(**_embedding_kwargs(settings, role))
                from cognee.infrastructure.databases.vector.create_vector_engine import (
                    _create_vector_engine,
                )

                _refresh_embedding_role(_create_vector_engine)
                if operation == "add":
                    with (Path(settings["root"]) / "inputs.jsonl").open("a", encoding="utf-8") as f:
                        for chunk in request["chunks"]:
                            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    await native.add(
                        data=[chunk["text"] for chunk in request["chunks"]],
                        dataset_name=settings["dataset"],
                        run_in_background=False,
                        embedding_config=embedding_config,
                    )
                    stage = "cognify"
                    await native.cognify(
                        datasets=[settings["dataset"]],
                        chunk_size=settings["chunk_size"],
                        chunks_per_batch=1,
                        data_per_batch=1,
                        run_in_background=False,
                        raise_on_error=True,
                        extractor="llm",
                        embedding_config=embedding_config,
                    )
                elif operation == "search":
                    result = await native.search(
                        query_text=request["query"],
                        query_type=native.SearchType.CHUNKS,
                        datasets=[settings["dataset"]],
                        top_k=request["top_k"],
                        verbose=True,
                        embedding_config=embedding_config,
                    )
                else:
                    raise ValueError("unknown native worker operation")
            protocol.write(json.dumps({"ok": True, "result": result}, default=str) + "\n")
        except Exception as error:
            protocol.write(
                json.dumps(
                    {
                        "ok": False,
                        "error_type": "native_failure",
                        "diagnostic": _diagnostics().native_diagnostic(error, stage),
                    }
                )
                + "\n"
            )
            break


if __name__ == "__main__" and sys.argv[1:] == ["--cognee-worker"]:
    asyncio.run(_cognee_worker())
