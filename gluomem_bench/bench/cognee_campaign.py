"""Fail-closed bridge for pinned native Cognee; no replacement memory algorithms.

All receipts are external operator-owned files. They contain native prompts and
responses (potentially sensitive), never upstream URL/auth headers. CHUNKS is a
narrow vector-chunk condition, NOT a complete Cognee graph-retrieval comparison.
"""

from __future__ import annotations

import base64
import json
import math
import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from gluomem_bench.bench.comparison_backends import CogneeSettings

CHUNKS_CONDITION = "cognee-1.6.5-native-cognify-vector-chunks-only-v1"
GRAPH_RETRIEVAL_GAP = (
    "Native cognify builds a graph, but SearchType.CHUNKS retrieves vector chunks, "
    "not graph triplets. Cognee 1.6.5 GraphCompletionRetriever exposes "
    "get_retrieved_objects then get_context_from_objects before get_completion_from_context. "
    "The frozen worker exposes only CHUNKS, not those graph-context methods in scoped "
    "dataset/embedding contexts; no graph retrieval or full Cognee quality comparison is admitted."
)


def verify_native_coverage(root, *, dataset, expected, quiescent=False):
    """Read committed SQLite status AND exact persisted UTF-8 input bytes.

    Call before QA with caller-trusted, chat-only Evidence, not observed inputs.
    This proves input/processing coverage, not graph semantic correctness. Native
    exact-text dedup is explicit: input IDs may map to one native data row. For a
    live worker keep quiescent=False to see WAL; only closed stores may use True.
    """
    import hashlib
    import sqlite3
    from collections import Counter
    from dataclasses import asdict
    from urllib.parse import unquote, urlsplit
    from urllib.request import url2pathname
    from uuid import UUID

    root = Path(root).resolve()
    expected = tuple(expected)
    if not expected or len({item.id for item in expected}) != len(expected):
        raise ValueError("nonempty uniquely identified trusted inputs required")
    inputs = [
        json.loads(line) for line in (root / "inputs.jsonl").read_bytes().splitlines() if line
    ]
    if inputs != [asdict(item) for item in expected]:
        raise ValueError("native input receipt differs from trusted ordered input")
    db = root / "system" / "databases" / "cognee_db"
    uri = db.as_uri() + ("?mode=ro&immutable=1" if quiescent else "?mode=ro")
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        datasets = list(conn.execute("SELECT id FROM datasets WHERE name = ?", (dataset,)))
        if len(datasets) != 1:
            raise ValueError("exact native dataset missing or ambiguous")
        dataset_id = datasets[0]["id"]
        native_rows = list(
            conn.execute(
                "SELECT id, raw_data_location, pipeline_status FROM data WHERE dataset_id = ?",
                (dataset_id,),
            )
        )
    wanted = {hashlib.sha256(item.text.encode("utf-8")).hexdigest() for item in expected}
    observed = []
    rows = []
    for row in native_rows:
        status = json.loads(row["pipeline_status"])
        for stage in ("add_pipeline", "cognify_pipeline"):
            scoped = status.get(stage, {})
            completion = [value for key, value in scoped.items() if UUID(key) == UUID(dataset_id)]
            if completion != ["DATA_ITEM_PROCESSING_COMPLETED"]:
                raise ValueError("native input processing incomplete")
        location = urlsplit(row["raw_data_location"])
        if location.scheme != "file" or location.netloc:
            raise ValueError("native raw data must be a scoped local file")
        raw = Path(url2pathname(unquote(location.path))).resolve()
        if not raw.is_relative_to(root):
            raise ValueError("native raw data escaped scoped root")
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        observed.append(digest)
        rows.append({"data_id": row["id"], "text_sha256": digest, "pipeline_status": status})
    if Counter(observed) != Counter(wanted):
        raise ValueError("native persisted input coverage mismatch")
    return {
        "condition": CHUNKS_CONDITION,
        "graph_retrieval_qualified": False,
        "graph_retrieval_gap": GRAPH_RETRIEVAL_GAP,
        "dataset_id": dataset_id,
        "input_count": len(expected),
        "native_data_count": len(rows),
        "processed_input_ids": [item.id for item in expected],
        "native_rows": rows,
        "inputs_sha256": hashlib.sha256((root / "inputs.jsonl").read_bytes()).hexdigest(),
        "quiescent": quiescent,
    }


def verify_generation_evidence(receipt, *, profile, expected_calls, starting_calls=0):
    """Offline exact wire/accounting verification; never infers missing spend as 0.

    ``expected_calls`` is bridge-local spend; ``starting_calls`` is the shared
    budget count captured before bridge use. Failed/partial evidence is rejected,
    never repaired into success. The returned digest binds the retained bytes.
    """
    import hashlib

    material = Path(receipt).read_bytes()
    rows = [json.loads(line) for line in material.splitlines() if line]
    if type(expected_calls) is not int or expected_calls < 1:
        raise ValueError("positive actual generation count required")

    def decoded(row):
        return json.loads(base64.b64decode(row["response_b64"], validate=True))

    models = [row for row in rows if row["kind"] == "/models"]
    if (
        len(models) != 1
        or models[0]["http_status"] != 200
        or profile.model not in {item["id"] for item in decoded(models[0])["data"]}
    ):
        raise ValueError("exact served model evidence missing")
    if rows and all('request_id' in row for row in rows):
        # Concurrent HTTP streams are grouped by immutable request identity,
        # not completion order. The one global discovery row precedes calls.
        by_request = {}
        for row in rows:
            if row['kind'] != '/models':
                by_request.setdefault(row['request_id'], []).append(row)
        streams = sorted(by_request.values(), key=lambda rs: next((r['calls'] for r in rs if r['kind']=='generation'), float('inf')))
        rows = models + [row for stream in streams for row in stream]
    groups, pending = [], []
    for row in rows:
        if row["kind"] == "generation":
            groups.append((pending, row))
            pending = []
        else:
            pending.append(row)
    if pending or len(groups) != expected_calls:
        raise ValueError("incomplete generation/accounting coverage")
    prompt_total = completion_total = 0
    for index, (before, call) in enumerate(groups, 1):
        if call["status"] != "complete" or call["calls"] != starting_calls + index:
            raise ValueError("failed or miscounted native generation")
        native = json.loads(base64.b64decode(call["native_request_b64"], validate=True))
        forwarded = {**native, "chat_template_kwargs": {"enable_thinking": False}}
        if (
            call["forwarded_request"] != forwarded
            or native["model"] != profile.model
            or native["temperature"] != 0
            or native.get("stream", False) is not False
        ):
            raise ValueError("native prompt/schema/model altered")
        kinds = [row["kind"] for row in before]
        expected_kinds = (["/models"] if index == 1 else []) + ["/tokenize"]
        if native.get("response_format") is not None:
            expected_kinds += ["/tokenize"]
        expected_kinds += ["dispatch-intent", "/chat/completions"]
        if kinds != expected_kinds:
            raise ValueError("missing or unexpected wire evidence")
        tokenize = [row for row in before if row["kind"] == "/tokenize"]
        first = tokenize[0]
        if first["http_status"] != 200 or first["request"] != {
            "model": profile.model,
            "messages": native["messages"],
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }:
            raise ValueError("served tokenizer prompt mismatch")
        count = decoded(first)["count"]
        reserve = profile.schema_reserve_tokens
        if len(tokenize) == 2:
            schema_row = tokenize[1]
            if schema_row["http_status"] != 200 or schema_row["request"] != {
                "model": profile.model,
                "prompt": json.dumps(native["response_format"], ensure_ascii=False),
                "add_special_tokens": False,
            }:
                raise ValueError("schema tokenizer mismatch")
            schema_count = decoded(schema_row)["count"]
            if type(schema_count) is not int or schema_count < 0:
                raise ValueError("invalid actual schema tokens")
            reserve = max(reserve, schema_count)
        output = native.get("max_tokens", native.get("max_completion_tokens"))
        if (
            type(count) is not int
            or count < 0
            or type(output) is not int
            or not 0 < output <= profile.output_budget("cognify")
            or count + output + reserve > profile.context_window
            or call["prompt_tokens"] != count
            or call["reserved_tokens"] != count + output + reserve
        ):
            raise ValueError("context admission evidence mismatch")
        intent, wire = before[-2:]
        if (
            intent["calls"] != call["calls"]
            or intent["native_request_b64"] != call["native_request_b64"]
            or intent["forwarded_request"] != forwarded
            or intent["reserved_tokens"] != call["reserved_tokens"]
            or wire["http_status"] != 200
            or wire["request"] != forwarded
            or wire["response_b64"] != call["response_b64"]
        ):
            raise ValueError("native dispatch/response evidence mismatch")
        result = decoded(call)
        choices = result["choices"]
        usage = result["usage"]
        if (
            result["model"] != profile.model
            or len(choices) != 1
            or choices[0]["finish_reason"] != "stop"
            or usage["prompt_tokens"] != count
            or type(usage["completion_tokens"]) is not int
            or not 0 <= usage["completion_tokens"] <= output
        ):
            raise ValueError("incomplete native response")
        if native.get("response_format") is not None:
            from jsonschema import Draft202012Validator

            fmt = native["response_format"]
            schema = (
                fmt["json_schema"]["schema"] if fmt["type"] == "json_schema" else {"type": "object"}
            )
            # Resolution has already been guarded at dispatch; offline verification
            # still supplies no remote resource registry.
            from referencing import Registry

            validator = Draft202012Validator(schema, registry=Registry())
            validator.validate(json.loads(choices[0]["message"]["content"]))
        prompt_total += count
        completion_total += usage["completion_tokens"]
    return {
        "calls": expected_calls,
        "prompt_tokens": prompt_total,
        "completion_tokens": completion_total,
        "receipt_sha256": hashlib.sha256(material).hexdigest(),
    }


def verify_native_evidence(root, *, dataset, expected, bridge, quiescent=False):
    """Required input/processing+generation gate BEFORE any benchmark QA call.

    This is not complete campaign admission: embedding qualification, retrieval
    scope and independent protocol review remain separate gates. On any failure,
    the same generation guard is latched so later reader calls cannot proceed.
    """
    with bridge._lock:
        try:
            bridge.check()
            coverage = verify_native_coverage(
                root, dataset=dataset, expected=expected, quiescent=quiescent
            )
            generation = verify_generation_evidence(
                bridge.receipt,
                profile=bridge.profile,
                expected_calls=bridge.budget.calls - bridge.starting_calls,
                starting_calls=bridge.starting_calls,
            )
            bridge.check()
            return {
                "input_processing_qualified": True,
                "coverage": coverage,
                "generation": generation,
                "condition": CHUNKS_CONDITION,
                "graph_retrieval_qualified": False,
                "complete_campaign_admission": False,
            }
        except BaseException:
            bridge.fail("native evidence verification failed")
            raise


def verify_embedding_role_evidence(rows, *, documents, query, recipe):
    """Verify a caller-frozen service-receipt slice, not active whole-service logs.

    Receipt format is the shared embedding_runtime service's complete/input_sha256/
    role/items/token_counts/recipe contract. Requires SAME single-item batch shape
    and exact original text, including newlines. This proves wire-role identity,
    NOT numeric vector equality or loaded weight identity from supplied labels.
    Bind recipe separately to the service's loaded-weight qualification artifact.
    """
    import hashlib

    if not documents or not query or not recipe:
        raise ValueError("explicit synthetic inputs and qualified recipe required")
    expected = [(text, "document") for text in dict.fromkeys(documents)] + [(query, "query")]
    matches = []
    for text, role in expected:
        digest = hashlib.sha256(json.dumps([text], ensure_ascii=False).encode("utf-8")).hexdigest()
        found = [
            (index, row) for index, row in enumerate(rows) if row.get("input_sha256") == digest
        ]
        if not found:
            raise ValueError("exact native embedding input absent from frozen service receipts")
        for index, row in found:
            counts = row.get("token_counts", [])
            if (
                row.get("complete") is not True
                or row.get("role") != role
                or row.get("items") != 1
                or row.get("recipe") != recipe
                or len(counts) != 1
                or type(counts[0]) is not int
                or not 0 <= counts[0] <= recipe["max_tokens"]
            ):
                raise ValueError("native embedding role/batch/recipe differs")
            matches.append({"receipt_index": index, "input_sha256": digest, "role": role})
    return {
        "wire_roles_qualified": True,
        "vector_parity_qualified": False,
        "matches": matches,
        "recipe": dict(recipe),
        "gap": "Receipts lack numeric vectors; exact same-shape vector parity remains unqualified.",
    }


async def qualify_native_synthetic(settings, *, profile, budget, out):
    """Run ONLY the fixed synthetic native add/cognify/CHUNKS/readback fixture.

    This function makes real native calls when invoked: require explicit operator
    admission and an idle shared server first. No benchmark questions or answers
    are accepted. Root must be exactly fresh ``out / 'native'``. It caps this
    qualification at 16 generations through the caller's SAME outer budget.
    Retain failed directories; never rerun into an existing output directory.
    """
    import hashlib
    from dataclasses import asdict

    from gluomem_bench.bench.comparison_backends import CogneeEvidenceBackend, Evidence
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    out = Path(out).resolve()
    if settings.root.resolve() != out / "native":
        raise ValueError("qualification root must be explicit out/native")
    if out.exists() or out.is_symlink():
        raise FileExistsError("qualification requires a fresh external output")
    budget.check()
    out.mkdir(parents=True)
    local = WorkloadBudget(max_calls=16, max_seconds=900, outer_budget=budget)
    summary = {
        "status": "running",
        "synthetic_only": True,
        "calls": local.calls,
        "condition": CHUNKS_CONDITION,
        "graph_retrieval_qualified": False,
        "complete_campaign_admission": False,
        "graph_retrieval_gap": GRAPH_RETRIEVAL_GAP,
    }

    def save(name, value):
        (out / name).write_bytes(
            (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode(
                "utf-8"
            )
        )

    chunks = [
        Evidence("synthetic-rhea", "Rhea owns a cobalt telescope.", {"fixture": "synthetic-only"})
    ]
    bridge = None
    backend = None
    failed = None
    try:
        with CogneeGenerationBridge(
            profile, budget=local, receipt=out / "generation.jsonl"
        ) as bridge:
            try:
                backend = CogneeEvidenceBackend(bridge.bind(settings))
                await backend.add(chunks)
                before = verify_native_evidence(
                    settings.root, dataset=settings.dataset, expected=chunks, bridge=bridge
                )
                save("before-search-coverage.json", before)
                hits = await backend.search("What does Rhea own?", top_k=3)
                bridge.check()
                save("synthetic-search.json", [asdict(hit) for hit in hits])
                if not hits or not any("cobalt telescope" in hit.text for hit in hits):
                    raise ValueError("synthetic native search did not return persisted fixture")
            finally:
                if backend is not None:
                    await backend.close()
            final = verify_native_evidence(
                settings.root,
                dataset=settings.dataset,
                expected=chunks,
                bridge=bridge,
                quiescent=True,
            )
            save("closed-coverage.json", final)
            summary.update(
                status="synthetic-native-lifecycle-qualified",
                generation=final["generation"],
                embedding_roles_qualified=False,
                vector_parity_qualified=False,
                embedding_gap="Wire-role/vector parity needs separate shared-encoder receipts.",
            )
            summary["native_files"] = {
                str(path.relative_to(settings.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(settings.root.rglob("*"))
                if path.is_file()
            }
            bridge.check()
    except BaseException as exc:
        failed = exc
        local.failed = True
        if bridge is not None:
            bridge.fail("native synthetic qualification failed")
        else:
            budget.failed = True
        summary.update(status="failed", error_type=type(exc).__name__)
    summary["calls"] = local.calls
    try:
        save("qualification.json", summary)
    except BaseException:
        budget.failed = True
        if bridge is not None:
            bridge.fail("qualification persistence failed")
        error = RuntimeError("Cognee qualification receipt persistence failed; incomplete")
        error.receipt = summary
        raise error from None
    if failed is not None:
        error = RuntimeError(
            "native Cognee qualification failed; inspect external qualification.json"
        )
        error.receipt = summary
        raise error from None
    return summary


class CogneeGenerationBridge:
    """Serialize native generation through a shared WorkloadBudget/JobContext.

    ``complete(raw_native_body)`` returns untouched upstream response bytes.
    Construction requires explicit RackProfile, budget and fresh receipt path.
    """

    def __init__(self, profile, *, budget, receipt: Path, transport=None,
                 operation_deadline=None, cleanup_reserve_seconds=0):
        if (operation_deadline is not None and
            (not math.isfinite(operation_deadline) or cleanup_reserve_seconds <= 0
             or not math.isfinite(cleanup_reserve_seconds))):
            raise ValueError("explicit deadline requires a finite positive cleanup reserve")
        self.operation_deadline = operation_deadline
        self.cleanup_reserve_seconds = cleanup_reserve_seconds
        self._closer = None
        self._close_error = None
        self.profile = profile
        self.budget = budget
        self.starting_calls = budget.calls
        self.receipt = Path(receipt)
        self.receipt.parent.mkdir(parents=True, exist_ok=True)
        self.receipt.open("xb").close()
        self._lock = threading.RLock()
        self._identity_lock = threading.Lock()
        self._request_local = threading.local()
        self._request_sequence = 0
        self._original_failure = None
        self._failure = None
        self._ready = False
        self._client = httpx.Client(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            timeout=profile.timeout_seconds,
        )

        self._server = None
        self._key = secrets.token_urlsafe(32)

    def __enter__(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                if bridge.operation_deadline is not None:
                    self.connection.settimeout(bridge._timeout())

            def log_message(self, *args):
                pass  # No request/URL/header logging, including native failures.

            def do_POST(self):
                status = 200
                try:
                    if (
                        self.path != "/v1/chat/completions"
                        or self.headers.get("Authorization") != "Bearer " + bridge._key
                        or self.headers.get("Transfer-Encoding") is not None
                    ):
                        raise ValueError("unsupported proxy request")
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= bridge.profile.max_input_chars * 8 + 1048576:
                        raise ValueError("invalid native body length")
                    self.connection.settimeout(bridge._timeout())
                    result = bridge.complete(self.rfile.read(size))
                except Exception:
                    with bridge._lock:
                        bridge._failure = bridge._failure or "proxy_protocol_failure"
                        bridge.budget.failed = True
                    status, result = 502, b'{"error":{"message":"guard failure; incomplete"}}'
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(result)))
                self.end_headers()
                self.wfile.write(result)

        class Server(ThreadingHTTPServer):
            def handle_error(self, request, client_address):
                if bridge.operation_deadline is None:
                    return super().handle_error(request, client_address)
                bridge.fail("proxy_protocol_failure")  # Never format a cancelled peer's traceback.

        self._server = Server(("127.0.0.1", 0), Handler)
        if self.operation_deadline is not None:
            self._server.daemon_threads = False  # server_close must drain owned handlers.
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.02} if self.operation_deadline is not None else {},
            daemon=True,
        )
        self._thread.start()
        return self

    def bind(self, settings: CogneeSettings) -> CogneeSettings:
        """Return settings with every native LLM role bound by the frozen worker.

        Reuse these settings in CogneeEvidenceBackend; close the backend BEFORE
        leaving the bridge context. The native worker already pins all three
        extraction/summarization/query stages, temperature=0, retries=0.
        """
        if self._server is None or settings.llm_model != self.profile.model:
            raise ValueError("active bridge and exact native model required")
        if settings.max_completion_tokens > self.profile.output_budget("cognify"):
            raise ValueError("native output allowance exceeds admitted extraction budget")
        if settings.operation_deadline is not None:
            if (settings.request_timeout_seconds != self.profile.timeout_seconds
                or settings.cleanup_reserve_seconds <= 0
                or (self.operation_deadline is not None and
                    (self.operation_deadline != settings.operation_deadline or
                     self.cleanup_reserve_seconds != settings.cleanup_reserve_seconds))):
                raise ValueError("native and proxy clocks must match the admitted study")
            self.operation_deadline = settings.operation_deadline
            self.cleanup_reserve_seconds = settings.cleanup_reserve_seconds
            self._server.daemon_threads = False
        return replace(
            settings,
            llm_url=f"http://127.0.0.1:{self._server.server_port}/v1",
            llm_api_key=self._key,
        )

    def __exit__(self, *args):
        if self.operation_deadline is not None:
            try:
                self.close()
            except BaseException:
                if len(args) < 2 or args[1] is None:
                    raise
            return
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join()
            self._server = None
        self._client.close()

    def close(self):
        """Drain within the original absolute clock; a live closer is NOT closed."""
        if self.operation_deadline is None:
            return self.__exit__()
        if self._closer is None:
            def drain():
                try:
                    if self._server is not None:
                        self._server.shutdown()
                        self._server.server_close()
                        self._thread.join()
                    self._client.close()
                    self._server = None
                except BaseException as exc:
                    self._close_error = exc
            self._closer = threading.Thread(target=drain, daemon=True, name="cognee-proxy-close")
            self._closer.start()
        self._closer.join(max(0, self.operation_deadline - time.monotonic()))
        if self._closer.is_alive():
            from gluomem_bench.bench.cognee_native_boundary import native_diagnostic
            error = TimeoutError("Cognee proxy closure unverified at original deadline")
            error.native_diagnostic = native_diagnostic(error, "close", "operation_deadline")
            raise error
        if self._close_error is not None:
            raise self._close_error

    def _timeout(self):
        remaining = self.budget.remaining()
        if self.operation_deadline is not None:
            remaining = min(remaining, self.operation_deadline - self.cleanup_reserve_seconds
                            - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("Cognee bridge work deadline exhausted")
        return min(self.profile.timeout_seconds, remaining)

    def fail(self, reason="native failure"):
        """Latch locally and through nested shared WorkloadBudget/JobContext guards."""
        with self._lock:
            self._failure = self._failure or reason
            budget = self.budget
            seen = set()
            while budget is not None and id(budget) not in seen:
                seen.add(id(budget))
                budget.failed = True
                budget = getattr(budget, "outer_budget", None)

    def check(self):
        with self._lock:
            if self._failure:
                raise RuntimeError("Cognee bridge failure latched; incomplete")
            self.budget.check()
            if (self.operation_deadline is not None and
                time.monotonic() >= self.operation_deadline - self.cleanup_reserve_seconds):
                raise TimeoutError("Cognee bridge work deadline exhausted")

    def _append(self, row):
        import os

        row = dict(row)
        if hasattr(self._request_local, 'identity'):
            row['request_id'] = self._request_local.identity
        with self._lock, self.receipt.open("ab") as stream:
            stream.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())

    def _request(self, method, path, payload=None):
        self.budget.check()
        base = (
            self.profile.base_url.removesuffix("/v1")
            if path == "/tokenize"
            else self.profile.base_url
        )
        response = self._client.request(
            method,
            base + path,
            json=payload,
            timeout=self._timeout(),
        )
        row = {"kind": path, "request": payload, "http_status": response.status_code}
        if response.is_success:
            row["response_b64"] = base64.b64encode(response.content).decode()
        else:
            # Arbitrary error bodies can echo source text, auth or provider data.
            # Retain only a bounded prefix digest, never a reversible encoding.
            import hashlib

            prefix = response.content[:4096]
            row["error_response"] = {
                "body_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                "captured_bytes": len(prefix),
                "truncated": len(response.content) > len(prefix),
            }
        self._append(row)
        response.raise_for_status()
        return response

    @contextmanager
    def _request_scope(self):
        with self._lock:
            self.check()
            self._request_sequence += 1
            self._request_local.identity = self._request_sequence
        try:
            yield
        finally:
            del self._request_local.identity

    def complete(self, native_body: bytes) -> bytes:
        with self._request_scope():
            if self._failure:
                raise RuntimeError("Cognee bridge failure latched; incomplete")
            row = {
                "kind": "generation",
                "native_request_b64": base64.b64encode(native_body).decode(),
            }
            initiating = None
            try:
                self.budget.check()
                body = json.loads(native_body)
                allowed = {
                    "model",
                    "messages",
                    "temperature",
                    "max_tokens",
                    "max_completion_tokens",
                    "response_format",
                    "stream",
                    "n",
                    "top_p",
                    "chat_template_kwargs",
                }
                output = body.get("max_tokens", body.get("max_completion_tokens"))
                if (
                    not isinstance(body, dict)
                    or set(body) - allowed
                    or body.get("model") != self.profile.model
                    or type(body.get("temperature")) not in (int, float)
                    or body.get("temperature") != 0
                    or body.get("stream", False) is not False
                    or type(body.get("n", 1)) is not int
                    or body.get("n", 1) != 1
                    or body.get("top_p", 1) != 1
                    or body.get("chat_template_kwargs", {"enable_thinking": False})
                    != {"enable_thinking": False}
                    or type(output) is not int
                    or not 0 < output <= self.profile.output_budget("cognify")
                    or ("max_tokens" in body and "max_completion_tokens" in body)
                    or not isinstance(body.get("messages"), list)
                    or not body["messages"]
                ):
                    raise ValueError("unsupported native generation request")
                response_format = body.get("response_format")
                validator = None
                if response_format is not None:
                    from jsonschema import Draft202012Validator

                    if response_format == {"type": "json_object"}:
                        schema = {"type": "object"}
                    elif (
                        isinstance(response_format, dict)
                        and response_format.get("type") == "json_schema"
                    ):
                        schema = response_format["json_schema"]["schema"]
                    else:
                        raise ValueError("unsupported native response format")

                    # Never allow JSON Schema reference resolution to contact another service.
                    def local_refs(value):
                        if isinstance(value, dict):
                            for key, item in value.items():
                                if key in {"$ref", "$dynamicRef"} and not str(item).startswith("#"):
                                    raise ValueError("remote schema reference forbidden")
                                local_refs(item)
                        elif isinstance(value, list):
                            for item in value:
                                local_refs(item)

                    local_refs(schema)
                    Draft202012Validator.check_schema(schema)
                    validator = Draft202012Validator(schema)
                with self._identity_lock:
                    if not self._ready:
                        models = self._request("GET", "/models").json()
                        if self.profile.model not in {x["id"] for x in models["data"]}:
                            raise ValueError("exact model unavailable")
                        self._ready = True
                forwarded = {**body, "chat_template_kwargs": {"enable_thinking": False}}
                count = self._request(
                    "POST",
                    "/tokenize",
                    {
                        "model": self.profile.model,
                        "messages": body["messages"],
                        "add_generation_prompt": True,
                        "chat_template_kwargs": forwarded["chat_template_kwargs"],
                    },
                ).json()["count"]
                output = body.get("max_tokens", body.get("max_completion_tokens"))
                if type(count) is not int or count < 0:
                    raise ValueError("actual served tokenizer count required")
                reserve = self.profile.schema_reserve_tokens
                if response_format is not None:
                    schema_count = self._request(
                        "POST",
                        "/tokenize",
                        {
                            "model": self.profile.model,
                            "prompt": json.dumps(response_format, ensure_ascii=False),
                            "add_special_tokens": False,
                        },
                    ).json()["count"]
                    if type(schema_count) is not int or schema_count < 0:
                        raise ValueError("actual schema tokenizer count required")
                    reserve = max(reserve, schema_count)
                tokens = count + output + reserve
                if (
                    tokens > self.profile.context_window
                    or len(json.dumps(body["messages"], ensure_ascii=False))
                    > self.profile.max_input_chars
                ):
                    raise ValueError("native prompt exceeds context; no truncation")
                row.update(prompt_tokens=count, reserved_tokens=tokens, forwarded_request=forwarded)
                with self.budget.lease(tokens):
                    self.budget.check()
                    with self._lock:
                        self.check()
                        self.budget.begin_call()
                        row['calls'] = self.budget.calls
                    self._append(
                        {
                            "kind": "dispatch-intent",
                            "calls": row['calls'],
                            "native_request_b64": row["native_request_b64"],
                            "forwarded_request": forwarded,
                            "reserved_tokens": tokens,
                        }
                    )
                    response = self._request("POST", "/chat/completions", forwarded)
                    row["response_b64"] = base64.b64encode(response.content).decode()
                    result = response.json()
                    choices = result.get("choices", [])
                    usage = result.get("usage", {})
                    if (
                        result.get("model") != self.profile.model
                        or len(choices) != 1
                        or choices[0].get("finish_reason") != "stop"
                        or not isinstance(choices[0].get("message", {}).get("content"), str)
                        or type(usage.get("prompt_tokens")) is not int
                        or usage["prompt_tokens"] != count
                        or type(usage.get("completion_tokens")) is not int
                        or not 0 <= usage["completion_tokens"] <= output
                    ):
                        raise ValueError("native response incomplete or tokenizer/model mismatch")
                    if validator is not None:
                        validator.validate(json.loads(choices[0]["message"]["content"]))
                    self.budget.check()
                    row["status"] = "complete"
                    return response.content
            except Exception as exc:
                initiating = exc
                self._original_failure = self._original_failure or exc
                self.fail('native_generation_failure')
                row.update(status="failed", error_type='native_generation_failure')
                raise RuntimeError("Cognee bridge guard failure; inspect private receipt") from exc
            finally:
                row.setdefault('calls', self.budget.calls)
                try:
                    self._append(row)
                except Exception:
                    self.fail("receipt_failure")
                    if initiating is None:
                        raise RuntimeError(
                            "Cognee receipt persistence failed; incomplete"
                        ) from None
