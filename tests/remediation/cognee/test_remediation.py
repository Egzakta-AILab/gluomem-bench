"""Authored offline contracts; no Cognee SDK/backend/service construction."""

import asyncio
import time
from dataclasses import replace
from pathlib import Path

import pytest

from gluomem_bench.bench.comparison_backends import CogneeEvidenceBackend, CogneeSettings


def settings(tmp_path, **kwargs):
    return CogneeSettings(
        python_executable="unused",
        root=tmp_path / "native",
        dataset="authored",
        llm_url="http://example.invalid/v1",
        llm_model="authored",
        llm_api_key="local",
        embedding_url="http://example.invalid/v1",
        embedding_model="authored",
        embedding_dimensions=3,
        embedding_api_key="local",
        **kwargs,
    )


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket

    connect, listen = socket.socket.connect, socket.socket.listen
    allowed = set()

    def registered(sock, *args):
        result = listen(sock, *args)
        address = sock.getsockname()
        if address[0] == "127.0.0.1":
            allowed.add(address)
        return result

    def guarded(sock, address):
        if address not in allowed:
            raise AssertionError("offline fixture: external socket denied")
        return connect(sock, address)

    def denied(*args, **kwargs):
        raise AssertionError("offline fixture: external socket denied")

    monkeypatch.setattr(socket.socket, "listen", registered)
    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)


def test_operation_can_contain_multiple_requests_longer_than_request_clock(tmp_path):
    async def exercise():
        end = time.monotonic() + 1
        backend = CogneeEvidenceBackend(
            settings(
                tmp_path,
                timeout_seconds=0.01,
                request_timeout_seconds=0.01,
                operation_deadline=end,
            )
        )
        requests = []

        async def exchange(operation, payload):
            # Each authored request is fast; their aggregate exceeds request timeout.
            for _ in range(5):
                await asyncio.sleep(0.005)
                requests.append(operation)
            return "done"

        backend._exchange = exchange
        assert await backend._rpc("add") == "done"
        assert len(requests) == 5
        assert backend.settings.operation_deadline == end
        await backend.close()

    asyncio.run(exercise())


def test_shared_deadline_covers_lock_init_timeout_and_cleanup(tmp_path):
    async def exercise():
        end = time.monotonic() + 0.04
        backend = CogneeEvidenceBackend(settings(tmp_path, operation_deadline=end))
        seen = []

        async def exchange(operation, payload):
            seen.append("init")
            await asyncio.sleep(0.2)

        async def terminate():
            seen.append(backend._deadline)

        backend._exchange, backend._terminate = exchange, terminate
        with pytest.raises(TimeoutError) as caught:
            await backend._rpc("add")
        assert seen == ["init", end]
        assert caught.value.native_diagnostic["clock"] == "operation_deadline"
        assert caught.value.native_diagnostic["stage"] == "rpc"
        assert backend.closed
        # A later settings update cannot reset the already admitted clock.
        backend.settings = replace(backend.settings, operation_deadline=end + 100)
        with pytest.raises(TimeoutError):
            await backend._rpc("search")
        assert seen == ["init", end, end]

    asyncio.run(exercise())


def test_legacy_timeout_default_and_explicit_clock_validation(tmp_path):
    legacy = settings(tmp_path)
    assert legacy.request_timeout_seconds is None
    assert legacy.operation_deadline is None
    for field in ("request_timeout_seconds", "operation_deadline", "native_llm_timeout_seconds"):
        for value in (0, -1, float("nan"), float("inf"), True):
            with pytest.raises(ValueError):
                replace(legacy, **{field: value})


def test_retriever_binds_operation_to_existing_budget_not_request_clock(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS

    from gluomem_bench.bench import cognee_retriever as module

    end = time.monotonic() + 40
    original = settings(
        tmp_path, timeout_seconds=1, request_timeout_seconds=1, operation_deadline=end
    )
    factory = NS(settings=original, budget=NS(remaining=lambda: 12))
    retriever = object.__new__(module.CogneeCampaignRetriever)
    retriever.factory = factory
    retriever.backend = NS(settings=original)
    monkeypatch.setattr(module.time, "monotonic", lambda: end - 20)
    retriever._bound_timeout()
    assert retriever.backend.settings.request_timeout_seconds == 1
    assert retriever.backend.settings.operation_deadline == end - 8
    factory.budget.remaining = lambda: 19
    retriever._bound_timeout()
    assert retriever.backend.settings.operation_deadline == end - 8


def test_initiating_diagnostic_crosses_rpc_without_private_content(tmp_path):
    import json
    from types import SimpleNamespace as NS

    from gluomem_bench.bench.cognee_native_boundary import native_diagnostic

    class PrivateGraphError(Exception):
        def __str__(self):
            raise AssertionError("PRIVATE_SQL_BODY")

        def __repr__(self):
            raise AssertionError("PRIVATE_QUERY")

    try:
        try:
            raise OSError(111, "PRIVATE_ENDPOINT")
        except OSError as exc:
            raise PrivateGraphError("PRIVATE_SQL_BODY") from exc
    except PrivateGraphError as original:
        diagnostic = native_diagnostic(original, "cognify")
    assert diagnostic["stage"] == "cognify"
    assert [c["error_class"] for c in diagnostic["causes"]] == ["unknown", "OSError"]
    assert diagnostic["causes"][1]["errno"] == 111
    assert diagnostic["causes"][0]["origin"]["source_sha256"]
    assert "PRIVATE" not in json.dumps(diagnostic)

    async def exercise():
        reader = asyncio.StreamReader()
        reader.feed_data(
            (
                json.dumps({"ok": False, "error_type": "PRIVATE_CLASS", "diagnostic": diagnostic})
                + "\n"
            ).encode()
        )

        class Writer:
            def write(self, data):
                pass

            async def drain(self):
                pass

        backend = CogneeEvidenceBackend(settings(tmp_path))
        backend._process = NS(stdin=Writer(), stdout=reader)
        with pytest.raises(RuntimeError) as caught:
            await backend._send({"operation": "add"})
        assert caught.value.native_diagnostic == diagnostic
        assert "PRIVATE" not in str(caught.value)

    asyncio.run(exercise())


def test_native_diagnostic_uses_bound_preflight_source_not_exception_text(tmp_path):
    import hashlib

    from gluomem_bench.bench.cognee_native_boundary import native_diagnostic

    # Authored ordinary exception proves unknown does not become a preflight by text.
    try:
        raise TimeoutError("test_embedding_connection timeout=30 PRIVATE_BODY")
    except TimeoutError as exc:
        row = native_diagnostic(exc, "add")
    assert row["stage"] == "add" and row["clock"] == "unknown"
    assert row["causes"][0]["error_class"] == "TimeoutError"
    assert (
        row["causes"][0]["origin"]["source_sha256"]
        == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "mode,stage,error_class",
    [("graph", "cognify", "RuntimeError"), ("preflight", "add", "TimeoutError")],
)
def test_authored_graph_worker_preserves_initiating_stage_through_actual_protocol(
    tmp_path, mode, stage, error_class
):
    import json
    import subprocess
    import sys

    from gluomem_bench.bench import cognee_graph

    source = Path(cognee_graph.__file__).parents[2]
    config = __import__("dataclasses").asdict(settings(tmp_path))
    config["root"] = str(tmp_path)
    config["llm_url"] = config["embedding_url"] = "http://127.0.0.1:1/v1"
    wire = [
        {"operation": "init", "settings": config},
        {"operation": "add", "chunks": [{"id": "authored", "text": "fixture"}]},
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-I",
            "-X",
            "utf8",
            str(Path(__file__).with_name("authored_worker_fixture.py")),
            str(source),
            mode,
        ],
        input="".join(json.dumps(r) + "\n" for r in wire),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    diagnostic = replies[1]["diagnostic"]
    assert diagnostic["stage"] == stage
    assert diagnostic["causes"][0]["error_class"] == error_class
    if mode == "graph":
        assert diagnostic["causes"][1]["errno"] == 5
    receipt = json.loads((tmp_path / "native-diagnostics.json").read_bytes())
    assert receipt["events"][-1]["diagnostic"] == diagnostic
    assert b"PRIVATE" not in (result.stdout + result.stderr).encode()
    assert all(b"PRIVATE" not in p.read_bytes() for p in tmp_path.rglob("*") if p.is_file())


def test_bound_preflight_fixture_has_distinct_prerequisite_clock(tmp_path, monkeypatch):
    from gluomem_bench.bench import cognee_native_boundary as module

    digest = "4804470f76196270601602a60e0a9550f89f96bd8b8ce67b2479887cbd461cdc"
    original = module.NativeBoundary.origin

    def authored_preflight_origin(filename, lineno, message=None):
        if filename == "<authored_preflight_fixture>":
            return {"source_sha256": digest, "line": 245}
        return original(filename, lineno, message)

    monkeypatch.setattr(module.NativeBoundary, "origin", staticmethod(authored_preflight_origin))
    namespace = {}
    exec(
        compile(
            "def test_embedding_connection():\n raise TimeoutError('PRIVATE')\n",
            "<authored_preflight_fixture>",
            "exec",
        ),
        namespace,
    )
    try:
        namespace["test_embedding_connection"]()
    except TimeoutError as exc:
        row = module.native_diagnostic(exc, "add")
    assert row["stage"] == "embedding_connection_preflight"
    assert row["clock"] == "native_embedding_preflight"
    assert row["stage_origin"] == {"source_sha256": digest, "line": 245}


def test_unbound_retry_and_sql_warnings_remain_blocking(tmp_path):
    from gluomem_bench.bench.cognee_native_boundary import NativeBoundary

    sink = NativeBoundary(tmp_path / "diagnostic.json")
    origin = {
        "source_sha256": "4679592140a66a2692b2e070ac0e9076c6cc6093effe073bada55497ac1e2f5f",
        "line": 64,
    }
    sink.capture("logging_warning", origin=origin)
    sink.capture("sql_warning", origin={"source_sha256": None, "line": 12})
    with pytest.raises(RuntimeError):
        sink.check()
    assert sink.snapshot()["blocking_diagnostics"] == 2
    assert sink.snapshot()["benign_startup_diagnostics"] == 0


def test_log_record_initiating_exception_is_retained_without_formatting(tmp_path):
    import logging

    from gluomem_bench.bench.cognee_native_boundary import NativeBoundary

    sink = NativeBoundary(tmp_path / "diagnostic.json")
    try:
        raise OSError(9, "PRIVATE_DB_QUERY")
    except OSError:
        import sys

        record = logging.LogRecord(
            "authored", logging.ERROR, __file__, 1, "PRIVATE", (), sys.exc_info()
        )
    sink.capture("logging_error", record)
    diagnostic = sink.snapshot()["events"][0]["diagnostic"]
    assert diagnostic["causes"][0]["error_class"] == "OSError"
    assert diagnostic["causes"][0]["errno"] == 9


@pytest.mark.parametrize("mode", ["healthy", "timeout"])
def test_authored_child_close_has_no_executor_thread_or_live_process(tmp_path, monkeypatch, mode):
    import sys
    import threading

    original = asyncio.create_subprocess_exec
    processes = []

    async def authored_spawn(*args, **kwargs):
        assert kwargs["limit"] == sys.maxsize, "RPC must not add an undeclared payload cap"
        process = await original(
            sys.executable,
            "-B",
            "-I",
            "-X",
            "utf8",
            str(Path(__file__).with_name("echo_worker_fixture.py")),
            mode,
            **kwargs,
        )
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", authored_spawn)

    async def exercise():
        baseline_threads = {t.ident for t in threading.enumerate()}
        end = time.monotonic() + (0.25 if mode == "timeout" else 2)
        backend = CogneeEvidenceBackend(
            settings(
                tmp_path,
                request_timeout_seconds=0.01,
                operation_deadline=end,
                cleanup_reserve_seconds=0.1,
            )
        )
        if mode == "timeout":
            with pytest.raises(TimeoutError):
                await backend._rpc("add")
        else:
            assert await backend._rpc("add") == "authored"
        await backend.close()
        assert backend.worker_closed
        assert processes[0].returncode is not None
        assert backend._log.closed
        assert {t.ident for t in threading.enumerate()} == baseline_threads
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(exercise())


def test_inner_timeout_is_not_relabelled_as_operation_expiration(tmp_path):
    async def exercise():
        backend = CogneeEvidenceBackend(settings(tmp_path, operation_deadline=time.monotonic() + 5))
        original = TimeoutError("PRIVATE_EMBEDDING_TRANSPORT")

        async def exchange(*args):
            raise original

        backend._exchange = exchange
        with pytest.raises(TimeoutError) as caught:
            await backend._rpc("add")
        assert caught.value is original
        assert caught.value.native_diagnostic["clock"] == "unknown"

    asyncio.run(exercise())


def test_retriever_retains_first_safe_diagnostic_and_current_exception_identity(tmp_path):
    from types import SimpleNamespace as NS

    from gluomem_bench.bench import cognee_retriever as module

    original = RuntimeError("PRIVATE_QUERY")
    diagnostic = {
        "protocol": "cognee-safe-failure@v2",
        "stage": "cognify",
        "clock": "unknown",
        "causes": [{"error_class": "RuntimeError"}],
    }
    original.native_diagnostic = diagnostic
    failures = []
    retriever = object.__new__(module.CogneeCampaignRetriever)
    retriever.factory = NS(budget=NS(calls=3), outer_budget=NS())
    retriever.failure = None
    retriever.bridge = NS(fail=failures.append)
    retriever._fail(original)
    retriever._fail(ValueError("PRIVATE_CLEANUP"))
    assert retriever.failure == diagnostic
    assert failures == ["native_failure", "native_failure"]
    assert original.native_generation_calls == 3


def test_native_llm_inner_clock_is_explicit_and_default_unchanged(tmp_path, monkeypatch):
    import json
    import os
    from dataclasses import asdict

    from gluomem_bench.bench.comparison_backends import _native_environment

    monkeypatch.setattr(os, "environ", dict(os.environ))
    config = settings(tmp_path)
    config.root.mkdir()
    _native_environment(asdict(config))
    assert json.loads(os.environ["LLM_ARGS"])["timeout"] == config.native_llm_timeout_seconds == 90
    _native_environment(asdict(replace(config, native_llm_timeout_seconds=2)))
    assert json.loads(os.environ["LLM_ARGS"])["timeout"] == 2


def test_diagnostic_failure_never_replaces_original_or_skips_cleanup(tmp_path, monkeypatch):
    from gluomem_bench.bench import comparison_backends as module

    def broken(*args, **kwargs):
        raise ValueError("PRIVATE_TELEMETRY")

    monkeypatch.setattr(module._diagnostics(), "native_diagnostic", broken)

    async def exercise():
        backend = CogneeEvidenceBackend(settings(tmp_path))
        original = OSError("PRIVATE_INITIATING")
        cleaned = []

        async def exchange(*args):
            raise original

        async def terminate():
            cleaned.append(True)
            raise RuntimeError("PRIVATE_CLEANUP")

        backend._exchange, backend._terminate = exchange, terminate
        with pytest.raises(OSError) as caught:
            await backend._rpc("add")
        assert caught.value is original
        assert cleaned == [True]

    asyncio.run(exercise())


def test_admission_error_keeps_initiating_logged_diagnostic(tmp_path):
    from gluomem_bench.bench.cognee_native_boundary import NativeBoundary

    sink = NativeBoundary(tmp_path / "receipt.json")
    error = OSError(5, "PRIVATE_GRAPH")
    sink.capture("structured_error", error)
    original = sink.snapshot()["events"][0]["diagnostic"]
    with pytest.raises(RuntimeError) as caught:
        sink.check()
    assert caught.value.native_diagnostic == original


def test_failure_ledger_retains_safe_diagnostic_and_honest_worker_close(tmp_path):
    import json
    from types import SimpleNamespace as NS

    from gluomem_bench.bench import cognee_retriever as module

    config = settings(tmp_path, request_timeout_seconds=2, operation_deadline=time.monotonic() + 10)

    async def backend_close():
        pass

    retriever = object.__new__(module.CogneeCampaignRetriever)
    retriever.factory = NS(settings=config, out=tmp_path, budget=NS(calls=2), outer_budget=NS())
    retriever.backend = NS(close=backend_close, closed=True, worker_closed=False, settings=config)
    retriever.bridge = NS(__exit__=lambda *args: None, _server=None)
    retriever.runner = asyncio.Runner()
    retriever.closed = False
    retriever.failed = True
    retriever.failure = {"stage": "cognify", "clock": "unknown"}
    retriever.started = time.monotonic()
    retriever.contexts, retriever.documents, retriever.pins = [], [], None
    retriever.close()
    ledger = json.loads((tmp_path / "native-ledger.json").read_bytes())
    assert ledger["native_diagnostic"] == retriever.failure
    assert ledger["native_worker_closed"] is False
    assert ledger["clocks"]["request_timeout_seconds"] == 2
    assert ledger["clocks"]["native_llm_timeout_seconds"] == 90
    assert ledger["clocks"]["operation_deadline"] == config.operation_deadline


def test_legacy_worker_exception_reply_is_safe_and_typed_without_traceback(tmp_path):
    import json
    import subprocess
    import sys
    from dataclasses import asdict

    from gluomem_bench.bench import comparison_backends as module

    config = asdict(settings(tmp_path))
    config["root"] = str(tmp_path)
    wire = [
        {"operation": "init", "settings": config},
        {"operation": "add", "chunks": [{"id": "authored", "text": "fixture"}]},
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-I",
            "-X",
            "utf8",
            str(Path(__file__).with_name("authored_worker_fixture.py")),
            str(Path(module.__file__).parents[2]),
            "graph",
            "legacy",
        ],
        input="".join(json.dumps(row) + "\n" for row in wire),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    reply = json.loads(result.stdout.splitlines()[1])
    assert reply["diagnostic"]["stage"] == "cognify"
    assert reply["diagnostic"]["causes"][1]["errno"] == 5
    assert "PRIVATE" not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "value",
    [
        {"stage": ["PRIVATE"], "clock": {"PRIVATE": 1}},
        {"stage": "PRIVATE", "clock": "PRIVATE", "causes": 123},
        {"causes": [{"error_class": ["PRIVATE"], "origin": ["PRIVATE"]}]},
        {"stage_origin": {"source_sha256": "PRIVATE", "line": "PRIVATE"}},
    ],
)
def test_rpc_projection_rejects_untyped_private_fields_without_formatting(value):
    import json

    from gluomem_bench.bench.cognee_native_boundary import clean_diagnostic

    assert "PRIVATE" not in json.dumps(clean_diagnostic(value))


def test_retriever_telemetry_failure_cannot_replace_initiating_error(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS

    from gluomem_bench.bench import cognee_retriever as module

    original = OSError("PRIVATE_INITIATING")
    retriever = object.__new__(module.CogneeCampaignRetriever)
    retriever.factory = NS(budget=NS(calls=2), outer_budget=NS())
    retriever.failure = None
    retriever.bridge = None

    def broken(*args):
        raise ValueError("PRIVATE_TELEMETRY")

    monkeypatch.setattr(module, "native_diagnostic", broken)
    retriever._fail(original)
    assert original.native_generation_calls == 2
    assert retriever.failed and retriever.factory.budget.failed
    assert retriever.failure["stage"] == "unknown"


def test_retriever_exposes_safe_failure_on_current_exception():
    from types import SimpleNamespace as NS

    from gluomem_bench.bench.cognee_retriever import CogneeCampaignRetriever

    retriever = object.__new__(CogneeCampaignRetriever)
    retriever.factory = NS(budget=NS(calls=1), outer_budget=NS())
    retriever.failure = None
    retriever.bridge = None
    original = ValueError("PRIVATE_GRAPH_QUERY")
    retriever._fail(original)
    assert original.native_diagnostic == retriever.failure
    assert original.native_diagnostic["causes"][0]["error_class"] == "ValueError"
