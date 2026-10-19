"""Offline authored privacy probes; assertions never publish sensitive markers."""
import hashlib
import io
import json
import urllib.error

import httpx
import pytest

from gluomem_bench.bench import embedding_runtime as runtime
from gluomem_bench.bench import suite_campaign as suite
from tests.bench.test_request_timeout_isolation import profile
from tests.bench.test_suite_campaign import conflict_adapter


def assert_absent(material, values):
    # Avoid pytest's assertion rewriting printing authored values on a RED run.
    if any(value.encode() in material for value in values):
        pytest.fail("untrusted text survived persisted privacy boundary", pytrace=False)


def test_untrusted_header_body_echo_is_not_diagnostic_text(monkeypatch):
    secret = "authored-header-" + "sensitive-opaque-value"
    text = "authored-query-" + "sensitive-opaque-value"
    body = ("backend echo " + secret + " " + text[:18]).encode()
    error = urllib.error.HTTPError("https://fixture.invalid/v1/embeddings", 400,
                                  text, {"Authorization": secret}, io.BytesIO(body))
    def reject(*args, **kwargs):
        raise error
    monkeypatch.setattr(runtime, "urlopen", reject)
    with pytest.raises(urllib.error.HTTPError) as caught:
        runtime.RemoteEmbedder("https://fixture.invalid/v1").embed_query(text)
    assert caught.value is error
    assert_absent(json.dumps(error.encoder_failure).encode(), [secret, text[:18]])
    assert error.encoder_failure["body_sha256"] == hashlib.sha256(body).hexdigest()
    assert "body" not in error.encoder_failure


@pytest.mark.parametrize("public_wrapper", [False, True])
def test_actual_history_collector_does_not_persist_http_reason(tmp_path, bench_cfg, monkeypatch,
                                                             public_wrapper):
    cfg, adapter = conflict_adapter.__wrapped__(tmp_path, bench_cfg)
    text = "authored-private-" + "input-in-http-reason"
    error = urllib.error.HTTPError("https://fixture.invalid/v1/embeddings", 400,
                                  text, {}, io.BytesIO(text.encode()))
    def reject(*args, **kwargs):
        raise error
    monkeypatch.setattr(runtime, "urlopen", reject)
    def factory(*args):
        runtime.RemoteEmbedder("https://fixture.invalid/v1").embed_query(text)
    out = tmp_path / "failed"
    with pytest.raises(urllib.error.HTTPError) as caught:
        call = suite.run_history if public_wrapper else suite._run_history
        extra = {} if public_wrapper else {"_accounting": {}}
        call(cfg=cfg, adapter=adapter, samples=adapter.load(), profile=profile(),
            out=out, system="dense", judge_source=None, **extra,
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [
                {"id": "offline", "max_model_len": 131072}]})), retriever_factory=factory)
    assert caught.value is error
    assert_absent((out / "failure.json").read_bytes(), [text])


@pytest.mark.parametrize("public_wrapper", [False, True])
def test_receipt_write_failure_preserves_original_exception(tmp_path, bench_cfg, monkeypatch,
                                                          public_wrapper):
    cfg, adapter = conflict_adapter.__wrapped__(tmp_path, bench_cfg)
    error = urllib.error.HTTPError("https://fixture.invalid/v1/embeddings", 400,
                                  "authored untrusted reason", {}, io.BytesIO(b"opaque"))
    cause = ValueError("authored original cause")
    error.__cause__ = cause
    def factory(*args):
        raise error
    write = suite._write
    def fail_receipt(path, value):
        if path.name == "failure.json":
            raise OSError("authored receipt failure")
        return write(path, value)
    monkeypatch.setattr(suite, "_write", fail_receipt)
    call = suite.run_history if public_wrapper else suite._run_history
    extra = {} if public_wrapper else {"_accounting": {}}
    with pytest.raises(urllib.error.HTTPError) as caught:
        call(cfg=cfg, adapter=adapter, samples=adapter.load(), profile=profile(),
             out=tmp_path/"failed", system="dense", judge_source=None, **extra,
             retriever_factory=factory, transport=httpx.MockTransport(lambda r:
                 httpx.Response(200, json={"data": [{"id": "offline", "max_model_len": 131072}]})))
    assert caught.value is error
    assert error.__cause__ is cause


def test_native_encoder_service_failure_receipt_is_categorical(tmp_path):
    import threading
    text = "authored-server-" + "opaque-private-value"
    class Encoder:
        recipe = runtime.EmbeddingRecipe()
        def encode(self, *args, **kwargs):
            raise ValueError(text)
    receipt = tmp_path/"native.jsonl"
    server = runtime.make_server(Encoder(), port=0, receipt=receipt)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(trust_env=False) as client:
            response = client.post(f"http://127.0.0.1:{server.server_port}/v1/embeddings",
                json={"model": Encoder.recipe.model, "input": [text], "input_type": text})
        assert response.status_code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert_absent(receipt.read_bytes(), [text])
    assert_absent(response.content, [text])


@pytest.mark.parametrize("fault", ["read", "digest", "attachment"])
def test_encoder_diagnostic_failure_cannot_replace_original(monkeypatch, fault):
    class BrokenBody(io.BytesIO):
        def read(self, size=-1):
            raise OSError("untrusted diagnostic read failure")
    class AttachmentError(urllib.error.HTTPError):
        def __setattr__(self, name, value):
            if name == "encoder_failure":
                raise OSError("untrusted diagnostic attachment failure")
            super().__setattr__(name, value)
    cls = AttachmentError if fault == "attachment" else urllib.error.HTTPError
    error = cls("https://fixture.invalid/v1/embeddings", 400, "opaque", {},
                BrokenBody(b"opaque") if fault == "read" else io.BytesIO(b"opaque"))
    cause = ValueError("opaque cause")
    error.__cause__ = cause
    def reject(*args, **kwargs):
        raise error
    monkeypatch.setattr(runtime, "urlopen", reject)
    if fault == "digest":
        def broken(*args, **kwargs):
            raise OSError("untrusted digest failure")
        monkeypatch.setattr(runtime.hashlib, "sha256", broken)
    with pytest.raises(urllib.error.HTTPError) as caught:
        runtime.RemoteEmbedder("https://fixture.invalid/v1").embed_query("opaque input")
    assert caught.value is error
    assert caught.value.__cause__ is cause
    if fault == "read":
        assert error.encoder_failure["body_capture"] == "unavailable"


def test_origin_metadata_rejects_untrusted_status_and_role(monkeypatch):
    value = "opaque-authored-" + "untrusted-metadata"
    error = urllib.error.HTTPError("https://fixture.invalid/v1/embeddings", value,
                                  "opaque", {}, io.BytesIO(b"opaque"))
    def reject(*args, **kwargs):
        raise error
    monkeypatch.setattr(runtime, "urlopen", reject)
    with pytest.raises(urllib.error.HTTPError) as caught:
        runtime.RemoteEmbedder("https://fixture.invalid/v1")._encode(["opaque"], value)
    assert caught.value is error
    assert_absent(json.dumps(error.encoder_failure).encode(), [value])
    assert "status" not in error.encoder_failure and "role" not in error.encoder_failure


def test_safe_failure_serializer_does_not_trust_names_strings_or_metadata():
    class Untrusted(Exception):
        def __str__(self):
            pytest.fail("serializer invoked untrusted exception formatting")
        __repr__ = __str__
    Untrusted.__name__ = "opaque untrusted exception class name"
    error = Untrusted()
    error.encoder_failure = dict(status="opaque", role="opaque", body="opaque",
        url="opaque", model="opaque", headers="opaque", input_sha256="opaque",
        body_sha256="opaque", body_bytes_captured=99999, body_truncated="opaque")
    assert runtime.safe_failure_diagnostic(error) == {
        "type": "Exception", "message": "operation_failed"}


def privacy_native_history_scenario(tmp_path, bench_cfg, monkeypatch, outer_form, mode):
    """Actual pinned native encoder seam + whole history + real coordinator.

    No native DB opens or model execution: phase-0 Memory scaffolding is DB-shaped.
    Returns coordinator state so an external acceptance probe can test its owner.
    """
    import ast
    from pathlib import Path
    from threading import Event
    from types import SimpleNamespace
    from gluomem_bench.bench.mem0 import SharedMem0Embedding
    from gluomem_bench.bench.parallel_campaign import CampaignPolicy, Job, run_jobs
    from gluomem_bench.bench.rack_trace import WorkloadBudget
    root = Path('tests/fixtures/native/operator-supplied')
    native = root/"longmemeval-full-continuation-v3/frozen-inputs/f63b856ff86b2b9da3730548116bfbc023560fe941dbfbd34075519b855c46d3"
    material = native.read_bytes()
    assert hashlib.sha256(material).hexdigest() == native.name
    memory_class = next(n for n in ast.parse(material).body
                        if isinstance(n, ast.ClassDef) and n.name == "Memory")
    method = next(n for n in memory_class.body
                  if isinstance(n, ast.FunctionDef) and n.name == "_add_to_vector_store")
    namespace = {"_build_session_scope": lambda filters: "authored-scope",
                 "parse_messages": lambda messages: "\n".join(
                     m["role"] + ": " + m["content"] for m in messages)}
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(native), "exec"), namespace)
    text = "opaque-authored-private-" * (500 if mode == "partial" else 1)
    secret = "opaque-authored-header-" + "private-secret"
    reason = text if mode == "reason" else "Bad Request"
    body = (secret + " prefix " + text).encode()
    error = urllib.error.HTTPError("https://fixture.invalid/v1/embeddings", 400, reason,
                                  {"Authorization": secret}, io.BytesIO(body))
    cause = ValueError("opaque native cause")
    error.__cause__ = cause
    attempts, contexts, accounting = [], {}, {}
    def reject(request, **kwargs):
        attempts.append(request)
        raise error
    monkeypatch.setattr(runtime, "urlopen", reject)
    shared = SharedMem0Embedding.__new__(SharedMem0Embedding)
    shared.encoder = runtime.RemoteEmbedder("https://fixture.invalid/v1")
    memory = SimpleNamespace(db=SimpleNamespace(get_last_messages=lambda *a, **kw: []),
                             embedding_model=shared)
    cfg, adapter = conflict_adapter.__wrapped__(tmp_path, bench_cfg)
    def factory(*args):
        for _ in range(22):
            accounting["budget"].begin_call()
        namespace["_add_to_vector_store"](memory, [{"role": "user", "content": text}],
                                          {}, {"user_id": "authored"}, True)
        pytest.fail("native encoder failure did not stop extraction")
    jobs = [Job("bad", "h", tmp_path/"bad", 100, 100),
            Job("healthy", "other", tmp_path/"healthy", 100, 100)]
    cancelled = Event()
    def run(job, context):
        contexts[job.job_id] = context
        if job.job_id == "healthy":
            context.begin_call()
            return {"healthy": True}
        outer = context if outer_form == "context" else WorkloadBudget(
            policy="role-budgets-v6", outer_budget=context)
        try:
            suite.run_history(cfg=cfg, adapter=adapter, samples=adapter.load(), profile=profile(),
                out=job.output, system="dense", judge_source=None, outer_budget=outer,
                retriever_factory=factory, transport=httpx.MockTransport(lambda r:
                    httpx.Response(200, json={"data": [{"id": "offline", "max_model_len": 131072}]})))
        except urllib.error.HTTPError as caught:
            assert caught is error and caught.__cause__ is cause
            if context.budget.failed:
                cancelled.set()
            raise
    # The public wrapper owns its WorkloadBudget; capture it at construction.
    original_init = WorkloadBudget.__init__
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        accounting["budget"] = self
    monkeypatch.setattr(WorkloadBudget, "__init__", init)
    result = run_jobs(jobs, policy=CampaignPolicy(1, 1, 131072, 200, 120, False),
                      out=tmp_path/"coordination", run_job=run, cancel=cancelled)
    assert [row["status"] for row in result["jobs"]] == ["failed", "complete"]
    assert [row["calls"] for row in result["jobs"]] == [22, 1]
    assert result["accounting"]["calls"] == 23
    assert not contexts["bad"].budget.failed and not cancelled.is_set()
    assert len(attempts) == 1
    failure = json.loads((jobs[0].output/"failure.json").read_bytes())
    assert failure["generation_calls"] == 22 and failure["completed_questions"] == 0
    assert failure["encoder_failure"]["status"] == 400
    assert failure["encoder_failure"]["body_bytes_captured"] == min(len(body), 4096)
    assert failure["encoder_failure"]["body_sha256"] == hashlib.sha256(body[:4096]).hexdigest()
    for path in jobs[0].output.rglob("*"):
        if path.is_file():
            assert_absent(path.read_bytes(), [text[:22], secret])
    assert not (jobs[0].output/"scores.json").exists()
    assert not (jobs[0].output/"manifest.json").exists()
    assert json.loads((tmp_path/"coordination/progress.json").read_bytes()) == result
    assert hashlib.sha256(native.read_bytes()).hexdigest() == native.name
    return result, [text[:22], secret]


@pytest.mark.parametrize("outer_form", ["context", "workload"])
@pytest.mark.parametrize("mode", ["reason", "header", "partial"])
def test_native_history_privacy_at_22_admissions_and_healthy_successor(
        tmp_path, bench_cfg, monkeypatch, outer_form, mode):
    privacy_native_history_scenario(tmp_path, bench_cfg, monkeypatch, outer_form, mode)
