"""Authored offline failures; no recovered server body or quality scores."""
import io
import json
import socket
import urllib.error

import httpx
import pytest

from gluomem_bench.bench.rack import RackClient
from gluomem_bench.bench.rack_trace import WorkloadBudget
from gluomem_bench.bench.parallel_campaign import CampaignBudget, Job, JobContext, RequestLimiter
from tests.bench.test_request_timeout_isolation import profile


@pytest.fixture(autouse=True)
def deny_real_connections(monkeypatch):
    def deny(*args, **kwargs):
        pytest.fail("real connections forbidden in encoder-history regression")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("fault", ["oserror", "urllib400", "httpx400"])
def test_external_exception_family_does_not_establish_shared_outage(tmp_path, direct, fault):
    campaign = CampaignBudget(max_calls=100, max_seconds=120)
    context = JobContext(Job("j", "h", tmp_path / "j", 100, 100), campaign,
                         RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))
    budget = context if direct else WorkloadBudget(policy="role-budgets-v6", outer_budget=context)
    request = httpx.Request("POST", "https://fixture.invalid/v1/embeddings")
    faults = {"oserror": OSError("unrelated native failure"),
              "urllib400": urllib.error.HTTPError(str(request.url), 400, "authored", {}, io.BytesIO(b"authored")),
              "httpx400": httpx.HTTPStatusError("authored rejection", request=request,
                                               response=httpx.Response(400, request=request))}
    client = RackClient(profile(), receipt=tmp_path / "calls.jsonl", budget=budget,
                        transport=httpx.MockTransport(lambda _: pytest.fail("no dispatch")))
    try:
        client._latch(faults[fault])
        assert budget.failed
        assert not campaign.failed
        with pytest.raises(Exception):
            client.check()
        assert campaign.calls == 0
        assert not client.receipt.exists()
    finally:
        client.close()


def test_history_catch_does_not_route_unrelated_native_errors_through_rack(
        tmp_path, bench_cfg, monkeypatch):
    from tests.bench.test_suite_campaign import conflict_adapter
    from gluomem_bench.bench.suite_campaign import _run_history
    cfg, adapter = conflict_adapter.__wrapped__(tmp_path, bench_cfg)
    accounting = {}
    relatches = []
    original = RackClient._latch
    def latch(client, exc):
        relatches.append(exc)
        return original(client, exc)
    monkeypatch.setattr(RackClient, "_latch", latch)
    def factory(*args):
        for _ in range(22):
            accounting["budget"].begin_call()
        raise OSError("authored native failure, not rack transport")
    out = tmp_path / "failed-history"
    def serve(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "offline", "max_model_len": 131072}]})
    with pytest.raises(OSError, match="authored native failure"):
        _run_history(cfg=cfg, adapter=adapter, samples=adapter.load(), profile=profile(),
                     out=out, system="dense", judge_source=None,
                     transport=httpx.MockTransport(serve), _accounting=accounting,
                     retriever_factory=factory)
    assert relatches == []
    assert accounting["budget"].failed
    failure = json.loads((out / "failure.json").read_bytes())
    assert failure["generation_calls"] == 22
    assert failure["completed_questions"] == 0
    assert not (out / "scores.json").exists()
    assert not (out / "manifest.json").exists()


def test_encoder_http400_keeps_bounded_safe_origin_metadata(monkeypatch):
    import hashlib
    from gluomem_bench.bench import embedding_runtime as runtime
    raw_input = "private authored query"
    attempts = []
    class BoundedBody(io.BytesIO):
        def read(self, n=-1):
            assert 0 < n <= 4097
            return super().read(n)
    body = BoundedBody(("authored rejection: " + raw_input + " " + "x" * 6000).encode())
    def reject(request, **kwargs):
        attempts.append(request)
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, body)
    monkeypatch.setattr(runtime, "urlopen", reject)
    encoder = runtime.RemoteEmbedder("https://user:password@fixture.invalid/v1?token=secret")
    with pytest.raises(urllib.error.HTTPError) as caught:
        encoder.embed_query(raw_input)
    meta = caught.value.encoder_failure
    assert meta["status"] == 400
    assert meta["role"] == "query"
    assert "model" not in meta and "url" not in meta
    assert meta["input_sha256"] == hashlib.sha256(json.dumps([raw_input], ensure_ascii=False).encode()).hexdigest()
    assert meta["body_truncated"] is True
    assert meta["body_bytes_captured"] == 4096
    assert "body" not in meta
    assert meta["body_sha256"] == hashlib.sha256(
        ("authored rejection: " + raw_input + " " + "x" * 6000).encode()[:4096]).hexdigest()
    assert all(secret not in json.dumps(meta) for secret in (raw_input, "password", "token=secret"))
    assert len(attempts) == 1


def test_bounded_encoder_body_does_not_publish_partial_echoed_input(monkeypatch):
    from gluomem_bench.bench import embedding_runtime as runtime
    text = "AUTHORED-PRIVATE-INPUT-" * 500
    def reject(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {},
                                     io.BytesIO(("rejected: " + text).encode()))
    monkeypatch.setattr(runtime, "urlopen", reject)
    with pytest.raises(urllib.error.HTTPError) as caught:
        runtime.RemoteEmbedder("https://encoder.invalid/v1").embed_query(text)
    assert "body" not in caught.value.encoder_failure
    assert caught.value.encoder_failure["body_bytes_captured"] == 4096
    assert "AUTHORED-PRIVATE-INPUT-" not in json.dumps(caught.value.encoder_failure)


def test_encoder_diagnostics_do_not_copy_headers_or_url_secret_values(monkeypatch):
    from gluomem_bench.bench import embedding_runtime as runtime
    headers = {"Authorization": "Bearer AUTHORED-HEADER-SECRET"}
    def reject(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", headers,
                                     io.BytesIO(b"authored error: AUTHORED-URL-SECRET"))
    monkeypatch.setattr(runtime, "urlopen", reject)
    with pytest.raises(urllib.error.HTTPError) as caught:
        runtime.RemoteEmbedder("https://encoder.invalid/v1?api_key=AUTHORED-URL-SECRET").embed("text")
    saved = json.dumps(caught.value.encoder_failure)
    assert "AUTHORED-HEADER-SECRET" not in saved
    assert "AUTHORED-URL-SECRET" not in saved
    assert "headers" not in saved


@pytest.mark.parametrize("outer_form", ["context", "workload"])
def test_failed_history_collector_retains_encoder_origin_and_healthy_successor(
        tmp_path, bench_cfg, monkeypatch, outer_form):
    from threading import Event
    from tests.bench.test_suite_campaign import conflict_adapter
    from gluomem_bench.bench import embedding_runtime as runtime
    from gluomem_bench.bench.suite_campaign import _run_history
    from gluomem_bench.bench.parallel_campaign import CampaignPolicy, run_jobs
    cfg, adapter = conflict_adapter.__wrapped__(tmp_path, bench_cfg)
    cancelled = Event()
    contexts, accounting, encoder_attempts, dispatched = {}, {}, [], []
    error = urllib.error.HTTPError("https://encoder.invalid/v1/embeddings", 400,
                                   "Bad Request", {}, io.BytesIO(b"authored rejection; not original body"))
    def reject(request, **kwargs):
        encoder_attempts.append(request)
        raise error
    monkeypatch.setattr(runtime, "urlopen", reject)
    encoder = runtime.RemoteEmbedder("https://encoder.invalid/v1")
    def factory(*args):
        for _ in range(22):
            accounting["budget"].begin_call()
        encoder.embed_query("authored private text")
        pytest.fail("encoder must fail before extraction")
    jobs = [Job("bad", "h", tmp_path / "bad", 100, 100),
            Job("healthy", "other", tmp_path / "healthy", 100, 100)]
    def serve(request):
        dispatched.append(request.url.path)
        assert request.url.path == "/v1/models", "no generation response or QA zero is authored"
        return httpx.Response(200, json={"data": [{"id": "offline", "max_model_len": 131072}]})
    def run(job, context):
        contexts[job.job_id] = context
        if job.job_id == "healthy":
            context.begin_call()
            return {"healthy": True}
        outer = context if outer_form == "context" else WorkloadBudget(
            policy="role-budgets-v6", outer_budget=context)
        try:
            _run_history(cfg=cfg, adapter=adapter, samples=adapter.load(), profile=profile(),
                         out=job.output, system="dense", judge_source=None,
                         transport=httpx.MockTransport(serve), _accounting=accounting,
                         outer_budget=outer, retriever_factory=factory)
        except Exception:
            if context.budget.failed:
                cancelled.set()
            raise
    result = run_jobs(jobs, policy=CampaignPolicy(1, 1, 131072, 200, 120, False),
                      out=tmp_path / "coordination", run_job=run, cancel=cancelled)
    assert [r["status"] for r in result["jobs"]] == ["failed", "complete"]
    assert result["accounting"]["calls"] == 23
    assert [r["calls"] for r in result["jobs"]] == [22, 1]
    assert accounting["budget"].failed
    assert not contexts["bad"].budget.failed
    assert not cancelled.is_set()
    assert len(encoder_attempts) == 1
    assert dispatched == ["/v1/models"]
    failure = json.loads((jobs[0].output / "failure.json").read_bytes())
    assert failure["encoder_failure"] == error.encoder_failure
    assert "body" not in failure["encoder_failure"]
    assert len(failure["encoder_failure"]["body_sha256"]) == 64
    assert failure["error_type"] == "HTTPError"
    assert failure["generation_calls"] == 22 and failure["completed_questions"] == 0
    assert "authored private text" not in json.dumps(failure)
    assert not (jobs[0].output / "scores.json").exists()
    assert not (jobs[0].output / "manifest.json").exists()
    assert json.loads((tmp_path / "coordination/progress.json").read_bytes()) == result


def test_history_setup_context_identity_failure_is_shared_at_origin(tmp_path, bench_cfg):
    from tests.bench.test_suite_campaign import conflict_adapter
    from gluomem_bench.bench.suite_campaign import _run_history
    from gluomem_bench.bench.rack import FatalRackError
    cfg, adapter = conflict_adapter.__wrapped__(tmp_path, bench_cfg)
    campaign = CampaignBudget(max_calls=100, max_seconds=120)
    context = JobContext(Job("j", "h", tmp_path / "j", 100, 100), campaign,
                         RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))
    accounting = {}
    with pytest.raises(FatalRackError, match="served context limit"):
        _run_history(cfg=cfg, adapter=adapter, samples=adapter.load(), profile=profile(),
                     out=tmp_path / "bad-identity", system="no_memory", judge_source=None,
                     transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
                         "data": [{"id": "offline", "max_model_len": 1}]})),
                     _accounting=accounting, outer_budget=context)
    assert campaign.failed
    assert accounting["budget"].failed
    assert campaign.calls == 0


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("status", [400, 422])
def test_generation_rejection_is_request_local_and_keeps_attempt(tmp_path, direct, status):
    campaign = CampaignBudget(max_calls=100, max_seconds=120)
    context = JobContext(Job("j", "h", tmp_path / "j", 100, 100), campaign,
                         RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))
    budget = context if direct else WorkloadBudget(policy="role-budgets-v6", outer_budget=context)
    attempts = []
    def serve(request):
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 20})
        attempts.append(request)
        return httpx.Response(status, text="authored request rejection")
    client = RackClient(profile(), receipt=tmp_path / "calls.jsonl", budget=budget,
                        transport=httpx.MockTransport(serve))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            client.generate_messages("answer", [{"role": "user", "content": "authored"}])
        assert budget.failed and not campaign.failed
        assert campaign.calls == 1
        with pytest.raises(Exception):
            client.generate_messages("answer", [{"role": "user", "content": "no retry"}])
        assert len(attempts) == 1
        rows = [json.loads(line) for line in client.receipt.read_text().splitlines()]
        assert rows[-1]["status"] == status
        assert rows[-1]["response_text"] == "authored request rejection"
        assert "error" in rows[-1]
    finally:
        client.close()
