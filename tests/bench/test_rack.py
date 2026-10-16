"""Controlled rack transport contracts; no live network in unit tests."""

import json

import httpx
import pytest


def test_operator_profile_requires_explicit_endpoint_and_model():
    from gluomem_bench.bench.rack import RackProfile

    with pytest.raises(ValueError):
        RackProfile.model_validate({})
    with pytest.raises(ValueError):
        RackProfile.model_validate({"base_url": "http://localhost:8000/v1"})
    with pytest.raises(ValueError):
        RackProfile.model_validate({"model": "example/served-model"})
    profile = RackProfile(base_url="http://localhost:8000/v1", model="example/served-model")
    assert profile.base_url == "http://localhost:8000/v1"
    assert profile.model == "example/served-model"
    with pytest.raises(ValueError):
        profile.model = "other"  # pyrefly: ignore[read-only]


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "https://example.com",
        "ftp://example.com/v1",
        "https://user:secret@example.com/v1",
        "https://example.com/v1?route=other",
        "https://example.com/v1#other",
    ],
)
def test_operator_profile_rejects_ambiguous_or_credentialed_endpoint(endpoint):
    from gluomem_bench.bench.rack import RackProfile

    with pytest.raises(ValueError):
        RackProfile(base_url=endpoint, model="example/served-model")


def test_controlled_config_routes_every_model_task_to_rack_and_ignores_env(tmp_path, monkeypatch):
    from gluomem_bench.bench.rack import RackProfile, controlled_config

    monkeypatch.setenv("GLUOMEM_DSPY__DEFAULT_MODEL", "haiku")
    cfg = controlled_config(
        RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
        data=tmp_path / "beam.json",
        sha256="a" * 64,
    )
    assert set(cfg.models) == {"rack-qwen", "minilm", "gliner_multi"}
    assert not cfg.safety.scanner.classifier.enabled
    assert all(task.model == "rack-qwen" for task in cfg.dspy.tasks.values())
    assert cfg.dspy.default_model == "rack-qwen"
    assert cfg.write.contradiction.model == "rack-qwen"
    assert cfg.write.parallelism == cfg.write.max_concurrent_llm == cfg.write.conv_parallelism == 1
    assert cfg.benchmark.parallelism == 1
    assert cfg.benchmark.judge.backbones == ["rack-qwen"]
    assert cfg.models["minilm"].revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert cfg.models["rack-qwen"].api_key_env == "GLUOMEM_RACK_API_KEY"
    assert cfg.models["rack-qwen"].api_base == "http://localhost:8000/v1"


@pytest.mark.parametrize("count", [131000, None])
def test_rack_context_gate_counts_served_tokenizer_and_refuses_overflow_or_unknown(tmp_path, count):
    from gluomem_bench.bench.rack import RackClient, RackProfile

    requests = []

    def serve(request):
        requests.append(request)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": count})
        return httpx.Response(
            200,
            json={
                "model": "example/served-model",
                "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":"blue"}'}}],
            },
        )

    client = RackClient(
        RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
        receipt=tmp_path / "calls.jsonl",
        transport=httpx.MockTransport(serve),
    )
    try:
        with pytest.raises(ValueError, match="unsupported"):
            client.complete("answer", {"context": "blue"}, {"answer": {"type": "string"}})
        assert len(requests) == 1
        assert requests[0].url.path == "/tokenize"
    finally:
        client.close()


@pytest.mark.parametrize("response_model", [None, "other/served-model"])
def test_generation_requires_exact_selected_response_model(tmp_path, response_model):
    from gluomem_bench.bench.rack import FatalRackError, RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    requests = []
    response = {"choices": [{"finish_reason": "stop", "message": {"content": '{"answer":"ok"}'}}]}
    if response_model is not None:
        response["model"] = response_model

    def serve(request):
        requests.append(request)
        return httpx.Response(
            200, json={"count": 10} if request.url.path == "/tokenize" else response
        )

    budget = WorkloadBudget()
    client = RackClient(
        RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
        receipt=tmp_path / "calls.jsonl",
        transport=httpx.MockTransport(serve),
        budget=budget,
    )
    try:
        with pytest.raises(FatalRackError, match="response model"):
            client.complete("answer", {}, {"answer": {"type": "string"}})
        before = len(requests)
        with pytest.raises(FatalRackError):
            client.preflight()
        assert len(requests) == before
        assert budget.failed
    finally:
        client.close()


def test_rack_preserves_non_json_provider_failure_body(tmp_path):
    from gluomem_bench.bench.rack import RackClient, RackProfile

    client = RackClient(
        RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
        receipt=tmp_path / "calls.jsonl",
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="backend unavailable")),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            client.preflight()
        row = json.loads((tmp_path / "calls.jsonl").read_text())
        assert row["status"] == 500 and row["response_text"] == "backend unavailable"
    finally:
        client.close()


def test_rack_profile_is_exact_and_only_sends_bounded_structured_requests(tmp_path):
    from gluomem_bench.bench.rack import RackClient, RackProfile

    requests = []

    def serve(request):
        requests.append(request)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 20})
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "example/served-model"}]})
        return httpx.Response(
            200,
            json={
                "model": "example/served-model",
                "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":"blue"}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5},
            },
        )

    client = RackClient(
        RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
        receipt=tmp_path / "calls.jsonl",
        transport=httpx.MockTransport(serve),
    )
    client.preflight()
    result = client.complete(
        "answer", {"context": "blue", "question": "color?"}, {"answer": {"type": "string"}}
    )
    assert result == {"answer": "blue"}
    body = json.loads(requests[-1].content)
    assert body["model"] == "example/served-model"
    assert body["temperature"] == 0
    assert body["max_tokens"] <= 512
    assert body["response_format"]["type"] == "json_schema"
    assert all(str(r.url).startswith("http://localhost:8000/") for r in requests)
    receipt = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert receipt[-1]["response"]["usage"]["prompt_tokens"] == 12
    assert receipt[-1]["elapsed_seconds"] >= 0
    with pytest.raises(ValueError, match="rack"):
        RackProfile(base_url="ftp://example.com/v1", model="example/served-model")
