"""Controlled rack transport contracts; no live network in unit tests."""

import json

import httpx
import pytest


@pytest.mark.parametrize('temperature,max_tokens', [(0, 256), (0.1, 4096)])
def test_v6_native_judge_parameters_are_explicit_and_wire_bound(tmp_path, temperature, max_tokens):
    from gluomem_bench.bench.rack import RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import WorkloadBudget
    budgets = dict(extraction=4096, consolidation=4096, reader=4096, judge=4096)
    with pytest.raises(ValueError):
        RackProfile(base_url='http://localhost:8000/v1', model='offline',
                    profile_version='role-budgets-v5', generation_budgets=budgets)
    profile = RackProfile(base_url='http://localhost:8000/v1', model='offline',
                          profile_version='role-budgets-v6', generation_budgets=budgets)
    sent = []
    def serve(request):
        body = json.loads(request.content)
        sent.append(body)
        if request.url.path == '/tokenize':
            return httpx.Response(200, json={'count': 100})
        return httpx.Response(200, json={'model': 'offline', 'choices': [
            {'finish_reason': 'stop', 'message': {'content': 'yes'}}]})
    client = RackClient(profile, receipt=tmp_path / 'calls.jsonl',
        transport=httpx.MockTransport(serve), budget=WorkloadBudget(policy='role-budgets-v6'))
    try:
        client.generate_messages('rubric', [{'role': 'user', 'content': 'judge'}],
                                 temperature=temperature, max_tokens=max_tokens)
        assert sent[-1]['temperature'] == temperature
        assert sent[-1]['max_tokens'] == max_tokens
        client.generate_messages('answer', [{'role': 'user', 'content': 'read'}])
        assert sent[-1]['temperature'] == 0 and sent[-1]['max_tokens'] == 4096
        rows = [json.loads(line) for line in client.receipt.read_bytes().splitlines()]
        assert rows[1]['request']['temperature'] == temperature
        assert rows[1]['request']['max_tokens'] == max_tokens
    finally:
        client.close()


@pytest.mark.parametrize('version,task,temp,tokens', [
    ('role-budgets-v5', 'rubric', 0.1, 256),
    ('role-budgets-v6', 'answer', 0.1, 256),
    ('role-budgets-v6', 'rubric', 0.2, 256),
    ('role-budgets-v6', 'rubric', 0.1, 4097),
])
def test_native_parameter_override_refusals_precede_any_network(tmp_path, version, task, temp, tokens):
    from gluomem_bench.bench.rack import RackClient, RackProfile, FatalRackError
    profile = RackProfile(base_url='http://localhost:8000/v1', model='offline',
        profile_version=version, generation_budgets=dict(extraction=512, consolidation=512,
        reader=4096, judge=1024 if version.endswith('v5') else 4096))
    sent = []
    client = RackClient(profile, receipt=tmp_path / 'calls.jsonl',
        transport=httpx.MockTransport(lambda request: sent.append(request)))
    try:
        with pytest.raises(FatalRackError, match='native|override'):
            client.generate_messages(task, [{'role': 'user', 'content': 'data'}],
                                     temperature=temp, max_tokens=tokens)
        assert not sent
    finally:
        client.close()


def test_v6_explicit_long_timeout_reaches_transport(tmp_path):
    from gluomem_bench.bench.rack import GenerationBudgets, RackClient, RackProfile

    profile = RackProfile(
        base_url='http://localhost:8000/v1', model='offline',
        profile_version='role-budgets-v6', timeout_seconds=600,
        generation_budgets=GenerationBudgets(extraction=4096, consolidation=2048,
                                            reader=4096, judge=4096),
    )
    timeouts = []

    def serve(request):
        timeouts.append(request.extensions['timeout'])
        return httpx.Response(200, json={'data': [{'id': 'offline'}]})

    client = RackClient(profile, receipt=tmp_path / 'calls.jsonl',
                        transport=httpx.MockTransport(serve))
    try:
        client.preflight()
        assert timeouts[0]['read'] == 600
    finally:
        client.close()
    with pytest.raises(ValueError):
        RackProfile.model_validate({**profile.model_dump(), 'timeout_seconds': 601})


@pytest.mark.parametrize('version', ['legacy-v1', 'role-budgets-v2',
                                    'role-budgets-v4', 'role-budgets-v5'])
def test_legacy_profiles_keep_120_second_request_ceiling(version):
    from gluomem_bench.bench.rack import RackProfile

    fields: dict[str, object] = dict(base_url='http://localhost:8000/v1', model='offline',
                                     profile_version=version, timeout_seconds=121)
    if version != 'legacy-v1':
        fields['generation_budgets'] = dict(extraction=512, consolidation=512,
                                           reader=512, judge=512)
    with pytest.raises(ValueError):
        RackProfile.model_validate(fields)


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


@pytest.mark.parametrize('fault', ['transport', 'identity', 'tokenizer', 'tokenizer_schema', 'models_schema', 'truncation'])
def test_v6_shared_faults_stop_campaign_but_truncation_only_latches_history(tmp_path, fault):
    from gluomem_bench.bench.rack import RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import WorkloadBudget
    from gluomem_bench.bench.parallel_campaign import CampaignBudget, JobContext, Job, RequestLimiter
    campaign = CampaignBudget(max_calls=20, max_seconds=120)
    context = JobContext(Job('j', 'h', tmp_path / 'j', 10, 60), campaign,
        RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))
    budget = WorkloadBudget(policy='role-budgets-v6', outer_budget=context)
    profile = RackProfile(base_url='http://localhost:8000/v1', model='offline',
        profile_version='role-budgets-v6', generation_budgets=dict(
            extraction=4096, consolidation=4096, reader=4096, judge=4096))
    def serve(request):
        if fault == 'transport':
            raise httpx.ConnectError('unavailable')
        if request.url.path == '/tokenize':
            if fault == 'tokenizer_schema':
                return httpx.Response(200, json=[])
            return httpx.Response(200, json={'count': None if fault == 'tokenizer' else 100})
        if request.url.path == '/v1/models':
            return httpx.Response(200, json={})
        return httpx.Response(200, json={'model': 'other' if fault == 'identity' else 'offline',
            'choices': [{'finish_reason': 'length' if fault == 'truncation' else 'stop',
                         'message': {'content': 'answer'}}]})
    client = RackClient(profile, receipt=tmp_path / 'calls.jsonl', budget=budget,
        transport=httpx.MockTransport(serve))
    try:
        with pytest.raises(Exception):
            if fault == 'models_schema':
                client.preflight()
            else:
                client.generate_messages('answer', [{'role': 'user', 'content': 'question'}])
        assert budget.failed is True
        assert campaign.failed is (fault != 'truncation')
        before = client.receipt.read_bytes()
        with pytest.raises(Exception):
            client.generate_messages('answer', [{'role': 'user', 'content': 'again'}])
        assert client.receipt.read_bytes() == before
    finally:
        client.close()


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


def test_dataset_overrides_cannot_escape_controlled_rack_configuration(tmp_path, monkeypatch):
    from gluomem_bench.bench.rack import RackProfile, controlled_config

    profile = RackProfile(base_url="http://localhost:8000/v1", model="example/served-model")
    cfg = controlled_config(profile, data=tmp_path / "source.json", sha256="a" * 64)
    monkeypatch.setenv("GLUOMEM_DSPY__DEFAULT_MODEL", "haiku")
    changed = cfg.with_overrides({"benchmark.answer.context_k": 10})
    assert type(changed) is type(cfg)
    assert changed.dspy.default_model == "rack-qwen"
    assert changed.models == cfg.models
    assert changed.dspy.tasks == cfg.dspy.tasks
    assert changed.benchmark.answer.context_k == 10
    assert changed.benchmark.datasets == cfg.benchmark.datasets
