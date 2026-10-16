"""Guards execute before provider calls and before completion sentinels."""

import json

import pytest


@pytest.mark.parametrize("reason", ["stop", "length", None])
def test_production_lm_captures_raw_and_rejects_incomplete_before_parsing(
    tmp_path, monkeypatch, reason
):
    import dspy
    import httpx

    from gluomem_bench.bench.rack import RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import RackLM, WorkloadBudget

    def provider(lm, *args, **kwargs):
        lm.history.append(
            {
                "response": {
                    "model": "example/served-model",
                    "choices": [{"finish_reason": reason, "message": {"content": "raw"}}],
                    "usage": {"completion_tokens": 512},
                }
            }
        )
        return ["raw"]

    monkeypatch.setattr(dspy.LM, "__call__", provider)
    p = RackProfile(base_url="http://localhost:8000/v1", model="example/served-model")
    client = RackClient(
        p,
        receipt=tmp_path / "http.jsonl",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"count": 20})),
    )
    budget = WorkloadBudget()
    lm = RackLM(
        "openai/" + p.model,
        client=client,
        task="event_claims",
        trace=tmp_path / "raw.jsonl",
        budget=budget,
        api_base=p.base_url,
        max_tokens=512,
        temperature=0,
        num_retries=0,
        cache=False,
    )
    try:
        if reason == "stop":
            assert lm(messages=[{"role": "user", "content": "source"}]) == ["raw"]
        else:
            with pytest.raises(ValueError, match="incomplete"):
                lm(messages=[{"role": "user", "content": "source"}])
            with pytest.raises(RuntimeError, match="workload"):
                budget.check()
            from gluomem_bench.bench.rack import FatalRackError

            with pytest.raises(FatalRackError, match="incomplete"):
                client.check()
        row = json.loads((tmp_path / "raw.jsonl").read_text())
        assert row["response"]["choices"][0]["message"]["content"] == "raw"
        assert row["prompt_token_count"] == 20
        assert row["task"] == "event_claims"
        assert budget.calls == 1
    finally:
        client.close()


@pytest.mark.parametrize("invalid_count", [None, 126000])
def test_production_extraction_uses_its_own_reserve_before_provider(
    tmp_path, monkeypatch, invalid_count
):
    import dspy
    import httpx

    from gluomem_bench.bench.rack import RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import RackLM, WorkloadBudget

    called = []
    monkeypatch.setattr(dspy.LM, "__call__", lambda *a, **k: called.append(True))
    p = RackProfile.model_validate(
        {
            "base_url": "http://localhost:8000/v1",
            "model": "example/served-model",
            "profile_version": "role-budgets-v2",
            "generation_budgets": {
                "extraction": 4096,
                "consolidation": 1024,
                "reader": 2048,
                "judge": 512,
            },
        }
    )
    client = RackClient(
        p,
        receipt=tmp_path / "http.jsonl",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"count": invalid_count})),
    )
    budget = WorkloadBudget()
    lm = RackLM(
        "openai/" + p.model,
        client=client,
        task="event_claims",
        trace=tmp_path / "raw.jsonl",
        budget=budget,
        api_base=p.base_url,
        max_tokens=4096,
        temperature=0,
        num_retries=0,
        cache=False,
    )
    try:
        with pytest.raises(ValueError, match="unsupported context"):
            lm(messages=[{"role": "user", "content": "source"}])
        assert not called and budget.calls == 0
    finally:
        client.close()


@pytest.mark.parametrize(
    "override",
    [
        {"api_base": "https://other.invalid"},
        {"timeout": 9999},
        {"n": 2},
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    ],
)
def test_production_call_cannot_override_model_route_or_bounded_transport(
    tmp_path, monkeypatch, override
):
    import dspy
    import httpx

    from gluomem_bench.bench.rack import RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import RackLM, WorkloadBudget

    called = []
    monkeypatch.setattr(dspy.LM, "__call__", lambda *a, **k: called.append(True))
    p = RackProfile(base_url="http://localhost:8000/v1", model="example/served-model")
    client = RackClient(
        p,
        receipt=tmp_path / "http.jsonl",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"count": 20})),
    )
    lm = RackLM(
        "openai/" + p.model,
        client=client,
        task="event_claims",
        trace=tmp_path / "raw.jsonl",
        budget=WorkloadBudget(),
        api_base=p.base_url,
        max_tokens=512,
        temperature=0,
        num_retries=0,
        cache=False,
    )
    try:
        with pytest.raises(ValueError, match="override"):
            lm(messages=[{"role": "user", "content": "source"}], **override)
        assert not called
    finally:
        client.close()


def test_production_refuses_latched_client_before_tokenizer_or_trace(tmp_path, monkeypatch):
    import dspy
    import httpx

    from gluomem_bench.bench.rack import FatalRackError, RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import RackLM, WorkloadBudget

    requests = []
    provider_calls = []
    monkeypatch.setattr(dspy.LM, "__call__", lambda *a, **k: provider_calls.append(True))
    p = RackProfile(base_url="http://localhost:8000/v1", model="example/served-model")
    client = RackClient(
        p,
        receipt=tmp_path / "http.jsonl",
        transport=httpx.MockTransport(lambda r: requests.append(r) or httpx.Response(200)),
    )
    trace = tmp_path / "raw.jsonl"
    lm = RackLM(
        "openai/" + p.model,
        client=client,
        task="event_claims",
        trace=trace,
        budget=WorkloadBudget(),
        api_base=p.base_url,
        max_tokens=512,
        temperature=0,
        num_retries=0,
        cache=False,
    )
    client._latch(FatalRackError("prior failure"))
    try:
        with pytest.raises(FatalRackError, match="prior failure"):
            lm(messages=[{"role": "user", "content": "source"}])
        assert requests == provider_calls == []
        assert not trace.exists()
    finally:
        client.close()


def test_shared_admission_preserves_distinct_reader_and_production_char_budgets(
    tmp_path, monkeypatch
):
    import dspy
    import httpx

    from gluomem_bench.bench.rack import RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import RackLM, WorkloadBudget

    provider_calls = []
    requests = []
    monkeypatch.setattr(dspy.LM, "__call__", lambda *a, **k: provider_calls.append(True))
    p = RackProfile(
        base_url="http://localhost:8000/v1", model="example/served-model", max_input_chars=2
    )

    def serve(request):
        requests.append(request)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 20})
        return httpx.Response(
            200,
            json={
                "model": p.model,
                "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":"ok"}'}}],
            },
        )

    budget = WorkloadBudget()
    client = RackClient(
        p, receipt=tmp_path / "http.jsonl", budget=budget, transport=httpx.MockTransport(serve)
    )
    trace = tmp_path / "raw.jsonl"
    lm = RackLM(
        "openai/" + p.model,
        client=client,
        task="event_claims",
        trace=trace,
        budget=budget,
        api_base=p.base_url,
        max_tokens=512,
        temperature=0,
        num_retries=0,
        cache=False,
    )
    messages = [{"role": "user", "content": "{}"}]
    try:
        assert client.complete("answer", {}, {"answer": {"type": "string"}}) == {"answer": "ok"}
        assert len(json.loads(requests[0].content)["messages"]) == 2
        with pytest.raises(ValueError, match="input budget"):
            lm(messages=messages)
        assert json.loads(requests[-1].content)["messages"] == messages
        assert [r.url.path for r in requests] == ["/tokenize", "/v1/chat/completions", "/tokenize"]
        assert provider_calls == []
        assert budget.calls == 1 and budget.failed
        assert json.loads(trace.read_text())["prompt_token_count"] == 20
        before = trace.read_bytes()
        with pytest.raises(ValueError):
            lm(messages=messages)
        assert trace.read_bytes() == before
    finally:
        client.close()


def test_whole_history_budget_latches_call_and_wall_failures():
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    clock = [0.0]
    budget = WorkloadBudget(max_calls=2, max_seconds=10, clock=lambda: clock[0])
    budget.begin_call()
    budget.begin_call()
    with pytest.raises(RuntimeError, match="workload"):
        budget.begin_call()
    assert budget.calls == 2
    with pytest.raises(RuntimeError, match="workload"):
        budget.check()
    budget = WorkloadBudget(max_calls=2, max_seconds=10, clock=lambda: clock[0])
    clock[0] = 11
    with pytest.raises(RuntimeError, match="workload"):
        budget.check()


def test_budget_is_shared_by_reader_and_judge_and_exhaustion_prevents_network(tmp_path):
    import httpx

    from gluomem_bench.bench.rack import RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    requests = []

    def serve(request):
        requests.append(request)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 10})
        return httpx.Response(
            200,
            json={
                "model": "example/served-model",
                "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":"ok"}'}}],
            },
        )

    budget = WorkloadBudget(max_calls=1)
    client = RackClient(
        RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
        receipt=tmp_path / "http.jsonl",
        budget=budget,
        transport=httpx.MockTransport(serve),
    )
    try:
        client.complete("answer", {}, {"answer": {"type": "string"}})
        requests.clear()
        with pytest.raises(RuntimeError, match="workload"):
            client.complete("rubric", {}, {"answer": {"type": "string"}})
        assert all(r.url.path != "/v1/chat/completions" for r in requests)
        assert budget.calls == 1
    finally:
        client.close()
