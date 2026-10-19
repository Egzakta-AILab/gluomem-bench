"""Offline authored integration, not MemoryArena answer-quality evidence."""

import json
import os
from pathlib import Path

import pytest


def frozen_capability(arena, tmp_path):
    """Authored receipt fixture, never evidence of real endpoint capability."""
    import hashlib

    model = "authored-offline"
    base = "https://example.invalid/v1"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "reasoning",
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    template = {
        "model": model,
        "messages": [{"role": "user", "content": "Authored sum"}],
        "tools": tools,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    rows = [
        {
            "method": "GET",
            "url": base + "/models",
            "status": 200,
            "response": {"data": [{"id": model, "max_model_len": 131072}]},
        },
        {
            "method": "POST",
            "url": base.removesuffix("/v1") + "/tokenize",
            "status": 200,
            "request": {**template, "add_generation_prompt": True},
            "response": {"count": 100, "max_model_len": 131072},
        },
        {
            "method": "POST",
            "url": base + "/chat/completions",
            "status": 200,
            "request": {
                **template,
                "tool_choice": "auto",
                "temperature": 0,
                "max_tokens": 128,
                "stream": False,
                "n": 1,
            },
            "response": {
                "model": model,
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "r1",
                                    "type": "function",
                                    "function": {
                                        "name": "reasoning",
                                        "arguments": '{"input":"Authored sum"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        },
    ]
    receipt = {
        "accounting": {"calls": 1},
        "jobs": [
            {
                "status": "complete",
                "calls": 1,
                "result": {
                    "generation_attempts": 1,
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "finish_reason": "tool_calls",
                    "tool_names": ["reasoning"],
                    "tool_was_executed": False,
                    "tools_sha256": hashlib.sha256(
                        json.dumps(tools, sort_keys=True).encode()
                    ).hexdigest(),
                },
            }
        ],
    }
    rp, wp = tmp_path / "capability.json", tmp_path / "qualification-wire.jsonl"
    rp.write_bytes(arena._json(receipt))
    wp.write_bytes(b"".join(arena._json(row).replace(b"\n", b" ") + b"\n" for row in rows))
    return arena.FrozenToolCapability(
        rp,
        hashlib.sha256(rp.read_bytes()).hexdigest(),
        wp,
        hashlib.sha256(wp.read_bytes()).hexdigest(),
    )


def test_capability_requires_frozen_bytes_and_checks_actual_wire_not_boolean(tmp_path):
    from gluomem_bench.bench import arena_campaign as arena
    from gluomem_bench.bench.rack import RackProfile

    capability = frozen_capability(arena, tmp_path)
    profile = RackProfile(
        base_url="https://example.invalid/v1", model="authored-offline", max_tokens=128
    )
    evidence = capability.validate(profile)
    assert evidence["generation_attempts"] == 1
    assert evidence["prompt_tokens"] == 100
    for changed in (
        profile.model_copy(update={"model": "different"}),
        profile.model_copy(update={"base_url": "https://elsewhere.invalid/v1"}),
    ):
        with pytest.raises(ValueError, match="qualification"):
            capability.validate(changed)
    capability.wire_path.write_bytes(capability.wire_path.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="SHA"):
        capability.validate(profile)


def native_root():
    root = Path(
        os.environ.get(
            "MEMORYARENA_NATIVE_ROOT",
            str(
                Path(__file__).resolve().parents[2]
                / 'tests/fixtures/native/operator-supplied'
            ),
        )
    )
    if not (root / ".git").exists():
        pytest.skip("pinned native MemoryArena checkout required; no download in tests")
    return root


@pytest.mark.parametrize(
    "mutation", ["usage", "template", "context", "finish", "tool", "arguments", "boolean-only"]
)
def test_refrozen_self_declared_capability_cannot_replace_actual_evidence(tmp_path, mutation):
    import hashlib
    from dataclasses import replace

    from gluomem_bench.bench import arena_campaign as arena
    from gluomem_bench.bench.rack import RackProfile

    cap = frozen_capability(arena, tmp_path)
    wire = [json.loads(line) for line in cap.wire_path.read_bytes().splitlines()]
    if mutation == "usage":
        wire[2]["response"]["usage"]["prompt_tokens"] += 1
    elif mutation == "template":
        wire[1]["request"]["chat_template_kwargs"]["enable_thinking"] = True
    elif mutation == "context":
        wire[0]["response"]["data"][0]["max_model_len"] = 8192
    elif mutation == "finish":
        wire[2]["response"]["choices"][0]["finish_reason"] = "length"
    elif mutation == "tool":
        wire[2]["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "coding"
    elif mutation == "arguments":
        wire[2]["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            '{"input":""}'
        )
    else:
        wire = []
        cap.receipt_path.write_bytes(b'{"qualified":true}')
    cap.wire_path.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in wire))
    cap = replace(
        cap,
        wire_sha256=hashlib.sha256(cap.wire_path.read_bytes()).hexdigest(),
        receipt_sha256=hashlib.sha256(cap.receipt_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="qualification"):
        cap.validate(
            RackProfile(
                base_url="https://example.invalid/v1", model="authored-offline", max_tokens=128
            )
        )


def test_live_transport_requires_frozen_capability_and_bounded_shared_budget(tmp_path):
    import httpx

    from gluomem_bench.bench import arena_campaign as arena

    offline, requests = scoped_client(arena, tmp_path, lambda r: httpx.Response(500))
    cap = frozen_capability(arena, tmp_path)
    try:
        for attestation in (None, True, "qualified", "offline-authored"):
            with pytest.raises(ValueError, match="gate"):
                arena.ArenaTransport(
                    offline.profile,
                    receipt=tmp_path / "never.jsonl",
                    budget=offline.budget,
                    transport=None,
                    tool_capability=attestation,
                )
        live = arena.ArenaTransport(
            offline.profile,
            receipt=tmp_path / "live.jsonl",
            budget=offline.budget,
            transport=None,
            tool_capability=cap,
        )
        try:
            assert live.live is True
            assert live.qualification["wire_sha256"] == cap.wire_sha256
            assert requests == []
        finally:
            live.close()
    finally:
        offline.close()


def test_qualified_live_typed_task_api_runs_native_loop_but_not_benchmark_cases(tmp_path):
    import httpx

    from gluomem_bench.bench import arena_campaign as arena
    from gluomem_bench.bench.trajectory_suite import ArenaSession, ArenaTask

    offline, _ = scoped_client(arena, tmp_path, lambda r: httpx.Response(500))
    capability = frozen_capability(arena, tmp_path)
    seen = []

    def reply(request):
        seen.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"data": [{"id": "authored-offline", "max_model_len": 131072}]}
            )
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100, "max_model_len": 131072})
        number = sum(r.url.path.endswith("/chat/completions") for r in seen)
        message = {"role": "assistant", "content": "yes" if number == 4 else "5"}
        finish = "stop"
        if number == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "r1",
                        "type": "function",
                        "function": {"name": "reasoning", "arguments": '{"input":"Compute 2+3."}'},
                    }
                ],
            }
            finish = "tool_calls"
        return httpx.Response(
            200,
            json={
                "model": "authored-offline",
                "choices": [{"message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    client = arena.ArenaTransport(
        offline.profile,
        receipt=tmp_path / "live/http.jsonl",
        budget=offline.budget,
        transport=httpx.MockTransport(reply),
        tool_capability=capability,
    )
    try:
        forbidden = ArenaTask(
            0, "formal_reasoning_math", (ArenaSession("Unadmitted real question"),), ("Gold",)
        )
        with pytest.raises(ValueError, match="frozen.*job|quality gate"):
            arena.run_math_task(
                forbidden,
                source=arena._sources(native_root()),
                transport=client,
                memory=arena.AuthoredExperienceMemory(client.budget),
                out=tmp_path / "forbidden",
            )
        assert seen == []
        assert not (tmp_path / "forbidden").exists()
        result = arena.run_math_task(
            arena.AUTHORED_LIVE_TASK,
            source=arena._sources(native_root()),
            transport=client,
            memory=arena.AuthoredExperienceMemory(client.budget),
            out=tmp_path / "live",
        )
        assert result["evidence_kind"] == "synthetic-live-native-integration-NOT-benchmark"
        assert result["calls"] == 4
        assert result["tool_capability"] == client.qualification
        assert result["live_transport_kind"] == "injected-test-transport"
        assert result["native_closed"] is True
        assert all(r["status"] == "unrun" for r in result["coverage"].values())
        assert arena.rescore(tmp_path / "live") == result["metrics"]
    finally:
        client.close()
        offline.close()


@pytest.mark.parametrize("drift", ["discovery", "tokenizer"])
def test_live_served_context_drift_aborts_before_generation_and_latches(tmp_path, drift):
    import httpx

    from gluomem_bench.bench import arena_campaign as arena

    offline, _ = scoped_client(arena, tmp_path, lambda r: httpx.Response(500))
    capability = frozen_capability(arena, tmp_path)
    seen = []

    def reply(request):
        seen.append(request.url.path)
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "authored-offline",
                            "max_model_len": 8192 if drift == "discovery" else 131072,
                        }
                    ]
                },
            )
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100, "max_model_len": 8192})
        pytest.fail("generation must not run on context drift")

    client = arena.ArenaTransport(
        offline.profile,
        receipt=tmp_path / "live.jsonl",
        budget=offline.budget,
        transport=httpx.MockTransport(reply),
        tool_capability=capability,
    )
    try:
        with pytest.raises(arena.NativeAbort, match="context"):
            client.create(
                "reasoning", model=client.profile.model, messages=[{"role": "user", "content": "Q"}]
            )
        assert seen == (["/v1/models"] if drift == "discovery" else ["/v1/models", "/tokenize"])
        before = list(seen)
        with pytest.raises(arena.NativeAbort):
            client.create(
                "reasoning", model=client.profile.model, messages=[{"role": "user", "content": "Q"}]
            )
        assert seen == before
        assert client.budget.calls == 0
    finally:
        client.close()
        offline.close()


@pytest.mark.parametrize(
    "violation", ["calls", "seconds", "outer-calls", "outer-seconds", "concurrency"]
)
def test_live_qualification_rejects_expanded_budget_before_client_or_http(tmp_path, violation):
    import httpx

    from gluomem_bench.bench import arena_campaign as arena

    offline, requests = scoped_client(
        arena, tmp_path, lambda r: httpx.Response(500), max_calls=9 if violation == "calls" else 8
    )
    cap = frozen_capability(arena, tmp_path)
    if violation == "seconds":
        from dataclasses import replace

        offline.budget.job = replace(offline.budget.job, seconds_budget=121)
    elif violation == "outer-calls":
        offline.budget.budget.max_calls = 9
    elif violation == "outer-seconds":
        offline.budget.budget.max_seconds = 121
    elif violation == "concurrency":
        offline.budget.limiter.max_in_flight = 2
    try:
        with pytest.raises(ValueError, match="budget gate"):
            arena.ArenaTransport(
                offline.profile,
                receipt=tmp_path / "never.jsonl",
                budget=offline.budget,
                transport=None,
                tool_capability=cap,
            )
        assert requests == []
        assert not (tmp_path / "never.jsonl").exists()
    finally:
        offline.close()


def test_offline_authored_trace_runs_native_actor_environment_and_official_metrics(tmp_path):
    from gluomem_bench.bench import arena_campaign as arena

    result = arena.run_offline_smoke(native_root=native_root(), out=tmp_path / "run")
    assert result["status"] == "complete"
    assert result["evidence_kind"] == "authored-offline-integration-not-quality"
    assert result["native_closed"] is True
    assert result["calls"] == 8  # actor -> reasoning -> final -> judge, twice
    assert result["bounds"] == {
        "actor_max_tokens": 128,
        "reasoning_max_tokens": 128,
        "judge_max_tokens": 128,
        "max_tool_iterations": 1,
        "reasoning_retries": 0,
        "sdk_retries": 0,
        "feedback_in_memory": False,
    }
    assert result["memory_system"] == "authored-experience-fixture-not-gluomem"
    assert result["metrics"]["avg_progress_score"] == 0.5
    assert result["metrics"]["overall_average_passrate"] == 1.0
    assert result["metrics"]["passrate_at_k"] == [0.0, 1.0]
    assert set(result["coverage"]) == set(arena.ARENA_DOMAINS)
    assert all(v["status"] == "unrun" for v in result["coverage"].values())
    steps = [json.loads(x) for x in (tmp_path / "run/steps.jsonl").read_text().splitlines()]
    assert len(steps) == 2
    assert steps[0]["action"]["tool_trace"][0]["tool"] == "reasoning"
    assert steps[0]["observation"]["judge_result"] == "no"
    assert steps[1]["observation"]["judge_result"] == "yes"
    assert "## Judge:" not in steps[0]["experience"]["content"]
    assert steps[0]["experience"]["authorship"] == "agent"
    assert "ground_truth" not in steps[0]["experience"]
    assert arena.rescore(tmp_path / "run")["passrate_at_k"] == [0.0, 1.0]
    wire = [json.loads(x) for x in (tmp_path / "run/http.jsonl").read_text().splitlines()]
    generations = [r["request"] for r in wire if r["url"].endswith("/chat/completions")]
    assert all(p["max_tokens"] == 128 for p in generations)
    assert all(
        "AUTHORED_GOLD" not in json.dumps(p)
        for p in generations
        if "math expert" not in json.dumps(p)
    )
    assert any(any(m["role"] == "tool" for m in p["messages"]) for p in generations)


def scoped_client(arena, tmp_path, handler, max_calls=8):
    import httpx

    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )
    from gluomem_bench.bench.rack import RackProfile

    profile = RackProfile(
        base_url="https://example.invalid/v1", model="authored-offline", max_tokens=128
    )
    outer = CampaignBudget(max_calls=max_calls, max_seconds=120)
    job = Job("test", "test", tmp_path / "run", max_calls, 120)
    outer.reserve(job)
    budget = JobContext(job, outer, RequestLimiter(max_in_flight=1, max_in_flight_tokens=8192))
    requests = []

    def http(request):
        requests.append(
            (request.url.path, json.loads(request.content) if request.content else None)
        )
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": profile.model}]})
        return handler(request)

    client = arena.ArenaTransport(
        profile,
        receipt=tmp_path / "run/http.jsonl",
        budget=budget,
        transport=httpx.MockTransport(http),
        tool_capability="offline-authored",
    )
    return client, requests


def test_coding_name_is_rejected_before_native_actor_no_retry_or_host_execution(tmp_path):
    import httpx

    from gluomem_bench.bench import arena_campaign as arena
    from gluomem_bench.bench.trajectory_suite import ArenaSession, ArenaTask

    def malicious(request):
        return httpx.Response(
            200,
            json={
                "model": "authored-offline",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "bad",
                                    "type": "function",
                                    "function": {
                                        "name": "coding",
                                        "arguments": '{"input":"print(123)"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 3},
            },
        )

    client, requests = scoped_client(arena, tmp_path, malicious)
    task = ArenaTask(0, "formal_reasoning_math", (ArenaSession("Question"),), ("GOLD_ONLY_JUDGE",))
    try:
        with pytest.raises(RuntimeError, match="incomplete"):
            arena.run_math_task(
                task,
                source=arena._sources(native_root()),
                transport=client,
                memory=arena.AuthoredExperienceMemory(client.budget),
                out=tmp_path / "run",
            )
        failure = json.loads((tmp_path / "run/failure.json").read_text())
        assert failure["calls"] == 1
        assert failure["native_closed"] is True
        assert "undeclared" in failure["error"]
        assert len([x for x in requests if x[0] == "/v1/chat/completions"]) == 1
        assert requests[0][0] == "/v1/models"  # identity before any tokenizer/generation
        assert not (tmp_path / "run/result.json").exists()
        assert not (tmp_path / "run/scores").exists()
        assert client.budget.limiter.snapshot() == {"in_flight": 0, "in_flight_tokens": 0}
    finally:
        client.close()


def test_orphan_tool_message_refused_before_tokenizer(tmp_path):
    import httpx

    from gluomem_bench.bench import arena_campaign as arena

    def reply(request):
        return httpx.Response(
            200,
            json={
                "model": "authored-offline",
                "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 1},
            },
        )

    client, requests = scoped_client(arena, tmp_path, reply)
    try:
        with pytest.raises(arena.NativeAbort, match="tool message"):
            client.create(
                "actor",
                model=client.profile.model,
                messages=[
                    {
                        "role": "tool",
                        "tool_call_id": "orphan",
                        "name": "reasoning",
                        "content": "untrusted",
                    }
                ],
            )
        assert requests == []
        assert client.budget.calls == 0
    finally:
        client.close()


@pytest.mark.parametrize(
    "options",
    [
        {"tools": [{"type": "function", "function": {"name": "reasoning", "parameters": {}}}]},
        {"tool_choice": "required"},
    ],
)
def test_unqualified_tool_schema_or_choice_is_not_silently_rewritten(tmp_path, options):
    import httpx

    from gluomem_bench.bench import arena_campaign as arena

    client, requests = scoped_client(arena, tmp_path, lambda r: httpx.Response(500))
    try:
        with pytest.raises(arena.NativeAbort, match="tool"):
            client.create(
                "actor",
                model=client.profile.model,
                messages=[{"role": "user", "content": "Q"}],
                **options,
            )
        assert requests == []
        assert client.budget.calls == 0
    finally:
        client.close()


@pytest.mark.parametrize("completion", [None, True, -1, 129])
def test_actual_completion_usage_must_be_integer_within_declared_cap(tmp_path, completion):
    import httpx

    from gluomem_bench.bench import arena_campaign as arena

    client, requests = scoped_client(
        arena,
        tmp_path,
        lambda r: httpx.Response(
            200,
            json={
                "model": "authored-offline",
                "choices": [{"finish_reason": "stop", "message": {"content": "5"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": completion},
            },
        ),
    )
    try:
        with pytest.raises(arena.NativeAbort, match="usage"):
            client.create(
                "reasoning", model=client.profile.model, messages=[{"role": "user", "content": "Q"}]
            )
        assert client.budget.calls == 1
        assert len(requests) == 2
    finally:
        client.close()


def test_document_experience_preserves_agent_authorship_without_chat_user_relabel():
    from gluomem.write.events import SourceEvent

    from gluomem_bench.bench import arena_campaign as arena

    experience = {
        "content": "## Task: Q\n## solution: authored A\n",
        "authorship": "agent",
        "source_ref": "memoryarena/formal_reasoning_math/0/session/0",
        "session_index": 0,
        "feedback_in_memory": False,
        "native_source_commit": arena.PIN,
    }
    event = arena.document_experience(experience)
    assert isinstance(event, SourceEvent)
    assert event.source_type == "document"
    assert event.messages == []
    assert event.kind is None  # real extractor decides, never explicit-memory bypass
    assert json.loads(event.content) == experience
    assert event.source_ref == experience["source_ref"]
    assert event.event_id == experience["source_ref"]


def test_all_five_domains_are_counted_without_qa_execution():
    from gluomem_bench.bench import arena_campaign as arena

    root = native_root().parent
    inventory = arena.inventory_domains(
        root / "memoryarena-data", json.loads((root / "memoryarena-pins.json").read_text())
    )
    assert set(inventory) == set(arena.ARENA_DOMAINS)
    assert all(
        row["status"] == "unrun" and row["tasks"] > 0 and row["sessions"] >= row["tasks"]
        for row in inventory.values()
    )
    assert inventory["formal_reasoning_math"]["tasks"] == 40


def test_adapted_direct_branch_keeps_native_feedback_policy_and_measured_time(tmp_path):
    import httpx

    from gluomem_bench.bench import arena_campaign as arena
    from gluomem_bench.bench.trajectory_suite import ArenaSession, ArenaTask

    def reply(request):
        return httpx.Response(
            200,
            json={
                "model": "authored-offline",
                "choices": [{"finish_reason": "stop", "message": {"content": "yes"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 1},
            },
        )

    client, requests = scoped_client(arena, tmp_path, reply, max_calls=2)
    task = ArenaTask(
        1,
        "formal_reasoning_math",
        (ArenaSession("Allowed question", "Allowed background"),),
        ("SECRET_GOLD",),
    )
    try:
        result = arena.run_math_task(
            task,
            source=arena._sources(native_root()),
            transport=client,
            memory=arena.AuthoredExperienceMemory(client.budget),
            out=tmp_path / "run",
            mode="adapted-native-direct-reasoning",
            feedback_in_memory=True,
        )
        row = json.loads((tmp_path / "run/steps.jsonl").read_text())
        assert row["time"] > 0
        assert row["action"]["type"] == "tool_result"
        assert "## Judge: CORRECT" in row["experience"]["content"]
        assert "SECRET_GOLD" not in json.dumps(row["experience"])
        assert result["protocol"].startswith("adapted-native-direct-reasoning:")
        assert result["calls"] == 2
        generations = [p for path, p in requests if path == "/v1/chat/completions"]
        assert all("tools" not in p for p in generations)
        assert "SECRET_GOLD" not in json.dumps(generations[0])
        assert "SECRET_GOLD" in json.dumps(generations[1])
    finally:
        client.close()


@pytest.mark.parametrize("cap_fields", [{}, {"generation_budgets": None}])
def test_implicit_default_output_cap_is_not_an_admission_declaration(tmp_path, cap_fields):
    import httpx

    from gluomem_bench.bench import arena_campaign as arena
    from gluomem_bench.bench.rack import RackProfile

    client, _ = scoped_client(arena, tmp_path, lambda r: httpx.Response(500))
    try:
        with pytest.raises(ValueError, match="explicit.*cap"):
            arena.ArenaTransport(
                RackProfile(
                    base_url="https://example.invalid/v1", model="authored-offline", **cap_fields
                ),
                receipt=tmp_path / "never.jsonl",
                budget=client.budget,
                transport=httpx.MockTransport(lambda r: httpx.Response(500)),
                tool_capability="offline-authored",
            )
    finally:
        client.close()


def test_rescore_rejects_changed_http_evidence_without_dataset_or_network(tmp_path, monkeypatch):
    import socket

    from gluomem_bench.bench import arena_campaign as arena

    arena.run_offline_smoke(native_root=native_root(), out=tmp_path / "run")
    monkeypatch.setattr(
        socket.socket, "connect", lambda *a, **k: pytest.fail("rescore used network")
    )
    monkeypatch.setattr(arena, "_sources", lambda *a: pytest.fail("rescore used native checkout"))
    original = arena.rescore(tmp_path / "run")
    assert original["passrate_at_k"] == [0.0, 1.0]
    receipt = tmp_path / "run/http.jsonl"
    receipt.write_bytes(receipt.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="HTTP.*hash"):
        arena.rescore(tmp_path / "run")


@pytest.mark.parametrize("options,role", [({"max_tokens": 129}, "actor"), ({}, "unknown")])
def test_undeclared_role_or_mismatched_output_cap_refused_before_http(tmp_path, options, role):
    import httpx

    from gluomem_bench.bench import arena_campaign as arena

    client, requests = scoped_client(
        arena,
        tmp_path,
        lambda r: httpx.Response(
            200,
            json={
                "model": "authored-offline",
                "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 1},
            },
        ),
    )
    try:
        with pytest.raises(arena.NativeAbort):
            client.create(
                role,
                model=client.profile.model,
                messages=[{"role": "user", "content": "question"}],
                **options,
            )
        assert requests == []
        assert client.budget.calls == 0
    finally:
        client.close()


def test_rescore_binds_native_judge_feedback_to_saved_http_and_step(tmp_path):
    import hashlib

    from gluomem_bench.bench import arena_campaign as arena

    arena.run_offline_smoke(native_root=native_root(), out=tmp_path / "run")
    steps = tmp_path / "run/steps.jsonl"
    rows = [json.loads(line) for line in steps.read_text().splitlines()]
    rows[0]["observation"]["judge_result"] = "yes"
    steps.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
    manifest = tmp_path / "run/result.json"
    saved = json.loads(manifest.read_text())
    saved["steps_sha256"] = hashlib.sha256(steps.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="judge.*feedback"):
        arena.rescore(tmp_path / "run")


@pytest.mark.parametrize(
    "options",
    [
        {"max_tool_iterations": 0},
        {"max_tool_iterations": True},
        {"max_tool_iterations": 10000},
        {"feedback_in_memory": "false"},
        {"mode": "native-default-reproduction"},
    ],
)
def test_invalid_campaign_admission_precedes_outputs_or_http(tmp_path, options):
    import httpx

    from gluomem_bench.bench import arena_campaign as arena
    from gluomem_bench.bench.trajectory_suite import ArenaSession, ArenaTask

    client, requests = scoped_client(
        arena,
        tmp_path,
        lambda r: httpx.Response(
            200,
            json={
                "model": "authored-offline",
                "choices": [{"finish_reason": "stop", "message": {"content": "yes"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 1},
            },
        ),
    )
    task = ArenaTask(0, "formal_reasoning_math", (ArenaSession("Q"),), ("G",))
    try:
        with pytest.raises(ValueError, match="admission"):
            arena.run_math_task(
                task,
                source=arena._sources(native_root()),
                transport=client,
                memory=arena.AuthoredExperienceMemory(client.budget),
                out=tmp_path / "run",
                **options,
            )
        assert not (tmp_path / "run").exists()
        assert requests == []
    finally:
        client.close()


@pytest.mark.parametrize("failure", ["judge-format", "judge-length", "http", "budget"])
def test_failed_native_dispatch_never_scores_or_retries_and_preserves_spend(tmp_path, failure):
    import httpx

    from gluomem_bench.bench import arena_campaign as arena
    from gluomem_bench.bench.trajectory_suite import ArenaSession, ArenaTask

    dispatched = []

    def reply(request):
        dispatched.append(request)
        if failure == "http":
            return httpx.Response(500, text="authored failure")
        judge = len(dispatched) == 2
        content = "maybe yes" if judge and failure == "judge-format" else "yes"
        finish = "length" if judge and failure == "judge-length" else "stop"
        return httpx.Response(
            200,
            json={
                "model": "authored-offline",
                "choices": [{"finish_reason": finish, "message": {"content": content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 1},
            },
        )

    client, _ = scoped_client(arena, tmp_path, reply, max_calls=1 if failure == "budget" else 8)
    task = ArenaTask(0, "formal_reasoning_math", (ArenaSession("Q"),), ("G",))
    try:
        with pytest.raises(RuntimeError, match="incomplete"):
            arena.run_math_task(
                task,
                source=arena._sources(native_root()),
                transport=client,
                memory=arena.AuthoredExperienceMemory(client.budget),
                out=tmp_path / "run",
                mode="adapted-native-direct-reasoning",
            )
        saved = json.loads((tmp_path / "run/failure.json").read_text())
        assert saved["calls"] == len(dispatched)
        assert len(dispatched) == (1 if failure in {"http", "budget"} else 2)
        assert saved["completed_sessions"] == 0
        assert saved["native_closed"] is True
        assert not (tmp_path / "run/result.json").exists()
        assert not (tmp_path / "run/scores").exists()
        before = len(dispatched)
        with pytest.raises(arena.NativeAbort):
            client.create(
                "actor",
                model=client.profile.model,
                messages=[{"role": "user", "content": "no retry"}],
            )
        assert len(dispatched) == before
        assert client.budget.limiter.snapshot() == {"in_flight": 0, "in_flight_tokens": 0}
    finally:
        client.close()
