"""Explicit versioned role budgets; legacy evidence stays reproducible."""

import json

import httpx
import pytest

from gluomem_bench.bench.rack import RackProfile, controlled_config


def test_role_budgets_are_explicit_bounded_and_route_production_tasks(tmp_path):
    profile = RackProfile.model_validate(
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
    cfg = controlled_config(profile, data=tmp_path / "beam.json", sha256="a" * 64)
    assert cfg.dspy.tasks["event_claims"].max_tokens == 4096
    assert cfg.dspy.tasks["consolidate"].max_tokens == 1024
    assert cfg.dspy.tasks["answer"].max_tokens == 2048
    assert profile.output_budget("rubric") == 512
    assert (
        RackProfile(
            base_url="http://localhost:8000/v1", model="example/served-model"
        ).output_budget("event_claims")
        == 512
    )
    with pytest.raises(ValueError):
        RackProfile.model_validate({"generation_budgets": {"extraction": 4096}})
    with pytest.raises(ValueError):
        RackProfile.model_validate({"profile_version": "role-budgets-v2"})
    with pytest.raises(ValueError):
        RackProfile.model_validate(
            {
                "profile_version": "role-budgets-v2",
                "generation_budgets": {
                    "extraction": 999999,
                    "consolidation": 1024,
                    "reader": 2048,
                    "judge": 512,
                },
            }
        )


def test_reader_sends_role_budget_and_reserves_it_before_generation(tmp_path):
    from gluomem_bench.bench.rack import RackClient

    requests = []
    count = 100

    def serve(request):
        requests.append(request)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": count})
        return httpx.Response(
            200,
            json={
                "model": "example/served-model",
                "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":"ok"}'}}],
            },
        )

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
    client = RackClient(p, receipt=tmp_path / "calls.jsonl", transport=httpx.MockTransport(serve))
    try:
        client.complete("answer", {}, {"answer": {"type": "string"}})
        assert json.loads(requests[-1].content)["max_tokens"] == 2048
        count = p.context_window - p.schema_reserve_tokens - 2048 + 1
        requests.clear()
        with pytest.raises(ValueError, match="unsupported context"):
            client.complete("answer", {}, {"answer": {"type": "string"}})
        assert len(requests) == 1
    finally:
        client.close()
