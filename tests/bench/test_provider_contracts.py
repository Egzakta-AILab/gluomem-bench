"""Critical transport/currency/privacy contracts with authored in-memory HTTP.

No captured responses, sockets, native services, credentials, or inference.
"""

import copy
import json
from contextlib import closing
from decimal import Decimal

import httpx
import pytest

from gluomem_bench.bench.rack import RackClient, RackProfile
from gluomem_bench.bench.rack_trace import WorkloadBudget

DEEPSEEK = "deepseek/deepseek-v4.1-flash-1017"


def deepseek_profile(base, **changes):
    fields = dict(
        profile_version="provider-roles-v8",
        provider="openrouter-baidu-fp8",
        requested_model=DEEPSEEK,
        model=DEEPSEEK,
        base_url=base + "/api/v1",
        context_window=1048576,
        token_count_policy="utf8-conservative-v1",
        api_key_env="OPENROUTER_API_KEY",
        generation_budgets=dict(extraction=4096, consolidation=2048, reader=1024, judge=1024),
    )
    return RackProfile.model_validate({**fields, **changes})


def completion(model=DEEPSEEK, provider="Baidu", content="yes", **changes):
    return {
        "id": "authored-completion",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "provider": provider,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 23, "completion_tokens": 3, "total_tokens": 26},
        **changes,
    }


def qwen_profile(base, **changes):
    fields = {
        **deepseek_profile(base).model_dump(),
        "provider": "qwen-vllm",
        "requested_model": "qwen-3.8-27b",
        "model": "Qwen/Qwen3.8-27B-FP8",
        "base_url": base + "/v1",
        "context_window": 131072,
        "api_key_env": "GLUOMEM_RACK_API_KEY",
        "token_count_policy": "served-tokenizer",
    }
    return RackProfile.model_validate({**fields, **changes})


POOL_VERSION = "provider-roles-v10"
ENDPOINTS = [
    {
        "tag": tag + "/fp8",
        "provider_name": name,
        "quantization": "fp8",
        "pricing": {"prompt": "0.0000001", "completion": "0.0000004", "request": "0"},
    }
    for tag, name in [
        ("nebius", "Nebius"),
        ("parasail", "Parasail"),
        ("crusoe", "Crusoe"),
        ("atlas-cloud", "AtlasCloud"),
        ("morph", "Morph"),
        ("deepinfra", "DeepInfra"),
    ]
]


def pool_profile(base, **changes):
    from gluomem_bench.bench.rack import RackProfile

    fields = deepseek_profile(base).model_dump()
    fields.update(
        profile_version=POOL_VERSION,
        provider="openrouter",
        openrouter_pool={
            "protocol": "client-owned-fp8-pool-v1",
            "endpoints": copy.deepcopy(ENDPOINTS),
            "max_attempts_per_logical_call": 6,
            "logical_deadline_seconds": 300,
            "cooldown_seconds": 1,
            "max_backoff_seconds": 30,
            "jitter_fraction": 0.1,
        },
    )
    return RackProfile.model_validate({**fields, **changes})


def test_direct_serialization_and_shared_budget_latch(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "authored-placeholder-not-a-secret")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completion(), request=request)

    profile = deepseek_profile("http://fixture.invalid")
    budget = WorkloadBudget(policy=profile.profile_version)
    with closing(
        RackClient(
            profile,
            receipt=tmp_path / "wire.jsonl",
            budget=budget,
            transport=httpx.MockTransport(handler),
        )
    ) as client:
        result = client.generate_messages(
            "rubric", [{"role": "user", "content": "Authored rubric"}]
        )
        assert result["text"] == "yes" and budget.calls == 1
        body = json.loads(requests[0].content)
        assert body["provider"] == {
            "only": ["baidu/fp8"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "quantizations": ["fp8"],
        }
        assert body["reasoning"] == {"enabled": False}
        assert "chat_template_kwargs" not in body
        row = json.loads(client.receipt.read_text())
        assert row["provider_evidence"]["returned_provider"] == "Baidu"
        assert row["token_accounting"]["actual_prompt_tokens"] == 23


def test_provider_identity_mismatch_never_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "authored-placeholder-not-a-secret")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completion(provider="WrongProvider"), request=request)

    profile = deepseek_profile("http://fixture.invalid")
    budget = WorkloadBudget(policy=profile.profile_version)
    client = RackClient(
        profile,
        receipt=tmp_path / "wire.jsonl",
        budget=budget,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ValueError):
            client.generate_messages("answer", [{"role": "user", "content": "Authored request"}])
        with pytest.raises(ValueError):
            client.generate_messages("answer", [{"role": "user", "content": "Authored request"}])
        assert len(requests) == budget.calls == 1 and budget.failed
    finally:
        client.close()


def test_pool_physical_attempts_preserve_model_and_exact_declared_routes(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "authored-placeholder-not-a-secret")
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(503, json={"error": {"code": 503}}, request=request)
        return httpx.Response(200, json=completion(provider="Parasail"), request=request)

    monkeypatch.setattr(httpx, "HTTPTransport", lambda **kw: httpx.MockTransport(handler))
    profile = pool_profile("http://fixture.invalid")
    budget = WorkloadBudget(policy=POOL_VERSION)
    client = RackClient(
        profile,
        receipt=tmp_path / "pool.jsonl",
        budget=budget,
        transport=httpx.MockTransport(handler),
    )
    try:
        result = client.generate_messages(
            "answer", [{"role": "user", "content": "Authored request"}]
        )
        assert result["text"] == "yes"
        assert len(requests) == budget.calls == 2 and not budget.failed
        assert [body["provider"]["only"] for body in requests] == [["nebius/fp8"], ["parasail/fp8"]]
        assert all(
            body["model"] == DEEPSEEK and not body["provider"]["allow_fallbacks"]
            for body in requests
        )
        rows = [json.loads(line) for line in client.receipt.read_text().splitlines()]
        assert [row["pool_attempt"]["outcome"] for row in rows] == [
            "infrastructure-retry",
            "success",
        ]
    finally:
        client.close()


@pytest.mark.parametrize("system", ["mem0", "cognee"])
def test_pool_publication_cannot_open_unqualified_native_cells(system):
    from gluomem_bench.bench.provider_study_evidence import validate_history_scope

    with pytest.raises(ValueError, match="unqualified|unsupported"):
        validate_history_scope(pool_profile("http://fixture.invalid"), system)


def test_currency_is_append_only_no_refund_and_billed_cost_stays_unknown(tmp_path):
    from gluomem_bench.bench.provider_study import CurrencyReservations

    currency = CurrencyReservations(
        max_usd="0.003", max_rate_per_million="1", receipt=tmp_path / "currency.jsonl"
    )
    currency.reserve(1000, "authored-job")
    currency.reserve(1000, "authored-job")
    with pytest.raises(RuntimeError, match="exhausted"):
        currency.reserve(2000, "authored-job")
    assert currency.reserved == Decimal("0.002") and currency.failed
    rows = [json.loads(line) for line in currency.receipt.read_text().splitlines()]
    assert [row["reservation_id"] for row in rows] == [1, 2]
    assert currency.snapshot()["billed_usd"] is None


@pytest.mark.parametrize("value", [True, 1.0, -1, "NaN", "1_0", "+0", "-0", "Infinity"])
def test_price_classification_never_coerces_unbounded_or_nondecimal_values(value):
    from gluomem_bench.bench.rack_pricing import decimal_price

    with pytest.raises(ValueError):
        decimal_price(value)


def test_price_tiers_and_cache_write_require_explicit_version():
    from gluomem_bench.bench.rack_pricing import (
        ProviderPricingPolicy,
        validate_prices,
        wire_price_limits,
    )

    fields = dict(
        version="bounded-provider-prices-v2",
        max_prompt_usd_per_token="0.000001",
        max_completion_usd_per_token="0.000005",
        max_request_usd="0",
        prompt_token_cap=65536,
        output_token_cap=16384,
    )
    policy = ProviderPricingPolicy(**fields)
    price = {
        "prompt": "0.0000001",
        "completion": "0.0000004",
        "request": "0",
        "input_cache_write": "0.000002",
    }
    with pytest.raises(ValueError, match="ceiling"):
        validate_prices(policy, price)
    legacy = ProviderPricingPolicy(**{**fields, "version": "bounded-provider-prices-v1"})
    assert validate_prices(legacy, price)["prompt"] == "0.0000001"
    price["input_cache_write"] = "0.000001"
    assert validate_prices(policy, price)["prompt"] == "0.000001"
    assert wire_price_limits(policy) == {"prompt": 1, "completion": 5, "request": 0}
    with pytest.raises(ValueError, match="unclassified"):
        validate_prices(policy, {**price, "unknown_fee": "0"})


def authored_failed_study(root):
    """Full V8 root with pinned native input and a failed, charged attempt."""
    from hashlib import sha256

    from gluomem_bench.bench import provider_study as study

    def save(name, value):
        raw = json.dumps(value).encode()
        (root / name).write_bytes(raw)
        return sha256(raw).hexdigest()

    dataset_hash = save(
        "dataset.json",
        [
            {
                "question_id": "authored-history",
                "question_type": "single-session-user",
                "question": "What is the beacon code?",
                "answer": "C2",
                "question_date": "2026/01/04 (Sun) 00:00",
                "haystack_dates": ["2026/01/03 (Sat) 00:00"],
                "haystack_session_ids": ["s1"],
                "answer_session_ids": ["s1"],
                "haystack_sessions": [[{"role": "user", "content": "The beacon code is C2."}]],
            }
        ],
    )
    judge_hash = save("judge.json", {"authored": True})
    spec = {
        "protocol": study.PROTOCOL,
        "unsupported_cells": study.UNSUPPORTED,
        "dataset": {"path": "dataset.json", "sha256": dataset_hash},
        "judge_source": "judge.json",
        "embedding_recipe": None,
        "config_overrides": {},
        "profiles": {
            "qwen": qwen_profile(
                "http://fixture.invalid", judge_profile=deepseek_profile("http://fixture.invalid")
            ).model_dump(mode="json")
        },
        "jobs": [
            {
                "job_id": "authored-job",
                "system": "no_memory",
                "sample_id": "authored-history",
                "track": "qwen",
            }
        ],
        "authority": {"max_rate_per_million": "1", "max_usd": "1"},
    }
    spec_hash = save("execution-spec.json", spec)
    save("independent-approval.json", {"spec_sha256": spec_hash, "approved": True})
    currency = study.CurrencyReservations(
        max_usd="1",
        max_rate_per_million="1",
        receipt=root / "currency.jsonl",
    )
    reservation = currency.reserve(1000, "authored-job")
    currency.append(
        {
            "kind": "dispatch-intent",
            "reservation_id": reservation["reservation_id"],
            "job_id": "authored-job",
            "call_ordinal": 1,
            "campaign_call_ordinal": 1,
            "reserved_usd": reservation["reserved_usd"],
        }
    )
    # A lease abandoned before dispatch remains a conservative debit.
    currency.reserve(2000, "authored-job")
    result = {
        "protocol": study.PROTOCOL,
        "spec_sha256": spec_hash,
        "publication": "verified",
        "unsupported_cells": study.UNSUPPORTED,
        "complete": False,
        "frozen_inputs": [
            {"path": name, "snapshot": name, "sha256": digest}
            for name, digest in [("dataset.json", dataset_hash), ("judge.json", judge_hash)]
        ],
        "jobs": [{"job_id": "authored-job", "status": "failed", "calls": 1, "artifact_sha256": {}}],
        "accounting": {"calls": 1},
        "currency": currency.snapshot(),
    }
    save("root-status.json", result)
    return study, result


@pytest.mark.parametrize("fault", [None, "missing", "late", "duplicate", "job", "amount", "tokens"])
def test_root_reconciles_dispatch_with_preceding_reservation(tmp_path, fault):
    study, result = authored_failed_study(tmp_path)
    assert study.verify_root(tmp_path) == result
    path = tmp_path / "currency.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if fault is None:
        assert result["currency"]["reserved_usd"] == "0.003"
        return
    if fault == "missing":
        rows = [rows[1]]
        result["currency"].update(reserved_usd="0", reservations=0)
    elif fault == "late":
        rows[0], rows[1] = rows[1], rows[0]
    elif fault == "duplicate":
        rows.insert(1, copy.deepcopy(rows[0]))
    elif fault == "job":
        rows[1]["job_id"] = "other-job"
    elif fault == "amount":
        rows[1]["reserved_usd"] = "0"
    else:
        rows[1]["tokens"] = rows[0]["tokens"] + 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "root-status.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="reservation"):
        study.verify_root(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reservation_id", True),
        ("reservation_id", 1.0),
        ("tokens", 0),
        ("tokens", -1000),
        ("tokens", True),
        ("tokens", 1000.0),
    ],
)
def test_root_currency_requires_typed_positive_reservation_identity(tmp_path, field, value):
    study, result = authored_failed_study(tmp_path)
    path = tmp_path / "currency.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = value
    if field == "tokens":
        # Keep arithmetic internally consistent so only token validity can refuse.
        amount = Decimal(int(value)) / Decimal(1000000)
        rows[0].update(reserved_usd=str(amount), cumulative_reserved_usd=str(amount))
        rows[1]["reserved_usd"] = str(amount)
        rows[2]["cumulative_reserved_usd"] = str(amount + Decimal("0.002"))
        result["currency"]["reserved_usd"] = rows[2]["cumulative_reserved_usd"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "root-status.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="reservation"):
        study.verify_root(tmp_path)


def test_tracking_opt_out_never_constructs_client_or_changes_result(monkeypatch):
    from gluomem_bench import tracking

    monkeypatch.delenv("GLUOMEM_MLFLOW_ENABLED", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("optional SDK boundary was touched")

    monkeypatch.setattr(tracking, "_client_campaign", forbidden)
    value = object()

    @tracking.track_experiment("beam", kind="condition")
    def work():
        return value

    assert work() is value
