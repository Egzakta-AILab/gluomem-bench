"""Pool receipt mirrors are not additional physical model calls."""

import copy
import json

import pytest


def wire(ordinal, logical, task, status=200):
    packet = {
        "url": "http://owned/api/v1/chat/completions",
        "request": {"messages": [{"role": "user", "content": "unit-only-private-payload"}]},
        "status": status,
        "response": {}
        if status != 200
        else {"usage": {"prompt_tokens": 23, "completion_tokens": 3}},
    }
    attempt = {
        "pool_key": "a" * 64,
        "logical_ordinal": logical,
        "ordinal": ordinal,
        "intent": f"{ordinal:064x}",
        "wire": copy.deepcopy(packet),
    }
    return {
        **packet,
        "task": task,
        "pool_key": attempt["pool_key"],
        "logical_ordinal": logical,
        "pool_attempt": attempt,
    }


def save(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_native_pool_mirrors_and_separate_common_judge_are_counted_once(tmp_path):
    from gluomem_bench.tracking_receipts import collect_record

    primary = [
        wire(1, 1, "event_claims", 503),
        wire(2, 1, "event_claims"),
        wire(3, 2, "consolidate"),
        wire(4, 3, "answer"),
    ]
    judge = [wire(5, 4, "rubric")]
    native = [
        {
            "task": "event_claims",
            "ordinal": 2,
            "pool_key": "a" * 64,
            "logical_ordinal": 1,
            "pool_attempts": [r["pool_attempt"] for r in primary[:2]],
        },
        {
            "task": "consolidate",
            "ordinal": 3,
            "pool_key": "a" * 64,
            "logical_ordinal": 2,
            "pool_attempts": [primary[2]["pool_attempt"]],
        },
    ]
    save(tmp_path / "calls.jsonl", primary)
    save(tmp_path / "calls-judge.jsonl", judge)
    save(tmp_path / "production-raw.jsonl", native)
    record = collect_record({"out": tmp_path}, {"generation_calls": 5})
    usage = record["usage"]
    assert usage["traced_attempts"] == usage["generation_calls"] == 5
    assert usage["known_usage_calls"] == 4
    assert usage["unknown_usage_calls"] == 1
    assert usage["prompt_tokens_observed"] == 4 * 23
    assert usage["cost_usd"] is None
    assert usage["roles"]["rubric"]["calls"] == 1
    assert usage["roles"]["consolidate"]["calls"] == 1
    assert "unit-only-private-payload" not in json.dumps(record)


def test_distinct_pool_attempts_with_identical_payloads_are_not_merged(tmp_path):
    from gluomem_bench.tracking_receipts import collect_record

    save(tmp_path / "calls.jsonl", [wire(1, 1, "answer"), wire(2, 2, "answer")])
    result = collect_record({"out": tmp_path}, {"generation_calls": 2})
    assert result["usage"]["traced_attempts"] == 2
    assert result["usage"]["known_usage_calls"] == 2


@pytest.mark.parametrize("fault", ["duplicate-authoritative", "conflicting-mirror"])
def test_duplicate_identity_or_conflicting_pool_mirror_still_refuses(tmp_path, fault):
    from gluomem_bench.tracking_receipts import collect_record

    first = wire(1, 1, "event_claims")
    if fault == "duplicate-authoritative":
        save(tmp_path / "calls.jsonl", [first, first])
        message = "duplicate authoritative pool physical receipt"
    else:
        save(tmp_path / "calls.jsonl", [first])
        attempt = copy.deepcopy(first["pool_attempt"])
        attempt["wire"]["response"]["usage"]["prompt_tokens"] += 1
        save(
            tmp_path / "production-raw.jsonl",
            [{"ordinal": 1, "task": "event_claims", "pool_attempts": [attempt]}],
        )
        message = "conflicting mirrored pool physical receipt"
    with pytest.raises(ValueError, match=message):
        collect_record({"out": tmp_path}, {"generation_calls": 2})


def test_native_only_pool_trace_retains_each_attempt_and_unknown_usage(tmp_path):
    from gluomem_bench.tracking_receipts import collect_record

    attempts = [wire(1, 1, "event_claims", 503), wire(2, 1, "event_claims")]
    save(
        tmp_path / "production-raw.jsonl",
        [
            {
                "ordinal": 2,
                "task": "event_claims",
                "pool_attempts": [r["pool_attempt"] for r in attempts],
            }
        ],
    )
    result = collect_record({"out": tmp_path}, {"generation_calls": 2})
    assert result["usage"]["traced_attempts"] == 2
    assert result["usage"]["known_usage_calls"] == result["usage"]["unknown_usage_calls"] == 1
