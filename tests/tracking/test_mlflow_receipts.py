"""Metric projections use actual existing driver receipt schemas."""

from __future__ import annotations

import json

import pytest


@pytest.mark.parametrize(
    "family,payload,path,value",
    [
        (
            "longmemeval",
            {"questions": 4, "by_category": {"knowledge-update": {"n": 4, "mean": 0.75}}},
            "longmemeval.native.scores.by_category.knowledge-update.mean",
            0.75,
        ),
        (
            "memoryagentbench",
            {"conflict_resolution": {"exact_match": 0.5, "n": 2}},
            "memoryagentbench.native.scores.conflict_resolution.exact_match",
            0.5,
        ),
        (
            "beam",
            {"event_ordering": {"f1": 1.0, "tau_norm": 0.5, "final_score": 0.5}},
            "beam.native.scores.event_ordering.tau_norm",
            0.5,
        ),
        (
            "memoryarena_shopping",
            {"official_metrics": {"success_rate": 0.25}},
            "memoryarena_shopping.native.scores.official_metrics.success_rate",
            0.25,
        ),
        (
            "retrieval_ablation",
            {"rows": [{"ndcg": 0.5, "recall": 0.75}]},
            "retrieval_ablation.native.scores.rows.0.ndcg",
            0.5,
        ),
    ],
)
def test_native_receipt_objectives_are_not_silently_discarded(
    tmp_path, family, payload, path, value
):
    from gluomem_bench.tracking_receipts import collect_record, metric_values

    (tmp_path / "scores.json").write_text(json.dumps(payload), encoding="utf-8")
    metrics = metric_values(collect_record({"out": tmp_path}, {}), family)
    assert metrics.get(path) == value, "existing native objective disappeared from MLflow"


def test_legacy_tagged_scores_keep_their_metric_identity(tmp_path):
    from gluomem_bench.tracking_receipts import collect_record, metric_values

    score = {
        "scope": "overall",
        "value": 0.75,
        "n": 4,
        "tag": {
            "metric_family": "qa_accuracy",
            "judge_family": "single",
            "backbone": "reader-v1",
            "dataset_sha": "a" * 64,
            "protocol": "native:lme@v1",
        },
    }
    (tmp_path / "scores.json").write_text(json.dumps([score]), encoding="utf-8")
    record = collect_record({"out": tmp_path}, {})
    entry = record["receipts"]["scores.json"][0]
    assert entry.get("tag", {}).get("metric_family") == "qa_accuracy", "metric name lost"
    assert any(
        "qa_accuracy" in k and v == 0.75 for k, v in metric_values(record, "longmemeval").items()
    )


def test_ingestion_errors_retain_a_numeric_failure_count_not_private_text():
    from dataclasses import dataclass

    from gluomem_bench.tracking import _failed
    from gluomem_bench.tracking_receipts import collect_record

    @dataclass
    class Report:
        sessions: int
        extraction_errors: list[str]

    record = collect_record({}, Report(2, ["private extraction failure message"]))
    assert _failed(record["result"]), "ingestion failure disappeared from the safe view"
    assert "private extraction failure message" not in json.dumps(record)


def test_cognee_nested_reader_and_native_dispatches_are_counted_once(tmp_path):
    import base64

    from gluomem_bench.tracking_receipts import collect_record

    reader = tmp_path / "reader"
    graph = tmp_path / "graph"
    reader.mkdir()
    graph.mkdir()
    (reader / "calls.jsonl").write_text(
        json.dumps(
            {
                "url": "http://local/v1/chat/completions",
                "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
            }
        )
        + "\n"
    )
    encoded = base64.b64encode(
        json.dumps({"usage": {"prompt_tokens": 8, "completion_tokens": 3}}).encode()
    ).decode()
    rows = [
        {"kind": "dispatch-intent"},
        {"kind": "/chat/completions", "response_b64": encoded},
        {"kind": "generation", "response_b64": encoded},  # wrapper duplicate, not dispatch
        {"kind": "dispatch-intent"},
    ]
    (graph / "generation.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    record = collect_record({"out": tmp_path}, {"generation_calls": 3, "status": "failed"})
    assert record["usage"].get("traced_attempts") == 3, "native/reader nested attempts were omitted"
    assert record["usage"]["known_usage_calls"] == 2
    assert record["usage"]["unknown_usage_calls"] == 1
    assert record["usage"]["prompt_tokens_observed"] == 18
    assert record["usage"]["completion_tokens"] is None


def test_resource_reservations_are_not_reported_as_actual_generation_spend():
    from gluomem_bench.tracking_receipts import collect_record

    result = {
        "accounting": {"calls": 40, "prior_calls": 37, "new_calls": 3},
        "resource_accounting": {
            "prior_debit_kind": "reserved",
            "conservative_prior_allowance": {"calls": 37},
            "actual_additional_spend": {"calls": 3},
        },
    }
    record = collect_record({}, result)
    assert record["usage"].get("generation_calls") == 3, "reservation became actual model spend"
    assert record["usage"]["reserved_prior_calls"] == 37
    assert record["usage"]["call_count_kind"] == "actual_additional_only"


def test_pending_native_quality_remains_explicitly_weak_not_complete():
    from gluomem_bench.tracking import _outcome, project_result

    raw = {"quality_status": "verified_pending_post_exit", "generation_calls": 2}
    assert project_result(raw).get("quality_status") == "verified_pending_post_exit"
    assert _outcome(raw) == "weak"


def test_private_category_labels_are_hashed_even_inside_numeric_metrics():
    from gluomem_bench.tracking import project_result

    private = "identifier-from-user-history"
    result = project_result(
        {"metrics": {private: {"accuracy": 0.5}, "event_ordering": {"f1": 1.0}}}
    )
    assert private not in json.dumps(result), "private input category escaped via metric key"
    assert result["metrics"]["event_ordering"]["f1"] == 1.0
    assert len(result["metrics"]) == 2, "privacy must not silently discard the numeric objective"


def test_explicit_core_default_model_is_part_of_effective_config_identity():
    from gluomem_bench.tracking import experiment_params

    cfg = {
        "dspy": {"model": "model-a", "tasks": {"answer": {"model": None}}},
        "models": {
            "model-a": {"provider_model": "explicit/model-a"},
            "model-b": {"provider_model": "explicit/model-b"},
        },
    }
    first = experiment_params(lambda: None, {"cfg": cfg}, code={})
    cfg["dspy"]["model"] = "model-b"
    second = experiment_params(lambda: None, {"cfg": cfg}, code={})
    assert first["config.sha256"] != second["config.sha256"], (
        "effective model change had identical experiment identity"
    )
    assert first["profile.sha256"] != second["profile.sha256"]
    assert first["profile.model"] == "model-a"
