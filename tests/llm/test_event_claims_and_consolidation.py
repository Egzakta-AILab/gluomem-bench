"""Unit tests for unified EventClaimExtraction and ClaimConsolidation scoring & optimization."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import dspy
import pytest

from gluomem_bench.config import load_config
from gluomem_bench.llm.extraction_eval import (
    SEALED_SPLIT,
    SealedSplitError,
    load_fixture,
    score_consolidation_case,
    score_event_claims_case,
)
from gluomem_bench.llm.optimize import (
    check_trainset,
    fixture_examples,
    metric_for,
    optimize_task,
)
from gluomem_bench.llm.outputs import (
    ConsolidationDecision,
    ExtractedClaimSpec,
    KGEntity,
    KGRelation,
    StoredCandidate,
)
from gluomem_bench.llm.signatures import (
    TASK_SIGNATURES,
    ClaimConsolidation,
    ConsolidateClaimSignature,
    EventClaimExtraction,
    ExtractEventClaimsSignature,
)


def test_signatures_registered_and_aliased() -> None:
    assert TASK_SIGNATURES["event_claims"] is EventClaimExtraction
    assert TASK_SIGNATURES["consolidate"] is ClaimConsolidation
    assert ExtractEventClaimsSignature is EventClaimExtraction
    assert ConsolidateClaimSignature is ClaimConsolidation


def test_score_event_claims_case_against_fixture_and_inline_cases() -> None:
    fx = load_fixture()
    case = fx.select(task="fact_extraction", splits=("train",))[0]

    # Invalid schema scores 0
    bad = score_event_claims_case(case, "not-a-list")
    assert bad.score == 0.0
    assert bad.errors == {"invalid_schema": 1}

    # Inline event_claims case with verbatim span and date
    inline_case = {
        "case_id": "ec-inline-1",
        "split": "train",
        "language": "en",
        "source_text": "[m1] user: We switched our primary queue to RabbitMQ on 2026-03-15.",
        "reference_date": "2026-03-15T00:00:00Z",
        "claims": [
            {
                "claim_id": "c1",
                "text": "Primary queue is RabbitMQ",
                "verbatim_span": "switched our primary queue to RabbitMQ on 2026-03-15",
                "date": "2026-03-15",
            }
        ],
    }
    good_pred = [
        ExtractedClaimSpec(
            text="We switched our primary queue to RabbitMQ on 2026-03-15.",
            kind="decision",
            confidence=0.95,
            event_date=datetime(2026, 3, 15, tzinfo=UTC),
            verbatim_span="switched our primary queue to RabbitMQ on 2026-03-15",
            entities=[KGEntity(name="RabbitMQ", entity_type="concept")],
            relations=[
                KGRelation(
                    source="primary queue",
                    relation_type="uses",
                    target="RabbitMQ",
                )
            ],
        )
    ]
    res = score_event_claims_case(inline_case, good_pred)
    assert res.score == 1.0
    assert res.components["span_fidelity"] == 1.0
    assert "unsupported_span" not in res.errors

    # Fabricated verbatim span is flagged in errors and span_fidelity
    bad_span_pred = [
        ExtractedClaimSpec(
            text="We switched our primary queue to RabbitMQ on 2026-03-15.",
            kind="decision",
            confidence=0.95,
            event_date=datetime(2026, 3, 15, tzinfo=UTC),
            verbatim_span="completely fabricated quote not in source",
        )
    ]
    res_bad_span = score_event_claims_case(inline_case, bad_span_pred)
    assert res_bad_span.components["span_fidelity"] == 0.0
    assert res_bad_span.errors.get("unsupported_span") == 1

    # Sealed split is refused for optimization
    sealed_case = fx.select(splits=(SEALED_SPLIT,))[0]
    with pytest.raises(SealedSplitError):
        score_event_claims_case(sealed_case, [], purpose="optimize")


def test_score_consolidation_case_actions_and_targets() -> None:
    candidates = [
        StoredCandidate(frame_id=10, content="SSH bastion port is 22.", kind="fact"),
        StoredCandidate(frame_id=11, content="Deploy window is Tuesday.", kind="fact"),
    ]
    case = {
        "case_id": "cons-1",
        "split": "train",
        "new_claim": "SSH bastion port was changed to 22022.",
        "candidates": candidates,
        "decision": {"action": "SUPERSEDE", "target_frame_id": 10},
    }

    exact = score_consolidation_case(
        case,
        ConsolidationDecision(
            action="SUPERSEDE",
            target_frame_id=10,
            rationale="Port changed from 22 to 22022.",
        ),
    )
    assert exact.score == 1.0
    assert exact.errors == {}

    wrong_target = score_consolidation_case(
        case,
        ConsolidationDecision(action="SUPERSEDE", target_frame_id=11),
    )
    assert wrong_target.score == 0.5
    assert wrong_target.errors == {"wrong_target": 1}

    invalid_target = score_consolidation_case(
        case,
        ConsolidationDecision(action="SUPERSEDE", target_frame_id=999),
    )
    assert invalid_target.score == 0.5
    assert "invalid_target" in invalid_target.errors

    wrong_action = score_consolidation_case(
        case,
        ConsolidationDecision(action="ADD", target_frame_id=None),
    )
    assert wrong_action.score == 0.0
    assert "wrong_action" in wrong_action.errors

    bad_schema = score_consolidation_case(case, {"action": "UNKNOWN"})
    assert bad_schema.score == 0.0
    assert bad_schema.errors == {"invalid_schema": 1}


def test_optimize_wiring_for_event_claims_and_consolidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    examples = fixture_examples("event_claims", splits=("train",))
    assert len(examples) > 0
    check_trainset("event_claims", examples)

    ec_metric = metric_for("event_claims")
    pred_empty = dspy.Prediction(claims=[])
    scalar = ec_metric(examples[0], pred_empty)
    assert isinstance(scalar, float)
    gepa_pred = ec_metric(examples[0], pred_empty, pred_name="event_claims")
    assert hasattr(gepa_pred, "score") and hasattr(gepa_pred, "feedback")

    cons_examples = [
        dspy.Example(
            new_claim="Redis port is 6380.",
            candidates=[{"frame_id": 1, "content": "Redis port is 6379.", "kind": "fact"}],
            decision={"action": "SUPERSEDE", "target_frame_id": 1},
        ).with_inputs("new_claim", "candidates")
    ]
    check_trainset("consolidate", cons_examples)
    cons_metric = metric_for("consolidate")
    assert (
        cons_metric(
            cons_examples[0],
            dspy.Prediction(decision=ConsolidationDecision(action="SUPERSEDE", target_frame_id=1)),
        )
        == 1.0
    )

    cfg = load_config().with_overrides({"dspy.artifacts.dir": str(tmp_path / "artifacts")})

    class FakeTeleprompter:
        def __init__(self, metric: object) -> None:
            self.metric = metric

        def compile(self, student: dspy.Module, *, trainset: object) -> dspy.Module:
            assert len(list(trainset)) > 0  # type: ignore[arg-type]
            return student

    out_ec = optimize_task(
        "event_claims",
        cfg,
        examples[:2],
        teleprompter_factory=FakeTeleprompter,
    )
    assert (out_ec / "program.pkl").exists()

    out_cons = optimize_task(
        "consolidate",
        cfg,
        cons_examples,
        teleprompter_factory=FakeTeleprompter,
    )
    assert (out_cons / "program.pkl").exists()
