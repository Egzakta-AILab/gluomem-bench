"""Frozen source-backed extraction evaluator and CachedExtractors tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gluomem_bench.llm.extraction_eval import (
    CAPABILITIES,
    FIXTURE_PATH,
    SEALED_SPLIT,
    SPLITS,
    TASKS,
    FixtureError,
    SealedSplitError,
    evaluator_fingerprint,
    load_fixture,
    normalize,
    parse_fixture,
    score_case,
    scorecard,
)


def _facts(*texts: str) -> list[dict]:
    return [{"text": t, "confidence": 1.0} for t in texts]


def _events(*rows: tuple[str, str | None]) -> list[dict]:
    return [{"text": t, "event_date": d, "confidence": 1.0} for t, d in rows]


ANA_GOLD = _facts(
    "Ana moved from Belgrade to Novi Sad in March 2026.",
    "Marko lives in Belgrade.",
    "Ana does not own a car.",
    "Ana might buy a bicycle next year.",
)

GOLDEN = [
    pytest.param("en-ana-move", ANA_GOLD, 1.0, {}, id="verbatim"),
    pytest.param(
        "en-ana-move",
        _facts(
            "Ana relocated to Novi Sad.",
            "Ana's brother Marko is based in Belgrade.",
            "Ana doesn't have a car.",
            "Ana is considering getting a bike.",
        ),
        1.0,
        {},
        id="supported-paraphrase",
    ),
    pytest.param(
        "en-ana-move",
        _facts("Ana lives in Paris.", "Ana works at a bank."),
        0.0,
        {"unsupported": 2},
        id="valid-but-unsupported",
    ),
    pytest.param(
        "en-ana-move",
        _facts(
            "Marko moved to Novi Sad.",
            "Marko lives in Belgrade.",
            "Ana does not own a car.",
            "Ana might buy a bicycle next year.",
        ),
        0.75,
        {"wrong_subject": 1},
        id="wrong-subject",
    ),
    pytest.param(
        "en-ana-move",
        _facts(
            "She moved to Novi Sad.",
            "Marko lives in Belgrade.",
            "Ana does not own a car.",
            "Ana might buy a bicycle next year.",
        ),
        0.75,
        {"missing_subject": 1},
        id="pronoun-subject",
    ),
    pytest.param(
        "en-ana-move",
        _facts(
            "Ana moved to Novi Sad.",
            "Marko lives in Belgrade.",
            "Ana owns a car.",
            "Ana might buy a bicycle next year.",
        ),
        0.75,
        {"negation_lost": 1},
        id="negation-loss",
    ),
    pytest.param(
        "en-ana-move",
        _facts(
            "Ana moved to Novi Sad.",
            "Marko lives in Belgrade.",
            "Ana does not own a car.",
            "Ana bought a bicycle.",
        ),
        0.75,
        {"hypothetical_as_completed": 1},
        id="hypothetical-as-completed",
    ),
    pytest.param(
        "en-ana-move",
        [*ANA_GOLD, {"text": "Ana lives in Belgrade."}],
        0.8889,
        {"stale_as_current": 1},
        id="stale-as-current",
    ),
    pytest.param(
        "en-ana-move",
        [*ANA_GOLD, {"text": "Ana previously lived in Belgrade."}],
        1.0,
        {},
        id="superseded-marked-previous",
    ),
    pytest.param(
        "en-ana-move",
        _facts(
            "Ana moved to Novi Sad.",
            "Marko used to live in Belgrade.",
            "Ana does not own a car.",
            "Ana might buy a bicycle next year.",
        ),
        0.75,
        {"current_as_stale": 1},
        id="current-as-stale",
    ),
    pytest.param(
        "sr-latn-ana-move",
        _facts(
            "Ana se preselila iz Beograda u Novi Sad.",
            "Marko živi u Beogradu.",
            "Ana nema automobil.",
            "Ana će možda kupiti bicikl.",
        ),
        1.0,
        {},
        id="sr-latn-native",
    ),
    pytest.param(
        "sr-latn-ana-move",
        _facts(
            "Ana se preselila u Novi Sad.",
            "Marko živi u Beogradu.",
            "Ana ima automobil.",
            "Ana je kupila bicikl.",
        ),
        0.5,
        {"negation_lost": 1, "hypothetical_as_completed": 1},
        id="sr-latn-negation-and-modality",
    ),
    pytest.param("en-greeting", [], 1.0, {}, id="correct-abstention"),
    pytest.param(
        "sr-cyrl-greeting",
        _facts("Корисник поздравља."),
        0.0,
        {"unsupported": 1},
        id="spurious-on-small-talk",
    ),
    pytest.param("en-ana-move", [], 0.0, {}, id="over-abstention"),
    pytest.param(
        "en-demo-dates",
        _events(
            ("Project kickoff", "2026-02-03T00:00:00"),
            ("Client demo", "2026-03-24T00:00:00"),
        ),
        1.0,
        {},
        id="events-current-dates",
    ),
    pytest.param(
        "en-demo-dates",
        _events(
            ("Project kickoff", "2026-02-03T00:00:00"),
            ("Client demo", "2026-03-10T00:00:00"),
        ),
        0.5,
        {"stale_as_current": 1},
        id="events-stale-date-as-current",
    ),
    pytest.param(
        "en-demo-dates",
        _events(
            ("Project kickoff", "2026-02-04T00:00:00"),
            ("Client demo", None),
        ),
        0.0,
        {"wrong_date": 1, "missing_date": 1},
        id="events-wrong-and-missing-date",
    ),
    pytest.param(
        "sr-cyrl-plant-migration",
        _events(("Фабрика у Крагујевцу је прешла на SAP.", "2025-03-01T00:00:00")),
        1.0,
        {},
        id="sr-cyrl-month-precision",
    ),
    pytest.param(
        "en-plant-migration",
        _facts(
            "The Kragujevac plant migrated its ERP to SAP in 2025.",
            "The Čačak plant has not migrated yet.",
            "The Čačak plant plans to migrate in 2027.",
        ),
        1.0,
        {},
        id="plan-kept-as-plan",
    ),
    pytest.param(
        "en-nikola-clinic",
        _facts(
            "Nikola now practices at the Zemun hospital.",
            "Nikola previously worked at the Vračar clinic.",
            "Milica works at the Vračar clinic.",
            "Nikola does not treat pediatric patients.",
            "Nikola might open a dental lab in autumn.",
        ),
        1.0,
        {},
        id="en-nikola-clinic-full",
    ),
    pytest.param(
        "sr-cyrl-nikola-clinic",
        _facts(
            "Никола сада ради у болници у Земуну.",
            "Никола је раније радио на Врачару.",
            "Милица ради на Врачару.",
            "Никола не прима педијатријске пацијенте.",
            "Никола ће можда отворити стоматолошку ординацију.",
        ),
        1.0,
        {},
        id="sr-cyrl-nikola-clinic-full",
    ),
    pytest.param(
        "en-datacenter-cooling",
        _facts(
            "Hall A uses liquid cooling.",
            "Hall B uses air cooling.",
            "Hall A does not run diesel generators indoors.",
            "Hall B plans to expand in 2027.",
        ),
        1.0,
        {},
        id="en-datacenter-cooling-full",
    ),
    pytest.param(
        "sr-latn-demo-dates",
        _events(
            ("Početak projekta", "2026-02-03T00:00:00"),
            ("Prezentacija za klijenta", "2026-03-24T00:00:00"),
        ),
        1.0,
        {},
        id="sr-latn-demo-dates-full",
    ),
]


@pytest.mark.parametrize(("case_id", "prediction", "expected", "errors"), GOLDEN)
def test_golden_scores(case_id: str, prediction: list, expected: float, errors: dict) -> None:
    case = load_fixture().case(case_id)
    result = score_case(case, prediction)
    assert result.score == pytest.approx(expected, abs=1e-4)
    assert dict(result.errors) == errors


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "Ana moved.",
        [{"text": ""}],
        [{"text": "x", "confidence": 2}],
        [{"text": "x", "event_date": "not-a-date"}],
        [42],
    ],
)
def test_invalid_schema_scores_zero(bad: object) -> None:
    result = score_case(load_fixture().case("en-ana-move"), bad)
    assert result.score == 0.0 and result.errors == {"invalid_schema": 1}


def test_evaluator_components_redaction_and_normalization() -> None:
    assert normalize("Чачак, Đurđevdan, živi") == ["cacak", "djurdjevdan", "zivi"]
    assert normalize("Ana doesn't") == ["ana", "does", "not"]

    fx = load_fixture()
    case = fx.case("en-ana-move")
    res = score_case(
        case, _facts("Ana moved to Novi Sad.", "Ana owns a car.", "Ana works at a bank.")
    )
    assert res.components["schema_validity"] == 1.0
    assert res.components["support_precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert res.components["polarity_fidelity"] == 0.0

    sec = score_case(
        case,
        [
            {
                "text": "Ana's api_key=sk-live-abcdef1234567890 is stored.",
                "reasoning": "HIDDEN-COT",
            },
            {"text": "Bearer abc.def.ghi unlocks it."},
        ],
    )
    assert "HIDDEN-COT" not in sec.feedback and "sk-live" not in sec.feedback
    assert "[redacted]" in sec.feedback
    assert "Ana's" not in json.dumps(sec.record())


def test_splits_sealing_fixture_validation_and_scorecard(tmp_path: Path) -> None:
    fx = load_fixture()
    assert len(fx.cases) >= 24
    for split in SPLITS:
        split_cases = [c for c in fx.cases if c.split == split]
        assert len(split_cases) >= 8
        assert {c.language for c in split_cases} == {"en", "sr-Latn", "sr-Cyrl"}
        assert {c.task for c in split_cases} == set(TASKS)
        assert {cap for c in split_cases for cl in c.claims for cap in cl.capabilities} == set(
            CAPABILITIES
        )
    manifest = fx.split_manifest()
    assert manifest["policy"]["group_by"] == "source_family"
    assert {"en-jelena-role", "sr-latn-jelena-role"} <= set(
        manifest["splits"][SEALED_SPLIT]["cases"]
    )

    sealed_case = fx.case("en-jelena-role")
    with pytest.raises(SealedSplitError):
        score_case(sealed_case, _facts("Jelena leads the payments team."))
    rep = score_case(sealed_case, _facts("Jelena leads the payments team."), purpose="report")
    assert rep.score > 0 and rep.feedback == ""

    doc = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    doc_split = json.loads(json.dumps(doc))
    doc_split["cases"][1]["split"] = "train"
    with pytest.raises(FixtureError, match="belongs to one split"):
        parse_fixture(doc_split, sha256="x")

    doc_span = json.loads(json.dumps(doc))
    doc_span["cases"][0]["claims"][0]["support"] = "Ana moved to Paris."
    with pytest.raises(FixtureError, match="verbatim"):
        parse_fixture(doc_span, sha256="x")

    edited = tmp_path / "extraction_v1.json"
    doc_ed = json.loads(json.dumps(doc))
    doc_ed["cases"][0]["claims"].pop()
    edited.write_text(json.dumps(doc_ed), encoding="utf-8")
    with pytest.raises(FixtureError, match="frozen pin"):
        load_fixture(edited, expected_sha256=fx.sha256)

    dev = [
        score_case(fx.case("en-ana-move"), ANA_GOLD),
        score_case(fx.case("en-greeting"), []),
    ]
    card = scorecard(dev)
    assert card["split"] == "dev" and card["evaluator"] == evaluator_fingerprint()
    with pytest.raises(ValueError, match="one scorecard per split"):
        scorecard([*dev, rep])
