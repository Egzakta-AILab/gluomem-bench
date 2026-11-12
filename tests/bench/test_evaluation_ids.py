"""Authored public-ID boundary cases; no captured observations or authority."""

import json
import runpy
import sys
from copy import deepcopy

import pytest

import gluomem_bench.evaluation as q


@pytest.fixture(params=["api", "module"])
def evaluate_case(request, tmp_path, monkeypatch, capsys):
    def invoke(case, *, expected_exit=0):
        if request.param == "api":
            return q.evaluate(case["results"], case["checks"], case["snapshot"])
        path = tmp_path / "authored-case.json"
        path.write_text(json.dumps(case), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["gluomem_bench.evaluation", str(path)])
        with pytest.raises(SystemExit) as exited:
            runpy.run_module("gluomem_bench.evaluation", run_name="__main__", alter_sys=True)
        assert exited.value.code == expected_exit
        return json.loads(capsys.readouterr().out)

    return invoke


def literal_case():
    return {
        "checks": [{"id": "authored", "contains": ["fixture-value"]}],
        "results": [
            {
                "id": "authored",
                "answer": "fixture-value",
                "evidence": {"context": "fixture-value", "hits": []},
            }
        ],
        "snapshot": {},
    }


@pytest.mark.parametrize("counterexample", ["bool-alias", "float-alias", "duplicate-result"])
def test_review_counterexamples_refuse(evaluate_case, counterexample):
    # Exact authored inputs supplied with BEVAL-PUB-01, not captured evidence.
    case = literal_case()
    if counterexample == "duplicate-result":
        wrong = deepcopy(case["results"][0])
        wrong["answer"] = "wrong"
        case["results"].insert(0, wrong)
    else:
        case["checks"][0]["id"] = 1
        case["results"][0]["id"] = True if counterexample == "bool-alias" else 1.0
    with pytest.raises(q.Refusal, match="(invalid|duplicate)-(check|result)-id"):
        evaluate_case(case)


@pytest.mark.parametrize("collection,label", [("checks", "check"), ("results", "result")])
@pytest.mark.parametrize("identifier", [None, True, 1, 1.0, [], {}, "", " ", "\t\n", "missing"])
def test_unsupported_and_missing_ids_refuse(evaluate_case, collection, label, identifier):
    case = literal_case()
    if identifier == "missing":
        del case[collection][0]["id"]
    else:
        case[collection][0]["id"] = identifier
    with pytest.raises(q.Refusal, match="invalid-" + label + "-id"):
        evaluate_case(case)


@pytest.mark.parametrize("collection,label", [("checks", "check"), ("results", "result")])
@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_ids_refuse_in_both_orders(evaluate_case, collection, label, reverse):
    case = literal_case()
    duplicate = deepcopy(case[collection][0])
    duplicate["answer" if collection == "results" else "contains"] = "wrong"
    case[collection].append(duplicate)
    if reverse:
        case[collection].reverse()
    with pytest.raises(q.Refusal, match="duplicate-" + label + "-id"):
        evaluate_case(case)


@pytest.mark.parametrize("identifier", ["authored", " authored ", "two words", "é", "a\tb"])
def test_ordinary_exact_string_positive(evaluate_case, identifier):
    case = literal_case()
    case["checks"][0]["id"] = case["results"][0]["id"] = identifier
    rows = evaluate_case(case)
    assert len(rows) == 1 and rows[0]["passed"] is True
    assert rows[0]["id"] == identifier


def test_ids_are_not_trimmed_casefolded_or_repaired(evaluate_case):
    case = literal_case()
    for identifier in [" authored ", "AUTHORED", "different"]:
        case["results"][0]["id"] = identifier
        assert evaluate_case(case, expected_exit=1) == [
            {"id": "authored", "passed": False, "reason": "unrun"}
        ]
    case["results"] = []
    assert evaluate_case(case, expected_exit=1) == [
        {"id": "authored", "passed": False, "reason": "unrun"}
    ]
    case["checks"] = []
    assert evaluate_case(case) == []


def test_id_refusal_precedes_policy_and_literal_grading(monkeypatch):
    from gluomem_bench.evaluation import state_contracts

    def forbidden(*args, **kwargs):
        pytest.fail("identity preflight ran too late")

    monkeypatch.setattr(state_contracts, "validate_correction_policy", forbidden)
    case = literal_case()
    case["checks"][0]["correction_context_policy"] = {}
    case["checks"].append({"id": None})
    with pytest.raises(q.Refusal, match="invalid-check-id"):
        q.evaluate(case["results"], case["checks"], case["snapshot"])


def test_module_id_refusal_precedes_source_reads_and_origin_binding(tmp_path, monkeypatch, capsys):
    case = literal_case()
    case["results"].append(deepcopy(case["results"][0]))
    case["source"] = {"path": "MUST-NOT-BE-READ", "sha256": "0" * 64, "scope": {}}
    case["materializations"] = []
    path = tmp_path / "authored-case.json"
    path.write_text(json.dumps(case), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(sys, "argv", ["gluomem_bench.evaluation", str(path)])
    with pytest.raises(q.Refusal, match="duplicate-result-id"):
        runpy.run_module("gluomem_bench.evaluation", run_name="__main__", alter_sys=True)
    assert capsys.readouterr().out == ""
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_string_subclasses_are_not_builtin_ids():
    class AuthoredString(str):
        pass

    case = literal_case()
    case["checks"][0]["id"] = AuthoredString("authored")
    with pytest.raises(q.Refusal, match="invalid-check-id"):
        q.evaluate(case["results"], case["checks"], case["snapshot"])
