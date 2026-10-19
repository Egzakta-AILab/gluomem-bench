"""Authored offline native reporting contracts; no native execution or retained data."""

import copy
import json
from types import SimpleNamespace

import pytest

from gluomem_bench.bench import lme_first
from gluomem_bench.bench import lme_first_execute as native


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def root_case(tmp_path, monkeypatch):
    jobs = [
        dict(job_id=f"job-{i}", sample_id="authored", system=system)
        for i, system in enumerate(("gluomem", "mem0", "cognee", "no_memory", "gluomem"))
    ]
    rows = [
        dict(status=status, calls=1)
        for status in ("complete", "failed", "complete", "unrun", "complete")
    ]
    plan = dict(
        jobs=jobs,
        sample_ids=["authored"],
        judge_source="authored-scorer.py",
        future_budget_proposal=dict(max_seconds=100),
    )
    coordination = dict(
        jobs=rows,
        accounting=dict(calls=5, elapsed_seconds=2),
        limiter=dict(peak_in_flight=2),
        peak_active_history_jobs=3,
    )
    candidate = dict(output_root=str(tmp_path))
    for index, scores in ((0, [1, 0]), (2, [1]), (4, [])):
        output = tmp_path / "jobs" / jobs[index]["job_id"]
        if jobs[index]["system"] == "cognee":
            output /= "reader"
        write(output / "predictions.json", [dict(score=score) for score in scores])
    write(tmp_path / "input-cache.json", {"authored": {"source": "authored"}})
    events = []
    monkeypatch.setattr(
        native,
        "native_records",
        lambda *args: [dict(job, output=str(tmp_path / "jobs" / job["job_id"])) for job in jobs],
    )
    monkeypatch.setattr(lme_first, "verify_journal", lambda *args: coordination)
    monkeypatch.setattr(native, "verify_input_cache", lambda *args, **kw: events.append("source"))
    monkeypatch.setattr(native, "effective_config", lambda *args: ("cfg", "profile"))
    monkeypatch.setattr(native, "restore_input", lambda *args: ("adapter", "sample"))

    def authenticate(**kw):
        assert (kw["cfg"], kw["profile"], kw["adapter"], kw["sample"]) == (
            "cfg",
            "profile",
            "adapter",
            "sample",
        )
        assert kw["context"].calls == 1
        assert kw["judge_source"].name == "authored-scorer.py"
        events.append(kw["job"].job_id)

    monkeypatch.setattr(native, "verify_native_job", authenticate)
    return SimpleNamespace(
        root=tmp_path, plan=plan, candidate=candidate, coordination=coordination, events=events
    )


def verify(case, **kwargs):
    return native.verify_root(
        case.candidate,
        plan=case.plan,
        qualification={},
        candidate_path=case.root / "root-status.json",
        **kwargs,
    )


def test_publication_and_authenticated_verification_share_only_summary_fold(root_case, monkeypatch):
    case = root_case
    fold = getattr(native, "_system_summaries", None)
    assert callable(fold), "publication and verification need one shared native summary fold"
    snapshots = copy.deepcopy((case.plan, case.coordination))

    def observed(*args, **kwargs):
        case.events.append("fold")
        return fold(*args, **kwargs)

    monkeypatch.setattr(native, "_system_summaries", observed)
    report = native._root_report(case.candidate, case.plan, case.coordination)
    assert case.events == ["fold"]  # Publication does not authenticate native evidence.
    assert report["systems"] == {
        "gluomem": dict(planned=2, complete=2, failed=0, unrun=0, score_sum=1, scores=2),
        "mem0": dict(planned=1, complete=0, failed=1, unrun=0, score_sum=0, scores=0),
        "cognee": dict(planned=1, complete=1, failed=0, unrun=0, score_sum=1, scores=1),
        "no_memory": dict(planned=1, complete=0, failed=0, unrun=1, score_sum=0, scores=0),
    }  # Existing system counters are not the remediation matrix's nullable QA values.
    assert report["quality_complete"] is False
    write(case.root / "root-status.json", report)
    case.events.clear()
    assert verify(case) == report
    assert case.events == ["source", "job-0", "job-2", "job-4", "fold"]
    assert (case.plan, case.coordination) == snapshots


@pytest.mark.parametrize("boundary", ["verify_input_cache", "verify_native_job"])
def test_authentication_failure_precedes_arithmetic(root_case, monkeypatch, boundary):
    case = root_case
    report = native._root_report(case.candidate, case.plan, case.coordination)
    report["systems"]["cognee"]["score_sum"] = -99
    write(case.root / "root-status.json", report)

    def reject(*args, **kwargs):
        raise ValueError("authored evidence rejected")

    def forbidden(*args, **kwargs):
        pytest.fail("arithmetic ran before authentication finished")

    monkeypatch.setattr(native, boundary, reject)
    monkeypatch.setattr(native, "_system_summaries", forbidden)
    with pytest.raises(ValueError, match="authored evidence rejected"):
        verify(case)


def test_saved_totals_are_compared_after_authentication(root_case):
    case = root_case
    report = native._root_report(case.candidate, case.plan, case.coordination)
    report["systems"]["cognee"]["score_sum"] = 0
    write(case.root / "root-status.json", report)
    with pytest.raises(ValueError, match="native score aggregation differs"):
        verify(case)
    assert case.events == ["source", "job-0", "job-2", "job-4"]


@pytest.mark.parametrize(
    "damage, error",
    [
        ("missing", FileNotFoundError),
        ("invalid-json", json.JSONDecodeError),
        ("null-score", TypeError),
        ("short-journal", ValueError),
        ("unknown-status", KeyError),
    ],
)
def test_fold_fails_closed_without_manufacturing_scores(root_case, damage, error):
    case = root_case
    path = case.root / "jobs" / "job-2" / "reader" / "predictions.json"
    if damage == "missing":
        path.unlink()
    elif damage == "invalid-json":
        path.write_text("{", encoding="utf-8")
    elif damage == "null-score":
        write(path, [dict(score=None)])
    elif damage == "short-journal":
        case.coordination["jobs"].pop()
    else:
        case.coordination["jobs"][0]["status"] = "not-terminal"
    with pytest.raises(error):
        native._system_summaries(case.plan, case.coordination, case.root)


def test_verification_keeps_per_job_and_outer_budget_checks(root_case, monkeypatch):
    case = root_case
    write(
        case.root / "root-status.json",
        native._root_report(case.candidate, case.plan, case.coordination),
    )
    budget = SimpleNamespace(check=lambda: case.events.append("budget"))
    verify(case, budget=budget)
    assert case.events[:9] == [
        "source",
        "budget",
        "job-0",
        "budget",
        "budget",
        "job-2",
        "budget",
        "budget",
        "job-4",
    ]
    assert case.events[9:] == ["budget"] * 6  # Fold checks plus final outer check.

    class Expired(Exception):
        pass

    def expire():
        raise Expired()

    monkeypatch.setattr(budget, "check", expire)
    with pytest.raises(Expired):
        verify(case, budget=budget)


def test_completion_receipt_and_revocation_stay_outside_fold(root_case):
    case = root_case
    write(
        case.root / "root-status.json",
        native._root_report(case.candidate, case.plan, case.coordination),
    )
    receipt = dict(
        root_sha256=native.bound(case.root / "root-status.json")["sha256"],
        calls=5,
        elapsed_seconds=3,
    )
    write(case.root / "completion-receipt.json", receipt)
    assert native.verify_root(case.candidate, plan=case.plan, qualification={})["planned"] == 5
    receipt["elapsed_seconds"] = 100
    write(case.root / "completion-receipt.json", receipt)
    with pytest.raises(ValueError, match="outer clock closure differs"):
        native.verify_root(case.candidate, plan=case.plan, qualification={})
    write(case.root / "incomplete.json", {})
    with pytest.raises(ValueError, match="revocation overrides"):
        native.verify_root(case.candidate, plan=case.plan, qualification={})
