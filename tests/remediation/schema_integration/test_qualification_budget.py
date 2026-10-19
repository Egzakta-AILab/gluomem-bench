"""Qualification cannot create another unmetered phase allowance."""
from types import SimpleNamespace

import pytest
from gluomem_bench.bench import lme_first_execute as native


def arguments(tmp_path):
    return dict(plan_path=tmp_path / "missing-plan", plan_sha256="a" * 64,
                budget_decision_path=tmp_path / "missing-permit", budget_decision_sha256="b" * 64,
                source_path=tmp_path / "missing-source", out=tmp_path / "must-not-exist",
                cognee_python="not-a-program", host_reserve_mib=8192, history_commit_limit_mib=4096)


def test_supplied_qualification_budget_is_checked_before_source_reads(tmp_path):
    class Expired(Exception):
        pass

    def check():
        raise Expired()

    budget = SimpleNamespace(max_calls=192, max_seconds=1200, check=check)
    with pytest.raises(Expired):
        native.run_authored_qualification(**arguments(tmp_path), budget=budget)
    assert not (tmp_path / "must-not-exist").exists()


def test_supplied_qualification_budget_cannot_expand_declared_caps(tmp_path):
    budget = SimpleNamespace(max_calls=193, max_seconds=1200, check=lambda: None)
    with pytest.raises(ValueError, match="qualification budget"):
        native.run_authored_qualification(**arguments(tmp_path), budget=budget)
