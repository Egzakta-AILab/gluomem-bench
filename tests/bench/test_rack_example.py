"""Portable example freezes bounded roles and a shared whole-run allowance."""

from pathlib import Path

import pytest

from gluomem_bench.bench.rack import RackProfile
from gluomem_bench.bench.rack_trace import WorkloadBudget


def test_portable_example_caps_whole_run_at_256_without_resetting_roles():
    profile = RackProfile.model_validate_json(
        (Path(__file__).parents[2] / "profiles/rack-example.json").read_text()
    )
    assert profile.profile_version == "role-budgets-v2"
    assert profile.base_url == "https://rack.example.com/v1"
    assert profile.model == "example/served-model"
    assert list((Path(__file__).parents[2] / "profiles").glob("rack-*.json")) == [
        Path(__file__).parents[2] / "profiles/rack-example.json"
    ]
    assert profile.max_generation_calls == 256
    assert profile.max_run_seconds == 900
    assert profile.timeout_seconds == 120
    assert [
        profile.output_budget(t) for t in ["event_claims", "consolidate", "answer", "rubric"]
    ] == [16384, 1024, 2048, 512]
    budget = WorkloadBudget(
        max_calls=profile.max_generation_calls, max_seconds=profile.max_run_seconds
    )
    for _ in range(256):
        budget.begin_call()
    with pytest.raises(RuntimeError, match="call budget"):
        budget.begin_call()
    assert budget.failed
    with pytest.raises(RuntimeError):
        budget.check()


def test_no_profile_can_make_budget_unlimited():
    with pytest.raises(ValueError):
        WorkloadBudget(max_calls=257)
    with pytest.raises(ValueError):
        WorkloadBudget(max_seconds=901)
    assert (
        RackProfile(
            base_url="http://localhost:8000/v1", model="example/served-model"
        ).max_generation_calls
        == 128
    )
