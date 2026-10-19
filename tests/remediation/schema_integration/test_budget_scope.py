"""Narrow scopes retain a single cumulative counter, lock and original deadline."""
import pytest
from gluomem_bench.bench.parallel_campaign import CampaignBudget, CampaignStopped
from gluomem_bench.bench import remediation_study as budget_scope


def test_scoped_debit_updates_the_original_counter_once():
    now = [100.0]
    original = CampaignBudget(max_calls=1000, max_seconds=1000, prior_calls=13,
                              prior_elapsed_seconds=17, clock=lambda: now[0])
    assert hasattr(budget_scope, "BudgetScope"), "bounded original-budget view missing"
    scope = budget_scope.BudgetScope(original, max_calls=3, max_seconds=20)
    assert scope.lock is original.lock and scope.deadline <= original.deadline
    with scope.lock:
        scope.calls += 1
    assert scope.calls == 1 and original.calls == 14
    scope.begin_call()
    assert scope.calls == 2 and original.calls == 15
    now[0] += 4
    assert scope.snapshot() == {"calls": 2, "elapsed_seconds": 4.0}
    assert original.snapshot()["new_calls"] == 2
    assert original.snapshot()["new_elapsed_seconds"] == 4.0


def test_scoped_limits_never_extend_or_reset_original_authority():
    now = [100.0]
    original = CampaignBudget(max_calls=10, max_seconds=100, clock=lambda: now[0])
    assert hasattr(budget_scope, "BudgetScope"), "bounded original-budget view missing"
    scope = budget_scope.BudgetScope(original, max_calls=2, max_seconds=10)
    scope.begin_call()
    scope.begin_call()
    with pytest.raises(CampaignStopped):
        scope.begin_call()
    assert original.calls == 2
    fresh = CampaignBudget(max_calls=10, max_seconds=100, clock=lambda: now[0])
    timed = budget_scope.BudgetScope(fresh, max_calls=2, max_seconds=10)
    now[0] += 10
    with pytest.raises(CampaignStopped):
        timed.check()
    assert fresh.failed
    with pytest.raises(ValueError):
        budget_scope.BudgetScope(CampaignBudget(max_calls=1, max_seconds=1), max_calls=2, max_seconds=10)
