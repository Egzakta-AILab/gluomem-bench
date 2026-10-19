"""Parent-owned native guard origin gate; no SDK/model/encoder or listener."""
from contextlib import nullcontext
from threading import Event, RLock
from types import SimpleNamespace
import pytest
from gluomem_bench.bench import lme_first_execute as x
from gluomem_bench.bench.parallel_campaign import CampaignBudget, JobContext
from test_broker_origin import Connection


@pytest.mark.parametrize('op', ['check', 'remaining', 'lease', 'begin'])
@pytest.mark.parametrize('reason', ['cancelled', 'campaign-failed', 'campaign-deadline',
                                  'job-deadline', 'job-failed'])
def test_native_guard_reports_actual_reason_at_boundary(op, reason):
    _assert_native_reason(op, reason)


def test_native_call_cap_refusal_reports_actual_reason():
    _assert_native_reason('begin', 'job-call-cap')


def _assert_native_reason(op, reason):
    budget = CampaignBudget(max_calls=10, max_seconds=100, clock=lambda: 10.0)
    job = SimpleNamespace(job_id='authored-native', seconds_budget=20, call_budget=2)
    context = JobContext(job, budget,
        SimpleNamespace(request=lambda *args, **kwargs: nullcontext()), cancel=Event())
    if reason == 'cancelled': context.cancel.set()
    elif reason == 'campaign-failed': budget.failed = True
    elif reason == 'campaign-deadline': budget.deadline = 5.0
    elif reason == 'job-deadline': context.deadline = 5.0
    elif reason == 'job-failed': context.failed = True
    elif reason == 'job-call-cap': context.calls = 2
    prior_calls = context.calls
    conn = Connection(['handle', (op, 1), ('close', None)])
    broker = object.__new__(x.ContextBroker)
    broker.contexts, broker.lock, broker.clients = {'handle': context}, RLock(), {conn}
    broker._serve(conn)
    report = next(row for row in conn.sent if not row['ok'])['guard_refusal']
    assert report['reason'] == reason
    assert report['source_operation'] == op
    assert report['call_ordinal'] == prior_calls == context.calls
    if reason in ('job-deadline', 'job-failed', 'job-call-cap', 'cancelled'):
        assert not budget.failed
