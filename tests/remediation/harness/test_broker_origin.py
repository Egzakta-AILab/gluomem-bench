"""Accounting-only in-memory persistent connection fixtures, no listener."""
from collections import deque
from types import SimpleNamespace
from threading import RLock
import hashlib
import json
import pytest
from gluomem_bench.bench import lme_first_execute as x


class Connection:
    def __init__(self, incoming):
        self.incoming, self.sent, self.closed = deque(incoming), [], False
    def recv(self):
        return self.incoming.popleft()
    def send(self, value):
        self.sent.append(value)
    def close(self):
        self.closed = True


def refuse(reason):
    error = RuntimeError('PRIVATE native error body')
    if reason is not None:
        error.guard_reason = reason
    return error


@pytest.mark.parametrize('op', ['check', 'remaining', 'lease', 'begin', 'release'])
@pytest.mark.parametrize('reason', ['job-deadline', 'job-failed', 'campaign-failed',
    'cancelled', 'job-call-cap', None])
def test_actual_broker_boundary_carries_origin_without_exception_echo(op, reason):
    error = refuse(reason)
    def fail(*args):
        raise error
    class Lease:
        def __enter__(self):
            if op == 'lease':
                fail()
        def __exit__(self, *args):
            if op == 'release':
                fail()
    context = SimpleNamespace(
        check=fail, remaining=fail, begin_call=fail, lease=lambda value: Lease(),
        budget=SimpleNamespace(clock=lambda: 10.5, deadline=100.0, failed=False),
        deadline=20.0, calls=7, job=SimpleNamespace(job_id='private-job'), failed=False)
    incoming = ['handle'] + ([('lease', 12)] if op == 'release' else []) + [(op, None), ('close', None)]
    conn = Connection(incoming)
    broker = object.__new__(x.ContextBroker)
    broker.contexts, broker.lock, broker.clients = {'handle': context}, RLock(), {conn}
    broker._serve(conn)
    reply = next(row for row in conn.sent if not row['ok'])
    receipt = reply['guard_refusal']
    assert receipt == {
        'protocol': 'native-job-guard-refusal@v1', 'reason': reason or 'unknown',
        'source_operation': op, 'clock_monotonic': 10.5,
        'job_deadline_monotonic': 20.0, 'campaign_deadline_monotonic': 100.0,
        'call_ordinal': 7, 'job_sha256': hashlib.sha256(b'private-job').hexdigest()}
    assert 'PRIVATE' not in json.dumps(reply)
    remote = x.RemoteContext([], b'', 'handle', context.job)
    wire = Connection([reply])
    remote._acquire = lambda: wire
    remote._return = lambda conn: None
    with pytest.raises(x.DispatchStopped) as failure:
        remote._rpc(op)
    assert type(failure.value) is x.RemoteContextDispatchStopped
    assert failure.value.guard_refusal == receipt
    assert context.calls == 7 and not context.budget.failed
    assert conn.closed

@pytest.mark.parametrize('commands', [
    [('release', None)], [('lease', 1), ('lease', 1)], [('unknown-op', None)],
])
def test_broker_accounting_invariant_has_origin_and_retains_global_latch(commands):
    from contextlib import nullcontext
    context = SimpleNamespace(lease=lambda value: nullcontext(),
        budget=SimpleNamespace(clock=lambda: 10.0, deadline=30.0, failed=False),
        deadline=20.0, calls=3, job=SimpleNamespace(job_id='authored'), failed=False)
    conn = Connection(['handle', *commands, ('close', None)])
    broker = object.__new__(x.ContextBroker)
    broker.contexts, broker.lock, broker.clients = {'handle': context}, RLock(), {conn}
    broker._serve(conn)
    report = next(row for row in conn.sent if not row['ok'])['guard_refusal']
    assert report['reason'] == 'accounting-invalid'
    assert context.budget.failed and context.calls == 3


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -1, True, 'PRIVATE'])
def test_refusal_numbers_and_reason_are_validated_without_formatting(invalid):
    context = SimpleNamespace(budget=SimpleNamespace(clock=lambda: invalid, deadline=invalid),
        deadline=invalid, calls=invalid, job=SimpleNamespace(job_id=object()))
    error = refuse(invalid)
    report = x.guard_refusal(context, 'PRIVATE', error)
    assert report['reason'] == 'unknown' and report['source_operation'] == 'unknown'
    assert all(report[field] is None for field in ('clock_monotonic',
        'job_deadline_monotonic', 'campaign_deadline_monotonic', 'call_ordinal', 'job_sha256'))
    assert 'PRIVATE' not in json.dumps(report)
