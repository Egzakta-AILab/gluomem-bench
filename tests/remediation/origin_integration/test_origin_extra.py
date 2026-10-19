from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
import json
import pytest
from gluomem_bench.bench import lme_first as lme, parallel_campaign as pc, rack_trace as rt

@pytest.mark.parametrize('method,reason', [(m,r) for r in ('job-failed','job-deadline') for m in ('check','remaining','begin_call')] + [('begin_call','job-call-cap')])
def test_workload_origin(method, reason):
    now = [0.0]
    b = rt.WorkloadBudget(max_calls=1, max_seconds=1, clock=lambda: now[0])
    if reason == 'job-failed': b.failed = True
    elif reason == 'job-deadline': now[0] = 2
    else:
        b.begin_call()
    with pytest.raises(RuntimeError) as caught: getattr(b, method)()
    assert type(caught.value) is RuntimeError
    assert caught.value.__dict__.get('guard_reason') == reason
    assert b.calls == (1 if reason == 'job-call-cap' else 0)


def test_campaign_cap_origin_no_spend():
    b = pc.CampaignBudget(max_calls=1, max_seconds=10, clock=lambda: 0)
    b.begin_call()
    with pytest.raises(pc.CampaignStopped) as caught: b.begin_call()
    assert caught.value.__dict__.get('guard_reason') == 'campaign-call-cap'
    assert b.calls == 1 and b.failed

@pytest.mark.parametrize('kind', ['plain', 'fair'])
@pytest.mark.parametrize('reason', ['cancelled', 'job-deadline'])
def test_limiter_origin(kind, reason):
    limiter = (pc.RequestLimiter if kind == 'plain' else lme.FairRequestLimiter)(max_in_flight=1, max_in_flight_tokens=2)
    cancel = Event()
    if reason == 'cancelled': cancel.set()
    with pytest.raises(RuntimeError) as caught:
        with limiter.request(1, deadline=0 if reason == 'job-deadline' else 10, cancel=cancel, clock=lambda: 0):
            pytest.fail('refused lease admitted')
    assert caught.value.__dict__.get('guard_reason') == reason
    assert limiter.snapshot()['in_flight'] == 0

@pytest.mark.parametrize('fault', ['no-lease', 'duplicate', 'nested'])
def test_actual_inner_accounting_origin(tmp_path, fault):
    errors = []
    jobs = [dict(job_id='j', history_id='h', system='mem0', output=str(tmp_path/'leaf'), call_budget=2, seconds_budget=5, memory_mib=1)]
    def work(job, context):
        try:
            if fault == 'no-lease': context.begin_call()
            else:
                with context.lease(1):
                    if fault == 'nested':
                        with context.lease(1): pass
                    else:
                        context.begin_call()
                        context.begin_call()
        except lme.DispatchStopped as error:
            errors.append(error)
            raise
    result = lme.run_isolated_jobs(jobs, out=tmp_path/'coord', run_job=work, max_calls=2, max_seconds=10, max_workers=1, native_limits={'mem0':1}, memory_limit_mib=1, max_in_flight_tokens=2)
    assert errors and errors[0].__dict__.get('guard_reason') == 'accounting-invalid'
    assert result['accounting']['calls'] == (1 if fault == 'duplicate' else 0)


def test_safe_causal_receipts_no_dynamic_text():
    from gluomem_bench.bench.lme_first_execute import RemoteContextDispatchStopped
    from gluomem_bench.bench.rack import failure_origin
    receipt = dict(protocol='native-job-guard-refusal@v1', reason='job-call-cap', source_operation='begin', clock_monotonic=10.0, job_deadline_monotonic=12.0, campaign_deadline_monotonic=20.0, call_ordinal=2, job_sha256='a'*64, private='SECRET')
    inner = RemoteContextDispatchStopped(receipt)
    inner.native_diagnostic = dict(stage='cognify', clock='operation_deadline', causes=[dict(error_class='TimeoutError', errno=10060, origin=dict(source_sha256='b'*64, line=12), body='SECRET')], body='SECRET')
    outer = RuntimeError('SECRET')
    outer.__cause__ = inner
    inner.__context__ = outer
    result = rt.failure_diagnostic(outer)
    assert result['guard_refusal'] == {k:v for k,v in receipt.items() if k != 'private'}
    assert result['native_diagnostic']['stage'] == 'cognify'
    assert 'SECRET' not in json.dumps(result)
    assert failure_origin(outer)['guard_refusal'] == result['guard_refusal']
    assert failure_origin(outer)['native_diagnostic'] == result['native_diagnostic']

@pytest.mark.parametrize('value', [None, 'SECRET', {'protocol':'wrong', 'reason':'cancelled'}, {'protocol':'native-job-guard-refusal@v1', 'reason': ['SECRET'], 'call_ordinal':True, 'clock_monotonic':float('nan'), 'job_sha256':'SECRET', 'source_operation':'SECRET'}])
def test_safe_receipt_validation(value):
    error = RuntimeError('SECRET')
    error.guard_refusal = value
    result = rt.failure_diagnostic(error)
    assert 'SECRET' not in json.dumps(result)
    if type(value) is dict and value.get('protocol') == 'native-job-guard-refusal@v1':
        row = result['guard_refusal']
        assert row['reason'] == row['source_operation'] == 'unknown'
        assert row['call_ordinal'] is None and row['clock_monotonic'] is None and row['job_sha256'] is None
    else: assert 'guard_refusal' not in result


@pytest.mark.parametrize('fault', ['wrapped', 'diagnostic-write'])
def test_actual_racklm_receipt_and_cause(tmp_path, monkeypatch, fault):
    import dspy
    import httpx
    from gluomem_bench.bench.rack import RackClient, RackProfile
    from gluomem_bench.bench.lme_first_execute import RemoteContextDispatchStopped
    inner = RemoteContextDispatchStopped(dict(protocol='native-job-guard-refusal@v1', reason='job-call-cap', source_operation='begin'))
    inner.native_diagnostic = dict(stage='search', clock='operation_deadline', causes=[])
    error = RuntimeError('SECRET')
    error.__cause__ = inner
    def fail(*args, **kwargs): raise error
    monkeypatch.setattr(dspy.LM, '__call__', fail)
    profile = RackProfile(base_url='http://fixture.invalid/v1', model='authored/model')
    client = RackClient(profile, receipt=tmp_path/'calls.jsonl', transport=httpx.MockTransport(lambda _: httpx.Response(200,json={'count':1})))
    budget = rt.WorkloadBudget()
    trace = tmp_path/'raw.jsonl'
    lm = rt.RackLM('openai/'+profile.model, client=client, task='event_claims', trace=trace, budget=budget, api_base=profile.base_url, max_tokens=512, temperature=0, cache=False, num_retries=0)
    if fault == 'diagnostic-write':
        monkeypatch.setattr(rt, 'failure_diagnostic', lambda _: (_ for _ in ()).throw(OSError('SECRET')))
        from pathlib import Path
        original = Path.open
        def refuse(path, *args, **kwargs):
            if path == trace: raise OSError('SECRET')
            return original(path,*args,**kwargs)
        monkeypatch.setattr(Path, 'open', refuse)
    try:
        with pytest.raises(RuntimeError) as caught: lm(messages=[{'role':'user','content':'authored'}])
        assert caught.value is error and error.__cause__ is inner
        assert budget.calls == 1 and budget.failed
        if fault == 'wrapped':
            row = json.loads(trace.read_text())['error']
            assert row['guard_refusal']['reason'] == 'job-call-cap'
            assert row['native_diagnostic']['stage'] == 'search'
            assert 'SECRET' not in trace.read_text()
    finally: client.close()


def test_job_lease_preserves_hostile_error_without_formatting():
    class SecretFailure(Exception):
        def __str__(self): raise AssertionError('exception formatted')
        def __repr__(self): raise AssertionError('exception formatted')
    error = SecretFailure()
    budget = pc.CampaignBudget(max_calls=1, max_seconds=10, clock=lambda:0)
    job = SimpleNamespace(job_id='j', seconds_budget=5, call_budget=1)
    context = pc.JobContext(job, budget, pc.RequestLimiter(max_in_flight=1,max_in_flight_tokens=1))
    with pytest.raises(SecretFailure) as caught:
        with context.lease(1): raise error
    assert caught.value is error and context.failed and not budget.failed
    assert context.failure == 'local request failed'
    assert context.calls == budget.calls == 0


def test_unknown_class_never_formats_or_names():
    class SECRET(RuntimeError):
        def __str__(self): raise AssertionError('formatted')
        def __repr__(self): raise AssertionError('formatted')
    result = rt.failure_diagnostic(SECRET())
    assert result == {'type':'Exception', 'message':'operation_failed'}
