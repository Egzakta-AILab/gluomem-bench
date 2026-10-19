import json
from pathlib import Path
import pytest
from gluomem_bench.bench import remediation_study as s
from gluomem_bench.bench.parallel_campaign import CampaignBudget, CampaignStopped, guard_stopped

class Hostile(BaseException):
    def __str__(self):
        raise AssertionError('exception text accessed')
    def __repr__(self):
        raise AssertionError('exception repr accessed')

def plan():
    return {'jobs': [], 'full_matrix': []}

def test_publish_preserves_actual_guard_cause_and_spend(tmp_path, monkeypatch):
    clock = [10.0]
    budget = CampaignBudget(max_calls=20, max_seconds=100, prior_calls=3,
        prior_elapsed_seconds=5, started_monotonic=0, clock=lambda: clock[0])
    budget.begin_call()
    error = guard_stopped(CampaignStopped('PRIVATE URL / message'), 'campaign-failed')
    def failed(*args):
        clock[0] = 11.0
        raise error
    monkeypatch.setattr(s, '_publish', failed)
    packet = {'output_root': str(tmp_path/'root')}
    with pytest.raises(CampaignStopped) as caught:
        s.publish(packet, plan(), {}, {}, {}, budget)
    assert caught.value is error
    receipt = json.loads((tmp_path/'root.revoked.json').read_bytes())
    assert receipt['failure']['category'] == 'execution-or-publication-failed'
    diagnostic = receipt['failure']['diagnostic']
    assert diagnostic['protocol'] == 'root-causal-diagnostic@v4'
    assert diagnostic['phase'] == 'publish'
    assert diagnostic['guard_reason'] == 'campaign-failed'
    assert diagnostic['origin']['code'] == 'runtime-guard'
    assert diagnostic['observed_monotonic'] == 11.0
    assert diagnostic['observed_call_ordinal'] == 4
    assert receipt['accounting'] == budget.snapshot()
    assert 'PRIVATE' not in json.dumps(receipt)


def test_cli_authored_failure_retains_cause_and_trusted_raise_site(tmp_path, monkeypatch, capsys):
    authority_file = tmp_path/'authority.json'
    authority_file.write_text('{}', encoding='utf8')
    import hashlib
    digest = hashlib.sha256(authority_file.read_bytes()).hexdigest()
    # Real CLI budget -> authority -> load path. No service/workload is entered.
    code = s.main(['budget', '--authority', str(authority_file), '--authority-sha256', digest])
    assert code == 2
    output = json.loads(capsys.readouterr().err)
    assert output['category'] == 'contract'  # Existing exact-type consumer unchanged.
    diagnostic = output['diagnostic']
    assert diagnostic['phase'] == 'setup'
    assert diagnostic['diagnostic']['type'] == 'ValueError'
    sites = diagnostic['sources']
    assert sites
    owned_file = Path(s.__file__)
    expected_hash = hashlib.sha256(owned_file.read_bytes()).hexdigest()
    assert any(site['source'] == 'gluomem_bench/bench/remediation_study.py'
               and site['sha256'] == expected_hash and site['line'] > 0 for site in sites)
    encoded = json.dumps(output)
    assert str(tmp_path) not in encoded and 'authority.json' not in encoded
    (tmp_path/'cli-failure.json').write_text(json.dumps(output), encoding='utf8')


def test_trusted_sources_reject_external_frames_and_wrong_inventory(tmp_path):
    import hashlib
    module_file = Path(s.__file__)
    manifest_file = tmp_path/'source.json'
    manifest_file.write_text(json.dumps({'gluomem_bench/bench/remediation_study.py': '0'*64,
                                        '../private.py': '0'*64}), encoding='utf8')
    packet = {'source_manifest': {'path': str(manifest_file),
        'sha256': hashlib.sha256(manifest_file.read_bytes()).hexdigest()}}
    try:
        s.load({'path': str(tmp_path/'PRIVATE-absent'), 'sha256': '0'*64})
    except BaseException as exc:
        diagnostic = s.root_failure_diagnostic(exc, phase='setup', packet=packet)
    assert diagnostic['sources'] == []
    assert str(tmp_path) not in json.dumps(diagnostic)
    assert module_file.is_file()


@pytest.fixture
def offline_execution(tmp_path, monkeypatch):
    from types import SimpleNamespace as N
    from gluomem_bench.bench import lme_first, upstream_scoring
    packet = {'output_root': str(tmp_path/'root'), 'worker': {}}
    candidate = tmp_path/'candidate.json'
    candidate.write_text(json.dumps(packet), encoding='utf8')
    approval = tmp_path/'approval.json'
    approval.write_text('{}', encoding='utf8')
    execution_plan = dict(plan(), native_constraints={'local_memory_mib': 12288},
        judge_source='unused', selected_systems=[])
    worker = {'history_commit_limits_mib': dict.fromkeys(s.SYSTEMS, 4096),
        'native_limits': {}, 'native_mode': 'process', 'max_workers': 1}
    budget = CampaignBudget(max_calls=20, max_seconds=100, clock=lambda: 10, started_monotonic=0)
    monkeypatch.setattr(s, 'validate_candidate', lambda *a, **kw: (packet, execution_plan, worker, {}))
    monkeypatch.setattr(s.native, 'validate_resource_contract', lambda *a, **kw: None)
    monkeypatch.setattr(s.native, 'verify_qualification_resources', lambda *a, **kw: None)
    monkeypatch.setattr(upstream_scoring, 'load_longmemeval_prompt', lambda *a: None)
    monkeypatch.setattr(s.native, 'build_input_cache', lambda *a: {})
    monkeypatch.setattr(s.native, 'native_records', lambda *a: [])
    monkeypatch.setattr(s.native, 'MemoryAdmission', lambda **kw: N(pressure_latched=False,
        release=lambda *a: None, snapshot=lambda: {'resident_jobs': 0}))
    class Broker:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(s.native, 'ContextBroker', Broker)
    monkeypatch.setattr(lme_first, 'run_isolated_jobs', lambda *a, **kw: {})
    monkeypatch.setattr(s.native, '_root_report', lambda *a: {})
    monkeypatch.setattr(s, 'publish', lambda *a: {'quality_complete': False, 'selected_quality_complete': False})
    return N(packet=packet, plan=execution_plan, worker=worker, budget=budget,
             candidate=candidate, approval=approval, broker=Broker, lme_first=lme_first)


@pytest.mark.parametrize('boundary,expected_phase', [
    ('resource', 'resource'), ('input-cache', 'input-cache'), ('broker-enter', 'worker'),
    ('run', 'run'), ('broker-exit', 'worker'), ('verify', 'verify')])
def test_execute_causal_receipt_is_actual_phase(tmp_path, monkeypatch, offline_execution, boundary, expected_phase):
    setup = offline_execution
    error = Hostile()
    def fail(*a, **kw):
        raise error
    if boundary == 'resource':
        monkeypatch.setattr(s.native, 'validate_resource_contract', fail)
    elif boundary == 'input-cache':
        monkeypatch.setattr(s.native, 'build_input_cache', fail)
    elif boundary == 'broker-enter':
        monkeypatch.setattr(setup.broker, '__enter__', fail)
    elif boundary == 'run':
        monkeypatch.setattr(setup.lme_first, 'run_isolated_jobs', fail)
    elif boundary == 'broker-exit':
        monkeypatch.setattr(setup.broker, '__exit__', fail)
    else:
        original = s.validate_candidate
        monkeypatch.setattr(s, 'validate_candidate', lambda *a, **kw:
            fail() if kw.get('post_exit') else original(*a, **kw))
    with pytest.raises(Hostile) as caught:
        s.execute(setup.candidate, setup.approval, budget=setup.budget)
    assert caught.value is error
    receipt = json.loads((tmp_path/'root.revoked.json').read_bytes())
    diagnostic = receipt['failure']['diagnostic']
    assert diagnostic['phase'] == expected_phase
    assert diagnostic['diagnostic']['type'] == 'Exception'
    assert 'Hostile' not in json.dumps(receipt)


def test_cli_execute_preserves_hostile_category_and_terminal_phase(tmp_path, monkeypatch, capsys, offline_execution):
    setup = offline_execution
    error = Hostile()
    def fail(*a, **kw):
        raise error
    monkeypatch.setattr(s, 'campaign_budget', lambda *a: setup.budget)
    monkeypatch.setattr(s.native, 'build_input_cache', fail)
    code = s.main(['execute', '--candidate', str(setup.candidate), '--approval', str(setup.approval),
                  '--authority', 'unused', '--authority-sha256', '0'*64])
    assert code == 2
    cli = json.loads(capsys.readouterr().err)
    root = json.loads((tmp_path/'root.revoked.json').read_bytes())
    assert cli['category'] == 'unknown'
    assert cli['diagnostic']['diagnostic']['type'] == 'Exception'
    assert cli['diagnostic']['phase'] == root['failure']['diagnostic']['phase'] == 'input-cache'
    assert 'Hostile' not in json.dumps(cli)


@pytest.mark.parametrize('boundary', ['diagnostic', 'revocation'])
def test_telemetry_failure_never_replaces_actual_exception(tmp_path, monkeypatch, boundary):
    error = Hostile()
    secondary = Hostile()
    budget = CampaignBudget(max_calls=5, max_seconds=100, started_monotonic=0, clock=lambda: 1)
    def original(*a, **kw):
        raise error
    def broken(*a, **kw):
        raise secondary
    monkeypatch.setattr(s, '_publish', original)
    if boundary == 'diagnostic':
        monkeypatch.setattr(s, 'root_failure_diagnostic', broken)
    else:
        monkeypatch.setattr(s, 'write_new', broken)
    with pytest.raises(Hostile) as caught:
        s.publish({'output_root': str(tmp_path/'root')}, plan(), {}, {}, {}, budget)
    assert caught.value is error
    if boundary == 'diagnostic':
        assert json.loads((tmp_path/'root.revoked.json').read_bytes())['quality_complete'] is False


def test_sources_distinguish_trusted_caller_from_unmapped_raise_site(tmp_path, monkeypatch):
    error = Hostile()
    def external_callback(*a):
        raise error
    monkeypatch.setattr(s, '_publish', external_callback)
    budget = CampaignBudget(max_calls=5, max_seconds=100, started_monotonic=0, clock=lambda: 1)
    with pytest.raises(Hostile):
        s.publish({'output_root': str(tmp_path/'root')}, plan(), {}, {}, {}, budget)
    diagnostic = json.loads((tmp_path/'root.revoked.json').read_bytes())['failure']['diagnostic']
    assert diagnostic['sources']
    assert diagnostic['sources'][0]['is_innermost'] is False


def test_broker_teardown_failure_keeps_terminal_and_run_cause(tmp_path, monkeypatch, offline_execution):
    setup = offline_execution
    run_error, terminal_error = Hostile(), OSError(10054, 'PRIVATE reset endpoint')
    def run(*a, **kw):
        raise run_error
    def teardown(*a):
        raise terminal_error
    monkeypatch.setattr(setup.lme_first, 'run_isolated_jobs', run)
    monkeypatch.setattr(setup.broker, '__exit__', teardown)
    with pytest.raises(OSError) as caught:
        s.execute(setup.candidate, setup.approval, budget=setup.budget)
    assert caught.value is terminal_error
    assert caught.value.__context__ is run_error
    diagnostic = json.loads((tmp_path/'root.revoked.json').read_bytes())['failure']['diagnostic']
    assert diagnostic['phase'] == 'worker'
    assert diagnostic['origin']['errno'] == 10054
    assert diagnostic['origin']['code'] == 'os-error'
    assert len(diagnostic['sources']) == 2
    assert 'PRIVATE' not in json.dumps(diagnostic)


def test_existing_native_guard_receipt_survives_without_reclassifying_unknown():
    from types import SimpleNamespace as N
    error = guard_stopped(Hostile(), 'job-call-cap')
    context = N(budget=N(clock=lambda: 4, deadline=100), deadline=10, calls=3,
                job=N(job_id='PRIVATE-job-identity'))
    error.guard_refusal = s.native.guard_refusal(context, 'begin', error)
    wrapper = Hostile()
    wrapper.__cause__ = error
    error.__context__ = wrapper  # Bounded and cycle-safe.
    diagnostic = s.root_failure_diagnostic(wrapper, phase='run')
    assert diagnostic['diagnostic']['type'] == 'Exception'
    assert diagnostic['origin']['code'] == 'operation-failed'
    assert diagnostic['guard_reason'] == 'job-call-cap'
    refusal = diagnostic['origin']['guard_refusal']
    assert refusal['source_operation'] == 'begin'
    assert refusal['call_ordinal'] == 3 and refusal['clock_monotonic'] == 4
    assert diagnostic['observed_monotonic'] is None
    assert diagnostic['observed_call_ordinal'] is None
    assert 'PRIVATE' not in json.dumps(diagnostic)


def test_diagnostic_work_is_charged_to_same_original_clock(tmp_path, monkeypatch):
    clock = [10]
    budget = CampaignBudget(max_calls=20, max_seconds=100, prior_calls=7,
        prior_elapsed_seconds=5, started_monotonic=0, clock=lambda: clock[0])
    budget.begin_call()
    error = Hostile()
    def fail(*a):
        raise error
    original = s.root_failure_diagnostic
    def collect(*a, **kw):
        clock[0] += 2
        return original(*a, **kw)
    original_write = s.write_new
    def write(*a, **kw):
        result = original_write(*a, **kw)
        clock[0] += 1  # Original clock continues through durable publication.
        return result
    monkeypatch.setattr(s, '_publish', fail)
    monkeypatch.setattr(s, 'root_failure_diagnostic', collect)
    monkeypatch.setattr(s, 'write_new', write)
    with pytest.raises(Hostile) as caught:
        s.publish({'output_root': str(tmp_path/'root')}, plan(), {}, {}, {}, budget)
    assert caught.value is error
    receipt = json.loads((tmp_path/'root.revoked.json').read_bytes())
    assert receipt['accounting']['calls'] == 8
    assert receipt['accounting']['elapsed_seconds'] == 17
    assert budget.snapshot()['elapsed_seconds'] == 18
    assert (budget.started, budget.deadline, budget.prior_calls) == (0, 95, 7)


def test_actual_cli_entrypoint_authored_failure_no_workloads(tmp_path, monkeypatch, capsys):
    import hashlib
    import runpy
    import sys
    authority_file = tmp_path/'authored-invalid-authority.json'
    authority_file.write_text('{}', encoding='utf8')
    digest = hashlib.sha256(authority_file.read_bytes()).hexdigest()
    monkeypatch.setattr(sys, 'argv', [s.__file__, 'budget', '--authority', str(authority_file),
                                    '--authority-sha256', digest])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(s.__file__, run_name='__main__')
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ''
    output = json.loads(captured.err)
    assert output['category'] == 'contract'
    diagnostic = output['diagnostic']
    assert diagnostic['diagnostic']['type'] == 'ValueError'
    assert diagnostic['sources'][0]['is_innermost'] is True
    assert diagnostic['sources'][0]['sha256'] == hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest()
    (tmp_path/'actual-cli-stderr.json').write_text(captured.err, encoding='utf8')


def test_unknown_origin_not_inferred_from_late_budget_flags():
    budget = CampaignBudget(max_calls=5, max_seconds=100, started_monotonic=0, clock=lambda: 1)
    budget.failed = True
    diagnostic = s.root_failure_diagnostic(Hostile(), phase='PRIVATE', budget=budget)
    assert diagnostic['phase'] == 'unknown'
    assert diagnostic['guard_reason'] == 'unknown'
    assert diagnostic['origin']['code'] == 'operation-failed'
    assert diagnostic['sources'] == []
    assert 'PRIVATE' not in json.dumps(diagnostic)


def test_causal_traversal_is_bounded_for_long_unknown_chain():
    head = Hostile()
    cursor = head
    for _ in range(30):
        cursor.__cause__ = Hostile()
        cursor = cursor.__cause__
    cursor.guard_reason = 'job-deadline'
    diagnostic = s.root_failure_diagnostic(head, phase='run')
    assert diagnostic['guard_reason'] == 'unknown'
    assert diagnostic['sources'] == []


@pytest.mark.parametrize('bad_hash', [None, '', 'not-a-hash'])
def test_declared_manifest_requires_actual_sha256(tmp_path, bad_hash):
    import hashlib
    manifest_file = tmp_path/'source.json'
    manifest_file.write_text(json.dumps({'gluomem_bench/bench/remediation_study.py': bad_hash}), encoding='utf8')
    packet = {'source_manifest': {'path': str(manifest_file),
        'sha256': hashlib.sha256(manifest_file.read_bytes()).hexdigest()}}
    try:
        s.load({'path': str(tmp_path/'absent'), 'sha256': '0'*64})
    except BaseException as exc:
        diagnostic = s.root_failure_diagnostic(exc, phase='setup', packet=packet)
    assert diagnostic['sources'] == []
