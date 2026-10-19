"""Authored offline contracts; no campaign dispatch or real endpoints."""
import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

from gluomem_bench.bench.parallel_campaign import CampaignStopped

OWN = Path(__file__).resolve().parents[2]


def study():
    return importlib.import_module('gluomem_bench.bench.remediation_study')


def put(path, data):
    path.write_bytes(json.dumps(data, sort_keys=True).encode())
    import hashlib
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def specification(tmp_path):
    from gluomem_bench.bench.rack import RackProfile
    ids = [f'authored-{i:03}' for i in range(500)]
    profile = RackProfile(base_url='http://authored.invalid/v1', model='authored/qwen',
                          max_generation_calls=4, max_run_seconds=30).model_dump(mode='json')
    base = {'sample_ids': ids, 'profile': profile, 'config_overrides': {},
            'dataset': {'name': 'longmemeval_s', 'path': str(tmp_path/'data.json'), 'sha256': '0'*64},
            'embedding_url': 'http://encoder.invalid/v1', 'embedding_recipe': {},
            'judge_source': str(tmp_path/'judge.py'), 'native_constraints': {'local_memory_mib': 12288},
            'jobs': [{'job_id': f'old-{i}-{s}', 'sample_id': sid, 'system': s}
                     for i, sid in enumerate(ids) for s in ('gluomem','mem0','cognee','no_memory')]}
    spec = {'protocol': 'lme-remediation-study-spec@v1', 'study_id': 'authored-study',
            'base_plan': put(tmp_path/'base.json', base),
            'profile_file': put(tmp_path/'profile.json', profile),
            'stage': 'development', 'selected_ids': ids[:1], 'selected_systems': ['no_memory'],
            'changed_condition': 'authored fixture operating point, not an old retry',
            'methods': {s: {'kind': 'native', 'version': 'authored-v1'}
                        for s in ('gluomem','mem0','cognee','no_memory')},
            'config_overrides': {}, 'experimental_overrides': {}, 'local_memory_mib': 12288}
    return spec


def test_prepare_is_explicit_new_experiment_and_preserves_full_target(tmp_path, specification):
    s = study()
    budget = s.campaign_budget(clock=lambda: 10)
    assert budget.calls == 141363
    assert budget.remaining() == 138408
    binding = s.prepare(put(tmp_path/'spec.json', specification), tmp_path/'plan.json', budget=budget)
    plan = s.load(binding)
    assert len(plan['sample_ids']) == 500
    assert len(plan['full_matrix']) == 2000
    assert len(plan['jobs']) == 1
    assert plan['full_scope'] is False
    assert plan['jobs'][0]['job_id'].startswith('authored-study--')
    assert all(not r['job_id'].startswith('old-') for r in plan['full_matrix'])
    assert s.validate_plan(plan) == plan
    forged = copy.deepcopy(plan)
    forged['profile']['temperature'] = 1
    with pytest.raises(ValueError):
        s.validate_plan(forged)


def test_prepare_forwards_bound_remediation_metadata_without_sdk_patches(tmp_path, specification):
    s = study()
    metadata = {'cognee_clocks': {'request_seconds': 600, 'cleanup_reserve_seconds': 12},
                'declared_condition': {'native_or_adapted': 'adapted', 'version': 'authored-v1'}}
    specification['remediation'] = metadata
    binding = put(tmp_path/'spec.json', specification)
    plan = s.assemble(binding)
    assert plan['remediation'] == metadata
    assert plan['experimental_overrides'] == {}
    forged = copy.deepcopy(plan)
    forged['remediation']['cognee_clocks']['request_seconds'] = 601
    with pytest.raises(ValueError):
        s.validate_plan(forged)
    specification['remediation'] = ['not-a-typed-object']
    with pytest.raises(ValueError):
        s.assemble(put(tmp_path/'invalid-spec.json', specification))


def test_freeze_binds_source_profile_runtime_and_refuses_mutation(tmp_path, specification, monkeypatch):
    s = study()
    source = tmp_path/'private-source'
    source.mkdir()
    producer = source/'producer.py'
    producer.write_bytes(b'# authored source closure\n')
    monkeypatch.setattr(s, 'SOURCE', source)
    budget = s.campaign_budget()
    plan = s.prepare(put(tmp_path/'spec.json', specification), tmp_path/'plan.json', budget=budget)
    runtime = tmp_path/'runtime.py'
    runtime.write_bytes(b'# authored installed runtime\n')
    template = {'protocol': s.native.EXECUTION_PROTOCOL, 'files': [dict(s.native.bound(runtime),
                snapshot=str(runtime), worker_check=True)], 'source_root': str(tmp_path/'old-source'),
                'native_mode': 'process', 'history_commit_limits_mib': dict.fromkeys(s.SYSTEMS, 4096),
                'host_reserve_mib': 8192, 'native_limits': dict.fromkeys(s.SYSTEMS, 1), 'max_workers': 1,
                'resource_protocol': s.native.RESOURCE_PROTOCOL,
                'qualification': put(tmp_path/'qualification.json', {'scope': 'authored-unit-only'}),
                'cognee_python': sys.executable}
    candidate = s.freeze(plan, put(tmp_path/'template.json', template), tmp_path/'run',
                         tmp_path/'frozen/candidate.json', budget=budget)
    packet, saved_plan, worker, _ = s.validate_candidate(candidate['path'], budget=budget)
    assert saved_plan == s.load(plan)
    assert packet['execution_authorized'] is False
    assert not (tmp_path/'run').exists()
    closure = {Path(r['path']): r for r in worker['files']}
    assert producer in closure and runtime in closure
    assert closure[producer]['worker_check'] is True
    assert Path(closure[producer]['snapshot']).read_bytes() == producer.read_bytes()
    producer.write_bytes(b'# changed deferred code\n')
    with pytest.raises(ValueError, match='binding'):
        s.validate_candidate(candidate['path'], budget=budget)


def test_budget_refuses_before_any_freeze_setup(tmp_path):
    s = study()
    clock = [0]
    budget = s.campaign_budget(clock=lambda: clock[0])
    clock[0] = 138408
    with pytest.raises((OSError, ValueError, CampaignStopped)):
        s.freeze({}, {}, tmp_path/'run', tmp_path/'must-not-exist/candidate.json', budget=budget)
    assert not (tmp_path/'must-not-exist').exists()


def test_full_matrix_accounting_retains_failed_and_unrun_null_qa(tmp_path, specification):
    s = study()
    specification.update(stage='full', selected_ids=s.load(specification['base_plan'])['sample_ids'],
                         selected_systems=list(s.SYSTEMS))
    plan = s.assemble(put(tmp_path/'spec.json', specification))
    records = [dict(r, output=str(tmp_path/'jobs'/r['job_id']), memory_mib=1) for r in plan['jobs']]
    from gluomem_bench.bench.lme_first import run_isolated_jobs, verify_journal
    from gluomem_bench.bench.lme_first_continue import NewAttemptBudget
    budget = s.campaign_budget()
    view = NewAttemptBudget(budget)
    def run(job, context):
        with context.lease(10):
            context.begin_call()
        raise ValueError('authored local failure')
    report = run_isolated_jobs(records, out=tmp_path/'coordination', run_job=run,
        max_calls=view.max_calls, max_seconds=view.max_seconds, budget=view, max_workers=4,
        native_limits=dict.fromkeys(s.SYSTEMS, 4), memory_limit_mib=4, max_in_flight_tokens=39321600)
    replay = verify_journal(tmp_path/'coordination', records)
    assert replay == report
    matrix = s.coverage(plan, report['jobs'])
    assert len(matrix) == 2000
    assert all(r['status'] == 'failed' and r['qa_score'] is None for r in matrix)
    assert budget.calls == 143363
    assert report['accounting']['calls'] == 2000
    assert len(s.coverage(plan, [])) == 2000
    assert all(r['status'] == 'unrun' for r in s.coverage(plan, []))


def test_cli_executes_owned_worker_reader_scorer_and_verifies_root(tmp_path, specification, monkeypatch, capsys):
    """Authored admission fixture; real worker/RPC/OS monitor/scorer, no native SDK construction."""
    import subprocess
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe
    original_popen = subprocess.Popen
    def isolated_popen(command, *args, **kwargs):
        assert command[1:3] == ['-B', '-I']
        assert Path(command[3]).name == 'lme_first_execute.py'
        command = command[:3] + [str(Path(__file__).with_name('offline_worker.py'))] + command[3:]
        return original_popen(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', isolated_popen)
    s = study()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def reply(self, data):
            raw = json.dumps(data).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def do_GET(self):
            assert self.path == '/v1/models'
            self.reply({'data': [{'id': 'authored/qwen', 'max_model_len': 131072}]})
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/tokenize':
                self.reply({'count': 10})
                return
            assert self.path == '/v1/chat/completions'
            answer = 'cobalt telescope' if data['messages'][0]['role'] == 'system' else 'yes'
            self.reply({'model': 'authored/qwen', 'choices': [{'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': answer}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}})
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = s.load(specification['base_plan'])
        dataset = [{'question_id': sid, 'question': 'What does Rhea own?', 'answer': 'cobalt telescope',
            'question_type': 'single-session-user', 'question_date': '2024/02/02 (Fri) 12:00',
            'answer_session_ids': ['answer_private'], 'haystack_dates': ['2024/02/01 (Thu) 12:00'],
            'haystack_session_ids': ['answer_private'],
            'haystack_sessions': [[{'role': 'user', 'content': ' Rhea owns a cobalt telescope. '},
                                    {'role': 'assistant', 'content': ''}]]} for sid in base['sample_ids']]
        data_binding = put(tmp_path/'data.json', dataset)
        base['dataset'].update(data_binding)
        original = json.loads((OWN/'baseline-binding.json').read_bytes())
        old = json.loads(Path(original['baseline_candidate']).read_bytes())
        old_plan = s.load(old['ledger'])['plan_semantics']
        base['judge_source'] = old_plan['judge_source']
        base['embedding_recipe'] = EmbeddingRecipe().as_dict()
        base['config_overrides'] = {'benchmark.datasets.longmemeval_s': {'url': 'https://authored.invalid/source.json',
            'file': 'data.json', 'sha256': data_binding['sha256']}}
        recipe = base['embedding_recipe']
        base['config_overrides'].update({'models.minilm.provider_model': recipe['model'],
            'models.minilm.revision': recipe['revision'], 'models.minilm.dim': recipe['dim'],
            'storage.vector_dim': recipe['dim']})
        profile = base['profile']
        profile['base_url'] = f'http://127.0.0.1:{server.server_port}/v1'
        specification['base_plan'] = put(tmp_path/'base.json', base)
        specification['profile_file'] = put(tmp_path/'profile.json', profile)
        plan = s.assemble(put(tmp_path/'spec.json', specification))
        root = tmp_path/'run'
        worker = {'protocol': s.native.EXECUTION_PROTOCOL, 'native_mode': 'process',
            'files': [dict(s.native.bound(s.native.__file__), worker_check=True)],
            'plan': put(tmp_path/'plan.json', plan), 'output_root': str(root),
            'host_reserve_mib': 8192, 'history_commit_limits_mib': dict.fromkeys(s.SYSTEMS, 4096),
            'native_limits': dict.fromkeys(s.SYSTEMS, 1), 'max_workers': 1,
            'resource_protocol': s.native.RESOURCE_PROTOCOL, 'cognee_python': sys.executable}
        qualification = s.load(s.load(old['worker'])['qualification'])
        packet = {'protocol': s.PROTOCOL, 'output_root': str(root),
            'worker': put(tmp_path/'worker.json', worker), 'plan': worker['plan']}
        (tmp_path/'sealed').mkdir()
        put(tmp_path/'sealed/candidate.json', packet)
        put(tmp_path/'approval.json', {'scope': 'authored-test-only'})
        monkeypatch.setattr(s, 'validate_candidate', lambda *a, **kw: (packet, plan, worker, qualification))
        assert s.main(['execute', '--candidate', str(tmp_path/'sealed/candidate.json'),
                       '--approval', str(tmp_path/'approval.json')]) == 0, capsys.readouterr()
        result = json.loads((root/'root-status.json').read_bytes())
        assert result['quality_complete'] is False
        assert result['selected_quality_complete'] is True
        assert len(result['full_matrix']) == 2000
        assert sum(r['status'] == 'complete' for r in result['full_matrix']) == 1
        assert result['accounting']['calls'] == 141365
        assert result['native']['accounting']['calls'] == 2
        outcome = json.loads(next((root/'workers').glob('*/outcome.json')).read_bytes())
        assert outcome['status'] == 'native-closed-verified'
        assert outcome['resource']['commit_limit_mib'] == 4096
        for mutation in ('drop', 'duplicate', 'promote-unrun', 'prior-reset', 'calls', 'deadline'):
            forged = copy.deepcopy(result)
            if mutation == 'drop':
                forged['full_matrix'].pop()
            elif mutation == 'duplicate':
                forged['full_matrix'][-1] = forged['full_matrix'][0]
            elif mutation == 'promote-unrun':
                forged['full_matrix'][0].update(status='complete', qa_score=1)
            elif mutation == 'prior-reset':
                forged['accounting']['prior_calls'] = 0
            elif mutation == 'calls':
                forged['accounting']['calls'] += 1
            else:
                forged['clock_deadline'] += 1
            with pytest.raises(ValueError):
                s.verify_report(packet, plan, worker, qualification, forged)
        before_verify = {str(p): s.native.bound(p)['sha256'] for p in root.rglob('*') if p.is_file()}
        offline = json.loads(next((root/'workers').glob('*/offline-sockets.json')).read_bytes())
        assert offline['denied_connections'] == 0
        assert offline['denied_databases'] == 0
        assert offline['allowed_fixture_connections'] > 0
        assert s.main(['verify', '--candidate', str(tmp_path/'sealed/candidate.json'),
                       '--out', str(tmp_path/'external-verification.json')]) == 0
        assert json.loads((tmp_path/'external-verification.json').read_bytes())['verified'] is True
        assert {str(p): s.native.bound(p)['sha256'] for p in root.rglob('*') if p.is_file()} == before_verify
        put(tmp_path/'run.revoked.json', {'quality_complete': False})
        with pytest.raises(ValueError, match='revocation'):
            s.verify_report(packet, plan, worker, qualification, result)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize('boundary', ['native-write', 'stage-write', 'verify', 'stage-read', 'replace',
                                      'root-read', 'receipt-write', 'receipt-read', 'deadline'])
def test_publication_fault_always_has_external_revocation(tmp_path, specification, monkeypatch, boundary):
    s = study()
    plan = s.assemble(put(tmp_path/'spec.json', specification))
    root = tmp_path/'run'
    root.mkdir()
    packet = {'output_root': str(root)}
    clock = [0]
    budget = s.campaign_budget(clock=lambda: clock[0])
    native_report = {'quality_complete': False, 'jobs': []}
    original_write, original_read = s.write_new, Path.read_bytes
    def write(path, value):
        if (boundary, Path(path).name) in (('native-write', 'native-status.json'),
                ('stage-write', 'root-status.pending.json'), ('receipt-write', 'completion-receipt.json')):
            raise OSError('authored write fault')
        return original_write(path, value)
    def read(path):
        if (boundary, path.name) in (('stage-read', 'root-status.pending.json'),
                ('root-read', 'root-status.json'), ('receipt-read', 'completion-receipt.json')):
            raise OSError('authored read fault')
        raw = original_read(path)
        if boundary == 'deadline' and path.name == 'completion-receipt.json':
            clock[0] = 138408
        return raw
    def check(*args, **kwargs):
        if boundary == 'verify':
            raise ValueError('authored full verifier failure')
    def replace(*args):
        raise OSError('authored promotion failure')
    monkeypatch.setattr(s, 'write_new', write)
    monkeypatch.setattr(Path, 'read_bytes', read)
    monkeypatch.setattr(s, 'verify_report', check)
    if boundary == 'replace':
        monkeypatch.setattr(s.os, 'replace', replace)
    with pytest.raises((OSError, ValueError, CampaignStopped)):
        s.publish(packet, plan, {}, {}, native_report, budget)
    revoked = json.loads(original_read(tmp_path/'run.revoked.json'))
    assert revoked['quality_complete'] is False
    assert len(revoked['full_matrix']) == 2000
    assert revoked['accounting']['calls'] == 141363
    assert all(r['qa_score'] is None for r in revoked['full_matrix'])


def test_shared_exit_receipt_uses_actual_group_contract(tmp_path, specification, monkeypatch):
    from types import SimpleNamespace as N

    from gluomem_bench.bench.lme_first import run_isolated_jobs
    from gluomem_bench.bench.lme_first_continue import NewAttemptBudget
    s = study()
    specification['selected_systems'] = ['gluomem']
    plan = s.assemble(put(tmp_path/'spec.json', specification))
    root = tmp_path/'run'
    root.mkdir()
    worker = {'output_root': str(root), 'native_mode': 'gluomem-shared-process',
              'history_commit_limits_mib': dict.fromkeys(s.SYSTEMS[1:], 4096),
              'gluomem_group_commit_limit_mib': 4096, 'native_limits': dict.fromkeys(s.SYSTEMS, 1),
              'max_workers': 1}
    qualification = {'memory_samples': {name: {'rss_mib': 1, 'private_mib': 1} for name in s.SYSTEMS[:3]}}
    budget = s.campaign_budget(clock=lambda: 0)
    view = NewAttemptBudget(budget)
    records = s.native.native_records(worker, plan, qualification)
    def fail(*args):
        raise ValueError('authored local failure')
    coordination = run_isolated_jobs(records, out=root/'coordination', run_job=fail,
        max_calls=view.max_calls, max_seconds=view.max_seconds, budget=view,
        max_workers=1, native_limits=worker['native_limits'], memory_limit_mib=12288,
        max_in_flight_tokens=39321600)
    groupdir = root/'workers/gluomem-group'
    groupdir.mkdir(parents=True)
    process = N(pid=123, returncode=0, poll=lambda: 0)
    group = N(process=process, monitor=N(join=lambda: None), workdir=groupdir,
              created=456, limit=4096, failed=False)
    s.native.SharedGluomemWorker.__exit__(group)
    native_report = {'quality_complete': False, 'jobs': coordination['jobs']}
    report = dict(protocol=s.PROTOCOL, native=native_report, full_matrix=s.coverage(plan, coordination['jobs']),
        planned=2000, stage=plan['stage'], historical_quality_pooled=False, selected_quality_complete=False,
        quality_complete=False, accounting=budget.snapshot(), clock_origin=budget.started,
        clock_deadline=budget.deadline, **{k: plan[k] for k in ('declaration', 'methods', 'changed_condition', 'remediation')})
    monkeypatch.setattr(s.native, 'verify_root', lambda *args, **kwargs: native_report)
    assert s.verify_report({'output_root': str(root)}, plan, worker, qualification, report) == report


def test_cli_never_formats_hostile_exception(tmp_path, monkeypatch, capsys):
    s = study()
    class Hostile(Exception):
        def __str__(self):
            raise AssertionError('must never format private exception')
    def fail(*args, **kwargs):
        raise Hostile()
    monkeypatch.setattr(s, 'prepare', fail)
    assert s.main(['prepare', '--spec', 'unused', '--spec-sha256', '0'*64, '--out', str(tmp_path/'no')]) == 2
    output = capsys.readouterr()
    assert output.out == ''
    assert json.loads(output.err)['category'] == 'unknown'
