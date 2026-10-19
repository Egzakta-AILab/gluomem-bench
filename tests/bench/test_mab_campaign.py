"""Offline integration evidence only: authored answers, never quality claims."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def deny_real_endpoints_in_authored_mab_tests(monkeypatch):
    """Deny external sockets in parent AND actual isolated native worker."""
    import ipaddress
    import socket
    import subprocess
    original_connect = socket.socket.connect
    def connect(sock, address):
        if isinstance(address, tuple):
            try:
                allowed = ipaddress.ip_address(address[0]).is_loopback
            except ValueError:
                allowed = address[0] == 'localhost'
            if not allowed:
                raise OSError('authored MAB tests deny real endpoints')
        return original_connect(sock, address)
    monkeypatch.setattr(socket.socket, 'connect', connect)
    original_lookup = socket.getaddrinfo
    def lookup(host, *args, **kwargs):
        try:
            allowed = ipaddress.ip_address(host).is_loopback
        except ValueError:
            allowed = host in ('localhost', None)
        if not allowed:
            raise OSError('authored MAB tests deny real endpoint DNS')
        return original_lookup(host, *args, **kwargs)
    monkeypatch.setattr(socket, 'getaddrinfo', lookup)
    original_popen = subprocess.Popen
    guard = '''import ipaddress, runpy, sys

def audit(event, args):
    if event == 'socket.getaddrinfo':
        try:
            allowed = ipaddress.ip_address(args[0]).is_loopback
        except ValueError:
            allowed = args[0] in ('localhost', None)
        if not allowed:
            raise OSError('authored native MAB test denies real endpoint DNS')
    if event == 'socket.connect' and isinstance(args[1], tuple):
        try:
            allowed = ipaddress.ip_address(args[1][0]).is_loopback
        except ValueError:
            allowed = args[1][0] == 'localhost'
        if not allowed:
            raise OSError('authored native MAB test denies real endpoints')
sys.addaudithook(audit)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    def popen(args, *positional, **kwargs):
        if isinstance(args, list) and '--graph-worker' in args:
            assert args[1] == '-I'
            args = [args[0], '-I', '-c', guard, *args[2:]]
        return original_popen(args, *positional, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', popen)


def api():
    return importlib.import_module('gluomem_bench.bench.mab_campaign')


def test_continuation_admission_exposes_versioned_exact_bridge_and_original_cap_api():
    report = api().admission()['continuation']
    assert report['campaign_mode'] == 'full-continuation-v1'
    assert report['bridge_protocol'] == 'mab-readonly-semantic-bridge@v1'
    assert report['prior_accounting'] == 'accounting_continuation.prior.coordinator'
    assert report['policy_caps'] == 'original total ceilings; prior calls/time imported once'
    assert report['cognee'] == 'all rows unrun; requires separately versioned wrapper-native qualification'


@pytest.mark.parametrize('fault', [None, 'verify', 'write', 'replace', 'final_read', 'deadline'])
def test_guarded_campaign_publication_verifies_exact_stage_and_revokes_authority(tmp_path, monkeypatch, fault):
    module = api()
    root = tmp_path / 'publication'
    root.mkdir()
    payload = {'complete': True, 'generation_calls': 7}
    class Budget:
        def check(self):
            if fault == 'deadline' and (root / 'root-status.pending.json').exists():
                raise RuntimeError('deadline')
        def snapshot(self):
            return {'calls': 7, 'elapsed_seconds': 2}
    seen = []
    def verifier(out, actual):
        assert out == root
        assert json.loads((root / 'root-status.pending.json').read_bytes()) == actual == payload
        seen.append(actual)
        if fault == 'verify':
            raise ValueError('verify')
    monkeypatch.setattr(module, '_verify_campaign_payload', verifier)
    original_write = module._write
    def write(path, value):
        if fault == 'write' and path.name.endswith('pending.json'):
            raise OSError('write')
        original_write(path, value)
    monkeypatch.setattr(module, '_write', write)
    original_replace = Path.replace
    def replace(path, target):
        if fault == 'replace':
            raise OSError('replace')
        value = original_replace(path, target)
        if fault == 'final_read':
            Path(target).write_bytes(b'{}')
        return value
    monkeypatch.setattr(Path, 'replace', replace)
    if fault:
        with pytest.raises((RuntimeError, ValueError, OSError)) as caught:
            module._publish_campaign(root, payload, Budget())
        assert caught.value.generation_calls == 7
        assert not (root / 'root-status.json').exists()
        assert json.loads((root / 'failure.json').read_bytes())['generation_calls'] == 7
    else:
        assert module._publish_campaign(root, payload, Budget()) == payload
        assert json.loads((root / 'root-status.json').read_bytes()) == payload
        assert seen == [payload]
    assert not (root / 'root-status.pending.json').exists()


def test_continuation_plan_retains_exact_original_rows_and_original_total_debits(pinned_root):
    """Read-only real retained metadata; no old DB/client/encoder construction."""
    import copy
    module = api()
    directory = pinned_root / 'full-continuation-candidate-v3'
    candidate = json.loads((directory / 'full-spec-v3.json').read_bytes())
    old_root = pinned_root / 'full-matrix-v1/root-status.json'
    old_spec = pinned_root / 'full-spec-v1.json'
    archive = pinned_root / 'full-producers-v1.zip'
    candidate['continuation'] = {key: {'path': str(path), 'sha256': module._hash(path)}
        for key, path in [('original_spec', old_spec), ('original_root', old_root),
                          ('original_archive', archive)]}
    candidate['campaign_mode'] = 'full-continuation-v1'
    candidate['execution_kind'] = module.PROTOCOL
    candidate['policy']['max_calls'] = candidate['accounting_continuation']['original_campaign_max_calls']
    candidate['policy']['max_seconds'] = candidate['accounting_continuation']['original_campaign_max_seconds']
    plan = module.continuation_plan(candidate)
    assert len(plan['original_jobs']) == 1022 and len(plan['dispatch_job_ids']) == 1008
    assert len(plan['retained']) == 160
    assert sum(len(j['case_ids']) for j in candidate['jobs'] if j['job_id'] in plan['dispatch_job_ids']) == 24295
    assert plan['prior_calls'] == 4148
    assert plan['prior_elapsed_seconds'] == 1595.865880599973
    assert plan['reuse_job_ids'] == ['full-0021-gluomem', 'full-0022-mem0', 'full-0023-no_memory']
    for fault in ('row_order', 'failed_reuse', 'prior', 'remaining_caps', 'case_order', 'old_hash'):
        forged = copy.deepcopy(candidate)
        if fault == 'row_order':
            forged['jobs'][0:2] = reversed(forged['jobs'][0:2])
        elif fault == 'failed_reuse':
            forged['jobs'][0]['disposition'] = 'reuse_old_complete_conditional_binding'
        elif fault == 'prior':
            forged['jobs'][0]['prior_generation_attempts'] = 0
        elif fault == 'remaining_caps':
            forged['policy']['max_calls'] -= plan['prior_calls']
        elif fault == 'case_order':
            forged['jobs'][0]['case_ids'].reverse()
        else:
            forged['continuation']['original_root']['sha256'] = '0' * 64
        with pytest.raises(ValueError):
            module.continuation_plan(forged)


def test_three_real_old_complete_leaves_rescore_on_copies_without_old_db_mutation(pinned_root, tmp_path):
    """Authored bridge fixture ONLY, not a campaign approval or quality claim."""
    from dataclasses import asdict
    from gluomem_bench.bench.mab_suite import MABSuite
    import gluomem
    module = api()
    directory = pinned_root / 'full-continuation-candidate-v3'
    spec = json.loads((directory / 'full-spec-v3.json').read_bytes())
    spec.update(campaign_mode=module.CONTINUATION_MODE, execution_kind=module.PROTOCOL)
    spec['continuation'] = {key: dict(path=str(path), sha256=module._hash(path)) for key, path in [
        ('original_spec', pinned_root / 'full-spec-v1.json'),
        ('original_root', pinned_root / 'full-matrix-v1/root-status.json'),
        ('original_archive', pinned_root / 'full-producers-v1.zip')]}
    original = json.loads((pinned_root / 'full-spec-v1.json').read_bytes())
    spec['policy']['max_calls'] = original['policy']['max_calls']
    spec['policy']['max_seconds'] = original['policy']['max_seconds']
    new_root = tmp_path / 'authored-full-root-not-campaign'
    spec['output_root'] = str(new_root)
    for job in spec['jobs']:
        job['output'] = str(new_root / 'jobs' / job['job_id'])
    package_roots = {'gluomem_bench': Path(module.__file__).resolve().parents[1],
                     'gluomem': Path(gluomem.__file__).resolve().parent}
    producer_semantics = []
    current = {str(p): dict(path=str(p), sha256=module._hash(p)) for p in module.required_files(pinned_root)}
    for item in original['files']:
        if not item['path'].endswith('.py'):
            continue
        parts = Path(item['path']).parts
        target = Path(item['path'])
        for package, root in package_roots.items():
            if package in parts:
                target = root.joinpath(*parts[parts.index(package) + 1:])
                break
        current[str(target)] = dict(path=str(target), sha256=module._hash(target))
        producer_semantics.append(dict(original_path=item['path'], original_sha256=item['sha256'],
            current_path=str(target), current_sha256=module._hash(target)))
    spec['files'] = list(current.values())
    plan = module.continuation_plan(spec)
    bridge = dict(protocol='mab-readonly-semantic-bridge@v1', original_bindings=dict(spec['continuation']),
        candidate_semantics_sha256=module.continuation_semantics(spec),
        current_wrapper_sha256=module._hash(module.__file__), reuse_job_ids=plan['reuse_job_ids'],
        producer_semantics=producer_semantics)
    path = tmp_path / 'authored-test-bridge-not-approval.json'
    module._write(path, bridge)
    spec['continuation']['semantic_bridge'] = dict(path=str(path), sha256=module._hash(path))
    old_root = pinned_root / 'full-matrix-v1'
    before = {p: module._hash(p) for p in old_root.rglob('*') if p.is_file()}
    suite = MABSuite.from_directory(pinned_root)
    assert module._verify_reuse(spec, {'approved_gates': {'mab_continuation_bridge': module._hash(path)}},
        plan, {c.sample.id: json.loads(json.dumps(asdict(c))) for c in suite.cases}, suite) == bridge
    # Exercise the actual full-size root verifier, not merely three leaf replays.
    # A pre-set cancel token leaves all 1008 eligible fresh rows genuinely unrun.
    from gluomem_bench.bench.parallel_campaign import CampaignBudget, CampaignPolicy, Job, run_jobs
    new_root.mkdir()
    module._write(new_root / 'execution-spec.json', spec)
    module._write(new_root / 'execution-approval.json', dict(approved=True,
        spec_sha256=module._hash(new_root / 'execution-spec.json'),
        approved_gates={'mab_continuation_bridge': module._hash(path)}))
    module._write(new_root / 'planned-cases.json', [asdict(c) for c in suite.cases])
    budget = CampaignBudget(max_calls=spec['policy']['max_calls'], max_seconds=spec['policy']['max_seconds'],
        prior_calls=plan['prior_calls'], prior_elapsed_seconds=plan['prior_elapsed_seconds'])
    cancelled = threading.Event()
    cancelled.set()
    jobs = [Job(r['job_id'], r['history_id'], Path(r['output']), r['call_budget'], r['seconds_budget'])
            for r in spec['jobs'] if r['job_id'] in plan['dispatch_job_ids']]
    def forbidden(*args):
        raise AssertionError('offline full coverage must never dispatch')
    coordination = run_jobs(jobs, policy=CampaignPolicy(**spec['policy']),
        out=new_root / 'coordination', run_job=forbidden, cancel=cancelled, budget=budget)
    records = []
    for r in spec['jobs']:
        status = {module.REUSE: 'complete', module.QUALITY_FAILED: 'failed', module.INFRA_HELD: 'held',
                  module.NEVER_DISPATCHED: 'unrun', module.NATIVE_UNRUN: 'unrun'}[r['disposition']]
        row = {k: r[k] for k in ('job_id', 'history_id', 'system', 'case_ids', 'disposition')}
        row.update(status=status, output=Path(r['output']).relative_to(new_root).as_posix(),
                   generation_calls=r.get('prior_generation_attempts', 0))
        if r['disposition'] == module.REUSE:
            row['reused_output'] = r['original_output']
        records.append(row)
    result = dict(protocol=module.PROTOCOL, campaign_mode=module.CONTINUATION_MODE,
        complete=False, coordinator_complete=True, jobs=records, generation_calls=4148,
        prior_generation_calls=4148, new_generation_calls=0, accounting=budget.snapshot(),
        artifact_sha256={p.relative_to(new_root).as_posix(): module._hash(p)
                         for p in new_root.rglob('*') if p.is_file()},
        **module._full_scope(new_root, records, spec))
    assert module._publish_campaign(new_root, result, budget) == result
    assert module.verify_campaign(new_root) == result
    assert len(result['jobs']) == 1168
    assert result['planned_history_occurrences'] == 146
    assert result['planned_question_occurrences'] == 3671
    assert all(row['planned'] == 3671 for row in result['coverage'].values())
    assert coordination['accounting']['calls'] == 4148
    assert {p: module._hash(p) for p in old_root.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('fault', [None, 'bridge', 'producer_mapping', 'deadline_setup', 'deadline_aggregate', 'verifier',
                                 'stage_write', 'replace', 'final_read'])
def test_executable_continuation_keeps_failed_and_native_unrun_boundaries(tmp_path, pinned_root, monkeypatch, fault):
    """Actual native reader/scorer + localhost-only transport; no real endpoints."""
    from dataclasses import asdict
    from gluomem_bench.bench.mab_suite import MABSuite
    from gluomem_bench.bench.rack import RackProfile
    from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe
    from gluomem_bench.bench.comprehensive_driver import PROTOCOL as ENVELOPE
    from gluomem_bench.bench import parallel_campaign as scheduler
    import socket
    import zipfile
    module = api()
    original_connect = socket.socket.connect
    def local_connect(sock, address):
        if isinstance(address, tuple) and address[0] not in ('127.0.0.1', '::1'):
            raise OSError('test denies real endpoint')
        return original_connect(sock, address)
    monkeypatch.setattr(socket.socket, 'connect', local_connect)
    suite = MABSuite.from_rows({'Accurate_Retrieval': [
        {'context': 'Station locker code is cobalt.', 'questions': [f'Question {i}'],
         'answers': [['cobalt']], 'metadata': {'source': 'ruler_qa1_197K', 'qa_pair_ids': [f'q{i}']}}
        for i in range(4)]})
    suite.root, suite.receipt = pinned_root, {'authored_integration_only': True}
    monkeypatch.setattr(MABSuite, 'from_directory', lambda root: suite)
    phase = ['old']
    import httpx
    original_request = httpx.Client.request
    def request(client, method, url, **kwargs):
        response = original_request(client, method, url, **kwargs)
        body = kwargs.get('json')
        if phase[0] == 'old' and str(url).endswith('/chat/completions'):
            if 'Question 1' in json.dumps(body):
                raise httpx.ReadTimeout('authored local timeout')
            if 'Question 2' in json.dumps(body):
                raise module.SharedRackError('authored global identity cancellation')
        return response
    monkeypatch.setattr(httpx.Client, 'request', request)
    with model_server() as (url, calls):
        old_root = tmp_path / 'old'
        profile = RackProfile(base_url=url, model='example/offline', profile_version='role-budgets-v6',
            generation_budgets=dict(extraction=4096, consolidation=4096, reader=4096, judge=4096))
        old = {'protocol': ENVELOPE, 'execution_kind': module.PROTOCOL,
            'campaign_mode': module.FULL_MODE, 'output_root': str(old_root), 'mab_root': str(pinned_root),
            'case_manifest_sha256': module._digest([asdict(c) for c in suite.cases]),
            'profile': profile.model_dump(mode='json'), 'embedding_url': url,
            'embedding_recipe': EmbeddingRecipe().as_dict(), 'case_ids': [c.sample.id for c in suite.cases],
            'systems': ['no_memory'], 'chunk_chars': 4096, 'top_k': 5, 'evidence_token_budget': 4096,
            'judge_qualification': None,
            'policy': dict(max_workers=1, max_in_flight=1, max_in_flight_tokens=131072,
                           max_calls=12, max_seconds=120, stop_on_failure=False),
            'jobs': [dict(job_id=f'j{i}', history_id=c.sample.conv_id, system='no_memory',
                case_ids=[c.sample.id], output=str(old_root / 'jobs' / f'j{i}'),
                call_budget=2, seconds_budget=60) for i, c in enumerate(suite.cases)],
            'files': [{'path': str(p), 'sha256': module._hash(p)} for p in module.required_files(pinned_root)]}
        old_spec, old_approval = tmp_path / 'old-spec.json', tmp_path / 'old-approval.json'
        def seal(path, approval, value, gates=None):
            module._write(path, value)
            module._write(approval, dict(approved=True, reviewer_kind='AI-assisted',
                spec_sha256=module._hash(path), approved_gates=gates or {}))
        seal(old_spec, old_approval, old)
        old_result = module.execute(old_spec, old_approval)
        assert [r['status'] for r in old_result['jobs']] == ['complete', 'failed', 'failed', 'unrun']
        assert old_result['generation_calls'] == 3  # Local timeout did not cancel healthy admission.
        old_bytes = {p: p.read_bytes() for p in old_root.rglob('*') if p.is_file()}
        archive_path = tmp_path / 'old-producers.zip'
        with zipfile.ZipFile(archive_path, 'w') as archive:
            sources = []
            for i, item in enumerate(old['files']):
                member = f'producer/{i}'
                archive.writestr(member, Path(item['path']).read_bytes())
                sources.append(dict(item, member=member))
            archive.writestr('producer-bindings.json', json.dumps(sources))
        import copy
        spec = copy.deepcopy(old)
        root = tmp_path / 'continued'
        spec.update(campaign_mode='full-continuation-v1', output_root=str(root), systems=['no_memory', 'cognee'])
        spec['profile']['timeout_seconds'] = 600
        accounting = json.loads((old_root / 'coordination/progress.json').read_bytes())['accounting']
        spec['accounting_continuation'] = {'prior': {'coordinator': accounting}}
        spec['continuation'] = {key: dict(path=str(path), sha256=module._hash(path))
            for key, path in [('original_spec', old_spec), ('original_root', old_root / 'root-status.json'),
                              ('original_archive', archive_path)]}
        dispositions = [module.REUSE, module.QUALITY_FAILED, module.INFRA_HELD, module.NEVER_DISPATCHED]
        for i, row in enumerate(spec['jobs']):
            row.update(disposition=dispositions[i], original_output=row['output'],
                       prior_generation_attempts=old_result['jobs'][i]['generation_calls'],
                       output=str(root / 'jobs' / f'j{i}'))
        spec['jobs'] += [dict(job_id=f'c{i}', history_id=c.sample.conv_id, system='cognee',
            case_ids=[c.sample.id], output=str(root / 'jobs' / f'c{i}'), call_budget=2,
            seconds_budget=60, prior_generation_attempts=0, disposition=module.NATIVE_UNRUN)
            for i, c in enumerate(suite.cases)]
        bridge = dict(protocol='mab-readonly-semantic-bridge@v1',
            original_bindings=copy.deepcopy(spec['continuation']),
            candidate_semantics_sha256=module.continuation_semantics(spec),
            current_wrapper_sha256=module._hash(module.__file__), reuse_job_ids=['j0'],
            producer_semantics=[dict(original_path=r['path'], original_sha256=r['sha256'],
                current_path=r['path'], current_sha256=r['sha256']) for r in old['files'] if r['path'].endswith('.py')])
        bridge_path = tmp_path / 'bridge.json'
        if fault == 'producer_mapping':
            bridge['producer_semantics'][0]['current_path'] = str(Path(module.__file__).resolve())
            bridge['producer_semantics'][0]['current_sha256'] = module._hash(module.__file__)
        module._write(bridge_path, bridge)
        spec['continuation']['semantic_bridge'] = dict(path=str(bridge_path), sha256=module._hash(bridge_path))
        gates = {'mab_continuation_bridge': module._hash(bridge_path)}
        if fault == 'bridge':
            gates.clear()
        spec_path, approval_path = tmp_path / 'new-spec.json', tmp_path / 'new-approval.json'
        seal(spec_path, approval_path, spec, gates)
        phase[0] = 'new'
        before = len(calls)
        budgets = []
        created = []
        original_budget_init = scheduler.CampaignBudget.__init__
        def capture_budget(budget, **kwargs):
            original_budget_init(budget, **kwargs)
            created.append(budget)
        monkeypatch.setattr(scheduler.CampaignBudget, '__init__', capture_budget)
        original_run = scheduler.run_jobs
        def run_jobs(*args, **kwargs):
            budgets.append(kwargs['budget'])
            assert kwargs['budget'] is created[0]
            assert kwargs['budget'].calls == 3
            return original_run(*args, **kwargs)
        monkeypatch.setattr(scheduler, 'run_jobs', run_jobs)
        original_scope = module._full_scope
        def scope(*args):
            result = original_scope(*args)
            if fault == 'deadline_aggregate':
                budgets[0].deadline = budgets[0].clock() - 1
            return result
        monkeypatch.setattr(module, '_full_scope', scope)
        original_from = MABSuite.from_directory
        def from_directory(path):
            assert created[0].prior_calls == 3  # Starts BEFORE native setup.
            if fault == 'deadline_setup':
                created[0].deadline = created[0].clock() - 1
            return original_from(path)
        monkeypatch.setattr(MABSuite, 'from_directory', from_directory)
        original_write = module._write
        def write(path, value):
            if fault == 'stage_write' and path == root / 'root-status.pending.json':
                raise OSError('authored stage write')
            original_write(path, value)
        monkeypatch.setattr(module, '_write', write)
        original_replace = Path.replace
        def replace(path, target):
            if path == root / 'root-status.pending.json':
                if fault == 'replace':
                    raise OSError('authored replace')
                result = original_replace(path, target)
                if fault == 'final_read':
                    Path(target).write_bytes(b'{}')
                return result
            return original_replace(path, target)
        monkeypatch.setattr(Path, 'replace', replace)
        if fault == 'verifier':
            monkeypatch.setattr(module, '_verify_campaign_payload',
                lambda *args: (_ for _ in ()).throw(ValueError('authored root verifier')))
        if fault:
            with pytest.raises((ValueError, OSError, RuntimeError)) as caught:
                module.execute(spec_path, approval_path)
            assert caught.value.generation_calls == (3 if fault in ('bridge', 'producer_mapping', 'deadline_setup') else 4)
            assert not (root / 'root-status.json').exists()
            assert not (root / 'root-status.pending.json').exists()
        else:
            result = module.execute(spec_path, approval_path)
            assert module.verify_campaign(root) == result
            assert result['generation_calls'] == 4
            assert result['prior_generation_calls'] == 3 and result['new_generation_calls'] == 1
            assert [r['status'] for r in result['jobs']] == ['complete', 'failed', 'held', 'complete'] + ['unrun'] * 4
            assert result['coverage']['no_memory']['graded'] == 2
            assert result['coverage']['cognee']['graded'] == 0
            assert result['native_metrics']['cognee']['sources']['ruler_qa1_197K']['completed_only_mean'] is None
            assert not result['complete']
            assert budgets[0].snapshot()['calls'] == 4
            assert sum(path.endswith('/chat/completions') for path, _ in calls[before:]) == 1
            authority = root / 'root-status.json'
            import copy
            for tamper in ('missing', 'duplicate', 'promote_failed', 'promote_native', 'prior', 'metrics', 'reuse_path'):
                forged = copy.deepcopy(result)
                if tamper == 'missing':
                    forged['jobs'].pop()
                elif tamper == 'duplicate':
                    forged['jobs'][-1] = forged['jobs'][0]
                elif tamper == 'promote_failed':
                    forged['jobs'][1]['status'] = 'complete'
                elif tamper == 'promote_native':
                    forged['jobs'][-1]['status'] = 'complete'
                elif tamper == 'prior':
                    forged['accounting']['prior_calls'] -= 1
                elif tamper == 'metrics':
                    forged['coverage']['no_memory']['graded'] += 1
                else:
                    forged['jobs'][0]['reused_output'] = str(root / 'jobs/j3')
                module._write(authority, forged)
                with pytest.raises(ValueError):
                    module.verify_campaign(root)
            module._write(authority, result)
            leaf_out = root / 'jobs/j3'
            source_path, protocol_path = leaf_out / 'source-inputs.json', leaf_out / 'protocol.json'
            source = json.loads(source_path.read_bytes())
            source[0]['content'] = 'Unrelated forged source.'
            module._write(source_path, source)
            protocol = json.loads(protocol_path.read_bytes())
            protocol['context_sha256'] = hashlib.sha256(source[0]['content'].encode()).hexdigest()
            module._write(protocol_path, protocol)
            leaf = json.loads((leaf_out / 'root-status.json').read_bytes())
            for path in (source_path, protocol_path):
                leaf['artifact_sha256'][path.relative_to(leaf_out).as_posix()] = module._hash(path)
            module._write(leaf_out / 'root-status.json', leaf)
            for path in (source_path, protocol_path, leaf_out / 'root-status.json'):
                result['artifact_sha256'][path.relative_to(root).as_posix()] = module._hash(path)
            module._write(authority, result)
            with pytest.raises(ValueError, match='source|context'):
                module.verify_campaign(root)
        assert all(p.read_bytes() == data for p, data in old_bytes.items())
        assert {p for p in old_root.rglob('*') if p.is_file()} == set(old_bytes)


def native_graph_output(body):
    """Authored responses ONLY at the real native package network boundary."""
    if not body.get('response_format'):
        if len(body['messages']) == 1 and body['messages'][0]['role'] == 'user':
            return 'yes'
        if 'following string: "test"' in body['messages'][0]['content']:
            return 'test'
        return 'cobalt'
    fmt = body['response_format']
    schema = fmt.get('json_schema', {}).get('schema', {})
    if fmt == {'type': 'json_object'}:
        schema = json.loads(body['messages'][0]['content'].split('```json\n', 1)[1].split('```', 1)[0])
    if 'nodes' in schema.get('properties', {}):
        return {'nodes': [
            {'id': 'station', 'name': 'Station locker', 'type': 'Object',
             'description': 'Station locker code is cobalt.'},
            {'id': 'cobalt', 'name': 'cobalt', 'type': 'Code', 'description': 'Locker code.'}],
            'edges': [{'source_node_id': 'station', 'target_node_id': 'cobalt',
                       'relationship_name': 'has_code', 'description': 'Station locker code is cobalt.'}]}
    if 'summary' in schema.get('properties', {}):
        return {'summary': 'Station locker code is cobalt.', 'description': 'Authored document.'}
    raise AssertionError('Unexpected native schema')


def test_cognee_real_native_documents_reach_same_mab_reader_and_native_scorer(tmp_path, pinned_root):
    from gluomem_bench.bench.comparison_backends import CogneeSettings
    native = (Path(__file__).resolve().parents[2] / 'tests/fixtures/native/operator-supplied'
              / 'cognee/venv/Scripts/python.exe')
    assert native.is_file(), 'pre-existing isolated native package required; no install'
    with model_server(output_for=native_graph_output) as (url, calls):
        args, outer = setup_run(url, tmp_path, authored_suite(pinned_root))
        args.update(system='cognee', chunk_chars=4096, cognee_settings=CogneeSettings(
            python_executable=str(native), root=args['out'] / 'graph/native', dataset='authored-mab',
            llm_url=url, llm_model=args['profile'].model, llm_api_key='local',
            embedding_url=url, embedding_model='text-embedding-3-small', embedding_dimensions=768,
            embedding_api_key='local', max_completion_tokens=1200, timeout_seconds=100))
        result = api().run_history(**args)
    assert api().verify_history(args['out']) == result
    ledger = json.loads((args['out'] / 'graph/native-ledger.json').read_bytes())
    row, = json.loads((args['out'] / 'predictions.json').read_bytes())
    assert result['generation_calls'] == outer.calls == ledger['generation_calls'] + 1
    assert outer.calls == sum(path.endswith('/chat/completions') for path, _ in calls)
    assert row['score']['metric'] == 'substring_exact_match' and row['score']['value'] == 1.0
    assert 'Connections:' in row['evidence_receipt']['original_text']
    assert row['question'] == args['suite'].reader_question(args['suite'].cases[0])
    assert row['retrieval']['native_final_completion_used'] is False
    assert ledger['source_inputs'][0]['text'] == args['suite'].histories[0].context
    assert not any(word in json.dumps(ledger['source_inputs']) for word in
                   ('1970', 'What is the station', 'qa_pair', 'ruler_qa'))
    assert ledger['native_worker_closed'] and ledger['proxy_closed']
    # Rehash EVERY enclosing inventory: semantic native readback must still win.
    saved = {p: p.read_bytes() for p in args['out'].rglob('*') if p.is_file()}
    contexts = json.loads((args['out'] / 'graph/graph-contexts.json').read_bytes())
    api()._verify_cognee_rendered_context(args['out'] / 'graph/native', contexts,
                                          python_executable=str(native))
    import copy
    forged = copy.deepcopy(contexts)
    forged[0]['hits'][0]['text'] = 'A model answer substituted for graph context.'
    with pytest.raises(ValueError, match='Cognee.*context'):
        api()._verify_cognee_rendered_context(args['out'] / 'graph/native', forged,
                                              python_executable=str(native))
    contexts_path = args['out'] / 'graph/graph-contexts.json'
    contexts = json.loads(contexts_path.read_bytes())
    refs = contexts[0]['hits'][0]['provenance']['native_evidence']
    assert any(ref['kind'] == 'segment' for ref in refs)
    for ref in refs:
        if ref['kind'] == 'segment':
            ref['source_ref_key'] = 'source_ref:v1:forged-scope:forged-data'
    contexts_path.write_bytes(json.dumps(contexts).encode())
    row['retrieval'] = contexts[0]
    (args['out'] / 'predictions.json').write_bytes(json.dumps([row]).encode())
    gate_path = args['out'] / 'graph/before-reader-000001.json'
    gate = json.loads(gate_path.read_bytes())
    gate['graph_context'] = contexts[0]
    gate_path.write_bytes(json.dumps(gate).encode())
    ledger['artifact_sha256'] = {p.relative_to(args['out'] / 'graph').as_posix(): api()._hash(p)
        for p in (args['out'] / 'graph').rglob('*') if p.is_file() and p.name != 'native-ledger.json'}
    (args['out'] / 'graph/native-ledger.json').write_bytes(json.dumps(ledger).encode())
    result['artifact_sha256'] = {str(p.relative_to(args['out'])): api()._hash(p)
        for p in saved if p.name != 'root-status.json'}
    (args['out'] / 'root-status.json').write_bytes(json.dumps(result).encode())
    with pytest.raises(ValueError, match='Cognee.*lineage|source.*lineage'):
        api().verify_history(args['out'])


@pytest.mark.parametrize('fault', ['root_readback', 'native_failure', 'outer_cap', 'query_generation',
                                  'query_failure', 'coverage', 'cleanup', 'verifier'])
def test_cognee_all_native_and_reader_failures_retain_one_history_spend(
    tmp_path, pinned_root, monkeypatch, fault
):
    from gluomem_bench.bench.comparison_backends import CogneeSettings
    from gluomem_bench.bench.cognee_graph import CogneeGraphEvidenceBackend
    module = api()
    native = (Path(__file__).resolve().parents[2] / 'tests/fixtures/native/operator-supplied'
              / 'cognee/venv/Scripts/python.exe')
    if fault in ('query_generation', 'query_failure'):
        original_search = CogneeGraphEvidenceBackend.search
        async def query_helper(backend, query, **kwargs):
            # Inject an authored query-time helper at the actual native proxy
            # boundary. Native retrieval still runs unchanged; no fake backend.
            import httpx
            hits = await original_search(backend, query, **kwargs)
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.post(backend.settings.llm_url + '/chat/completions',
                    headers={'Authorization': 'Bearer ' + backend.settings.llm_api_key},
                    json={'model': backend.settings.llm_model, 'temperature': 0, 'max_tokens': 1200,
                          'messages': [{'role': 'system', 'content': 'authored-query-helper'}]})
                response.raise_for_status()
            return hits
        monkeypatch.setattr(CogneeGraphEvidenceBackend, 'search', query_helper)
    if fault == 'cleanup':
        original_close = CogneeGraphEvidenceBackend.close
        async def bad_close(backend):
            await original_close(backend)
            raise OSError('authored native cleanup fault')
        monkeypatch.setattr(CogneeGraphEvidenceBackend, 'close', bad_close)
    original_write = module._write
    def write(path, value):
        if fault == 'root_readback' and path.name == 'root-status.json':
            return original_write(path, {**value, 'generation_calls': 999})
        original_write(path, value)
        if fault == 'coverage' and path.name == 'coverage.json':
            value['processed_events'] = 0
            original_write(path, value)
    monkeypatch.setattr(module, '_write', write)
    if fault == 'verifier':
        monkeypatch.setattr(module, '_verify_payload',
            lambda *a: (_ for _ in ()).throw(ValueError('authored verifier fault')))
    def finish(body):
        return ('length' if (fault == 'native_failure' and body.get('response_format'))
                or (fault == 'query_failure' and 'authored-query-helper' in json.dumps(body)) else 'stop')
    with model_server(output_for=native_graph_output, finish=finish) as (url, calls):
        args, outer = setup_run(url, tmp_path, authored_suite(pinned_root))
        args.update(system='cognee', chunk_chars=4096, cognee_settings=CogneeSettings(
            python_executable=str(native), root=args['out'] / 'graph/native', dataset='authored-mab',
            llm_url=url, llm_model=args['profile'].model, llm_api_key='local',
            embedding_url=url, embedding_model='text-embedding-3-small', embedding_dimensions=768,
            embedding_api_key='local', max_completion_tokens=1200, timeout_seconds=100))
        if fault == 'outer_cap':
            outer.job = type(outer.job)(outer.job.job_id, outer.job.history_id, outer.job.output, 1, 120)
        if fault == 'query_generation':
            result = module.run_history(**args)
            ledger = json.loads((args['out'] / 'graph/native-ledger.json').read_bytes())
            contexts = json.loads((args['out'] / 'graph/graph-contexts.json').read_bytes())
            before = json.loads((args['out'] / 'graph/before-search.json').read_bytes())
            assert contexts[0]['native_calls_after_search'] == before['generation']['calls'] + 1
            assert result['generation_calls'] == outer.calls == ledger['generation_calls'] + 1
            assert module.verify_history(args['out']) == result
            return
        with pytest.raises(Exception) as caught:
            module.run_history(**args)
    dispatched = sum(path.endswith('/chat/completions') for path, _ in calls)
    assert caught.value.generation_calls == outer.calls == dispatched > 0
    assert not (args['out'] / 'root-status.json').exists()
    assert not (args['out'] / 'scores.json').exists()
    ledger = json.loads((args['out'] / 'graph/native-ledger.json').read_bytes())
    assert ledger['native_worker_closed'] and ledger['proxy_closed']
    assert json.loads((args['out'] / 'failure.json').read_bytes())['generation_calls'] == outer.calls
    if fault in ('coverage', 'native_failure', 'outer_cap', 'query_failure'):
        assert not (args['out'] / 'predictions.json').exists()


def test_cognee_complete_projection_gate_binds_real_qualification_and_all_histories(pinned_root, tmp_path, monkeypatch):
    from dataclasses import asdict
    from gluomem_bench.bench.mab_suite import MABSuite
    from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe
    module = api()
    suite = MABSuite.from_directory(pinned_root)
    qualification = (Path(__file__).resolve().parents[2] / 'tests/fixtures/native/operator-supplied'
                     / 'cognee/graph-full-readiness-20261017/verified-qualification.json')
    qualified = json.loads(qualification.read_bytes())
    native = qualification.parent.parent / 'venv/Scripts/python.exe'
    package = native.parent.parent / 'Lib/site-packages/cognee'
    source_files = sorted(package.rglob('*.py'))
    files = qualified['evidence_files'] + [{'path': str(qualification), 'sha256': module._hash(qualification)}]
    files += [{'path': str(p), 'sha256': module._hash(p)} for p in source_files]
    projection = [{'history_id': h.id, 'context_sha256': h.context_sha256,
        'documents_sha256': module._digest([asdict(d) for d in module.cognee_documents(
            module.document_events(h), chunk_chars=4096)])}
        for h in suite.histories]
    spec = {'campaign_mode': module.FULL_MODE, 'case_ids': [c.sample.id for c in suite.cases],
        'chunk_chars': 4096, 'embedding_recipe': EmbeddingRecipe().as_dict(),
        'profile': {'model': qualified['real_model']}, 'files': files,
        'cognee': {'qualification': {'path': str(qualification), 'sha256': module._hash(qualification)},
            'scope': 'all-pinned-mab-history-question-occurrences',
            'source_manifest_sha256': module._digest(projection),
            'native': {'python_executable': str(native), 'chunk_size': 4096, 'timeout_seconds': 120,
                'package_sha256': module._digest({p.relative_to(package).as_posix(): module._hash(p)
                                                for p in source_files})}}}
    approval = {'approved_gates': {'cognee_native_graph': module._hash(qualification)}}
    assert module._qualify_cognee_spec(spec, approval, suite)['source_manifest_sha256'] == module._digest(projection)
    import copy
    for fault in ('scope', 'selection', 'qualification', 'approval', 'recipe', 'source', 'package'):
        bad, review = copy.deepcopy(spec), copy.deepcopy(approval)
        if fault == 'scope':
            bad['cognee']['scope'] = 'four-case-pilot'
        elif fault == 'selection':
            bad['case_ids'] = bad['case_ids'][:4]
        elif fault == 'qualification':
            bad['cognee']['qualification']['sha256'] = '0' * 64
        elif fault == 'approval':
            review['approved_gates'] = {}
        elif fault == 'recipe':
            bad['embedding_recipe']['dim'] = 3
        elif fault == 'package':
            bad['cognee']['native']['package_sha256'] = '0' * 64
        else:
            bad['cognee']['source_manifest_sha256'] = '0' * 64
        with pytest.raises(ValueError, match='Cognee|qualification|source|scope'):
            module._qualify_cognee_spec(bad, review, suite)
    assert len(projection) == 146 and len(spec['case_ids']) == 3671
    # The approved entry point also executes genuine native Cognee with the
    # SAME MAB reader and its native judge, using ONLY authored input/answers.
    from gluomem_bench.bench.rack import RackProfile
    from gluomem_bench.bench.comprehensive_driver import PROTOCOL as ENVELOPE
    suite = MABSuite.from_rows({'Accurate_Retrieval': [{
        'context': 'Station locker code is cobalt.',
        'questions': ['What is the station locker code?'], 'answers': [['cobalt']],
        'metadata': {'source': 'longmemeval_s*', 'qa_pair_ids': ['authored'],
                     'question_types': ['single-session-user'], 'question_ids': ['authored']}}]})
    suite.root, suite.receipt = pinned_root, {'authored_integration_only': True}
    monkeypatch.setattr(MABSuite, 'from_directory', lambda root: suite)
    projection = [{'history_id': h.id, 'context_sha256': h.context_sha256,
        'documents_sha256': module._digest([asdict(d) for d in module.cognee_documents(
            module.document_events(h), chunk_chars=4096)])}
        for h in suite.histories]
    spec['cognee']['source_manifest_sha256'] = module._digest(projection)
    spec['files'] += [{'path': str(p), 'sha256': module._hash(p)} for p in module.required_files(pinned_root)]
    with model_server(output_for=native_graph_output, model=qualified['real_model']) as (url, calls):
        root = tmp_path / 'full-cognee'
        profile = RackProfile(base_url=url, model=qualified['real_model'], profile_version='role-budgets-v6',
            generation_budgets=dict(extraction=4096, consolidation=4096, reader=4096, judge=4096))
        spec.update(protocol=ENVELOPE, execution_kind=module.PROTOCOL,
            output_root=str(root), mab_root=str(pinned_root), profile=profile.model_dump(mode='json'),
            systems=['cognee'], case_ids=[c.sample.id for c in suite.cases],
            case_manifest_sha256=module._digest([asdict(c) for c in suite.cases]),
            embedding_url=url, top_k=5, evidence_token_budget=4096,
            judge_qualification={'protocol': 'native-judge-parameters@v2', 'model': profile.model,
                                 'temperatures': [0, 0.1], 'output_budget': 4096},
            policy={'max_workers': 1, 'max_in_flight': 1, 'max_in_flight_tokens': 131072,
                    'max_calls': 12, 'max_seconds': 180, 'stop_on_failure': False},
            jobs=[{'job_id': 'authored', 'history_id': suite.histories[0].id, 'system': 'cognee',
                   'case_ids': [c.sample.id for c in suite.cases], 'output': str(root / 'jobs/authored'),
                   'call_budget': 12, 'seconds_budget': 120}])
        spec_path, approval_path = tmp_path / 'spec.json', tmp_path / 'approval.json'
        spec_path.write_bytes(json.dumps(spec).encode())
        approval.update(approved=True, reviewer_kind='AI-assisted', spec_sha256=module._hash(spec_path))
        approval_path.write_bytes(json.dumps(approval).encode())
        result = module.execute(spec_path, approval_path)
    assert result['complete'] and result['coverage']['cognee']['graded'] == 1
    assert module.verify_campaign(root) == result
    leaf = root / 'jobs/authored'
    ledger = json.loads((leaf / 'graph/native-ledger.json').read_bytes())
    row, = json.loads((leaf / 'predictions.json').read_bytes())
    assert row['score']['metric'] == 'llm_judge_accuracy'
    assert row['score']['value'] == 1.0  # authored yes; not quality evidence
    assert result['generation_calls'] == ledger['generation_calls'] + 2
    assert result['generation_calls'] == sum(p.endswith('/chat/completions') for p, _ in calls)
    assert module._hash(root / 'cognee-qualification.json') == module.COGNEE_QUALIFICATION_SHA256
    assert json.loads((root / 'cognee-source-manifest.json').read_bytes()) == projection
    assert json.loads((leaf / 'protocol.json').read_bytes())['cognee_admission'] == spec['cognee']
    manifest_path = root / 'cognee-source-manifest.json'
    forged = json.loads(manifest_path.read_bytes())
    forged[0]['documents_sha256'] = '0' * 64
    manifest_path.write_bytes(json.dumps(forged).encode())
    result['artifact_sha256']['cognee-source-manifest.json'] = module._hash(manifest_path)
    (root / 'root-status.json').write_bytes(json.dumps(result).encode())
    with pytest.raises(ValueError, match='Cognee.*source|Cognee.*scope'):
        module.verify_campaign(root)


def test_v6_all_native_judge_protocols_preserve_declared_parameters(tmp_path, pinned_root):
    from gluomem_bench.bench.mab_suite import MABSuite
    from gluomem_bench.bench.rack import RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import WorkloadBudget
    import httpx
    suite = MABSuite.from_directory(pinned_root)
    scorer = suite.scorer()
    profile = RackProfile(base_url='http://localhost:8000/v1', model='offline',
        profile_version='role-budgets-v6', generation_budgets=dict(
            extraction=4096, consolidation=4096, reader=4096, judge=4096))
    qualification = {'protocol': 'native-judge-parameters@v2', 'model': profile.model,
                     'temperatures': [0, 0.1], 'output_budget': 4096}
    for source in ('longmemeval_s*', 'infbench_sum_eng_shots2'):
        case = next(c for c in suite.cases if c.source == source)
        requests = scorer.judge_requests(case, 'candidate')
        sent = []
        outputs = {'answer': 'yes', 'fluency': '{"fluency":1}',
                   'recall': '{"recall":0}', 'precision': '{"precision":0,"sentence_count":1}'}
        def serve(request):
            body = json.loads(request.content)
            if request.url.path == '/tokenize':
                return httpx.Response(200, json={'count': 100})
            sent.append(body)
            native = requests[len(sent)-1]
            return httpx.Response(200, json={'model': profile.model, 'choices': [
                {'finish_reason': 'stop', 'message': {'content': outputs[native.role]}}]})
        client = RackClient(profile, receipt=tmp_path / (source.replace('*', '') + '.jsonl'),
            transport=httpx.MockTransport(serve), budget=WorkloadBudget(policy=profile.profile_version))
        try:
            _score, judgments = api().score_prediction(scorer, case, 'candidate', client=client,
                                                      judge_qualification=qualification)
            assert len(sent) == len(judgments) == len(requests)
            for wire, native in zip(sent, requests, strict=True):
                assert wire['temperature'] == native.temperature
                assert wire['max_tokens'] == native.max_tokens
                assert wire['messages'] == [{'role': 'user', 'content': native.prompt}]
        finally:
            client.close()


@pytest.fixture
def pinned_root():
    root = os.environ.get('MAB_ROOT')
    if not root:
        pytest.skip('MAB_ROOT required for pinned native templates/scorers')
    return Path(root)


@contextmanager
def model_server(*, finish='stop', overflow=False, token_count=None, mem0_output=None,
                 output_for=None, response_model=None, model='example/offline'):
    """Only the network boundary is fake; DSPy, SQLite and search are real."""
    from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            calls.append((self.path, None))
            self.reply({'data': [{'id': model, 'max_model_len': 131072}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append((self.path, body))
            if self.path == '/tokenize':
                return self.reply({'count': token_count(body['messages']) if token_count else (200000 if overflow else 100)})
            if self.path.endswith('/embeddings'):
                return self.reply({'recipe': EmbeddingRecipe().as_dict(),
                    'role': body['input_type'], 'data': [
                        {'index': i, 'embedding': [1.0] + [0.0] * 767}
                        for i, _ in enumerate(body['input'])]})
            messages = json.dumps(body['messages'])
            if '## New Messages' in messages:
                output = mem0_output if mem0_output is not None else {'memory': [
                    {'text': 'Station locker code is cobalt.', 'event': 'ADD'}]}
            elif 'supporting verbatim quote' in messages:
                output = {'claims': [
                    {'text': text, 'verbatim_span': text}
                    for text in ('Station locker code is cobalt.', 'Station locker code is willow.')
                    if text in body['messages'][-1]['content']]}
            elif 'target_frame_id' in messages:
                output = {'decision': {'action': 'ADD'}}
            else:
                output = 'cobalt'
            if output_for:
                output = output_for(body)
            self.reply({'id': 'offline', 'object': 'chat.completion', 'created': 1,
                'model': response_model(body) if response_model else model,
                'choices': [{'index': 0, 'finish_reason': finish(body) if callable(finish) else finish,
                'message': {'role': 'assistant', 'content': json.dumps(output) if isinstance(output, dict) else output}}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}})
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def authored_suite(root):
    from gluomem_bench.bench.mab_suite import MABSuite
    suite = MABSuite.from_rows({'Accurate_Retrieval': [{
        'context': 'Station locker code is cobalt.Station locker code is willow.',
        'questions': ['What is the station locker code?'], 'answers': [['cobalt']],
        'metadata': {'source': 'ruler_qa1_197K', 'qa_pair_ids': ['authored']}}]})
    suite.root = root  # Native source bytes remain hash-checked, not authored scorer copies.
    return suite


def setup_run(url, tmp_path, suite):
    from gluomem_bench.bench.embedding_runtime import RemoteEmbedder
    from gluomem_bench.bench.parallel_campaign import CampaignBudget, Job, JobContext, RequestLimiter
    from gluomem_bench.bench.rack import RackProfile, controlled_config
    profile = RackProfile(base_url=url, model='example/offline', max_generation_calls=12,
        max_input_chars=2000000, profile_version='role-budgets-v5',
        generation_budgets={'extraction': 1200, 'consolidation': 1200, 'reader': 1024, 'judge': 256})
    embedder = RemoteEmbedder(url)
    cfg = controlled_config(profile, data=tmp_path / 'authored.json', sha256='0' * 64, embedder=embedder)
    job = Job('offline', suite.histories[0].id, tmp_path / 'run', 12, 120)
    outer = JobContext(job, CampaignBudget(max_calls=12, max_seconds=120),
                       RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))
    return dict(suite=suite, history_id=job.history_id, case_ids=[c.sample.id for c in suite.cases],
                cfg=cfg, profile=profile, out=job.output, system='gluomem',
                embedder=embedder, outer_budget=outer, authored=True,
                chunk_chars=len('Station locker code is cobalt.')), outer


def test_reader_ceiling_is_complete_messages_whole_line_prefix_and_bound_to_dispatch(tmp_path, pinned_root):
    def count(messages):
        return 100 + len(json.loads(messages[-1]['content'])['evidence']) * 100
    with model_server(token_count=count) as (url, calls):
        args, outer = setup_run(url, tmp_path, authored_suite(pinned_root))
        args['system'] = 'bm25'
        result = api().run_history(**args)
    row, = json.loads((args['out'] / 'predictions.json').read_bytes())
    receipt = row['evidence_receipt']
    assert receipt['budget'] == 4096 and receipt['kept_tokens'] <= 4096
    assert receipt['original_tokens'] > 4096
    assert receipt['original_text'].startswith(receipt['kept_text'])
    assert receipt['kept_text'] in ('Station locker code is cobalt.\n', 'Station locker code is willow.\n')
    assert row['evidence'] == receipt['kept_text']
    assert row['reader_receipt']['admitted_prompt_tokens'] == receipt['kept_tokens']
    assert receipt['kept_sha256'] == hashlib.sha256(row['evidence'].encode()).hexdigest()
    assert api().verify_history(args['out']) == result
    assert outer.calls == 1


@pytest.mark.parametrize('native_output', [None, {'memory': []}, 'not JSON', {'facts': []}])
def test_stock_mem0_source_document_strings_real_native_network_boundary(
    tmp_path, pinned_root, monkeypatch, native_output
):
    monkeypatch.setenv('MEM0_TELEMETRY', 'False')
    # A stable process-local import-time home outside all source repositories.
    monkeypatch.setenv('MEM0_DIR', str(tmp_path.parent / 'native-home'))
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    with model_server(mem0_output=native_output) as (url, calls):
        args, outer = setup_run(url, tmp_path, authored_suite(pinned_root))
        args['system'] = 'mem0'
        args['chunk_chars'] = 4096
        if native_output in ('not JSON', {'facts': []}):
            with pytest.raises(Exception, match='extraction|warning|diagnostic'):
                api().run_history(**args)
            assert not (args['out'] / 'root-status.json').exists()
            assert not (args['out'] / 'scores.json').exists()
            assert (args['out'] / 'mem0-native.jsonl').exists()
            assert outer.calls == 1
            return
        result = api().run_history(**args)
    assert result['generation_calls'] == outer.calls == 2
    row, = json.loads((args['out'] / 'predictions.json').read_bytes())
    assert '1970' not in row['evidence'] and 'created_at' not in row['evidence']
    coverage = json.loads((args['out'] / 'coverage.json').read_bytes())
    assert coverage['source_events'] == coverage['processed_events'] == 1
    assert coverage['source_input_kind'] == 'stock memory.add(str, infer=True, user_id=history.id)'
    receipts = json.loads((args['out'] / 'mem0-input-receipts.json').read_bytes())
    assert receipts[0]['document_text'] == args['suite'].histories[0].context
    assert receipts[0]['extraction']['status'] == ('valid_empty' if native_output == {'memory': []} else 'valid')
    assert receipts[0]['rolling_buffer'] == [['user', args['suite'].histories[0].context]]
    assert bool(receipts[0]['added']) is (native_output != {'memory': []})
    assert api().verify_history(args['out']) == result
    embedding = [body for path, body in calls if path.endswith('/embeddings')]
    assert any(body['input_type'] == 'query' for body in embedding)
    assert any(body['input_type'] == 'document' for body in embedding) is (native_output != {'memory': []})


@pytest.mark.parametrize('tamper', ['coverage', 'document', 'scope', 'native', 'source', 'database', 'retrieval'])
def test_mem0_rehashed_native_coverage_and_committed_readback_reconciliation(tmp_path, pinned_root, monkeypatch, tamper):
    monkeypatch.setenv('MEM0_TELEMETRY', 'False')
    monkeypatch.setenv('MEM0_DIR', str(tmp_path.parent / 'native-home'))
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    with model_server() as (url, _calls):
        args, _outer = setup_run(url, tmp_path, authored_suite(pinned_root))
        args.update(system='mem0', chunk_chars=4096)
        result = api().run_history(**args)
    out = args['out']
    if tamper == 'database':
        import sqlite3
        path = out / 'stores' / args['history_id'] / 'history.db'
        db = sqlite3.connect(path)
        db.execute('DELETE FROM history')
        db.commit()
        db.close()
    else:
        name = {'coverage': 'coverage.json', 'document': 'mem0-input-receipts.json',
            'scope': 'mem0-memories.json', 'native': 'mem0-native.jsonl',
            'source': 'mem0-source.json', 'retrieval': 'predictions.json'}[tamper]
        path = out / name
        value = json.loads(path.read_bytes())
        if tamper == 'coverage':
            value['processed_events'] = 0
        elif tamper == 'document':
            value[0]['document_text'] = 'forged document'
        elif tamper == 'scope':
            value[0]['user_id'] = 'wrong-history'
        elif tamper == 'native':
            value['response']['choices'][0]['message']['content'] = '{"memory": []}'
        elif tamper == 'source':
            value['files']['memory/main.py'] = '0' * 64
        else:
            value[0]['retrieval']['query'] = 'forged query'
        path.write_bytes(json.dumps(value).encode() + b'\n')
    result['artifact_sha256'][str(path.relative_to(out))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / 'root-status.json').write_bytes(json.dumps(result).encode())
    with pytest.raises(ValueError, match='Mem0|mem0|native|committed|coverage|retrieval'):
        api().verify_history(out)


@pytest.mark.parametrize('fault', ['coverage', 'native_close', 'verifier', 'root_write', 'outer_cap'])
def test_mem0_document_faults_never_publish_completion_and_keep_shared_spend(tmp_path, pinned_root, monkeypatch, fault):
    from gluomem_bench.bench import mem0 as native
    monkeypatch.setenv('MEM0_TELEMETRY', 'False')
    monkeypatch.setenv('MEM0_DIR', str(tmp_path.parent / 'native-home'))
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    module = api()
    closed = []
    original_close = native.close_mem0_memory
    def close(memory):
        closed.append(memory)
        original_close(memory)
        if fault == 'native_close':
            raise OSError('injected native close failure')
    monkeypatch.setattr(native, 'close_mem0_memory', close)
    original_write = module._write
    def write(path, value):
        if fault == 'root_write' and path.name == 'root-status.json':
            raise OSError('injected native root write failure')
        original_write(path, value)
        if fault == 'coverage' and path.name == 'coverage.json':
            target = path.parent / 'mem0-input-receipts.json'
            receipts = json.loads(target.read_bytes())
            receipts[0]['added'][0]['user_id'] = 'forged-scope'
            target.write_bytes(json.dumps(receipts).encode())
    monkeypatch.setattr(module, '_write', write)
    if fault == 'verifier':
        def reject(out, result):
            assert not (out / 'root-status.json').exists()
            raise ValueError('injected native verifier rejection')
        monkeypatch.setattr(module, '_verify_payload', reject)
    with model_server() as (url, calls):
        args, outer = setup_run(url, tmp_path, authored_suite(pinned_root))
        args.update(system='mem0', chunk_chars=4096)
        if fault == 'outer_cap':
            outer.job = type(outer.job)(outer.job.job_id, outer.job.history_id, outer.job.output, 1, 120)
        with pytest.raises(Exception) as caught:
            module.run_history(**args)
    assert len(closed) == 1
    assert not (args['out'] / 'root-status.json').exists()
    assert not (args['out'] / 'scores.json').exists()
    assert caught.value.generation_calls == outer.calls == sum(path.endswith('/chat/completions') for path, _ in calls)
    assert outer.calls == (1 if fault in ('coverage', 'outer_cap') else 2)


@pytest.mark.parametrize('tamper', ['kept_tokens', 'kept_text', 'original_sha256', 'budget', 'dispatch_count'])
def test_rehashed_evidence_receipts_are_verified_against_actual_tokenizer_and_dispatch(tmp_path, pinned_root, tamper):
    with model_server() as (url, _calls):
        args, _outer = setup_run(url, tmp_path, authored_suite(pinned_root))
        args['system'] = 'no_memory'
        result = api().run_history(**args)
    path = args['out'] / 'predictions.json'
    rows = json.loads(path.read_bytes())
    if tamper == 'dispatch_count':
        rows[0]['reader_receipt']['admitted_prompt_tokens'] += 1
    else:
        rows[0]['evidence_receipt'][tamper] = {'kept_tokens': 101, 'kept_text': 'forged',
            'original_sha256': '0' * 64, 'budget': 8192}[tamper]
    path.write_bytes(json.dumps(rows).encode())
    result['artifact_sha256']['predictions.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args['out'] / 'root-status.json').write_bytes(json.dumps(result).encode())
    with pytest.raises(ValueError, match='evidence|token'):
        api().verify_history(args['out'])


@pytest.mark.parametrize('version', ['role-budgets-v5', 'role-budgets-v6'])
def test_authored_real_document_dspy_sqlite_reader_native_score(tmp_path, pinned_root, monkeypatch, version):
    monkeypatch.setenv('GLUOMEM_RACK_API_KEY', 'offline-placeholder')
    with model_server() as (url, calls):
        suite = authored_suite(pinned_root)
        args, outer = setup_run(url, tmp_path, suite)
        args['profile'] = args['profile'].model_copy(update={'profile_version': version})
        result = api().run_history(**args)
    assert result['complete'] is True
    assert result['evidence_kind'] == 'authored-offline-integration'
    assert result['case_ids'] == args['case_ids']
    assert api().verify_history(args['out']) == result
    rows = json.loads((args['out'] / 'predictions.json').read_bytes())
    assert rows[0]['prediction'] == 'cobalt'
    assert rows[0]['score']['metric'] == 'substring_exact_match'
    assert rows[0]['score']['value'] == 1.0  # authored server, not model-quality evidence
    production = [json.loads(x) for x in (args['out'] / 'production-raw.jsonl').read_bytes().splitlines()]
    assert {r['task'] for r in production} >= {'event_claims', 'consolidate'}
    extracts = [r for r in production if r['task'] == 'event_claims']
    assert all('1970' not in json.dumps(r['messages']) for r in extracts)
    assert all('What is the station' not in json.dumps(r['messages']) for r in extracts)
    inputs = json.loads((args['out'] / 'source-inputs.json').read_bytes())
    for event in inputs:
        assert event['source_type'] == 'document'
        assert event['created_at'] is None and event['event_time'] is None
        assert not event['messages']
        assert 'authored' not in json.dumps(event) and '197K' not in json.dumps(event)
    coverage = json.loads((args['out'] / 'coverage.json').read_bytes())
    assert coverage['source_events'] == coverage['processed_events'] == 2
    assert outer.calls == result['generation_calls'] == len(production) + 1
    assert any(path.endswith('/embeddings') for path, _ in calls)
    assert not list((args['out'] / 'stores').glob('*-wal'))
    assert not list((args['out'] / 'stores').glob('*-shm'))


@pytest.mark.parametrize('system', ['no_memory', 'bm25', 'dense', 'hybrid', 'full_context'])
def test_raw_controls_never_fabricate_document_dates_or_roles(tmp_path, pinned_root, system):
    with model_server() as (url, calls):
        suite = authored_suite(pinned_root)
        args, outer = setup_run(url, tmp_path, suite)
        args['system'] = system
        result = api().run_history(**args)
    assert result['generation_calls'] == outer.calls == 1
    assert not (args['out'] / 'production-raw.jsonl').exists()
    row, = json.loads((args['out'] / 'predictions.json').read_bytes())
    assert '1970' not in row['evidence']
    assert 'document:' not in row['evidence']
    if system == 'no_memory':
        assert row['evidence'] == ''
    elif system == 'full_context':
        assert row['evidence'] == suite.histories[0].context
    else:
        assert 'Station locker' in row['evidence']
    assert api().verify_history(args['out']) == result


def test_task_specific_judge_routes_shared_guard_and_summary_4096_is_not_squeezed(tmp_path, pinned_root):
    from gluomem_bench.bench.mab_suite import MABSuite
    from gluomem_bench.bench.rack import RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import WorkloadBudget
    import httpx
    suite = MABSuite.from_directory(pinned_root)
    scorer = suite.scorer()
    case = next(c for c in suite.cases if c.source == 'longmemeval_s*')
    sent = []
    def serve(request):
        body = json.loads(request.content)
        sent.append((request.url.path, body))
        if request.url.path == '/tokenize':
            return httpx.Response(200, json={'count': 100})
        return httpx.Response(200, json={'model': 'example/offline', 'choices': [
            {'finish_reason': 'stop', 'message': {'content': 'yes'}}]})
    profile = RackProfile(base_url='http://localhost:8000/v1', model='example/offline')
    client = RackClient(profile, receipt=tmp_path / 'judge.jsonl',
        budget=WorkloadBudget(), transport=httpx.MockTransport(serve))
    qualification = {'protocol': 'adapted-judge-role-budget@v1', 'model': profile.model,
                     'temperature': 0, 'output_budget': profile.output_budget('rubric')}
    try:
        score, judgments = api().score_prediction(scorer, case, 'candidate', client=client,
                                                 judge_qualification=qualification)
        assert score.metric == 'llm_judge_accuracy'
        assert score.details['judge_identity'] == 'adapted-judge'
        assert len(judgments) == client.budget.calls == 1
        request, = scorer.judge_requests(case, 'candidate')
        assert sent[-1][1]['messages'] == [{'role': 'user', 'content': request.prompt}]
        summary = next(c for c in suite.cases if c.source == 'infbench_sum_eng_shots2')
        prior = len(sent)
        with pytest.raises(ValueError, match='4096.*frozen.*judge|judge.*4096'):
            api().score_prediction(scorer, summary, 'candidate', client=client,
                                   judge_qualification=qualification)
        assert len(sent) == prior
    finally:
        client.close()


@pytest.mark.parametrize('fault', ['verifier', 'root_write'])
def test_root_is_only_published_after_verification_and_retains_spend_on_fault(
    tmp_path, pinned_root, monkeypatch, fault
):
    module = api()
    observed = []
    if fault == 'verifier':
        def reject(out, result):
            observed.append(out)
            assert not (out / 'root-status.json').exists()
            assert list((out / 'stores').glob('*.ingest.json'))
            raise ValueError('injected external verifier rejection')
        monkeypatch.setattr(module, '_verify_payload', reject)
    else:
        original = module._write
        def fail_root(path, value):
            if path.name == 'root-status.json':
                observed.append(path)
                raise OSError('injected root result write failure')
            return original(path, value)
        monkeypatch.setattr(module, '_write', fail_root)
    monkeypatch.setenv('GLUOMEM_RACK_API_KEY', 'offline-placeholder')
    with model_server() as (url, _calls):
        args, outer = setup_run(url, tmp_path, authored_suite(pinned_root))
        with pytest.raises((ValueError, OSError), match='injected') as error:
            module.run_history(**args)
    assert observed
    assert error.value.generation_calls == outer.calls > 0
    assert not (args['out'] / 'root-status.json').exists()
    assert list((args['out'] / 'stores').glob('*.ingest.json'))  # cannot promote the root
    failure = json.loads((args['out'] / 'failure.json').read_bytes())
    assert failure['generation_calls'] == outer.calls
    with pytest.raises((ValueError, FileNotFoundError)):
        module.verify_history(args['out'])


@pytest.mark.parametrize('invocation', ['api', 'cli', 'mem0_cli', 'verifier_fault', 'root_write_fault', 'root_failure_write_fault', 'deadline_fault'])
def test_scoped_hash_approval_executes_selected_native_occurrence_and_refuses_drift(
    tmp_path, pinned_root, capsys, invocation, monkeypatch
):
    from gluomem_bench.bench.mab_suite import MABSuite
    from gluomem_bench.bench.rack import RackProfile
    from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe
    from gluomem_bench.bench.comprehensive_driver import PROTOCOL as ENVELOPE
    suite = MABSuite.from_directory(pinned_root)
    case = next(c for c in suite.cases if c.source == 'factconsolidation_sh_6k')
    module = api()
    with model_server() as (url, calls):
        spec = {'protocol': ENVELOPE, 'execution_kind': module.PROTOCOL,
                'output_root': str(tmp_path / 'campaign'), 'mab_root': str(pinned_root),
                'profile': RackProfile(base_url=url, model='example/offline').model_dump(mode='json'),
                'embedding_url': url, 'embedding_recipe': EmbeddingRecipe().as_dict(),
                'case_ids': [case.sample.id], 'systems': ['no_memory'], 'chunk_chars': 1000,
                'top_k': 5, 'evidence_token_budget': 4096, 'judge_qualification': None,
                'policy': {'max_workers': 1, 'max_in_flight': 1, 'max_in_flight_tokens': 131072,
                           'max_calls': 2, 'max_seconds': 120, 'stop_on_failure': True},
                'jobs': [{'job_id': 'j1', 'history_id': case.sample.conv_id,
                          'system': 'no_memory', 'case_ids': [case.sample.id],
                          'output': str(tmp_path / 'campaign' / 'jobs' / 'j1'),
                          'call_budget': 2, 'seconds_budget': 60}],
                'files': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in module.required_files(pinned_root)]}
        if invocation == 'mem0_cli':
            monkeypatch.setenv('MEM0_TELEMETRY', 'False')
            monkeypatch.setenv('MEM0_DIR', str(tmp_path.parent / 'native-home'))
            monkeypatch.delenv('OPENAI_API_KEY', raising=False)
            monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
            spec['systems'] = ['mem0']
            spec['jobs'][0]['system'] = 'mem0'
            spec['chunk_chars'] = 1000000
            spec['profile']['max_input_chars'] = 2000000
        spec_path, approval_path = tmp_path / 'spec.json', tmp_path / 'approval.json'
        spec_path.write_bytes(json.dumps(spec).encode())
        approval = {'approved': True, 'reviewer_kind': 'AI-assisted',
                    'spec_sha256': hashlib.sha256(spec_path.read_bytes()).hexdigest()}
        approval_path.write_bytes(json.dumps(approval).encode())
        # Approval binds bytes, not merely a boolean.
        spec_path.write_bytes(spec_path.read_bytes() + b' ')
        with pytest.raises(ValueError, match='approval'):
            module.execute(spec_path, approval_path)
        assert not calls and not Path(spec['output_root']).exists()
        spec_path.write_bytes(json.dumps(spec).encode())
        for cap in (None, 8192, True):
            invalid = dict(spec)
            if cap is None:
                invalid.pop('evidence_token_budget')
            else:
                invalid['evidence_token_budget'] = cap
            spec_path.write_bytes(json.dumps(invalid).encode())
            approval_path.write_bytes(json.dumps({**approval,
                'spec_sha256': hashlib.sha256(spec_path.read_bytes()).hexdigest()}).encode())
            with pytest.raises(ValueError, match='evidence.*4096|4096.*evidence'):
                module.execute(spec_path, approval_path)
            assert not calls and not Path(spec['output_root']).exists()
        spec_path.write_bytes(json.dumps(spec).encode())
        approval_path.write_bytes(json.dumps(approval).encode())
        if invocation.endswith('_fault'):
            root = Path(spec['output_root'])
            if invocation == 'deadline_fault':
                original_verify = module._verify_campaign_payload
                def consume_deadline(out, result):
                    original_verify(out, result)
                    monkeypatch.setattr(module, '_clock', lambda: float('inf'), raising=False)
                monkeypatch.setattr(module, '_verify_campaign_payload', consume_deadline)
            elif invocation == 'verifier_fault':
                def reject_root(out, _result):
                    assert not (out / 'root-status.json').exists()
                    assert (Path(spec['jobs'][0]['output']) / 'root-status.json').exists()
                    raise ValueError('injected campaign verifier rejection')
                monkeypatch.setattr(module, '_verify_campaign_payload', reject_root)
            else:
                original_write = module._write
                def reject_write(path, value):
                    if path.parent == root and (path.name == 'root-status.pending.json' or (
                            invocation == 'root_failure_write_fault' and path.name == 'failure.json')):
                        raise OSError('injected campaign receipt write failure')
                    return original_write(path, value)
                monkeypatch.setattr(module, '_write', reject_write)
            with pytest.raises((ValueError, OSError, RuntimeError), match='injected|deadline') as caught:
                module.execute(spec_path, approval_path)
            assert caught.value.generation_calls == 1
            assert not (root / 'root-status.json').exists()
            assert (Path(spec['jobs'][0]['output']) / 'root-status.json').exists()
            progress = json.loads((root / 'coordination' / 'progress.json').read_bytes())
            assert progress['accounting']['calls'] == 1
            assert progress['jobs'][0]['status'] == 'complete'
            with pytest.raises((ValueError, FileNotFoundError)):
                module.verify_campaign(root)
            return
        if invocation == 'api':
            result = module.execute(spec_path, approval_path)
        else:
            assert module.main(['--spec', str(spec_path), '--approval', str(approval_path)]) == 0
            result = json.loads(capsys.readouterr().out)
    assert result['complete'] is True
    assert result['generation_calls'] == (2 if invocation == 'mem0_cli' else 1)
    assert result['jobs'][0]['case_ids'] == [case.sample.id]
    row, = json.loads((Path(spec['jobs'][0]['output']) / 'predictions.json').read_bytes())
    assert row['source_id'] == case.source_id
    assert row['question'] == suite.reader_question(case)
    assert 'larger serial number' in row['question']
    assert row['score']['metric'] == 'substring_exact_match'
    assert module.verify_campaign(Path(spec['output_root'])) == result
    assert module.main(['--verify-root', spec['output_root']]) == 0
    assert json.loads(capsys.readouterr().out) == result
    leaf_out = Path(spec['jobs'][0]['output'])
    protocol_path = leaf_out / 'protocol.json'
    original_protocol = protocol_path.read_bytes()
    protocol = json.loads(original_protocol)
    protocol['top_k'] += 1
    protocol_path.write_bytes(json.dumps(protocol).encode())
    leaf_path = leaf_out / 'root-status.json'
    original_leaf = leaf_path.read_bytes()
    leaf = json.loads(original_leaf)
    leaf['artifact_sha256']['protocol.json'] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    leaf_path.write_bytes(json.dumps(leaf).encode())
    forged = json.loads(json.dumps(result))
    for path in (protocol_path, leaf_path):
        forged['artifact_sha256'][path.relative_to(Path(spec['output_root'])).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    root_path = Path(spec['output_root']) / 'root-status.json'
    root_path.write_bytes(json.dumps(forged).encode())
    with pytest.raises(ValueError, match='frozen|binding|effective'):
        module.verify_campaign(Path(spec['output_root']))
    protocol_path.write_bytes(original_protocol)
    leaf_path.write_bytes(original_leaf)
    result['jobs'] = []
    result['generation_calls'] = 0
    (Path(spec['output_root']) / 'root-status.json').write_bytes(json.dumps(result).encode())
    with pytest.raises(ValueError, match='frozen|coverage|matrix'):
        module.verify_campaign(Path(spec['output_root']))


def test_native_rescore_uses_saved_pinned_sources_not_dataset_and_rejects_rehashed_wrong_score(
    tmp_path, pinned_root, monkeypatch
):
    with model_server() as (url, _calls):
        args, _outer = setup_run(url, tmp_path, authored_suite(pinned_root))
        args['system'] = 'no_memory'
        result = api().run_history(**args)
    # Deliberately update the preservation digest: rescoring must catch a value
    # inconsistent with native post_process, not merely a corrupt file hash.
    path = args['out'] / 'predictions.json'
    rows = json.loads(path.read_bytes())
    rows[0]['score']['value'] = 0.0
    path.write_bytes(json.dumps(rows).encode())
    result['artifact_sha256']['predictions.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args['out'] / 'root-status.json').write_bytes(json.dumps(result).encode())
    original = Path.read_bytes
    def block_dataset(path):
        if path.suffix == '.parquet' or path.is_relative_to(pinned_root):
            raise AssertionError('offline verifier opened external dataset/source snapshot')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', block_dataset)
    with pytest.raises(ValueError, match='native.*score|score.*native'):
        api().verify_history(args['out'])


@pytest.mark.parametrize('corruption', ['input', 'processed'])
def test_persistent_input_and_processing_readback_gate_runs_before_reader(
    tmp_path, pinned_root, monkeypatch, corruption
):
    module = api()
    original = module._write
    def corrupt_after_coverage(path, value):
        original(path, value)
        if path.name != 'coverage.json':
            return
        if corruption == 'input':
            target = path.parent / 'source-inputs.json'
            inputs = json.loads(target.read_bytes())
            inputs[0]['content'] = 'forged input'
            target.write_bytes(json.dumps(inputs).encode())
        else:
            value['processed_events'] = 0
            path.write_bytes(json.dumps(value).encode())
    monkeypatch.setattr(module, '_write', corrupt_after_coverage)
    monkeypatch.setenv('GLUOMEM_RACK_API_KEY', 'offline-placeholder')
    with model_server() as (url, calls):
        args, outer = setup_run(url, tmp_path, authored_suite(pinned_root))
        with pytest.raises(ValueError, match='coverage|input'):
            module.run_history(**args)
    generations = [body for path, body in calls if path.endswith('/chat/completions')]
    assert generations
    assert all('supporting verbatim quote' in json.dumps(body) or 'target_frame_id' in json.dumps(body)
               for body in generations)
    assert not (args['out'] / 'root-status.json').exists()
    assert outer.calls == len(generations)


def test_all_four_released_families_native_metrics_and_saved_summary_without_generation(pinned_root):
    from gluomem_bench.bench.mab_suite import MABSuite, JudgeResponse, primary_metric
    suite = MABSuite.from_directory(pinned_root)
    assert len(suite.histories) == 146 and len(suite.cases) == 3671
    assert len({c.sample.id for c in suite.cases}) == 3671
    scorer = suite.scorer()
    by_source = {c.source: c for c in reversed(suite.cases)}
    seen = set()
    for source, case in by_source.items():
        candidate = case.answers[0]
        if source.startswith('recsys_'):
            mapping = json.loads((pinned_root / 'entity2id.json').read_bytes())
            name = next(name for name, identifier in mapping.items() if identifier == int(case.answers[0]))
            candidate = '1.' + scorer.upstream['extract_movie_name'](name)
        requests = scorer.judge_requests(case, candidate)
        outputs = {'answer': 'yes', 'fluency': '{"fluency":1}',
                   'recall': json.dumps({'recall': len(case.metadata.get('keypoints') or [])}),
                   'precision': '{"precision":1,"sentence_count":1}'}
        judgments = [JudgeResponse(case_id=r.case_id, role=r.role,
            prediction_sha256=r.prediction_sha256, prompt_sha256=r.prompt_sha256,
            model=r.reference_model, output=outputs[r.role]) for r in requests]
        score, bound = api().score_prediction(scorer, case, candidate, judgments=judgments)
        assert score.metric == primary_metric(source)
        assert score.value == 1.0  # explicit GOLD-ECHO calibration, not model answers
        assert len(bound) == len(requests)
        seen.add(case.family)
    assert len(seen) == 4 and len(by_source) == 22


@pytest.mark.parametrize('failure', ['truncation', 'fullfit_overflow', 'call_cap', 'token_cap'])
def test_whole_system_guards_never_turn_failures_into_scores(tmp_path, pinned_root, monkeypatch, failure):
    monkeypatch.setenv('GLUOMEM_RACK_API_KEY', 'offline-placeholder')
    from gluomem_bench.bench.parallel_campaign import RequestLimiter
    with model_server(finish='length' if failure == 'truncation' else 'stop',
                      overflow=failure == 'fullfit_overflow') as (url, calls):
        args, outer = setup_run(url, tmp_path, authored_suite(pinned_root))
        if failure == 'fullfit_overflow':
            args['system'] = 'full_context'
        elif failure == 'call_cap':
            args['profile'] = args['profile'].model_copy(update={'max_generation_calls': 1})
        elif failure == 'token_cap':
            outer.limiter = RequestLimiter(max_in_flight=1, max_in_flight_tokens=1)
        if failure == 'fullfit_overflow':
            result = api().run_history(**args)
            assert result['complete'] is True
            assert result['generation_calls'] == outer.calls == 0
            row, = json.loads((args['out'] / 'predictions.json').read_bytes())
            assert row['status'] == 'not_applicable' and row['reason'] == 'context_overflow'
            assert row['prediction'] is None and row['score'] is None
            assert row['evidence'] == args['suite'].histories[0].context
            report = json.loads((args['out'] / 'scores.json').read_bytes())
            assert report['not_applicable'] == [{'case_id': row['case_id'], 'reason': 'context_overflow'}]
            source, = report['sources'].values()
            assert source['completed'] == 0
            assert api().verify_history(args['out']) == result
            return
        with pytest.raises(Exception) as caught:
            api().run_history(**args)
    assert not (args['out'] / 'root-status.json').exists()
    assert not (args['out'] / 'scores.json').exists()
    failure_receipt = json.loads((args['out'] / 'failure.json').read_bytes())
    assert failure_receipt['generation_calls'] == outer.calls == caught.value.generation_calls
    dispatched = sum(path.endswith('/chat/completions') for path, _ in calls)
    assert dispatched == outer.calls
    if failure in {'fullfit_overflow', 'token_cap'}:
        assert dispatched == 0
    else:
        assert dispatched == 1


@pytest.mark.parametrize('tamper', ['receipt', 'count', 'inventory'])
def test_offline_verifier_reconciles_native_reader_receipts_and_exact_inventory(tmp_path, pinned_root, tamper):
    with model_server() as (url, _calls):
        args, _outer = setup_run(url, tmp_path, authored_suite(pinned_root))
        args['system'] = 'no_memory'
        result = api().run_history(**args)
    out = args['out']
    if tamper == 'receipt':
        path = out / 'calls.jsonl'
        calls = [json.loads(line) for line in path.read_bytes().splitlines()]
        for call in calls:
            if call['url'].endswith('/chat/completions'):
                call['response']['choices'][0]['message']['content'] = 'forged wire answer'
        path.write_bytes(b''.join(json.dumps(c).encode() + b'\n' for c in calls))
        result['artifact_sha256']['calls.jsonl'] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif tamper == 'count':
        result['generation_calls'] += 1
    else:
        result['artifact_sha256'].pop('source-inputs.json')
    (out / 'root-status.json').write_bytes(json.dumps(result).encode())
    with pytest.raises(ValueError, match='receipt|reconcil|inventory'):
        api().verify_history(out)


def test_native_worker_cannot_bypass_scoped_approval_entrypoint(tmp_path, pinned_root):
    with model_server() as (url, calls):
        args, _outer = setup_run(url, tmp_path, authored_suite(pinned_root))
        args['authored'] = False
        # A claimed receipt must not grant execution authority.
        args['suite'].receipt = {'files': 'self-declared'}
        with pytest.raises(ValueError, match='execute|approval'):
            api().run_history(**args)
        assert not calls and not args['out'].exists()


def test_admission_is_explicit_about_frozen_document_and_summary_bridges():
    report = api().admission()
    assert set(report['executable_systems']) == {'gluomem', 'mem0', 'cognee', 'no_memory', 'bm25', 'dense', 'hybrid', 'full_context'}
    assert 'mem0' not in report['unadmitted_systems']
    assert 'cognee' not in report['unadmitted_systems']
    assert report['cognee_document_api']['qualification_sha256'] == api().COGNEE_QUALIFICATION_SHA256
    assert report['cognee_document_api']['full_only'] is True
    assert '4096' in report['live_summary_judge_gate']
    assert report['quality_claims_admitted'] is False
    assert report['full_campaign'] == {'campaign_mode': 'full-campaign-v1',
        'profile_version': 'role-budgets-v6', 'recommended_workers': 8, 'max_workers': 32,
        'stop_on_failure': False, 'reader_tokens': 4096, 'judge_tokens': 4096,
        'selection': 'all pinned history and question occurrences; no sampled full mode'}


def test_selected_occurrence_report_keeps_native_source_metrics_separate(tmp_path, pinned_root):
    with model_server() as (url, _calls):
        args, _outer = setup_run(url, tmp_path, authored_suite(pinned_root))
        args['system'] = 'no_memory'
        result = api().run_history(**args)
    report = json.loads((args['out'] / 'scores.json').read_bytes())
    assert set(report) == {'sources'}
    source, = report['sources'].values()
    assert source['metric'] == 'substring_exact_match'
    assert source['planned'] == source['completed'] == 1
    assert source['missing'] == 0
    assert api().verify_history(args['out']) == result


def test_reader_uses_byte_pinned_native_system_and_query_templates(tmp_path, pinned_root):
    suite = authored_suite(pinned_root)
    with model_server() as (url, calls):
        args, _outer = setup_run(url, tmp_path, suite)
        args['system'] = 'no_memory'
        api().run_history(**args)
    reader, = [body for path, body in calls if path.endswith('/chat/completions')]
    assert reader['messages'][0] == {'role': 'system', 'content':
        suite.templates()['get_template'](suite.cases[0].source, 'system', 'rag')}
    query = json.loads(reader['messages'][1]['content'])
    assert query['question'] == suite.reader_question(suite.cases[0])
    assert query['evidence'] == ''


def test_cli_reports_actual_admission_without_clients(capsys):
    assert api().main(['--admission']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == api().admission()


@pytest.mark.parametrize('workers,fault', [(w, f) for w in (1, 8) for f in
    (None, 'verifier', 'root_write')] + [(1, f) for f in
    ('identity', 'tokenizer', 'unrun', 'root_readback', 'judge_schema', 'not_applicable',
     'embedding_timeout', 'embedding_oserror', 'native_timeout', 'explicit_shared', 'evidenced_shared',
     'shared_budget', 'legacy_timeout')])
def test_full_mode_middle_failure_continues_real_jobs_and_publishes_only_verified_scope(
    tmp_path, pinned_root, monkeypatch, fault, workers
):
    from gluomem_bench.bench.mab_suite import MABSuite
    from gluomem_bench.bench.rack import RackProfile
    from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe
    from gluomem_bench.bench.comprehensive_driver import PROTOCOL as ENVELOPE
    module = api()
    source = 'longmemeval_s*' if fault == 'judge_schema' else 'ruler_qa1_197K'
    system = ('full_context' if fault == 'not_applicable' else
              'dense' if fault in ('embedding_timeout', 'embedding_oserror') else 'no_memory')
    request_fault = fault in ('embedding_timeout', 'embedding_oserror', 'native_timeout',
                              'explicit_shared', 'evidenced_shared', 'shared_budget', 'legacy_timeout')
    attempts, budgets = [], []
    if request_fault:
        import httpx
        from gluomem_bench.bench import embedding_runtime
        from gluomem_bench.bench.cognee_retriever import NativeWorkloadBudget
        original_init = NativeWorkloadBudget.__init__
        def capture_budget(budget, *args, **kwargs):
            original_init(budget, *args, **kwargs)
            budgets.append(budget)
        monkeypatch.setattr(NativeWorkloadBudget, '__init__', capture_budget)
        original_encode = embedding_runtime.urlopen
        original_request = httpx.Client.request
        class NativeTimeout(httpx.ReadTimeout):
            pass
        def encode(request, **kwargs):
            body = json.loads(request.data)
            if fault.startswith('embedding_') and 'Question 1' in json.dumps(body):
                attempts.append(body)
                error = TimeoutError if fault == 'embedding_timeout' else OSError
                raise error('authored single embedding request failure')
            return original_encode(request, **kwargs)
        def request(client, method, url, **kwargs):
            response = original_request(client, method, url, **kwargs)
            body = kwargs.get('json')
            if (not fault.startswith('embedding_') and str(url).endswith('/chat/completions')
                    and 'Question 1' in json.dumps(body)):
                attempts.append(body)
                if fault == 'shared_budget':
                    budgets[-1].outer_budget.budget.failed = True
                if fault == 'explicit_shared':
                    raise module.SharedRackError('authored shared identity fault')
                if fault == 'evidenced_shared':
                    error = TimeoutError('authored evidenced shared fault')
                    error.__dict__['shared_failure'] = True
                    raise error
                raise NativeTimeout('authored single native generation timeout')
            return response
        monkeypatch.setattr(embedding_runtime, 'urlopen', encode)
        monkeypatch.setattr(httpx.Client, 'request', request)
    suite = MABSuite.from_rows({'Accurate_Retrieval': [
        {'context': 'Station locker code is cobalt.', 'questions': [f'Question {i}'],
         'answers': [['cobalt']], 'metadata': {'source': source, 'qa_pair_ids': [f'q{i}'],
            'question_types': ['single-session-user'], 'question_ids': [f'q{i}']}}
        for i in range(3)]})
    suite.root, suite.receipt = pinned_root, {'authored_integration_only': True}
    monkeypatch.setattr(MABSuite, 'from_directory', lambda root: suite)
    def finish(body):
        if request_fault or fault in ('judge_schema', 'not_applicable'):
            return 'stop'
        return 'length' if 'Question 1' in json.dumps(body['messages']) else 'stop'
    def response_model(body):
        return 'wrong-model' if fault == 'identity' and 'Question 1' in json.dumps(body) else 'example/offline'
    def token_count(messages):
        if fault == 'not_applicable':
            return 200000
        return None if fault == 'tokenizer' and 'Question 1' in json.dumps(messages) else 100
    def output(body):
        if fault == 'judge_schema' and len(body['messages']) == 1:
            return 'not a verdict' if 'Question 1' in json.dumps(body) else 'yes'
        return 'cobalt'
    with model_server(finish=finish, response_model=response_model, token_count=token_count,
                      output_for=output) as (url, calls):
        root = tmp_path / 'full-campaign'
        profile = RackProfile(base_url=url, model='example/offline', profile_version='role-budgets-v6',
            generation_budgets=dict(extraction=4096, consolidation=4096, reader=4096, judge=4096))
        spec = {'protocol': ENVELOPE, 'execution_kind': module.PROTOCOL,
            'campaign_mode': 'full-campaign-v1', 'output_root': str(root), 'mab_root': str(pinned_root),
            'case_manifest_sha256': module._digest([__import__('dataclasses').asdict(c) for c in suite.cases]),
            'profile': profile.model_dump(mode='json'), 'embedding_url': url,
            'embedding_recipe': EmbeddingRecipe().as_dict(),
            'case_ids': [c.sample.id for c in suite.cases], 'systems': [system],
            'chunk_chars': 1000, 'top_k': 5, 'evidence_token_budget': 4096,
            'judge_qualification': ({'protocol': 'native-judge-parameters@v2', 'model': 'example/offline',
                'temperatures': [0, 0.1], 'output_budget': 4096} if fault == 'judge_schema' else None),
            'policy': {'max_workers': workers, 'max_in_flight': workers, 'max_in_flight_tokens': 1048576,
                       'max_calls': 3 if fault == 'unrun' else 6, 'max_seconds': 120, 'stop_on_failure': False},
            'jobs': [{'job_id': f'j{i}', 'history_id': c.sample.conv_id, 'system': system,
                      'case_ids': [c.sample.id], 'output': str(root / 'jobs' / f'j{i}'),
                      'call_budget': 2, 'seconds_budget': 60} for i, c in enumerate(suite.cases)],
            'files': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in module.required_files(pinned_root)]}
        if fault == 'legacy_timeout':
            spec['campaign_mode'] = 'legacy-strict-v1'
            spec['policy']['stop_on_failure'] = True
        spec_path, approval_path = tmp_path / 'spec.json', tmp_path / 'approval.json'
        spec_path.write_bytes(json.dumps(spec).encode())
        approval_path.write_bytes(json.dumps({'approved': True, 'reviewer_kind': 'AI-assisted',
            'spec_sha256': hashlib.sha256(spec_path.read_bytes()).hexdigest()}).encode())
        if fault is None and workers == 1:
            import copy
            for invalidity in ('partial', 'reader', 'v5', 'workers', 'stop'):
                invalid = copy.deepcopy(spec)
                if invalidity == 'partial':
                    invalid['case_ids'] = invalid['case_ids'][:-1]
                elif invalidity == 'reader':
                    invalid['profile']['generation_budgets']['reader'] = 1024
                elif invalidity == 'v5':
                    invalid['profile']['profile_version'] = 'role-budgets-v5'
                    invalid['profile']['generation_budgets']['judge'] = 1024
                elif invalidity == 'workers':
                    invalid['policy']['max_workers'] = 33
                else:
                    invalid['policy']['stop_on_failure'] = True
                spec_path.write_bytes(json.dumps(invalid).encode())
                approval_path.write_bytes(json.dumps({'approved': True, 'reviewer_kind': 'AI-assisted',
                    'spec_sha256': hashlib.sha256(spec_path.read_bytes()).hexdigest()}).encode())
                with pytest.raises(ValueError, match='full campaign'):
                    module.execute(spec_path, approval_path)
                assert not calls and not root.exists()
            spec_path.write_bytes(json.dumps(spec).encode())
            approval_path.write_bytes(json.dumps({'approved': True, 'reviewer_kind': 'AI-assisted',
                'spec_sha256': hashlib.sha256(spec_path.read_bytes()).hexdigest()}).encode())
        if fault == 'verifier':
            monkeypatch.setattr(module, '_verify_campaign_payload',
                lambda *args: (_ for _ in ()).throw(ValueError('injected verifier')))
        elif fault in ('root_write', 'root_readback'):
            original = module._write
            def reject(path, value):
                if path == root / 'root-status.pending.json':
                    if fault == 'root_write':
                        raise OSError('injected root write')
                    return original(path, {**value, 'generation_calls': 999})
                return original(path, value)
            monkeypatch.setattr(module, '_write', reject)
        if fault in ('verifier', 'root_write', 'root_readback'):
            with pytest.raises((ValueError, OSError), match='injected|readback') as caught:
                module.execute(spec_path, approval_path)
            assert caught.value.generation_calls == 3
            assert not (root / 'root-status.json').exists()
            return
        if fault == 'legacy_timeout':
            with pytest.raises(RuntimeError, match='campaign incomplete'):
                module.execute(spec_path, approval_path)
            progress = json.loads((root / 'coordination/progress.json').read_bytes())
            assert [j['status'] for j in progress['jobs']] == ['complete', 'failed', 'unrun']
            assert progress['accounting']['calls'] == 2
            assert len(attempts) == 1 and budgets[1].failed
            assert not (root / 'jobs/j2').exists()
            assert not (root / 'root-status.json').exists()
            assert json.loads((root / 'failure.json').read_bytes())['generation_calls'] == 2
            return
        result = module.execute(spec_path, approval_path)
    if request_fault:
        shared = fault in ('explicit_shared', 'evidenced_shared', 'shared_budget')
        assert [j['status'] for j in result['jobs']] == [
            'complete', 'failed', 'unrun' if shared else 'complete']
        assert len(attempts) == 1  # Never retry the failed request/job.
        failed = root / 'jobs' / 'j1'
        receipt = json.loads((failed / 'failure.json').read_bytes())
        assert receipt['generation_calls'] == (0 if fault.startswith('embedding_') else 1)
        assert budgets[1].failed
        with pytest.raises(Exception):
            budgets[1].check()
        trace = [json.loads(line) for line in (failed / 'calls.jsonl').read_text().splitlines()]
        assert trace[0]['url'].endswith('/models') and trace[0]['status'] == 200
        assert (failed / 'cases.json').exists() and (failed / 'source-inputs.json').exists()
        assert not (failed / 'scores.json').exists()
        assert not (failed / 'predictions.json').exists()
        assert not (failed / 'root-status.json').exists()
        assert result['generation_calls'] == (2 if shared or fault.startswith('embedding_') else 3)
        assert sum(path.endswith('/chat/completions') for path, _ in calls) == result['generation_calls']
        assert result['coverage'][system]['graded'] == (1 if shared else 2)
        assert result['coverage'][system]['failed'] == 1
        assert result['coverage'][system]['unrun'] == int(shared)
        assert module.verify_campaign(root) == result
        assert result['complete'] is False
        return
    if fault == 'not_applicable':
        assert result['complete'] is True
        assert result['coverage'][system] == dict(planned=3, complete=3, failed=0, unrun=0,
            graded=0, not_applicable=3, eligible_case_ids=[])
        assert result['native_metrics'][system]['sources'][source]['completed_only_mean'] is None
        assert result['generation_calls'] == 0
        assert module.verify_campaign(root) == result
        return
    if fault in ('identity', 'tokenizer', 'unrun'):
        assert [j['status'] for j in result['jobs']] == ['complete', 'failed', 'unrun']
        assert result['generation_calls'] == (1 if fault == 'tokenizer' else 2)
        assert result['coverage']['no_memory']['graded'] == 1
        assert result['coverage']['no_memory']['unrun'] == 1
        assert module.verify_campaign(root) == result
        assert not (root / 'jobs' / 'j2').exists()
        return
    assert result['complete'] is False
    assert result['coordinator_complete'] is True
    assert [j['status'] for j in result['jobs']] == ['complete', 'failed', 'complete']
    assert result['generation_calls'] == (6 if fault == 'judge_schema' else 3)
    assert result['coverage']['no_memory'] == dict(planned=3, complete=2, failed=1, unrun=0,
        graded=2, not_applicable=0, eligible_case_ids=[suite.cases[i].sample.id for i in (0, 2)])
    metric = result['native_metrics']['no_memory']['sources'][source]
    assert metric['planned'] == 3 and metric['completed'] == 2
    assert metric['missing_ids'] == [suite.cases[1].sample.id]
    assert module.verify_campaign(root) == result
    assert not (root / 'jobs' / 'j1' / 'root-status.json').exists()
    assert (root / 'jobs' / 'j1' / 'failure.json').exists()
    assert sum(path.endswith('/chat/completions') for path, _ in calls) == result['generation_calls']
    if fault is None and workers == 1:
        path = root / 'root-status.json'
        for key in ('complete', 'coverage', 'native_metrics'):
            forged = json.loads(json.dumps(result))
            if key == 'complete':
                forged['complete'] = True
            elif key == 'coverage':
                forged[key]['no_memory']['graded'] += 1
            else:
                forged[key]['no_memory']['sources'][source]['completed'] += 1
            path.write_bytes(json.dumps(forged).encode())
            with pytest.raises(ValueError, match='completion|coverage|metrics'):
                module.verify_campaign(root)
        path.write_bytes(json.dumps(result).encode())
        leaf_out = root / 'jobs' / 'j0'
        cases_path = leaf_out / 'cases.json'
        saved_cases = json.loads(cases_path.read_bytes())
        saved_cases[0]['answers'].append('forged alternate')
        cases_path.write_bytes(json.dumps(saved_cases).encode())
        leaf_path = leaf_out / 'root-status.json'
        leaf = json.loads(leaf_path.read_bytes())
        leaf['artifact_sha256']['cases.json'] = module._hash(cases_path)
        leaf_path.write_bytes(json.dumps(leaf).encode())
        for changed in (cases_path, leaf_path):
            result['artifact_sha256'][changed.relative_to(root).as_posix()] = module._hash(changed)
        path.write_bytes(json.dumps(result).encode())
        with pytest.raises(ValueError, match='planned|frozen.*case'):
            module.verify_campaign(root)


@pytest.mark.parametrize('tamper', ['temperature', 'max_tokens', 'judge_token_count'])
@pytest.mark.parametrize('source', ['longmemeval_s*', 'infbench_sum_eng_shots2'])
def test_v6_saved_native_judge_parameters_cannot_be_rehashed_away(
    tmp_path, pinned_root, tamper, source
):
    from gluomem_bench.bench.mab_suite import MABSuite
    from gluomem_bench.bench.rack import RackProfile
    family = 'Accurate_Retrieval' if source == 'longmemeval_s*' else 'Long_Range_Understanding'
    suite = MABSuite.from_rows({family: [{'context': 'Only source text.',
        'questions': ['Private question'], 'answers': [['Private answer']],
        'metadata': {'source': source, 'qa_pair_ids': ['private-id'], 'keypoints': ['point'],
                     'question_types': ['single-session-user'], 'question_ids': ['question-id']}}]})
    suite.root = pinned_root
    def output(body):
        if len(body['messages']) == 2:
            return 'candidate'
        prompt = body['messages'][0]['content']
        requests = suite.scorer().judge_requests(suite.cases[0], 'candidate')
        role = next(r.role for r in requests if r.prompt == prompt)
        return {'answer': 'yes', 'fluency': {'fluency': 1}, 'recall': {'recall': 1},
                'precision': {'precision': 1, 'sentence_count': 1}}[role]
    with model_server(output_for=output) as (url, _calls):
        args, _outer = setup_run(url, tmp_path, suite)
        args['profile'] = RackProfile(base_url=url, model='example/offline',
            profile_version='role-budgets-v6', generation_budgets=dict(
                extraction=4096, consolidation=4096, reader=4096, judge=4096))
        args.update(system='no_memory', judge_qualification={
            'protocol': 'native-judge-parameters@v2', 'model': 'example/offline',
            'temperatures': [0, 0.1], 'output_budget': 4096})
        result = api().run_history(**args)
    assert api().verify_history(args['out']) == result
    path = args['out'] / 'calls.jsonl'
    calls = [json.loads(line) for line in path.read_bytes().splitlines()]
    judge = next(r for r in calls if r['url'].endswith('/chat/completions')
                 and len(r['request']['messages']) == 1)
    if tamper == 'judge_token_count':
        for r in calls:
            if r['url'].endswith('/tokenize') and len(r['request']['messages']) == 1:
                r['response']['count'] = 131000
    else:
        judge['request'][tamper] = 0.1 if tamper == 'temperature' and source == 'longmemeval_s*' else 0
    path.write_bytes(b''.join(json.dumps(r).encode() + b'\n' for r in calls))
    result['artifact_sha256']['calls.jsonl'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args['out'] / 'root-status.json').write_bytes(json.dumps(result).encode())
    with pytest.raises(ValueError, match='wire|parameter|token|context'):
        api().verify_history(args['out'])


def test_cleanup_failure_is_counted_once_inside_finalization_boundary(tmp_path, pinned_root, monkeypatch):
    from gluomem_bench.bench.comparison_backends import RawEvidenceBackend
    original = RawEvidenceBackend.close
    closed = []
    async def failing_close(backend):
        closed.append(backend)
        await original(backend)
        raise OSError('injected store close failure')
    monkeypatch.setattr(RawEvidenceBackend, 'close', failing_close)
    with model_server() as (url, _calls):
        args, outer = setup_run(url, tmp_path, authored_suite(pinned_root))
        args['system'] = 'bm25'
        with pytest.raises(OSError, match='injected') as caught:
            api().run_history(**args)
    assert caught.value.generation_calls == outer.calls == 1
    assert len(closed) == 1
    assert not (args['out'] / 'root-status.json').exists()
    assert json.loads((args['out'] / 'failure.json').read_bytes())['generation_calls'] == 1


def test_unwritable_failure_receipt_does_not_hide_actual_spend(tmp_path, pinned_root, monkeypatch):
    module = api()
    original = module._write
    def reject(path, value):
        if path.name in {'root-status.json', 'failure.json'}:
            raise OSError('injected receipt filesystem failure')
        return original(path, value)
    monkeypatch.setattr(module, '_write', reject)
    with model_server() as (url, _calls):
        args, outer = setup_run(url, tmp_path, authored_suite(pinned_root))
        args['system'] = 'no_memory'
        with pytest.raises(OSError, match='injected') as caught:
            module.run_history(**args)
    assert caught.value.generation_calls == outer.calls == 1
    assert not (args['out'] / 'root-status.json').exists()
    assert any('failure receipt' in note for note in caught.value.__notes__)


def test_remote_embedding_timeout_is_bounded_by_same_whole_system_deadline(tmp_path, monkeypatch):
    from gluomem_bench.bench import embedding_runtime
    from gluomem_bench.bench.rack_trace import WorkloadBudget
    from gluomem_bench.bench.rack import RackProfile
    original = embedding_runtime.urlopen
    observed = []
    def inspect_timeout(request, **kwargs):
        observed.append(kwargs['timeout'])
        return original(request, **kwargs)
    monkeypatch.setattr(embedding_runtime, 'urlopen', inspect_timeout)
    with model_server() as (url, calls):
        budget = WorkloadBudget()
        monkeypatch.setattr(budget, 'remaining', lambda: 0.25)
        remote = embedding_runtime.RemoteEmbedder(url, timeout=120)
        wrapped = api().DeadlineEmbedder(remote, budget=budget, request_timeout=60)
        values = wrapped.embed_batch(['authored source'])
        assert len(values) == 1
        assert wrapped.fingerprint == remote.fingerprint
        assert wrapped.recipe == remote.recipe
        assert budget.calls == 0  # embedding is not a generation; wall time still bounded
    assert observed == [0.25]
    payload, = [body for path, body in calls if path.endswith('/embeddings')]
    assert payload['input'] == ['authored source'] and payload['input_type'] == 'document'


def test_history_and_question_occurrences_are_not_deduplicated_by_context_or_source_id(tmp_path):
    from gluomem_bench.bench.mab_suite import MABSuite
    row = {'context': 'Only source document bytes.', 'questions': ['private question', 'private question'],
           'answers': [['private gold'], ['private gold']],
           'metadata': {'source': 'ruler_qa1_197K', 'qa_pair_ids': ['repeat', 'repeat']}}
    suite = MABSuite.from_rows({'Accurate_Retrieval': [row, row]})
    assert len(suite.histories) == 2 and len(suite.cases) == 4
    assert len({c.sample.id for c in suite.cases}) == 4
    assert {c.source_id for c in suite.cases} == {'repeat'}
    left = api().document_events(suite.histories[0], chunk_chars=7)
    right = api().document_events(suite.histories[1], chunk_chars=7)
    assert {e.event_id for e in left}.isdisjoint(e.event_id for e in right)
    assert ''.join(e.content for e in left) == row['context']
    for event in left + right:
        assert event.source_type == 'document' and event.kind is None
        assert not event.messages and event.created_at is None and event.event_time is None
        serialized = event.model_dump_json()
        assert 'private question' not in serialized and 'private gold' not in serialized
    with model_server() as (url, calls):
        args, _outer = setup_run(url, tmp_path, suite)
        args['case_ids'] = [suite.cases[0].sample.id, suite.cases[-1].sample.id]
        with pytest.raises(ValueError, match='one history'):
            api().run_history(**args)
        assert not calls and not args['out'].exists()
