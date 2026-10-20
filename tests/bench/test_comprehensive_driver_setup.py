"""Approved authored setup fixtures: real driver/gates, denied endpoints, no QA."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from test_comprehensive_driver_chain import _authored_previous


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def approved_setup(tmp_path, monkeypatch):
    from gluomem_bench.bench import comprehensive_driver as d
    assert Path(d.__file__).resolve().parents[3].name == 'runtime', 'setup fixtures require the test-only private overlay'
    from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe
    from gluomem_bench.bench.rack import RackProfile, GenerationBudgets
    from gluomem_bench.bench import upstream_scoring
    # Only retained predecessor/native scorer are authored seams. The new builder,
    # complete inventory, approval, launch validator and entire setup are real.
    source = tmp_path/'authored.json'
    source.write_bytes(json.dumps([dict(question_id='q', question='Editor?', answer='Neovim',
        question_type='single-session-user', question_date='2026/01/02',
        haystack_dates=['2026/01/01 (Thu) 10:00'], haystack_session_ids=['s'],
        haystack_sessions=[[dict(role='user', content='I use Neovim.')]], answer_session_ids=['s'])]).encode())
    judge = tmp_path/'authored-judge.py'; judge.write_bytes(b'# authored scorer; no QA\n')
    previous = list(_authored_previous(tmp_path))
    old = previous[0]
    profile = RackProfile(base_url='https://fixture.invalid/v1', model='authored/model',
        profile_version='role-budgets-v6', max_generation_calls=32, max_run_seconds=120,
        generation_budgets=GenerationBudgets(extraction=32, consolidation=32, reader=32, judge=32))
    recipe = EmbeddingRecipe()
    old.update(dataset=dict(name='longmemeval_s', path=str(source), sha256=sha(source)),
        judge_source=str(judge), profile=profile.model_dump(mode='json'),
        embedding_url='https://fixture.invalid/v1', embedding_recipe=recipe.as_dict(),
        evidence_token_budget=4096, raw_chunk_chars=1000,
        config_overrides={'benchmark.datasets.longmemeval_s': dict(url='https://fixture.invalid/authored.json',file=source.name,sha256=sha(source)),
                          'benchmark.datasets_dir':str(tmp_path)},
        reuse_receipt={}, runtime_root=str(tmp_path/'old-runtime'),
        execution_environment=dict(MEM0_TELEMETRY='False', GLUOMEM_RACK_API_KEY='local-placeholder',
                                   unset=['OPENAI_API_KEY','OPENROUTER_API_KEY']),
        candidate_source_archive=dict(path='old.zip'),
        campaign_policy=dict(max_calls=57344000,max_seconds=75603600,max_workers=1,max_in_flight=1,max_in_flight_tokens=262144,stop_on_failure=False))
    for row in old['jobs']: row.update(call_budget=32, seconds_budget=120)
    old['files'] = d.frozen_inventory(source, judge)
    monkeypatch.setattr(d, '_load_chained_predecessor', lambda *a, **kw: copy.deepcopy(previous))
    monkeypatch.setattr(d, '_chain_semantic_bridge', lambda *a: [])
    monkeypatch.setattr(upstream_scoring, 'load_longmemeval_prompt', lambda *a: 'Authored judge {question}')
    # Real byte-bound cache assets/manifest/archive, not a seed-validation mock.
    import zipfile
    namespace = Path(d.__file__).resolve().parents[3].parent
    assets = namespace/'native-cache-assets'; assets.mkdir(exist_ok=True)
    asset = assets/'fixture.json'; asset.write_bytes(b'{"authored":true}')
    archive = tmp_path/'assets.zip'
    with zipfile.ZipFile(archive, 'w') as zipped: zipped.writestr('fixture.json', asset.read_bytes())
    manifest = namespace/'native-cache-assets.json'
    manifest.write_bytes(json.dumps(dict(source_archive=dict(path=str(archive),sha256=sha(archive)),files=[
        dict(relative_path='fixture.json',archive_member='fixture.json',sha256=sha(asset))])).encode())
    spec = d.build_chained_continuation_execution_spec(predecessor_binding=dict(root=str(tmp_path/'old')),
        output_root=tmp_path/'out', semantic_delta_manifest=dict(path='authored',sha256='a'*64),
        repaired_cohort='timeout600_serial1_repaired_guard_v1',
        launch_metadata=dict(namespace_root=str(namespace),read_only_assets=dict(path=str(manifest),sha256=sha(manifest))))
    from test_comprehensive_driver_resource import authored_allowance
    spec['prior_resource_allowance'] = authored_allowance(spec,tmp_path,
        calls=spec['prior_accounting']['calls'],seconds=spec['prior_accounting']['elapsed_seconds'])
    spec_path = tmp_path/'spec.json'; spec_path.write_bytes(json.dumps(spec).encode())
    approval_path = tmp_path/'authored-approval.json'
    binding = spec['prior_resource_allowance']
    approval_path.write_bytes(json.dumps(dict(approved=True, reviewer_kind='AI-assisted',spec_sha256=sha(spec_path),semantic_delta_manifest_sha256='a'*64,
        prior_resource_allowance_sha256=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        parent_reserve_approval_sha256=binding['parent_reserve_approval']['sha256'])).encode())
    for key, value in spec['execution_environment'].items():
        if key == 'unset':
            for name in value: monkeypatch.delenv(name, raising=False)
        else: monkeypatch.setenv(key, value)
    monkeypatch.chdir(spec['working_directory'])
    # Actual generation and encoder endpoints must never be reached by setup.
    import socket
    def denied(*a, **kw): raise AssertionError('fixture endpoint access forbidden')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    assert d.verify_execution_spec(spec_path, approval_path) == spec
    return d, spec, spec_path, approval_path


def test_actual_approved_setup_keeps_exact_frozen_environment(approved_setup, monkeypatch):
    d, spec, spec_path, approval_path = approved_setup
    from gluomem_bench.bench import parallel_campaign as pc
    reached = []
    class SetupObserved(BaseException): pass
    def stop(jobs, **kwargs):
        # This is after actual mkdir, seeding, spec/approval writes and preservation.
        reached.append(kwargs['budget'].snapshot())
        assert {key:os.environ.get(key) for key in spec['execution_environment'] if key != 'unset'} == {
            key:value for key,value in spec['launch_metadata']['execution_environment'].items() if key != 'unset'}
        raise SetupObserved()
    monkeypatch.setattr(pc, 'run_jobs', stop)
    with pytest.raises(pc.CampaignReceiptError) as failure:
        d.main(['--spec',str(spec_path),'--approval',str(approval_path)])
    assert isinstance(failure.value.__cause__, SetupObserved)
    assert reached[0]['prior_calls'] == 8225
    assert (Path(spec['output_root'])/'independent-approval.json').read_bytes() == approval_path.read_bytes()


def test_cache_seed_failure_before_saved_spec_retains_original_and_prior_debit(approved_setup, monkeypatch):
    d, spec, spec_path, approval_path = approved_setup
    from gluomem_bench.bench.parallel_campaign import CampaignReceiptError
    original = OSError('authored seed failure before spec preservation')
    def fail(spec): raise original
    monkeypatch.setattr(d, '_seed_chain_cache_assets', fail)
    with pytest.raises(CampaignReceiptError) as caught:
        d.main(['--spec',str(spec_path),'--approval',str(approval_path)])
    assert caught.value.__cause__ is original
    receipt = d.verify_setup_incomplete(spec['output_root'])
    assert receipt['complete'] is False and receipt['execution_complete'] is False
    assert receipt['accounting']['prior_calls'] == 8225
    assert receipt['accounting']['prior_elapsed_seconds'] == 14937.567625900003
    assert receipt['accounting']['calls'] == 8225
    assert receipt['accounting']['new_calls'] == 0
    assert receipt['spec_sha256'] == sha(spec_path)
    assert receipt['approval_sha256'] == sha(approval_path)
    assert receipt['output_root'] == spec['output_root']
    assert receipt['prior_spend_uncertainty'] == spec['prior_spend_uncertainty']


@pytest.mark.parametrize('fault', ['mkdir', 'spec-write', 'approval-write', 'cache-copy', 'preservation', 'locked-root', 'root-status-write'])
def test_each_setup_boundary_has_readonly_terminal_failure(approved_setup, monkeypatch, fault):
    d, spec, spec_path, approval_path = approved_setup
    from gluomem_bench.bench.parallel_campaign import CampaignReceiptError
    root = Path(spec['output_root'])
    original = PermissionError('authored '+fault)
    mkdir, write, read = Path.mkdir, Path.write_bytes, Path.read_bytes
    def fail_mkdir(path, *args, **kwargs):
        if path == root: raise original
        return mkdir(path, *args, **kwargs)
    def fail_write(path, material):
        name = {'spec-write':'execution-spec.json','approval-write':'independent-approval.json','root-status-write':'root-status.json'}[fault]
        if path == root/name: raise original
        return write(path, material)
    if fault == 'mkdir': monkeypatch.setattr(Path, 'mkdir', fail_mkdir)
    elif fault in ('spec-write','approval-write','root-status-write'): monkeypatch.setattr(Path, 'write_bytes', fail_write)
    elif fault == 'cache-copy':
        # Real seed path: authenticated asset, then first exclusive cache open.
        opened = Path.open
        def fail_open(path, *args, **kwargs):
            if path == Path(spec['execution_environment']['FASTEMBED_CACHE_PATH'])/'fixture.json': raise original
            return opened(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'open', fail_open)
    else:
        import shutil
        locked = [False]
        def fail_copy(*a, **kw):
            locked[0] = True
            raise original
        monkeypatch.setattr(shutil, 'copyfile', fail_copy)
        if fault == 'locked-root':
            def locked_read(path, *args, **kwargs):
                if path.is_relative_to(root): raise PermissionError('authored root locked')
                return read(path, *args, **kwargs)
            def locked_write(path, *args, **kwargs):
                if locked[0] and path.is_relative_to(root): raise PermissionError('authored root write locked')
                return write(path, *args, **kwargs)
            unlink = Path.unlink
            def locked_unlink(path, *args, **kwargs):
                if locked[0] and path.is_relative_to(root): raise PermissionError('authored root unlink locked')
                return unlink(path, *args, **kwargs)
            monkeypatch.setattr(Path, 'read_bytes', locked_read)
            monkeypatch.setattr(Path, 'write_bytes', locked_write)
            monkeypatch.setattr(Path, 'unlink', locked_unlink)
    with pytest.raises(CampaignReceiptError) as caught:
        d.main(['--spec',str(spec_path),'--approval',str(approval_path)])
    assert caught.value.__cause__ is original
    receipt = d.verify_campaign(root)
    assert receipt == d.verify_setup_incomplete(root)
    assert receipt['publication'] == 'incomplete' and receipt['complete'] is False
    assert receipt['accounting']['calls'] == 8225 and receipt['accounting']['new_calls'] == 0
    assert receipt['accounting']['prior_elapsed_seconds'] == 14937.567625900003
    assert receipt['spec_sha256'] == sha(spec_path)
    # Verifiers must be read-only, including when preserved provenance is absent.
    with monkeypatch.context() as readonly:
        readonly.setattr(Path, 'write_bytes', lambda *a: pytest.fail('verification wrote'))
        assert d.verify_campaign(root) == receipt
    # Failure evidence must prevent a later same-root admission, even mkdir failure.
    with pytest.raises((ValueError, FileExistsError)):
        d.verify_execution_spec(spec_path, approval_path)


def test_setup_provenance_cannot_change_after_real_admission(approved_setup, monkeypatch):
    d, spec, spec_path, approval_path = approved_setup
    verify = d.verify_execution_spec
    def mutate_approval(*args):
        admitted = verify(*args)
        changed = json.loads(approval_path.read_bytes()); changed['approved'] = False
        approval_path.write_bytes(json.dumps(changed).encode())
        return admitted
    monkeypatch.setattr(d, 'verify_execution_spec', mutate_approval)
    def stop(*args): raise OSError('must not reach setup')
    monkeypatch.setattr(d, '_seed_chain_cache_assets', stop)
    with pytest.raises(ValueError, match='changed during admission'):
        d.main(['--spec',str(spec_path),'--approval',str(approval_path)])
    assert not Path(spec['output_root']).exists()


def test_terminal_failure_never_promotes_or_accepts_rebound_provenance(approved_setup, monkeypatch):
    d, spec, spec_path, approval_path = approved_setup
    from gluomem_bench.bench.parallel_campaign import CampaignReceiptError
    def fail(*a): raise OSError('authored seed fault')
    monkeypatch.setattr(d, '_seed_chain_cache_assets', fail)
    with pytest.raises(CampaignReceiptError):
        d.main(['--spec',str(spec_path),'--approval',str(approval_path)])
    root = Path(spec['output_root']); marker = d._setup_failure_path(root)
    read = Path.read_bytes
    saved = json.loads(read(marker))
    for field in ['complete','execution_complete','terminal','output_root','spec_sha256','approval_sha256','prior_accounting','accounting','resource_accounting']:
        forged = copy.deepcopy(saved)
        if field in ['complete','execution_complete','terminal']: forged[field] = not saved[field]
        elif field == 'output_root': forged[field] += '-forged'
        elif field.endswith('sha256'): forged[field] = '0'*64
        elif field == 'prior_accounting': forged[field]['calls'] -= 1
        elif field == 'accounting': forged[field]['prior_calls'] -= 1
        elif field == 'resource_accounting': forged[field]['prior_debit_kind'] = 'measured'
        with monkeypatch.context() as overlay:
            overlay.setattr(Path,'read_bytes',lambda p: json.dumps(forged).encode() if p == marker else read(p))
            with pytest.raises(ValueError): d.verify_campaign(root)
    # Copying the whole otherwise valid receipt to another namespace cannot bind it.
    elsewhere = root.with_name('different-root')
    d._setup_failure_path(elsewhere).write_bytes(read(marker))
    with pytest.raises(ValueError): d.verify_setup_incomplete(elsewhere)
