"""Launch-only rebind contract; authored predecessor, actual imports/inventory."""
import copy
import hashlib
from pathlib import Path

import pytest

from test_comprehensive_driver_chain import _authored_previous


@pytest.fixture
def launch_candidate(tmp_path, monkeypatch):
    from gluomem_bench.bench import comprehensive_driver as d

    previous = _authored_previous(tmp_path)
    old = previous[0]
    old.update(runtime_root=str(tmp_path / 'old-runtime'), working_directory='old-work',
               pythonpath=['old-import'], execution_environment={'MODEL_SETTING': 'pinned'},
               candidate_source_archive={'path': 'old.zip', 'binding': 'archive SHA in SEAL.json; avoid circular self-hash'})
    # Native package/source enumeration remains real, not replaced by a test list.
    # Bind independent, authored dataset bytes as well as the private producer.
    dataset = tmp_path / 'dataset.json'
    dataset.write_bytes(b'[]')
    old['dataset'] = dict(path=str(dataset), sha256=hashlib.sha256(dataset.read_bytes()).hexdigest())
    old['judge_source'] = str(Path(d.__file__))
    old['files'] = d.frozen_inventory(old['dataset']['path'], old['judge_source'])
    monkeypatch.setattr(d, '_load_chained_predecessor', lambda *a, **kw: copy.deepcopy(previous))
    monkeypatch.setattr(d, '_chain_semantic_bridge', lambda *a: [])
    runtime = Path(d.__file__).resolve().parents[3]
    launch = dict(namespace_root=str(runtime.parent))
    spec = d.build_chained_continuation_execution_spec(
        predecessor_binding={'root': str(tmp_path / 'old')}, output_root=tmp_path / 'fresh-output',
        semantic_delta_manifest={'path': 'authored', 'sha256': 'a' * 64},
        repaired_cohort='timeout600_serial1_repaired', launch_metadata=launch,
    )
    return d, spec, previous


def test_launch_rebind_uses_actual_private_runtime_not_predecessor(tmp_path, launch_candidate):
    d, spec, previous = launch_candidate
    runtime = Path(d.__file__).resolve().parents[3]
    assert spec['runtime_root'] == str(runtime)
    assert spec['working_directory'] == str(runtime / 'gluomem-bench')
    assert spec['pythonpath'] == [str(runtime / 'gluomem-bench'), str(runtime / 'GluoMem')]
    assert spec['execution_environment']['MEM0_DIR'] == str(tmp_path / 'fresh-output/native-home')
    assert spec['execution_environment']['MODEL_SETTING'] == 'pinned'
    assert spec['candidate_source_archive']['path'] == str(runtime.parent / 'candidate-producers.zip')
    assert spec['prior_accounting']['calls'] == 8225
    assert spec['dispatch_job_ids'] == ['j1']
    assert d.validate_chained_continuation_spec(spec) == previous[1]
    assert not (tmp_path / 'fresh-output').exists()


@pytest.mark.parametrize('forgery', ['missing', 'stale', 'hash', 'duplicate'])
def test_launch_rejects_inventory_source_forgery(launch_candidate, forgery):
    d, spec, _ = launch_candidate
    bad = copy.deepcopy(spec)
    if forgery == 'missing': bad['files'] = []
    if forgery == 'stale': bad['files'][0]['path'] = str(Path(spec['runtime_root']).parent.parent / 'old.py')
    if forgery == 'hash': bad['files'][0]['sha256'] = '0' * 64
    if forgery == 'duplicate': bad['files'].append(bad['files'][0])
    with pytest.raises(ValueError, match='inventory'):
        d.validate_chained_continuation_spec(bad)


@pytest.mark.parametrize('target', ['old-root', 'old-runtime', 'ancestor', 'private-runtime'])
def test_launch_rejects_output_overlaps_in_both_directions(tmp_path, launch_candidate, target):
    d, spec, previous = launch_candidate
    paths = {'old-root': tmp_path / 'old/jobs/new', 'old-runtime': tmp_path / 'old-runtime/cache',
             'ancestor': tmp_path, 'private-runtime': Path(spec['runtime_root']) / 'new-output'}
    with pytest.raises(ValueError, match='overlap'):
        d._chain_launch_metadata(dict(previous[0], output_root=str(tmp_path / 'old')), paths[target])


def test_launch_routes_all_writable_caches_to_declared_output(launch_candidate):
    d, spec, _ = launch_candidate
    output = Path(spec['output_root'])
    for key in ['MEM0_DIR', 'DSPY_CACHEDIR', 'FASTEMBED_CACHE_PATH', 'HF_HOME',
                'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE', 'XDG_CACHE_HOME', 'TMP', 'TEMP', 'TMPDIR',
                'TORCH_HOME', 'NUMBA_CACHE_DIR']:
        assert Path(spec['execution_environment'][key]).is_relative_to(output)
    assert spec['launch_metadata']['writable_paths']


def test_launch_rejects_actual_import_from_uninventoried_old_tree(launch_candidate, monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    d, spec, _ = launch_candidate
    monkeypatch.setitem(sys.modules, 'gluomem.forged', SimpleNamespace(__file__=str(tmp_path / 'old-runtime/forged.py')))
    with pytest.raises(ValueError, match='import'):
        d.validate_chained_continuation_spec(spec)


def test_launch_rejects_changed_interpreter_or_dependency_pin(launch_candidate):
    import sys
    d, spec, previous = launch_candidate
    pin = next(p for p in previous[0]['files'] if Path(p['path']) == Path(sys.executable))
    pin['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='environment'):
        d.validate_chained_continuation_spec(spec)


def test_launch_process_refuses_ambient_environment_and_cwd(launch_candidate, monkeypatch, tmp_path):
    import os
    d, spec, _ = launch_candidate
    for key, value in spec['execution_environment'].items():
        if key != 'unset': monkeypatch.setenv(key, value)
    for key in spec['execution_environment'].get('unset', []): monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(spec['working_directory'])
    d._validate_chain_launch_process(spec)
    monkeypatch.setenv('MEM0_DIR', str(tmp_path / 'old-runtime'))
    with pytest.raises(ValueError, match='environment'):
        d._validate_chain_launch_process(spec)
    monkeypatch.setenv('MEM0_DIR', spec['execution_environment']['MEM0_DIR'])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='working'):
        d._validate_chain_launch_process(spec)


@pytest.mark.parametrize('field', ['runtime_root', 'working_directory', 'pythonpath', 'candidate_source_archive', 'cache', 'env', 'model', 'cap'])
def test_launch_rejects_stale_escape_and_semantic_forgery(launch_candidate, field, tmp_path):
    d, spec, _ = launch_candidate
    bad = copy.deepcopy(spec)
    if field in ['runtime_root', 'working_directory']: bad[field] = str(tmp_path / 'old-runtime')
    if field == 'pythonpath': bad[field] = [str(tmp_path / 'unrelated'), str(Path(spec['runtime_root']) / 'GluoMem')]
    if field == 'candidate_source_archive': bad[field]['path'] = str(tmp_path / 'old.zip')
    if field == 'cache': bad['execution_environment']['DSPY_CACHEDIR'] = str(tmp_path / '../unrelated-cache')
    if field == 'env': bad['execution_environment']['MODEL_SETTING'] = 'forged'
    if field == 'model': bad['profile']['reader_model'] = 'forged'
    if field == 'cap': bad['campaign_policy']['max_calls'] += 1
    if field in bad['launch_metadata']: bad['launch_metadata'][field] = copy.deepcopy(bad[field])
    if field in ['cache', 'env']: bad['launch_metadata']['execution_environment'] = bad['execution_environment']
    with pytest.raises(ValueError): d.validate_chained_continuation_spec(bad)


def test_launch_rejects_rehashed_actual_dataset_bytes(launch_candidate):
    d, spec, _ = launch_candidate
    source = Path(spec['dataset']['path'])
    source.write_bytes(b'["forged-source"]')
    # Rehashing a new local inventory must not admit different pinned input bytes.
    bad = copy.deepcopy(spec)
    bad['files'] = d.frozen_inventory(spec['dataset']['path'], spec['judge_source'])
    bad['launch_metadata'] = d._chain_launch_metadata(launch_candidate[2][0], spec['output_root'])
    with pytest.raises(ValueError, match='dataset'):
        d.validate_chained_continuation_spec(bad)


def test_launch_metadata_must_be_complete_and_frozen(launch_candidate):
    d, spec, _ = launch_candidate
    bad = copy.deepcopy(spec)
    bad['launch_metadata'] = {'namespace_root': spec['launch_metadata']['namespace_root']}
    with pytest.raises(ValueError, match='launch'):
        d.validate_chained_continuation_spec(bad)


def test_launch_actual_import_requires_exact_module_location(launch_candidate, monkeypatch):
    import sys
    from types import SimpleNamespace
    d, spec, _ = launch_candidate
    monkeypatch.setitem(sys.modules, 'gluomem.forged', SimpleNamespace(__file__=str(Path(spec['pythonpath'][1]) / 'gluomem/__init__.py')))
    with pytest.raises(ValueError, match='import'):
        d.validate_chained_continuation_spec(spec)


def test_readonly_cache_assets_are_byte_bound_and_seed_only_fresh_cache(tmp_path):
    import json
    import zipfile
    from gluomem_bench.bench import comprehensive_driver as d
    namespace = tmp_path / 'candidate'
    assets = namespace / 'native-cache-assets'
    assets.mkdir(parents=True)
    source = tmp_path / 'retained.zip'
    with zipfile.ZipFile(source, 'w') as z: z.writestr('assets/config.json', b'{}')
    (assets / 'config.json').write_bytes(b'{}')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = namespace / 'native-cache-assets.json'
    manifest.write_bytes(json.dumps(dict(source_archive=dict(path=str(source), sha256=digest(source)), files=[
        dict(relative_path='config.json', archive_member='assets/config.json', sha256=digest(assets / 'config.json'))])).encode())
    binding = dict(path=str(manifest), sha256=digest(manifest))
    verified = d._chain_read_only_assets(binding, namespace)
    assert verified[0]['path'] == str(assets / 'config.json')
    output = tmp_path / 'fresh'
    spec = dict(output_root=str(output), launch_metadata=dict(namespace_root=str(namespace), read_only_assets=binding),
                execution_environment=dict(FASTEMBED_CACHE_PATH=str(output / 'fastembed-cache')))
    d._seed_chain_cache_assets(spec)
    assert (output / 'fastembed-cache/config.json').read_bytes() == b'{}'
    (assets / 'config.json').write_bytes(b'{"forged":true}')
    with pytest.raises(ValueError): d._chain_read_only_assets(binding, namespace)
    (assets / 'config.json').write_bytes(b'{}')
    forged = json.loads(manifest.read_bytes())
    forged['files'][0]['relative_path'] = '../escape'
    manifest.write_bytes(json.dumps(forged).encode()); binding['sha256'] = digest(manifest)
    with pytest.raises(ValueError): d._chain_read_only_assets(binding, namespace)
