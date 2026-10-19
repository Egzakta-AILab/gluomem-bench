"""Versioned exact symbol guards: no whole-module or algorithm exemption."""
import copy,hashlib,json
from pathlib import Path
import pytest


def make_entry(d,old,new,name,exceptions,telemetry=(),imports=(),statements=()):
    return dict(contract='longmemeval-privacy-guard-symbols@v2',relative_path=name,
        old_path=str(old),old_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
        new_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),exception_symbols=list(exceptions),
        telemetry_symbols=list(telemetry),telemetry_imports=list(imports),telemetry_statements=list(statements),
        symbol_bindings=d.guard_delta_symbol_bindings(old.read_bytes(),new.read_bytes(),set(exceptions)|set(telemetry)|set(statements)))


@pytest.mark.parametrize('name,exceptions,telemetry,imports',[
    ('embedding_runtime.py',['RemoteEmbedder._encode','make_server.Handler.do_POST'],['safe_failure_diagnostic'],
     ['from urllib.parse import parse_qsl, urlsplit, urlunsplit']),
    ('suite_campaign.py',['_run_history','run_history'],[],
     ['from gluomem_bench.bench.embedding_runtime import safe_failure_diagnostic']),
    ('parallel_campaign.py',['run_jobs'],[],
     ['from gluomem_bench.bench.embedding_runtime import safe_failure_diagnostic']),
])
def test_actual_privacy_modules_have_exact_api_and_normal_path_parity(name,exceptions,telemetry,imports):
    from gluomem_bench.bench import comprehensive_driver as d
    root=Path('tests/fixtures/native/operator-supplied')
    old=root/'longmemeval-continuation-runtime-v5/runtime/gluomem-bench/gluomem_bench/bench'/name
    new=Path(d.__file__).parent/name
    entry=make_entry(d,old,new,name,exceptions,telemetry,imports)
    assert d._verify_guard_delta(new,entry)['symbol_bindings']==entry['symbol_bindings']
    bad=copy.deepcopy(entry);symbol=next(iter(bad['symbol_bindings']))
    bad['symbol_bindings'][symbol]['new_ast_sha256']='0'*64
    with pytest.raises(ValueError,match='symbol'):d._verify_guard_delta(new,bad)


def test_v2_mem0_telemetry_is_exact_named_definition_not_native_algorithm_exemption(tmp_path):
    from gluomem_bench.bench import comprehensive_driver as d
    old=tmp_path/'old.py';new=tmp_path/'mem0.py'
    old.write_bytes(b'class NativeMem0Diagnostics:\n def emit(self,record):\n  self.rows.append(record.getMessage())\ndef vector(x):\n return x\n')
    new.write_bytes(b'class NativeMem0Diagnostics:\n def emit(self,record):\n  self.rows.append("native-warning")\ndef vector(x):\n return x\n')
    entry=make_entry(d,old,new,'mem0.py',[],['NativeMem0Diagnostics.emit'])
    d._verify_guard_delta(new,entry)
    new.write_bytes(new.read_bytes().replace(b'return x',b'return 0'))
    entry=make_entry(d,old,new,'mem0.py',[],['NativeMem0Diagnostics.emit'])
    with pytest.raises(ValueError,match='normal-path'):d._verify_guard_delta(new,entry)
    new.write_bytes(b'class NativeMem0Diagnostics:\n def emit(self,record,permit=True):\n  self.rows.append("native-warning")\ndef vector(x):\n return x\n')
    entry=make_entry(d,old,new,'mem0.py',[],['NativeMem0Diagnostics.emit'])
    with pytest.raises(ValueError,match='signature'):d._verify_guard_delta(new,entry)


def test_actual_mem0_capture_plumbing_keeps_native_algorithms_and_check_exact(tmp_path):
    from gluomem_bench.bench import comprehensive_driver as d
    root=Path('tests/fixtures/native/operator-supplied')
    old=root/'longmemeval-continuation-runtime-v5/runtime/gluomem-bench/gluomem_bench/bench/mem0.py'
    live=Path('tests/fixtures/native/operator-supplied')
    new=tmp_path/'mem0.py';new.write_bytes(live.read_bytes())
    telemetry=['_capture_native_mem0_log','NativeMem0Diagnostics.__init__','NativeMem0Diagnostics.emit','NativeMem0Diagnostics.close']
    statements=['NativeMem0Diagnostics.__doc__','NativeMem0Diagnostics._capture_lock','NativeMem0Diagnostics._captures','NativeMem0Diagnostics._original_dispatch']
    entry=make_entry(d,old,new,'mem0.py',['GuardedMem0SDK.create'],telemetry,[])
    entry['telemetry_statements']=statements
    entry['symbol_bindings']=d.guard_delta_symbol_bindings(old.read_bytes(),new.read_bytes(),set(telemetry+statements+['GuardedMem0SDK.create']))
    d._verify_guard_delta(new,entry)
    # Existing warning/error admission is NOT a telemetry exemption.
    new.write_bytes(new.read_bytes().replace(b'if self.events:',b'if False:'))
    entry['new_sha256']=hashlib.sha256(new.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='normal-path'):d._verify_guard_delta(new,entry)


def test_v2_rejects_duplicate_exempt_definition_and_state(tmp_path):
    from gluomem_bench.bench import comprehensive_driver as d
    old=b''
    for new,symbol in [
        (b'def safe_failure_diagnostic(exc):\n return dangerous()\ndef safe_failure_diagnostic(exc):\n return {}\n','safe_failure_diagnostic'),
        (b'class NativeMem0Diagnostics:\n _captures = dangerous()\n _captures = set()\n','NativeMem0Diagnostics._captures')]:
        with pytest.raises(ValueError,match='duplicate'):
            d.guard_delta_symbol_bindings(old,new,{symbol})


def test_v2_cannot_declare_scoring_or_whole_module_telemetry(tmp_path):
    from gluomem_bench.bench import comprehensive_driver as d
    old=tmp_path/'old.py';new=tmp_path/'new.py'
    old.write_bytes(b'def score(x):\n return x\n');new.write_bytes(b'def score(x):\n return 0\n')
    entry=make_entry(d,old,new,'mem0.py',[],['score'])
    with pytest.raises(ValueError,match='symbol'):d._verify_guard_delta(new,entry)


@pytest.mark.parametrize('legacy_kind',['parallel-file','v2-contract'])
def test_chain_v2_binds_coordinator_and_helper_without_widening_legacy_v1(tmp_path,monkeypatch,legacy_kind):
    from gluomem_bench.bench import comprehensive_driver as d
    root=Path('tests/fixtures/native/operator-supplied')
    base=root/'longmemeval-continuation-runtime-v5/runtime/gluomem-bench/gluomem_bench/bench'
    declarations=[
        ('suite_campaign.py',['_run_history','run_history'],[],['from gluomem_bench.bench.embedding_runtime import safe_failure_diagnostic']),
        ('embedding_runtime.py',['RemoteEmbedder._encode','make_server.Handler.do_POST'],['safe_failure_diagnostic'],['from urllib.parse import parse_qsl, urlsplit, urlunsplit']),
        ('parallel_campaign.py',['run_jobs'],[],['from gluomem_bench.bench.embedding_runtime import safe_failure_diagnostic']),
        ('mem0.py',['GuardedMem0SDK.create'],['_capture_native_mem0_log','NativeMem0Diagnostics.__init__','NativeMem0Diagnostics.emit','NativeMem0Diagnostics.close'],[])]
    entries=[make_entry(d,base/name,Path(d.__file__).parent/name,name,ex,tele,imp,
        ['NativeMem0Diagnostics.'+v for v in ('__doc__','_capture_lock','_captures','_original_dispatch')] if name=='mem0.py' else [])
        for name,ex,tele,imp in declarations]
    previous=dict(files=[dict(path=str(base/name),sha256=hashlib.sha256((base/name).read_bytes()).hexdigest())
        for name in ('suite_campaign.py','mem0.py','embedding_runtime.py','rack.py','parallel_campaign.py')],
        original_binding=dict(verifier_path='authored-readonly-verifier'))
    monkeypatch.setattr(d,'_continuation_original',lambda *a:(copy.deepcopy(previous),{}))
    def semantic(original,verifier,*,guard_deltas):
        return [d._verify_guard_delta(Path(d.__file__).parent/name,guard_deltas[name])
            for name in ('suite_campaign.py','mem0.py')]
    monkeypatch.setattr(d,'_semantic_bridge',semantic)
    manifest=tmp_path/'authored-manifest.json'
    material=dict(protocol='longmemeval-reviewed-guard-delta@v2',files=entries)
    manifest.write_bytes(json.dumps(material).encode())
    binding=dict(path=str(manifest),sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    bridge=d._chain_semantic_bridge(previous,binding)
    assert any(row.get('relative_path')=='parallel_campaign.py' for row in bridge)
    material['protocol']='longmemeval-reviewed-guard-delta@v1'
    if legacy_kind=='v2-contract':material['files']=entries[:2]
    manifest.write_bytes(json.dumps(material).encode());binding['sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError):d._chain_semantic_bridge(previous,binding)
