"""Real process privacy regressions; no inference or remote sockets."""
import asyncio
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2] / 'tests/fixtures/native/operator-supplied'
NATIVE = ROOT / 'cognee/venv/Scripts/python.exe'


def test_graph_worker_failure_never_formats_private_cause(tmp_path):
    from gluomem_bench.bench import cognee_graph
    worker = Path(cognee_graph.__file__)
    # An untrusted operation reaches the actual isolated worker exception branch.
    payload = {'operation': 'PRIVATE_EXCEPTION_BODY_SQL_PARAMETER'}
    result = subprocess.run([str(NATIVE), '-I', str(worker), '--graph-worker'],
                            input=json.dumps(payload)+'\n', text=True,
                            capture_output=True, cwd=tmp_path, timeout=40)
    assert 'Traceback' not in result.stderr, 'raw traceback persisted by real graph worker'
    assert 'PRIVATE_EXCEPTION' not in result.stderr
    reply = json.loads(result.stdout)
    assert reply == {'ok': False, 'error_type': 'native_failure'}


def test_process_boundary_captures_root_child_warnings_structlog_before_format(tmp_path):
    from gluomem_bench.bench import cognee_native_boundary as boundary
    script = '''
import importlib.util,json,logging,sys,warnings
spec=importlib.util.spec_from_file_location('boundary',sys.argv[1]); b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
sink=b.NativeBoundary(sys.argv[2]);sink.install()
class Hostile:
 def __str__(self): raise AssertionError('PRIVATE_FORMATTED')
 def __repr__(self): raise AssertionError('PRIVATE_REPR')
root=logging.getLogger();root.addHandler(logging.FileHandler(sys.argv[3]))
child=logging.getLogger('native.sql');child.propagate=False;child.addHandler(logging.FileHandler(sys.argv[3]))
root.error(Hostile(),exc_info=(ValueError,ValueError('PRIVATE_CAUSE'),None))
child.warning('PRIVATE_SQL %s',Hostile())
warnings.warn('PRIVATE_WARNING_PARAMETERS',UserWarning)
from sqlalchemy.exc import SAWarning
warnings.filterwarnings('ignore', category=SAWarning)
warnings.warn('PRIVATE_SQL_PARAMETERS', SAWarning)
import structlog
structlog.get_logger().error('PRIVATE_STRUCTLOG',error=Hostile())
sink.save(); print(json.dumps(sink.snapshot()))
'''
    result = subprocess.run([str(NATIVE), '-I', '-c', script, boundary.__file__,
                             str(tmp_path/'diagnostics.json'), str(tmp_path/'raw.log')],
                            text=True, capture_output=True, timeout=40)
    assert result.returncode == 0, result.stderr
    material = b''.join(p.read_bytes() for p in tmp_path.iterdir() if p.is_file())
    assert b'PRIVATE_' not in material
    snapshot = json.loads(result.stdout)
    assert snapshot['blocking_diagnostics'] >= 5
    assert (tmp_path/'raw.log').read_bytes() == b''


def test_only_code_pinned_startup_notice_can_pass(tmp_path):
    from gluomem_bench.bench.cognee_native_boundary import NativeBoundary
    sink = NativeBoundary(tmp_path/'diagnostics.json')
    origin = {'source_sha256': 'a47784a889e2fc8a041ba02523ef29c65f20aeaa726dee70f079d99adb5c3c99',
              'line': 637, 'message_sha256': '2392b847aa39876b322358272871f3b81d32f7d83f19204d729e0852e0081a1b'}
    sink.capture('structured_warning', origin=origin)
    sink.check()
    assert sink.snapshot()['benign_startup_diagnostics'] == 1
    sink.startup = False
    sink.capture('structured_warning', origin=origin)
    import pytest
    with pytest.raises(RuntimeError, match='admission rejected'):
        sink.check()


def test_bridge_overlaps_native_generation_and_verifies_receipts(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import httpx
    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge, verify_generation_evidence
    from gluomem_bench.bench.rack import RackProfile
    from gluomem_bench.bench.rack_trace import WorkloadBudget
    barrier = threading.Barrier(2, timeout=2)
    def upstream(request):
        if request.url.path.endswith('/models'):
            return httpx.Response(200,json={'data':[{'id':'served/qwen'}]})
        if request.url.path.endswith('/tokenize'):
            return httpx.Response(200,json={'count':10})
        barrier.wait()
        return httpx.Response(200,json={'model':'served/qwen','choices':[{'finish_reason':'stop','message':{'content':'ok'}}],
                                       'usage':{'prompt_tokens':10,'completion_tokens':1}})
    profile = RackProfile(base_url='http://authored.invalid/v1',model='served/qwen')
    with CogneeGenerationBridge(profile,budget=WorkloadBudget(max_calls=2),receipt=tmp_path/'generation.jsonl',
                                transport=httpx.MockTransport(upstream)) as bridge:
        body = json.dumps({'model':'served/qwen','messages':[{'role':'user','content':'authored'}],
                           'temperature':0,'max_tokens':32}).encode()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(bridge.complete,[body,body]))
        assert len(results) == 2
        assert verify_generation_evidence(bridge.receipt,profile=profile,expected_calls=2)['calls'] == 2


def test_actual_native_cognify_extraction_summary_overlap(tmp_path, monkeypatch):
    import threading
    import time
    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge
    from tests.bench.test_cognee_graph import test_real_native_graph_lifecycle_network_only_fixture
    windows = []
    original = CogneeGenerationBridge._request
    def request(self, method, path, payload=None):
        if path != '/chat/completions':
            return original(self,method,path,payload)
        started = time.monotonic()
        # Delayed authored transport exposes native SDK scheduling, not model throughput.
        time.sleep(0.2)
        try:
            return original(self,method,path,payload)
        finally:
            windows.append({'start':started,'end':time.monotonic(),'thread':threading.get_ident()})
    monkeypatch.setattr(CogneeGenerationBridge,'_request',request)
    test_real_native_graph_lifecycle_network_only_fixture(tmp_path,NATIVE)
    events = sorted([(r['start'],1) for r in windows]+[(r['end'],-1) for r in windows])
    current = peak = 0
    for _,delta in events:
        current += delta
        peak = max(peak,current)
    (tmp_path/'native-sdk-concurrency.json').write_text(json.dumps({'scope':'actual SDK authored transport, not remote executing sequences',
          'windows':windows,'observed_overlap':peak}))
    assert peak >= 2
