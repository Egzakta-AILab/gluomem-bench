"""Executor contracts. Authored fixtures only; no real model endpoints."""
import importlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pytest


def executor():
    return importlib.import_module('gluomem_bench.bench.lme_first_execute')


def test_remote_context_accounts_through_actual_coordinator(tmp_path):
    x = executor()
    from gluomem_bench.bench.lme_first import run_isolated_jobs, verify_journal
    rows = [dict(job_id=str(i), history_id=str(i), system='mem0', output=str(tmp_path / str(i)),
                 memory_mib=1, call_budget=4, seconds_budget=10) for i in range(3)]
    with x.ContextBroker() as broker:
        def run(job, context):
            handle = broker.register(context)
            def dispatch():
                remote = x.RemoteContext(broker.address, broker.key, handle, job)
                try:
                    with remote.lease(100):
                        remote.begin_call()
                        assert remote.calls >= 1
                        assert remote.remaining() > 0
                finally:
                    remote.close()
            with ThreadPoolExecutor(2) as pool:
                list(pool.map(lambda _: dispatch(), range(2)))
            return {'native': 'fixture'}
        report = run_isolated_jobs(rows, out=tmp_path/'coord', run_job=run, max_calls=12,
                 max_seconds=30, max_workers=3, native_limits={'mem0':3}, memory_limit_mib=3,
                 max_in_flight_tokens=39321600)
    assert report['accounting']['calls'] == 6
    assert verify_journal(tmp_path/'coord', rows) == report


def test_remote_duplicate_begin_latches_global_accounting(tmp_path):
    x = executor()
    from gluomem_bench.bench.lme_first import run_isolated_jobs
    rows = [dict(job_id='a', history_id='a', system='mem0', output=str(tmp_path/'a'), memory_mib=1,
                 call_budget=4, seconds_budget=10)]
    with x.ContextBroker() as broker:
        def run(job, context):
            remote = x.RemoteContext(broker.address, broker.key, broker.register(context), job)
            try:
                with remote.lease(100):
                    remote.begin_call()
                    remote.begin_call()
            finally:
                remote.close()
        report = run_isolated_jobs(rows, out=tmp_path/'coord', run_job=run, max_calls=4,
                 max_seconds=30, max_workers=1, native_limits={'mem0':1}, memory_limit_mib=1,
                 max_in_flight_tokens=39321600)
    assert report['jobs'][0]['status'] == 'failed'
    assert report['accounting']['calls'] == 1


def test_memory_admission_uses_measured_tree_and_host_reserve():
    x = executor()
    admission = x.MemoryAdmission(host_reserve_mib=2048, available=lambda: 8192)
    assert not admission.admit('gluomem')  # No invented per-worker size.
    admission.observe('gluomem', rss_mib=128, private_mib=256)
    assert admission.admit('gluomem')
    admission.available = lambda: 2100
    assert not admission.admit('gluomem')
    assert admission.snapshot()['observed']['gluomem']['private_mib'] == 256


def test_memory_policy_tracks_rss_commit_and_atomic_startup_reservations():
    x = executor()
    host = {'available_physical_mib': 2000, 'commit_headroom_mib': 900}
    gate = x.MemoryAdmission(host_reserve_mib=100, host_status=lambda: dict(host))
    gate.observe('gluomem', rss_mib=100, private_mib=300)
    assert gate.acquire('first', 'gluomem')
    assert not gate.acquire('second', 'gluomem')  # 2*450 commit + reserve >900
    gate.update('first', rss_mib=100, private_mib=300)
    assert gate.snapshot()['startup_pending_commit_mib'] == 150
    assert gate.snapshot()['startup_pending_rss_mib'] == 50
    gate.release('first')
    assert gate.acquire('second', 'gluomem')
    host['commit_headroom_mib'] = 99
    assert not gate.acquire('third', 'gluomem')
    assert gate.snapshot()['pressure_latched'] is True
    host['commit_headroom_mib'] = 9000
    assert not gate.acquire('fourth', 'gluomem')  # Fail closed, no automatic recovery.


def test_chronological_process_tree_excludes_recycled_parent_pid():
    x = executor()
    rows = [dict(pid=10,parent_pid=1,created_100ns=100),
            dict(pid=20,parent_pid=10,created_100ns=90),  # recycled parent PID
            dict(pid=30,parent_pid=10,created_100ns=110),
            dict(pid=40,parent_pid=30,created_100ns=120)]
    assert [r['pid'] for r in x.chronological_tree(rows,10,100)] == [10,30,40]
    with pytest.raises(OSError):
        x.chronological_tree(rows,10,99)  # exact original identity required


def test_os_commit_limit_refuses_only_owned_child_allocation(tmp_path):
    import subprocess, sys
    x = executor()
    code = ("from gluomem_bench.bench.lme_first_execute import ProcessCommitLimit; "
            "cap=ProcessCommitLimit(128); "
            "exec('try:\\n b=bytearray(256*1048576)\\nexcept MemoryError:\\n print(\\\"bounded-oom\\\")\\nelse:\\n raise RuntimeError(\\\"allocation escaped cap\\\")'); "
            "cap.close()")
    result = subprocess.run([sys.executable,'-c',code],capture_output=True,timeout=20)
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.strip() == b'bounded-oom'


def test_scheduler_releases_external_reservation_when_budget_refuses(tmp_path):
    x = executor()
    from gluomem_bench.bench.lme_first import run_isolated_jobs
    reserved = set()
    def acquire(record):
        reserved.add(record['job_id'])
        return True
    rows = [dict(job_id='a',history_id='a',system='gluomem',output=str(tmp_path/'leaf'),
                 memory_mib=32,call_budget=2,seconds_budget=2)]
    result = run_isolated_jobs(rows,out=tmp_path/'coord',run_job=lambda *a: {},max_calls=1,
        max_seconds=30,max_workers=1,native_limits={'gluomem':1},memory_limit_mib=64,
        max_in_flight_tokens=39321600,admit=acquire,
        release_admission=lambda record: reserved.remove(record['job_id']))
    assert result['jobs'][0]['status'] == 'unrun'
    assert not reserved


def test_worker_enforces_declared_commit_limit_and_persists_exact_pids(tmp_path):
    x = executor()
    from gluomem_bench.bench.lme_first import run_isolated_jobs
    rows = [dict(job_id='p',history_id='p',system='gluomem',output=str(tmp_path/'leaf'),
                 memory_mib=128,call_budget=2,seconds_budget=15)]
    with x.ContextBroker() as broker:
        def run(job,context):
            outcome = x.launch_worker({'probe':True,'commit_limit_mib':128},broker=broker,
                context=context,workdir=tmp_path/'process')
            assert outcome['resource']['commit_limit_mib'] == 128
            return outcome
        result = run_isolated_jobs(rows,out=tmp_path/'coord',run_job=run,max_calls=2,
            max_seconds=30,max_workers=1,native_limits={'gluomem':1},memory_limit_mib=256,
            max_in_flight_tokens=39321600)
    assert result['jobs'][0]['status'] == 'complete'
    samples = [json.loads(line) for line in (tmp_path/'process/resource-samples.jsonl').read_bytes().splitlines()]
    assert samples and all(r['root_pid'] > 0 and r['root_created_100ns'] > 0 for r in samples)
    assert all(r['units'] == 'MiB (1048576 bytes)' and r['members'] for r in samples)
    assert all(r['rss_mib'] == sum(m['rss_mib'] for m in r['members']) for r in samples)
    assert all(r['private_mib'] == sum(m['private_mib'] for m in r['members']) for r in samples)
    proof = {k:x.bound(tmp_path/'process'/name) for k,name in
             [('samples','resource-samples.jsonl'),('outcome','outcome.json'),('exit','process-exit.json')]}
    measured = {k:max(r[k] for r in samples) for k in ('rss_mib','private_mib')}
    x.verify_resource_measurement(proof,commit_limit_mib=128,measured=measured)
    samples[0]['private_mib'] = 1  # A rehashed bookkeeping placeholder is not proof.
    path = tmp_path/'process/resource-samples.jsonl'
    path.write_bytes(b''.join(x.json_bytes(r) for r in samples))
    proof['samples'] = x.bound(path)
    with pytest.raises(ValueError,match='resource'):
        x.verify_resource_measurement(proof,commit_limit_mib=128,measured=measured)


def test_memory_policy_reserves_explicit_commit_envelope_not_tiny_peak():
    x = executor()
    gate = x.MemoryAdmission(host_reserve_mib=100,
        host_status=lambda: {'available_physical_mib':2000,'commit_headroom_mib':1000},
        commit_limits={'gluomem':700})
    gate.observe('gluomem',rss_mib=10,private_mib=20)
    assert gate.acquire('a','gluomem')
    assert not gate.acquire('b','gluomem')
    gate.update('a',rss_mib=10,private_mib=20)
    assert gate.snapshot()['startup_pending_commit_mib'] == 680
    assert gate.reservations('gluomem')['private_mib'] == 700


@pytest.mark.parametrize('mutation',['missing','threads','tiny','nan','protocol','reserve'])
def test_operational_resource_contract_fails_closed(mutation):
    x = executor()
    candidate = {'resource_protocol':x.RESOURCE_PROTOCOL,'native_mode':'process',
        'host_reserve_mib':8192,'history_commit_limits_mib':{s:4096 for s in ('gluomem','mem0','cognee','no_memory')}}
    qualification = {'resource_protocol':x.RESOURCE_PROTOCOL,
        'memory_samples':{s:{'rss_mib':300,'private_mib':1200} for s in ('gluomem','mem0','cognee')}}
    x.validate_resource_contract(candidate,qualification,local_memory_mib=16384)
    if mutation == 'missing': candidate.pop('history_commit_limits_mib')
    elif mutation == 'threads': candidate['native_mode'] = 'thread'
    elif mutation == 'tiny': candidate['history_commit_limits_mib']['mem0'] = 1
    elif mutation == 'nan': qualification['memory_samples']['mem0']['private_mib'] = float('nan')
    elif mutation == 'reserve': candidate['host_reserve_mib'] = 512
    else: qualification['resource_protocol'] = 'legacy-placeholder'
    with pytest.raises(ValueError):
        x.validate_resource_contract(candidate,qualification,local_memory_mib=16384)


def test_materialization_cli_is_outcome_blind_and_never_ingests(tmp_path):
    import subprocess,sys
    x = executor()
    source = tmp_path/'source.json'
    rows = [{'question_id':str(i),'question':'FORBIDDEN_QUESTION','answer':'FORBIDDEN_GOLD',
             'haystack_dates':['2024/02/01 (Thu) 12:00'],'haystack_session_ids':['answer_secret'],
             'haystack_sessions':[[{'role':'user','content':'facts '*(i+1)}]]} for i in range(5)]
    source.write_bytes(x.json_bytes(rows))
    out = tmp_path/'materialized'
    result = subprocess.run([sys.executable,'-m','gluomem_bench.bench.lme_first_execute',
        'materialize','--source',str(source),'--source-sha256',x.bound(source)['sha256'],
        '--out',str(out),'--host-reserve-mib','128','--commit-limit-mib','4096'],
        capture_output=True,timeout=45)
    assert result.returncode == 0, result.stdout+result.stderr
    report = json.loads((out/'materialization.json').read_bytes())
    assert report['generation_calls'] == report['encoder_calls'] == 0
    assert report['native_ingestion_complete'] is False
    assert report['full_history_peak_mib'] is None
    assert report['source_histories'] == 5
    assert [r['sample_id'] for r in report['selected']] == ['0','2','3','4']
    assert report['network_policy'] == 'deny-all-sockets'
    assert all(r['source_events'] == 1 for r in report['selected'])
    assert 'FORBIDDEN_GOLD' not in (out/'materialization.json').read_text()
    assert 'FORBIDDEN_QUESTION' not in (out/'materialization.json').read_text()
    assert not list(out.rglob('*.mind'))


def test_three_production_extractors_keep_shared_dspy_history_empty(tmp_path,authored_plan,monkeypatch):
    """Regression for the separately owned scoped RackLM history repair.

    Actual production DSPy extraction concurrently calls authored HTTP responses.
    Native algorithms are not replaced. No released QA or real service access.
    """
    monkeypatch.setenv('LITELLM_LOCAL_MODEL_COST_MAP','True')
    import dspy,httpx,threading
    from datetime import datetime,timezone
    from gluomem.llm.modules import DspyExtractors
    from gluomem_bench.bench.rack import RackClient
    from gluomem_bench.bench.rack_trace import RackRouter,WorkloadBudget
    from dspy.clients.base_lm import record_history
    global_history = record_history.__globals__['GLOBAL_HISTORY']
    before = len(global_history)
    barrier = threading.Barrier(3,timeout=15)
    facts = [f'AUTHORED_TENANT_{i} owns item_{i}.' for i in range(3)]
    def send(client,request,**kwargs):
        assert request.url.host == 'authored.invalid', 'real endpoint forbidden'
        if request.url.path == '/tokenize':
            return httpx.Response(200,json={'count':10},request=request)
        payload = json.loads(request.content)
        visible = json.dumps(payload['messages'])
        matches = [i for i,fact in enumerate(facts) if fact in visible]
        assert len(matches) == 1
        i = matches[0]
        barrier.wait()
        content = json.dumps({'claims':[{'text':facts[i],'kind':'fact','confidence':1,
            'verbatim_span':facts[i],'message_id':'s0t0'}]})
        return httpx.Response(200,json={'id':f'authored-{i}','object':'chat.completion',
            'created':0,'model':'authored/qwen','choices':[{'index':0,'finish_reason':'stop',
            'message':{'role':'assistant','content':content}}],
            'usage':{'prompt_tokens':10,'completion_tokens':10,'total_tokens':20}},request=request)
    monkeypatch.setattr(httpx.Client,'send',send)
    monkeypatch.setenv('GLUOMEM_RACK_API_KEY','local-endpoint-placeholder')
    environment = dict(__import__('os').environ)
    cfg,profile = executor().effective_config(authored_plan)
    cfg = cfg.with_overrides({'dspy.tasks.event_claims.module':'predict'})
    def extract(i):
        budget = WorkloadBudget(max_calls=2,max_seconds=30)
        client = RackClient(profile,receipt=tmp_path/f'calls-{i}.jsonl',budget=budget)
        try:
            router = RackRouter(cfg,client=client,trace=tmp_path/f'raw-{i}.jsonl',budget=budget)
            extractor = DspyExtractors(cfg,router=router)
            claims = extractor.extract_event_claims(facts[i],reference_date=datetime(2024,i+1,1,tzinfo=timezone.utc))
            assert claims[0].text == facts[i]
            assert budget.calls == 1
            return claims
        finally:
            client.close()
    try:
        with ThreadPoolExecutor(3) as pool:
            list(pool.map(extract,range(3)))
        assert dict(__import__('os').environ) == environment
        shared = json.dumps(global_history[before:],default=str)
        assert not global_history[before:], 'scoped RackLM may not append to DSPy GLOBAL_HISTORY'
        assert all(fact not in shared for fact in facts)
        for i in range(3):
            own = (tmp_path/f'raw-{i}.jsonl').read_text()
            assert facts[i] in own
            assert all(facts[j] not in own for j in range(3) if j!=i)
    finally:
        del global_history[before:]  # Restore test-only global records.


def test_qualification_closure_requires_native_backends_not_only_executor(tmp_path):
    x = executor()
    files = [{'path':str(tmp_path/name),'sha256':str(i)*64} for i,name in enumerate(
        ('lme_first_execute.py','core/source.py','mem0/native.py','cognee/native.py','schema.sql',
         'site-packages/mem0/prompt.json'),1)]
    qualification = {'executor_sources':files[:1],'source_closure':files[:1]}
    with pytest.raises(ValueError,match='closure'):
        x.validate_qualification_closure(files,qualification)
    qualification['source_closure'] = list(files)
    x.validate_qualification_closure(files,qualification)
    qualification['source_closure'][-1] = dict(files[-1],sha256='0'*64)
    with pytest.raises(ValueError,match='closure'):
        x.validate_qualification_closure(files,qualification)


def test_coordinator_reports_bookkeeping_separately_from_remote_slots(tmp_path):
    from gluomem_bench.bench.lme_first import run_isolated_jobs
    rows = [dict(job_id='a',history_id='a',system='gluomem',output=str(tmp_path/'leaf'),
                 memory_mib=32,call_budget=1,seconds_budget=2)]
    result = run_isolated_jobs(rows,out=tmp_path/'coord',run_job=lambda *a: {},max_calls=1,
        max_seconds=30,max_workers=3,native_limits={'gluomem':3},memory_limit_mib=64,
        max_in_flight_tokens=39321600)
    assert result['peak_active_history_jobs'] == 1
    assert result['memory_metric'] == 'scheduler-reservations-not-physical-measurement'
    assert result['policy']['max_in_flight'] == 300
    assert result['limiter']['peak_in_flight'] == 0


def test_tiny_commit_cap_fails_one_owned_worker_not_next_job(tmp_path):
    x = executor()
    from gluomem_bench.bench.lme_first import run_isolated_jobs
    rows = [dict(job_id=str(i),history_id=str(i),system='gluomem',output=str(tmp_path/str(i)),
                 memory_mib=128,call_budget=2,seconds_budget=15) for i in range(2)]
    with x.ContextBroker() as broker:
        def run(job,context):
            return x.launch_worker({'probe':True,'commit_limit_mib':1 if job.job_id=='0' else 128},
                broker=broker,context=context,workdir=tmp_path/('worker-'+job.job_id))
        result = run_isolated_jobs(rows,out=tmp_path/'coord',run_job=run,max_calls=4,
            max_seconds=40,max_workers=1,native_limits={'gluomem':1},memory_limit_mib=256,
            max_in_flight_tokens=39321600)
    assert [r['status'] for r in result['jobs']] == ['failed','complete']
    assert result['accounting']['calls'] == 1
    assert (tmp_path/'worker-0/process-exit.json').is_file()


def test_missing_os_sample_receipts_cannot_qualify_positive_summary():
    x = executor()
    with pytest.raises(ValueError,match='resource'):
        x.verify_qualification_resources({'memory_samples':{s:{'rss_mib':1,'private_mib':1}
            for s in ('gluomem','mem0','cognee')}},
            {s:4096 for s in ('gluomem','mem0','cognee','no_memory')})


def test_native_entry_uses_real_history_reader_and_scorer_seams(tmp_path, monkeypatch):
    x = executor()
    from types import SimpleNamespace
    from gluomem_bench.bench import suite_campaign, cognee_driver
    calls = []
    monkeypatch.setattr(suite_campaign, 'run_history', lambda **kw: calls.append(('history',kw)))
    monkeypatch.setattr(x, 'verify_native_job', lambda **kw: {'verified':True})
    monkeypatch.setattr(cognee_driver, 'run_graph_job', lambda **kw: calls.append(('graph',kw)))
    common = dict(context=object(),cfg=object(),adapter=object(),sample=object(),profile=object(),
                  judge_source=object(),embedder=object())
    for system in ('gluomem','mem0','no_memory','cognee'):
        job = SimpleNamespace(system=system,output=tmp_path/system)
        x.run_native_job(job=job,cognee_settings=object(),**common)
    assert [c[0] for c in calls] == ['history','history','history','graph']
    assert all(c[1].get('outer_budget',c[1].get('context')) is common['context'] for c in calls)
    assert calls[-1][1]['settings'] is not None


def test_external_clock_and_memory_gate_reach_scheduler(tmp_path):
    from gluomem_bench.bench.lme_first import run_isolated_jobs
    from gluomem_bench.bench.parallel_campaign import CampaignBudget
    budget = CampaignBudget(max_calls=2,max_seconds=30)
    budget.started -= 2
    rows = [dict(job_id='a',history_id='a',system='gluomem',output=str(tmp_path/'leaf'),
                 memory_mib=1,call_budget=1,seconds_budget=2)]
    result = run_isolated_jobs(rows,out=tmp_path/'coord',run_job=lambda *a: {},max_calls=2,
              max_seconds=30,max_workers=1,native_limits={'gluomem':1},memory_limit_mib=2,
              max_in_flight_tokens=39321600,budget=budget,admit=lambda record: False)
    assert result['jobs'][0]['status'] == 'unrun'
    assert result['accounting']['elapsed_seconds'] >= 2


def test_process_probe_uses_one_real_shared_journal(tmp_path):
    x = executor()
    from gluomem_bench.bench.lme_first import run_isolated_jobs, verify_journal
    rows = [dict(job_id='p',history_id='p',system='mem0',output=str(tmp_path/'leaf'),
                 memory_mib=1,call_budget=2,seconds_budget=15)]
    with x.ContextBroker() as broker:
        def run(job,context):
            return x.probe_subprocess(broker,job,context,tmp_path)
        report = run_isolated_jobs(rows,out=tmp_path/'coord',run_job=run,max_calls=2,
             max_seconds=30,max_workers=1,native_limits={'mem0':1},memory_limit_mib=2,
             max_in_flight_tokens=39321600)
    assert report['accounting']['calls'] == 1
    assert report['jobs'][0]['status'] == 'complete'
    assert verify_journal(tmp_path/'coord',rows) == report


def test_noop_native_output_cannot_be_verified(tmp_path):
    x = executor()
    from types import SimpleNamespace as N
    from gluomem_bench.bench.rack import RackProfile
    with pytest.raises(FileNotFoundError):
        x.verify_native_job(job=N(system='gluomem',output=tmp_path),context=N(calls=0),
            cfg=N(),adapter=N(name='longmemeval_s',sha='a'*64),sample=N(id='authored'),
            profile=RackProfile(base_url='http://authored.invalid/v1',model='authored'),
            judge_source=tmp_path/'no-scorer',require_real=True)


def test_candidate_validation_rejects_false_approval_before_plan(tmp_path):
    x = executor()
    path = tmp_path/'candidate.json'
    path.write_bytes(b'{}')
    with pytest.raises(ValueError,match='parent'):
        x.validate_candidate(path, {'approved':False})


@pytest.mark.parametrize('field',['none','duplicate','drop','spend','clock','summary','complete'])
def test_root_replays_real_journal_and_rejects_forged_authority(tmp_path,authored_plan,field):
    x = executor()
    from gluomem_bench.bench.parallel_campaign import CampaignBudget
    from gluomem_bench.bench.lme_first import run_isolated_jobs
    plan=authored_plan
    plan['jobs']=[dict(job_id=system,history_id=system+':authored',sample_id='authored',system=system,
                      call_budget=4,seconds_budget=30) for system in ('gluomem','mem0','cognee','no_memory')]
    plan['future_budget_proposal']={'max_seconds':40}
    qualification={'memory_samples':{s:{'rss_mib':1,'private_mib':1} for s in ('gluomem','mem0','cognee')}}
    candidate={'output_root':str(tmp_path)}
    inputs=x.build_input_cache(tmp_path,plan,CampaignBudget(max_calls=4,max_seconds=30))
    records=x.native_records(candidate,plan,qualification)
    coordination=run_isolated_jobs(records,out=tmp_path/'coordination',run_job=lambda *a: {},max_calls=1,
        max_seconds=40,max_workers=1,native_limits={s:1 for s in ('gluomem','mem0','cognee','no_memory')},
        memory_limit_mib=8,max_in_flight_tokens=39321600)
    report=x._root_report(candidate,plan,coordination)
    stage=tmp_path/'root-status.pending.json'
    stage.write_bytes(x.json_bytes(report))
    assert x.verify_root(candidate,plan=plan,candidate_path=stage,qualification=qualification)==report
    assert report['quality_complete'] is False
    if field=='none':
        return
    if field=='duplicate':
        report['jobs'][1]=report['jobs'][0]
    elif field=='drop':
        report['jobs'].pop()
    elif field=='spend':
        report['accounting']['calls']=1
    elif field=='summary':
        report['systems']['cognee']['score_sum']=1
    elif field=='complete':
        report['quality_complete']=True
    elif field=='clock':
        path=Path(inputs['authored']['path'])
        data=json.loads(path.read_bytes())
        data['sample']['question_date']='2026-10-18'
        path.write_bytes(x.json_bytes(data))
        inputs['authored']=x.bound(path)
        (tmp_path/'input-cache.json').write_bytes(x.json_bytes(inputs))
    stage.write_bytes(x.json_bytes(report))
    with pytest.raises(ValueError):
        x.verify_root(candidate,plan=plan,candidate_path=stage,qualification=qualification)


@pytest.fixture
def authored_plan(tmp_path):
    from gluomem_bench.bench.rack import RackProfile
    x = executor()
    source = tmp_path/'source.json'
    source.write_bytes(x.json_bytes([{'question_id':'authored','question':'What does Rhea own?',
        'answer':'cobalt telescope','question_type':'single-session-user',
        'question_date':'2024/02/02 (Fri) 12:00','answer_session_ids':['answer_private'],
        'haystack_dates':['2024/02/01 (Thu) 12:00'], 'haystack_session_ids':['answer_private'],
        'haystack_sessions':[[{'role':'user','content':'  Rhea owns a cobalt telescope.\\n'},
                              {'role':'assistant','content':''}]]}]))
    parent = Path(__file__).resolve().parents[2]/'tests/fixtures/native/operator-supplied'
    judge = json.loads(parent.read_bytes())['judge_source']
    return {'dataset':{'path':str(source),'sha256':x.bound(source)['sha256'],'name':'longmemeval_s'},
            'profile':RackProfile(base_url='http://authored.invalid/v1',model='authored/qwen',
                                  max_generation_calls=4,max_run_seconds=30).model_dump(mode='json'),
            'config_overrides':{'benchmark.datasets.longmemeval_s':{'url':'https://authored.invalid/source.json',
                'file':source.name,'sha256':x.bound(source)['sha256']}},
            'sample_ids':['authored'],'judge_source':judge}


@pytest.mark.parametrize('mutation',['none','text','clock','gold','drop'])
def test_exact_source_cache_reconstruction_rejects_rehashed_forgeries(tmp_path,authored_plan,mutation):
    x = executor()
    from gluomem_bench.bench.parallel_campaign import CampaignBudget
    inputs = x.build_input_cache(tmp_path,authored_plan,CampaignBudget(max_calls=4,max_seconds=30))
    x.verify_input_cache(tmp_path,authored_plan,inputs)
    if mutation == 'none':
        return
    path = Path(inputs['authored']['path'])
    data = json.loads(path.read_bytes())
    if mutation == 'text':
        data['conversation']['sessions'][0]['turns'][0]['text'] = 'forged'
    elif mutation == 'clock':
        data['sample']['question_date'] = '2026-10-18'
    elif mutation == 'gold':
        data['sample']['gold'] = 'forged'
    else:
        data['conversation']['sessions'][0]['turns'].pop()
    path.write_bytes(x.json_bytes(data))
    inputs['authored'] = x.bound(path)
    with pytest.raises(ValueError,match='source projection'):
        x.verify_input_cache(tmp_path,authored_plan,inputs)


def test_concrete_native_reader_scorer_closes_with_shared_budget(tmp_path,authored_plan):
    import httpx
    from types import SimpleNamespace as N
    x = executor()
    from gluomem_bench.bench.parallel_campaign import CampaignBudget
    from gluomem_bench.bench.lme_first import run_isolated_jobs
    inputs = x.build_input_cache(tmp_path,authored_plan,CampaignBudget(max_calls=4,max_seconds=30))
    adapter,sample = x.restore_input(inputs['authored'],authored_plan)
    cfg,profile = x.effective_config(authored_plan)
    def serve(request):
        if request.url.path.endswith('/models'):
            return httpx.Response(200,json={'data':[{'id':profile.model,'max_model_len':131072}]})
        if request.url.path == '/tokenize':
            return httpx.Response(200,json={'count':10})
        payload = json.loads(request.content)
        answer = 'cobalt telescope' if payload['messages'][0]['role']=='system' else 'yes'
        return httpx.Response(200,json={'model':profile.model,'choices':[{'finish_reason':'stop',
            'message':{'role':'assistant','content':answer}}],
            'usage':{'prompt_tokens':10,'completion_tokens':3,'total_tokens':13}})
    rows = [dict(job_id='native',history_id='authored',system='no_memory',output=str(tmp_path/'leaf'),
                 memory_mib=1,call_budget=4,seconds_budget=30)]
    def run(job,context):
        native_job = N(output=job.output,system='no_memory')
        return x.run_native_job(job=native_job,context=context,cfg=cfg,adapter=adapter,sample=sample,
            profile=profile,judge_source=Path(authored_plan['judge_source']),embedder=None,
            transport=httpx.MockTransport(serve),require_real=False)
    report = run_isolated_jobs(rows,out=tmp_path/'coord',run_job=run,max_calls=4,max_seconds=40,
        max_workers=1,native_limits={'no_memory':1},memory_limit_mib=1,max_in_flight_tokens=39321600)
    assert report['jobs'][0]['status']=='complete',report
    assert report['accounting']['calls']==2
    prediction = json.loads((tmp_path/'leaf/predictions.json').read_bytes())[0]
    assert prediction['sample']['question_date']==sample.question_date
    assert prediction['judge_verdict']=='yes'


def test_isolated_worker_executes_actual_reader_scorer(tmp_path,authored_plan):
    """Real isolated CLI + loopback HTTP fixture, not real Qwen or benchmark QA."""
    import threading
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    x = executor()
    from gluomem_bench.bench.parallel_campaign import CampaignBudget
    from gluomem_bench.bench.lme_first import run_isolated_jobs
    from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def reply(self,value):
            raw = json.dumps(value).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def do_GET(self):
            self.reply({'data':[{'id':'authored/qwen','max_model_len':131072}]})
        def do_POST(self):
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path=='/tokenize':
                self.reply({'count':10})
                return
            answer='cobalt telescope' if payload['messages'][0]['role']=='system' else 'yes'
            self.reply({'model':'authored/qwen','choices':[{'finish_reason':'stop',
                'message':{'role':'assistant','content':answer}}],
                'usage':{'prompt_tokens':10,'completion_tokens':3,'total_tokens':13}})
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    authored_plan['profile']['base_url']=f'http://127.0.0.1:{server.server_port}/v1'
    authored_plan['embedding_url']='http://authored.invalid/v1'
    recipe=EmbeddingRecipe()
    authored_plan['embedding_recipe']=recipe.as_dict()
    authored_plan['config_overrides'].update({'models.minilm.provider_model':recipe.model,
        'models.minilm.revision':recipe.revision,'models.minilm.dim':recipe.dim,
        'storage.vector_dim':recipe.dim})
    inputs=x.build_input_cache(tmp_path,authored_plan,CampaignBudget(max_calls=4,max_seconds=30))
    plan_path=tmp_path/'plan.json'
    plan_path.write_bytes(x.json_bytes(authored_plan))
    candidate_path=tmp_path/'authored-candidate.json'
    candidate_path.write_bytes(x.json_bytes({'plan':x.bound(plan_path),'files':[]}))
    rows=[dict(job_id='native-process',history_id='authored',system='no_memory',output=str(tmp_path/'leaf'),
               memory_mib=1,call_budget=4,seconds_budget=30)]
    try:
        with x.ContextBroker() as broker:
            def run(job,context):
                return x.launch_worker({'candidate':x.bound(candidate_path),'input':inputs['authored'],
                    'system':'no_memory'},broker=broker,context=context,workdir=tmp_path/'process')
            report=run_isolated_jobs(rows,out=tmp_path/'coord',run_job=run,max_calls=4,max_seconds=40,
                max_workers=1,native_limits={'no_memory':1},memory_limit_mib=1,max_in_flight_tokens=39321600)
        assert report['jobs'][0]['status']=='complete', json.loads((tmp_path/'process/outcome.json').read_bytes())
        assert report['accounting']['calls']==2
        assert (tmp_path/'leaf/manifest.json').is_file()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_cli_refuses_unapproved_before_output(tmp_path):
    x = executor()
    candidate = tmp_path/'candidate.json'
    candidate.write_text('{}')
    approval = tmp_path/'approval.json'
    approval.write_text('{}')
    assert x.main(['execute','--candidate',str(candidate),'--approval',str(approval)]) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ['approval.json','candidate.json']


def test_parent_budget_binds_same_windows_plan_identity(tmp_path):
    x=executor()
    base=Path(__file__).resolve().parents[2]/'tests/fixtures/native/operator-supplied'
    parent=base/'lme-first-parent-prospective-budget-decision-v1.json'
    decision=json.loads(parent.read_bytes())
    plan=x.bound(decision['plan']['path'])  # Native Windows backslashes vs parent forward slashes.
    candidate={'protocol':x.EXECUTION_PROTOCOL,'producer_id':'fixture-producer',
               'plan':plan,'budget_decision':x.bound(parent),'files':[]}
    target=tmp_path/'candidate.json'
    target.write_bytes(x.json_bytes(candidate))
    review={'approved':True,'blockers':[],'candidate_sha256':x.bound(target)['sha256'],
            'reviewer_kind':'AI-assisted','reviewer_id':'fixture-reviewer'}
    review_path=tmp_path/'review.json'
    review_path.write_bytes(x.json_bytes(review))
    approval={'approved':True,'approver_kind':'parent','operational_full_benchmark_execution_authorized':True,
              'candidate_sha256':x.bound(target)['sha256'],'independent_review':x.bound(review_path)}
    # Must reach the intended missing-source gate, not reject equivalent plan paths.
    with pytest.raises(ValueError,match='source/non-Python'):
        x.validate_candidate(target,approval)


def test_cli_never_formats_driver_exception(tmp_path,monkeypatch,capsys):
    x=executor()
    def fail(*args):
        raise RuntimeError('PRIVATE_BODY_HEADER_REASON_TOKEN')
    monkeypatch.setattr(x,'execute_candidate',fail)
    assert x.main(['execute','--candidate',str(tmp_path/'c'),'--approval',str(tmp_path/'a')])==1
    text=capsys.readouterr()
    assert 'PRIVATE_BODY_HEADER_REASON_TOKEN' not in text.out+text.err
    assert json.loads(text.out)['status']=='refused-or-incomplete'
