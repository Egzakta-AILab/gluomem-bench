"""Shared process contracts; authored/OS fixtures, never real services."""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from gluomem_bench.bench import lme_first_execute as x
from gluomem_bench.bench.lme_first import run_isolated_jobs, verify_journal
from tests.bench.test_lme_first_execute import authored_plan


@pytest.mark.parametrize('local_failure',[False,True])
def test_three_actual_native_factories_keep_stores_clocks_traces_and_close(tmp_path,authored_plan,monkeypatch,local_failure):
    """Real ingestion/search/reader/scorer; authored generation/vector fixtures only."""
    import copy
    import threading
    import httpx
    import numpy as np
    import dspy
    from types import SimpleNamespace
    from gluomem_bench.bench.embedding_runtime import RemoteEmbedder,EmbeddingRecipe
    from gluomem_bench.bench.parallel_campaign import CampaignBudget
    from gluomem.storage.db import MindDB
    from dspy.clients.base_lm import record_history
    global_history = record_history.__globals__['GLOBAL_HISTORY']
    before = list(global_history)
    monkeypatch.setenv('GLUOMEM_RACK_API_KEY','local-endpoint-placeholder')
    monkeypatch.setenv('LITELLM_LOCAL_MODEL_COST_MAP','True')
    count = 4 if local_failure else 3
    facts = [f'AUTHORED_TENANT_{i} owns unique_item_{i}.' for i in range(count)]
    barrier = threading.Barrier(3,timeout=20)
    seen = []
    lock = threading.Lock()
    def send(client,request,**kwargs):
        assert request.url.host == 'authored.invalid', 'real services forbidden'
        if request.url.path.endswith('/models'):
            return httpx.Response(200,json={'data':[{'id':'authored/qwen','max_model_len':131072}]},request=request)
        if request.url.path == '/tokenize':
            return httpx.Response(200,json={'count':10},request=request)
        payload = json.loads(request.content)
        visible = json.dumps(payload['messages'])
        matches = [i for i in range(count) if f'AUTHORED_TENANT_{i}' in visible]
        if len(matches) != 1:
            # Native scorer contains the authored QA (tenant-specific).
            raise AssertionError('cross-tenant or unidentified native input')
        i = matches[0]
        with lock:
            seen.append((i,payload))
        if 'claims' in visible and 'reference' in visible.lower():
            if i < 3:
                barrier.wait()
            content = json.dumps({'claims':[{'text':facts[i],'kind':'fact','confidence':1,
                'verbatim_span':facts[i],'message_id':'s0t0'}]})
        elif payload['messages'][0]['role'] == 'user':
            content = 'yes'
        else:
            content = f'unique_item_{i}'
        finish = 'length' if local_failure and i==1 else 'stop'
        return httpx.Response(200,json={'model':'authored/qwen','choices':[{'index':0,'finish_reason':finish,
            'message':{'role':'assistant','content':content}}],
            'usage':{'prompt_tokens':10,'completion_tokens':10,'total_tokens':20}},request=request)
    monkeypatch.setattr(httpx.Client,'send',send)
    recipe = EmbeddingRecipe()
    class AuthoredVectors(RemoteEmbedder):
        def _encode(self,texts,role):
            values = np.zeros((len(texts),768),dtype=np.float32)
            values[:,0] = 1
            return values
    original_close = MindDB.close
    opened,closed = [],[]
    original_init = MindDB.__init__
    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        with lock: opened.append(self)
    def close(self):
        with lock: closed.append(self)
        return original_close(self)
    monkeypatch.setattr(MindDB,'__init__',init)
    monkeypatch.setattr(MindDB,'close',close)
    environment = dict(os.environ)
    def run(job,context):
        i = int(job.job_id)
        root = tmp_path/f'native-{i}'
        root.mkdir()
        plan = copy.deepcopy(authored_plan)
        source = root/'source.json'
        raw = json.loads(open(plan['dataset']['path'],encoding='utf8').read())
        raw[0].update(question_id=f'authored-{i}',question=f'What does AUTHORED_TENANT_{i} own?',
            answer=f'unique_item_{i}',question_date=f'2024/0{i+2}/02 (Fri) 12:00',
            haystack_dates=[f'2024/0{i+1}/01 (Thu) 12:00','2023/12/31 (Sun) 12:00'],
            haystack_session_ids=['answer_private','older_assistant'],
            haystack_sessions=[[{'role':'user','content':facts[i]}],
                [{'role':'assistant','content':'Acknowledged previous message.'}]])
        source.write_bytes(x.json_bytes(raw))
        plan['dataset'].update(path=str(source),sha256=x.bound(source)['sha256'])
        plan['sample_ids'] = [f'authored-{i}']
        plan['config_overrides'].update({'benchmark.datasets_dir':str(root),
            'benchmark.datasets.longmemeval_s':{'url':'https://authored.invalid/source.json','file':source.name,
                'sha256':plan['dataset']['sha256']},'dspy.tasks.event_claims.module':'predict',
            'models.minilm.provider_model':recipe.model,'models.minilm.revision':recipe.revision,
            'models.minilm.dim':768,'storage.vector_dim':768,'search.abstain.enabled':False,
            'search.rerank.enabled':False})
        inputs = x.build_input_cache(root,plan,CampaignBudget(max_calls=8,max_seconds=90))
        adapter,sample = x.restore_input(inputs[f'authored-{i}'],plan)
        cfg,profile = x.effective_config(plan)
        return x.run_native_job(job=SimpleNamespace(system='gluomem',output=root/'leaf'),
            context=context,
            cfg=cfg,adapter=adapter,sample=sample,profile=profile,
            judge_source=__import__('pathlib').Path(plan['judge_source']),
            embedder=AuthoredVectors('http://authored.invalid/v1',recipe=recipe),require_real=False)
    try:
        rows = [dict(job_id=str(i),history_id=str(i),system='gluomem',
            output=str(tmp_path/f'native-{i}'/'leaf'),memory_mib=1,call_budget=4,
            seconds_budget=30) for i in range(count)]
        result = run_isolated_jobs(rows,out=tmp_path/'coord',run_job=run,
            max_calls=4*count,max_seconds=90,max_workers=3,native_limits={'gluomem':3},
            memory_limit_mib=3,max_in_flight_tokens=39321600)
        expected = ['complete','failed','complete','complete'] if local_failure else ['complete']*3
        assert [r['status'] for r in result['jobs']] == expected, result
        assert result['accounting']['calls'] == (10 if local_failure else 9)
        assert result['limiter']['peak_in_flight'] >= 3
        assert verify_journal(tmp_path/'coord',rows) == result
        assert global_history == before, 'RackLM may not append tenant data to DSPy GLOBAL_HISTORY'
        assert dict(os.environ) == environment
        assert opened and all(db in closed for db in opened), 'every native DB handle must close'
        for i in range(count):
            root = tmp_path/f'native-{i}'/'leaf'
            if local_failure and i==1:
                assert (root/'failure.json').is_file()
                assert not (root/'manifest.json').exists()
                continue
            raw_trace = (root/'production-raw.jsonl').read_text()
            assert facts[i] in raw_trace
            assert all(facts[j] not in raw_trace for j in range(count) if j != i)
            prediction = json.loads((root/'predictions.json').read_bytes())[0]
            sentinel = json.loads((root/'stores'/f'authored-{i}.ingest.json').read_bytes())
            assert [r['event_id'] for r in sentinel['committed_receipts']] == [
                f'authored-{i}:session:2',f'authored-{i}:session:1']
            assert prediction['sample']['question_date'] == f'2024/0{i+2}/02 (Fri) 12:00'
            assert any(facts[i] in d['text'] for d in prediction['prediction']['retrieved'])
            assert all(f'AUTHORED_TENANT_{j}' not in json.dumps(prediction) for j in range(count) if j!=i)
            extraction = [p for tenant,p in seen if tenant==i and 'claims' in json.dumps(p['messages'])]
            assert any(f'2024-0{i+1}-01' in json.dumps(p) for p in extraction)
    finally:
        global_history[:] = before



def test_shared_owned_process_accounts_three_jobs_and_one_cap(tmp_path):
    rows = [dict(job_id=str(i), history_id=str(i), system='gluomem',
                 output=str(tmp_path/str(i)), memory_mib=1, call_budget=2,
                 seconds_budget=15) for i in range(3)]
    with x.ContextBroker() as broker:
        with x.SharedGluomemWorker(workdir=tmp_path/'group', commit_limit_mib=256,
                                  max_workers=3) as group:
            def run(job, context):
                return group.run({'probe': True, 'system': 'gluomem'}, broker=broker,
                                 context=context, workdir=tmp_path/('worker-'+job.job_id))
            report = run_isolated_jobs(rows, out=tmp_path/'coord', run_job=run,
                max_calls=6, max_seconds=40, max_workers=3, native_limits={'gluomem':3},
                memory_limit_mib=3, max_in_flight_tokens=39321600)
        assert group.process.pid != os.getpid()
    assert [r['status'] for r in report['jobs']] == ['complete']*3
    assert report['accounting']['calls'] == 3
    assert verify_journal(tmp_path/'coord', rows) == report
    outcomes = [json.loads((tmp_path/('worker-'+str(i))/'outcome.json').read_bytes()) for i in range(3)]
    assert len({r['resource']['pid'] for r in outcomes}) == 1
    assert all(r['resource']['scope'] == 'shared-gluomem-group-not-per-history' for r in outcomes)
    proof = {k:x.bound(tmp_path/'group'/name) for k,name in
             [('samples','resource-samples.jsonl'),('outcome','outcome.json'),('exit','process-exit.json')]}
    samples = [json.loads(line) for line in (tmp_path/'group/resource-samples.jsonl').read_bytes().splitlines()]
    x.verify_resource_measurement(proof,commit_limit_mib=256,
        measured={k:max(r[k] for r in samples) for k in ('rss_mib','private_mib')})


def test_scheduler_charges_shared_group_once_and_replays_it(tmp_path):
    import threading
    barrier = threading.Barrier(3,timeout=3)
    rows = [dict(job_id=str(i),history_id=str(i),system='gluomem',
                 output=str(tmp_path/str(i)),memory_mib=256,memory_group='gluomem',
                 call_budget=1,seconds_budget=5) for i in range(3)]
    def run(job,context):
        barrier.wait()
        with context.lease(100):
            context.begin_call()
        return {}
    result = run_isolated_jobs(rows,out=tmp_path/'coord',run_job=run,max_calls=3,
        max_seconds=20,max_workers=3,native_limits={'gluomem':3},memory_limit_mib=256,
        max_in_flight_tokens=39321600)
    assert [r['status'] for r in result['jobs']] == ['complete']*3
    assert result['peak_memory_mib'] == 256
    assert result['peak_active_history_jobs'] == 3
    assert verify_journal(tmp_path/'coord',rows) == result


def test_post_recovery_permit_is_explicit_bound_authority(tmp_path):
    base = tmp_path/'budget.json'
    plan = tmp_path/'plan.json'
    plan.write_bytes(b'{}')
    base.write_bytes(x.json_bytes({'approver_kind':'parent','plan':x.bound(plan),
        'bounded_authored_real_backend_qualification_permitted':{
            'generation_admission_cap':192,'wall_seconds_cap':1200,'retries':0}}))
    permit = tmp_path/'permit.json'
    permit.write_bytes(x.json_bytes({'protocol':'lme-first-parent-post-recovery-qualification-permit@v1',
        'approver_kind':'parent','approved':True,'base_budget_decision':str(base),
        'new_generation_admission_cap':192,'new_wall_seconds_cap':1200,'new_attempts_per_backend':1,
        'generation_retries':0,'history_commit_limit_mib':4096,
        'encoder_repeat_dispatch_latch_required':True,
        'host_physical_and_commit_reserve_mib':8192,'bootstrap_resident_histories':1,
        'benchmark_released_qa_authorized':False,'full_campaign_execution_authorized':False}))
    kwargs = dict(plan_path=plan,plan_sha256=x.bound(plan)['sha256'],budget_decision_path=base,
        budget_decision_sha256=x.bound(base)['sha256'],source_path=tmp_path/'missing-source',
        out=tmp_path/'out',cognee_python='unused',host_reserve_mib=8192,
        history_commit_limit_mib=4096,post_recovery_permit_path=permit,
        post_recovery_permit_sha256=x.bound(permit)['sha256'])
    with pytest.raises(FileNotFoundError):
        x.run_authored_qualification(**kwargs)  # Authority accepted; source not opened earlier.
    assert not (tmp_path/'out').exists()
    kwargs['host_reserve_mib'] = 4096
    with pytest.raises(ValueError,match='post-recovery'):
        x.run_authored_qualification(**kwargs)
    kwargs['host_reserve_mib'] = 8192
    kwargs['post_recovery_permit_sha256'] = '0'*64
    with pytest.raises(ValueError,match='binding'):
        x.run_authored_qualification(**kwargs)
    material = json.loads(permit.read_bytes())
    material['encoder_repeat_dispatch_latch_required'] = False
    permit.write_bytes(x.json_bytes(material))
    kwargs['post_recovery_permit_sha256'] = x.bound(permit)['sha256']
    with pytest.raises(ValueError,match='post-recovery'):
        x.run_authored_qualification(**kwargs)


def test_shared_process_runs_actual_native_factories_and_healthy_successor(tmp_path,authored_plan,monkeypatch):
    import copy
    import threading
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    from pathlib import Path
    from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe
    from gluomem_bench.bench.parallel_campaign import CampaignBudget
    recipe = EmbeddingRecipe()
    facts = [f'AUTHORED_TENANT_{i} owns unique_item_{i}.' for i in range(5)]
    barrier = threading.Barrier(3,timeout=25)
    lock = threading.Lock()
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def reply(self,value):
            raw = json.dumps(value).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def do_GET(self):
            assert self.path == '/v1/models'
            self.reply({'data':[{'id':'authored/qwen','max_model_len':131072}]})
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/v1/embeddings':
                texts = payload['input']
                self.reply({'model':recipe.model,'recipe':recipe.as_dict(),'role':payload['input_type'],
                    'data':[{'index':i,'embedding':[1.0]+[0.0]*767} for i in range(len(texts))]})
                return
            if self.path == '/tokenize':
                self.reply({'count':10})
                return
            assert self.path == '/v1/chat/completions'
            visible = json.dumps(payload['messages'])
            matches = [i for i in range(5) if f'AUTHORED_TENANT_{i}' in visible]
            assert len(matches)==1, 'native tenant input contaminated'
            i = matches[0]
            with lock: requests.append((i,payload))
            extraction = 'claims' in visible and 'reference' in visible.lower()
            if extraction:
                if i<3: barrier.wait()
                content = json.dumps({'claims':[{'text':facts[i],'kind':'fact','confidence':1,
                    'verbatim_span':facts[i],'message_id':'s0t0'}]})
            else:
                content = 'yes' if payload['messages'][0]['role']=='user' else f'unique_item_{i}'
            self.reply({'model':'authored/qwen','choices':[{'index':0,
                'finish_reason':'length' if i==3 else 'stop','message':{'role':'assistant','content':content}}],
                'usage':{'prompt_tokens':10,'completion_tokens':10,'total_tokens':20}})
    server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    plan = copy.deepcopy(authored_plan)
    original = json.loads(Path(plan['dataset']['path']).read_bytes())[0]
    raw = []
    for i in range(5):
        item = copy.deepcopy(original)
        item.update(question_id=f'authored-{i}',question=f'What does AUTHORED_TENANT_{i} own?',
            answer=f'unique_item_{i}',question_date=f'2024/0{i+2}/02 (Fri) 12:00',
            haystack_dates=[f'2024/0{i+1}/01 (Thu) 12:00','2023/12/31 (Sun) 12:00'],
            haystack_session_ids=['answer_private','older_assistant'],
            haystack_sessions=[[{'role':'user','content':facts[i]}],
                [{'role':'assistant','content':'Acknowledged previous message.'}]])
        raw.append(item)
    source = tmp_path/'five-source.json'
    source.write_bytes(x.json_bytes(raw))
    plan['dataset'].update(path=str(source),sha256=x.bound(source)['sha256'])
    plan['sample_ids'] = [f'authored-{i}' for i in range(5)]
    plan['profile']['base_url'] = f'http://127.0.0.1:{server.server_port}/v1'
    plan['profile']['max_run_seconds'] = 60
    plan['embedding_url'] = plan['profile']['base_url']
    plan['embedding_recipe'] = recipe.as_dict()
    plan['config_overrides'].update({'benchmark.datasets_dir':str(tmp_path),
        'benchmark.datasets.longmemeval_s':{'url':'https://authored.invalid/source.json','file':source.name,
            'sha256':plan['dataset']['sha256']},'dspy.tasks.event_claims.module':'predict',
        'models.minilm.provider_model':recipe.model,'models.minilm.revision':recipe.revision,
        'models.minilm.dim':768,'storage.vector_dim':768,'search.abstain.enabled':False,
        'search.rerank.enabled':False})
    inputs = x.build_input_cache(tmp_path,plan,CampaignBudget(max_calls=20,max_seconds=120))
    plan_path = tmp_path/'plan.json'
    plan_path.write_bytes(x.json_bytes(plan))
    candidate = tmp_path/'candidate.json'
    candidate.write_bytes(x.json_bytes({'plan':x.bound(plan_path),'files':[]}))
    rows = [dict(job_id=str(i),history_id=str(i),sample_id=f'authored-{i}',system='gluomem',output=str(tmp_path/f'leaf-{i}'),
        memory_mib=4096,memory_group='gluomem',call_budget=4,seconds_budget=60) for i in range(5)]
    try:
        with x.ContextBroker() as broker:
            with x.SharedGluomemWorker(workdir=tmp_path/'group',commit_limit_mib=4096,max_workers=3,
                    network_endpoints=[broker.address,('127.0.0.1',server.server_port)],host_reserve_mib=8192) as group:
                def run(job,context):
                    # Failure's healthy successor starts only after failed native close.
                    if job.job_id=='4':
                        while not (tmp_path/'worker-3'/'outcome.json').exists():
                            context.check()
                            __import__('time').sleep(.01)
                    return group.run({'candidate':x.bound(candidate),'input':inputs[f'authored-{job.job_id}'],
                        'system':'gluomem'},broker=broker,context=context,
                        workdir=tmp_path/('worker-'+job.job_id))
                report = run_isolated_jobs(rows,out=tmp_path/'coord',run_job=run,max_calls=20,
                    max_seconds=120,max_workers=3,native_limits={'gluomem':3},memory_limit_mib=4096,
                    max_in_flight_tokens=39321600)
        assert [r['status'] for r in report['jobs']] == ['complete']*3+['failed','complete'], report
        assert report['accounting']['calls'] == 13
        assert report['limiter']['peak_in_flight'] >= 3
        assert verify_journal(tmp_path/'coord',rows)==report
        outcomes = [json.loads((tmp_path/f'worker-{i}'/'outcome.json').read_bytes()) for i in range(5)]
        assert len({r['resource']['pid'] for r in outcomes}) == 1
        assert outcomes[0]['resource']['pid'] != os.getpid()
        for i in (0,1,2,4):
            prediction = json.loads((tmp_path/f'leaf-{i}'/'predictions.json').read_bytes())[0]
            sentinel = json.loads((tmp_path/f'leaf-{i}'/'stores'/f'authored-{i}.ingest.json').read_bytes())
            assert [r['event_id'] for r in sentinel['committed_receipts']] == [
                f'authored-{i}:session:2',f'authored-{i}:session:1']
            assert any(facts[i] in hit['text'] for hit in prediction['prediction']['retrieved'])
            assert prediction['sample']['question_date'] == raw[i]['question_date']
            trace = (tmp_path/f'leaf-{i}'/'production-raw.jsonl').read_text()
            assert all(facts[j] not in trace for j in range(5) if i!=j)
        proof = {k:x.bound(tmp_path/'group'/name) for k,name in
            [('samples','resource-samples.jsonl'),('outcome','outcome.json'),('exit','process-exit.json')]}
        samples = [json.loads(line) for line in (tmp_path/'group/resource-samples.jsonl').read_bytes().splitlines()]
        measured = {k:max(r[k] for r in samples) for k in ('rss_mib','private_mib')}
        x.verify_resource_measurement(proof,commit_limit_mib=4096,measured=measured,host_reserve_mib=8192)
        host_tamper = tmp_path/'host-tamper.jsonl'
        damaged = copy.deepcopy(samples)
        damaged[0]['host_status']['commit_headroom_mib'] = 8191
        host_tamper.write_bytes(b''.join(x.json_bytes(row) for row in damaged))
        with pytest.raises(ValueError,match='host reserve'):
            x.verify_resource_measurement(dict(proof,samples=x.bound(host_tamper)),
                commit_limit_mib=4096,measured=measured,host_reserve_mib=8192)
        packet = {'protocol':'lme-first-shared-gluomem@v1','executor':x.bound(x.__file__),
            'coordinator':x.bound(__import__('gluomem_bench.bench.lme_first',fromlist=['']).__file__),
            'rack_boundary':x.bound(__import__('gluomem_bench.bench.rack_trace',fromlist=['']).__file__),
            'tests':x.bound(__file__),'plan':x.bound(plan_path),'inputs':inputs,'records':rows,
            'coordinator_pid':os.getpid(),
            'worker_outcomes':{str(i):x.bound(tmp_path/f'worker-{i}'/'outcome.json') for i in range(5)},
            'coordination':str(tmp_path/'coord'),'resource_evidence':proof,'memory_samples':measured,
            'group_commit_limit_mib':4096,'host_reserve_mib':8192,
            'assertions':{key:True for key in x.SHARED_ASSERTIONS},
            'evidence_files':[x.bound(p) for p in sorted(tmp_path.rglob('*')) if p.is_file()]}
        from gluomem_bench.bench.comprehensive_driver import frozen_inventory
        packet['source_closure'] = frozen_inventory(source,plan['judge_source'])
        packet_path = tmp_path/'shared-qualification.json'
        packet_path.write_bytes(x.json_bytes(packet))
        x.validate_shared_qualification(x.bound(packet_path),commit_limit_mib=4096,host_reserve_mib=8192)
        qualification = {'resource_protocol':x.RESOURCE_PROTOCOL,
            'memory_samples':{system:{'rss_mib':100,'private_mib':200} for system in ('gluomem','mem0','cognee')}}
        operational = {'native_mode':'gluomem-shared-process','resource_protocol':x.RESOURCE_PROTOCOL,
            'host_reserve_mib':8192,'history_commit_limits_mib':{s:4096 for s in
                ('mem0','cognee','no_memory')},'gluomem_group_commit_limit_mib':4096,
            'gluomem_qualification_process_commit_limit_mib':4096,
            'shared_gluomem_qualification':x.bound(packet_path)}
        x.validate_resource_contract(operational,qualification,local_memory_mib=32768)
        operational.pop('shared_gluomem_qualification')
        with pytest.raises(ValueError,match='shared'):
            x.validate_resource_contract(operational,qualification,local_memory_mib=32768)
        # Exercise the REAL execute_candidate dispatch and root finalization, not
        # merely the low-level group callback. Only exact parent admission is
        # fixture-substituted; all factories, RPC, caps and verifiers are real.
        execution_plan = copy.deepcopy(plan)
        execution_plan.update(jobs=rows,future_budget_proposal={'max_calls':20,'max_seconds':120},
                              native_constraints={'local_memory_mib':32768})
        execution_plan_path = tmp_path/'execute-plan.json'
        execution_plan_path.write_bytes(x.json_bytes(execution_plan))
        operational.update(protocol=x.EXECUTION_PROTOCOL,output_root=str(tmp_path/'executed'),
            plan=x.bound(execution_plan_path),files=[],max_workers=3,
            native_limits={'gluomem':3,'mem0':1,'cognee':1,'no_memory':1},
            shared_gluomem_qualification=x.bound(packet_path))
        execute_candidate = tmp_path/'execute-candidate.json'
        execute_candidate.write_bytes(x.json_bytes(operational))
        approval = tmp_path/'fixture-admission.json'
        approval.write_bytes(b'{}')
        monkeypatch.setattr(x,'validate_candidate',lambda *args,**kwargs:(operational,execution_plan,qualification))
        actual = x.execute_candidate(execute_candidate,approval)
        assert actual['quality_complete'] is False
        assert actual['accounting']['calls'] == 13
        assert [r['status'] for r in actual['jobs']] == ['complete']*3+['failed','complete']
        assert actual['memory_admission']['resident_jobs'] == 0
        assert 'gluomem' not in actual['memory_admission']['commit_limits_mib']
        assert actual['memory_admission']['commit_limits_mib']['gluomem-group'] == 4096
        assert actual['shared_gluomem_resources']['commit_limit_mib'] == 4096
        assert actual['native_mode'] == 'gluomem-shared-process'
        assert actual['configured_gluomem_history_capacity'] == 3
        assert actual['native_300_parallelism_certified'] is False
        assert (tmp_path/'executed'/'completion-receipt.json').is_file()
        assert x.verify_root(operational,plan=execution_plan,qualification=qualification)==actual
        forged = copy.deepcopy(actual)
        forged['shared_gluomem_resources']['memory_samples']['private_mib'] = 1
        runtime_tamper = tmp_path/'runtime-tamper.json'
        runtime_tamper.write_bytes(x.json_bytes(forged))
        with pytest.raises(ValueError,match='high-water'):
            x.verify_root(operational,plan=execution_plan,qualification=qualification,candidate_path=runtime_tamper)
        closure_tamper = copy.deepcopy(packet)
        closure_tamper['source_closure'][0]['sha256'] = '0'*64
        closure_path = tmp_path/'source-closure-tamper.json'
        closure_path.write_bytes(x.json_bytes(closure_tamper))
        with pytest.raises(ValueError,match='binding|closure'):
            x.validate_shared_qualification(x.bound(closure_path),commit_limit_mib=4096,host_reserve_mib=8192)
        packet['memory_samples']['private_mib'] = 1
        tampered = tmp_path/'shared-qualification-tamper.json'
        tampered.write_bytes(x.json_bytes(packet))
        with pytest.raises(ValueError,match='high-water'):
            x.validate_shared_qualification(x.bound(tampered),commit_limit_mib=4096,host_reserve_mib=8192)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize('identity',['authored-lme-native-v1','authored-clock-whitespace-empty-01'])
def test_qualification_preserves_retained_authored_identity(tmp_path,authored_plan,monkeypatch,identity):
    from pathlib import Path
    source = tmp_path/'authored.json'
    raw = json.loads(Path(authored_plan['dataset']['path']).read_bytes())
    raw[0]['question_id'] = identity
    source.write_bytes(x.json_bytes(raw))
    plan = tmp_path/'approved-plan.json'
    plan.write_bytes(x.json_bytes(authored_plan))
    decision = tmp_path/'base-decision.json'
    decision.write_bytes(x.json_bytes({'approver_kind':'parent','plan':x.bound(plan),
        'bounded_authored_real_backend_qualification_permitted':{
            'generation_admission_cap':192,'wall_seconds_cap':1200,'retries':0}}))
    def reached(out,qplan,budget):
        assert qplan['sample_ids'] == [identity], 'do not normalize the original source identifier'
        raise RuntimeError('accepted source; no client construction')
    monkeypatch.setattr(x,'build_input_cache',reached)
    with pytest.raises(RuntimeError,match='accepted source'):
        x.run_authored_qualification(plan_path=plan,plan_sha256=x.bound(plan)['sha256'],
            budget_decision_path=decision,budget_decision_sha256=x.bound(decision)['sha256'],
            source_path=source,out=tmp_path/'qualification',cognee_python='unused',
            host_reserve_mib=8192,history_commit_limit_mib=4096)


def test_resident_group_uses_one_reservation_and_latches_pressure():
    host = {'available_physical_mib':15000,'commit_headroom_mib':15000}
    gate = x.MemoryAdmission(host_reserve_mib=8192,host_status=lambda:dict(host),
        commit_limits={'gluomem-group':4096})
    gate.observe('gluomem-group',rss_mib=300,private_mib=1000)
    assert gate.acquire('owned-group','gluomem-group')
    assert not gate.acquire('second-group','gluomem-group')
    assert gate.admit_existing('owned-group')
    assert len(gate.live) == 1
    host['commit_headroom_mib'] = 8191
    assert not gate.admit_existing('owned-group')
    host['commit_headroom_mib'] = 20000
    assert not gate.admit_existing('owned-group')


def test_one_os_group_cap_bounds_aggregate_threads_not_coordinator(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    package = str(Path(x.__file__).resolve().parents[2])
    code = (f'import sys;sys.path.insert(0,{package!r})\n'
        'import json,threading\n'
        'from concurrent.futures import ThreadPoolExecutor\n'
        'from gluomem_bench.bench.lme_first_execute import ProcessCommitLimit\n'
        'cap=ProcessCommitLimit(192)\n'
        'barrier=threading.Barrier(3,timeout=10)\n'
        'def allocate(i):\n'
        ' buffer=None\n'
        ' try:\n'
        '  buffer=bytearray(80*1048576)\n'
        ' except MemoryError:\n'
        '  pass\n'
        ' barrier.wait()\n'
        ' return buffer is not None\n'
        'with ThreadPoolExecutor(3) as pool:\n'
        ' result=list(pool.map(allocate,range(3)))\n'
        'print(json.dumps(result))\n'
        'cap.close()\n')
    before = x.process_creation_time(os.getpid())
    result = subprocess.run([sys.executable,'-I','-c',code],capture_output=True,timeout=30)
    assert result.returncode == 0, result.stderr
    allocated = json.loads(result.stdout)
    assert len(allocated)==3 and any(allocated) and not all(allocated)
    assert x.process_creation_time(os.getpid()) == before
    # A fresh unrelated native worker still acquires/account-closes normally.
    with x.ContextBroker() as broker:
        rows = [dict(job_id='healthy',history_id='healthy',system='mem0',output=str(tmp_path/'leaf'),
            memory_mib=128,call_budget=1,seconds_budget=15)]
        report = run_isolated_jobs(rows,out=tmp_path/'coord',
            run_job=lambda job,ctx:x.launch_worker({'probe':True,'commit_limit_mib':128},
                broker=broker,context=ctx,workdir=tmp_path/'healthy-worker'),
            max_calls=1,max_seconds=30,max_workers=1,native_limits={'mem0':1},
            memory_limit_mib=128,max_in_flight_tokens=39321600)
    assert report['jobs'][0]['status'] == 'complete'


def test_process_worker_pressure_is_persisted_and_latched_before_generation(tmp_path,monkeypatch):
    monkeypatch.setattr(x,'host_memory_status',lambda:{'available_physical_mib':8191,
        'commit_headroom_mib':100000,'units':'MiB (1048576 bytes)',
        'source':'GlobalMemoryStatusEx + GetPerformanceInfo'})
    with x.ContextBroker() as broker:
        rows = [dict(job_id='pressured',history_id='p',system='mem0',output=str(tmp_path/'leaf'),
            memory_mib=128,call_budget=1,seconds_budget=15)]
        report = run_isolated_jobs(rows,out=tmp_path/'coord',
            run_job=lambda job,ctx:x.launch_worker({'probe':True,'commit_limit_mib':128},
                broker=broker,context=ctx,workdir=tmp_path/'worker',host_reserve_mib=8192),
            max_calls=1,max_seconds=30,max_workers=1,native_limits={'mem0':1},
            memory_limit_mib=128,max_in_flight_tokens=39321600)
    samples = [json.loads(line) for line in (tmp_path/'worker'/'resource-samples.jsonl').read_bytes().splitlines()]
    assert samples[0]['host_status']['available_physical_mib'] == 8191
    assert report['jobs'][0]['status'] == 'failed'
    assert report['accounting']['calls'] == 0
    assert json.loads((tmp_path/'worker'/'process-exit.json').read_bytes())['measurement_failed'] is True


def test_shared_group_refuses_non_gluomem_before_dispatch(tmp_path):
    with x.SharedGluomemWorker(workdir=tmp_path/'group',commit_limit_mib=256,max_workers=3) as group:
        with pytest.raises(ValueError,match='GluoMem'):
            group.run({'system':'mem0'},broker=None,context=None,workdir=tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()
