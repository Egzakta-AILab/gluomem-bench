"""Fixed-authored finalnative qualification, never released benchmark QA.

Only explicit parent 192-call/1200-second/no-retry permit is accepted. Native
algorithms are invoked via lme_first_execute.run_native_job in fresh processes.
"""
from __future__ import annotations
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def validate_permit(permit):
    if (permit.get('generation_admission_cap') != 192 or permit.get('wall_seconds_cap') != 1200
        or permit.get('retries') != 0 or permit.get('same_approved_endpoints_only') is not True
        or permit.get('all_spend_retained') is not True):
        raise ValueError('exact parent authored qualification permit required')
    return 192,1200


def save(path,value):
    Path(path).write_bytes((json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())


def bound(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


class EncoderProxy:
    """Exact-route local native adapter proxy; no provider/download fallback."""
    def __init__(self,upstream,*,receipt,timeout,transport=None):
        import httpx
        self.upstream=upstream.rstrip('/')
        self.receipt=Path(receipt)
        self.timeout=timeout
        self.lock=threading.Lock()
        self.rows=[]
        self._failure=None
        self.client=httpx.Client(transport=transport,trust_env=False,follow_redirects=False)

    def forward(self,payload,*,replay=False):
        with self.lock:
            failure=self._failure
        if failure is not None:
            raise RuntimeError('encoder proxy failure latched; no retry dispatch') from failure
        row={'request':payload,'replay':replay,'start':time.monotonic()}
        try:
            response=self.client.post(self.upstream+'/embeddings',json=payload,timeout=min(120,self.timeout()))
            row['http_status']=response.status_code
            if not response.is_success:
                row['failure']={'category':'encoder_http_failure','body_prefix_sha256':hashlib.sha256(response.content[:4096]).hexdigest(),
                                'captured_bytes':min(4096,len(response.content)),'truncated':len(response.content)>4096}
                raise RuntimeError('approved encoder request failed')
            result=response.json()
            row['response']=result
            return result
        except BaseException as exc:
            with self.lock:
                if self._failure is None:self._failure=exc
            row.setdefault('failure',{'category':'encoder_transport_failure'})
            raise
        finally:
            row['end']=time.monotonic()
            with self.lock:
                self.rows.append(row)
                with self.receipt.open('ab') as stream:
                    stream.write((json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n').encode())
                    stream.flush()

    def __enter__(self):
        proxy=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                status=200
                try:
                    if self.path!='/v1/embeddings': raise ValueError('unsupported encoder route')
                    size=int(self.headers.get('Content-Length','0'))
                    if not 0<size<=4_000_000: raise ValueError('encoder request size rejected')
                    body=json.loads(self.rfile.read(size))
                    result=proxy.forward(body)
                except BaseException:
                    status,result=502,{'error':{'category':'encoder_guard_failure'}}
                raw=json.dumps(result).encode()
                self.send_response(status);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.base_url=f'http://127.0.0.1:{self.server.server_port}/v1'
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        return self

    def replay(self):
        originals=list(self.rows)
        count=0
        for row in originals:
            if row.get('replay') or 'response' not in row: continue
            actual=self.forward(row['request'],replay=True)
            if actual['data']!=row['response']['data']:
                raise ValueError('exact same-role/text/batch vector replay differs')
            count+=1
        return {'exact_same_shape_matches':count,'tolerance':0,'numeric_space':'actual encoder JSON vectors'}

    def __exit__(self,*args):
        self.server.shutdown();self.server.server_close();self.thread.join();self.client.close()


class SocketBoundary:
    """Process-local exact endpoint allowlist, plus newly bound loopback listeners."""
    def __init__(self,endpoints):
        self.allowed=set()
        self.counts={'approved_connections':0,'denied_connections':0}
        for host,port in endpoints:
            self.allowed.add((host,port))
            for address in socket.getaddrinfo(host,port,type=socket.SOCK_STREAM):
                self.allowed.add((address[4][0],address[4][1]))

    def install(self):
        connect=socket.socket.connect
        listen=socket.socket.listen
        def guarded_connect(sock,address):
            permitted=isinstance(address,tuple) and (address[0],address[1]) in self.allowed
            self.counts['approved_connections' if permitted else 'denied_connections']+=1
            if not permitted: raise OSError('qualification outbound denied')
            return connect(sock,address)
        def guarded_listen(sock,*args):
            result=listen(sock,*args);address=sock.getsockname()
            if isinstance(address,tuple) and address[0] in ('127.0.0.1','::1'):
                self.allowed.add((address[0],address[1]))
            return result
        def guarded_connect_ex(sock,address):
            try: guarded_connect(sock,address);return 0
            except OSError:return 10013
        socket.socket.connect=guarded_connect;socket.socket.connect_ex=guarded_connect_ex;socket.socket.listen=guarded_listen


def worker(request_path):
    from types import SimpleNamespace
    from gluomem_bench.bench.lme_first_execute import RemoteContext,effective_config,run_native_job
    from gluomem_bench.bench.parallel_campaign import Job
    from gluomem_bench.bench.embedding_runtime import RemoteEmbedder,EmbeddingRecipe,safe_failure_diagnostic
    from gluomem_bench.bench.lme_projection import ExactLongMemEvalAdapter
    path=Path(request_path);request=json.loads(path.read_bytes());plan=request['plan']
    context=RemoteContext(request['address'],bytes.fromhex(os.environ.pop('LME_RPC_KEY')),request['handle'],
                          Job(**request['job']),max_calls=192,max_seconds=1200)
    endpoints=[tuple(request['address'])]
    for url in (plan['profile']['base_url'],request['encoder_proxy']):
        parsed=urlsplit(url);endpoints.append((parsed.hostname,parsed.port or (443 if parsed.scheme=='https' else 80)))
    network=SocketBoundary(endpoints);network.install()
    outcome={'status':'failed','calls':None};code=1
    try:
        for item in request['source_files']:
            if bound(item['path'])!=item:raise ValueError('qualified source bytes changed')
        cfg,profile=effective_config(plan)
        adapter=ExactLongMemEvalAdapter(cfg,'longmemeval_s');adapter.verify()
        samples=adapter.load()
        if len(samples)!=1 or samples[0].id!='authored-clock-whitespace-empty-01':
            raise ValueError('fixed authored qualification input required')
        sample=samples[0]
        embedder=RemoteEmbedder(request['encoder_proxy'],recipe=EmbeddingRecipe(**plan['embedding_recipe']),timeout=min(120,context.remaining()))
        job=SimpleNamespace(**request['job'],system=request['system']);job.output=Path(job.output)
        settings=None
        if job.system=='cognee':
            from gluomem_bench.bench.comparison_backends import CogneeSettings
            settings=CogneeSettings(python_executable=request['cognee_python'],root=job.output/'graph/native',dataset='authored-finalnative',
                llm_url=profile.base_url,llm_model=profile.model,llm_api_key='local-endpoint-placeholder',
                embedding_url=request['encoder_proxy'],embedding_model=plan['embedding_recipe']['model'],
                embedding_dimensions=plan['embedding_recipe']['dim'],embedding_api_key='local-endpoint-placeholder',
                max_completion_tokens=profile.output_budget('cognify'),timeout_seconds=min(profile.timeout_seconds,context.remaining()))
        result=run_native_job(job=job,context=context,cfg=cfg,adapter=adapter,sample=sample,profile=profile,
                              judge_source=Path(plan['judge_source']),embedder=embedder,cognee_settings=settings,require_real=True)
        outcome={'status':'native-closed-verified','calls':context.calls,'result':result};code=0
    except BaseException as exc:
        outcome['failure']=safe_failure_diagnostic(exc)
        try:outcome['calls']=context.calls
        except BaseException:pass
    finally:
        outcome['network']=network.counts
        try:context.close()
        finally:save(path.parent/'outcome.json',outcome)
    return code


def run(*,plan_path,decision_path,fixture_plan_path,out,cognee_python):
    from gluomem_bench.bench.parallel_campaign import CampaignBudget
    from gluomem_bench.bench.lme_first import run_isolated_jobs,verify_journal
    from gluomem_bench.bench.lme_first_execute import ContextBroker,process_tree_memory
    from gluomem_bench.bench.rack import RackClient,RackProfile
    import httpx
    budget=CampaignBudget(max_calls=192,max_seconds=1200)
    coordinator_before=process_tree_memory(os.getpid())
    decision=json.loads(Path(decision_path).read_bytes())
    validate_permit(decision['bounded_authored_real_backend_qualification_permitted'])
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('fresh authored qualification run required; no automatic rerun')
    out.mkdir(parents=True)
    original_plan=json.loads(Path(plan_path).read_bytes())
    plan=json.loads(json.dumps(original_plan))
    fixture_plan=json.loads(Path(fixture_plan_path).read_bytes())
    raw=dict(fixture_plan['synthetic_source_fixture'])
    eval_only=fixture_plan['evaluator_only_fixture']
    raw.update(question_date=eval_only['question_date'],question=eval_only['question'],answer=eval_only['gold'],
               question_type=eval_only['category'],answer_session_ids=[])
    source=out/'authored-input.json';save(source,[raw])
    digest=bound(source)['sha256']
    plan['dataset']={'name':'longmemeval_s','path':str(source),'sha256':digest}
    plan['config_overrides']['benchmark.datasets_dir']=str(out)
    plan['config_overrides']['benchmark.datasets.longmemeval_s']={'url':'https://authored.invalid/fixed.json','file':source.name,'sha256':digest}
    save(out/'effective-authored-plan.json',plan)
    source_files=[bound(p) for p in Path(__file__).resolve().parent.glob('*.py')]
    save(out/'source-files.json',source_files)
    profile=RackProfile.model_validate(plan['profile'])
    preflight={}
    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge
    client=CogneeGenerationBridge(profile,budget=budget,receipt=out/'preflight-qwen.jsonl')
    try:
        preflight['model_discovery']=client._request('GET','/models').json()
        if profile.model not in {r['id'] for r in preflight['model_discovery']['data']}:
            raise ValueError('approved Qwen served model unavailable')
        messages=[{'role':'user','content':'Fixed authored finalnative qualification.'}]
        counts=[client._request('POST','/tokenize',{'model':profile.model,'messages':messages,
                'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}}).json()['count'] for _ in range(2)]
        if counts[0]!=counts[1]:raise ValueError('served tokenizer nondeterministic')
        preflight['tokenizer_counts']=counts
        preflight['chat_template_kwargs']={'enable_thinking':False}
    finally:
        client.__exit__()
    with httpx.Client(trust_env=False,follow_redirects=False) as client:
        response=client.get(plan['embedding_url'].rstrip('/')+'/models',timeout=min(120,budget.deadline-budget.clock()))
        response.raise_for_status();preflight['encoder_discovery']=response.json()
        if plan['embedding_recipe']['model'] not in {r['id'] for r in response.json()['data']}:
            raise ValueError('approved encoder model unavailable')
    save(out/'preflight.json',preflight)
    rows=[dict(job_id='authored-'+s,history_id='authored:'+s,system=s,output=str(out/'jobs'/s),
               memory_mib=1,call_budget=64,seconds_budget=350) for s in ('gluomem','mem0','cognee')]
    samples={};outcomes={}
    with EncoderProxy(plan['embedding_url'],receipt=out/'encoder-wire.jsonl',timeout=lambda:budget.deadline-budget.clock()) as encoder:
        with ContextBroker() as broker:
            def execute(job,context):
                system=next(r['system'] for r in rows if r['job_id']==job.job_id)
                work=out/'workers'/system;work.mkdir(parents=True)
                request={'plan':plan,'encoder_proxy':encoder.base_url,'address':broker.address,'handle':broker.register(context),
                         'system':system,'cognee_python':str(cognee_python),'source_files':source_files,
                         'job':{'job_id':job.job_id,'history_id':job.history_id,'output':str(job.output),
                                'call_budget':job.call_budget,'seconds_budget':job.seconds_budget}}
                request_path=work/'request.json';save(request_path,request)
                env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('API_KEY','TOKEN','SECRET'))}
                env.update(MEM0_TELEMETRY='False',MEM0_DIR=str(work/'mem0-home'),GLUOMEM_RACK_API_KEY='local-endpoint-placeholder',
                           LME_RPC_KEY=broker.key.hex(),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LITELLM_LOCAL_MODEL_COST_MAP='true',
                           PYTHONDONTWRITEBYTECODE='1')
                process=subprocess.Popen([sys.executable,'-I','-c',
                    'import sys;sys.path.insert(0,sys.argv[1]);from gluomem_bench.bench.native_backend_qualification import worker;sys.exit(worker(sys.argv[2]))',
                    str(Path(__file__).resolve().parents[2]),str(request_path)],cwd=Path(__file__).resolve().parents[2],env=env,
                    stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                measured=[]
                while process.poll() is None:
                    try:measured.append(dict(time=budget.clock(),**process_tree_memory(process.pid)))
                    except OSError:pass
                    time.sleep(.1)
                save(work/'memory-samples.json',measured)
                if measured:
                    samples[system]={'rss_mib':max(r['rss_mib'] for r in measured),'private_mib':max(r['private_mib'] for r in measured),
                                     'first':measured[0],'last_alive':measured[-1],'scope':'tiny authored whole-worker process tree; not full-history maximum'}
                outcome=json.loads((work/'outcome.json').read_bytes());outcomes[system]=outcome
                if process.returncode!=0 or outcome['status']!='native-closed-verified':
                    raise RuntimeError('actual isolated native qualification failed')
                if outcome['calls']!=context.calls:raise ValueError('shared qualification accounting differs')
                return outcome
            coordination=run_isolated_jobs(rows,out=out/'coordination',run_job=execute,max_calls=192,max_seconds=1200,
                max_workers=1,native_limits={'gluomem':1,'mem0':1,'cognee':1},memory_limit_mib=3,
                max_in_flight_tokens=39321600,budget=budget)
        replay=encoder.replay()
    verified=verify_journal(out/'coordination',rows)
    if verified!=coordination:raise ValueError('actual qualification journal replay differs')
    complete=all(r['status']=='complete' for r in coordination['jobs'])
    report={'protocol':'lme-first-authored-finalnative-qualification@v2','qualified':complete,'systems':['gluomem','mem0','cognee'],
            'candidate_code_sha256':bound(Path(__file__).with_name('lme_first_execute.py'))['sha256'],
            'accounting':coordination['accounting'],'jobs':coordination['jobs'],'limiter':coordination['limiter'],
            'outcomes':outcomes,'memory_samples':samples,'encoder_parity':replay,'full_history_ram_maximum':None,
            'coordinator_memory_before':coordinator_before,'coordinator_memory_after':process_tree_memory(os.getpid()),
            'released_qa_dispatched':False,'retries':0,'mem0_observation_date':'Native execution-day UTC Observation Date; not historical question_date. No clock monkeypatch.',
            'assertions':{'native_factory_reader_scorer':complete,'all_generation_shared_accounting':True,'no_aux_sdk_bypass':complete,
              'privacy_categorical':complete,'source_clock_exact':complete,'mem0_execution_date_disclosed':True,'native_process_isolation':True}}
    report['evidence_files']=[bound(p) for p in out.rglob('*') if p.is_file()]
    save(out/'qualification.json',report)
    budget.check()
    return report


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--decision',type=Path,required=True)
    parser.add_argument('--fixture-plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cognee-python',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        result=run(plan_path=args.plan,decision_path=args.decision,fixture_plan_path=args.fixture_plan,out=args.out,cognee_python=args.cognee_python)
    except BaseException as exc:
        from gluomem_bench.bench.embedding_runtime import safe_failure_diagnostic
        evidence={'qualified':False,'failure':safe_failure_diagnostic(exc),'retry_authorized':False,
                  'accounting_source':str(args.out/'coordination/journal.jsonl')}
        if args.out.is_dir():save(args.out/'qualification-blocker.json',evidence)
        print(json.dumps(evidence))
        return 1
    print(json.dumps({'qualified':result['qualified'],'accounting':result['accounting'],'report':str(args.out/'qualification.json')}))
    return 0 if result['qualified'] else 1

if __name__=='__main__':raise SystemExit(main())
