"""Scoped real DSPy history boundary; authored transports, denied real sockets."""
import json
import os
import socket
from concurrent.futures import ThreadPoolExecutor

from threading import Barrier
from types import SimpleNamespace

import dspy
import httpx
import pytest
from dspy.clients.base_lm import record_history
from litellm import ModelResponse

from gluomem_bench.bench.rack import RackClient, RackProfile, controlled_config
from gluomem_bench.bench.rack_trace import RackLM, RackRouter, WorkloadBudget


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
    monkeypatch.setenv('GLUOMEM_RACK_API_KEY', 'local-endpoint-placeholder')
    def deny(*args, **kwargs):
        raise AssertionError('real network forbidden')
    # Windows asyncio creates a socketpair via one ephemeral loopback listener.
    # Register only binds made within that exact socketpair call, not localhost generally.
    from threading import local
    state = local()
    original_bind, original_pair = socket.socket.bind, socket.socketpair
    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex
    def bind(sock, address):
        result = original_bind(sock, address)
        if getattr(state, 'pairs', None) is not None:
            state.pairs.add(sock.getsockname())
        return result
    def pair(*args, **kwargs):
        prior = getattr(state, 'pairs', None)
        state.pairs = set()
        try:
            return original_pair(*args, **kwargs)
        finally:
            state.pairs = prior
    def connect(sock, address):
        if address in (getattr(state, 'pairs', None) or ()):
            return original_connect(sock, address)
        return deny()
    def connect_ex(sock, address):
        if address in (getattr(state, 'pairs', None) or ()):
            return original_connect_ex(sock, address)
        return deny()
    monkeypatch.setattr(socket.socket, 'bind', bind)
    monkeypatch.setattr(socket, 'socketpair', pair)
    monkeypatch.setattr(socket.socket, 'connect', connect)
    monkeypatch.setattr(socket.socket, 'connect_ex', connect_ex)
    monkeypatch.setattr(socket, 'create_connection', deny)


def make_lm(tmp_path, *, trace=None):
    profile = RackProfile(base_url='http://authored.invalid/v1', model='authored/qwen')
    client = RackClient(profile, receipt=tmp_path/'calls.jsonl',
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'count':10})))
    budget = WorkloadBudget()
    lm = RackLM('openai/'+profile.model, client=client, task='event_claims',
        trace=trace or tmp_path/'raw.jsonl', budget=budget, api_base=profile.base_url,
        max_tokens=512, temperature=0, num_retries=0, cache=False)
    return lm, client, budget


def response(text='private response'):
    return ModelResponse(model='authored/qwen', choices=[{'index':0,'finish_reason':'stop',
        'message':{'role':'assistant','content':text}}],
        usage={'prompt_tokens':10,'completion_tokens':7,'total_tokens':17})


def test_instance_history_only_preserves_native_usage_and_nonrack_global_behavior(tmp_path):
    lm, client, _ = make_lm(tmp_path)
    global_history = record_history.__globals__['GLOBAL_HISTORY']
    before = list(global_history)
    caller = SimpleNamespace(history=[])
    global_lm = dspy.settings.lm
    try:
        from dspy.utils.usage_tracker import track_usage
        with track_usage() as usage, dspy.context(caller_modules=[caller]):
            raw = response()
            assert lm._process_lm_response(raw, None, [{'role':'user','content':'private prompt'}]) == ['private response']
            assert lm.history[-1]['response'] is raw
            assert {k:usage.get_total_tokens()[lm.model][k] for k in ('prompt_tokens','completion_tokens','total_tokens')} == {'prompt_tokens':10,'completion_tokens':7,'total_tokens':17}
        assert global_history == before
        assert caller.history == []
        assert dspy.settings.lm is global_lm
        ordinary = dspy.LM('openai/authored/qwen', cache=False)
        with dspy.context(caller_modules=[caller]):
            ordinary._process_lm_response(response('ordinary'), None, [{'role':'user','content':'ordinary'}])
        assert global_history[-1] is ordinary.history[-1] is caller.history[-1]
    finally:
        global_history[:] = before
        client.close()


def test_three_production_ingests_retrieve_scoped_facts_clocks_and_close_handles(tmp_path, monkeypatch):
    from gluomem.embedding.providers import MockEmbedder
    from gluomem.frames.store import FrameStore
    from gluomem.kg.graph import KnowledgeGraph
    from gluomem.llm.modules import DspyExtractors
    from gluomem.search.hybrid import HybridSearch
    from gluomem.storage.db import MindDB
    from gluomem.storage.scope import MemoryScope
    from gluomem.write.events import SourceEvent, SourceEventMessage, ingest_source_event

    facts = [f'AUTHORED_TENANT_{i} owns cobalt telescope_{i}.' for i in range(3)]
    clocks = [f'2024-0{i+1}-01T12:00:00+00:00' for i in range(3)]
    barrier = Barrier(3, timeout=30)
    global_history = record_history.__globals__['GLOBAL_HISTORY']
    before = list(global_history)
    global_lm = dspy.settings.lm
    env = dict(os.environ)
    def send(client, request, **kwargs):
        assert request.url.host == 'authored.invalid'
        if request.url.path == '/tokenize':
            return httpx.Response(200, json={'count':10}, request=request)
        assert request.url.path == '/v1/chat/completions'
        payload = json.loads(request.content)
        visible = json.dumps(payload['messages'])
        selected = [i for i, fact in enumerate(facts) if fact in visible]
        assert len(selected) == 1
        i = selected[0]
        assert clocks[i][:10] in visible
        assert all(clocks[j][:10] not in visible for j in range(3) if j != i)
        barrier.wait()
        content = json.dumps({'claims':[{'text':facts[i], 'kind':'fact', 'confidence':1,
            'verbatim_span':facts[i], 'message_id':'s0t0'}]})
        return httpx.Response(200, json={'id':f'authored-{i}', 'object':'chat.completion',
            'created':0, **response(content).model_dump()}, request=request)
    monkeypatch.setattr(httpx.Client, 'send', send)
    profile = RackProfile(base_url='http://authored.invalid/v1', model='authored/qwen')
    cfg = controlled_config(profile, data=tmp_path/'unused.json', sha256='0'*64)
    cfg = cfg.with_overrides({'dspy.tasks.event_claims.module':'predict',
                              'search.rerank.enabled':False})
    clients, dbs, callers, lms, modules = [], [], [], [], []
    native_lm_for = RackRouter.lm_for
    def observed_lm_for(router, task):
        lm = native_lm_for(router, task)
        lms.append(lm)
        return lm
    monkeypatch.setattr(RackRouter, 'lm_for', observed_lm_for)
    def ingest(i):
        scope = MemoryScope(tenant_id=f'tenant_{i}', principal_id=f'person_{i}', namespace=f'job_{i}')
        db = MindDB(tmp_path/f'store-{i}.mind', dim=384)
        dbs.append(db)
        embedder = MockEmbedder(dim=384)
        store = FrameStore(db, embedder=embedder, chunking=cfg.chunking, scope=scope)
        kg = KnowledgeGraph(db, scope=scope)
        budget = WorkloadBudget(max_calls=2, max_seconds=60)
        client = RackClient(profile, receipt=tmp_path/f'calls-{i}.jsonl', budget=budget)
        clients.append(client)
        caller = SimpleNamespace(history=[])
        callers.append(caller)
        try:
            router = RackRouter(cfg, client=client, trace=tmp_path/f'raw-{i}.jsonl', budget=budget)
            extractors = DspyExtractors(cfg, router=router)
            event = SourceEvent(event_id=f'event_{i}', source_ref=f'authored/{i}',
                source_type='chat_turn', event_time=clocks[i], scope=scope,
                messages=[SourceEventMessage(message_id='s0t0', role='user',
                           content=facts[i], timestamp=clocks[i])])
            with dspy.context(caller_modules=[caller]):
                committed = ingest_source_event(event, store=store, kg=kg, cfg=cfg, extractors=extractors)
            assert committed.event_row_id > 0
            modules.extend(extractors._modules.values())
            assert extractors.usage.snapshot() == {
                'openai/authored/qwen': {'prompt_tokens':10, 'completion_tokens':7}}
            assert budget.calls == 1
            rows = [dict(r) for r in db.fetchall('SELECT * FROM source_events')]
            assert len(rows) == 1
            assert rows[0]['tenant_id'] == scope.tenant_id
            assert clocks[i] in rows[0]['payload_json']
            provenance = [dict(r) for r in db.fetchall('SELECT * FROM memory_provenance')]
            assert provenance and all(r['tenant_id'] == scope.tenant_id for r in provenance)
            search = HybridSearch(db, store, cfg, embedder=embedder, kg=kg)
            hits = search.search(facts[i], scope=scope).hits
            assert hits and any(facts[i] in h.frame.content for h in hits)
            assert all(facts[j] not in str(hits) for j in range(3) if j != i)
            for j in range(3):
                if j != i:
                    assert not search.search(facts[j], scope=MemoryScope(
                        tenant_id=f'tenant_{j}', principal_id=f'person_{j}', namespace=f'job_{j}')).hits
            return i
        finally:
            client.close()
            db.close()
    try:
        with ThreadPoolExecutor(3) as pool:
            assert sorted(pool.map(ingest, range(3))) == [0,1,2]
        assert global_history == before
        assert all(c.history == [] for c in callers)
        assert len(lms) == len(modules) == 3
        assert all(module.history == [] for module in modules)
        for lm in lms:
            assert len(lm.history) == 1
            visible = json.dumps(lm.history, default=str)
            assert sum(fact in visible for fact in facts) == 1
        assert dspy.settings.lm is global_lm
        assert dict(os.environ) == env
        assert all(c._client.is_closed for c in clients)
        for i in range(3):
            trace = (tmp_path/f'raw-{i}.jsonl').read_text()
            assert facts[i] in trace and clocks[i][:10] in trace
            assert all(facts[j] not in trace for j in range(3) if j != i)
            # Windows rename after close exercises that owned SQLite handles are released.
            path = tmp_path/f'store-{i}.mind'
            path.rename(path.with_suffix('.closed'))
    finally:
        global_history[:] = before


def test_repeated_calls_keep_one_raw_response_and_validate_fresh_identity(tmp_path, monkeypatch):
    lm, client, budget = make_lm(tmp_path)
    def provider(self, *args, **kwargs):
        return self._process_lm_response(response(str(budget.calls)), None, kwargs['messages'])
    monkeypatch.setattr(dspy.LM, '__call__', provider)
    try:
        for i in range(3):
            assert lm(messages=[{'role':'user','content':'own source'}]) == [str(i+1)]
            assert len(lm.history) == 1
        assert budget.calls == 3
        assert len((tmp_path/'raw.jsonl').read_text().splitlines()) == 3
    finally:
        client.close()


@pytest.mark.parametrize('fault', ['normal', 'hostile', 'trace_write', 'diagnostic'])
@pytest.mark.parametrize('shared', [False, True])
def test_failure_diagnostics_never_format_reason_preserve_original_spend_and_logs(
        tmp_path, monkeypatch, caplog, fault, shared):
    from gluomem_bench.bench import embedding_runtime
    from dspy.utils.exceptions import LMTransportError
    from gluomem_bench.bench.parallel_campaign import CampaignBudget, Job, JobContext, RequestLimiter
    secret = 'PRIVATE_FAILURE_REASON_TENANT_42'
    class HostileError(Exception):
        def __str__(self):
            raise AssertionError('untrusted __str__ invoked')
    error = (LMTransportError(secret) if shared else
             HostileError() if fault == 'hostile' else ValueError(secret))
    lm, client, budget = make_lm(tmp_path)
    campaign = CampaignBudget(max_calls=20, max_seconds=120)
    # An actual outer budget demonstrates exactly-once debit and failure scope.
    budget.outer_budget = JobContext(Job('authored', 'tenant', tmp_path/'job', 20, 120),
        campaign, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))
    client.budget = budget
    client.profile = client.profile.model_copy(update={'profile_version':'role-budgets-v6'})
    def provider(*args, **kwargs):
        raise error
    monkeypatch.setattr(dspy.LM, '__call__', provider)
    if fault == 'trace_write':
        path = tmp_path/'not-directory'
        path.write_text('occupied')
        lm.trace = path/'trace.jsonl'
    if fault == 'diagnostic':
        def diagnostic_failure(*args):
            raise OSError(secret)
        monkeypatch.setattr(embedding_runtime, 'safe_failure_diagnostic', diagnostic_failure)
    try:
        with caplog.at_level('DEBUG'):
            with pytest.raises(Exception) as caught:
                lm(messages=[{'role':'user','content':'allowed task source'}])
        assert caught.value is error
        assert budget.calls == campaign.calls == 1
        assert budget.failed
        assert bool(campaign.failed) is shared
        assert secret not in client._failure
        assert secret not in caplog.text
        assert secret not in str(budget.outer_budget.failure)
        if fault != 'trace_write':
            row = json.loads(lm.trace.read_text())
            assert row['ordinal'] == 1
            assert row['error']['message'] == 'operation_failed'
            assert secret not in lm.trace.read_text()
        before = lm.trace.read_bytes() if lm.trace.is_file() else None
        with pytest.raises(Exception):
            lm(messages=[{'role':'user','content':'must not dispatch'}])
        assert budget.calls == campaign.calls == 1
        assert (lm.trace.read_bytes() if lm.trace.is_file() else None) == before
    finally:
        client.close()


def test_zero_module_history_limit_still_retains_scoped_raw_response(tmp_path):
    lm, client, _ = make_lm(tmp_path)
    try:
        with dspy.context(max_history_size=0):
            lm._process_lm_response(response(), None, [{'role':'user','content':'own source'}])
        assert len(lm.history) == 1
    finally:
        client.close()


def test_stale_instance_history_cannot_certify_missing_response(tmp_path, monkeypatch):
    lm, client, budget = make_lm(tmp_path)
    lm.update_history({'response':response()})
    monkeypatch.setattr(dspy.LM, '__call__', lambda *a, **k: ['unreceipted'])
    try:
        with pytest.raises(ValueError, match='missing raw history'):
            lm(messages=[{'role':'user','content':'own source'}])
        assert budget.calls == 1 and budget.failed
    finally:
        client.close()


def test_three_failed_jobs_keep_coordinator_and_root_child_thread_logs_private(tmp_path, monkeypatch):
    import logging
    from gluomem_bench.bench.parallel_campaign import Job, CampaignPolicy, run_jobs
    secrets = [f'PRIVATE_THREAD_REASON_{i}' for i in range(3)]
    barrier = Barrier(3, timeout=20)
    root = logging.getLogger()
    child = logging.getLogger('rack_boundary.authored_nonpropagating')
    path = tmp_path/'thread-logs.txt'
    handler = logging.FileHandler(path, encoding='utf-8')
    root.addHandler(handler)
    child.addHandler(handler)
    prior_propagate = child.propagate
    child.propagate = False
    jobs = [Job(str(i), f'tenant_{i}', tmp_path/f'job-{i}', 2, 60) for i in range(3)]
    contexts = []
    def run(job, context):
        contexts.append(context)
        lm, client, _ = make_lm(job.output)
        lm.budget = WorkloadBudget(outer_budget=context)
        client.budget = lm.budget

        def dispatch(*args, **kwargs):
            barrier.wait()
            raise ValueError(secrets[int(job.job_id)])
        # Each task owns its callable; no process-global LM binding is changed.
        lm.forward = dispatch
        try:
            lm(messages=[{'role':'user','content':'allowed task source'}])
        finally:
            client.close()
    # Bind the dispatcher through the owning LM rather than a thread-global marker.
    monkeypatch.setattr(dspy.LM, '__call__', lambda self, *a, **k: self.forward(*a, **k))
    try:
        result = run_jobs(jobs, policy=CampaignPolicy(3,3,39321600,6,120,False),
            out=tmp_path/'coordinator', run_job=run)
        assert [r['status'] for r in result['jobs']] == ['failed']*3
        assert result['accounting']['calls'] == 3
        assert all(c.failed and not c.budget.failed for c in contexts)
        handler.flush()
        material = json.dumps(result) + path.read_text()
        for p in (tmp_path/'coordinator').rglob('*.json*'):
            material += p.read_text()
        for secret in secrets:
            assert secret not in material
    finally:
        root.removeHandler(handler)
        child.removeHandler(handler)
        child.propagate = prior_propagate
        handler.close()
