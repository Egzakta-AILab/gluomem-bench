"""Authored V2 concrete factory and cache contracts; no benchmark QA."""
import importlib
import json

import pytest


def test_factory_admission_rejects_static_overbudget_before_native_import(tmp_path):
    v = importlib.import_module('gluomem_bench.bench.v2_driver')
    from gluomem_bench.bench.rack_trace import WorkloadBudget
    module = importlib.import_module('gluomem_bench.bench.v2_memory_factories')
    docs = [{'id': 'a', 'text': 'unaltered', 'sha256': v.file_hash(__file__)}]
    proof = {'complete': True, 'documents_sha256': v.digest(docs)}
    with pytest.raises(v.ResourceNotAdmitted, match='lower bound'):
        module.mem0_factory(cfg=None, group=None, documents=docs, source_proof=proof,
                            out=tmp_path, client=None,
                            budget=WorkloadBudget(max_calls=1, max_seconds=120),
                            embedder=None, qa_reserve=1)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('fault', ['none', 'closed_verifier'])
def test_native_mem0_v2_reader_finalization_and_failure_spend(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace as NS
    from tests.bench.test_mab_campaign import model_server
    from tests.bench.test_v2_driver import NATIVE_ROOT
    from gluomem_bench.bench import v2_driver as v, v2_memory_factories as m
    from gluomem_bench.bench.embedding_runtime import RemoteEmbedder
    from gluomem_bench.bench.rack import RackClient, RackProfile, controlled_config
    from gluomem_bench.bench.rack_trace import WorkloadBudget
    monkeypatch.setenv('MEM0_TELEMETRY', 'False')
    monkeypatch.setenv('MEM0_DIR', str(tmp_path.parent/'native-home'))
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    q = {'question': 'What is the locker code?', 'answer': 'cobalt',
         'eval_function': 'norm_phrase_set_match', 'domain': 'web'}
    data = NS(questions={'q': q}, reader_question=lambda qid: {**q, 'image': None},
        histories=lambda group: iter([{'states': [{'action': 'incoming', 'accessibility_tree': 'Station locker code is cobalt.'}]}]))
    group = v.HistoryGroup('authored', 'small', 'web', ('trajectory',), ('q',))
    if fault == 'closed_verifier':
        def fail(*args, **kwargs): raise ValueError('authored closed verification failure')
        monkeypatch.setattr(m, 'verify_closed_native', fail)
    with model_server() as (url, calls):
        encoder = RemoteEmbedder(url)
        profile = RackProfile(base_url=url, model='example/offline', max_input_chars=2000000)
        cfg = controlled_config(profile, data=tmp_path/'unused', sha256='0'*64, embedder=encoder)
        budget = WorkloadBudget(max_calls=12, max_seconds=120)
        client = RackClient(profile, receipt=tmp_path/'calls.jsonl', budget=budget)
        native = v.NativeInterface(NATIVE_ROOT); native.use_served_context_counter(client)
        result = v.run_group(data=data, group=group, method='mem0', native=native,
            client=client, budget=budget, out=tmp_path, cfg=cfg, embedder=encoder,
            document_factories={'mem0': m.mem0_factory}, authored=True)
        client.close()
    assert result['usage']['generation_calls'] == budget.calls == 2
    assert result['usage']['prompt_tokens'] == 200
    assert result['status'] == ('complete' if fault == 'none' else 'failed')
    assert (tmp_path/'source-documents.json').is_file()
    if fault != 'none':
        assert 'verification failure' in result['error']


def test_real_mem0_v2_factory_network_boundary_and_exact_readback(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from tests.bench.test_mab_campaign import model_server
    from gluomem_bench.bench.embedding_runtime import RemoteEmbedder
    from gluomem_bench.bench.rack import RackClient, RackProfile, controlled_config
    from gluomem_bench.bench.rack_trace import WorkloadBudget
    from gluomem_bench.bench import v2_driver as v, v2_memory_factories as m
    monkeypatch.setenv('MEM0_TELEMETRY', 'False')
    monkeypatch.setenv('MEM0_DIR', str(tmp_path.parent / 'native-home'))
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    with model_server() as (url, calls):
        encoder = RemoteEmbedder(url)
        profile = RackProfile(base_url=url, model='example/offline', max_generation_calls=12, max_input_chars=2000000)
        cfg = controlled_config(profile, data=tmp_path/'unused.json', sha256='0'*64, embedder=encoder)
        budget = WorkloadBudget(max_calls=12, max_seconds=120)
        client = RackClient(profile, receipt=tmp_path/'calls.jsonl', budget=budget)
        text = 'Station locker code is cobalt.\nOriginal source state incoming action.'
        import hashlib
        docs = [{'id': 'document-0', 'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'offset': 0}]
        proof = {'complete': True, 'documents_sha256': v.digest(docs)}
        memory = m.mem0_factory(cfg=cfg, group=NS(id='authored'), documents=docs,
            source_proof=proof, out=tmp_path, client=client, budget=budget, embedder=encoder, qa_reserve=1)
        try:
            observed = memory.verify_source_coverage(documents=docs, source_proof=proof)
            assert observed['ordered_document_hashes'] == [('document-0', docs[0]['sha256'])]
            items = memory.query('locker code?', query_image='reader-only.png')
            assert all(set(i) == {'type', 'value'} for i in items)
            contexts = json.loads((tmp_path/'native-mem0/query-contexts.json').read_bytes())
            assert [hit['memory'] for hit in contexts[0]['hits']] == [item['value'] for item in items]
            assert budget.calls == 1
            assert memory.memory.db.connection.execute('SELECT role, content FROM messages').fetchall() == [('user', text)]
        finally:
            memory.close()
            client.close()
        assert (tmp_path/'native-mem0/native-closed.json').is_file()
        m.verify_closed_native(tmp_path, method='mem0', documents=docs,
                               source_proof=proof, profile=profile)
        receipt = tmp_path/'native-mem0/mem0-input-receipts.json'
        saved = json.loads(receipt.read_bytes()); saved[0]['document_text'] = 'forged'
        receipt.write_bytes(json.dumps(saved).encode())
        with pytest.raises(ValueError, match='hash|binding|differs'):
            m.verify_closed_native(tmp_path, method='mem0', documents=docs,
                                   source_proof=proof, profile=profile)


def test_real_cognee_v2_factory_graph_evidence_network_boundary(tmp_path, bench_cfg):
    import hashlib
    import httpx
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path
    from types import SimpleNamespace as NS
    from gluomem_bench.bench import v2_driver as v, v2_memory_factories as m
    from gluomem_bench.bench.rack import RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import WorkloadBudget
    python = Path(__file__).resolve().parents[2]/'tests/fixtures/native/operator-supplied'
    assert python.is_file(), 'required isolated native Cognee runtime missing'
    calls = []
    graph = {'nodes': [
        {'id': 'rhea', 'name': 'Rhea', 'type': 'Person', 'description': 'Rhea owns a cobalt telescope.'},
        {'id': 'scope', 'name': 'cobalt telescope', 'type': 'Object', 'description': "Rhea's telescope."}],
        'edges': [{'source_node_id': 'rhea', 'target_node_id': 'scope', 'relationship_name': 'owns',
                   'description': 'Rhea owns a cobalt telescope.'}]}
    def upstream(request):
        body = json.loads(request.content) if request.content else {}
        if request.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [{'id': 'served/qwen', 'max_model_len': 131072}]})
        if request.url.path == '/tokenize':
            return httpx.Response(200, json={'count': 32})
        calls.append(body)
        schema = body.get('response_format', {}).get('json_schema', {}).get('schema', {})
        if body.get('response_format') == {'type': 'json_object'}:
            schema = json.loads(body['messages'][0]['content'].split('```json\n', 1)[1].split('```', 1)[0])
        props = schema.get('properties', {})
        if 'nodes' in props:
            result = graph
        elif 'summary' in props:
            result = {'summary': 'Rhea owns a cobalt telescope.', 'description': 'Authored source.'}
        elif not body.get('response_format') and 'following string: "test"' in body['messages'][0]['content']:
            result = 'test'
        else:
            raise AssertionError('unexpected native generation; no answer allowed')
        return httpx.Response(200, json={'id': 'authored', 'object': 'chat.completion', 'created': 1,
            'model': 'served/qwen', 'choices': [{'index': 0, 'finish_reason': 'stop',
            'message': {'role': 'assistant', 'content': json.dumps(result)}}],
            'usage': {'prompt_tokens': 32, 'completion_tokens': 16, 'total_tokens': 48}})
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            assert self.path == '/v1/embeddings'
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            texts = body['input'] if isinstance(body['input'], list) else [body['input']]
            raw = json.dumps({'object': 'list', 'model': 'encoder', 'data': [
                {'object': 'embedding', 'index': i, 'embedding': [0.8, 0.6, 0.0]}
                for i in range(len(texts))], 'usage': {'prompt_tokens': len(texts), 'total_tokens': len(texts)}}).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    budget = WorkloadBudget(max_calls=12, max_seconds=300)
    profile = RackProfile(base_url='http://authored.invalid/v1', model='served/qwen')
    client = RackClient(profile, receipt=tmp_path/'calls.jsonl', budget=budget)
    encoder = NS(base_url=f'http://127.0.0.1:{server.server_port}/v1', recipe=NS(model='text-embedding-3-small'), dim=3)
    text = 'Rhea owns a cobalt telescope.\n'
    docs = [{'id': 'doc-0', 'text': text, 'offset': 0, 'sha256': hashlib.sha256(text.encode()).hexdigest()}]
    proof = {'complete': True, 'documents_sha256': v.digest(docs)}
    memory = None
    try:
        memory = m.cognee_factory(cfg=bench_cfg, group=NS(id='v2-authored'), documents=docs,
            source_proof=proof, out=tmp_path, client=client, budget=budget, embedder=encoder,
            qa_reserve=1, python_executable=str(python), transport=httpx.MockTransport(upstream))
        assert memory.verify_source_coverage(documents=docs, source_proof=proof)['persistent_readback']
        evidence = memory.query('What does Rhea own?', query_image='reader-only.png')
        assert evidence and 'cobalt telescope' in json.dumps(evidence)
        assert all(set(i) == {'type', 'value'} for i in evidence)
        assert budget.calls == len(calls) > 0
        assert all('GOLD_ONLY' not in json.dumps(call) for call in calls)
    finally:
        if memory is not None: memory.close()
        client.close(); server.shutdown(); server.server_close(); thread.join()
    ledger = json.loads((tmp_path/'native-cognee/native-ledger.json').read_bytes())
    assert ledger['status'] == 'closed_verified'
    assert ledger['native_worker_closed'] and ledger['proxy_closed']
    assert ledger['generation_calls'] == budget.calls
    m.verify_closed_native(tmp_path, method='cognee', documents=docs,
                           source_proof=proof, profile=profile)


def test_cognee_factory_rejects_changed_source_before_native_dispatch(tmp_path):
    from gluomem_bench.bench import v2_memory_factories as m
    from gluomem_bench.bench.rack_trace import WorkloadBudget
    with pytest.raises(ValueError, match='trusted source'):
        m.cognee_factory(cfg=None, group=None, documents=[],
                         source_proof={'complete': True, 'documents_sha256': 'wrong'},
                         out=tmp_path, client=None,
                         budget=WorkloadBudget(max_calls=12, max_seconds=120),
                         embedder=None, qa_reserve=1, python_executable='unused')
    assert not list(tmp_path.iterdir())
