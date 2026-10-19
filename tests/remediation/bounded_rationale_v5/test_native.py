"""Actual DSPy native schema factory/SDK over owned loopback HTTP only.

Usage/tokenizer counts are authored fixture evidence, not real-model quality.
"""
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from gluomem.llm.modules import DspyExtractors
from gluomem.llm.outputs import ExtractionError, StoredCandidate
from gluomem_bench.bench import rack_schema
from gluomem_bench.bench.rack import RackProfile, RackClient, controlled_config, SharedRackError
from gluomem_bench.bench.rack_trace import WorkloadBudget
def material(**updates):
    value = dict(base_url='http://owned.invalid/v1', model='authored/qwen',
        profile_version='role-budgets-v7', timeout_seconds=1800,
        max_generation_calls=16384, max_run_seconds=21600,
        generation_budgets=dict(extraction=16384, consolidation=32768, reader=4096, judge=4096))
    value.update(updates)
    return value


@pytest.fixture(params=[(False,'zero'),(True,'zero'),(True,'json'),(True,'program')])
def native(tmp_path, request):
    import __main__
    requests, leases = [], []
    state = dict(prompt_tokens=20, finish='stop', completion_tokens=8192, model='authored/qwen')
    rationale = 'é' * 8192
    state['rationale'] = rationale
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((self.path, body))
            if self.path == '/tokenize':
                response = {'count': state['prompt_tokens']}
            else:
                assert self.path == '/v1/chat/completions'
                response = {'id': 'owned-fixture', 'object': 'chat.completion', 'created': 1,
                    'model': state['model'], 'choices': [{'index': 0, 'finish_reason': state['finish'],
                    'message': {'role': 'assistant', 'content': json.dumps({'decision': {
                        'action': 'ADD', 'target_frame_id': None, 'rationale': state['rationale']}})}}],
                    'usage': {'prompt_tokens': state['prompt_tokens'],
                        'completion_tokens': state['completion_tokens'],
                        'total_tokens': state['prompt_tokens'] + state['completion_tokens']}}
            encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    __main__.allowed.add(server.server_address)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    profile = RackProfile.model_validate(material(base_url=f'http://127.0.0.1:{server.server_port}/v1'))
    cfg = controlled_config(profile, data=tmp_path/'unused.json', sha256='a'*64)
    cfg = cfg.with_overrides({'dspy.schema_mode': 'required', 'dspy.schema_strict': True,
        'dspy.output_field_limits': {'consolidate': {'decision.rationale': 8192}}})
    canonicalize, artifact = request.param
    cfg = cfg.with_overrides({'dspy.canonicalize': canonicalize,
        'dspy.artifacts.dir': str(tmp_path/'artifacts')})
    if artifact != 'zero':
        from gluomem.llm.modules import build_module
        from gluomem.llm.bootstrap import save_artifact, save_json_artifact
        legacy = cfg.with_overrides({'dspy.output_field_limits': {}})
        module = build_module('consolidate',legacy)
        saver = save_json_artifact if artifact == 'json' else save_artifact
        saver(module,cfg.dspy.artifacts.dir,'consolidate',model_ref=profile.model)
    class LeaseRecorder:
        @contextmanager
        def lease(self, tokens):
            leases.append(tokens)
            yield
        def check(self):
            pass
        def remaining(self):
            return 21600
        def begin_call(self):
            pass
    budget = WorkloadBudget(max_calls=profile.max_generation_calls,
        max_seconds=profile.max_run_seconds, policy=profile.profile_version, outer_budget=LeaseRecorder())
    client = RackClient(profile, receipt=tmp_path/'http.jsonl', budget=budget)
    trace = tmp_path/'native.jsonl'
    router = rack_schema.make_router(cfg, client=client, trace=trace, budget=budget)
    extractors = DspyExtractors(cfg, router=router)
    try:
        yield dict(profile=profile, cfg=cfg, budget=budget, client=client, router=router,
            extractors=extractors, trace=trace, requests=requests, state=state,
            rationale=rationale, leases=leases)
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join()
        __main__.allowed.remove(server.server_address)


def consolidate(native):
    return native['extractors'].consolidate_claim('Rhea owns a telescope.',
        [StoredCandidate(frame_id=1, content='Rhea owns a camera.')])



def test_native_schema_prompt_wire_and_independent_verifier(native):
    decision = consolidate(native)
    assert decision.rationale == native['rationale'] and decision.target_frame_id is None
    body = native['requests'][-1][1]
    assert body['max_tokens'] == 32768
    schema = body['response_format']['json_schema']['schema']
    assert schema['$defs']['ConsolidationDecision']['properties']['rationale']['maxLength'] == 8192
    assert 'maxLength' in json.dumps(body['messages'])
    row = json.loads(native['trace'].read_text())
    cfg, profile = native['cfg'], native['profile']
    expected = rack_schema.production_expectation('gluomem',profile,cfg)
    assert rack_schema.validate_schema_row(row,profile,cfg.model_dump(mode='json'),expected,expected) == row['model']
    assert expected['rack_schema_transport']['output_field_limits'] == {'consolidate': {'decision.rationale': 8192}}
    import copy
    stale = copy.deepcopy(row)
    del stale['generation_kwargs']['response_format']['json_schema']['schema']['$defs']['ConsolidationDecision']['properties']['rationale']['maxLength']
    with pytest.raises(ValueError):
        rack_schema.validate_schema_row(stale,profile,cfg.model_dump(mode='json'),expected,expected)


def test_native_stop_overlimit_is_parser_rejection_not_decoder_proof(native):
    native['state']['rationale'] = 'é' * 8193
    with pytest.raises(ExtractionError):
        consolidate(native)
    assert native['budget'].calls == 1
    assert [p for p,_ in native['requests']] == ['/tokenize','/v1/chat/completions']
    row = json.loads(native['trace'].read_text())
    assert row['response']['choices'][0]['finish_reason'] == 'stop'
    assert len(json.loads(row['response']['choices'][0]['message']['content'])['decision']['rationale']) == 8193
