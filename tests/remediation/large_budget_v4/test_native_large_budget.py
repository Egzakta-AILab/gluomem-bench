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
from test_large_budget import material


@pytest.fixture
def native(tmp_path):
    import __main__
    requests, leases = [], []
    state = dict(prompt_tokens=20, finish='stop', completion_tokens=8192, model='authored/qwen')
    rationale = 'Evidence sentence. ' * 8192
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
                        'action': 'ADD', 'target_frame_id': None, 'rationale': rationale}})}}],
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
    cfg = cfg.with_overrides({'dspy.schema_mode': 'required', 'dspy.schema_strict': True})
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


def test_native_factory_accepts_complete_large_rationale_with_exact_reservation(native):
    state = native['state']
    profile = native['profile']
    state['prompt_tokens'] = profile.context_window - profile.output_budget('consolidate') - profile.schema_reserve_tokens
    decision = consolidate(native)
    assert decision.action == 'ADD' and decision.rationale == native['rationale']
    assert native['budget'].calls == 1
    assert native['leases'] == [131072]
    assert [p for p, _ in native['requests']] == ['/tokenize', '/v1/chat/completions']
    body = native['requests'][-1][1]
    assert body['max_tokens'] == 32768 and body['temperature'] == 0
    assert body['model'] == profile.model
    assert body['chat_template_kwargs'] == {'enable_thinking': False}
    assert body['response_format']['type'] == 'json_schema'
    assert 'rationale' in json.dumps(body['response_format'])
    lm = native['router'].lm_for('consolidate')
    assert lm.kwargs['max_tokens'] == 32768 and lm.num_retries == 0 and lm.engine == 'lm15'
    assert lm.kwargs['timeout'] == 1800
    row = json.loads(native['trace'].read_text())
    rack_schema.validate_native_projection(row['response'], row['native_response'])
    assert row['response']['usage']['completion_tokens'] == 8192
    assert row['output_budget'] == 32768
    assert row['prompt_token_count'] == state['prompt_tokens']
    assert row['generation_kwargs']['response_format'] == body['response_format']
    expected = rack_schema.production_expectation('gluomem', profile, native['cfg'])
    assert expected == {'rack_schema_transport': rack_schema.schema_binding(profile, native['cfg'].dspy)}
    assert rack_schema.validate_schema_row(row, profile, native['cfg'].model_dump(mode='json'),
        expected, expected) == lm.model


def test_native_context_overflow_rejected_before_generation(native):
    profile = native['profile']
    native['state']['prompt_tokens'] = profile.context_window - 32768 - profile.schema_reserve_tokens + 1
    with pytest.raises(ExtractionError):
        consolidate(native)
    assert native['budget'].calls == 0 and native['leases'] == []
    assert [p for p, _ in native['requests']] == ['/tokenize']


def test_native_length_at_large_allowance_is_fatal_without_feedback_retry(native):
    native['state'].update(finish='length', completion_tokens=32768)
    with pytest.raises(ExtractionError):
        consolidate(native)
    assert native['budget'].calls == 1
    assert [p for p, _ in native['requests']] == ['/tokenize', '/v1/chat/completions']
    assert native['budget'].failed
    with pytest.raises(ExtractionError):
        consolidate(native)
    assert native['budget'].calls == 1


def test_native_wrong_served_model_remains_fatal(native):
    native['state']['model'] = 'wrong/model'
    with pytest.raises(ExtractionError):
        consolidate(native)
    assert native['budget'].calls == 1 and native['budget'].failed


def test_v7_whole_history_policy_is_available():
    budget = WorkloadBudget(max_calls=16384, max_seconds=21600, policy='role-budgets-v7')
    assert budget.max_calls == 16384 and budget.max_seconds == 21600
