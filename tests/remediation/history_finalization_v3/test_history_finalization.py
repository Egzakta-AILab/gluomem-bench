"""Authored data/generations/vectors only; actual run_history and evidence validators."""
import json
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from gluomem_bench.bench import lme_first_execute as x
from gluomem_bench.bench import suite_campaign as campaign
from gluomem_bench.bench.embedding_runtime import RemoteEmbedder, EmbeddingRecipe
from gluomem_bench.bench.parallel_campaign import CampaignBudget
from gluomem_bench.bench.rack import RackProfile
from gluomem_bench.bench.rack_schema import production_expectation


@pytest.mark.parametrize('mode', ['auto', 'required'])
def test_actual_history_finalizes_with_independent_identity(tmp_path, mode):
    import __main__
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            assert self.path == '/v1/models'
            self.reply({'data': [{'id': 'authored/qwen', 'max_model_len': 131072}]})
        def reply(self, response):
            encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/tokenize':
                self.reply({'count': 10})
                return
            assert self.path == '/v1/chat/completions'
            seen.append(body)
            visible = json.dumps(body['messages'])
            if 'claims' in visible and 'reference' in visible.lower():
                content = json.dumps({'claims': [{'text': 'Rhea owns a cobalt telescope.',
                    'kind': 'fact', 'confidence': 1, 'verbatim_span': 'Rhea owns a cobalt telescope.',
                    'message_id': 's0t0'}]})
            elif body['messages'][0]['role'] == 'user':
                content = 'yes'
            else:
                content = 'cobalt telescope'
            self.reply({'id': 'authored-fixture', 'object': 'chat.completion', 'created': 1,
                'model': 'authored/qwen', 'choices': [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': content}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20}})
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    __main__.allowed.add(server.server_address)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        source = tmp_path/'source.json'
        source.write_bytes(x.json_bytes([{'question_id': 'authored', 'question': 'What does Rhea own?',
            'answer': 'cobalt telescope', 'question_type': 'single-session-user',
            'question_date': '2024/02/02 (Fri) 12:00', 'answer_session_ids': ['answer_private'],
            'haystack_dates': ['2024/02/01 (Thu) 12:00'], 'haystack_session_ids': ['answer_private'],
            'haystack_sessions': [[{'role': 'user', 'content': 'Rhea owns a cobalt telescope.'}]]}]))
        parent = Path(__file__).resolve().parents[2]/'tests/fixtures/native/operator-supplied'
        judge = Path(json.loads(parent.read_bytes())['judge_source'])
        profile = RackProfile(base_url=f'http://127.0.0.1:{server.server_port}/v1',
            model='authored/qwen', max_generation_calls=4, max_run_seconds=30)
        recipe = EmbeddingRecipe()
        plan = {'dataset': {'path': str(source), 'sha256': x.bound(source)['sha256'], 'name': 'longmemeval_s'},
            'profile': profile.model_dump(mode='json'), 'sample_ids': ['authored'], 'judge_source': str(judge),
            'config_overrides': {'benchmark.datasets_dir': str(tmp_path),
                'benchmark.datasets.longmemeval_s': {'url': 'https://authored.invalid/source.json',
                    'file': source.name, 'sha256': x.bound(source)['sha256']},
                'dspy.schema_mode': mode, 'dspy.schema_strict': mode == 'required',
                'dspy.tasks.event_claims.module': 'predict',
                'models.minilm.provider_model': recipe.model, 'models.minilm.revision': recipe.revision,
                'models.minilm.dim': 768, 'storage.vector_dim': 768,
                'search.abstain.enabled': False, 'search.rerank.enabled': False}}
        inputs = x.build_input_cache(tmp_path, plan, CampaignBudget(max_calls=4, max_seconds=30))
        adapter, sample = x.restore_input(inputs['authored'], plan)
        cfg, profile = x.effective_config(plan)
        class AuthoredVectors(RemoteEmbedder):
            def _encode(self, texts, role):
                values = np.zeros((len(texts), 768), dtype=np.float32)
                values[:, 0] = 1
                return values
        out = tmp_path/'leaf'
        result = campaign.run_history(cfg=cfg, adapter=adapter, samples=[sample], profile=profile,
            out=out, system='gluomem', judge_source=judge,
            embedder=AuthoredVectors('http://authored.invalid/v1', recipe=recipe))
        expected = {'dataset': adapter.name, 'system': 'gluomem', 'sample_ids': [sample.id],
            **production_expectation('gluomem', profile, cfg)}
        assert campaign.verify_history(out, expected_job=expected) == result
        assert result['questions'] == 1
        assert len(seen) == 3
        assert json.loads((out/'usage.json').read_bytes())['generation_calls'] == 3
        assert not (out/'failure.json').exists()
        assert not (out/'manifest.pending.json').exists()
        prediction = json.loads((out/'predictions.json').read_bytes())[0]
        assert prediction['judge_verdict'] == 'yes'
        assert prediction['sample']['question_date'] == sample.question_date
        assert any('cobalt telescope' in row['text'] for row in prediction['prediction']['retrieved'])
        for key, wrong in [('dataset', 'wrong'), ('system', 'wrong'), ('sample_ids', ['wrong'])]:
            with pytest.raises(ValueError, match='workload identity'):
                campaign.verify_history(out, expected_job={**expected, key: wrong})
        if mode == 'required':
            assert seen[0]['response_format']['json_schema']['strict'] is True
            for invalid in [None, {k: v for k, v in expected.items() if k != 'rack_schema_transport'},
                {**expected, 'rack_schema_transport': {'transport': 'unknown'}}]:
                with pytest.raises(ValueError, match='unbound declared production schema transport'):
                    campaign.verify_history(out, expected_job=invalid)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        __main__.allowed.remove(server.server_address)
