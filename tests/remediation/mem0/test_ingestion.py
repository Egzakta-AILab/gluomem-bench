"""Native-shaped offline fixture, NOT native SDK or model fidelity evidence."""
import json
import sqlite3
import sys
import types
from types import SimpleNamespace as NS
from datetime import datetime
import pytest
from gluomem_bench.bench import mem0
from gluomem_bench.ingest import Conversation, Session, Turn


@pytest.fixture
def ingestion(tmp_path, monkeypatch):
    monkeypatch.setenv('MEM0_TELEMETRY', 'False')
    monkeypatch.setenv('MEM0_DIR', str(tmp_path / 'native-home'))
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    parsed = lambda messages: ''.join(f"{m['role']}: {m['content']}\n" for m in messages)
    for name, attrs in {
        'mem0.configs.base': {'mem0_dir': str(tmp_path / 'native-home')},
        'mem0.configs.prompts': {'ADDITIVE_EXTRACTION_PROMPT': 'fixture extraction'},
        'mem0.memory.telemetry': {'MEM0_TELEMETRY': False},
        'mem0.memory.utils': {'parse_messages': parsed},
        'mem0.utils.entity_extraction': {'extract_entities': lambda _: ['fixture']},
    }.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(mem0, '_installed_versions', lambda: {'mem0ai': '2.2.1'})
    calls = []
    state = NS(mode='healthy', closed=False, embedding_queries=[], constructions=0)
    profile = NS(model='fixture/model', base_url='http://fixture.invalid/v1',
                 output_budget=lambda _: 64)
    budget = NS(calls=0, check=lambda: None)
    client = NS(profile=profile, check=lambda: None, _latch=lambda exc: None)

    def generate(task, messages, **kwargs):
        budget.calls += 1
        raw = '{"memory": []}' if state.mode != 'malformed' else 'broken'
        if state.mode in ('commit', 'wrong_scope'):
            raw = '{"memory": [{"text": "fixture committed fact", "event": "ADD"}]}'
        if state.mode == 'length':
            raise RuntimeError('authored length rejection')
        return {'response': {'id': 'fixture', 'object': 'chat.completion', 'created': 1,
                'model': profile.model, 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': raw}}]}}
    client.generate_messages = generate
    class Memory:
        @classmethod
        def from_config(cls, config):
            self = cls()
            self.llm = NS(config=NS(model=profile.model),
                          client=NS(base_url=profile.base_url, close=lambda: None))
            self.embedding_model = NS(config=NS(model='fixture/encoder'))
            self.vector_store = NS(is_local=True, _get_bm25_encoder=lambda: object(),
                                   client=NS(close=lambda: None))
            state.constructions += 1
            db = sqlite3.connect(tmp_path / f'fixture-{state.constructions}.db')
            db.execute('CREATE TABLE messages (role TEXT, content TEXT)')
            db.execute('CREATE TABLE history (memory_id TEXT, new_memory TEXT, event TEXT)')
            self.db = NS(connection=db)
            self.saved = {}
            return self
        def add(self, messages, *, user_id, metadata, infer):
            calls.append({'messages': messages, 'user_id': user_id, 'metadata': metadata, 'infer': infer})
            state.embedding_queries.append(parsed(messages))
            self.llm.client.create(model=profile.model, temperature=0, top_p=1.0,
                max_tokens=64, response_format={'type': 'json_object'}, messages=[
                    {'role': 'system', 'content': 'fixture extraction'}, {'role': 'user',
                     'content': '## New Messages\n' + parsed(messages) + '\n\n## Observation Date\n2026-10-19'}])
            self.db.connection.executemany('INSERT INTO messages VALUES (?, ?)',
                                          [(m['role'], m['content']) for m in messages])
            self.db.connection.execute('DELETE FROM messages WHERE rowid NOT IN (SELECT rowid FROM messages ORDER BY rowid DESC LIMIT 10)')
            self.db.connection.commit()
            if state.mode == 'bad_ack':
                return {}
            if state.mode in ('commit', 'wrong_scope'):
                identity = str(len(self.saved))
                row = {'id': identity, 'memory': 'fixture committed fact',
                       'user_id': user_id if state.mode == 'commit' else 'other'}
                self.saved[identity] = row
                self.db.connection.execute('INSERT INTO history VALUES (?, ?, ?)',
                                            (identity, row['memory'], 'ADD'))
                self.db.connection.commit()
                return {'results': [dict(row, event='ADD')]}
            return {'results': []}
        def get(self, identity):
            return self.saved.get(identity)
        def get_all(self, **kwargs):
            return {'results': list(self.saved.values())}
        def close(self):
            self.db.connection.close()
            state.closed = True
    monkeypatch.setattr(mem0, '_import_memory_cls', lambda: Memory)
    monkeypatch.setattr(mem0, 'build_mem0_config', lambda *a, **k: {'llm': {
        'provider': 'openai', 'config': {'model': profile.model,
        'openai_base_url': profile.base_url, 'api_key': 'fixture', 'temperature': 0,
        'top_p': 1.0, 'max_tokens': profile.output_budget('event_claims')}}})
    cfg = NS(models={'encoder': NS(provider_model='fixture/encoder')},
             benchmark=NS(mem0=NS(embedder_model='encoder'), answer=NS(context_k=50)))
    conv = Conversation('opaque-history', ('user', 'assistant'), [
        Session(7, datetime(2023, 1, 2), [Turn('user', ' α\n' * 50), Turn('user', ''), Turn('assistant', ' tail\n')])])
    out = tmp_path / 'run'
    out.mkdir()
    return NS(cfg=cfg, conv=conv, out=out, client=client, budget=budget, calls=calls, state=state)


def test_adapted_ingestion_binds_original_blocks_and_shared_calls(ingestion):
    f = ingestion
    memory = mem0.ingest_guarded_mem0(f.cfg, conv=f.conv, out=f.out, client=f.client,
        budget=f.budget, max_batch_input_tokens=60, count_query_tokens=len, clock_mode='native')
    try:
        plan = json.loads((f.out / 'mem0-input-plan.json').read_bytes())
        originals = json.loads((f.out / 'mem0-original-inputs.json').read_bytes())
        inputs = json.loads((f.out / 'mem0-source-inputs.json').read_bytes())
        summary = json.loads((f.out / 'mem0-ingestion.json').read_bytes())
        mem0.verify_mem0_input_plan(originals, plan, scope=f.conv.conv_id,
                                    max_batch_input_tokens=60, count_query_tokens=len)
        assert inputs == plan['blocks']
        assert len(inputs) == f.budget.calls == len(f.calls) == len(f.state.embedding_queries) > 1
        assert all(len(s) <= 60 for s in f.state.embedding_queries)
        assert all(c['user_id'] == f.conv.conv_id and c['infer'] for c in f.calls)
        assert summary['original_inputs_sha256'] == mem0._mem0_digest(originals)
        assert summary['original_submitted_turns'] == 3
        assert summary['input_protocol'] == plan['protocol']
        assert all(r['native_add_ack'] is True for r in summary['sessions'])
        assert summary['native_add_calls'] == summary['native_extraction_calls'] == f.budget.calls
        assert summary['temporal_contract']['source_clock_supported'] is False
        # Compose the new independent source guard with the UNCHANGED native
        # evidence validator. Never exempt the original-source mismatch globally.
        from gluomem_bench.bench.suite_evidence import validate_mem0_evidence
        inventory = {p.relative_to(f.out).as_posix(): 'fixture-only' for p in f.out.rglob('*') if p.is_file()}
        native = [json.loads(line) for line in (f.out / 'mem0-native.jsonl').read_bytes().splitlines()]
        direct = [{'status': 200, 'request': dict(n['request'], chat_template_kwargs={'enable_thinking': False}),
                   'response': n['response']} for n in native]
        with pytest.raises(ValueError, match='source inputs differ'):
            validate_mem0_evidence(f.out, inventory, f.client.profile,
                {'mem0_source_inputs_sha256': mem0._mem0_digest(originals)}, direct)
        mem0.verify_mem0_block_receipts(originals, plan, summary['sessions'], scope=f.conv.conv_id,
                                        max_batch_input_tokens=60, count_query_tokens=len)
        derived = {'mem0_source_inputs_sha256': mem0._mem0_digest(plan['blocks'])}
        expected = dict(derived, profile_sha256=None, dataset_sha256=None,
                        mem0_extractor_system_sha256=__import__('hashlib').sha256(b'fixture extraction').hexdigest())
        assert validate_mem0_evidence(f.out, inventory, f.client.profile, derived, direct,
                                      expected=expected) == len(plan['blocks'])
    finally:
        mem0.close_mem0_memory(memory)


def test_native_add_requires_explicit_results_ack(ingestion):
    f = ingestion
    f.state.mode = 'bad_ack'
    with pytest.raises(Exception, match='ADD acknowledgment'):
        mem0.ingest_guarded_mem0(f.cfg, conv=f.conv, out=f.out, client=f.client,
            budget=f.budget, max_batch_input_tokens=60, count_query_tokens=len)
    assert not (f.out / 'mem0-ingestion.json').exists()
    assert f.state.closed


@pytest.mark.parametrize('mode', ['malformed', 'length', 'wrong_scope'])
def test_unhealthy_extraction_never_promotes_and_healthy_successor_runs(ingestion, mode):
    f = ingestion
    f.state.mode = mode
    with pytest.raises(Exception):
        mem0.ingest_guarded_mem0(f.cfg, conv=f.conv, out=f.out, client=f.client,
            budget=f.budget, max_batch_input_tokens=60, count_query_tokens=len)
    assert not (f.out / 'mem0-ingestion.json').exists()
    assert not (f.out / 'stores' / f.conv.conv_id / 'ingest_meta.json').exists()
    assert f.state.closed
    used = f.budget.calls
    assert used == 1
    f.state.mode = 'commit'
    next_out = f.out.parent / 'healthy-successor'
    next_out.mkdir()
    memory = mem0.ingest_guarded_mem0(f.cfg, conv=f.conv, out=next_out, client=f.client,
        budget=f.budget, max_batch_input_tokens=60, count_query_tokens=len)
    try:
        summary = json.loads((next_out / 'mem0-ingestion.json').read_bytes())
        assert summary['complete'] and summary['memories'] > 0
        assert summary['native_extraction_calls'] == f.budget.calls - used
        assert all(row['user_id'] == f.conv.conv_id for receipt in summary['sessions'] for row in receipt['added'])
    finally:
        mem0.close_mem0_memory(memory)


def test_source_clock_refuses_before_native_construction_and_persists_gap(ingestion):
    f = ingestion
    with pytest.raises(ValueError, match='source clock unsupported'):
        mem0.ingest_guarded_mem0(f.cfg, conv=f.conv, out=f.out, client=f.client,
                                 budget=f.budget, clock_mode='source')
    contract = json.loads((f.out / 'mem0-temporal-contract.json').read_bytes())
    assert contract['status'] == 'unsupported'
    assert not contract['add_timestamp_supported'] and not contract['search_reference_date_supported']
    assert f.state.constructions == f.budget.calls == 0


def test_legacy_default_keeps_exact_whole_session_messages(ingestion):
    f = ingestion
    memory = mem0.ingest_guarded_mem0(f.cfg, conv=f.conv, out=f.out, client=f.client, budget=f.budget)
    try:
        summary = json.loads((f.out / 'mem0-ingestion.json').read_bytes())
        assert len(f.calls) == f.budget.calls == 1
        assert f.calls[0]['messages'] == mem0._session_messages(f.conv.sessions[0])
        assert not (f.out / 'mem0-input-plan.json').exists()
        assert summary['submitted_turns'] == 3
    finally:
        mem0.close_mem0_memory(memory)
