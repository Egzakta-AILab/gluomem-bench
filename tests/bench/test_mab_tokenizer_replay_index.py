"""Offline authored tokenizer receipts only; no QA, dataset, DB or endpoints."""
from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / 'gluomem_bench/bench/mab_campaign.py'


def tokenizer_class(source, calls, *, collide=False):
    """Execute the actual nested production AST without importing clients."""
    tree = ast.parse(source)
    verifier = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_verify_payload')
    cls = next(n for n in verifier.body if isinstance(n, ast.ClassDef) and n.name == 'SavedTokenizer')
    namespace = {'calls': calls, 'profile': SimpleNamespace(model='fixture-model')}
    if collide:
        namespace['hash'] = lambda value: 0
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['SavedTokenizer']


def messages(i):
    return [{'role': 'user', 'content': f'authored fixture {i}'}]


def receipt(message, count=7):
    return {'url': 'https://fixture.invalid/tokenize', 'request': {
        'messages': message, 'model': 'fixture-model', 'add_generation_prompt': True,
        'chat_template_kwargs': {'enable_thinking': False}},
        'status': 200, 'response': {'count': count}}


class CountingCalls(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.visits = 0
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        for value in super().__iter__():
            self.visits += 1
            yield value


def test_saved_tokenizer_consumes_receipts_once_not_per_query():
    calls = CountingCalls(receipt(messages(i), i) for i in range(64))
    tokenizer = tokenizer_class(SOURCE.read_bytes(), calls)()
    for i in range(64):
        assert tokenizer.prompt_token_count(messages(i)) == i
    assert calls.visits <= len(calls), f'{calls.visits} receipt visits, expected at most one pass'
    assert calls.passes == 1


@pytest.fixture
def old_source():
    path = os.environ.get('MAB_TOKENIZER_OLD_SOURCE')
    if not path:
        pytest.skip('MAB_TOKENIZER_OLD_SOURCE required for actual preserved old AST differential')
    return Path(path).read_bytes()


def outcome(tokenizer, query):
    try:
        return ('return', tokenizer.prompt_token_count(query))
    except Exception as exc:
        return ('raise', type(exc).__name__, str(exc))


@pytest.mark.parametrize('collide', [False, True])
@pytest.mark.parametrize('variant', [
    'same', 'different', 'duplicate', 'conflict', 'missing', 'role', 'model',
    'add_false', 'add_int', 'thinking_true', 'thinking_missing', 'extra_setting',
    'non200', 'error', 'not_tokenize', 'bad_plus_good', 'equal_numeric_counts',
    'missing_count', 'null_count', 'unhashable_count', 'reordered_dict',
    'list_order', 'nested_content', 'numeric_equality', 'dict_messages',
    'null_messages', 'absent_messages', 'string_messages', 'empty_list',
    'list_vs_dict', 'int_float_equal', 'signed_zero_equal', 'unicode',
])
def test_actual_old_ast_admissibility_and_count_consistency(old_source, variant, collide):
    query = messages(1)
    calls = [receipt(copy.deepcopy(query))]
    request = calls[0]['request']
    if variant == 'different':
        query = messages(2)
    elif variant == 'duplicate':
        calls.append(copy.deepcopy(calls[0]))
    elif variant == 'conflict':
        calls.append(receipt(copy.deepcopy(query), 8))
    elif variant == 'missing':
        calls.clear()
    elif variant == 'role':
        request['messages'][0]['role'] = 'assistant'
    elif variant == 'model':
        request['model'] = 'forged-model'
    elif variant in ('add_false', 'add_int'):
        request['add_generation_prompt'] = False if variant == 'add_false' else 1
    elif variant == 'thinking_true':
        request['chat_template_kwargs']['enable_thinking'] = True
    elif variant == 'thinking_missing':
        request.pop('chat_template_kwargs')
    elif variant == 'extra_setting':
        request['chat_template_kwargs']['other'] = False
    elif variant == 'non200':
        calls[0]['status'] = 201
    elif variant == 'error':
        calls[0]['error'] = 'authored rejection'
    elif variant == 'not_tokenize':
        calls[0]['url'] = 'https://fixture.invalid/chat/completions'
    elif variant == 'bad_plus_good':
        calls[0]['status'] = 500
        calls.append(receipt(copy.deepcopy(query), 8))
    elif variant == 'equal_numeric_counts':
        calls[0]['response']['count'] = 1
        calls.extend([receipt(copy.deepcopy(query), True), receipt(copy.deepcopy(query), 1.0)])
    elif variant == 'missing_count':
        calls[0]['response'].clear()
    elif variant == 'null_count':
        calls[0]['response']['count'] = None
    elif variant == 'unhashable_count':
        calls[0]['response']['count'] = []
    elif variant == 'reordered_dict':
        request['messages'] = [{'content': 'authored fixture 1', 'role': 'user'}]
    elif variant == 'list_order':
        query = [*query, {'role': 'assistant', 'content': 'different order'}]
        request['messages'] = list(reversed(query))
    elif variant == 'nested_content':
        query = [{'role': 'user', 'content': [{'type': 'text', 'text': 'fixture'}, {'nested': [None, True, 1.5]}]}]
        request['messages'] = copy.deepcopy(query)
    elif variant == 'numeric_equality':
        query = [{'role': 'user', 'content': {'x': 1, 'y': [False, None]}}]
        request['messages'] = [{'content': {'y': [0.0, None], 'x': True}, 'role': 'user'}]
    elif variant == 'dict_messages':
        query = {'arbitrary': [1, {'x': 'y'}]}
        request['messages'] = copy.deepcopy(query)
    elif variant in ('null_messages', 'absent_messages'):
        query = None
        request['messages'] = None
        if variant == 'absent_messages':
            request.pop('messages')
    elif variant == 'string_messages':
        query = 'valid JSON scalar, historical seam accepts equality'
        request['messages'] = query
    elif variant == 'empty_list':
        query = []
        request['messages'] = []
    elif variant == 'list_vs_dict':
        query = []
        request['messages'] = {}
    elif variant == 'int_float_equal':
        query = [2**53]
        request['messages'] = [float(2**53)]
    elif variant == 'signed_zero_equal':
        query = [0]
        request['messages'] = [-0.0]
    elif variant == 'unicode':
        query = [{'role': 'user', 'content': '\u2603\n\u00e9'}]
        request['messages'] = copy.deepcopy(query)
    old = tokenizer_class(old_source, calls)()
    new = tokenizer_class(SOURCE.read_bytes(), calls, collide=collide)()
    assert outcome(new, query) == outcome(old, query)
    # Cache hits must obey the same contract as initial lookups.
    assert outcome(new, copy.deepcopy(query)) == outcome(old, copy.deepcopy(query))
    if variant in ('different', 'conflict', 'missing', 'role', 'model', 'add_false',
                   'add_int', 'thinking_true', 'thinking_missing', 'extra_setting',
                   'non200', 'error', 'not_tokenize', 'list_order', 'list_vs_dict'):
        assert outcome(new, query)[0] == 'raise'


@pytest.mark.parametrize('collide', [False, True])
def test_hash_collision_cannot_reuse_wrong_message_receipt_or_cached_count(collide):
    calls = [receipt(messages(1), 11), receipt(messages(2), 22)]
    tokenizer = tokenizer_class(SOURCE.read_bytes(), calls, collide=collide)()
    assert tokenizer.prompt_token_count(messages(1)) == 11
    assert tokenizer.prompt_token_count(messages(2)) == 22
    assert tokenizer.prompt_token_count(messages(1)) == 11
    with pytest.raises(ValueError, match='missing or inconsistent'):
        tokenizer.prompt_token_count(messages(3))


def test_duplicate_counts_are_checked_once_then_cached():
    class CountReads(dict):
        reads = 0

        def get(self, key, *args):
            type(self).reads += key == 'count'
            return super().get(key, *args)

    calls = [receipt(messages(1)) for _ in range(64)]
    for call in calls:
        call['response'] = CountReads(call['response'])
    tokenizer = tokenizer_class(SOURCE.read_bytes(), calls)()
    for _ in range(64):
        assert tokenizer.prompt_token_count(messages(1)) == 7
    assert CountReads.reads == 64


def test_unmatched_receipt_with_missing_response_does_not_change_lookup(old_source):
    calls = [receipt(messages(1), 11), receipt(messages(2), 22)]
    calls[1].pop('response')
    old = tokenizer_class(old_source, calls)()
    new = tokenizer_class(SOURCE.read_bytes(), calls)()
    assert outcome(new, messages(1)) == outcome(old, messages(1)) == ('return', 11)
    assert outcome(new, messages(2)) == outcome(old, messages(2))


def test_verifier_constructs_one_shared_index_for_evidence_and_dispatch():
    tree = ast.parse(SOURCE.read_bytes())
    verifier = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_verify_payload')
    construction = [n for n in ast.walk(verifier) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == 'SavedTokenizer']
    assert len(construction) == 1
    assert any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Name) and n.value.func.id == 'SavedTokenizer'
               for n in verifier.body)
