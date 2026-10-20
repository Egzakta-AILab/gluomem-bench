"""Authored fixtures only: no native clients, services, or model calls."""
import copy
import pytest
from gluomem_bench.bench import mem0


def original():
    return [{"session_index": 7, "session_date": "2023-01-02T00:00:00", "messages": [
        {"role": "user", "content": "[2023-01-02] user: α\n  "},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "next\r\n"}]}]


def test_exact_boundary_lossless_roles_and_empty_turns():
    inputs = original()
    serialized = "".join(f"{m['role']}: {m['content']}\n" for m in inputs[0]['messages'])
    plan = mem0.plan_mem0_inputs(inputs, scope="opaque-history", max_batch_input_tokens=len(serialized),
                                 count_query_tokens=len)
    assert len(plan['blocks']) == 1
    assert plan['blocks'][0]['messages'] == inputs[0]['messages']
    assert plan['protocol'] == 'adapted:mem0-token-blocks@v1'
    assert mem0.verify_mem0_input_plan(inputs, plan, scope="opaque-history", max_batch_input_tokens=len(serialized), count_query_tokens=len) is None
    assert inputs == original()


def test_oversized_single_turn_reconstructs_exact_bytes():
    inputs = original()
    plan = mem0.plan_mem0_inputs(inputs, scope="h", max_batch_input_tokens=17, count_query_tokens=len)
    restored = [''] * len(inputs[0]['messages'])
    seen = []
    for block in plan['blocks']:
        assert len(''.join(f"{m['role']}: {m['content']}\n" for m in block['messages'])) <= 17
        for msg, part in zip(block['messages'], block['source_parts'], strict=True):
            restored[part['turn_index']] += msg['content']
            seen.append(part['turn_index'])
    assert [s.encode() for s in restored] == [m['content'].encode() for m in inputs[0]['messages']]
    assert 1 in seen
    mem0.verify_mem0_input_plan(inputs, plan, scope="h", max_batch_input_tokens=17, count_query_tokens=len)


@pytest.mark.parametrize('mutation', ['content', 'role', 'date', 'order', 'offset', 'scope', 'cap', 'original'])
def test_forged_coverage_refused(mutation):
    inputs = original()
    plan = mem0.plan_mem0_inputs(inputs, scope='h', max_batch_input_tokens=17, count_query_tokens=len)
    if mutation == 'content': plan['blocks'][0]['messages'][0]['content'] += 'x'
    if mutation == 'role': plan['blocks'][0]['messages'][0]['role'] = 'assistant'
    if mutation == 'date': plan['blocks'][0]['session_date'] = '2026-10-19'
    if mutation == 'order': plan['blocks'].reverse()
    if mutation == 'offset': plan['blocks'][0]['source_parts'][0]['start'] = 1
    if mutation == 'scope': plan['scope'] = 'other'
    if mutation == 'cap': plan['max_batch_input_tokens'] = 18
    if mutation == 'original':
        inputs[0]['messages'][0]['content'] += 'changed original'
    with pytest.raises(ValueError, match='coverage'):
        mem0.verify_mem0_input_plan(inputs, plan, scope='h', max_batch_input_tokens=17, count_query_tokens=len)


def test_block_receipts_require_native_ack_and_order():
    inputs = original()
    plan = mem0.plan_mem0_inputs(inputs, scope='h', max_batch_input_tokens=17, count_query_tokens=len)
    receipts = [{'session_index': b['session_index'], 'session_date': b['session_date'],
                 'source_session_ordinal': b['source_session_ordinal'], 'block_index': b['block_index'],
                 'source_parts': b['source_parts'], 'messages_sha256': mem0._mem0_digest(b['messages']),
                 'submitted_turns': len(b['messages']), 'ordinal': i, 'native_add_ack': True}
                for i, b in enumerate(plan['blocks'], 1)]
    verify = lambda r: mem0.verify_mem0_block_receipts(inputs, plan, r, scope='h',
                         max_batch_input_tokens=17, count_query_tokens=len)
    verify(receipts)
    for field, value in [('native_add_ack', False), ('ordinal', 2), ('messages_sha256', 'forged'),
                         ('source_parts', []), ('session_date', 'wrong')]:
        bad = copy.deepcopy(receipts)
        bad[0][field] = value
        with pytest.raises(ValueError): verify(bad)
    with pytest.raises(ValueError): verify(receipts[:-1])


@pytest.mark.parametrize('cap', [24, 32, 64, 128])
def test_injected_prefix_special_count_multisession_exact_reconstruction(cap):
    prefix = 'query: '
    count = lambda text: len(prefix + text) + 2  # Authored, not real tokenizer parity.
    inputs = original()
    inputs[0]['messages'][0]['content'] += '\r\n\u2028😀' * 10
    inputs.append(copy.deepcopy(inputs[0]))  # Same index/date, distinct source occurrence.
    plan = mem0.plan_mem0_inputs(inputs, scope='h', max_batch_input_tokens=cap, count_query_tokens=count)
    restored = [['' for m in s['messages']] for s in inputs]
    occurrences = []
    for b in plan['blocks']:
        assert b['query_tokens'] <= cap
        for m, part in zip(b['messages'], b['source_parts'], strict=True):
            si, ti = b['source_session_ordinal'], part['turn_index']
            assert len(restored[si][ti]) == part['start']
            assert m['role'] == inputs[si]['messages'][ti]['role']
            restored[si][ti] += m['content']
            occurrences.append((si, ti))
    for si, source in enumerate(inputs):
        for ti, m in enumerate(source['messages']):
            assert restored[si][ti].encode() == m['content'].encode()
            assert (si, ti) in occurrences
    assert [b['source_session_ordinal'] for b in plan['blocks']] == sorted(b['source_session_ordinal'] for b in plan['blocks'])


@pytest.mark.parametrize('cap', [None, 0, -1, True, 1.5])
def test_explicit_adapted_policy_requires_positive_integer_cap(cap):
    with pytest.raises(ValueError):
        mem0.plan_mem0_inputs(original(), scope='h', max_batch_input_tokens=cap, count_query_tokens=len)


def test_atomic_unfit_refuses_without_dropping_source():
    inputs = original()
    with pytest.raises(ValueError, match='atomic'):
        mem0.plan_mem0_inputs(inputs, scope='h', max_batch_input_tokens=1, count_query_tokens=len)
    assert inputs == original()
