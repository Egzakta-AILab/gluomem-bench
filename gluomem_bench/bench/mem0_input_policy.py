"""Lossless, explicitly adapted Mem0 input grouping; no tokenizer/model imports."""
from __future__ import annotations

import hashlib
import json


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _parsed(messages):
    # Pinned mem0 2.2.1 memory.utils.parse_messages for admitted textual roles.
    return ''.join(f"{m['role']}: {m['content']}\n" for m in messages)


def plan_mem0_inputs(inputs, *, scope, max_batch_input_tokens, count_query_tokens):
    """Split only when exact native serialized-query count exceeds the bound.

    count_query_tokens(text) must include the pinned encoder query prefix AND
    special tokens; it is injected by the caller, never downloaded here. This
    bounds ingestion's existing-memory query, not the full extraction prompt or
    output. Source parts use zero-based session/turn ordinals and half-open Python
    Unicode code-point offsets in ORIGINAL dated message content, not token IDs.
    Prefixes/newlines are sliced verbatim, never repeated on continuation parts.
    """
    if type(max_batch_input_tokens) is not int or max_batch_input_tokens <= 0:
        raise ValueError('positive explicit Mem0 input token bound required')
    if not callable(count_query_tokens) or not scope:
        raise ValueError('injected query token counter and scope required')

    def count(messages):
        value = count_query_tokens(_parsed(messages))
        if type(value) is not int or value < 0:
            raise ValueError('invalid exact query token count')
        return value

    blocks = []
    for si, source in enumerate(inputs):
        pending, parts = [], []
        block_index = 0

        def flush():
            nonlocal pending, parts, block_index
            if not pending:
                return
            blocks.append({**source, 'messages': pending,
                           'source_session_ordinal': si, 'block_index': block_index,
                           'source_parts': parts, 'query_tokens': count(pending)})
            block_index += 1
            pending, parts = [], []

        for ti, message in enumerate(source['messages']):
            if (message['role'] not in ('user', 'assistant')
                    or type(message['content']) is not str):
                raise ValueError('unsupported Mem0 source role/content')
            text, start = message['content'], 0
            while True:  # Empty source turns still have one explicit 0:0 part.
                remaining = {**message, 'content': text[start:]}
                if count(pending + [remaining]) <= max_batch_input_tokens:
                    pending.append(remaining)
                    parts.append({'turn_index': ti, 'start': start, 'end': len(text)})
                    break
                if pending:
                    flush()
                    continue
                # Exact checks, not character/token estimates. Token counts need
                # not be monotonic: maximal packing is NOT a protocol guarantee.
                end = len(text)
                while end > start and count([{**message, 'content': text[start:end]}]) > max_batch_input_tokens:
                    end = start + (end - start) // 2
                if end == start:
                    raise ValueError('Mem0 token bound cannot fit an atomic source part')
                pending.append({**message, 'content': text[start:end]})
                parts.append({'turn_index': ti, 'start': start, 'end': end})
                flush()
                start = end
        flush()
    return {'protocol': 'adapted:mem0-token-blocks@v1', 'scope': scope,
            'max_batch_input_tokens': max_batch_input_tokens,
            'count_policy': 'native-serialized-query-with-encoder-prefix-and-special-tokens',
            'offset_unit': 'unicode-code-point', 'original_inputs_sha256': _digest(inputs),
            'original_sessions': len(inputs),
            'original_turns': sum(len(s['messages']) for s in inputs), 'blocks': blocks}


def verify_mem0_input_plan(original_inputs, plan, *, scope, max_batch_input_tokens,
                           count_query_tokens):
    """Recompute from independently admitted ORIGINAL input, scope and policy.

    Never use the block file's self-declared original hash/cap as authority.
    Exact recomputation binds order, roles, dates, empty turns and all offsets.
    """
    wanted = plan_mem0_inputs(original_inputs, scope=scope,
                             max_batch_input_tokens=max_batch_input_tokens,
                             count_query_tokens=count_query_tokens)
    if _digest(plan) != _digest(wanted):
        raise ValueError('Mem0 source block coverage differs from independent original input')


def verify_mem0_block_receipts(original_inputs, plan, receipts, **policy):
    """Additional adapted guard; does NOT replace native HTTP/ADD validation."""
    verify_mem0_input_plan(original_inputs, plan, **policy)
    if len(receipts) != len(plan['blocks']):
        raise ValueError('Mem0 block receipt count differs')
    for ordinal, (block, receipt) in enumerate(zip(plan['blocks'], receipts, strict=True), 1):
        expected = {k: block[k] for k in ('session_index', 'session_date',
                    'source_session_ordinal', 'block_index', 'source_parts')}
        expected.update(messages_sha256=_digest(block['messages']),
                        submitted_turns=len(block['messages']), ordinal=ordinal,
                        native_add_ack=True)
        if _digest({k: receipt.get(k) for k in expected}) != _digest(expected):
            raise ValueError('Mem0 block native ADD acknowledgment differs')
