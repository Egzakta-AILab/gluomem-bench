"""Pinned-source AST temporal checks; no native imports or model construction."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from gluomem_bench.bench import mem0
from gluomem_bench.bench.types import Sample





def test_retriever_explicit_contract_keeps_reader_date_and_scope(tmp_path):
    calls, reader = [], []
    def search(query, **kwargs):
        calls.append((query, kwargs))
        return {'results': [{'memory': 'fixture', 'metadata': {'session_date': '2023-01-01'}}]}
    cfg = NS(benchmark=NS(mem0=NS(top_k=50), answer=NS(context_k=50)))
    def answer(question, context):
        reader.append(question)
        return NS(text='fixture answer', prompt_tokens=1, completion_tokens=1)
    r = mem0.Mem0Retriever(cfg, db_dir=tmp_path, answer_fn=answer,
                         prepared_memories={'h': NS(search=search)}, clock_mode='native')
    sample = Sample(id='q', conv_id='h', question='When?', gold='', category=1,
                    category_name='fixture', question_date='2023-02-03')
    assert r.answer(sample).answer == 'fixture answer'
    assert calls == [('When?', {'filters': {'user_id': 'h'}, 'top_k': 50})]
    assert reader == [sample.reader_question]
    with pytest.raises(ValueError, match='source clock unsupported'):
        mem0.Mem0Retriever(cfg, db_dir=tmp_path, answer_fn=answer,
                           prepared_memories={'h': NS(search=search)}, clock_mode='source')
    assert len(calls) == 1
