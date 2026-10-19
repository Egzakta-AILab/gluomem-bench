"""Vertical contract tests; real-release checks opt in via MAB_ROOT."""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest


def release_root():
    value = os.environ.get("MAB_ROOT")
    if not value:
        pytest.skip("set MAB_ROOT to the external pinned MAB snapshot")
    return Path(value)


@pytest.mark.parametrize("change", ["unknown_source", "wrong_family", "short_ids", "empty_alias", "bad_dates"])
def test_malformed_rows_fail_closed(change):
    raw = row()
    if change == "unknown_source":
        raw['metadata']['source'] = 'invented'
    elif change == "wrong_family":
        raw['metadata']['source'] = 'detective_qa'
    elif change == "short_ids":
        raw['metadata']['qa_pair_ids'].pop()
    elif change == "empty_alias":
        raw['answers'][0] = []
    else:
        raw['metadata']['question_dates'] = ['one']
    with pytest.raises(ValueError):
        api().MABSuite.from_rows({'Accurate_Retrieval': [raw]})


def test_actual_release_all_rows_counts_ids_overlap_and_hash_receipts():
    m = api()
    suite = m.MABSuite.from_directory(release_root())
    inventory = suite.inventory()
    assert inventory['history_occurrences'] == 146
    assert inventory['question_occurrences'] == 3671
    assert inventory['unique_contexts'] == 136
    assert inventory['unique_source_qa_ids'] == 2231
    assert len(inventory['sources']) == 22
    assert len({c.sample.id for c in suite.cases}) == 3671
    assert inventory['sources']['detective_qa']['questions'] == 71
    assert inventory['sources']['infbench_sum_eng_shots2']['histories'] == 100
    assert suite.receipt['data_revision'] == m.DATA_REVISION
    assert suite.receipt['files'] == m.PINNED_FILES
    for history in suite.histories:
        assert history.context
    for case in suite.cases:
        assert case.source_id == case.metadata['qa_pair_ids'][case.question_index]


def test_unpinned_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='pinned'):
        api().MABSuite.from_directory(tmp_path)


def test_generated_history_ids_are_portable_store_filenames(tmp_path):
    suite = api().MABSuite.from_rows({'Accurate_Retrieval': [row()]})
    for history in suite.histories:
        assert not set(history.id) & set('<>:"/\\|?*')
        path = tmp_path / f'{history.id}.mind'
        path.write_bytes(b'authored storage-path probe')
        assert path.read_bytes() == b'authored storage-path probe'
    assert suite.cases[0].source_id == 'same'


def api():
    return importlib.import_module("gluomem_bench.bench.mab_suite")


def row(source="ruler_qa1_197K"):
    return {"context": " exact\ncontext \n", "questions": ["Q", "Q"],
            "answers": [["France", "France", "FR"], ["yes"]],
            "metadata": {"source": source, "qa_pair_ids": ["same", "same"],
                         "demo": None, "question_dates": None}}


def test_native_reader_templates_and_lossless_document_conversation_seam():
    m = api()
    suite = m.MABSuite.from_directory(release_root())
    readers = suite.reader_samples(agent='rag')
    assert len(readers) == len(suite.cases)
    for case, sample in zip(suite.cases, readers, strict=True):
        assert sample.id == case.sample.id
        assert sample.gold == case.sample.gold
        assert sample.reader_question == suite.reader_question(case, agent='rag')
        assert case.sample.question in sample.reader_question
    conflict = next(c for c in suite.cases if c.source.startswith('factconsolidation'))
    prompt = suite.reader_question(conflict)
    assert 'larger serial number' in prompt
    assert 'rather than the real facts in real world' in prompt
    longmem = next(c for c in suite.cases if c.source == 'longmemeval_s*')
    assert suite.reader_question(longmem).count('Current Date:') == 1
    conversations = suite.conversations(chunk_chars=4096)
    assert len(conversations) == 146
    for history, conversation in zip(suite.histories, conversations, strict=True):
        assert conversation.conv_id == history.id
        turns = [t for s in conversation.sessions for t in s.turns]
        assert ''.join(t.text for t in turns) == history.context
        assert all(t.speaker == 'document' for t in turns)
    assert suite.admission()['native_ingestion'] is False


@pytest.mark.parametrize('source,prediction,metric,expected', [
    ('ruler_qa1_197K', 'Answer: The FRANCE.', 'substring_exact_match', 1),
    ('eventqa_full', 'FRANCE or Germany', 'substring_exact_match', 1),
    ('factconsolidation_sh_6k', 'France or wrong', 'substring_exact_match', 1),
    ('icl_nlu_8296shot_balance', 'label: France', 'exact_match', 0),
    ('icl_nlu_8296shot_balance', 'Answer: France', 'exact_match', 1),
    ('detective_qa', 'reasoning\nAnswer: France', 'exact_match', 1),
])
def test_native_scoring_routes_real_upstream_postprocessing(source, prediction, metric, expected):
    m = api()
    suite = m.MABSuite.from_directory(release_root())
    fixture = m.MABSuite.from_rows({m.SOURCE_FAMILIES[source]: [row(source)]})
    case = fixture.cases[0]
    score = suite.scorer().score(case, prediction)
    assert score.metric == metric
    assert score.value == expected
    assert score.source == source
    assert score.metrics[metric] == expected
    assert score.scorer_revision == m.CODE_REVISION


def test_recsys_native_entity_mapping_recall_and_tie_gate():
    import json
    m = api()
    suite = m.MABSuite.from_directory(release_root())
    case = next(c for c in suite.cases if c.source == 'recsys_redial_full')
    scorer = suite.scorer()
    mapping = json.loads((release_root() / 'entity2id.json').read_bytes())
    name = next(name for name, identifier in mapping.items() if identifier == int(case.answers[0]))
    movie = scorer.upstream['extract_movie_name'](name)
    scored = scorer.score(case, '1.' + movie)
    assert scored.metric == 'recsys_recall@5'
    assert scored.metrics['recsys_recall@1'] == 1
    assert 'set-order edit-distance tie breaking' in scorer.gates['recsys_redial_full']


def judged(request, output, model=None):
    return api().JudgeResponse(case_id=request.case_id, role=request.role,
        prediction_sha256=request.prediction_sha256, prompt_sha256=request.prompt_sha256,
        model=model or request.reference_model, output=output)


def test_longmem_native_judge_prompt_binding_and_saved_response_scoring():
    suite = api().MABSuite.from_directory(release_root())
    scorer = suite.scorer()
    cases = [c for c in suite.cases if c.source == 'longmemeval_s*']
    assert len(cases) == 300
    for case in cases:
        requests = scorer.judge_requests(case, 'candidate')
        assert len(requests) == 1
        assert case.sample.question in requests[0].prompt
        assert str(list(case.answers)) in requests[0].prompt
    case = next(c for c in cases if c.metadata['question_types'][c.question_index] == 'temporal-reasoning')
    request, = scorer.judge_requests(case, 'candidate')
    assert 'off-by-one' in request.prompt
    score = scorer.score(case, 'candidate', judgments=[judged(request, 'yes')])
    assert score.metric == 'llm_judge_accuracy'
    assert score.value == 1
    assert score.details['judge_model'] == 'gpt-4o'
    with pytest.raises(ValueError, match='binding'):
        scorer.score(case, 'different answer', judgments=[judged(request, 'yes')])
    with pytest.raises(ValueError, match='invalid'):
        scorer.score(case, 'candidate', judgments=[judged(request, 'probably yes')])
    with pytest.raises(ValueError, match='requires'):
        scorer.score(case, 'candidate')


def test_summary_native_three_prompt_fluency_gated_f1_and_invalid_gates():
    suite = api().MABSuite.from_directory(release_root())
    scorer = suite.scorer()
    cases = [c for c in suite.cases if c.source == 'infbench_sum_eng_shots2']
    assert len(cases) == 100
    for case in cases:
        requests = scorer.judge_requests(case, 'candidate summary')
        assert [r.role for r in requests] == ['fluency', 'recall', 'precision']
        assert case.answers[0] in requests[2].prompt
        assert case.metadata['keypoints'][0] in requests[1].prompt
    case = cases[0]
    requests = scorer.judge_requests(case, 'candidate summary')
    responses = [judged(requests[0], '{"fluency":1}'),
                 judged(requests[1], '{"recall":25,"supported_key_points":[]}'),
                 judged(requests[2], '{"precision":1,"sentence_count":2}')]
    score = scorer.score(case, 'candidate summary', judgments=responses)
    assert score.metric == 'llm_summary_f1'
    assert score.value == pytest.approx(2 / 3)
    responses[0] = judged(requests[0], '{"fluency":0}')
    assert scorer.score(case, 'candidate summary', judgments=responses).value == 0
    responses[1] = judged(requests[1], '{"recall":9999}')
    with pytest.raises(ValueError, match='invalid'):
        scorer.score(case, 'candidate summary', judgments=responses)
    with pytest.raises(ValueError, match='requires'):
        scorer.score(case, 'candidate summary', judgments=responses[:1])


def test_source_specific_reporting_preserves_missing_denominators_without_pooling():
    suite = api().MABSuite.from_directory(release_root())
    scorer = suite.scorer()
    case = next(c for c in suite.cases if c.source == 'factconsolidation_mh_6k')
    score = scorer.score(case, case.answers[0])
    report = suite.summarize([score])
    assert 'overall' not in report
    source = report['sources'][case.source]
    assert source['metric'] == 'substring_exact_match'
    assert source['planned'] == 100
    assert source['completed'] == 1
    assert source['missing'] == 99
    assert source['completed_only_mean'] == 1
    assert len(report['sources']) == 22
    with pytest.raises(ValueError, match='duplicate'):
        suite.summarize([score, score])


def test_admission_cli_writes_real_inventory_without_inference(tmp_path):
    import json
    output = tmp_path / 'admission.json'
    assert api().main([str(release_root()), '--output', str(output)]) == 0
    report = json.loads(output.read_bytes())
    assert report['inventory']['question_occurrences'] == 3671
    assert report['scoring']['longmemeval_s*']['metric'] == 'llm_judge_accuracy'
    assert report['scoring']['infbench_sum_eng_shots2']['judge_calls_per_question'] == 3
    assert report['scoring']['recsys_redial_full']['metric'] == 'recsys_recall@5'
    assert report['receipt']['files']['upstream/README.md']
    assert report['inference_performed'] is False


def test_source_categories_are_distinct_in_sample_seam():
    m = api()
    suite = m.MABSuite.from_rows({'Accurate_Retrieval': [row('ruler_qa1_197K'), row('ruler_qa2_421K')]})
    assert len({c.sample.category for c in suite.cases}) == 2


def test_rows_preserve_occurrences_aliases_raw_metadata_and_shared_history():
    m = api()
    raw = row()
    suite = m.MABSuite.from_rows({"Accurate_Retrieval": [raw, raw]})
    cases = suite.cases
    assert len(cases) == 4
    assert len({c.sample.id for c in cases}) == 4
    assert [c.source_id for c in cases] == ["same"] * 4
    assert cases[0].answers == ("France", "France", "FR")
    assert cases[0].sample.question == "Q"
    assert cases[0].sample.gold == '["France", "France", "FR"]'
    assert [h.row_index for h in suite.histories] == [0, 1]
    assert suite.histories[0].context == raw["context"]
    assert suite.histories[0].metadata == raw["metadata"]
    assert suite.histories[0].context_sha256 == suite.histories[1].context_sha256
    assert suite.histories[0].id != suite.histories[1].id
    assert [c.sample.conv_id for c in cases] == [suite.histories[0].id] * 2 + [suite.histories[1].id] * 2
