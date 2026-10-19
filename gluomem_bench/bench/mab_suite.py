"""Lossless official MemoryAgentBench admission, not a campaign quality claim.

Histories are parquet row occurrences, NOT unique contexts. Source QA IDs are
not unique in the release; occurrence-qualified Sample IDs prevent data loss.
Gold/metadata never enter document ingestion. See ``admission`` for gates.

Usage::

    suite = MABSuite.from_directory(external_snapshot)
    samples = suite.reader_samples(agent='rag')  # exact upstream query templates
    histories = suite.conversations()             # explicitly adapted ingestion
    scorer = suite.scorer()
    result = scorer.score(suite.cases[0], saved_answer)
    requests = scorer.judge_requests(case, saved_answer)  # no inference
    result = scorer.score(case, saved_answer, judgments=saved_bound_responses)

``from_rows`` is an unpinned parsing/testing seam, not release admission.
``load`` exposes raw questions; use ``reader_samples`` for inference. Native
objective scoring additionally needs numpy, nltk, rouge-score and editdistance;
parquet admission needs pyarrow. No packages are installed by this module.
The upstream MIT license is retained in the external verified snapshot.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any
from pathlib import Path
from collections import Counter, defaultdict

CODE_REVISION = '538026089d1a8a8eff05121d0db89b388f360eba'
DATA_REVISION = '7ea066982b140a19337e17e60d45d4076e042faf'
FAMILIES = ('Accurate_Retrieval', 'Test_Time_Learning',
            'Long_Range_Understanding', 'Conflict_Resolution')
PINNED_FILES = {
    'data/Accurate_Retrieval-00000-of-00001.parquet': '56c3cd80fb6731a3e53cd1a6be3148f54df60ff2d290ee50e28f8acebf9655c1',
    'data/Conflict_Resolution-00000-of-00001.parquet': '24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45',
    'data/Long_Range_Understanding-00000-of-00001.parquet': '5ab175461954db67770d4a4cb69e569b513ebb96aceb9ee79b57f67488bcd539',
    'data/Test_Time_Learning-00000-of-00001.parquet': '5338753be48f925d03318eed66117286e3489025fabe050a547bd086cd7d79c0',
    'entity2id.json': '63353aca481bc9558b502f91cb98f6fa26438796fdd7e0bc06b5a1532126e8b5',
    'README.md': '5a9fd0dc2fbab5beb081eae40f24964ad1e47da75b9a7a01f0007791845b1580',
    'upstream/utils/templates.py': '07f323f7e1569a4cfc9c9a979059e2c25818b312c8aa383777ea61d2fbff9214',
    'upstream/utils/eval_other_utils.py': 'd23a5ba9f2750229315cd308463ae99aef9ff36ed0859b69d4851f804976b815',
    'upstream/llm_based_eval/longmem_qa_evaluate.py': '5c98e9ad6db0d3e90993db3e01019a2862337479987dc9b81d5a48858461aeb2',
    'upstream/llm_based_eval/summarization_evaluate.py': '0139db690c4537c8f6ca5fe766ab60caf16050d172412f18ad0535577e708801',
    'upstream/LICENSE': '6d4d21d2fffa4511768cfe93acb8d652f90e24c12f8c173da378acc8dbcf9fc4',
    'upstream/README.md': '366fc1e90b5a4cacdba10552f89f37a0084bcee2208638ec631784206b009133',
    'upstream/conversation_creator.py': 'a0d293a73341e46e30be209c0c58bca88428334705ef076eed16c99d8a7a4af7',
    'upstream/utils/eval_data_utils.py': '0306c3316c955c465d62a34ab311bcabb23a5868d1aa4d551b011c1c8251794f',
}

SOURCE_FAMILIES = {
    **{s: FAMILIES[0] for s in ('ruler_qa1_197K', 'ruler_qa2_421K',
       'eventqa_full', 'eventqa_65536', 'eventqa_131072', 'longmemeval_s*')},
    **{s: FAMILIES[1] for s in ('recsys_redial_full', 'icl_banking77_5900shot_balance',
       'icl_clinic150_7050shot_balance', 'icl_nlu_8296shot_balance',
       'icl_trec_coarse_6600shot_balance', 'icl_trec_fine_6400shot_balance')},
    **{s: FAMILIES[2] for s in ('infbench_sum_eng_shots2', 'detective_qa')},
    **{f'factconsolidation_{hop}_{size}': FAMILIES[3]
       for hop in ('mh', 'sh') for size in ('6k', '32k', '64k', '262k')},
}


def _validate_row(family, row):
    if family not in FAMILIES:
        raise ValueError(f'unknown MAB family: {family}')
    if not isinstance(row, dict) or not isinstance(row.get('metadata'), dict):
        raise ValueError('MAB row and metadata mappings required')
    meta = row['metadata']
    if SOURCE_FAMILIES.get(meta.get('source')) != family:
        raise ValueError('unknown source or wrong source family')
    if not isinstance(row.get('context'), str) or not row['context']:
        raise ValueError('nonempty context required')
    for field in ('questions', 'answers'):
        if not isinstance(row.get(field), list) or not row[field]:
            raise ValueError(f'nonempty {field} list required')
    n = len(row['questions'])
    if len(row['answers']) != n or len(meta.get('qa_pair_ids') or []) != n:
        raise ValueError('parallel question/answer/source ID arrays required')
    for q, aliases, qid in zip(row['questions'], row['answers'], meta['qa_pair_ids'], strict=True):
        if not isinstance(q, str) or not q or not isinstance(qid, str) or not qid:
            raise ValueError('nonempty string question and source ID required')
        if not isinstance(aliases, list) or not aliases or any(not isinstance(a, str) or not a for a in aliases):
            raise ValueError('nonempty string aliases required')
    for field in ('question_dates', 'question_ids', 'question_types', 'previous_events', 'haystack_sessions'):
        if meta.get(field) is not None and (not isinstance(meta[field], list) or len(meta[field]) != n):
            raise ValueError(f'parallel metadata {field} required')

from gluomem_bench.bench.types import Sample


@dataclass(frozen=True)
class MABHistory:
    id: str
    family: str
    source: str
    row_index: int
    context: str
    context_sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MABCase:
    sample: Sample
    family: str
    source: str
    row_index: int
    question_index: int
    source_id: str
    answers: tuple[str, ...]
    metadata: dict[str, Any]


class MABSuite:
    def __init__(self, histories, cases):
        self.histories = tuple(histories)
        self.cases = tuple(cases)
        self.root: Path | None = None
        self.receipt: dict[str, Any] = {}
        self._templates: dict[str, Any] | None = None

    @classmethod
    def from_rows(cls, splits):
        histories, cases = [], []
        for family, rows in splits.items():
            for ri, row in enumerate(rows):
                _validate_row(family, row)
                meta = copy.deepcopy(row['metadata'])
                source = meta['source']
                # Generated storage IDs must also be valid Windows filenames.
                # Native source QA IDs remain unchanged in MABCase.source_id.
                hid = f'mab-{family}-row-{ri}'
                histories.append(MABHistory(hid, family, source, ri, row['context'],
                    hashlib.sha256(row['context'].encode('utf-8')).hexdigest(), meta))
                for qi, (question, answers, qid) in enumerate(zip(row['questions'],
                        row['answers'], meta['qa_pair_ids'], strict=True)):
                    sample = Sample(f'{hid}:qa:{qi}', hid, question,
                                    json.dumps(answers, ensure_ascii=False),
                                    tuple(SOURCE_FAMILIES).index(source) + 1, source)
                    cases.append(MABCase(sample, family, source, ri, qi, qid,
                                         tuple(answers), copy.deepcopy(meta)))
        return cls(histories, cases)

    @classmethod
    def from_directory(cls, root):
        """Verify actual pinned code/data bytes before reading all parquet rows."""
        root = Path(root)
        actual = {}
        for relative, expected in PINNED_FILES.items():
            path = root / relative
            if not path.is_file():
                raise ValueError(f'missing pinned file: {relative}')
            actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual[relative] != expected:
                raise ValueError(f'pinned SHA256 mismatch: {relative}')
        import pyarrow.parquet as pq
        suite = cls.from_rows({family: pq.read_table(root / 'data' /
            f'{family}-00000-of-00001.parquet').to_pylist() for family in FAMILIES})
        suite.root = root
        suite.receipt = {'data_revision': DATA_REVISION, 'code_revision': CODE_REVISION,
                         'files': actual, 'dataset_license': 'MIT (dataset card)',
                         'code_license': 'MIT, Copyright (c) 2026 Yuanzhe Hu',
                         'component_license_gate': 'derived corpus rights not independently audited'}
        return suite

    def inventory(self):
        sources = {}
        for history in self.histories:
            sources.setdefault(history.source, {'family': history.family, 'histories': 0,
                'questions': 0, 'context_hashes': []})
            sources[history.source]['histories'] += 1
            sources[history.source]['context_hashes'].append(history.context_sha256)
        for case in self.cases:
            sources[case.source]['questions'] += 1
        overlaps = defaultdict(list)
        for history in self.histories:
            overlaps[history.context_sha256].append(history.id)
        duplicates = Counter(case.source_id for case in self.cases)
        return {'history_occurrences': len(self.histories), 'question_occurrences': len(self.cases),
                'unique_contexts': len(overlaps), 'unique_source_qa_ids': len(duplicates),
                'shared_histories': {k: v for k, v in overlaps.items() if len(v) > 1},
                'duplicate_source_qa_ids': {k: v for k, v in duplicates.items() if v > 1},
                'sources': sources}

    def _source_bytes(self, relative):
        if self.root is None:
            raise ValueError('pinned from_directory suite required for native source execution')
        data = (self.root / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != PINNED_FILES[relative]:
            raise ValueError(f'pinned SHA256 mismatch: {relative}')
        return data

    def templates(self):
        """Use the byte-pinned upstream module; no remote import or regeneration."""
        if self._templates is None:
            namespace = {}
            exec(compile(self._source_bytes('upstream/utils/templates.py'),
                         'pinned_mab_templates', 'exec'), namespace)
            self._templates = namespace
        return self._templates

    def reader_question(self, case, *, agent='rag'):
        template = self.templates()['get_template'](case.source, 'query', agent)
        return template.format(question=case.sample.question)

    def reader_samples(self, *, agent='rag'):
        """Reader-ready Sample clones; raw questions remain in cases/load().

        MAB LongMemEval questions already embed their query clock. Do not use
        Sample.question_date to prepend it again. Exact native templates have
        suffixes, so Sample.task_instructions alone cannot represent them.
        """
        from dataclasses import replace
        return [replace(c.sample, question=self.reader_question(c, agent=agent)) for c in self.cases]

    def conversations(self, *, chunk_chars=4096):
        """Adapted lossless document chunks, NOT upstream token/sentence chunks.

        Epoch dates are placeholders only, not a claim of document chronology.
        Every row occurrence remains a separate conversation, even shared text.
        """
        if type(chunk_chars) is not int or chunk_chars < 1:
            raise ValueError('positive integer chunk_chars required')
        from datetime import datetime
        from gluomem_bench.ingest import Conversation, Session, Turn
        return [Conversation(conv_id=h.id, speakers=('document', 'document'),
            sessions=[Session(index=1, date_time=datetime(1970, 1, 1), turns=[
                Turn(speaker='document', text=h.context[i:i + chunk_chars],
                     dia_id=f'{h.id}:chunk:{i // chunk_chars}')
                for i in range(0, len(h.context), chunk_chars)])]) for h in self.histories]

    def admission(self):
        return {'pinned_bytes_verified': bool(self.receipt), 'native_ingestion': False,
                'missing_gates': [
                    'freeze system/reader/judge models, generation limits and shared budgets',
                    'admit full prompts with served tokenizer and output reserve',
                    'document ingestion is adapted lossless characters, not native incremental token chunks',
                    'verify production extraction preserves document order/serial numbers and coverage',
                    'freeze shared-history clustering and duplicate-ID occurrence selection',
                    'qualify isolated native scorer dependencies: numpy, nltk, rouge-score, editdistance',
                    'Recsys upstream set-order edit-distance ties require frozen Python hash seed/candidate order',
                    'substituted judge models are adapted-judge results, not upstream-model reproductions',
                    'review derived corpus licensing beyond declared MIT dataset/code license',
                    'complete predictions and independent final evidence review before quality claims']}

    def summarize(self, scores):
        """Per-source native metrics only; never pool different tasks or judges.

        All released cases remain in planned denominators. Missing cases are
        reported, NOT filled with zeros or silently removed from coverage.
        """
        import math
        cases = {c.sample.id: c for c in self.cases}
        grouped = defaultdict(list)
        seen = set()
        for score in scores:
            if score.case_id in seen:
                raise ValueError('duplicate score occurrence')
            seen.add(score.case_id)
            case = cases.get(score.case_id)
            if case is None or score.source != case.source or score.metric != primary_metric(case.source):
                raise ValueError('score/source/native metric binding mismatch')
            if not math.isfinite(score.value) or not 0 <= score.value <= 1:
                raise ValueError('invalid score range')
            grouped[score.source].append(score)
        sources = {}
        for source, counts in self.inventory()['sources'].items():
            values = grouped[source]
            judge_models = {tuple(v.details.get('judge_models', [v.details.get('judge_model', 'none')])) for v in values}
            if len(judge_models) > 1:
                raise ValueError('incompatible judge models must not aggregate')
            planned = [c.sample.id for c in self.cases if c.source == source]
            sources[source] = {'metric': primary_metric(source), 'planned': counts['questions'],
                'completed': len(values), 'missing': len(planned) - len(values),
                'missing_ids': [identifier for identifier in planned if identifier not in seen],
                'completed_only_mean': sum(v.value for v in values) / len(values) if values else None,
                'judge_models': list(next(iter(judge_models))) if judge_models else []}
        return {'sources': sources}

    def scorer(self):
        return MABScorer(self)

    def load(self):
        """Raw Sample seam; use reader_samples for exact official reader queries."""
        return [case.sample for case in self.cases]


@dataclass(frozen=True)
class NativeScore:
    case_id: str
    source: str
    metric: str
    value: float
    metrics: dict[str, float]
    details: dict[str, Any]
    scorer_revision: str = CODE_REVISION


def primary_metric(source):
    if source not in SOURCE_FAMILIES:
        raise ValueError(f'unknown MAB source: {source}')
    if source == 'longmemeval_s*':
        return 'llm_judge_accuracy'
    if source == 'infbench_sum_eng_shots2':
        return 'llm_summary_f1'
    if source.startswith('recsys_'):
        return 'recsys_recall@5'
    if source.startswith(('icl_', 'detective_')):
        return 'exact_match'
    return 'substring_exact_match'


@dataclass(frozen=True)
class JudgeRequest:
    case_id: str
    role: str
    prompt: str
    prediction_sha256: str
    prompt_sha256: str
    reference_model: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class JudgeResponse:
    case_id: str
    role: str
    prediction_sha256: str
    prompt_sha256: str
    model: str
    output: str


def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class MABScorer:
    """Execute pinned official post_process, without upstream import side effects.

    Native metrics are never substituted with generic F1, ROUGE or pooled QA
    accuracy. Optional scorer packages are caller-installed; missing packages
    fail closed. No model clients are constructed and no inference is run.
    """
    gates = {'recsys_redial_full': 'set-order edit-distance tie breaking',
             'longmemeval_s*': 'requires saved task-specific judge responses and frozen judge identity',
             'infbench_sum_eng_shots2': 'requires saved fluency/recall/precision judge responses'}

    def __init__(self, suite):
        self.suite = suite
        self._upstream = None

    @property
    def upstream(self):
        if self._upstream is None:
            import ast
            import io
            import os
            import re
            import string
            try:
                import numpy as np
                from editdistance import eval as edit_distance
                from rouge_score import rouge_scorer
            except ImportError as exc:
                raise ValueError('native scorer requires numpy, nltk, rouge-score and editdistance; install in an isolated scorer environment') from exc

            def pinned_open(path, *args, **kwargs):
                expected = './processed_data/Recsys_Redial/entity2id.json'
                if str(path).replace('\\', '/') != expected:
                    raise ValueError(f'unexpected upstream scorer open: {path}')
                return io.StringIO(self.suite._source_bytes('entity2id.json').decode('utf-8'))

            namespace = {'np': np, 'edit_distance': edit_distance, 'os': os, 're': re,
                'string': string, 'json': json, 'Counter': Counter, 'open': pinned_open,
                'rouge_scorer_instance': rouge_scorer.RougeScorer(['rougeL', 'rougeLsum'], use_stemmer=True)}
            relative = 'upstream/utils/eval_other_utils.py'
            tree = ast.parse(self.suite._source_bytes(relative))
            # Function definitions only: no imports, network, NLTK downloads or
            # top-level mutable setup. Called functions retain original bodies.
            tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            exec(compile(tree, relative, 'exec'), namespace)
            self._upstream = namespace
        return self._upstream

    def _summary_namespace(self):
        import ast
        relative = 'upstream/llm_based_eval/summarization_evaluate.py'
        tree = ast.parse(self.suite._source_bytes(relative))
        names = {'fluency_prompt_book', 'recall_prompt_book', 'precision_prompt_book'}
        namespace = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in names:
                    namespace[target.id] = ast.literal_eval(node.value)
        return namespace

    def judge_requests(self, case, prediction):
        """Prepare native prompts only. Caller owns transport/model/budget gates."""
        if case.source == 'infbench_sum_eng_shots2':
            namespace = self._summary_namespace()
            points = case.metadata.get('keypoints')
            if not points:
                raise ValueError('summary requires native keypoints')
            fields = {'text': prediction.strip(), 'summary': prediction.strip(),
                'expert_summary': case.answers[0],
                'keypoints': '\n'.join(f'{i+1}. {point}' for i, point in enumerate(points))}
            result = []
            for role in ('fluency', 'recall', 'precision'):
                prompt = namespace[f'{role}_prompt_book'].format(**fields)
                result.append(JudgeRequest(case.sample.id, role, prompt, _digest(prediction),
                    _digest(prompt), 'gpt-4o-2024-05-13', 0.1, 4096))
            return tuple(result)
        if case.source != 'longmemeval_s*':
            return ()
        import ast
        relative = 'upstream/llm_based_eval/longmem_qa_evaluate.py'
        tree = ast.parse(self.suite._source_bytes(relative))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'get_anscheck_prompt']
        namespace = {}
        exec(compile(tree, relative, 'exec'), namespace)
        index = case.question_index
        prompt = namespace['get_anscheck_prompt'](
            case.metadata['question_types'][index], case.sample.question,
            list(case.answers), prediction,
            abstention='_abs' in case.metadata['question_ids'][index])
        return (JudgeRequest(case.sample.id, 'answer', prompt, _digest(prediction),
                             _digest(prompt), 'gpt-4o', 0, 10),)

    def _bound_judgments(self, case, prediction, judgments):
        requests = self.judge_requests(case, prediction)
        responses = {j.role: j for j in judgments}
        if not requests or len(responses) != len(judgments) or set(responses) != {r.role for r in requests}:
            raise ValueError('requires complete unique saved native judge responses')
        for request in requests:
            response = responses[request.role]
            if (response.case_id != request.case_id
                    or response.prediction_sha256 != request.prediction_sha256
                    or response.prompt_sha256 != request.prompt_sha256
                    or not response.model or not isinstance(response.output, str)):
                raise ValueError('judge response binding mismatch')
        return responses

    def score(self, case, prediction, *, judgments=()):
        if not isinstance(prediction, str):
            raise ValueError('complete string prediction required; failures are not quality scores')
        metric = primary_metric(case.source)
        if metric == 'llm_judge_accuracy':
            responses = self._bound_judgments(case, prediction, judgments)
            response = responses['answer']
            if response.output.strip().lower() not in ('yes', 'no'):
                raise ValueError('invalid native judge verdict; expected yes or no')
            value = float('yes' in response.output.strip().lower())
            return NativeScore(case.sample.id, case.source, metric, value, {metric: value},
                {'judge_model': response.model, 'judge_output': response.output,
                 'judge_identity': 'upstream-reference' if response.model == 'gpt-4o' else 'adapted-judge'})
        if metric == 'llm_summary_f1':
            import ast
            import re
            responses = self._bound_judgments(case, prediction, judgments)
            relative = 'upstream/llm_based_eval/summarization_evaluate.py'
            tree = ast.parse(self.suite._source_bytes(relative))
            tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'parse_json']
            namespace = {'json': json, 're': re}
            exec(compile(tree, relative, 'exec'), namespace)
            try:
                parsed = {role: namespace['parse_json'](response.output) for role, response in responses.items()}
                f, r, p = parsed['fluency'], parsed['recall'], parsed['precision']
                counts = (f['fluency'], r['recall'], p['precision'], p['sentence_count'])
                if any(type(value) is not int or value < 0 for value in counts):
                    raise ValueError('invalid judge counters')
                if f['fluency'] not in (0, 1) or r['recall'] > len(case.metadata['keypoints']) or p['precision'] > p['sentence_count']:
                    raise ValueError('invalid judge counters')
                rec = r['recall'] / len(case.metadata['keypoints']) if case.metadata['keypoints'] else 0
                prec = p['precision'] / p['sentence_count'] if p['sentence_count'] > 0 else 0
                value = f['fluency'] * 2 * (rec * prec) / (rec + prec) if rec + prec > 0 else 0
            except (KeyError, TypeError, IndexError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError('invalid native summary judge response') from exc
            models = {j.model for j in responses.values()}
            return NativeScore(case.sample.id, case.source, metric, value,
                {metric: value, 'fluency': float(f['fluency']), 'recall': rec, 'precision': prec},
                {'judge_models': sorted(models), 'judgments': parsed,
                 'judge_outputs': {role: response.output for role, response in responses.items()},
                 'judge_identity': 'upstream-reference' if models == {'gpt-4o-2024-05-13'} else 'adapted-judge'})
        metrics, details = self.upstream['post_process'](
            {'output': prediction}, list(case.answers), {'sub_dataset': case.source})
        return NativeScore(case.sample.id, case.source, metric, float(metrics[metric]),
                           {key: float(value) for key, value in metrics.items()}, details)


def main(argv=None):
    """Executable offline admission: python -m ...mab_suite ROOT --output FILE."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='external snapshot containing data/ and upstream/')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    suite = MABSuite.from_directory(args.root)
    scoring = {}
    for source in suite.inventory()['sources']:
        metric = primary_metric(source)
        scoring[source] = {'metric': metric,
            'judge_calls_per_question': 3 if metric == 'llm_summary_f1' else int(metric == 'llm_judge_accuracy'),
            'gate': MABScorer.gates.get(source),
            'reader_template_sha256': _digest(suite.templates()['get_template'](source, 'query', 'rag'))}
    report = {'receipt': suite.receipt, 'inventory': suite.inventory(),
              'admission': suite.admission(), 'scoring': scoring,
              'inference_performed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'output': str(args.output), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'histories': len(suite.histories), 'questions': len(suite.cases)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
