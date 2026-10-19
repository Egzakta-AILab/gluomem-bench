"""Executable MAB document seam. Authored loopback tests are not quality evidence.

``run_history`` is a low-level, one-history/system worker; it requires a shared
JobContext (call/deadline/token lease). ``verify_history`` trusts only the root
status written after closed-store hashing and native scoring verification.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from gluomem_bench.bench.error_privacy import safe_error
from gluomem_bench.bench.rack import FatalRackError, RackClient, SharedRackError

PROTOCOL = 'mab-native-task-document-campaign@v1'
FULL_MODE = 'full-campaign-v1'
CONTINUATION_MODE = 'full-continuation-v1'
REUSE = 'reuse_old_complete_conditional_binding'
QUALITY_FAILED = 'retain_quality_failure_no_rerun'
INFRA_HELD = 'hold_infra_retry_unresolved_origin'
NEVER_DISPATCHED = 'remaining_cancelled_never_dispatched'
NATIVE_UNRUN = 'remaining_native_graph_unrun'
EVIDENCE_TOKEN_BUDGET = 4096


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _reader_evidence(client, question, evidence, *, system_prompt, system):
    """Count complete reader messages; keep only a ranked whole-line prefix."""
    original = evidence
    counts = []
    def count(text):
        messages = _messages(question, text, system_prompt=system_prompt)
        tokens = client.prompt_token_count(messages)
        if type(tokens) is not int or tokens < 0:
            raise FatalRackError('invalid served evidence token count')
        counts.append({'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                       'messages_sha256': _digest(messages), 'tokens': tokens})
        return tokens
    original_tokens = kept_tokens = count(original)
    if system != 'full_context' and original_tokens > EVIDENCE_TOKEN_BUDGET:
        evidence = ''
        kept_tokens = count(evidence)
        if kept_tokens > EVIDENCE_TOKEN_BUDGET:
            raise FatalRackError('evidence ceiling smaller than complete reader-message overhead')
        for line in original.splitlines(keepends=True):
            tokens = count(evidence + line)
            if tokens > EVIDENCE_TOKEN_BUDGET:
                break
            evidence, kept_tokens = evidence + line, tokens
    return evidence, {
        'original_text': original, 'kept_text': evidence,
        'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'kept_sha256': hashlib.sha256(evidence.encode()).hexdigest(),
        'original_tokens': original_tokens, 'kept_tokens': kept_tokens,
        'original_lines': len(original.splitlines()), 'kept_lines': len(evidence.splitlines()),
        'budget': EVIDENCE_TOKEN_BUDGET, 'exempt': system == 'full_context',
        'counts': counts,
    }


_EXECUTION_AUTHORITY = object()
_clock = time.monotonic
SCORER_FILES = ('entity2id.json', 'upstream/utils/templates.py',
                'upstream/utils/eval_other_utils.py',
                'upstream/llm_based_eval/longmem_qa_evaluate.py',
                'upstream/llm_based_eval/summarization_evaluate.py')


def _write(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8'))


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _publish_campaign(root, result, budget):
    """Only the exact verified staged bytes may acquire root authority."""
    pending = root / 'root-status.pending.json'
    target = root / 'root-status.json'
    try:
        budget.check()
        _write(pending, result)
        material = pending.read_bytes()
        staged = json.loads(material)
        if staged != result:
            raise ValueError('campaign staged root readback differs')
        budget.check()
        _verify_campaign_payload(root, staged)
        if pending.read_bytes() != material:
            raise ValueError('campaign staged root changed during verification')
        budget.check()
        pending.replace(target)
        if target.read_bytes() != material:
            raise ValueError('campaign final root readback differs')
        budget.check()
        return staged
    except Exception as exc:
        spend = budget.snapshot()
        exc.__dict__['generation_calls'] = spend['calls']
        for path in (pending, target):
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup:
                exc.add_note(f'root authority revocation failed: {safe_error(cleanup)}')
        try:
            _write(root / 'failure.json', {'complete': False, 'generation_calls': spend['calls'],
                'accounting': spend, 'error': safe_error(exc)})
        except Exception as persistence:
            exc.add_note(
                f'failure receipt unavailable: {safe_error(persistence)}; '
                f'accounting={spend}'
            )
        raise


def _pinned_record(record):
    import re
    path = Path(record['path'])
    digest = record['sha256']
    if (not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None
            or not path.is_absolute() or _hash(path) != digest):
        raise ValueError('continuation exact source/root/archive binding differs')
    return path


def continuation_plan(spec):
    """Read-only evidence admission; failed rows NEVER redispatch.

    Pass ORIGINAL total ceilings to the shared budget with retained concurrent
    wall time. This validates bindings, but grants no semantic reuse authority.
    """
    import zipfile
    if spec.get('campaign_mode') != CONTINUATION_MODE:
        raise ValueError('explicit continuation mode required')
    bindings = spec['continuation']
    original = json.loads(_pinned_record(bindings['original_spec']).read_bytes())
    original_root_path = _pinned_record(bindings['original_root'])
    old = json.loads(original_root_path.read_bytes())
    archive_path = _pinned_record(bindings['original_archive'])
    if (original.get('campaign_mode') != FULL_MODE or old.get('campaign_mode') != FULL_MODE
            or original.get('execution_kind') != PROTOCOL or old.get('protocol') != PROTOCOL
            or spec['case_ids'] != original['case_ids']
            or spec['case_manifest_sha256'] != original['case_manifest_sha256']
            or spec['policy']['max_calls'] != original['policy']['max_calls']
            or spec['policy']['max_seconds'] != original['policy']['max_seconds']):
        raise ValueError('continuation must retain full original scope and ORIGINAL total caps')
    prior_systems = original['systems']
    if spec['systems'] != prior_systems + (['cognee'] if 'cognee' not in prior_systems else []):
        raise ValueError('continuation must retain original ordered methods plus native unrun')
    identity = lambda r: {k: r[k] for k in ('job_id', 'system', 'history_id', 'case_ids')}
    rows = spec['jobs']
    original_rows = [r for r in rows if r['system'] in prior_systems]
    if ([identity(r) for r in original_rows] != [identity(r) for r in original['jobs']]
            or [identity(r) for r in old['jobs']] != [identity(r) for r in original['jobs']]
            or len({r['job_id'] for r in rows}) != len(rows)):
        raise ValueError('continuation original matrix IDs/case occurrence row order differs')
    old_rows = {r['job_id']: r for r in old['jobs']}
    progress_path = original_root_path.parent / 'coordination/progress.json'
    progress = json.loads(progress_path.read_bytes())
    if [r['job_id'] for r in progress['jobs']] != [r['job_id'] for r in old['jobs']]:
        raise ValueError('original coordination row order differs')
    dispatch, retained, reuse = [], [], []
    for record in rows:
        disposition = record['disposition']
        previous = old_rows.get(record['job_id'])
        if previous is None:
            if (record['system'] != 'cognee' or disposition != NATIVE_UNRUN
                    or record.get('prior_generation_attempts', 0) != 0):
                raise ValueError('new native rows must remain unrun until wrapper qualification')
            retained.append(record['job_id'])
            continue
        calls = previous['generation_calls']
        previous_path = original_root_path.parent / previous['output']
        if (record.get('prior_generation_attempts') != calls
                or Path(record['original_output']).resolve() != previous_path.resolve()
                or not previous_path.resolve().is_relative_to(original_root_path.parent.resolve())):
            raise ValueError('retained output/prior spend binding differs')
        if previous['status'] == 'complete':
            if disposition != REUSE:
                raise ValueError('old complete rows require conditional reuse, not redispatch')
            reuse.append(record['job_id'])
        elif previous['status'] == 'failed':
            if disposition not in (QUALITY_FAILED, INFRA_HELD):
                raise ValueError('failed old rows cannot be promoted/retried')
        elif previous['status'] == 'unrun':
            if disposition != NEVER_DISPATCHED or calls != 0:
                raise ValueError('only genuinely never-dispatched rows may run')
            dispatch.append(record['job_id'])
            continue
        else:
            raise ValueError('unknown original authoritative status')
        retained.append(record['job_id'])
    prior_calls = sum(r['generation_calls'] for r in old['jobs'])
    if (prior_calls != old['generation_calls'] or progress['accounting']['calls'] != prior_calls
            or sum(r.get('calls', 0) for r in progress['jobs']) != prior_calls):
        raise ValueError('original retained call reconciliation differs')
    prior_elapsed = progress['accounting']['elapsed_seconds']
    old_root = original_root_path.parent
    inventory = old['artifact_sha256']
    actual = {p.relative_to(old_root).as_posix() for p in old_root.rglob('*')
              if p.is_file() and p != original_root_path}
    if {Path(p).as_posix() for p in inventory} != actual:
        raise ValueError('original complete evidence inventory differs')
    for name, digest in inventory.items():
        path = old_root / name
        if not path.resolve().is_relative_to(old_root.resolve()) or _hash(path) != digest:
            raise ValueError('original retained artifact bytes differ')
    if _hash(old_root / 'execution-spec.json') != bindings['original_spec']['sha256']:
        raise ValueError('original root/spec mismatch')
    pins = {r['path']: r['sha256'] for r in original['files']}
    with zipfile.ZipFile(archive_path) as archive:
        sources = json.loads(archive.read('producer-bindings.json'))
        if len({r['path'] for r in sources}) != len(sources):
            raise ValueError('duplicate archived producer bindings')
        archived = {}
        for item in sources:
            if (pins.get(item['path']) != item['sha256']
                    or hashlib.sha256(archive.read(item['member'])).hexdigest() != item['sha256']):
                raise ValueError('archived producer semantics/source bytes differ')
            archived[item['path']] = item['sha256']
        for item in original['files']:
            if item['path'] not in archived and _hash(item['path']) != item['sha256']:
                raise ValueError('original nonarchived dataset/source bytes differ')
    return {'original_spec': original, 'original_root': old, 'original_jobs': original_rows,
            'dispatch_job_ids': dispatch, 'retained': retained, 'reuse_job_ids': reuse,
            'prior_calls': prior_calls, 'prior_elapsed_seconds': prior_elapsed}


def continuation_semantics(spec):
    """Non-circular review target; neither approvals nor bridge hashes self-bind."""
    keys = ('campaign_mode', 'output_root', 'case_ids', 'case_manifest_sha256', 'systems',
            'profile', 'embedding_url', 'embedding_recipe', 'chunk_chars', 'top_k',
            'evidence_token_budget', 'judge_qualification', 'policy', 'jobs', 'files')
    return _digest({key: spec[key] for key in keys})


def _verify_reuse(spec, approval, plan, planned, suite):
    """Rehash and rescore old leaves on disposable copies, never open old DBs."""
    import os
    import shutil
    import tempfile
    pin = spec['continuation'].get('semantic_bridge')
    if (not pin or approval.get('approved_gates', {}).get('mab_continuation_bridge') != pin['sha256']):
        raise ValueError('exact independently approved continuation semantic bridge required')
    bridge = json.loads(_pinned_record(pin).read_bytes())
    original_bindings = {k: spec['continuation'][k] for k in
                         ('original_spec', 'original_root', 'original_archive')}
    if (bridge.get('protocol') != 'mab-readonly-semantic-bridge@v1'
            or bridge.get('original_bindings') != original_bindings
            or bridge.get('candidate_semantics_sha256') != continuation_semantics(spec)
            or bridge.get('current_wrapper_sha256') != _hash(__file__)
            or bridge.get('reuse_job_ids') != plan['reuse_job_ids']):
        raise ValueError('continuation bridge exact producer/scope/old evidence binding differs')
    old_spec = plan['original_spec']
    old_pins = {r['path']: r['sha256'] for r in old_spec['files'] if r['path'].endswith('.py')}
    new_pins = {r['path']: r['sha256'] for r in spec['files']}
    semantic = bridge.get('producer_semantics', [])
    if len({r['original_path'] for r in semantic}) != len(semantic) or {
            r['original_path']: r['original_sha256'] for r in semantic} != old_pins:
        raise ValueError('continuation complete old producer semantic inventory required')
    import gluomem
    package_roots = {'gluomem_bench': Path(__file__).resolve().parents[1],
                     'gluomem': Path(gluomem.__file__).resolve().parent}
    for row in semantic:
        original_parts = Path(row['original_path']).parts
        current_path = Path(row['current_path']).resolve()
        expected_path = Path(row['original_path']).resolve()
        for package, package_root in package_roots.items():
            if package in original_parts:
                expected_path = package_root.joinpath(*original_parts[original_parts.index(package) + 1:])
                break
        if (current_path != expected_path
                or new_pins.get(row['current_path']) != row['current_sha256']
                or _hash(current_path) != row['current_sha256']):
            raise ValueError('continuation actual imported producer semantic identity/bytes differ')
    old_profile = old_spec['profile']
    if ({**old_profile, 'timeout_seconds': 600} != spec['profile']
            or spec['policy']['max_workers'] != 1 or spec['policy']['max_in_flight'] != 1
            or spec['profile']['concurrency'] != 1):
        raise ValueError('continuation requires explicitly fresh600s serial1 cohort; old profile retained')
    for key in ('chunk_chars', 'top_k', 'evidence_token_budget', 'judge_qualification',
                'embedding_url', 'embedding_recipe'):
        if spec[key] != old_spec[key]:
            raise ValueError('continuation changed old reader/scorer/encoder semantics')
    records = {r['job_id']: r for r in spec['jobs']}
    previous = {r['job_id']: r for r in plan['original_root']['jobs']}
    for job_id in plan['reuse_job_ids']:
        record, old = records[job_id], previous[job_id]
        out = Path(record['original_output'])
        leaf = json.loads((out / 'root-status.json').read_bytes())
        if (leaf.get('complete') is not True or leaf['case_ids'] != record['case_ids']
                or leaf['generation_calls'] != old['generation_calls']
                or json.loads((out / 'profile.json').read_bytes()) != old_profile
                or json.loads((out / 'cases.json').read_bytes()) != [planned[key] for key in record['case_ids']]):
            raise ValueError('reused complete leaf native cases/profile/spend differ')
        protocol = json.loads((out / 'protocol.json').read_bytes())
        _verify_native_source(out, record, suite, spec['chunk_chars'])
        if (protocol['system'] != record['system'] or protocol['history_id'] != record['history_id']
                or protocol['case_ids'] != record['case_ids']
                or any(protocol[key] != old_spec[key] for key in
                       ('chunk_chars', 'top_k', 'evidence_token_budget', 'judge_qualification'))):
            raise ValueError('reused complete leaf protocol differs from original specification')
        # The immutable root's full inventory has already been rehashed by plan.
        # SQLite mode=ro can create WAL sidecars, so verify only the byte-copy.
        with tempfile.TemporaryDirectory(prefix='mab-readonly-reuse-', dir=os.environ.get('TMPDIR')) as temporary:
            copied = Path(temporary) / 'leaf'
            shutil.copytree(out, copied)
            verify_history(copied)
    return bridge


class DeadlineEmbedder:
    """Transparent RemoteEmbedder delegation with the shared wall deadline.

    Embeddings are not charged as generations. No text/role/batch/fingerprint
    rewriting, no global provider mutation and no replacement encoding logic.
    """
    def __init__(self, embedder, *, budget, request_timeout):
        from gluomem_bench.bench.embedding_runtime import RemoteEmbedder
        if not isinstance(embedder, RemoteEmbedder):
            raise ValueError('MAB requires the qualified deadline-bound RemoteEmbedder seam')
        self.inner = embedder
        self.budget = budget
        self.request_timeout = request_timeout

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _call(self, method, value):
        from gluomem_bench.bench.embedding_runtime import RemoteEmbedder
        self.budget.check()
        bounded = RemoteEmbedder(self.inner.base_url, recipe=self.inner.recipe,
            timeout=min(self.inner.timeout, self.request_timeout, self.budget.remaining()))
        try:
            result = getattr(bounded, method)(value)
            self.budget.check()
            return result
        except Exception:
            self.budget.failed = True
            raise

    def embed(self, text):
        return self._call('embed', text)

    def embed_batch(self, texts):
        return self._call('embed_batch', texts)

    def embed_query(self, text):
        return self._call('embed_query', text)


def document_events(history, *, chunk_chars=4096):
    """Lossless adapted character chunks, without fabricated time or chat roles."""
    from gluomem.write.events import SourceEvent
    if type(chunk_chars) is not int or chunk_chars < 1:
        raise ValueError('positive integer chunk_chars required')
    return [SourceEvent(event_id=f'{history.id}:chunk:{i // chunk_chars}',
                       source_ref=f'memoryagentbench/{history.id}/chunk/{i // chunk_chars}',
                       source_type='document', content=history.context[i:i + chunk_chars])
            for i in range(0, len(history.context), chunk_chars)]


def cognee_documents(events, *, chunk_chars):
    """The identical raw-control projection: body + opaque source coordinates."""
    from gluomem_bench.bench.comparison_backends import Evidence
    return tuple(Evidence(id=e.event_id, text=e.content,
                          provenance={'source_ref': e.source_ref, 'offset': i * chunk_chars})
                 for i, e in enumerate(events))


def cognee_source_manifest(suite):
    """Every ordered history occurrence; hash exact common Evidence, not QA."""
    return [{'history_id': h.id, 'context_sha256': h.context_sha256,
             'documents_sha256': _digest([asdict(d) for d in cognee_documents(
                 document_events(h), chunk_chars=4096)])} for h in suite.histories]


def _messages(question, evidence, *, system_prompt):
    return [{'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': json.dumps({'question': question, 'evidence': evidence}, ensure_ascii=False)}]


def _summary(suite, scores, rows):
    report = suite.summarize(scores)
    na = [r for r in rows if r.get('status') == 'not_applicable']
    if na:
        report['not_applicable'] = [{'case_id': r['case_id'], 'reason': r['reason']} for r in na]
        sources = {c.sample.id: c.source for c in suite.cases}
        for source, record in report['sources'].items():
            ids = {r['case_id'] for r in na if sources[r['case_id']] == source}
            record['not_applicable'] = len(ids)
            record['applicable'] = record['planned'] - len(ids)
            record['missing_ids'] = [key for key in record['missing_ids'] if key not in ids]
            record['missing'] = len(record['missing_ids'])
    return report


def _coverage(out, history, events):
    from gluomem_bench.bench.beam_ingest import _read_rows, _validate_rows
    committed, retention = _validate_rows(_read_rows(out / 'stores' / f'{history.id}.mind'), events)
    sentinel = json.loads((out / 'stores' / f'{history.id}.ingest.json').read_bytes())
    if (sentinel.get('complete') is not True or sentinel.get('committed_receipts') != committed
            or type(sentinel.get('events')) is not int or sentinel.get('events') != len(events)
            or sentinel.get('coverage_protocol') != PROTOCOL
            or sentinel.get('dataset_sha256') != history.context_sha256):
        raise FatalRackError('persistent processed-input coverage mismatch')
    return {'history_id': history.id, 'context_sha256': history.context_sha256,
            'source_events': len(events), 'processed_events': len(committed),
            'committed_receipts': committed, **retention,
            'processing': 'synchronous production ingest acknowledgments; not frame-retention counts'}


def _ingest_mem0_documents(cfg, *, history, events, out, client, budget, embedder):
    """Stock 2.2.1 source-string API; shared guard/encoder, no chat adapter."""
    import os
    from collections import Counter
    from gluomem_bench.bench import mem0 as native
    if (os.environ.get('MEM0_TELEMETRY') != 'False' or not os.environ.get('MEM0_DIR')
            or os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENROUTER_API_KEY')):
        raise FatalRackError('Mem0 requires isolated import-time home, telemetry and providers')
    home = Path(os.environ['MEM0_DIR']).resolve()
    import gluomem
    if any(home.is_relative_to(root) for root in (Path(__file__).resolve().parents[2],
            Path(gluomem.__file__).resolve().parent.parent)):
        raise FatalRackError('Mem0 home must be outside source repositories')
    memory_cls = native._import_memory_cls()
    from mem0.configs.base import mem0_dir
    from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT
    from mem0.memory.telemetry import MEM0_TELEMETRY
    from mem0.memory.utils import parse_messages
    from mem0.utils.entity_extraction import extract_entities
    if MEM0_TELEMETRY or Path(mem0_dir).resolve() != home:
        raise FatalRackError('Mem0 was imported before environment isolation')
    if native._installed_versions()['mem0ai'] != '2.2.1':
        raise FatalRackError('unadmitted Mem0 source version')
    conv_dir = out / 'stores' / history.id
    conv_dir.mkdir()
    config = native.build_mem0_config(cfg, conv_dir=conv_dir, profile=client.profile, embedder=embedder)
    _write(out / 'mem0-config.json', native._safe_mem0_config(config))
    import mem0
    source_root = Path(mem0.__file__).parent
    hashes = {}
    for path in sorted(source_root.rglob('*.py')):
        relative = path.relative_to(source_root)
        target = out / 'mem0-source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        hashes[relative.as_posix()] = _hash(target)
    _write(out / 'mem0-source.json', {'versions': native._installed_versions(), 'files': hashes,
        'extractor_system': ADDITIVE_EXTRACTION_PROMPT,
        'clock_policy': 'stock observation/current date; no source/query date invented',
        'search': {'threshold': 0.1, 'rerank': False, 'top_k': cfg.benchmark.answer.context_k}})
    memory = None
    diagnostics = native.NativeMem0Diagnostics(out / 'mem0-diagnostics.json')
    receipts = []
    allowed = []
    try:
        memory = memory_cls.from_config(config)
        memory._benchmark_diagnostics = diagnostics
        _write(out / 'mem0-embedding.json', native.install_shared_mem0_encoder(memory, cfg=cfg, embedder=embedder))
        if (memory.llm.config.model != client.profile.model
                or str(memory.llm.client.base_url).rstrip('/') != client.profile.base_url
                or not memory.vector_store.is_local
                or memory.vector_store._get_bm25_encoder() is None
                or not extract_entities('Microsoft Azure in Prague and United Airlines')):
            raise FatalRackError('native Mem0 provider/auxiliary qualification failed')
        diagnostics.check()
        memory.llm.client.close()
        sdk = native.GuardedMem0SDK(client, trace=out / 'mem0-native.jsonl', expected_system=ADDITIVE_EXTRACTION_PROMPT)
        memory.llm.client = sdk
        diagnostics.phase = 'document_ingestion'
        for event in events:
            budget.check()
            before = len(sdk.records)
            added_result = memory.add(event.content, infer=True, user_id=history.id)
            if len(sdk.records) != before + 1 or sdk.records[-1].get('error'):
                raise FatalRackError('Mem0 extraction request coverage differs')
            record = sdk.records[-1]
            segment = '## New Messages\n' + parse_messages([{'role': 'user', 'content': event.content}]) + '\n\n## Observation Date\n'
            if segment not in record['request']['messages'][1]['content']:
                raise FatalRackError('Mem0 request lacks complete native-normalized source document')
            extraction = native.inspect_native_extraction(record['response']['choices'][0]['message']['content'])
            committed = []
            for item in added_result.get('results', []):
                saved = memory.get(item['id'])
                if (item.get('event') != 'ADD' or saved is None or saved.get('user_id') != history.id
                        or saved.get('memory') != item.get('memory')):
                    raise FatalRackError('Mem0 ADD lacks committed scoped readback')
                committed.append(saved)
            allowed.append(('user', event.content))
            buffered = [list(r) for r in memory.db.connection.execute('SELECT role, content FROM messages ORDER BY rowid').fetchall()]
            if len(buffered) > 10 or Counter(map(tuple, buffered)) - Counter(allowed):
                raise FatalRackError('Mem0 buffer contains unsubmitted source')
            receipts.append({'event_id': event.event_id, 'document_text': event.content,
                'ordinal': record['ordinal'], 'native_request_sha256': _digest(record['request']),
                'native_response_sha256': _digest(record['response']), 'extraction': extraction,
                'added': committed, 'rolling_buffer': buffered})
            _write(out / 'mem0-input-receipts.json', receipts)
            if extraction['status'] not in ('valid', 'valid_empty'):
                raise FatalRackError('native Mem0 extraction ' + extraction['status'])
            if extraction['extracted_memories'] != len(committed):
                raise FatalRackError('native Mem0 extraction/ADD coverage differs')
            diagnostics.check()
            budget.check()
        snapshot = memory.get_all(filters={'user_id': history.id}, top_k=10000)['results']
        history_rows = [list(r) for r in memory.db.connection.execute('SELECT memory_id, new_memory, event FROM history ORDER BY rowid').fetchall()]
        if (len(snapshot) == 10000 or any(m.get('user_id') != history.id for m in snapshot)
                or Counter((r[0], r[1]) for r in history_rows if r[2] == 'ADD') != Counter((m['id'], m['memory']) for m in snapshot)):
            raise FatalRackError('Mem0 committed history/vector inventory differs')
        _write(out / 'mem0-memories.json', snapshot)
        _write(out / 'mem0-history.json', history_rows)
        coverage = {'history_id': history.id, 'context_sha256': history.context_sha256,
            'source_events': len(events), 'processed_events': len(receipts),
            'source_input_kind': 'stock memory.add(str, infer=True, user_id=history.id)',
            'committed_memories': len(snapshot), 'input_receipts_sha256': _digest(receipts)}
        _write(conv_dir / 'ingest_meta.json', coverage)
        return memory, coverage
    except Exception:
        if memory is not None:
            native.close_mem0_memory(memory)
        else:
            diagnostics.close()
        raise


def _verify_mem0_coverage(out, history, events, profile, calls):
    """Reconcile persisted source attempts, guarded native calls and scoped ADDs."""
    from collections import Counter
    from gluomem_bench.bench.mem0 import inspect_native_extraction
    def read(name):
        return json.loads((out / name).read_bytes())
    source = read('mem0-source.json')
    if source['versions']['mem0ai'] != '2.2.1' or not source['files']:
        raise ValueError('Mem0 native source version/inventory missing')
    if set(source['files']) != {p.relative_to(out / 'mem0-source').as_posix()
                                for p in (out / 'mem0-source').rglob('*.py')}:
        raise ValueError('Mem0 native source inventory differs')
    if any(_hash(out / 'mem0-source' / name) != digest for name, digest in source['files'].items()):
        raise ValueError('Mem0 native source hash differs')
    if read('mem0-diagnostics.json'):
        raise ValueError('Mem0 native diagnostics are not clean')
    records = [json.loads(line) for line in (out / 'mem0-native.jsonl').read_bytes().splitlines()]
    receipts = read('mem0-input-receipts.json')
    direct = [r for r in calls if r['url'].endswith('/chat/completions')]
    if len(records) != len(events) or len(receipts) != len(events) or len(direct) < len(events):
        raise ValueError('Mem0 source-input attempt coverage differs')
    added = []
    allowed = []
    for ordinal, (event, record, receipt, wire) in enumerate(zip(events, records, receipts, direct), 1):
        request = record['request']
        extraction = inspect_native_extraction(record['response']['choices'][0]['message']['content'])
        expected_keys = {'model', 'temperature', 'top_p', 'max_tokens', 'messages', 'response_format'}
        segment = '## New Messages\nuser: ' + event.content + '\n\n\n## Observation Date\n'
        if (set(request) != expected_keys or record.get('error') or record['ordinal'] != ordinal
                or receipt['ordinal'] != ordinal or receipt['event_id'] != event.event_id
                or receipt['document_text'] != event.content
                or request['model'] != profile.model or request['temperature'] != profile.temperature
                or request['max_tokens'] != profile.output_budget('event_claims') or request['top_p'] != 1.0
                or request['response_format'] != {'type': 'json_object'}
                or len(request['messages']) != 2
                or request['messages'][0] != {'role': 'system', 'content': source['extractor_system']}
                or request['messages'][1]['role'] != 'user' or segment not in request['messages'][1]['content']
                or wire['request'] != {**request, 'chat_template_kwargs': {'enable_thinking': False}}
                or wire['response'] != record['response']
                or receipt['native_request_sha256'] != _digest(request)
                or receipt['native_response_sha256'] != _digest(record['response'])
                or receipt['extraction'] != extraction or extraction['status'] not in ('valid', 'valid_empty')
                or extraction['extracted_memories'] != len(receipt['added'])):
            raise ValueError('Mem0 native request/response/source extraction binding differs')
        allowed.append(('user', event.content))
        if len(receipt['rolling_buffer']) > 10 or Counter(map(tuple, receipt['rolling_buffer'])) - Counter(allowed):
            raise ValueError('Mem0 native normalized source buffer differs')
        added.extend(receipt['added'])
    snapshot = read('mem0-memories.json')
    saved_history = read('mem0-history.json')
    if (any(m.get('user_id') != history.id for m in added + snapshot)
            or Counter((m['id'], m['memory']) for m in snapshot) != Counter((m['id'], m['memory']) for m in added)
            or Counter((r[0], r[1]) for r in saved_history if r[2] == 'ADD') != Counter((m['id'], m['memory']) for m in snapshot)):
        raise ValueError('Mem0 committed scoped ADD/history inventory differs')
    db_path = out / 'stores' / history.id / 'history.db'
    db = sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        db_history = [list(r) for r in db.execute('SELECT memory_id, new_memory, event FROM history ORDER BY rowid')]
        buffer = [list(r) for r in db.execute('SELECT role, content FROM messages ORDER BY rowid')]
    finally:
        db.close()
    if db_history != saved_history or (receipts and buffer != receipts[-1]['rolling_buffer']):
        raise ValueError('Mem0 committed native SQLite readback differs')
    coverage = {'history_id': history.id, 'context_sha256': history.context_sha256,
        'source_events': len(events), 'processed_events': len(receipts),
        'source_input_kind': 'stock memory.add(str, infer=True, user_id=history.id)',
        'committed_memories': len(snapshot), 'input_receipts_sha256': _digest(receipts)}
    if read('coverage.json') != coverage or read(f'stores/{history.id}/ingest_meta.json') != coverage:
        raise ValueError('Mem0 persistent input/committed coverage differs')
    return records, snapshot


def _qualify_judges(requests, profile, qualification):
    if not requests:
        return
    required = max(r.max_tokens for r in requests)
    if required > profile.output_budget('rubric'):
        raise ValueError(f'native judge requires {required} tokens; frozen judge role budget is insufficient')
    expected = {'protocol': 'adapted-judge-role-budget@v1', 'model': profile.model,
                'temperature': profile.temperature, 'output_budget': profile.output_budget('rubric')}
    if profile.profile_version == 'role-budgets-v6':
        expected = {'protocol': 'native-judge-parameters@v2', 'model': profile.model,
                    'temperatures': [0, 0.1], 'output_budget': profile.output_budget('rubric')}
        if any(type(r.max_tokens) is not int or r.max_tokens < 1
               or type(r.temperature) not in (int, float) or r.temperature not in (0, 0.1)
               for r in requests):
            raise ValueError('unsupported native judge request parameters')
    if qualification != expected:
        raise ValueError('explicit frozen adapted judge role/model/budget qualification required')


def score_prediction(scorer, case, prediction, *, client=None, judge_qualification=None,
                     judgments=()):
    """Pinned native post-processing; never generic EM or an unguarded judge.

    Saved bound judgments allow inference-free native rescore. If a client is
    supplied every required judge request must dispatch through that guard.
    """
    requests = scorer.judge_requests(case, prediction)
    if client is not None and requests:
        if judgments:
            raise ValueError('cannot mix supplied judgments with guarded judge execution')
        _qualify_judges(requests, client.profile, judge_qualification)
        from gluomem_bench.bench.mab_suite import JudgeResponse
        judgments = []
        try:
            for request in requests:
                params = ({'temperature': request.temperature, 'max_tokens': request.max_tokens}
                          if client.profile.profile_version == 'role-budgets-v6' else {})
                result = client.generate_messages('rubric', [{'role': 'user', 'content': request.prompt}], **params)
                judgments.append(JudgeResponse(case_id=request.case_id, role=request.role,
                    prediction_sha256=request.prediction_sha256, prompt_sha256=request.prompt_sha256,
                    model=client.profile.model, output=result['text']))
            score = scorer.score(case, prediction, judgments=judgments)
        except Exception as exc:
            client._latch(exc)
            raise
    else:
        score = scorer.score(case, prediction, judgments=judgments)
    return score, tuple(judgments)


def run_history(*, suite, history_id, case_ids, cfg, profile, out, system,
                embedder, outer_budget, authored=False, chunk_chars=4096,
                judge_qualification=None, cognee_settings=None, cognee_admission=None,
                _execution_authority=None):
    """Run exactly selected question occurrences, sharing ingestion/reader budget.

    Unpinned rows require authored=True and are permanently labelled integration
    evidence. Source-only document events never contain questions, gold, task
    instructions or row metadata. No injectable extractor or raw-claim fallback.
    """
    if not authored and _execution_authority is not _EXECUTION_AUTHORITY:
        raise ValueError('native execution requires scoped approval through execute(spec_path, approval_path)')
    if system not in {'gluomem', 'mem0', 'cognee', 'no_memory', 'bm25', 'dense', 'hybrid', 'full_context'}:
        raise ValueError('system not admitted by MAB document seam')
    if not authored and not suite.receipt:
        raise ValueError('pinned MAB suite required')
    if not callable(getattr(outer_budget, 'lease', None)):
        raise ValueError('shared whole-system call/time/token lease required')
    outer_budget.check()
    histories = {h.id: h for h in suite.histories}
    if history_id not in histories:
        raise ValueError('unknown history occurrence')
    history = histories[history_id]
    selected = [c for c in suite.cases if c.sample.id in case_ids]
    if (not case_ids or len(set(case_ids)) != len(case_ids)
            or [c.sample.id for c in selected] != list(case_ids)
            or any(c.sample.conv_id != history_id for c in selected)):
        raise ValueError('exact ordered question occurrences over one history required')
    events = document_events(history, chunk_chars=chunk_chars)
    questions = {c.sample.id: suite.reader_question(c, agent='rag') for c in selected}
    reader_system = suite.templates()['get_template'](history.source, 'system', 'rag')
    scorer = suite.scorer()
    _ = scorer.upstream  # qualify actual native objective dependencies before output/client
    for case in selected:
        _qualify_judges(scorer.judge_requests(case, ''), profile, judge_qualification)
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    import gluomem
    roots = (Path(__file__).resolve().parents[2], Path(gluomem.__file__).resolve().parent.parent)
    if any(out.resolve().is_relative_to(root) for root in roots):
        raise ValueError('artifacts must be outside source repositories')
    from gluomem.llm.modules import DspyExtractors
    from gluomem_bench.bench.beam_ingest import _ingest_conversation
    from gluomem_bench.bench.rack_trace import RackRouter, WorkloadBudget
    from gluomem_bench.bench.retriever import GluoMemRetriever
    # Retain a debit even when JobContext's coordination persistence fails.
    from gluomem_bench.bench.cognee_retriever import NativeWorkloadBudget
    budget = NativeWorkloadBudget(max_calls=profile.max_generation_calls,
                            max_seconds=profile.max_run_seconds,
                            policy=profile.profile_version, outer_budget=outer_budget)
    if embedder is not None:
        embedder = DeadlineEmbedder(embedder, budget=budget, request_timeout=profile.timeout_seconds)
    out.mkdir(parents=True)
    client = None
    retriever = None
    raw_backend = None
    mem0_memory = None
    cognee_factory = None
    cognee_reader = None
    rows = []
    scores = []

    def close_resources():
        nonlocal client, retriever, raw_backend, mem0_memory
        callbacks = []
        if cognee_factory is not None:
            def close_native():
                # Native coverage's sqlite context managers end transactions but
                # leave cyclic read connections for GC. Release those handles
                # BEFORE the worker exits and its closed inventory is hashed.
                import gc
                gc.collect()
                cognee_factory.close()
                native_out = out / 'graph'
                db_path = native_out / 'native/system/databases/cognee_db'
                if db_path.is_file():
                    db = sqlite3.connect(db_path)
                    try:
                        if db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0] != 0:
                            raise ValueError('Cognee closed native checkpoint busy')
                    finally:
                        db.close()
                    ledger_path = native_out / 'native-ledger.json'
                    ledger = json.loads(ledger_path.read_bytes())
                    ledger['artifact_sha256'] = {p.relative_to(native_out).as_posix(): _hash(p)
                        for p in native_out.rglob('*') if p.is_file() and p != ledger_path}
                    _write(ledger_path, ledger)
            callbacks.append(close_native)
        if mem0_memory is not None:
            from gluomem_bench.bench.mem0 import close_mem0_memory
            owned, mem0_memory = mem0_memory, None
            callbacks.append(lambda: close_mem0_memory(owned))
        if retriever is not None:
            callbacks.extend(stack._db.close for stack in retriever._stacks.values())
            retriever = None
        if raw_backend is not None:
            close = raw_backend.close
            raw_backend = None
            callbacks.append(lambda: asyncio.run(close()))
        if client is not None:
            callbacks.append(client.close)
            client = None
        errors = []
        for callback in callbacks:
            try:
                callback()
            except Exception as error:
                errors.append(error)
        if errors:
            for other in errors[1:]:
                errors[0].add_note(f'additional cleanup failure: {safe_error(other)}')
            raise errors[0]

    try:
        _write(out / 'protocol.json', {'protocol': PROTOCOL, 'history_id': history_id,
            'case_ids': list(case_ids), 'system': system, 'chunk_chars': chunk_chars,
            'evidence_kind': 'authored-offline-integration' if authored else 'native-tasks-adapted-document-ingestion',
            'completion_authority': 'root-status.json', 'context_sha256': history.context_sha256,
            'family': history.family, 'source': history.source, 'row_index': history.row_index,
            'top_k': cfg.benchmark.answer.context_k, 'judge_qualification': judge_qualification,
            'evidence_token_budget': EVIDENCE_TOKEN_BUDGET,
            'evidence_count_policy': 'complete reader messages; whole-line ranked prefix',
            'reader_system': reader_system, 'reader_template_agent': 'rag',
            'cognee_admission': cognee_admission,
            'reader_packaging': 'adapted JSON evidence/question; no fabricated memorize dialogue'})
        _write(out / 'cases.json', [asdict(c) for c in selected])
        _write(out / 'profile.json', profile.model_dump(mode='json'))
        _write(out / 'effective-config.json', cfg.model_dump(mode='json'))
        _write(out / 'source-inputs.json', [e.model_dump(mode='json') for e in events])
        for relative in SCORER_FILES:
            target = out / 'scorer-source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(suite._source_bytes(relative))
        client = RackClient(profile, receipt=out / 'calls.jsonl', budget=budget)
        served = next(m for m in client.preflight()['data'] if m['id'] == profile.model)
        if served.get('max_model_len') != profile.context_window:
            raise SharedRackError('served context differs from frozen profile')
        stores = out / 'stores'
        stores.mkdir()
        if system == 'gluomem':
            router = RackRouter(cfg, client=client, trace=out / 'production-raw.jsonl', budget=budget)
            extractors = DspyExtractors(cfg, router=router)
            _ingest_conversation(cfg, SimpleNamespace(sha=history.context_sha256), stores,
                extractors, embedder, SimpleNamespace(conv_id=history_id), budget.check,
                source_events=events, coverage_protocol=PROTOCOL)
            coverage = _coverage(out, history, events)
            retriever = GluoMemRetriever(cfg, db_dir=stores, answer_fn=None,
                                        content_to_dias={}, embedder=embedder)
        elif system == 'mem0':
            mem0_memory, coverage = _ingest_mem0_documents(cfg, history=history, events=events,
                out=out, client=client, budget=budget, embedder=embedder)
        elif system == 'cognee':
            from gluomem_bench.bench.cognee_retriever import CogneeRetrieverFactory
            if cognee_settings is None:
                raise ValueError('explicit native Cognee settings required')
            if (cognee_settings.root.resolve() != (out / 'graph/native').resolve()
                    or cognee_settings.llm_model != profile.model
                    or cognee_settings.llm_url != profile.base_url):
                raise ValueError('native Cognee source/provider scope differs')
            _write(out / 'cognee-runtime.json', {
                'python_executable': cognee_settings.python_executable,
                'python_sha256': _hash(cognee_settings.python_executable),
                'package_sha256': _digest({p.relative_to(Path(cognee_settings.python_executable).parent.parent
                    / 'Lib/site-packages/cognee').as_posix(): _hash(p)
                    for p in sorted((Path(cognee_settings.python_executable).parent.parent
                        / 'Lib/site-packages/cognee').rglob('*.py'))}),
                'chunk_size': cognee_settings.chunk_size,
                'timeout_seconds': cognee_settings.timeout_seconds})
            cognee_factory = CogneeRetrieverFactory(settings=cognee_settings, profile=profile,
                outer_budget=budget, out=out / 'graph',
                documents=cognee_documents(events, chunk_chars=chunk_chars))
            # This callback executes this module's SAME native reader pack, not LME.
            def native_answer(question, original):
                nonlocal evidence, evidence_receipt, response
                evidence, evidence_receipt = _reader_evidence(client, question, original,
                    system_prompt=reader_system, system=system)
                response = client.generate_messages('answer', _messages(question, evidence,
                    system_prompt=reader_system))
                return SimpleNamespace(text=response['text'], prompt_tokens=response['prompt_tokens'],
                                       completion_tokens=response['completion_tokens'])
            cognee_reader = cognee_factory(cfg, None, native_answer)
            before = json.loads((out / 'graph/before-search.json').read_bytes())
            coverage = {'history_id': history_id, 'context_sha256': history.context_sha256,
                'source_events': len(events), 'processed_events': before['coverage']['input_count'],
                'processing': 'native add/cognify committed source-document readback',
                'native_coverage': before['coverage']}
        else:
            from gluomem_bench.bench.comparison_backends import Evidence, RawEvidenceBackend
            chunks = [Evidence(id=e.event_id, text=e.content,
                               provenance={'source_ref': e.source_ref, 'offset': i * chunk_chars})
                      for i, e in enumerate(events)]
            if system in {'bm25', 'dense', 'hybrid'}:
                raw_backend = RawEvidenceBackend(system,
                    encoder=embedder.embed_batch if system != 'bm25' else None,
                    query_encoder=embedder.embed_query if system != 'bm25' else None)
                asyncio.run(raw_backend.add(chunks))
            persisted = json.loads((out / 'source-inputs.json').read_bytes())
            if persisted != [e.model_dump(mode='json') for e in events]:
                raise FatalRackError('persistent allowed-input coverage mismatch')
            coverage = {'history_id': history_id, 'context_sha256': history.context_sha256,
                        'source_events': len(events), 'processed_events': len(events),
                        'processing': 'raw allowed documents; no semantic extraction',
                        'mode': system}
        _write(out / 'coverage.json', coverage)
        persisted_inputs = json.loads((out / 'source-inputs.json').read_bytes())
        if persisted_inputs != [e.model_dump(mode='json') for e in events]:
            raise FatalRackError('persistent allowed-input coverage mismatch')
        readback = json.loads((out / 'coverage.json').read_bytes())
        if (any(type(readback.get(key)) is not int or readback[key] != len(events)
                for key in ('source_events', 'processed_events'))
                or readback.get('context_sha256') != history.context_sha256
                or readback.get('history_id') != history_id):
            raise FatalRackError('persistent processed-input coverage mismatch')
        if system == 'gluomem' and readback != _coverage(out, history, events):
            raise FatalRackError('production committed-input coverage mismatch')
        if system == 'mem0':
            _verify_mem0_coverage(out, history, events, profile,
                [json.loads(line) for line in (out / 'calls.jsonl').read_bytes().splitlines()])
        for case in selected:
            budget.check()
            retrieval = None
            # Reuse production search, but do not promote ingestion created_at into
            # document chronology (legacy chat reader adds it unconditionally).
            if system == 'gluomem':
                result = retriever._search_for(history_id).search(
                    case.sample.question, limit=cfg.benchmark.answer.context_k)
                evidence = '\n'.join(hit.frame.content for hit in result.hits) if not result.abstained else ''
            elif system == 'full_context':
                evidence = history.context  # full fit or RackClient refuses; never truncate
            elif system == 'no_memory':
                evidence = ''
            elif system == 'mem0':
                mem0_memory._benchmark_diagnostics.phase = 'document_search'
                hits = mem0_memory.search(case.sample.question, filters={'user_id': history_id},
                    top_k=cfg.benchmark.answer.context_k, threshold=0.1, rerank=False)['results']
                mem0_memory._benchmark_diagnostics.check()
                if any(h.get('user_id') != history_id for h in hits):
                    raise FatalRackError('Mem0 search returned an unscoped memory')
                retrieval = {'query': case.sample.question, 'filters': {'user_id': history_id},
                    'reference_date': None, 'results': hits}
                evidence = '\n'.join(h['memory'] for h in hits)
            elif system == 'cognee':
                cognee_reader.answer(SimpleNamespace(id=case.sample.id,
                    question=case.sample.question, reader_question=questions[case.sample.id]))
                retrieval = cognee_reader.contexts[-1]
            else:
                hits = asyncio.run(raw_backend.search(case.sample.question,
                    limit=cfg.benchmark.answer.context_k))
                evidence = '\n'.join(hit.text for hit in hits)
            if system != 'cognee':
                evidence, evidence_receipt = _reader_evidence(client, questions[case.sample.id], evidence,
                    system_prompt=reader_system, system=system)
            if system == 'full_context':
                try:
                    client.admit_context(evidence_receipt['kept_tokens'],
                        output_budget=profile.output_budget('answer'),
                        input_chars=len(json.dumps(_messages(questions[case.sample.id], evidence,
                            system_prompt=reader_system), ensure_ascii=False)))
                except ValueError as overflow:
                    rows.append({'case_id': case.sample.id, 'source_id': case.source_id,
                        'question': questions[case.sample.id], 'evidence': evidence,
                        'evidence_receipt': evidence_receipt, 'retrieval': None,
                        'status': 'not_applicable', 'reason': 'context_overflow',
                        'detail': str(overflow), 'prediction': None, 'reader_receipt': None,
                        'score': None, 'judgments': []})
                    _write(out / 'predictions.json', rows)
                    continue
            if system != 'cognee':
                response = client.generate_messages('answer', _messages(questions[case.sample.id], evidence,
                                                                        system_prompt=reader_system))
            if response['admitted_prompt_tokens'] != evidence_receipt['kept_tokens']:
                raise SharedRackError('served tokenizer changed between evidence count and dispatch')
            prediction = response['text']
            score, judgments = score_prediction(scorer, case, prediction, client=client,
                                                judge_qualification=judge_qualification)
            scores.append(score)
            rows.append({'case_id': case.sample.id, 'source_id': case.source_id,
                         'question': questions[case.sample.id], 'evidence': evidence,
                         'evidence_receipt': evidence_receipt,
                         'retrieval': retrieval,
                         'prediction': prediction, 'reader_receipt': response,
                         'score': asdict(score), 'judgments': [asdict(j) for j in judgments]})
            _write(out / 'predictions.json', rows)
        budget.check()
        from gluomem_bench.bench.mab_suite import MABSuite
        _write(out / 'scores.json', _summary(MABSuite([history], selected), scores, rows))
        close_resources()
        # All writers/readers above have closed. Explicit checkpoint owns a short
        # final handle; no immutable read of a potentially live writer.
        for db_path in stores.glob('*.mind'):
            db = sqlite3.connect(db_path)
            try:
                db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            finally:
                db.close()
        budget.check()
        paths = sorted(p for p in out.rglob('*') if p.is_file())
        result = {'complete': True, 'protocol': PROTOCOL, 'case_ids': list(case_ids),
                  'evidence_kind': 'authored-offline-integration' if authored else 'native-tasks-adapted-document-ingestion',
                  'generation_calls': budget.calls,
                  'artifact_sha256': {str(p.relative_to(out)): _hash(p) for p in paths}}
        _verify_payload(out, result)
        budget.check()
        _write(out / 'root-status.json', result)
        if json.loads((out / 'root-status.json').read_bytes()) != result:
            raise ValueError('MAB root readback differs from verified result')
        return result
    except Exception as exc:
        budget.failed = True
        if cognee_factory is not None and cognee_factory.retriever is not None:
            cognee_factory.retriever._fail(exc)
        # RackClient classifies its own request failures at their origin. Do not
        # reclassify unrelated encoder/native/cleanup OSErrors as rack outages;
        # the workload budget above already forbids any further job requests.
        exc.__dict__.update(generation_calls=budget.calls, completed_questions=len(rows))
        try:
            close_resources()
        except Exception as cleanup_error:
            exc.add_note(f'cleanup failure retained: {safe_error(cleanup_error)}')
        for name in ('root-status.json', 'scores.json'):
            try:
                (out / name).unlink(missing_ok=True)
            except OSError as cleanup_error:
                exc.add_note(f'could not remove incomplete {name}: {safe_error(cleanup_error)}')
        try:
            _write(out / 'failure.json', {'complete': False, 'error': safe_error(exc),
                                        'generation_calls': budget.calls, 'completed_questions': len(rows)})
        except Exception as persistence_error:
            exc.add_note(
                f'failure receipt unavailable; generation_calls={budget.calls}; '
                f'{safe_error(persistence_error)}'
            )
        raise


def verify_history(out):
    """Offline preservation/coverage verification; root status alone admits completion."""
    out = Path(out)
    result = json.loads((out / 'root-status.json').read_bytes())
    return _verify_payload(out, result)


def _verify_cognee_rendered_context(native_root, contexts, *, python_executable, timeout=120):
    """Replay the pinned native formatter over its closed graph, with NO network.

    Reading Ladybug requires its pre-existing isolated interpreter. Only the
    graph's read-only storage boundary is adapted; native Edge/Node, evidence,
    conflict formatting and GraphCompletionRetriever context methods are real.
    """
    import os
    import subprocess
    import tempfile
    from gluomem_bench.bench.cognee_graph import NATIVE_SOURCE_PINS
    program = r'''
import asyncio, hashlib, importlib.metadata, json, sys
from pathlib import Path
from types import SimpleNamespace as NS
payload = json.load(sys.stdin)
# Windows creates a loopback socketpair for its event loop's self-pipe.
# Construct that plumbing first; all native imports/readback remain denied I/O.
loop = asyncio.new_event_loop()
def audit(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise RuntimeError('offline graph verifier forbids network')
sys.addaudithook(audit)
dist = importlib.metadata.distribution('cognee')
assert dist.version == '1.6.5'
for name, digest in payload['pins'].items():
    assert hashlib.sha256(Path(dist.locate_file('cognee/' + name)).read_bytes()).hexdigest() == digest
import cognee_db_workers
from ladybug import Connection
from ladybug.database import Database
from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Node, Edge
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.brute_force_triplet_search import get_memory_fragment
from cognee.modules.retrieval.utils.evidence import graph_context_evidence
paths = list(Path(payload['root']).rglob('*.lbug'))
assert len(paths) == 1
store = Database(str(paths[0]), read_only=True, buffer_pool_size=67108864)
connection = Connection(store)
nodes, edges = {}, {}
try:
    result = connection.execute('MATCH (a:Node)-[r:EDGE]->(b:Node) RETURN a,r,b')
    while result.has_next():
        a, r, b = result.get_next()
        for value in (a, b):
            nodes[value['id']] = LadybugAdapter._parse_node_properties(None, dict(value))
        attrs = json.loads(r['properties']) if r.get('properties') else {}
        attrs.setdefault('relationship_name', r['relationship_name'])
        key = (a['id'], b['id'], r['relationship_name'])
        assert key not in edges
        edges[key] = attrs
    async def neighborhood(chunk_ids, **kwargs):
        return list(nodes.items()), [(s, t, relation, attrs) for (s, t, relation), attrs in edges.items()]
    retriever = GraphCompletionRetriever()
    retriever._unified_engine = NS(graph=NS(get_neighborhood=neighborhood))
    async def verify():
        async def graph_data():
            return list(nodes.items()), [(s, t, rel, attrs) for (s, t, rel), attrs in edges.items()]
        memory = await get_memory_fragment(graph_engine=NS(get_graph_data=graph_data))
        projected = {(e.node1.id, e.node2.id, e.attributes['relationship_type']): e
                     for e in memory.get_edges()}
        for context in payload['contexts']:
            for hit in context['hits']:
                refs = hit['provenance']['native_evidence']
                native_edges = []
                for ref in refs:
                    if ref['kind'] != 'graph_edge':
                        continue
                    key = (ref['source_node_id'], ref['target_node_id'], ref['relationship_name'])
                    native_edges.append(projected[key])
                observed = [r.model_dump(mode='json') for r in graph_context_evidence(
                    native_edges, dataset_id=hit['provenance']['dataset_id'])]
                assert observed == [r for r in refs if r['kind'] in ('graph_node', 'graph_edge')]
                rendered = await retriever.get_context_from_objects(query=context['query'],
                                                                    retrieved_objects=native_edges)
                assert rendered == hit['text']
    loop.run_until_complete(verify())
finally:
    loop.close()
    connection.close()
    store.close()
print('MAB_NATIVE_CONTEXT_VERIFIED')
'''
    allowed = {'PATH', 'SYSTEMROOT', 'WINDIR', 'HOME', 'USERPROFILE', 'LOCALAPPDATA',
               'APPDATA', 'TEMP', 'TMP', 'TMPDIR'}
    env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
    with tempfile.TemporaryDirectory(prefix='mab-native-readback-', dir=os.environ.get('TMPDIR')) as scratch:
        scratch = Path(scratch)
        (scratch / 'isolated.env').write_bytes(b'')
        env.update(PYTHONUTF8='1', COGNEE_ENV_FILE=str(scratch / 'isolated.env'),
            TELEMETRY_DISABLED='true', COGNEE_TRACING_ENABLED='false', LOG_LEVEL='ERROR',
            LITELLM_LOCAL_MODEL_COST_MAP='true',
            SYSTEM_ROOT_DIRECTORY=str(scratch / 'system'), DATA_ROOT_DIRECTORY=str(scratch / 'data'),
            CACHE_ROOT_DIRECTORY=str(scratch / 'cache'), COGNEE_LOGS_DIR=str(scratch / 'logs'))
        completed = subprocess.run([str(python_executable), '-I', '-c', program],
            input=json.dumps({'root': str(Path(native_root).resolve()), 'contexts': contexts,
                              'pins': NATIVE_SOURCE_PINS}),
            text=True, encoding='utf-8', capture_output=True, env=env, cwd=scratch,
            timeout=timeout, check=False)
    if completed.returncode != 0 or 'MAB_NATIVE_CONTEXT_VERIFIED' not in completed.stdout.splitlines():
        raise ValueError('Cognee closed native context/graph readback differs')


def _verify_cognee_lineage(native_root, provenance, coverage):
    """Read the actual closed native provenance sidecar; never trust labels alone."""
    from uuid import UUID
    def uid(value):
        return UUID(str(value)).hex
    refs = provenance['native_evidence']
    dataset = uid(coverage['dataset_id'])
    data_ids = {uid(r['data_id']) for r in coverage['native_rows']}
    if any(uid(ref['dataset_id']) != dataset for ref in refs):
        raise ValueError('Cognee source lineage dataset differs')
    db = sqlite3.connect((native_root / 'system/databases/cognee_db').resolve().as_uri()
                         + '?mode=ro&immutable=1', uri=True)
    db.row_factory = sqlite3.Row
    try:
        records = [dict(r) for r in db.execute(
            'SELECT p.*, d.name AS document_name FROM provenance_edge_evidence p '
            'JOIN data d ON d.id = p.data_id WHERE p.dataset_id = ?', (dataset,))]
    finally:
        db.close()
    edges = {r['artifact_id']: r for r in refs if r['kind'] == 'graph_edge'}
    segments = [r for r in refs if r['kind'] == 'segment']
    expected = []
    for r in records:
        edge_id = str(UUID(r['edge_id']))
        if edge_id not in edges:
            continue
        edge = edges[edge_id]
        if (uid(edge['source_node_id']) != r['source_node_id']
                or uid(edge['target_node_id']) != r['destination_node_id']
                or edge['relationship_name'] != r['relationship_name'] or r['data_id'] not in data_ids):
            raise ValueError('Cognee source lineage native edge differs')
        expected.append((edge_id, str(UUID(r['data_id'])), str(UUID(r['chunk_id'])),
                         r['chunk_index'], r['document_name']))
    observed = []
    for ref in segments:
        if (ref['role'] != 'supports_assertion' or ref['artifact_id'] != ref['chunk_id']
                or ref['source_ref_key'] != f"source_ref:v1:{UUID(dataset)}:{UUID(ref['data_id'])}"):
            raise ValueError('Cognee source lineage reference differs')
        observed.append((ref['assertion_id'], ref['data_id'], ref['chunk_id'],
                         ref['chunk_index'], ref['document_name']))
    from collections import Counter
    if Counter(expected) != Counter(observed) or (edges and not expected):
        raise ValueError('Cognee source lineage committed sidecar differs')


def _verify_cognee(out, history, events, cases, rows, profile, protocol):
    """Closed native inventory, exact source processing and graph/reader binding."""
    from gluomem_bench.bench.cognee_campaign import verify_native_coverage, verify_generation_evidence
    from gluomem_bench.bench.cognee_graph import GRAPH_CONDITION, NATIVE_SOURCE_PINS, graph_context_evidence
    from gluomem_bench.bench.comparison_backends import COGNEE_WHEEL_SHA256, Evidence
    native_out = out / 'graph'
    def read(name):
        return json.loads((native_out / name).read_bytes())
    ledger = read('native-ledger.json')
    documents = cognee_documents(events, chunk_chars=protocol['chunk_chars'])
    pins = {'version': '1.6.5', 'wheel_sha256': COGNEE_WHEEL_SHA256,
            'source_sha256': NATIVE_SOURCE_PINS}
    if (ledger['status'] != 'closed_verified' or ledger['condition'] != GRAPH_CONDITION
            or ledger['native_worker_closed'] is not True or ledger['proxy_closed'] is not True
            or ledger['source_inputs'] != [asdict(d) for d in documents] or ledger['source_pins'] != pins):
        raise ValueError('Cognee closed native source/scope/pins differ')
    actual = {p.relative_to(native_out).as_posix(): _hash(p) for p in native_out.rglob('*')
              if p.is_file() and p != native_out / 'native-ledger.json'}
    if ledger['artifact_sha256'] != actual:
        raise ValueError('Cognee closed native inventory/hash differs')
    coverage = verify_native_coverage(native_out / 'native', dataset=ledger['dataset'],
        expected=documents, quiescent=True)
    coverage = {k: v for k, v in coverage.items()
                if k not in {'condition', 'graph_retrieval_qualified', 'graph_retrieval_gap'}}
    if coverage != ledger['verification']['coverage']:
        raise ValueError('Cognee closed native processing coverage differs')
    before = read('before-search.json')
    initial = {**coverage, 'quiescent': False}
    expected_coverage = {'history_id': history.id, 'context_sha256': history.context_sha256,
        'source_events': len(events), 'processed_events': coverage['input_count'],
        'processing': 'native add/cognify committed source-document readback', 'native_coverage': initial}
    if (json.loads((out / 'coverage.json').read_bytes()) != expected_coverage
            or before['coverage'] != initial or before['source_pins'] != pins):
        raise ValueError('Cognee before-QA source coverage differs')
    generation = verify_generation_evidence(native_out / 'generation.jsonl', profile=profile,
        expected_calls=ledger['generation_calls'])
    if generation != ledger['generation'] or generation != ledger['verification']['generation']:
        raise ValueError('Cognee native generation ledger differs')
    contexts = read('graph-contexts.json')
    runtime = json.loads((out / 'cognee-runtime.json').read_bytes())
    if _hash(runtime['python_executable']) != runtime['python_sha256']:
        raise ValueError('Cognee native readback interpreter bytes differ')
    package = Path(runtime['python_executable']).parent.parent / 'Lib/site-packages/cognee'
    if _digest({p.relative_to(package).as_posix(): _hash(p) for p in sorted(package.rglob('*.py'))}) != runtime['package_sha256']:
        raise ValueError('Cognee native readback source package differs')
    _verify_cognee_rendered_context(native_out / 'native', contexts,
        python_executable=runtime['python_executable'], timeout=profile.timeout_seconds)
    if len(contexts) != len(cases) or ledger['graph_context_receipts'] != len(cases):
        raise ValueError('Cognee native graph context coverage differs')
    prior = before['generation']['calls']
    for ordinal, (case, row, context) in enumerate(zip(cases, rows, contexts, strict=True), 1):
        gate = read(f'before-reader-{ordinal:06d}.json')
        if (context != row['retrieval'] or gate['graph_context'] != context
                or gate['coverage'] != initial or gate['source_pins'] != pins
                or context['sample_id'] != case.sample.id or context['query'] != case.sample.question
                or context['native_final_completion_used'] is not False
                or not prior <= context['native_calls_after_search'] <= generation['calls']
                or gate['generation']['calls'] != context['native_calls_after_search']
                or row['evidence_receipt']['original_text'] != '\n'.join(h['text'] for h in context['hits'])):
            raise ValueError('Cognee native graph/MAB reader context binding differs')
        prior = context['native_calls_after_search']
        for hit in context['hits']:
            p = hit['provenance']
            _verify_cognee_lineage(native_out / 'native', p, coverage)
            mapped = graph_context_evidence({'dataset_name': p['dataset_name'],
                'dataset_id': p['dataset_id'], 'context': hit['text'],
                'evidence': p['native_evidence'], 'completion': p.get('completion')}, dataset=ledger['dataset'])
            if (mapped != [Evidence(**hit)] or p['condition'] != GRAPH_CONDITION
                    or p['dataset_id'].replace('-', '') != coverage['dataset_id'].replace('-', '')):
                raise ValueError('Cognee native graph scope/context lineage differs')
    return generation['calls']


def _verify_payload(out, result):
    if result.get('complete') is not True or result.get('protocol') != PROTOCOL:
        raise ValueError('incomplete MAB root')
    inventory = result['artifact_sha256']
    required = {'protocol.json', 'cases.json', 'profile.json', 'effective-config.json',
                'coverage.json', 'source-inputs.json', 'calls.jsonl', 'predictions.json', 'scores.json'}
    required.update('scorer-source/' + name for name in SCORER_FILES)
    if not required <= {Path(name).as_posix() for name in inventory}:
        raise ValueError('required MAB evidence inventory missing')
    actual_names = {p.relative_to(out).as_posix() for p in out.rglob('*')
                    if p.is_file() and p != out / 'root-status.json'}
    if actual_names != {Path(name).as_posix() for name in inventory}:
        raise ValueError('exact MAB artifact inventory differs')
    if any(name.endswith(('-wal', '-shm')) for name in actual_names):
        raise ValueError('unclosed MAB store sidecar in artifact inventory')
    for relative, expected in inventory.items():
        path = out / relative
        if not path.resolve().is_relative_to(out.resolve()) or _hash(path) != expected:
            raise ValueError('MAB artifact hash mismatch')
    rows = json.loads((out / 'predictions.json').read_bytes())
    if (not result['case_ids'] or len(set(result['case_ids'])) != len(result['case_ids'])
            or [r['case_id'] for r in rows] != result['case_ids']):
        raise ValueError('incomplete question occurrence coverage')
    from gluomem.write.events import SourceEvent
    from gluomem_bench.bench.mab_suite import MABCase, MABHistory, MABSuite, JudgeResponse
    from gluomem_bench.bench.types import Sample
    protocol = json.loads((out / 'protocol.json').read_bytes())
    cases = [MABCase(**{**c, 'sample': Sample(**c['sample']), 'answers': tuple(c['answers'])})
             for c in json.loads((out / 'cases.json').read_bytes())]
    events = [SourceEvent.model_validate(e) for e in json.loads((out / 'source-inputs.json').read_bytes())]
    context = ''.join(e.content or '' for e in events)
    if hashlib.sha256(context.encode('utf-8')).hexdigest() != protocol['context_sha256']:
        raise ValueError('allowed source context hash mismatch')
    history = MABHistory(protocol['history_id'], protocol['family'], protocol['source'],
                         protocol['row_index'], context, protocol['context_sha256'], {})
    expected_events = document_events(history, chunk_chars=protocol['chunk_chars'])
    if events != expected_events:
        raise ValueError('source event identity/order/content/clock mismatch')
    saved_suite = MABSuite([history], cases)
    saved_suite.root = out / 'scorer-source'
    scorer = saved_suite.scorer()
    if [c.sample.id for c in cases] != result['case_ids']:
        raise ValueError('saved native case coverage mismatch')
    scores = []
    for case, row in zip(cases, rows, strict=True):
        if (case.sample.conv_id != history.id or row['source_id'] != case.source_id
                or row['question'] != saved_suite.reader_question(case)):
            raise ValueError('saved native reader/source binding mismatch')
        if row.get('status') == 'not_applicable':
            if (protocol['system'] != 'full_context' or row['reason'] != 'context_overflow'
                    or any(row[key] is not None for key in ('score', 'prediction', 'reader_receipt'))
                    or row['judgments']):
                raise ValueError('invalid not-applicable native score')
            continue
        score, _ = score_prediction(scorer, case, row['prediction'],
                                    judgments=[JudgeResponse(**j) for j in row['judgments']])
        if json.loads(json.dumps(asdict(score))) != row['score']:
            raise ValueError('saved score differs from pinned native rescore')
        scores.append(score)
    if _summary(saved_suite, scores, rows) != json.loads((out / 'scores.json').read_bytes()):
        raise ValueError('per-source native score summary differs')
    from gluomem_bench.bench.rack import RackProfile
    profile = RackProfile.model_validate_json((out / 'profile.json').read_bytes())
    calls = [json.loads(line) for line in (out / 'calls.jsonl').read_bytes().splitlines() if line]
    if protocol.get('evidence_token_budget') != EVIDENCE_TOKEN_BUDGET:
        raise ValueError('frozen reader evidence token budget differs')
    class SavedTokenizer:
        @staticmethod
        def _key(value):
            # JSON structural equality, including dict order independence and
            # Python's existing bool/int/float equality. Do not stringify numbers.
            if isinstance(value, dict):
                return ('object', frozenset((k, SavedTokenizer._key(v)) for k, v in value.items()))
            if isinstance(value, list):
                return ('array', tuple(SavedTokenizer._key(v) for v in value))
            return ('scalar', value)

        def __init__(self):
            self._receipts = {}
            self._counts = {}
            for r in calls:
                if (r['url'].endswith('/tokenize')
                        and r['request'].get('model') == profile.model
                        and r['request'].get('add_generation_prompt') is True
                        and r['request'].get('chat_template_kwargs') == {'enable_thinking': False}
                        and r.get('status') == 200 and not r.get('error')):
                    messages = r['request'].get('messages')
                    key = hash(self._key(messages))
                    self._receipts.setdefault(key, []).append((messages, r))

        def prompt_token_count(self, messages):
            key = hash(self._key(messages))
            # Hashes select buckets only; exact original equality is mandatory
            # on both receipt lookup and cache hits, including hash collisions.
            for saved_messages, count in self._counts.get(key, ()):
                if saved_messages == messages:
                    return count
            counts = {r['response'].get('count') for saved_messages, r in self._receipts.get(key, ())
                      if saved_messages == messages}
            if len(counts) != 1:
                raise ValueError('served evidence tokenizer receipt missing or inconsistent')
            count = counts.pop()
            self._counts.setdefault(key, []).append((messages, count))
            return count
    saved_tokenizer = SavedTokenizer()
    for row in rows:
        receipt = row['evidence_receipt']
        kept, expected_receipt = _reader_evidence(saved_tokenizer, row['question'],
            receipt['original_text'], system_prompt=protocol['reader_system'], system=protocol['system'])
        na = row.get('status') == 'not_applicable'
        if (receipt != expected_receipt or row['evidence'] != kept
                or (not na and row['reader_receipt']['admitted_prompt_tokens'] != receipt['kept_tokens'])
                or (protocol['system'] == 'no_memory' and receipt['original_text'] != '')
                or (protocol['system'] == 'full_context' and receipt['original_text'] != context)):
            raise ValueError('reader evidence/token receipt binding mismatch')
        if na:
            messages = _messages(row['question'], kept, system_prompt=protocol['reader_system'])
            if (receipt['kept_tokens'] + profile.output_budget('answer') + profile.schema_reserve_tokens
                    <= profile.context_window and len(json.dumps(messages, ensure_ascii=False)) <= profile.max_input_chars):
                raise ValueError('not-applicable context overflow is not supported by served tokens')
            continue
        matching = [i for i, r in enumerate(calls) if r['url'].endswith('/chat/completions')
                    and r['request'].get('messages') == _messages(row['question'], kept,
                        system_prompt=protocol['reader_system'])]
        if not matching or any(i == 0 or not calls[i-1]['url'].endswith('/tokenize')
                or calls[i-1]['request']['messages'] != calls[i]['request']['messages']
                or calls[i-1]['response']['count'] != receipt['kept_tokens'] for i in matching):
            raise ValueError('reader dispatch evidence tokenizer binding mismatch')
    direct = [r for r in calls if r['url'].endswith('/chat/completions')]
    expected = []
    native_calls = (_verify_cognee(out, history, events, cases, rows, profile, protocol)
                    if protocol['system'] == 'cognee' else 0)
    if protocol['system'] == 'mem0':
        native_records, snapshot = _verify_mem0_coverage(out, history, events, profile, calls)
        expected.extend(('event_claims', r['request']['messages'],
            r['response']['choices'][0]['message']['content'], profile.temperature,
            profile.output_budget('event_claims')) for r in native_records)
        inventory = {m['id']: m for m in snapshot}
        for case, row in zip(cases, rows, strict=True):
            retrieval = row['retrieval']
            hits = retrieval['results']
            if (retrieval['query'] != case.sample.question or retrieval['filters'] != {'user_id': history.id}
                    or retrieval['reference_date'] is not None or len(hits) > protocol['top_k']
                    or any(h.get('user_id') != history.id or h['id'] not in inventory
                        or h['memory'] != inventory[h['id']]['memory'] for h in hits)
                    or row['evidence_receipt']['original_text'] != '\n'.join(h['memory'] for h in hits)):
                raise ValueError('Mem0 native query/retrieval/source-clock binding differs')
    for case, row in zip(cases, rows, strict=True):
        native_system = saved_suite.templates()['get_template'](case.source, 'system', 'rag')
        if protocol['reader_system'] != native_system:
            raise ValueError('native system prompt receipt mismatch')
        if row.get('status') == 'not_applicable':
            continue
        expected.append(('answer', _messages(row['question'], row['evidence'],
                                             system_prompt=native_system), row['prediction'],
                         profile.temperature, profile.output_budget('answer')))
        if row['reader_receipt']['text'] != row['prediction']:
            raise ValueError('saved reader receipt differs from prediction')
        requests = scorer.judge_requests(case, row['prediction'])
        _qualify_judges(requests, profile, protocol['judge_qualification'])
        for request, judgment in zip(requests, row['judgments'], strict=True):
            native = profile.profile_version == 'role-budgets-v6'
            expected.append(('rubric', [{'role': 'user', 'content': request.prompt}], judgment['output'],
                             request.temperature if native else profile.temperature,
                             request.max_tokens if native else profile.output_budget('rubric')))
    if len(expected) != len(direct):
        raise ValueError('reader/judge receipt coverage mismatch')
    for receipt, (task, messages, text, temperature, max_tokens) in zip(direct, expected, strict=True):
        payload, response = receipt['request'], receipt['response']
        if profile.profile_version == 'role-budgets-v6':
            tokens = saved_tokenizer.prompt_token_count(messages)
            index = next(i for i, row in enumerate(calls) if row is receipt)
            if (type(tokens) is not int or tokens < 0
                    or tokens + max_tokens + profile.schema_reserve_tokens > profile.context_window
                    or len(json.dumps(messages, ensure_ascii=False)) > profile.max_input_chars
                    or index == 0 or not calls[index-1]['url'].endswith('/tokenize')
                    or calls[index-1]['request']['messages'] != messages):
                raise ValueError('native generation served-tokenizer/context binding differs')
        choices = response.get('choices', [])
        if (receipt.get('status') != 200 or receipt.get('error')
                or payload.get('messages') != messages
                or payload.get('model') != profile.model
                or payload.get('max_tokens') != max_tokens
                or (profile.profile_version == 'role-budgets-v6' and payload.get('temperature') != temperature)
                or response.get('model') != profile.model
                or len(choices) != 1 or choices[0].get('finish_reason') != 'stop'
                or choices[0].get('message', {}).get('content') != text):
            raise ValueError('reader/judge wire receipt binding mismatch')
    raw_path = out / 'production-raw.jsonl'
    raw = [json.loads(line) for line in raw_path.read_bytes().splitlines() if line] if raw_path.exists() else []
    if (type(result['generation_calls']) is not int
            or result['generation_calls'] != len(raw) + len(direct) + native_calls):
        raise ValueError('whole-system generation call reconciliation mismatch')
    if protocol['system'] == 'gluomem':
        if not raw or any(r.get('error') or r.get('finish_reasons') != ['stop'] for r in raw):
            raise ValueError('production extraction/consolidation receipt incomplete')
        if sum(r['task'] == 'event_claims' for r in raw) != sum(bool(e.content.strip()) for e in events):
            raise ValueError('production extraction receipt coverage mismatch')
    return result


def required_files(mab_root):
    """First-party deferred producer closure + locks + every pinned upstream byte."""
    import gluomem
    from gluomem_bench.bench.mab_suite import PINNED_FILES
    bench = Path(__file__).resolve().parents[1]
    core = Path(gluomem.__file__).resolve().parent
    paths = set(bench.rglob('*.py')) | set(core.rglob('*.py'))
    for project in (bench.parent, core.parent):
        for name in ('pyproject.toml', 'uv.lock'):
            if (project / name).is_file():
                paths.add(project / name)
        paths.update(project.glob('*.yaml'))
        paths.update(project.glob('*.yml'))
    paths.update(Path(mab_root) / relative for relative in PINNED_FILES)
    return tuple(sorted(p.resolve() for p in paths))


COGNEE_QUALIFICATION_SHA256 = '7e65689fca29887972c953629542aa01449b0e3f57bf84f1206a8173d2b1cc04'


def _qualify_cognee_spec(spec, approval, suite):
    """Read-only native qualification + full source projection + package closure."""
    from gluomem_bench.bench.cognee_graph import GRAPH_CONDITION
    entry = spec.get('cognee', {})
    item = entry.get('qualification', {})
    path = Path(item.get('path', ''))
    if (spec.get('campaign_mode') != FULL_MODE
            or entry.get('scope') != 'all-pinned-mab-history-question-occurrences'
            or spec['case_ids'] != [c.sample.id for c in suite.cases]
            or spec['chunk_chars'] != 4096):
        raise ValueError('Cognee requires full source scope, all occurrences and common 4096-char documents')
    inventory = {str(Path(r['path']).resolve()): r['sha256'] for r in spec['files']}
    def checked(record):
        target = Path(record['path'])
        if (not target.is_absolute() or not target.is_file()
                or inventory.get(str(target.resolve())) != record['sha256']
                or _hash(target) != record['sha256']):
            raise ValueError('Cognee qualification/source frozen evidence differs')
        return target
    if (not path.is_absolute() or item.get('sha256') != COGNEE_QUALIFICATION_SHA256
            or approval.get('approved_gates', {}).get('cognee_native_graph') != COGNEE_QUALIFICATION_SHA256):
        raise ValueError('Cognee independent exact native qualification approval required')
    qualification = json.loads(checked(item).read_bytes())
    for key in ('qualified', 'synthetic_only', 'native_graph_context', 'served_tokenizer_used',
                'dataset_scope_negative_qualified', 'wire_roles_qualified', 'vector_parity_qualified'):
        if qualification.get(key) is not True:
            raise ValueError('Cognee native graph qualification incomplete')
    if (qualification.get('condition') != GRAPH_CONDITION
            or qualification.get('native_final_completion_used') is not False
            or qualification.get('real_model') != spec['profile']['model']
            or qualification.get('recipe') != spec['embedding_recipe']):
        raise ValueError('Cognee qualified model/encoder/context scope differs')
    for record in qualification['evidence_files']:
        checked(record)
    native = entry.get('native', {})
    executable = Path(native.get('python_executable', ''))
    if (not executable.is_absolute() or not executable.is_file()
            or type(native.get('chunk_size')) is not int or native['chunk_size'] < 1
            or type(native.get('timeout_seconds')) not in (int, float)
            or not 0 < native['timeout_seconds'] <= 1200):
        raise ValueError('Cognee native executable/chunk/deadline scope required')
    package = executable.parent.parent / 'Lib/site-packages/cognee'
    sources = sorted(package.rglob('*.py'))
    if not sources:
        raise ValueError('Cognee native source package missing')
    hashes = {p.relative_to(package).as_posix(): _hash(p) for p in sources}
    if (native.get('package_sha256') != _digest(hashes)
            or any(inventory.get(str(p.resolve())) != hashes[p.relative_to(package).as_posix()]
                   for p in sources)):
        raise ValueError('Cognee native source package closure differs')
    projection = cognee_source_manifest(suite)
    if entry.get('source_manifest_sha256') != _digest(projection):
        raise ValueError('Cognee full source projection differs')
    return entry


def execute(spec_path, approval_path):
    """Own one clock/counter from before admission/setup through final readback."""
    from gluomem_bench.bench.parallel_campaign import CampaignBudget
    raw = json.loads(Path(spec_path).read_bytes())
    prior = (raw.get('accounting_continuation', {}).get('prior', {}).get('coordinator', {})
             if raw.get('campaign_mode') == CONTINUATION_MODE else {})
    budget = CampaignBudget(max_calls=raw['policy']['max_calls'],
        max_seconds=raw['policy']['max_seconds'], prior_calls=prior.get('calls', 0),
        prior_elapsed_seconds=prior.get('elapsed_seconds', 0), clock=lambda: _clock())
    root = Path(raw['output_root'])
    initially_absent = not root.exists()
    try:
        return _execute(spec_path, approval_path, budget)
    except Exception as exc:
        spend = budget.snapshot()
        exc.__dict__['generation_calls'] = spend['calls']
        exc.__dict__['accounting'] = spend
        if initially_absent and root.is_dir():
            for name in ('root-status.pending.json', 'root-status.json'):
                try:
                    (root / name).unlink(missing_ok=True)
                except OSError as cleanup:
                    exc.add_note(f'root revocation failed: {safe_error(cleanup)}')
            try:
                _write(root / 'failure.json', {'complete': False, 'generation_calls': spend['calls'],
                    'accounting': spend, 'error': safe_error(exc)})
            except Exception as persistence:
                exc.add_note(
                    f'failure receipt unavailable: {safe_error(persistence)}; '
                    f'accounting={spend}'
                )
        raise


def _execute(spec_path, approval_path, budget):
    """Native campaign entry point; exact byte/matrix gate before model I/O.

    Uses the comprehensive_driver approval envelope plus execution_kind=PROTOCOL.
    Jobs group ordered occurrence-qualified case_ids by history, not source IDs
    or context hashes. Campaign root-status.json alone admits whole completion.
    """
    from gluomem_bench.bench.comprehensive_driver import verify_execution_spec
    from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe, RemoteEmbedder
    from gluomem_bench.bench.mab_suite import MABSuite
    from gluomem_bench.bench.parallel_campaign import CampaignPolicy, Job, run_jobs
    from gluomem_bench.bench.rack import RackProfile, controlled_config
    spec = verify_execution_spec(spec_path, approval_path)
    budget.check()
    if spec.get('execution_kind') != PROTOCOL:
        raise ValueError('explicit native MAB execution_kind required')
    if type(spec.get('evidence_token_budget')) is not int or spec['evidence_token_budget'] != EVIDENCE_TOKEN_BUDGET:
        raise ValueError('explicit frozen evidence_token_budget=4096 required')
    declared = {Path(item['path']).resolve() for item in spec['files']}
    if not set(required_files(spec['mab_root'])) <= declared:
        raise ValueError('scoped producer/data file inventory incomplete')
    suite = MABSuite.from_directory(spec['mab_root'])
    profile = RackProfile.model_validate(spec['profile'])
    policy = CampaignPolicy(**spec['policy'])
    mode = spec.get('campaign_mode', 'legacy-strict-v1')
    if mode not in ('legacy-strict-v1', FULL_MODE, CONTINUATION_MODE):
        raise ValueError('unknown MAB campaign mode')
    continuation = mode == CONTINUATION_MODE
    full = mode in (FULL_MODE, CONTINUATION_MODE)
    plan = continuation_plan(spec) if continuation else None
    if continuation and (budget.prior_calls != plan['prior_calls']
            or budget.prior_elapsed_seconds != plan['prior_elapsed_seconds']):
        raise ValueError('imported original campaign prior accounting differs')
    if full:
        if (profile.profile_version != 'role-budgets-v6' or policy.stop_on_failure
                or profile.output_budget('answer') != 4096
                or not 1 <= policy.max_workers <= 32 or not 1 <= policy.max_in_flight <= 32):
            raise ValueError('full campaign requires v6, reader4096, <=32 workers/requests and stop_on_failure=False')
    elif policy.max_workers != 1 or not policy.stop_on_failure:
        raise ValueError('MAB initial admission requires sequential fail-closed jobs')
    supported = {'gluomem', 'mem0', 'cognee', 'no_memory', 'bm25', 'dense', 'hybrid', 'full_context'}
    systems = spec['systems']
    if not systems or len(set(systems)) != len(systems) or not set(systems) <= supported:
        raise ValueError('unadmitted MAB system')
    cognee_spec = None
    if 'cognee' in systems and not continuation:
        cognee_spec = _qualify_cognee_spec(spec, json.loads(Path(approval_path).read_bytes()), suite)
    canonical = {c.sample.id: c for c in suite.cases}
    ids = spec['case_ids']
    if (not ids or len(set(ids)) != len(ids) or not set(ids) <= canonical.keys()
            or [c.sample.id for c in suite.cases if c.sample.id in ids] != ids):
        raise ValueError('exact frozen question occurrence order required')
    selected = [canonical[key] for key in ids]
    if full and (ids != [c.sample.id for c in suite.cases]
                 or spec.get('case_manifest_sha256') != _digest([asdict(c) for c in suite.cases])):
        raise ValueError('full campaign requires entire pinned occurrence matrix and case_manifest_sha256')
    if continuation:
        _verify_reuse(spec, json.loads(Path(approval_path).read_bytes()), plan,
                      {c.sample.id: json.loads(json.dumps(asdict(c))) for c in selected}, suite)
    budget.check()
    scorer = suite.scorer()
    _ = scorer.upstream
    for case in selected:
        suite.reader_question(case)
        _qualify_judges(scorer.judge_requests(case, ''), profile, spec['judge_qualification'])
    root = Path(spec['output_root']).resolve()
    import gluomem
    if any(root.is_relative_to(p) for p in (Path(__file__).resolve().parents[2],
                                          Path(gluomem.__file__).resolve().parent.parent)):
        raise ValueError('campaign artifacts must be outside source repositories')
    if type(spec['chunk_chars']) is not int or spec['chunk_chars'] < 1:
        raise ValueError('positive chunk_chars required')
    if type(spec['top_k']) is not int or spec['top_k'] < 1:
        raise ValueError('positive effective top_k required')
    groups = {}
    for case in selected:
        groups.setdefault(case.sample.conv_id, []).append(case.sample.id)
    jobs, bindings, matrix = [], {}, []
    for record in spec['jobs']:
        if (record['history_id'] not in groups or record['system'] not in systems
                or record['case_ids'] != groups[record['history_id']]):
            raise ValueError('job differs from frozen history/question occurrence grouping')
        job = Job(record['job_id'], record['history_id'], Path(record['output']),
                  record['call_budget'], record['seconds_budget'])
        if (job.output == root or not job.output.is_relative_to(root)
                or job.output.is_relative_to(root / 'coordination') or job.output.exists()):
            raise ValueError('job requires fresh isolated path under campaign root')
        if job.call_budget > profile.max_generation_calls or job.seconds_budget > profile.max_run_seconds:
            raise ValueError('job envelope exceeds frozen whole-history profile')
        bindings[job.job_id] = record
        matrix.append((record['system'], record['history_id']))
        if not continuation or job.job_id in plan['dispatch_job_ids']:
            jobs.append(job)
    expected = {(system, hid) for system in systems for hid in groups}
    if len(set(matrix)) != len(matrix) or set(matrix) != expected:
        raise ValueError('exact complete system/history job matrix required')
    if len(bindings) != len(spec['jobs']):
        raise ValueError('duplicate job identity')
    budget.check()
    encoder = RemoteEmbedder(spec['embedding_url'], recipe=EmbeddingRecipe(**spec['embedding_recipe']))
    source = Path(spec['mab_root']) / 'data' / 'Accurate_Retrieval-00000-of-00001.parquet'
    cfg = controlled_config(profile, data=source, sha256=_hash(source), embedder=encoder)
    cfg = cfg.with_overrides({'benchmark.answer.context_k': spec['top_k']})
    root.mkdir(parents=True)
    coordination = None
    try:
        (root / 'execution-spec.json').write_bytes(Path(spec_path).read_bytes())
        (root / 'execution-approval.json').write_bytes(Path(approval_path).read_bytes())
        if cognee_spec is not None:
            (root / 'cognee-qualification.json').write_bytes(Path(cognee_spec['qualification']['path']).read_bytes())
            _write(root / 'cognee-source-manifest.json', cognee_source_manifest(suite))
        if full:
            _write(root / 'planned-cases.json', [asdict(c) for c in selected])
        from threading import Event
        cancel = Event()
        def worker(job, context):
            record = bindings[job.job_id]
            native_settings = None
            if record['system'] == 'cognee':
                from gluomem_bench.bench.comparison_backends import CogneeSettings
                native = cognee_spec['native']
                native_settings = CogneeSettings(
                    python_executable=native['python_executable'], root=job.output / 'graph/native',
                    dataset='mab-' + _digest({'history_id': job.history_id, 'job_id': job.job_id}),
                    llm_url=profile.base_url, llm_model=profile.model, llm_api_key='guard-bound-placeholder',
                    embedding_url=spec['embedding_url'], embedding_model=spec['embedding_recipe']['model'],
                    embedding_dimensions=spec['embedding_recipe']['dim'], embedding_api_key='local-placeholder',
                    max_completion_tokens=profile.output_budget('cognify'), chunk_size=native['chunk_size'],
                    timeout_seconds=min(native['timeout_seconds'], context.remaining()))
            try:
                return run_history(suite=suite, history_id=job.history_id,
                    case_ids=record['case_ids'], cfg=cfg, profile=profile, out=job.output,
                    system=record['system'], embedder=encoder, outer_budget=context,
                    chunk_chars=spec['chunk_chars'], judge_qualification=spec['judge_qualification'],
                    cognee_settings=native_settings,
                    cognee_admission=cognee_spec if record['system'] == 'cognee' else None,
                    _execution_authority=_EXECUTION_AUTHORITY)
            except Exception as exc:
                if full and (context.budget.failed or isinstance(exc, SharedRackError)
                             or getattr(exc, 'shared_failure', False)):
                    context.budget.failed = True
                    cancel.set()
                raise
        coordination = run_jobs(jobs, policy=policy, out=root / 'coordination', run_job=worker,
                                cancel=cancel, budget=budget)
        if not full and any(r['status'] != 'complete' for r in coordination['jobs']):
            raise RuntimeError('MAB campaign incomplete; see coordination/progress.json')
        records = []
        for job, progress in zip(jobs, coordination['jobs'], strict=True):
            record = bindings[job.job_id]
            status = progress['status']
            if full and status in ('cancelled', 'queued'):
                status = 'unrun'
            verification_error = None
            if status == 'complete':
                try:
                    leaf = verify_history(job.output)
                    if leaf['case_ids'] != record['case_ids'] or leaf['generation_calls'] != progress.get('calls', 0):
                        raise ValueError('verified leaf differs from frozen cases/spend')
                except Exception as exc:
                    if not full:
                        raise
                    status = 'failed'
                    verification_error = safe_error(exc)
            item = {'job_id': job.job_id, 'output': job.output.relative_to(root).as_posix(),
                            'system': record['system'], 'history_id': job.history_id,
                            'case_ids': record['case_ids'], 'generation_calls': progress.get('calls', 0)}
            if full:
                item['status'] = status
                if verification_error:
                    item['verification_error'] = verification_error
            records.append(item)
        if continuation:
            dispatched = {r['job_id']: r for r in records}
            records = []
            for record in spec['jobs']:
                disposition = record['disposition']
                if record['job_id'] in dispatched:
                    item = dispatched[record['job_id']]
                else:
                    item = {key: record[key] for key in ('job_id', 'system', 'history_id', 'case_ids')}
                    item.update(output=Path(record['output']).relative_to(root).as_posix(),
                        status={REUSE: 'complete', QUALITY_FAILED: 'failed', INFRA_HELD: 'held',
                                NATIVE_UNRUN: 'unrun'}[disposition],
                        generation_calls=record.get('prior_generation_attempts', 0))
                    if disposition == REUSE:
                        item['reused_output'] = record['original_output']
                item['disposition'] = disposition
                records.append(item)
        result = {'complete': all(r['status'] == 'complete' for r in records) if full else True,
                  'protocol': PROTOCOL, 'jobs': records,
                  'generation_calls': coordination['accounting']['calls'],
                  'artifact_sha256': {p.relative_to(root).as_posix(): _hash(p)
                                     for p in sorted(root.rglob('*')) if p.is_file()}}
        if full:
            result.update(campaign_mode=mode, coordinator_complete=True,
                          **_full_scope(root, records, spec))
        if continuation:
            result.update(prior_generation_calls=plan['prior_calls'],
                          new_generation_calls=budget.calls - plan['prior_calls'],
                          accounting=budget.snapshot(), cohort='fresh600s-serial1; reused leaves retain old profile')
        def finalization_check():
            # A shared failure forbids further dispatch, not an honest incomplete
            # scope receipt. The same absolute deadline still bounds publication.
            if budget.clock() >= budget.deadline:
                raise RuntimeError('campaign deadline exhausted during finalization')
        return _publish_campaign(root, result,
            SimpleNamespace(check=finalization_check, snapshot=budget.snapshot))
    except Exception as exc:
        spend = coordination['accounting']['calls'] if coordination is not None else None
        if hasattr(exc, 'receipt'):
            spend = exc.receipt.get('accounting', {}).get('calls')
        exc.__dict__['generation_calls'] = spend
        try:
            (root / 'root-status.json').unlink(missing_ok=True)
        except OSError as cleanup_error:
            exc.add_note(f'could not remove incomplete root status: {safe_error(cleanup_error)}')
        try:
            _write(root / 'failure.json', {'complete': False, 'generation_calls': spend,
                                          'error': safe_error(exc)})
        except Exception as persistence_error:
            exc.add_note(
                f'failure receipt unavailable; generation_calls={spend}; '
                f'{safe_error(persistence_error)}'
            )
        raise


def _verify_campaign_payload(root, result):
    if result.get('campaign_mode') == CONTINUATION_MODE:
        return _verify_continuation_payload(root, result)
    full = result.get('campaign_mode') == FULL_MODE
    if (not full and result.get('complete') is not True) or result.get('protocol') != PROTOCOL:
        raise ValueError('campaign root incomplete')
    required = {'execution-spec.json', 'execution-approval.json',
                'coordination/plan.json', 'coordination/progress.json'}
    inventory = result['artifact_sha256']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p not in (root / 'root-status.json', root / 'root-status.pending.json')}
    if not required <= inventory.keys() or set(inventory) != actual:
        raise ValueError('frozen campaign evidence inventory incomplete')
    for relative, expected in result['artifact_sha256'].items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or _hash(path) != expected:
            raise ValueError('campaign artifact hash mismatch')
    spec_bytes = (root / 'execution-spec.json').read_bytes()
    spec = json.loads(spec_bytes)
    if full != (spec.get('campaign_mode') == FULL_MODE):
        raise ValueError('frozen campaign mode differs')
    planned = {}
    if full:
        frozen_cases = json.loads((root / 'planned-cases.json').read_bytes())
        if _digest(frozen_cases) != spec.get('case_manifest_sha256'):
            raise ValueError('frozen planned case manifest differs')
        planned = {c['sample']['id']: c for c in frozen_cases}
    if type(spec.get('evidence_token_budget')) is not int or spec['evidence_token_budget'] != EVIDENCE_TOKEN_BUDGET:
        raise ValueError('frozen campaign evidence_token_budget=4096 required')
    approval = json.loads((root / 'execution-approval.json').read_bytes())
    if (approval.get('approved') is not True
            or approval.get('spec_sha256') != hashlib.sha256(spec_bytes).hexdigest()
            or spec.get('execution_kind') != PROTOCOL):
        raise ValueError('frozen campaign approval binding differs')
    cognee_manifest = {}
    if 'cognee' in spec['systems']:
        entry = spec.get('cognee', {})
        manifest = json.loads((root / 'cognee-source-manifest.json').read_bytes())
        if (not full or entry.get('scope') != 'all-pinned-mab-history-question-occurrences'
                or entry.get('qualification', {}).get('sha256') != COGNEE_QUALIFICATION_SHA256
                or approval.get('approved_gates', {}).get('cognee_native_graph') != COGNEE_QUALIFICATION_SHA256
                or _hash(root / 'cognee-qualification.json') != COGNEE_QUALIFICATION_SHA256
                or _digest(manifest) != entry.get('source_manifest_sha256')
                or [m['history_id'] for m in manifest] != list(dict.fromkeys(
                    c['sample']['conv_id'] for c in frozen_cases))):
            raise ValueError('Cognee frozen native qualification/source scope differs')
        cognee_manifest = {m['history_id']: m for m in manifest}
    declared = [{key: job[key] for key in ('job_id', 'history_id', 'system', 'case_ids')}
                | {'output': Path(job['output']).resolve().relative_to(
                    Path(spec['output_root']).resolve()).as_posix()} for job in spec['jobs']]
    observed = [{key: job[key] for key in ('job_id', 'history_id', 'system', 'case_ids', 'output')}
                for job in result['jobs']]
    if not declared or declared != observed:
        raise ValueError('complete frozen system/history/occurrence matrix coverage required')
    progress = json.loads((root / 'coordination' / 'progress.json').read_bytes())
    if ([job['job_id'] for job in progress['jobs']] != [job['job_id'] for job in declared]
            or (not full and any(job['status'] != 'complete' for job in progress['jobs']))
            or progress['accounting']['calls'] != result['generation_calls']):
        raise ValueError('frozen campaign coordinator coverage/spend differs')
    for record, progress_job in zip(result['jobs'], progress['jobs'], strict=True):
        if full:
            status = record['status']
            expected_status = 'unrun' if progress_job['status'] == 'cancelled' else progress_job['status']
            if (status not in ('complete', 'failed', 'unrun')
                    or (status != expected_status and not (status == 'failed'
                        and expected_status == 'complete' and record.get('verification_error')))
                    or record['generation_calls'] != progress_job.get('calls', 0)):
                raise ValueError('full campaign condition status/spend binding differs')
            if status != 'complete':
                continue
        leaf = verify_history(root / record['output'])
        if leaf['case_ids'] != record['case_ids'] or leaf['generation_calls'] != record['generation_calls']:
            raise ValueError('campaign leaf coverage/spend mismatch')
        leaf_out = root / record['output']
        if full and json.loads((leaf_out / 'cases.json').read_bytes()) != [planned[key] for key in record['case_ids']]:
            raise ValueError('leaf differs from frozen planned cases')
        protocol = json.loads((leaf_out / 'protocol.json').read_bytes())
        if record['system'] == 'cognee':
            from gluomem.write.events import SourceEvent
            events = [SourceEvent.model_validate(e) for e in json.loads((leaf_out / 'source-inputs.json').read_bytes())]
            manifest = cognee_manifest[record['history_id']]
            runtime = json.loads((leaf_out / 'cognee-runtime.json').read_bytes())
            if (protocol.get('cognee_admission') != spec['cognee']
                    or protocol['context_sha256'] != manifest['context_sha256']
                    or _digest([asdict(d) for d in cognee_documents(events, chunk_chars=spec['chunk_chars'])])
                        != manifest['documents_sha256']
                    or runtime['python_executable'] != spec['cognee']['native']['python_executable']
                    or runtime['package_sha256'] != spec['cognee']['native']['package_sha256']
                    or runtime['chunk_size'] != spec['cognee']['native']['chunk_size']):
                raise ValueError('Cognee native source/protocol scope binding differs')
        leaf_profile = json.loads((leaf_out / 'profile.json').read_bytes())
        if (protocol['system'] != record['system'] or protocol['history_id'] != record['history_id']
                or protocol['case_ids'] != record['case_ids'] or protocol['top_k'] != spec['top_k']
                or protocol['chunk_chars'] != spec['chunk_chars']
                or protocol['evidence_token_budget'] != spec['evidence_token_budget']
                or protocol['judge_qualification'] != spec['judge_qualification']
                or leaf_profile != spec['profile']):
            raise ValueError('campaign frozen leaf protocol/profile binding differs')
    if sum(r['generation_calls'] for r in result['jobs']) != result['generation_calls']:
        raise ValueError('campaign call reconciliation mismatch')
    if full:
        if (result.get('complete') is not all(r['status'] == 'complete' for r in result['jobs'])
                or result.get('coordinator_complete') is not True):
            raise ValueError('full campaign completion scope differs')
        scope = _full_scope(root, result['jobs'], spec)
        if any(result.get(key) != value for key, value in scope.items()):
            raise ValueError('full campaign graded coverage/native metrics differ')
    return result


def _record_output(root, record):
    return Path(record['reused_output']) if 'reused_output' in record else root / record['output']


def _verify_native_source(out, record, suite, chunk_chars):
    history = next(h for h in suite.histories if h.id == record['history_id'])
    expected = [e.model_dump(mode='json') for e in document_events(history, chunk_chars=chunk_chars)]
    protocol = json.loads((out / 'protocol.json').read_bytes())
    if (json.loads((out / 'source-inputs.json').read_bytes()) != expected
            or protocol['context_sha256'] != history.context_sha256
            or protocol['family'] != history.family or protocol['source'] != history.source
            or protocol['row_index'] != history.row_index):
        raise ValueError('continuation native source/context binding differs')


def _verify_continuation_payload(root, result):
    required = {'execution-spec.json', 'execution-approval.json', 'planned-cases.json',
                'coordination/plan.json', 'coordination/progress.json'}
    inventory = result['artifact_sha256']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and p not in (root / 'root-status.json', root / 'root-status.pending.json')}
    if not required <= inventory.keys() or set(inventory) != actual:
        raise ValueError('continuation complete root evidence inventory differs')
    for name, digest in inventory.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or _hash(path) != digest:
            raise ValueError('continuation root artifact hash mismatch')
    spec = json.loads((root / 'execution-spec.json').read_bytes())
    approval = json.loads((root / 'execution-approval.json').read_bytes())
    if (spec.get('campaign_mode') != CONTINUATION_MODE or spec.get('execution_kind') != PROTOCOL
            or result.get('protocol') != PROTOCOL or approval.get('approved') is not True
            or approval.get('spec_sha256') != _hash(root / 'execution-spec.json')):
        raise ValueError('continuation frozen spec/approval differs')
    frozen = json.loads((root / 'planned-cases.json').read_bytes())
    if (_digest(frozen) != spec['case_manifest_sha256']
            or [c['sample']['id'] for c in frozen] != spec['case_ids']):
        raise ValueError('continuation native case occurrence order differs')
    planned = {c['sample']['id']: c for c in frozen}
    groups = {}
    for c in frozen:
        groups.setdefault(c['sample']['conv_id'], []).append(c['sample']['id'])
    pairs = [(r['system'], r['history_id']) for r in spec['jobs']]
    if (len(set(pairs)) != len(pairs) or set(pairs) != {
            (s, h) for s in spec['systems'] for h in groups}
            or any(r['case_ids'] != groups[r['history_id']] for r in spec['jobs'])):
        raise ValueError('continuation full native system/history coverage differs')
    plan = continuation_plan(spec)
    from gluomem_bench.bench.mab_suite import MABSuite
    suite = MABSuite.from_directory(spec['mab_root'])
    if json.loads(json.dumps([asdict(c) for c in suite.cases])) != frozen:
        raise ValueError('continuation native source/case reconstruction differs')
    _verify_reuse(spec, approval, plan, planned, suite)
    progress = json.loads((root / 'coordination/progress.json').read_bytes())
    scheduled = json.loads((root / 'coordination/plan.json').read_bytes())
    dispatch = [r for r in spec['jobs'] if r['job_id'] in plan['dispatch_job_ids']]
    expected_scheduled = [{k: r[k] for k in ('job_id', 'history_id', 'output', 'call_budget', 'seconds_budget')}
                          for r in dispatch]
    if (scheduled['jobs'] != expected_scheduled or scheduled['policy'] != spec['policy']
            or [r['job_id'] for r in progress['jobs']] != plan['dispatch_job_ids']):
        raise ValueError('continuation scheduled subset/order differs')
    progress_rows = {r['job_id']: r for r in progress['jobs']}
    if len(result['jobs']) != len(spec['jobs']):
        raise ValueError('continuation full matrix row count differs')
    for record, declared in zip(result['jobs'], spec['jobs'], strict=True):
        if (any(record[k] != declared[k] for k in ('job_id', 'system', 'history_id', 'case_ids', 'disposition'))
                or record['output'] != Path(declared['output']).relative_to(Path(spec['output_root'])).as_posix()):
            raise ValueError('continuation full matrix row binding differs')
        disposition = declared['disposition']
        if disposition != NEVER_DISPATCHED:
            status = {REUSE: 'complete', QUALITY_FAILED: 'failed', INFRA_HELD: 'held', NATIVE_UNRUN: 'unrun'}[disposition]
            if (record['status'] != status or record['generation_calls'] != declared.get('prior_generation_attempts', 0)
                    or (disposition == REUSE and record.get('reused_output') != declared['original_output'])
                    or (disposition != REUSE and 'reused_output' in record)):
                raise ValueError('continuation retained failed/held/unrun/reuse boundary differs')
            continue
        row = progress_rows[record['job_id']]
        status = 'unrun' if row['status'] in ('cancelled', 'queued') else row['status']
        if ('reused_output' in record or record['generation_calls'] != row.get('calls', 0)
                or (record['status'] != status and not (record['status'] == 'failed'
                    and status == 'complete' and record.get('verification_error')))):
            raise ValueError('continuation new dispatch status/spend differs')
        if record['status'] != 'complete':
            continue
        out = root / record['output']
        _verify_native_source(out, record, suite, spec['chunk_chars'])
        leaf = verify_history(out)
        protocol = json.loads((out / 'protocol.json').read_bytes())
        if (leaf['case_ids'] != record['case_ids'] or leaf['generation_calls'] != record['generation_calls']
                or json.loads((out / 'cases.json').read_bytes()) != [planned[k] for k in record['case_ids']]
                or json.loads((out / 'profile.json').read_bytes()) != spec['profile']
                or protocol['system'] != record['system'] or protocol['history_id'] != record['history_id']
                or any(protocol[k] != spec[k] for k in
                       ('chunk_chars', 'top_k', 'evidence_token_budget', 'judge_qualification'))):
            raise ValueError('continuation new complete leaf binding differs')
    new_calls = sum(r.get('calls', 0) for r in progress['jobs'])
    accounting = result['accounting']
    if (result['prior_generation_calls'] != plan['prior_calls'] or result['new_generation_calls'] != new_calls
            or result['generation_calls'] != plan['prior_calls'] + new_calls
            or sum(r['generation_calls'] for r in result['jobs']) != result['generation_calls']
            or progress['accounting']['calls'] != result['generation_calls']
            or accounting['calls'] != result['generation_calls']
            or accounting['prior_calls'] != plan['prior_calls'] or accounting['new_calls'] != new_calls
            or accounting['prior_elapsed_seconds'] != plan['prior_elapsed_seconds']
            or accounting['elapsed_seconds'] != accounting['prior_elapsed_seconds'] + accounting['new_elapsed_seconds']
            or accounting['elapsed_seconds'] < progress['accounting']['elapsed_seconds']
            or accounting['elapsed_seconds'] >= spec['policy']['max_seconds']
            or result['generation_calls'] > spec['policy']['max_calls']):
        raise ValueError('continuation prior/new call/time reconciliation differs')
    scope = _full_scope(root, result['jobs'], spec)
    if (result.get('complete') is not all(r['status'] == 'complete' for r in result['jobs'])
            or result.get('coordinator_complete') is not True
            or any(result.get(k) != v for k, v in scope.items())):
        raise ValueError('continuation complete native coverage/metrics differ')
    return result


def _full_scope(root, records, spec):
    """Dataset-free completed-condition metrics, retaining all planned denominators."""
    from gluomem_bench.bench.mab_suite import MABCase, MABSuite, NativeScore
    from gluomem_bench.bench.types import Sample
    frozen = json.loads((root / 'planned-cases.json').read_bytes())
    if (_digest(frozen) != spec.get('case_manifest_sha256')
            or [c['sample']['id'] for c in frozen] != spec['case_ids']):
        raise ValueError('full frozen planned case manifest differs')
    cases = [MABCase(**{**c, 'sample': Sample(**c['sample']), 'answers': tuple(c['answers'])}) for c in frozen]
    histories = {}
    for case in cases:
        histories.setdefault(case.sample.conv_id, SimpleNamespace(id=case.sample.conv_id,
            source=case.source, family=case.family, context_sha256=None))
    suite = MABSuite(list(histories.values()), cases)
    coverage, metrics = {}, {}
    for system in spec['systems']:
        scores, rows, eligible = [], [], []
        counts: dict = dict(planned=0, complete=0, failed=0, unrun=0, graded=0, not_applicable=0)
        if spec.get('campaign_mode') == CONTINUATION_MODE:
            counts['held'] = 0
        for record in records:
            if record['system'] != system:
                continue
            counts['planned'] += len(record['case_ids'])
            counts[record['status']] += len(record['case_ids'])
            if record['status'] != 'complete':
                continue
            saved = json.loads((_record_output(root, record) / 'predictions.json').read_bytes())
            for row in saved:
                rows.append(row)
                if row.get('status') == 'not_applicable':
                    counts['not_applicable'] += 1
                else:
                    scores.append(NativeScore(**row['score']))
                    eligible.append(row['case_id'])
                    counts['graded'] += 1
        counts['eligible_case_ids'] = eligible
        coverage[system] = counts
        metrics[system] = _summary(suite, scores, rows)
    return {'coverage': coverage, 'native_metrics': metrics,
            'planned_history_occurrences': len(histories), 'planned_question_occurrences': len(cases)}


def verify_campaign(root):
    root = Path(root)
    return _verify_campaign_payload(root, json.loads((root / 'root-status.json').read_bytes()))


def admission():
    """Executable scope and honest blocked seams; not an inference approval."""
    return {
        'protocol': PROTOCOL,
        'executable_systems': ['gluomem', 'mem0', 'cognee', 'no_memory', 'bm25', 'dense', 'hybrid', 'full_context'],
        'required_spec_fields': {'evidence_token_budget': EVIDENCE_TOKEN_BUDGET},
        'continuation': {'campaign_mode': CONTINUATION_MODE,
            'bridge_protocol': 'mab-readonly-semantic-bridge@v1',
            'bindings': ['continuation.original_spec', 'continuation.original_root',
                         'continuation.original_archive', 'continuation.semantic_bridge'],
            'approval_gate': 'approved_gates.mab_continuation_bridge',
            'bridge_fields': ['original_bindings', 'candidate_semantics_sha256',
                              'current_wrapper_sha256', 'reuse_job_ids', 'producer_semantics'],
            'producer_semantics_fields': ['original_path', 'original_sha256', 'current_path', 'current_sha256'],
            'prior_accounting': 'accounting_continuation.prior.coordinator',
            'policy_caps': 'original total ceilings; prior calls/time imported once',
            'operating_point': 'fresh600s-serial1; old complete profiles retained',
            'cognee': 'all rows unrun; requires separately versioned wrapper-native qualification'},
        'full_campaign': {'campaign_mode': FULL_MODE, 'profile_version': 'role-budgets-v6',
            'recommended_workers': 8, 'max_workers': 32, 'stop_on_failure': False,
            'reader_tokens': 4096, 'judge_tokens': 4096,
            'selection': 'all pinned history and question occurrences; no sampled full mode'},
        'mem0_document_api': 'stock 2.2.1 memory.add(str, infer=True, user_id=history.id); guarded shared budget/encoder',
        'cognee_document_api': {'factory': 'CogneeRetrieverFactory(documents=tuple[Evidence, ...])',
            'backend': 'CogneeGraphEvidenceBackend; native graph context, never CHUNKS or model answer',
            'full_only': True, 'qualification_sha256': COGNEE_QUALIFICATION_SHA256,
            'spec_fields': ['cognee.qualification', 'cognee.scope', 'cognee.source_manifest_sha256',
                            'cognee.native.python_executable', 'cognee.native.chunk_size',
                            'cognee.native.timeout_seconds', 'cognee.native.package_sha256'],
            'approval_gate': 'approved_gates.cognee_native_graph',
            'source_projection': 'same raw body and source_ref/offset as controls; no QA, date or roles'},
        'unadmitted_systems': {},
        'live_summary_judge_gate': ('Native summary requests require 4096 output tokens and temperature 0.1; '
                                   'explicit role-budgets-v6 plus native-judge-parameters@v2 preserves each request. '
                                   'Legacy profiles retain judge <=1024 and temperature 0 and refuse this path.'),
        'quality_claims_admitted': False,
        'remaining_gates': ['operator execution-spec byte approval',
                            'loaded embedding weights/device/role parity qualification',
                            'served tokenizer/template and role output feasibility qualification',
                            'explicit adapted character-document ingestion acceptance',
                            'Recsys fixed process hash seed/candidate-order qualification',
                            'derived-corpus licensing review and independent final evidence review'],
    }


def main(argv=None):
    """python -m gluomem_bench.bench.mab_campaign --spec SPEC --approval APPROVAL."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--spec', type=Path)
    mode.add_argument('--verify-root', type=Path)
    mode.add_argument('--admission', action='store_true')
    parser.add_argument('--approval', type=Path)
    args = parser.parse_args(argv)
    if args.spec is not None:
        if args.approval is None:
            parser.error('--spec requires --approval')
        result = execute(args.spec, args.approval)
    elif args.verify_root is not None:
        if args.approval is not None:
            parser.error('--approval is only valid with --spec')
        result = verify_campaign(args.verify_root)
    else:
        if args.approval is not None:
            parser.error('--approval is only valid with --spec')
        result = admission()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
