"""Fresh, explicitly changed LME study; legacy validators remain unchanged.

No command authorizes itself. Freeze only after integration is byte-stable;
execute requires an independent review and separate parent authority.
"""
from __future__ import annotations

import copy
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gluomem_bench.bench import lme_first_execute as native
from gluomem_bench.bench.lme_first import json_bytes, read_bound
from gluomem_bench.bench.lme_first_continue import NewAttemptBudget
from gluomem_bench.bench.parallel_campaign import CampaignBudget, CampaignStopped, guard_stopped

PROTOCOL = 'lme-remediation-study@v1'
SYSTEMS = ('gluomem', 'mem0', 'cognee', 'no_memory')
SOURCE = Path(__file__).resolve().parents[2]


class BudgetScope(NewAttemptBudget):
    """Narrow admission coordinates over the same original counter and lock."""

    def __init__(self, budget, *, max_calls, max_seconds):
        import math
        budget.check()
        if (type(max_calls) is not int or not 0 < max_calls <= budget.max_calls - budget.calls - budget.reserved_calls
                or type(max_seconds) not in (int, float) or not math.isfinite(max_seconds)
                or not 0 < max_seconds <= budget.remaining()):
            raise ValueError('scope exceeds original remaining allowance')
        super().__init__(budget)
        self.offset = budget.calls
        self.started = budget.clock()
        self.max_calls, self.max_seconds = max_calls, max_seconds
        self.deadline = min(budget.deadline, self.started + max_seconds)

    @property
    def calls(self):
        return self.outer.calls - self.offset

    @calls.setter
    def calls(self, value):
        self.outer.calls = self.offset + value

    def check(self):
        self.outer.check()
        if self.clock() >= self.deadline:
            self.failed = True
            raise guard_stopped(CampaignStopped('bounded scope deadline exhausted'), 'campaign-deadline')

    def begin_call(self):
        self.check()
        if self.calls >= self.max_calls:
            self.failed = True
            raise guard_stopped(CampaignStopped('bounded scope call allowance exhausted'), 'campaign-call-cap')
        self.outer.begin_call()

    def snapshot(self):
        return {'calls': self.calls, 'elapsed_seconds': self.clock() - self.started}


def clock_host():
    if os.name == 'nt':
        import ctypes
        data = ctypes.create_string_buffer(64)
        status = ctypes.windll.ntdll.NtQuerySystemInformation(90, data, len(data), None)
        if status != 0:
            raise ValueError('boot identity unavailable')
        boot = data.raw[:16].hex()
    else:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    return {'host': platform.node(), 'boot_id': boot}


def authority(binding):
    current = item = load(binding)
    seen = set()
    while True:
        digest = json_bytes(item)
        if digest in seen:
            raise ValueError('cyclic budget authority')
        seen.add(digest)
        if (item.get('protocol') != 'lme-study-budget-authority@v1'
            or not isinstance(item.get('historical_unknown'), str) or not item['historical_unknown'].strip()
            or item.get('clock_host') != clock_host()):
            raise ValueError('explicit budget authority on the original host/boot required')
        # Constructor validates numeric caps, debit and origin without granting fresh time.
        CampaignBudget(max_calls=item['max_calls'], max_seconds=item['max_seconds'],
            prior_calls=item['retained_calls'], prior_elapsed_seconds=item['retained_reserved_seconds'],
            started_monotonic=item['clock_origin'])
        load(item['debit_evidence'])
        if item['retained_calls'] >= item['max_calls'] or item['retained_reserved_seconds'] >= item['max_seconds']:
            raise ValueError('positive future phase allowance required')
        if 'previous_authority' not in item:
            operator = load(item['operator'])
            if item != dict(operator, operator=item['operator'], clock_origin=item['clock_origin'], clock_host=item['clock_host']):
                raise ValueError('initial authority differs from bound operator declaration')
            break
        previous = load(item['previous_authority'])
        if (any(item[k] != previous[k] for k in ('max_calls', 'max_seconds', 'retained_reserved_seconds',
                'clock_origin', 'clock_host', 'historical_unknown', 'operator'))
            or type(item['retained_calls']) is not int or item['retained_calls'] < previous['retained_calls']):
            raise ValueError('budget refund, ceiling change or clock reset refused')
        item = previous
    return current


def campaign_budget(binding, *, clock=time.monotonic):
    item = authority(binding)
    budget = CampaignBudget(max_calls=item['max_calls'], max_seconds=item['max_seconds'],
        prior_calls=item['retained_calls'], prior_elapsed_seconds=item['retained_reserved_seconds'],
        started_monotonic=item['clock_origin'], clock=clock)
    budget.authority = binding
    budget.check()
    return budget


def bootstrap(operator_binding, out, *, clock=time.monotonic):
    origin = clock()  # Before operator hashing, mkdir or any preparation work.
    item = load(operator_binding)
    if set(item) != {'protocol', 'max_calls', 'max_seconds', 'retained_calls',
                     'retained_reserved_seconds', 'historical_unknown', 'debit_evidence'}:
        raise ValueError('explicit initial operator declaration required')
    budget = CampaignBudget(max_calls=item['max_calls'], max_seconds=item['max_seconds'],
        prior_calls=item['retained_calls'], prior_elapsed_seconds=item['retained_reserved_seconds'],
        started_monotonic=origin, clock=clock)
    load(item['debit_evidence'])
    if item['protocol'] != 'lme-study-budget-authority@v1' or not isinstance(item['historical_unknown'], str) or not item['historical_unknown'].strip():
        raise ValueError('explicit phase and unknown historical accounting required')
    budget.check()
    result = write_new(out, dict(item, operator=operator_binding, clock_origin=origin, clock_host=clock_host()))
    budget.check()
    return result


def load(binding):
    return json.loads(read_bound(binding['path'], binding['sha256']))


def write_new(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    return native.bound(path)


def assemble(binding):
    spec = load(binding)
    debit = authority(spec['budget_authority'])
    base, profile = load(spec['base_plan']), load(spec['profile_file'])
    ids, selected, systems = base['sample_ids'], spec['selected_ids'], spec['selected_systems']
    remediation = spec.get('remediation', {})
    if type(remediation) is not dict:
        raise ValueError('remediation must be an explicit JSON object; consuming bridges validate feature schemas')
    json_bytes(remediation)
    source_pairs = [(r['sample_id'], r['system']) for r in base['jobs']]
    primary_pairs = [(sid, s) for sid in ids for s in SYSTEMS[:3]] + [(sid, 'no_memory') for sid in ids]
    if (spec['protocol'] != 'lme-remediation-study-spec@v1'
        or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', spec['study_id'])
        or len(ids) != 500 or len(set(ids)) != 500
        or source_pairs not in (primary_pairs, [(sid, s) for sid in ids for s in SYSTEMS])
        or spec['stage'] not in ('development', 'stage', 'full')
        or not selected or selected != [sid for sid in ids if sid in selected]
        or not systems or systems != [s for s in SYSTEMS if s in systems]
        or spec['stage'] == 'full' and (selected != ids or systems != list(SYSTEMS))
        or not isinstance(spec['changed_condition'], str) or not spec['changed_condition'].strip()
        or set(spec['methods']) != set(SYSTEMS)
        or any(m.get('kind') not in ('native', 'adapted') or not m.get('version') for m in spec['methods'].values())
        or type(spec['local_memory_mib']) is not int or not 8192 < spec['local_memory_mib'] <= 32678
        or spec['experimental_overrides']):
        raise ValueError('explicit fresh study declaration required; unsupported experimental override')
    for key in ('model', 'context_window', 'temperature'):
        if profile[key] != base['profile'][key]:
            raise ValueError('common model/context/temperature changed')
    from gluomem_bench.bench.rack import RackProfile
    RackProfile.model_validate(profile)
    plan = copy.deepcopy(base)
    plan.update(protocol=PROTOCOL, declaration=binding, execution_authorized=False,
                scope='explicit changed-condition study; only stage=full covers the released primary target',
                new_experiment=True, historical_quality_pooled=False, profile=profile,
                selected_ids=selected, selected_systems=systems, stage=spec['stage'],
                changed_condition=spec['changed_condition'], methods=spec['methods'],
                remediation=copy.deepcopy(remediation), experimental_overrides={}, full_scope=spec['stage'] == 'full',
                budget_authority=spec['budget_authority'],
                future_budget_proposal={'max_calls': debit['max_calls'] - debit['retained_calls'],
                                        'max_seconds': debit['max_seconds'] - debit['retained_reserved_seconds']},
                historical_accounting={k: debit[k] for k in ('retained_calls', 'retained_reserved_seconds', 'historical_unknown')},
                operating_point={'max_in_flight': 300, 'max_in_flight_tokens': 39321600,
                                 'native_workers': 'explicit byte-bound runtime template', 'retries': 0})
    plan['config_overrides'].update(spec['config_overrides'])
    plan['native_constraints']['local_memory_mib'] = spec['local_memory_mib']
    plan['full_matrix'] = [dict(job_id=f"{spec['study_id']}--{i:04d}--{s}", sample_id=sid,
        history_id=f"{spec['study_id']}:{s}:{sid}", system=s,
        phase='control-after-primary' if s == 'no_memory' else 'primary',
        call_budget=profile['max_generation_calls'], seconds_budget=profile['max_run_seconds'])
        for i, (sid, s) in enumerate([(sid, s) for sid in ids for s in SYSTEMS[:3]] +
                                     [(sid, 'no_memory') for sid in ids])]
    if set(r['job_id'] for r in base['jobs']) & set(r['job_id'] for r in plan['full_matrix']):
        raise ValueError('historical job identities cannot be reused')
    plan['jobs'] = [r for r in plan['full_matrix'] if r['sample_id'] in selected and r['system'] in systems]
    return plan


def validate_plan(plan):
    if plan != assemble(plan['declaration']):
        raise ValueError('study plan differs from bound declaration')
    return plan


def prepare(specification, out, *, budget):
    budget.check()
    plan = assemble(specification)
    if plan['budget_authority'] != budget.authority:
        raise ValueError('preparation budget authority differs')
    budget.check()
    return write_new(out, plan)


def freeze(plan_binding, template_binding, output_root, out, *, budget):
    budget.check()
    plan = validate_plan(load(plan_binding))
    if plan['budget_authority'] != budget.authority:
        raise ValueError('freeze budget authority differs')
    worker = copy.deepcopy(load(template_binding))
    root, out = Path(output_root).resolve(), Path(out).resolve()
    if (root.exists() or root.is_symlink() or SOURCE.is_relative_to(root) or root.is_relative_to(SOURCE)
        or root.is_relative_to(out.parent) or out.parent.is_relative_to(root)
        or any(p.exists() for p in (out, out.with_suffix('.worker.json'), out.with_suffix('.source.json'),
                                    out.with_suffix('.objects')))):
        raise ValueError('fresh disjoint output and freeze paths required')
    old_source = Path(worker.pop('source_root')).resolve()
    files = [r for r in worker['files'] if not Path(r['path']).resolve().is_relative_to(old_source)]
    objects = out.with_suffix('.objects')
    source = {}
    for path in sorted(SOURCE.rglob('*'), key=lambda p: p.relative_to(SOURCE).as_posix()):
        budget.check()
        if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo'):
            continue
        item = native.bound(path)
        snapshot = objects/item['sha256']
        if not snapshot.exists():
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            with path.open('rb') as src, snapshot.open('xb') as dst:
                import shutil
                shutil.copyfileobj(src, dst, 1048576)
        files.append(dict(item, snapshot=str(snapshot), worker_check=True))
        source[path.relative_to(SOURCE).as_posix()] = item['sha256']
    worker.update(plan=plan_binding, files=files, output_root=str(root), execution_authorized=False)
    source_binding = write_new(out.with_suffix('.source.json'), source)
    worker_binding = write_new(out.with_suffix('.worker.json'), worker)
    packet = dict(protocol=PROTOCOL, execution_authorized=False, plan=plan_binding,
                  worker=worker_binding, source_manifest=source_binding, runtime_template=template_binding,
                  source_root=str(SOURCE), output_root=str(root), producer_id='remediation-study-v1',
                  retained_debit={'calls': budget.prior_calls, 'reserved_seconds': budget.prior_elapsed_seconds})
    budget.check()
    result = write_new(out, packet)
    validate_candidate(out, budget=budget)
    return result


def validate_candidate(path, *, budget, approval=None, post_exit=False):
    budget.check()
    packet = json.loads(Path(path).read_bytes())
    if (packet.get('protocol') != PROTOCOL or packet.get('execution_authorized') is not False
        or packet.get('source_root') != str(SOURCE)
        or packet.get('retained_debit') != {'calls': budget.prior_calls, 'reserved_seconds': budget.prior_elapsed_seconds}):
        raise ValueError('bound study/debit contract required')
    plan, worker, source = validate_plan(load(packet['plan'])), load(packet['worker']), load(packet['source_manifest'])
    if plan['budget_authority'] != budget.authority:
        raise ValueError('candidate budget authority differs')
    root = Path(packet['output_root'])
    if ((not post_exit and (root.exists() or root.is_symlink()))
        or root.with_name(root.name+'.revoked.json').exists()
        or root.is_relative_to(SOURCE) or SOURCE.is_relative_to(root)
        or worker['output_root'] != str(root) or worker['plan'] != packet['plan']
        or worker['protocol'] != native.EXECUTION_PROTOCOL
        or worker['host_reserve_mib'] != 8192
        or any(v != 4096 for v in worker['history_commit_limits_mib'].values())
        or worker.get('gluomem_group_commit_limit_mib', 4096) != 4096):
        raise ValueError('fresh unchanged OS caps/reserve/worker envelope required')
    paths = {str(Path(r['path']).resolve()): r for r in worker['files']}
    if len(paths) != len(worker['files']):
        raise ValueError('duplicate runtime closure path')
    template = copy.deepcopy(load(packet['runtime_template']))
    old_source = Path(template.pop('source_root')).resolve()
    retained = [r for r in template['files'] if not Path(r['path']).resolve().is_relative_to(old_source)]
    expected_worker = dict(template, plan=packet['plan'], output_root=str(root), execution_authorized=False,
        files=retained + [dict(path=str(SOURCE/rel), sha256=sha, worker_check=True,
            snapshot=str(Path(path).resolve().with_suffix('.objects')/sha)) for rel, sha in sorted(source.items())])
    if worker != expected_worker:
        raise ValueError('source/runtime launch rebinding differs')
    for item in worker['files']:
        budget.check()
        native.check_file(item['path'], item['sha256'])
        native.check_file(item['snapshot'], item['sha256'])
    actual = {p.relative_to(SOURCE).as_posix(): native.bound(p)['sha256'] for p in SOURCE.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')}
    if actual != source or any(paths.get(str(SOURCE/rel), {}).get('sha256') != sha
         or not paths.get(str(SOURCE/rel), {}).get('worker_check', True) for rel, sha in source.items()):
        raise ValueError('source closure binding differs')
    if approval is not None:
        for name, module in list(sys.modules.items()):
            if (name.split('.')[0] in ('gluomem', 'gluomem_bench') and getattr(module, '__file__', None)
                and (not Path(module.__file__).resolve().is_relative_to(SOURCE)
                     or str(Path(module.__file__).resolve()) not in paths)):
                raise ValueError('actual imported package closure differs')
        review = load(approval['independent_review'])
        if (approval.get('protocol') != PROTOCOL+'-parent-approval'
            or approval.get('approver_kind') != 'parent' or approval.get('approved') is not True
            or approval.get('execution_authorized') is not True
            or approval.get('candidate_sha256') != native.bound(path)['sha256']
            or approval.get('retained_debit') != packet['retained_debit']
            or review.get('approved') is not True or review.get('blockers') != []
            or review.get('candidate_sha256') != approval['candidate_sha256']
            or review.get('source_manifest_sha256') != packet['source_manifest']['sha256']
            or review.get('reviewer_kind') not in ('human', 'AI-assisted')
            or not review.get('reviewer_id') or review['reviewer_id'] == packet['producer_id']):
            raise ValueError('separate byte-bound independent review and parent authority required')
    return packet, plan, worker, load(worker['qualification'])


def coverage(plan, rows, *, root=None):
    by_id = {r['job_id']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) - {r['job_id'] for r in plan['jobs']}:
        raise ValueError('duplicate or undeclared matrix identity')
    result = []
    for record in plan['full_matrix']:
        row = by_id.get(record['job_id'], {'status': 'unrun', 'calls': 0})
        if row['status'] not in ('complete', 'failed', 'unrun'):
            raise ValueError('nonterminal matrix state')
        score = None
        if row['status'] == 'complete' and root is not None:
            reader = Path(root)/'jobs'/record['job_id']
            if record['system'] == 'cognee':
                reader /= 'reader'
            predictions = json.loads((reader/'predictions.json').read_bytes())
            if len(predictions) != 1 or type(predictions[0]['score']) not in (int, float) or predictions[0]['score'] not in (0, 1):
                raise ValueError('native binary QA coverage differs')
            score = predictions[0]['score']
        result.append(dict(record, status=row['status'], calls=row['calls'], qa_score=score))
    return result


def verify_report(packet, plan, worker, qualification, report, *, budget=None):
    from gluomem_bench.bench.lme_first import verify_journal
    debit = authority(plan['budget_authority'])
    prior, reserved = debit['retained_calls'], debit['retained_reserved_seconds']
    future = plan['future_budget_proposal']
    root = Path(packet['output_root'])
    if root.with_name(root.name+'.revoked.json').exists():
        raise ValueError('external revocation overrides publication')
    native_report = native.verify_root(worker, plan=plan, candidate_path=root/'native-status.json',
                                      budget=budget, qualification=qualification)
    coordination = verify_journal(root/'coordination', native.native_records(worker, plan, qualification))
    if (report.get('protocol') != PROTOCOL or report.get('native') != native_report
        or report.get('full_matrix') != coverage(plan, coordination['jobs'], root=root) or report.get('planned') != 2000
        or report.get('stage') != plan['stage'] or report.get('historical_quality_pooled') is not False
        or any(report.get(k) != plan[k] for k in ('declaration', 'methods', 'changed_condition', 'remediation'))
        or report.get('selected_quality_complete') != native_report['quality_complete']
        or report.get('quality_complete') != (plan['full_scope'] and native_report['quality_complete'])
        or report['accounting']['calls'] != prior + coordination['accounting']['calls']):
        raise ValueError('full study coverage/accounting/native verification differs')
    accounting = report['accounting']
    policy = dict(**future, max_workers=worker['max_workers'],
        native_limits=worker['native_limits'], memory_limit_mib=plan['native_constraints']['local_memory_mib'],
        max_in_flight=300, max_in_flight_tokens=39321600)
    if (coordination['policy'] != policy
        or accounting.get('prior_calls', 0) != prior or accounting.get('prior_elapsed_seconds', 0) != reserved
        or accounting.get('new_calls') != coordination['accounting']['calls']
        or accounting.get('elapsed_seconds') != reserved + accounting.get('new_elapsed_seconds', -1)
        or not coordination['accounting']['elapsed_seconds'] <= accounting['new_elapsed_seconds'] < future['max_seconds']
        or report['clock_origin'] != debit['clock_origin']
        or report['clock_deadline'] != debit['clock_origin'] + future['max_seconds']):
        raise ValueError('declared scheduler policy or once-only accounting/clock differs')
    for record, row in zip(plan['jobs'], coordination['jobs'], strict=True):
        if row['status'] == 'unrun':
            continue
        grouped = worker['native_mode'] == 'gluomem-shared-process' and record['system'] == 'gluomem'
        directory = 'gluomem-group' if grouped else record['job_id']
        closure = json.loads((root/'workers'/directory/'process-exit.json').read_bytes())
        if (type(closure.get('returncode')) is not int
            or closure.get('measurement_failed') is not False
            or not grouped and closure.get('drain_finished') is not True
            or any(type(closure.get(k)) is not int or closure[k] <= 0
                   for k in ('root_pid', 'root_created_100ns'))):
            raise ValueError('all attempted worker process closures required')
    return report


def root_failure_diagnostic(exc, *, phase='unknown', budget=None, packet=None):
    """Bounded causal receipt; never stringify failures or copy private payloads.

    Observations describe this collector, not an invented originating admission.
    Existing rack/encoder projections remain the classification authority.
    """
    from gluomem_bench.bench.embedding_runtime import safe_failure_diagnostic
    phases = {'setup', 'resource', 'input-cache', 'worker', 'run', 'verify', 'publish', 'unknown'}
    phase = phase if type(phase) is str and phase in phases else 'unknown'
    result = dict(protocol='root-causal-diagnostic@v4', phase=phase,
                  diagnostic=safe_failure_diagnostic(exc), guard_reason='unknown', sources=[],
                  observed_monotonic=None, observed_call_ordinal=None)
    # Only declared owned source bytes are eligible. Never open traceback paths.
    inventory = {}
    try:
        manifest_bound = packet is not None and 'source_manifest' in packet
        if manifest_bound:
            declared = load(packet['source_manifest'])
        else:
            declared = {f'gluomem_bench/bench/{name}.py': None for name in (
                'remediation_study', 'parallel_campaign', 'lme_first', 'lme_first_continue',
                'lme_first_execute', 'rack', 'rack_trace')}
        if type(declared) is dict:
            for rel, sha in declared.items():
                if manifest_bound and (type(sha) is not str or not re.fullmatch(r'[0-9a-f]{64}', sha)):
                    continue
                if (type(rel) is not str or '\\' in rel or Path(rel).is_absolute()
                        or '..' in Path(rel).parts or Path(rel).suffix != '.py'):
                    continue
                path = (SOURCE/rel).resolve()
                if path.is_relative_to(SOURCE):
                    inventory[str(path)] = (rel, sha)
    except BaseException:
        pass
    try:
        from gluomem_bench.bench.rack import failure_origin
        result['origin'] = failure_origin(exc)
    except BaseException:
        pass  # Diagnostics cannot supersede the actual BaseException.
    pending, seen = [exc], set()
    while pending and len(seen) < 12:
        item = pending.pop(0)
        if not isinstance(item, BaseException) or id(item) in seen:
            continue
        seen.add(id(item))
        fields = BaseException.__dict__['__dict__'].__get__(item)
        site = None
        trace = BaseException.__traceback__.__get__(item)
        for _ in range(256):
            if trace is None:
                break
            entry = inventory.get(trace.tb_frame.f_code.co_filename)
            if entry is not None:
                rel, expected = entry
                try:
                    sha = native.bound(SOURCE/rel)['sha256']
                    if expected is None or sha == expected:
                        site = dict(source=rel, sha256=sha, line=trace.tb_lineno,
                                    exception_ordinal=len(seen), is_innermost=trace.tb_next is None)
                except BaseException:
                    pass
            trace = trace.tb_next
        if site is not None:
            result['sources'].append(site)
        reason = fields.get('guard_reason')
        if type(reason) is str and reason in native.GUARD_REASONS:
            if result['guard_reason'] == 'unknown':
                result['guard_reason'] = reason
        pending.extend((BaseException.__cause__.__get__(item), BaseException.__context__.__get__(item)))
    if budget is not None:
        result['observed_monotonic'] = budget.clock()
        result['observed_call_ordinal'] = budget.calls
    return result


def revoke(packet, plan, budget, exc=None, *, phase='unknown'):
    root = Path(packet['output_root'])
    rows = []
    try:
        rows = json.loads((root/'coordination/progress.json').read_bytes())['jobs']
    except (OSError, ValueError, KeyError):
        pass
    try:
        matrix = coverage(plan, rows)
        unresolved = False
    except (ValueError, KeyError, TypeError):
        matrix, unresolved = coverage(plan, []), True
    failure = {'category': 'execution-or-publication-failed'}
    if exc is not None:
        try:
            failure['diagnostic'] = root_failure_diagnostic(exc, phase=phase, budget=budget, packet=packet)
        except BaseException:
            pass
    write_new(root.with_name(root.name+'.revoked.json'), dict(protocol=PROTOCOL, quality_complete=False,
        status='incomplete', accounting=budget.snapshot(), full_matrix=matrix, coordination_unresolved=unresolved,
        coverage_verified=False, failure=failure))


def publish(packet, plan, worker, qualification, native_report, budget):
    try:
        return _publish(packet, plan, worker, qualification, native_report, budget)
    except BaseException as exc:
        try:
            BaseException.__dict__['__dict__'].__get__(exc)['_study_failure_phase'] = 'publish'
            revoke(packet, plan, budget, exc, phase='publish')
        except BaseException:
            pass
        try:
            (Path(packet['output_root'])/'root-status.pending.json').unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _publish(packet, plan, worker, qualification, native_report, budget):
    root = Path(packet['output_root'])
    budget.check()
    write_new(root/'native-status.json', native_report)
    report = dict(protocol=PROTOCOL, stage=plan['stage'], planned=2000,
        declaration=plan['declaration'], methods=plan['methods'], changed_condition=plan['changed_condition'],
        remediation=plan['remediation'],
        quality_complete=plan['full_scope'] and native_report['quality_complete'],
        selected_quality_complete=native_report['quality_complete'], historical_quality_pooled=False,
        full_matrix=coverage(plan, native_report['jobs'], root=root), native=native_report,
        accounting=budget.snapshot(), clock_origin=budget.started, clock_deadline=budget.deadline)
    staged = root/'root-status.pending.json'
    write_new(staged, report)
    verify_report(packet, plan, worker, qualification, json.loads(staged.read_bytes()), budget=budget)
    material = staged.read_bytes()
    budget.check()
    os.replace(staged, root/'root-status.json')
    if (root/'root-status.json').read_bytes() != material:
        raise ValueError('publication readback differs')
    budget.check()
    receipt = dict(root_sha256=native.bound(root/'root-status.json')['sha256'],
                   clock_origin=budget.started, clock_deadline=budget.deadline, **budget.snapshot())
    write_new(root/'completion-receipt.json', receipt)
    if json.loads((root/'completion-receipt.json').read_bytes()) != receipt:
        raise ValueError('completion receipt readback differs')
    budget.check()
    return report


def execute(candidate_path, approval_path, *, budget):
    from contextlib import ExitStack
    from types import SimpleNamespace
    from urllib.parse import urlsplit

    from gluomem_bench.bench.lme_first import run_isolated_jobs
    from gluomem_bench.bench.lme_first_continue import NewAttemptBudget
    budget.check()
    approval = json.loads(Path(approval_path).read_bytes())
    packet, plan, worker, qualification = validate_candidate(candidate_path, budget=budget, approval=approval)
    root = Path(packet['output_root'])
    run_failure = None
    try:
        phase = 'resource'
        native.validate_resource_contract(worker, qualification, local_memory_mib=plan['native_constraints']['local_memory_mib'])
        native.verify_qualification_resources(qualification, dict.fromkeys(SYSTEMS[:3], 4096))
        phase = 'setup'
        from gluomem_bench.bench.upstream_scoring import load_longmemeval_prompt
        load_longmemeval_prompt(Path(plan['judge_source']))
        budget.check()
        root.mkdir(parents=True, exist_ok=False)
        write_new(root/'candidate.json', json.loads(Path(candidate_path).read_bytes()))
        write_new(root/'execution-approval.json', approval)
        phase = 'input-cache'
        inputs = native.build_input_cache(root, plan, budget)
        phase = 'resource'
        records = native.native_records(worker, plan, qualification)
        bindings = {r['job_id']: r for r in records}
        memory = native.MemoryAdmission(host_reserve_mib=8192, commit_limits=worker['history_commit_limits_mib'])
        for system in worker['native_limits']:
            measured = qualification['memory_samples'][system if system != 'no_memory' else 'gluomem']
            memory.observe(system, **{k: measured[k] for k in ('rss_mib', 'private_mib')})
        shared = worker['native_mode'] == 'gluomem-shared-process' and 'gluomem' in plan['selected_systems']
        identity = 'owned-gluomem-process-group'
        phase = 'worker'
        with native.ContextBroker() as broker, ExitStack() as lifecycle:
            group = None
            if shared:
                proof = load(worker['shared_gluomem_qualification'])
                memory.commit_limits['gluomem-group'] = 4096
                memory.observe('gluomem-group', **proof['memory_samples'])
                if not memory.acquire(identity, 'gluomem-group'):
                    raise ValueError('shared group resource admission refused')
                endpoints = [broker.address]
                for url in (plan['profile']['base_url'], plan['embedding_url']):
                    part = urlsplit(url)
                    endpoints.append((part.hostname, part.port or (443 if part.scheme == 'https' else 80)))
                group = lifecycle.enter_context(native.SharedGluomemWorker(workdir=root/'workers/gluomem-group',
                    commit_limit_mib=4096, max_workers=worker['native_limits']['gluomem'], host_reserve_mib=8192,
                    network_endpoints=endpoints, memory_failure=memory.latch_pressure,
                    sample_memory=lambda r: memory.update(identity, rss_mib=r['rss_mib'], private_mib=r['private_mib'])))
            def admit(record):
                if group is not None and record['system'] == 'gluomem':
                    return not group.failed and group.process.poll() is None and memory.admit_existing(identity)
                return memory.acquire(record['job_id'], record['system'])
            def run(job, context):
                record = bindings[job.job_id]
                try:
                    request = dict(candidate=packet['worker'], input=inputs[record['sample_id']], system=record['system'])
                    if group is not None and record['system'] == 'gluomem':
                        group.run(request, broker=broker, context=context, workdir=root/'workers'/job.job_id)
                    else:
                        native.launch_worker(dict(request, commit_limit_mib=4096), broker=broker, context=context,
                            workdir=root/'workers'/job.job_id, host_reserve_mib=8192, memory_failure=memory.latch_pressure,
                            sample_memory=lambda r: memory.update(job.job_id, rss_mib=r['rss_mib'], private_mib=r['private_mib']))
                    adapter, sample = native.restore_input(inputs[record['sample_id']], plan)
                    native_job = SimpleNamespace(**dict(record, output=Path(record['output'])))
                    cfg, profile = native.effective_config(plan)
                    return native.verify_native_job(job=native_job, context=context, cfg=cfg, adapter=adapter,
                        sample=sample, profile=profile, judge_source=Path(plan['judge_source']))
                finally:
                    memory.release(job.job_id)
            view = NewAttemptBudget(budget)
            phase = 'run'
            try:
                coordination = run_isolated_jobs(records, out=root/'coordination', run_job=run,
                    max_calls=view.max_calls, max_seconds=view.max_seconds, budget=view, max_workers=worker['max_workers'],
                    native_limits=worker['native_limits'], memory_limit_mib=plan['native_constraints']['local_memory_mib'],
                    max_in_flight_tokens=39321600, admit=admit, release_admission=lambda r: memory.release(r['job_id']))
            except BaseException as exc:
                run_failure = exc
                raise
            finally:
                phase = 'worker'  # Includes actual group/broker context teardown.
        phase = 'resource'
        memory.release(identity)
        if memory.pressure_latched or memory.snapshot()['resident_jobs']:
            raise ValueError('resource closure unavailable')
        phase = 'verify'
        report = native._root_report(worker, plan, coordination)
        report['memory_admission'] = memory.snapshot()
        if shared:
            evidence = {k: native.bound(root/'workers/gluomem-group'/f) for k, f in
                (('samples', 'resource-samples.jsonl'), ('outcome', 'outcome.json'), ('exit', 'process-exit.json'))}
            samples = [json.loads(line) for line in Path(evidence['samples']['path']).read_bytes().splitlines()]
            report['shared_gluomem_resources'] = dict(commit_limit_mib=4096, evidence=evidence,
                memory_samples={k: max(r[k] for r in samples) for k in ('rss_mib', 'private_mib')})
        validate_candidate(candidate_path, budget=budget, approval=approval, post_exit=True)
        phase = 'publish'
        return publish(packet, plan, worker, qualification, report, budget)
    except BaseException as exc:
        if exc is run_failure:
            phase = 'run'
        try:
            BaseException.__dict__['__dict__'].__get__(exc)['_study_failure_phase'] = phase
            revoke(packet, plan, budget, exc, phase=phase)
        except BaseException:
            pass
        raise


def verify(candidate_path, out, *, budget):
    packet, plan, worker, qualification = validate_candidate(candidate_path, budget=budget, post_exit=True)
    root = Path(packet['output_root'])
    approval = json.loads((root/'execution-approval.json').read_bytes())
    validate_candidate(candidate_path, budget=budget, approval=approval, post_exit=True)
    if json.loads((root/'candidate.json').read_bytes()) != json.loads(Path(candidate_path).read_bytes()):
        raise ValueError('executed candidate differs')
    report = verify_report(packet, plan, worker, qualification, json.loads((root/'root-status.json').read_bytes()))
    closure = json.loads((root/'completion-receipt.json').read_bytes())
    if (closure['root_sha256'] != native.bound(root/'root-status.json')['sha256']
        or closure['calls'] != report['accounting']['calls']
        or closure['clock_origin'] != report['clock_origin'] or closure['clock_deadline'] != report['clock_deadline']
        or not report['accounting']['elapsed_seconds'] <= closure['elapsed_seconds'] < budget.max_seconds):
        raise ValueError('same-clock final publication receipt differs')
    out = Path(out).resolve()
    if out.is_relative_to(root) or out.is_relative_to(Path(candidate_path).resolve().parent) or out.is_relative_to(SOURCE):
        raise ValueError('verification output must be outside sealed evidence/source')
    budget.check()
    return write_new(out, dict(protocol=PROTOCOL, verified=True, quality_complete=report['quality_complete'],
                               candidate=native.bound(candidate_path), root=native.bound(root/'root-status.json')))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('bootstrap', 'budget', 'prepare', 'freeze', 'execute', 'verify'):
        sub = commands.add_parser(command)
        names = {'bootstrap': ('operator', 'operator-sha256', 'out'), 'budget': (),
                 'prepare': ('spec', 'spec-sha256', 'out'),
                 'freeze': ('plan', 'plan-sha256', 'template', 'template-sha256', 'output-root', 'out'),
                 'execute': ('candidate', 'approval'), 'verify': ('candidate', 'out')}[command]
        for name in names + (() if command == 'bootstrap' else ('authority', 'authority-sha256')):
            sub.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    budget = None
    phase = 'setup'
    try:
        if args.command == 'bootstrap':
            print(json.dumps(bootstrap({'path': args.operator, 'sha256': args.operator_sha256}, args.out), sort_keys=True))
            return 0
        budget = campaign_budget({'path': args.authority, 'sha256': args.authority_sha256})
        if args.command == 'budget':
            print(json.dumps(dict(authority=budget.authority, clock_origin=budget.started,
                clock_deadline=budget.deadline, remaining_seconds=budget.remaining(), **budget.snapshot()), sort_keys=True))
            return 0
        if args.command == 'prepare':
            result = prepare({'path': args.spec, 'sha256': args.spec_sha256}, args.out, budget=budget)
        elif args.command == 'freeze':
            result = freeze({'path': args.plan, 'sha256': args.plan_sha256},
                            {'path': args.template, 'sha256': args.template_sha256}, args.output_root, args.out, budget=budget)
        elif args.command == 'execute':
            report = execute(args.candidate, args.approval, budget=budget)
            result = dict(protocol=PROTOCOL, quality_complete=report['quality_complete'],
                          selected_quality_complete=report['selected_quality_complete'])
        else:
            phase = 'verify'
            result = verify(args.candidate, args.out, budget=budget)
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as exc:
        category = {ValueError: 'contract', OSError: 'io', FileNotFoundError: 'missing-file',
                    KeyError: 'missing-field'}.get(type(exc), 'unknown')
        failure = dict(protocol=PROTOCOL, status='refused-or-incomplete', category=category)
        try:
            phase = BaseException.__dict__['__dict__'].__get__(exc).get('_study_failure_phase', phase)
            failure['diagnostic'] = root_failure_diagnostic(exc, phase=phase, budget=budget)
        except BaseException:
            pass
        print(json.dumps(failure), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
