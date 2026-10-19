"""Authored offline budget authority; no model/encoder/native calls."""
import copy
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from gluomem_bench.bench import remediation_study as s

OLD_TESTS = Path(__file__).resolve().parents[1]/'evaluation/study_fixtures.py'
spec = importlib.util.spec_from_file_location('driver_fixtures', OLD_TESTS)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
put = fixtures.put


def operator(tmp_path, calls=141363):
    return {'protocol': 'lme-study-budget-authority@v1', 'max_calls': 32749905,
            'max_seconds': 170578, 'retained_calls': calls, 'retained_reserved_seconds': 32170,
            'historical_unknown': 'prior campaign outside this phase unknown; not reset',
            'debit_evidence': put(tmp_path/'ledger.json', {'calls': calls, 'scope': 'authored only'})}


def authority(tmp_path, calls=141363, clock=time.monotonic):
    return s.bootstrap(put(tmp_path/'operator.json', operator(tmp_path, calls)),
                       tmp_path/'authority.json', clock=clock)


@pytest.mark.parametrize('origin', [float('nan'), float('inf'), -1, 21, True])
def test_constructor_rejects_invalid_retained_origins(origin):
    from gluomem_bench.bench.parallel_campaign import CampaignBudget
    with pytest.raises(ValueError):
        CampaignBudget(max_calls=10, max_seconds=100, clock=lambda: 20, started_monotonic=origin)


def test_constructor_retains_origin_and_default_unchanged():
    from gluomem_bench.bench.parallel_campaign import CampaignBudget
    b = CampaignBudget(max_calls=10, max_seconds=100, prior_elapsed_seconds=5,
                       clock=lambda: 20, started_monotonic=10)
    assert (b.started, b.deadline, b.remaining()) == (10, 105, 85)
    assert b.snapshot()['elapsed_seconds'] == 15
    assert CampaignBudget(max_calls=10, max_seconds=100, clock=lambda: 20).started == 20


def test_extra_qualification_debits_once_and_cannot_refund(tmp_path):
    binding = authority(tmp_path, clock=lambda: 10)
    first = s.campaign_budget(binding, clock=lambda: 20)
    assert first.calls == 141363
    assert first.remaining() == 138398
    newer = s.load(binding)
    newer['previous_authority'] = binding
    newer['retained_calls'] += 3
    newer['debit_evidence'] = put(tmp_path/'qualification-ledger.json', {'calls': 141366})
    updated = put(tmp_path/'matrix-authority.json', newer)
    second = s.campaign_budget(updated, clock=lambda: 30)
    assert second.calls == 141366
    assert second.max_calls - second.calls == 32608539
    assert second.started == first.started == 10
    assert second.deadline == first.deadline == 138418
    assert second.snapshot()['elapsed_seconds'] == 32190
    for field, value in [('retained_calls', 141362), ('max_calls', 32749906),
                         ('retained_reserved_seconds', 32169), ('clock_origin', 11)]:
        forged = dict(newer, **{field: value})
        with pytest.raises(ValueError):
            s.campaign_budget(put(tmp_path/f'forged-{field}.json', forged), clock=lambda: 30)
    with pytest.raises(ValueError):
        s.campaign_budget(dict(binding, sha256='0'*64), clock=lambda: 30)
    exhausted = dict(newer, retained_calls=newer['max_calls'])
    with pytest.raises(ValueError):
        s.campaign_budget(put(tmp_path/'exhausted.json', exhausted), clock=lambda: 30)


def test_real_cross_process_cli_keeps_origin_before_prepare_setup(tmp_path):
    binding = authority(tmp_path)
    source = str(s.SOURCE)
    code = ('import sys;sys.path.insert(0,sys.argv[1]);'
            'from gluomem_bench.bench.remediation_study import campaign_budget;'
            'import json;b=campaign_budget(json.loads(sys.argv[2]));'
            'print(json.dumps(dict(started=b.started,deadline=b.deadline,**b.snapshot())))')
    def read():
        return json.loads(subprocess.check_output([sys.executable, '-B', '-I', '-X', 'utf8',
                          '-c', code, source, json.dumps(binding)], cwd=source))
    before = read()
    time.sleep(0.05)
    after = read()
    assert before['started'] == after['started'] == s.load(binding)['clock_origin']
    assert before['deadline'] == after['deadline']
    assert after['new_elapsed_seconds'] > before['new_elapsed_seconds']


def test_actual_cli_bootstrap_prepare_budget_share_absolute_deadline(tmp_path):
    op = put(tmp_path/'operator.json', operator(tmp_path))
    driver = Path(s.__file__)
    command = [sys.executable, '-B', '-I', '-X', 'utf8', str(driver)]
    def cli(*args):
        return json.loads(subprocess.check_output([*command, *args], cwd=s.SOURCE))
    before = time.monotonic()
    bound = cli('bootstrap', '--operator', op['path'], '--operator-sha256', op['sha256'],
                '--out', str(tmp_path/'authority.json'))
    common = ['--authority', bound['path'], '--authority-sha256', bound['sha256']]
    start = cli('budget', *common)
    declaration = fixtures.specification.__wrapped__(tmp_path)
    declaration['budget_authority'] = bound
    declaration['remediation'] = {'cognee_clocks': {'request_seconds': 600, 'cleanup_reserve_seconds': 12}}
    spec_binding = put(tmp_path/'spec.json', declaration)
    plan_binding = cli('prepare', '--spec', spec_binding['path'], '--spec-sha256', spec_binding['sha256'],
                       '--out', str(tmp_path/'plan.json'), *common)
    later = cli('budget', *common)
    plan = s.load(plan_binding)
    assert before <= start['clock_origin'] <= later['clock_origin']
    assert start['clock_origin'] == later['clock_origin']
    assert start['clock_deadline'] == later['clock_deadline']
    assert later['remaining_seconds'] < start['remaining_seconds']
    assert later['new_elapsed_seconds'] > start['new_elapsed_seconds']
    assert plan['budget_authority'] == bound
    assert plan['remediation'] == declaration['remediation']
    forged = s.load(bound)
    forged['clock_host']['boot_id'] = 'not-this-boot'
    with pytest.raises(ValueError):
        s.campaign_budget(put(tmp_path/'wrong-host.json', forged))


def test_prepare_binds_authority_and_refuses_different_budget(tmp_path):
    declaration = fixtures.specification.__wrapped__(tmp_path)
    binding = authority(tmp_path)
    declaration['budget_authority'] = binding
    budget = s.campaign_budget(binding)
    plan = s.load(s.prepare(put(tmp_path/'spec.json', declaration), tmp_path/'plan.json', budget=budget))
    assert plan['budget_authority'] == binding
    assert plan['future_budget_proposal'] == {'max_calls': 32608542, 'max_seconds': 138408}
    assert plan['historical_accounting']['retained_calls'] == 141363
    assert plan['remediation'] == {}
    other = copy.deepcopy(s.load(binding))
    other['retained_calls'] += 1
    other['previous_authority'] = binding
    other_binding = put(tmp_path/'other.json', other)
    with pytest.raises(ValueError):
        s.prepare(put(tmp_path/'spec2.json', declaration), tmp_path/'never.json',
                  budget=s.campaign_budget(other_binding))
    assert not (tmp_path/'never.json').exists()
