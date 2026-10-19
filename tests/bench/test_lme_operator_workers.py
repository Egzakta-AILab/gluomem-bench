"""Operator-visible worker ceilings must reach canonical preparation and freeze."""
import copy

import pytest

from gluomem_bench.bench import lme_first as plans
from gluomem_bench.bench import lme_first_execute as execution
from tests.bench.test_lme_first import source_spec


def make_plan(tmp_path, **kwargs):
    source, digest = source_spec(tmp_path)
    return plans.build_plan(
        source_spec=source, source_spec_sha256=digest,
        future_max_calls=32768000, future_max_seconds=172800,
        local_memory_mib=12288, **kwargs,
    )


def test_canonical_shared_300_worker_proposal_counts_group_memory_once(tmp_path):
    plan = make_plan(tmp_path, gluomem_workers=300, gluomem_group_memory_mib=4096)
    assert plan['native_constraints']['workers']['gluomem']['workers'] == 300
    assert plan['native_constraints']['reserved_memory_mib'] == 10240
    assert plan['operating_point']['max_in_flight'] == 300
    assert plans.validate_plan(plan) == 2000
    forged = copy.deepcopy(plan)
    forged['native_constraints']['workers']['gluomem']['workers'] = 299
    with pytest.raises(ValueError):
        plans.validate_plan(forged)


@pytest.mark.parametrize('workers', [0, 301, True, '300'])
def test_invalid_operator_worker_ceiling_refused(tmp_path, workers):
    with pytest.raises(ValueError):
        make_plan(tmp_path, gluomem_workers=workers, gluomem_group_memory_mib=4096)


def test_legacy_plan_still_reconstructs_without_new_worker_proposal(tmp_path):
    plan = make_plan(tmp_path)
    assert 'worker_proposal' not in plan
    assert plan['native_constraints']['workers']['gluomem']['workers'] == 24
    assert plans.validate_plan(plan) == 2000


def test_freeze_cli_forwards_explicit_native_limits(tmp_path, monkeypatch):
    captured = {}
    def freeze(**kwargs):
        captured.update(kwargs)
        return {'path': str(tmp_path / 'candidate'), 'sha256': 'a' * 64}
    monkeypatch.setattr(execution, 'freeze_candidate', freeze)
    args = ['freeze']
    for flag in ['plan', 'budget-decision', 'qualification']:
        args += ['--' + flag, str(tmp_path / flag), '--' + flag + '-sha256', 'a' * 64]
    args += ['--output-root', str(tmp_path / 'output'), '--out', str(tmp_path / 'candidate'),
             '--cognee-python', 'python.exe', '--host-reserve-mib', '8192',
             '--history-commit-limit-mib', '4096', '--max-workers', '300',
             '--native-limits', 'gluomem=300,mem0=4,cognee=2,no_memory=20']
    assert execution._main(args) == 0
    assert captured['native_limits'] == {'gluomem': 300, 'mem0': 4, 'cognee': 2, 'no_memory': 20}


@pytest.mark.parametrize('value', [
    'gluomem=300,gluomem=24,mem0=4,cognee=2,no_memory=20',
    'gluomem=300,mem0=4,cognee=2',
    'gluomem=300,mem0=4,cognee=3,no_memory=20',
    'gluomem=0,mem0=4,cognee=2,no_memory=20',
])
def test_native_limit_argument_rejects_duplicates_missing_and_unqualified_caps(value):
    assert callable(getattr(execution, 'parse_native_limits', None))
    with pytest.raises(Exception):
        execution.parse_native_limits(value)
