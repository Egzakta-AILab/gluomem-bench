"""Adapt unchanged authored driver tests to mandatory budget authority API."""
import importlib.util
from pathlib import Path

import pytest
from gluomem_bench.bench import remediation_study as s

HERE = Path(__file__).resolve().parent
legacy = HERE.parent/'evaluation/study_fixtures.py'
spec = importlib.util.spec_from_file_location('legacy_study_tests', legacy)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
for name in dir(old):
    if name.startswith('test_'):
        globals()[name] = getattr(old, name)


@pytest.fixture
def specification(tmp_path):
    from test_authority import authority
    value = old.specification.__wrapped__(tmp_path)
    value['budget_authority'] = authority(tmp_path)
    return value


@pytest.fixture(autouse=True)
def mandatory_authority(tmp_path, specification, monkeypatch, request):
    from test_authority import operator
    real_budget, real_main = s.campaign_budget, s.main
    active = [specification['budget_authority']]
    def budget(**kwargs):
        if 'clock' in kwargs:
            op = old.put(tmp_path/(request.node.name[:40] + '-clock-operator.json'), operator(tmp_path))
            active[0] = s.bootstrap(op, tmp_path/'clock-authority.json', clock=kwargs['clock'])
            specification['budget_authority'] = active[0]
        return real_budget(active[0], **kwargs)
    def main(argv):
        return real_main([*argv, '--authority', active[0]['path'], '--authority-sha256', active[0]['sha256']])
    # Shared-exit test constructs its plan before asking for a constant clock;
    # establish that authored origin before plan assembly, preserving its semantics.
    if request.node.name == 'test_shared_exit_receipt_uses_actual_group_contract':
        budget(clock=lambda: 0)
        def budget(**kwargs):
            return real_budget(active[0], **kwargs)
    monkeypatch.setattr(s, 'campaign_budget', lambda binding=None, **kw: real_budget(binding, **kw) if binding is not None else budget(**kw))
    monkeypatch.setattr(s, 'main', main)
