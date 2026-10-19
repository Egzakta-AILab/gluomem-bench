"""Fresh proposed API fixtures; never execute retained sampler AST."""
import pytest
from gluomem_bench.bench import lme_resource_lifecycle as r


def test_parent_gate_prunes_old_branch_before_querying_grandchild():
    calls = []
    def lookup(pid):
        calls.append(pid)
        if pid == 3:
            raise PermissionError('inaccessible excluded grandchild')
        return {1: 100, 2: 50, 4: 110, 5: 120}[pid]
    rows = r.discover_members({1: 0, 2: 1, 3: 2, 4: 1, 5: 4}, 1, 100, lookup)
    assert [v['pid'] for v in rows] == [1, 4, 5]
    assert 2 in calls and 3 not in calls


@pytest.mark.parametrize('parents, values, member', [
    ({1: 0, 2: 1}, {1: 100}, 2),
    ({1: 0, 2: 1, 3: 2}, {1: 100, 2: 110}, 3),
    ({1: 0}, {1: 99}, 1),
])
def test_unknown_genuine_member_or_recycled_root_refuses(parents, values, member):
    def lookup(pid):
        if pid not in values:
            raise PermissionError('not readable')
        return values[pid]
    with pytest.raises(r.ResourceFailure) as failure:
        r.discover_members(parents, 1, 100, lookup)
    assert failure.value.stage == 'identity'
    assert failure.value.member['pid'] == member


def test_verified_equal_time_children_and_cycles_are_bounded():
    calls = []
    def lookup(pid):
        calls.append(pid)
        return 100
    rows = r.discover_members({1: 3, 2: 1, 3: 2}, 1, 100, lookup)
    assert {v['pid'] for v in rows} == {1, 2, 3}
    assert calls == [1, 2, 3]
