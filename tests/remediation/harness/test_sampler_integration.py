"""Current private sampler exercised through authored fake Win32 calls only."""
import ctypes
import pytest
from gluomem_bench.bench import lme_first_execute as x


class Function:
    def __init__(self, implementation): self.implementation = implementation
    def __call__(self, *args): return self.implementation(*args)


@pytest.mark.parametrize('unknown_direct_child', [False, True])
def test_private_sampler_uses_parent_gated_discovery_before_memory(monkeypatch, unknown_direct_child):
    parents = [(1, 0), (2, 1), (3, 2), (4, 1)]
    queries, memory_queries, closed = [], [], []
    cursor = iter(parents)
    def entry(snapshot, pointer):
        try: pid, parent = next(cursor)
        except StopIteration:
            ctypes.set_last_error(18)
            return 0
        pointer._obj.pid, pointer._obj.parent = pid, parent
        return 1
    def memory(handle, pointer, size):
        memory_queries.append(handle)
        pointer._obj.rss, pointer._obj.private, pointer._obj.peak_rss = 1048576, 2097152, 1048576
        return 1
    class Kernel:
        CreateToolhelp32Snapshot = Function(lambda flags, pid: 900)
        OpenProcess = Function(lambda rights, inherited, pid: pid)
        CloseHandle = Function(lambda handle: closed.append(handle))
        WaitForSingleObject = Function(lambda handle, timeout: 258)
        Process32FirstW = Function(entry)
        Process32NextW = Function(entry)
    class Psapi:
        GetProcessMemoryInfo = Function(memory)
    monkeypatch.setattr(ctypes, 'WinDLL', lambda name, **kwargs: Kernel if name == 'kernel32' else Psapi)
    def identity(pid):
        queries.append(pid)
        if pid == 3 or (unknown_direct_child and pid == 4):
            raise PermissionError('PRIVATE unavailable process')
        return {1: 100, 2: 50, 4: 110}[pid]
    monkeypatch.setattr(x, 'process_creation_time', identity)
    if unknown_direct_child:
        with pytest.raises(x.ResourceFailure) as failed:
            x.process_tree_memory(1, created_100ns=100)
        assert failed.value.stage == 'identity'
        assert failed.value.member == {'pid': 4, 'created_100ns': 0}
        assert not memory_queries
    else:
        sample = x.process_tree_memory(1, created_100ns=100)
        assert [row['pid'] for row in sample['members']] == [1, 4]
        assert sample['rss_mib'] == 2 and sample['private_mib'] == 4
        assert sample['processes'] == 2
        assert memory_queries == [1, 4]
    assert 2 in queries and 3 not in queries
    assert 900 in closed
