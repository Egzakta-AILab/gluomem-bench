"""Pure authored terminal-proof fixtures, no native identity query or child."""
import json
from types import SimpleNamespace
import pytest
from gluomem_bench.bench import lme_resource_lifecycle as r
from gluomem_bench.bench.lme_first import DispatchStopped


@pytest.mark.parametrize('omitted', ['live', 'unknown', 'exited'])
@pytest.mark.parametrize('stage', ['memory-query', 'member-open', 'identity'])
def test_only_exact_launcher_excluded_all_retained_identities_proven(omitted, stage):
    launcher = dict(pid=10, created_100ns=100)
    native = dict(pid=11, created_100ns=110)
    prior = dict(root_pid=10, root_created_100ns=100, private_mib=100,
        members=[launcher, native], host_status=dict(available_physical_mib=9000, commit_headroom_mib=9000))
    queried = []
    class Identities:
        handles = {(10,100): 'launcher', (11,110): 'native', (12,120): omitted,
                   (10,90): 'same-pid-not-launcher'}
        def exited(self, member):
            key = (member['pid'], member['created_100ns'])
            queried.append(key)
            return key in ((11,110), (10,90)) or (key == (12,120) and omitted == 'exited')
    marker = json.dumps(dict(status='native-closed-verified', calls=3,
        resource=dict(native, commit_limit_mib=4096,
                      enforcement='Windows inherited job commit ceiling'))).encode()
    proof = r.terminal_proof(r.ResourceFailure(stage, native), marker, prior, Identities(),
        pid=10, created=100, cap=4096, calls=3, host=prior['host_status'], reserve=8192)
    assert (proof is not None) == (stage != 'identity' and omitted == 'exited')
    if proof:
        assert (10,100) not in queried and (10,90) in queried
        assert (12,120) in queried


@pytest.mark.parametrize('stage', ['process-snapshot', 'identity', 'host-status', 'pressure'])
def test_identity_or_host_loss_cannot_use_native_close_marker(tmp_path, monkeypatch, stage):
    member = dict(pid=11, created_100ns=110)
    host = dict(available_physical_mib=9000, commit_headroom_mib=9000)
    row = dict(root_pid=10, root_created_100ns=100, private_mib=100,
        members=[dict(pid=10, created_100ns=100), member], host_status=host)
    class Process:
        pid, returncode, polls = 10, 0, 0
        def poll(self):
            self.polls += 1
            return None if self.polls <= 2 else 0
    class Identities:
        handles = {}
        def remember(self, row): pass
        def close(self): pass
        def exited(self, member): return True
    monkeypatch.setattr(r, 'Identities', Identities)
    monkeypatch.setattr(r.time, 'sleep', lambda seconds: None)
    def sample(*args, **kwargs):
        if process.polls == 2: raise r.ResourceFailure(stage, member)
        return row
    (tmp_path/'outcome.json').write_text(json.dumps(dict(status='native-closed-verified', calls=1,
        resource=dict(member, commit_limit_mib=4096, enforcement='Windows inherited job commit ceiling'))))
    process = Process()
    ctx = SimpleNamespace(deadline=100, budget=SimpleNamespace(clock=lambda: 10), calls=1, failed=False)
    native = SimpleNamespace(process_tree_memory=sample, host_memory_status=lambda: host,
        json_bytes=lambda value: (json.dumps(value)+'\n').encode(), DispatchStopped=DispatchStopped)
    latched = []
    with pytest.raises(DispatchStopped):
        r.monitor(process, native=native, context=ctx, workdir=tmp_path, created=100, cap=4096,
                  sample_memory=None, memory_failure=lambda: latched.append(True), reserve=8192)
    receipt = json.loads((tmp_path/'process-exit.json').read_bytes())
    assert receipt['measurement_failed'] and ctx.failed and latched
    assert receipt['resource_failure']['phase'] == stage
    assert receipt['verified_terminal_teardown'] is None
