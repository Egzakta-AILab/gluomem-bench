"""Categorical fail-closed resource lifecycle; terminal exemption is proof-bound."""
import ctypes as c
from ctypes import wintypes as w
import hashlib,json,math,time

STAGES=frozenset(('process-snapshot','member-open','memory-query','identity','host-status','sample-write','admission-update','pressure','terminal-proof','drain-deadline'))
class ResourceFailure(OSError):
    def __init__(self,stage,member=None):
        if stage not in STAGES:raise ValueError('unknown resource stage')
        self.stage=stage
        self.member=None if member is None else {k:member[k] for k in ('pid','created_100ns')}
        super().__init__('resource-'+stage)

def discover_members(parents, pid, created, lookup):
    """Expand only verified chronological parents; unknown genuine members refuse."""
    children = {}
    for child, parent in parents.items():
        children.setdefault(parent, []).append(child)
    pending = [(pid, created)]
    seen = set()
    members = []
    for candidate, parent_created in pending:
        if candidate in seen:
            continue
        seen.add(candidate)
        member = {'pid': candidate, 'created_100ns': created if candidate == pid else 0}
        try:
            actual = lookup(candidate)
        except OSError:
            raise ResourceFailure('identity', member) from None
        if type(actual) is not int or actual <= 0 or (candidate == pid and actual != created):
            raise ResourceFailure('identity', member)
        if actual < parent_created:
            continue
        members.append(dict(member, created_100ns=actual, parent_pid=parents.get(candidate, 0)))
        pending.extend((child, actual) for child in children.get(candidate, ()))
    return sorted(members, key=lambda row: (row['created_100ns'], row['pid']))


class Identities:
    """Query/synchronize handles bind exit proof to creation identity, not PID."""
    def __init__(self):
        self.kernel=c.WinDLL('kernel32',use_last_error=True)
        self.kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];self.kernel.OpenProcess.restype=w.HANDLE
        self.kernel.GetProcessTimes.argtypes=[w.HANDLE]+[c.POINTER(w.FILETIME)]*4
        self.kernel.WaitForSingleObject.argtypes=[w.HANDLE,w.DWORD]
        self.kernel.CloseHandle.argtypes=[w.HANDLE]
        self.handles={}
    def remember(self,row):
        for m in row['members']:
            key=(m['pid'],m['created_100ns'])
            if key in self.handles:continue
            handle=self.kernel.OpenProcess(0x100000|0x1000,False,m['pid'])
            if not handle:raise ResourceFailure('identity',m)
            times=[w.FILETIME() for _ in range(4)]
            if not self.kernel.GetProcessTimes(handle,*(c.byref(t) for t in times)) or (times[0].dwHighDateTime<<32)|times[0].dwLowDateTime!=m['created_100ns']:
                self.kernel.CloseHandle(handle);raise ResourceFailure('identity',m)
            self.handles[key]=handle
    def exited(self,member):
        handle=self.handles.get((member['pid'],member['created_100ns']))
        return handle is not None and self.kernel.WaitForSingleObject(handle,0)==0
    def close(self):
        for handle in self.handles.values():self.kernel.CloseHandle(handle)
        self.handles.clear()

def healthy_host(host,reserve):
    if reserve is None:return False
    vals=[host[k] for k in ('available_physical_mib','commit_headroom_mib')]
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in vals):raise ResourceFailure('host-status')
    if min(vals)<reserve:raise ResourceFailure('pressure')
    return True

def terminal_proof(error,marker,prior,identities,*,pid,created,cap,calls,host,reserve):
    if type(error) is not ResourceFailure or error.stage not in ('member-open','memory-query') or not marker or prior is None:return None
    try:
        outcome=json.loads(marker)
        resource=outcome['resource']
        native={'pid':resource['pid'],'created_100ns':resource['created_100ns']}
        if (outcome['status']!='native-closed-verified' or outcome['calls']!=calls or resource['commit_limit_mib']!=cap
            or resource['enforcement']!='Windows inherited job commit ceiling' or native['pid']==pid
            or prior['root_pid']!=pid or prior['root_created_100ns']!=created
            or not 0<prior['private_mib']<=cap or not healthy_host(prior['host_status'],reserve)
            or not healthy_host(host,reserve) or error.member not in [{k:m[k] for k in ('pid','created_100ns')} for m in prior['members']]
            or native not in [{k:m[k] for k in ('pid','created_100ns')} for m in prior['members']]):return None
        # No surviving nonlauncher work and the exact failed member is terminal.
        if not identities.exited(error.member):return None
        if not all(identities.exited({'pid':known_pid,'created_100ns':known_created})
                   for known_pid,known_created in identities.handles
                   if (known_pid,known_created)!=(pid,created)):return None
        return {'protocol':'verified-terminal-native-close@v1','marker_sha256':hashlib.sha256(marker).hexdigest(),
                'native_identity':native,'origin_phase':error.stage,'failed_member':error.member,
                'identity_exit_proof':'retained-query-synchronize-handles-signaled','unknown_sample_recorded':False}
    except (KeyError,TypeError,ValueError,ResourceFailure):return None

def monitor(process,*,native,context,workdir,created,cap,sample_memory,memory_failure,reserve):
    prior=None;first_failure=None;terminal=None;marker=None
    identities=Identities()
    def latch(stage,member=None):
        nonlocal first_failure
        if first_failure is None:first_failure={'protocol':'resource-failure-origin@v1','phase':stage,'member':member,'native_work_state':'terminal-proof' if terminal else 'unverified-or-active'}
        context.failed=True
        if memory_failure is not None:memory_failure()
    # Both deadlines use the SAME original budget clock, never a new work clock.
    deadline=context.deadline
    drain_deadline=deadline
    try:
        with (workdir/'resource-samples.jsonl').open('xb') as samples:
            while process.poll() is None:
                stage='process-snapshot'
                try:
                    if context.budget.clock()>=drain_deadline:raise ResourceFailure('drain-deadline')
                    if terminal is not None or first_failure is not None:
                        stage='host-status';host=native.host_memory_status() if reserve is not None else None
                        if host is not None:healthy_host(host,reserve)
                        time.sleep(.01);continue
                    # Capture BEFORE sampling. A later-created marker is not at-failure authority.
                    stage='terminal-proof'
                    try:marker=(workdir/'outcome.json').read_bytes()
                    except FileNotFoundError:marker=None
                    stage='process-snapshot'
                    row=native.process_tree_memory(process.pid,created_100ns=created)
                    stage='host-status';host=native.host_memory_status() if reserve is not None else None
                    if host is not None:
                        row['host_status']=host;healthy_host(host,reserve)
                    stage='identity';identities.remember(row)
                    stage='sample-write';samples.write(native.json_bytes(row));samples.flush()
                    stage='admission-update'
                    if sample_memory is not None:sample_memory(row)
                    prior=row
                except Exception as error:
                    phase=error.stage if type(error) is ResourceFailure else stage
                    # Never discard a host/callback/write failure just because poll changed.
                    recovered=False
                    if first_failure is None and terminal is None and phase in ('member-open','memory-query'):
                        try:
                            host=native.host_memory_status()
                            terminal=terminal_proof(error,marker,prior,identities,pid=process.pid,created=created,
                                cap=cap,calls=context.calls,host=host,reserve=reserve)
                            recovered=terminal is not None
                        except Exception:terminal=None
                    if recovered:
                        drain_deadline=min(deadline,context.budget.clock()+2.0)
                    else:
                        latch(phase,getattr(error,'member',None) if type(error) is ResourceFailure else None)
                        if phase=='drain-deadline':break
                time.sleep(.1)
    except Exception:
        latch('sample-write')
    finally:
        identities.close()
    process.poll()
    receipt={'root_pid':process.pid,'root_created_100ns':created,'returncode':process.returncode,
        'commit_limit_mib':cap,'measurement_failed':first_failure is not None,'native_completion_claimed':False,
        'resource_failure':first_failure,'verified_terminal_teardown':terminal,'drain_finished':process.returncode is not None}
    try:(workdir/'process-exit.json').write_bytes(native.json_bytes(receipt))
    except Exception:
        latch('sample-write');raise native.DispatchStopped('resource receipt persistence failed') from None
    if first_failure is not None:raise native.DispatchStopped('resource guard failed; categorical receipt retained')
    return receipt
