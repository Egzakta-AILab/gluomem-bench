"""Native LME-S executor. Fresh prospective cohort; separate exact-byte approval.

Workers are fresh interpreters; their entire outer budget protocol terminates in
one coordinator. No SDK, memory algorithm, reader, scorer or clock is replaced.
The authenticated local RPC is accounting-only (never a generation endpoint).
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

# Absolute -I worker launch: never import neighboring bench/types.py as stdlib.
if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gluomem_bench.bench.lme_first import DispatchStopped, json_bytes, read_bound
from gluomem_bench.bench.lme_resource_lifecycle import ResourceFailure, discover_members, monitor


# Serialize preparation, not native histories or generation leases. Large bound
# candidate/plan inventories are released before the next history prepares.
_WORKER_BOOTSTRAP_LOCK = threading.Lock()


GUARD_REASONS = frozenset(('cancelled', 'campaign-failed', 'campaign-deadline',
    'job-deadline', 'job-failed', 'job-call-cap', 'campaign-call-cap', 'accounting-invalid', 'unknown'))
GUARD_OPERATIONS = frozenset(('check', 'remaining', 'lease', 'begin', 'release'))


def guard_refusal(context, op, error):
    """Origin reason only: never infer it from failure flags or exception text."""
    import math
    reason = error.__dict__.get('guard_reason')
    result = {'protocol': 'native-job-guard-refusal@v1',
              'reason': reason if type(reason) is str and reason in GUARD_REASONS else 'unknown',
              'source_operation': op if type(op) is str and op in GUARD_OPERATIONS else 'unknown'}
    for key, read in (
        ('clock_monotonic', context.budget.clock),
        ('job_deadline_monotonic', lambda: context.deadline),
        ('campaign_deadline_monotonic', lambda: context.budget.deadline),
        ('call_ordinal', lambda: context.calls),
        ('job_sha256', lambda: context.job.job_id),
    ):
        try:
            value = read()
            if key == 'job_sha256':
                value = hashlib.sha256(value.encode('utf-8')).hexdigest() if type(value) is str else None
            elif key == 'call_ordinal':
                value = value if type(value) is int and 0 <= value < 2**63 else None
            else:
                value = value if type(value) in (int, float) and 0 <= value < 1e15 and math.isfinite(value) else None
        except Exception:
            value = None
        result[key] = value
    return result


class RemoteContextDispatchStopped(DispatchStopped):
    """Categorical authenticated broker response; no native exception payload."""
    def __init__(self, receipt):
        super().__init__('remote accounting/job guard refused')
        self.guard_refusal = receipt


def _accounting_refusal(message):
    error = DispatchStopped(message)
    error.guard_reason = 'accounting-invalid'
    return error


class ContextBroker:
    """Each worker thread owns one persistent authenticated connection/lease.

    A server thread enters/exits the original Context's lease AND begin_call, so
    its thread-local exactly-once accounting survives native process isolation.
    A lost client closes any live lease and fails its job, never refunds spend.
    """
    def __init__(self):
        from multiprocessing.connection import Listener
        self.key = secrets.token_bytes(32)
        self.listener = Listener(('127.0.0.1', 0), backlog=512, authkey=self.key)
        self.address = self.listener.address
        self.contexts = {}
        self.lock = threading.RLock()
        self.threads = []
        self.clients = set()
        self.stopping = False

    def register(self, context):
        handle = secrets.token_hex(32)
        with self.lock:
            self.contexts[handle] = context
        return handle

    def __enter__(self):
        self.acceptor = threading.Thread(target=self._accept, daemon=True)
        self.acceptor.start()
        return self

    def _accept(self):
        while not self.stopping:
            try:
                conn = self.listener.accept()
            except (OSError, EOFError):
                return
            if self.stopping:
                conn.close()
                return
            thread = threading.Thread(target=self._serve, args=(conn,), daemon=True)
            with self.lock:
                self.clients.add(conn)
                self.threads.append(thread)
            thread.start()

    def _serve(self, conn):
        context = lease = None
        try:
            handle = conn.recv()
            with self.lock:
                context = self.contexts[handle]
            conn.send({'ok': True, 'value': None})
            while True:
                op, value = conn.recv()
                try:
                    result = None
                    if op == 'lease':
                        if lease is not None:
                            context.budget.failed = True
                            raise _accounting_refusal('duplicate remote lease')
                        candidate = context.lease(value)
                        candidate.__enter__()
                        lease = candidate
                    elif op == 'release':
                        if lease is None:
                            context.budget.failed = True
                            raise _accounting_refusal('missing remote lease')
                        active, lease = lease, None
                        active.__exit__(None, None, None)
                    elif op == 'begin':
                        context.begin_call()
                    elif op == 'check':
                        context.check()
                    elif op == 'remaining':
                        result = context.remaining()
                    elif op == 'calls':
                        result = context.calls
                    elif op == 'failed':
                        result = context.failed
                    elif op == 'fail':
                        context.failed = True
                    elif op == 'shared_failed':
                        result = context.budget.failed
                    elif op == 'fail_shared':
                        context.budget.failed = True
                        if context.cancel is not None:
                            context.cancel.set()
                    elif op == 'close':
                        conn.send({'ok': True, 'value': None})
                        break
                    else:
                        context.budget.failed = True
                        raise _accounting_refusal('unknown accounting operation')
                    conn.send({'ok': True, 'value': result})
                except BaseException as error:
                    # Capture before any further RPC can mutate this job's state.
                    conn.send({'ok': False, 'category': 'accounting-or-job-guard-refused',
                               'guard_refusal': guard_refusal(context, op, error)})
        except (EOFError, OSError, KeyError):
            if context is not None and lease is not None:
                context.failed = True
        finally:
            if lease is not None:
                try:
                    lease.__exit__(None, None, None)
                except BaseException:
                    if context is not None:
                        context.budget.failed = True
            conn.close()
            with self.lock:
                self.clients.discard(conn)

    def __exit__(self, *args):
        from multiprocessing.connection import Client
        self.stopping = True
        try:
            wake = Client(self.address, authkey=self.key)
            wake.close()
        except (OSError, EOFError):
            pass
        self.listener.close()
        self.acceptor.join(timeout=5)
        for thread in self.threads:
            thread.join(timeout=5)
        if any(t.is_alive() for t in self.threads):
            raise DispatchStopped('native accounting connection still live')


class RemoteCampaign:
    def __init__(self, context, max_calls, max_seconds):
        self.context, self.max_calls, self.max_seconds = context, max_calls, max_seconds

    @property
    def failed(self):
        return self.context._rpc('shared_failed')

    @failed.setter
    def failed(self, value):
        if value:
            self.context._rpc('fail_shared')


class RemoteContext:
    """JobContext's actual transport protocol, not a private worker allowance."""
    def __init__(self, address, key, handle, job, *, max_calls=None, max_seconds=None):
        self.address, self.key, self.handle, self.job = tuple(address), key, handle, job
        self.local = threading.local()
        self.connections = []
        self.idle = []
        self.condition = threading.Condition()
        self.opening = 0
        self.closed = False
        self.budget = RemoteCampaign(self, max_calls, max_seconds)

    def _acquire(self):
        from multiprocessing.connection import Client
        with self.condition:
            while True:
                if self.closed:
                    raise DispatchStopped('remote accounting context closed')
                if self.idle:
                    return self.idle.pop()
                if len(self.connections) + self.opening < 300:
                    self.opening += 1
                    break
                self.condition.wait()
        conn = None
        try:
            conn = Client(self.address, authkey=self.key)
            conn.send(self.handle)
            if not conn.recv()['ok']:
                raise DispatchStopped('accounting identity refused')
            with self.condition:
                if self.closed:
                    raise DispatchStopped('remote accounting context closed')
                self.connections.append(conn)
            return conn
        except BaseException:
            if conn is not None:
                conn.close()
            raise
        finally:
            with self.condition:
                self.opening -= 1
                self.condition.notify_all()

    def _return(self, conn):
        with self.condition:
            if conn in self.connections and not self.closed:
                self.idle.append(conn)
            self.condition.notify_all()

    def _discard(self, conn):
        with self.condition:
            if conn in self.connections:
                self.connections.remove(conn)
            conn.close()
            self.condition.notify_all()

    def _rpc(self, op, value=None):
        pinned = getattr(self.local, 'conn', None)
        conn = pinned if pinned is not None else self._acquire()
        try:
            conn.send((op, value))
            reply = conn.recv()
            if not reply['ok']:
                raise RemoteContextDispatchStopped(reply.get('guard_refusal'))
            return reply['value']
        except (OSError, EOFError):
            self._discard(conn)
            raise
        finally:
            if pinned is None:
                self._return(conn)

    def check(self):
        self._rpc('check')

    def remaining(self):
        return self._rpc('remaining')

    def begin_call(self):
        self._rpc('begin')

    @property
    def calls(self):
        return self._rpc('calls')

    @property
    def failed(self):
        return self._rpc('failed')

    @failed.setter
    def failed(self, value):
        if value:
            self._rpc('fail')

    @contextmanager
    def lease(self, tokens):
        # Pin the whole lease, begin_call and release to ONE broker thread.
        # Outside leases RPCs briefly borrow idle connections; dead native caller
        # threads no longer strand a socket + broker thread until history close.
        if getattr(self.local, 'conn', None) is not None:
            self._rpc('lease', tokens)  # Preserve duplicate-lease global guard.
            raise DispatchStopped('nested remote lease')
        conn = self._acquire()
        self.local.conn = conn
        leased = False
        try:
            self._rpc('lease', tokens)
            leased = True
            yield
        finally:
            try:
                if leased and not conn.closed:
                    self._rpc('release')
            finally:
                del self.local.conn
                self._return(conn)

    def close(self):
        # Native lifecycle has joined its own request threads before close.
        with self.condition:
            if self.closed:
                return
            self.closed = True
            connections = list(self.connections)
            self.connections.clear()
            self.idle.clear()
            self.condition.notify_all()
        first_error = None
        for conn in connections:
            try:
                conn.send(('close', None))
                conn.recv()
            except (OSError, EOFError) as exc:
                first_error = first_error or exc
            finally:
                conn.close()
        if first_error is not None:
            raise DispatchStopped('remote accounting close failed') from None


class MemoryAdmission:
    """Measured admission, never treating tiny qualification RSS as an upper bound.

    Start only after real native measurements have seeded each backend. Subsequent
    complete process-tree samples can raise (never lower) the reservation. Host
    available RAM and an explicit reserve are rechecked before each launch.
    """
    def __init__(self, *, host_reserve_mib, available=None, host_status=None, commit_limits=None):
        if type(host_reserve_mib) is not int or host_reserve_mib < 1:
            raise ValueError('explicit positive host reserve required')
        self.host_reserve_mib = host_reserve_mib
        self.available = available
        self.host_status = host_status or host_memory_status
        self.commit_limits = dict(commit_limits or {})
        if any(type(v) is not int or v < 1 for v in self.commit_limits.values()):
            raise ValueError('explicit positive commit limits required')
        self.observed = {}
        self.live = {}
        self.pressure_latched = False
        self.last_host = None
        self.lock = threading.RLock()

    def observe(self, system, *, rss_mib, private_mib):
        import math
        if any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0
               for v in (rss_mib,private_mib)):
            raise ValueError('positive finite measured whole-tree memory required')
        with self.lock:
            old = self.observed.get(system, {'rss_mib':0, 'private_mib':0})
            self.observed[system] = {k:max(old[k], v) for k,v in
                                    {'rss_mib':rss_mib,'private_mib':private_mib}.items()}

    def reservation(self, system):
        with self.lock:
            return max(self.reservations(system).values())

    def reservations(self, system):
        with self.lock:
            cap = self.commit_limits.get(system,0)
            return {k:max(v*1.5,cap) for k,v in self.observed[system].items()}

    def _pending(self):
        pending = {'rss_mib':0, 'private_mib':0}
        for system, measured in self.live.values():
            for key,value in self.reservations(system).items():
                pending[key] += max(0,value-measured[key])
        return pending

    def _admit(self, system, *, existing=False):
        import math
        if self.pressure_latched or system not in self.observed:
            return False
        try:
            host = self.host_status()
            # Legacy test injection affects physical memory only; commit remains
            # independently OS-measured, never inferred from RSS or free RAM.
            if self.available is not None:
                host = dict(host,available_physical_mib=self.available())
            if any(type(host[k]) not in (int,float) or not math.isfinite(host[k]) or host[k] < 0
                   for k in ('available_physical_mib','commit_headroom_mib')):
                raise OSError('invalid host memory sample')
            self.last_host = dict(host)
        except Exception:
            self.pressure_latched = True
            return False
        if min(host['available_physical_mib'],host['commit_headroom_mib']) < self.host_reserve_mib:
            self.pressure_latched = True
            return False
        pending = self._pending()
        required = {'rss_mib':0,'private_mib':0} if existing else self.reservations(system)
        return (host['available_physical_mib']-self.host_reserve_mib-pending['rss_mib'] >= required['rss_mib']
                and host['commit_headroom_mib']-self.host_reserve_mib-pending['private_mib'] >= required['private_mib'])

    def admit(self, system):
        with self.lock:
            return self._admit(system)

    def admit_existing(self, identity):
        """Recheck physical/commit pressure without charging an owned group twice."""
        with self.lock:
            return identity in self.live and self._admit(self.live[identity][0],existing=True)

    def acquire(self, identity, system):
        with self.lock:
            if identity in self.live:
                raise ValueError('duplicate memory reservation')
            if not self._admit(system):
                return False
            self.live[identity] = (system,{'rss_mib':0,'private_mib':0})
            return True

    def update(self, identity, *, rss_mib, private_mib):
        with self.lock:
            system,_ = self.live[identity]
            self.observe(system,rss_mib=rss_mib,private_mib=private_mib)
            self.live[identity] = (system,{'rss_mib':rss_mib,'private_mib':private_mib})

    def release(self, identity):
        with self.lock:
            self.live.pop(identity,None)

    def latch_pressure(self):
        with self.lock:
            self.pressure_latched = True

    def snapshot(self):
        with self.lock:
            pending = self._pending()
            return {'host_reserve_mib':self.host_reserve_mib,
                    'observed':json.loads(json.dumps(self.observed)),
                    'commit_limits_mib':dict(self.commit_limits),
                    'last_host':self.last_host,'pressure_latched':self.pressure_latched,
                    'resident_jobs':len(self.live),
                    'startup_pending_rss_mib':pending['rss_mib'],
                    'startup_pending_commit_mib':pending['private_mib'],
                    'reservation_policy':'max(explicit inherited-tree commit cap, separate 1.5 * observed RSS/private-commit high-waters); startup deficits reserved atomically; not full-history maxima'}


def available_memory_mib():
    return host_memory_status()['available_physical_mib']


def host_memory_status():
    """OS physical availability and system commit headroom, both binary MiB.

    GetPerformanceInfo commit fields are PAGE COUNTS, not bytes or working set.
    GlobalMemoryStatusEx availPageFile is process-specific and is not used as
    system commit headroom.
    """
    import ctypes
    from ctypes import wintypes
    class Status(ctypes.Structure):
        _fields_ = [('length',wintypes.DWORD),('load',wintypes.DWORD)] + [(n,ctypes.c_ulonglong) for n in
                    ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
    status = Status()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('host memory measurement unavailable')
    class Performance(ctypes.Structure):
        _fields_ = [('cb',wintypes.DWORD)] + [(n,ctypes.c_size_t) for n in
            ('commit_total','commit_limit','commit_peak','physical_total','physical_available',
             'system_cache','kernel_total','kernel_paged','kernel_nonpaged','page_size')] + [
             ('handles',wintypes.DWORD),('processes',wintypes.DWORD),('threads',wintypes.DWORD)]
    performance = Performance()
    performance.cb = ctypes.sizeof(performance)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(performance),performance.cb):
        raise OSError('system commit measurement unavailable')
    if not performance.page_size or performance.commit_total > performance.commit_limit:
        raise OSError('invalid system commit measurement')
    return {'available_physical_mib':status.available / 1048576,
            'total_physical_mib':status.total / 1048576,
            'commit_headroom_mib':(performance.commit_limit-performance.commit_total)*performance.page_size / 1048576,
            'commit_total_mib':performance.commit_total*performance.page_size / 1048576,
            'commit_limit_mib':performance.commit_limit*performance.page_size / 1048576,
            'sample_monotonic':time.monotonic(),'units':'MiB (1048576 bytes)',
            'source':'GlobalMemoryStatusEx + GetPerformanceInfo'}


def source_rows(path):
    """Stream a JSON array, retaining at most one raw source row plus a chunk."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding='utf-8') as stream:
        buffer = ''
        eof = False
        def extend():
            nonlocal buffer,eof
            chunk = stream.read(1048576)
            buffer += chunk
            eof = not chunk
        extend()
        buffer = buffer.lstrip()
        if not buffer.startswith('['):
            raise ValueError('source JSON array required')
        buffer = buffer[1:]
        first = True
        while True:
            buffer = buffer.lstrip()
            while not buffer and not eof:
                extend()
                buffer = buffer.lstrip()
            if buffer.startswith(']'):
                buffer = buffer[1:]
                if buffer.strip() or stream.read().strip():
                    raise ValueError('unexpected source suffix')
                return
            if not first:
                if not buffer.startswith(','):
                    raise ValueError('source row separator missing')
                buffer = buffer[1:].lstrip()
            while True:
                try:
                    row,end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    extend()
                    buffer = buffer.lstrip()
            buffer = buffer[end:]
            first = False
            if not isinstance(row,dict):
                raise ValueError('source rows must be objects')
            yield row


def materialize_source(*, source, source_sha256, out, host_reserve_mib, commit_limit_mib):
    """ZERO inference. Source length selection and full allowed-event materialization.

    Execute in its own CLI process: the OS ceiling applies only to that process.
    This never opens a memory DB, constructs an LM/encoder, or calls ingestion.
    The report is explicitly NOT a completed-ingestion RAM qualification.
    """
    import socket
    from dataclasses import asdict
    if type(host_reserve_mib) is not int or host_reserve_mib < 1:
        raise ValueError('positive explicit host reserve required')
    out = Path(out)
    if out.exists():
        raise FileExistsError('fresh materialization output required')
    host = host_memory_status()
    if min(host['available_physical_mib'],host['commit_headroom_mib']) < host_reserve_mib+commit_limit_mib:
        raise DispatchStopped('materialization envelope exceeds OS headroom and reserve')
    check_file(source,source_sha256)
    cap = ProcessCommitLimit(commit_limit_mib)
    original_connect,original_connect_ex = socket.socket.connect,socket.socket.connect_ex
    denied = 0
    def deny(*args,**kwargs):
        nonlocal denied
        denied += 1
        raise OSError('materialization network denied')
    socket.socket.connect = socket.socket.connect_ex = deny
    try:
        from gluomem_bench.bench.lme_projection import project_history
        from gluomem_bench.bench.suite_campaign import make_source_events
        out.mkdir(parents=True,exist_ok=False)
        rows = []
        samples = []
        def measure():
            current = host_memory_status()
            if min(current['available_physical_mib'],current['commit_headroom_mib']) < host_reserve_mib:
                raise DispatchStopped('materialization host pressure')
            sample = process_tree_memory(os.getpid())
            samples.append(sample)
            return sample
        measure()
        for raw in source_rows(source):
            conv = project_history(raw)
            characters = sum(len(t.text) for s in conv.sessions for t in s.turns)
            rows.append({'sample_id':raw['question_id'],'characters':characters})
            measure()
        if not rows or len({r['sample_id'] for r in rows}) != len(rows):
            raise ValueError('distinct source history identities required')
        ordered = sorted(rows,key=lambda r:(r['characters'],r['sample_id']))
        indices = (0,(len(rows)-1)//2,((len(rows)-1)*9)//10,len(rows)-1)
        selected = [dict(ordered[i],stratum=s) for s,i in zip(('minimum','median','p90','maximum'),indices)]
        identities = {r['sample_id'] for r in selected}
        evidence = {}
        for raw in source_rows(source):
            if raw['question_id'] not in identities:
                continue
            conv = project_history(raw)
            events = make_source_events(conv,'longmemeval_s')
            value = asdict(conv)
            for session in value['sessions']:
                session['date_time'] = session['date_time'].isoformat()
            evidence[raw['question_id']] = {'source_events':len(events),
                'source_messages':sum(len(e.messages) for e in events),
                'projection_sha256':hashlib.sha256(json_bytes(value)).hexdigest(),
                'materialized_sample':measure()}
        for row in selected:
            row.update(evidence[row['sample_id']])
        check_file(source,source_sha256)
        report = {'resource_protocol':RESOURCE_PROTOCOL,'scope':'allowed source and production SourceEvent materialization only',
            'source':{'path':str(Path(source).resolve()),'sha256':source_sha256},
            'source_histories':len(rows),'selected':selected,'selection':'source character length only; stable sample-id ties; floor((N-1)*quantile)',
            'generation_calls':0,'encoder_calls':0,'native_ingestion_complete':False,
            'full_history_peak_mib':None,'network_policy':'deny-all-sockets','denied_connections':denied,
            'commit_limit_mib':commit_limit_mib,'host_reserve_mib':host_reserve_mib,
            'sampled_process_highwater':{k:max(r[k] for r in samples) for k in ('rss_mib','private_mib')},
            'executor':bound(__file__),'samples':samples}
        (out/'materialization.json').write_bytes(json_bytes(report))
        return report
    finally:
        socket.socket.connect,socket.socket.connect_ex = original_connect,original_connect_ex
        cap.close()


def main(argv=None):
    try:
        return _main(argv)
    except Exception:
        # Do not let a native DatasetError/SDK failure print arbitrary bodies,
        # headers, reasons, source content, dynamic class names or tracebacks.
        print(json.dumps({'status':'refused-or-incomplete','error':{
            'type':'native-executor-error','message':'admission or execution failed'}}))
        return 1


def parse_native_limits(value):
    """Parse explicit CLI ceilings without silently collapsing duplicate keys."""
    import argparse
    try:
        pairs = [part.split("=") for part in value.split(",")]
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError
        limits = {name: int(count) for name, count in pairs}
        if (len(limits) != len(pairs)
            or set(limits) != {"gluomem", "mem0", "cognee", "no_memory"}
            or any(not 1 <= count <= 500 for count in limits.values())
            or limits["cognee"] > 2):
            raise ValueError
        return limits
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(
            "unique gluomem,mem0,cognee,no_memory integer ceilings required; "
            "range 1..500 and Cognee at most 2"
        ) from None


def _main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    execute = sub.add_parser('execute')
    execute.add_argument('--candidate',type=Path,required=True)
    execute.add_argument('--approval',type=Path,required=True)
    worker = sub.add_parser('worker')
    worker.add_argument('--request',type=Path,required=True)
    group_worker = sub.add_parser('group-worker')
    group_worker.add_argument('--request',type=Path,required=True)
    for name in ('verify',):
        verify = sub.add_parser(name)
        verify.add_argument('--candidate',type=Path,required=True)
        verify.add_argument('--approval',type=Path,required=True)
    freeze = sub.add_parser('freeze')
    for flag in ('plan','budget-decision','qualification'):
        freeze.add_argument('--'+flag,type=Path,required=True)
        freeze.add_argument('--'+flag+'-sha256',required=True)
    freeze.add_argument('--output-root',type=Path,required=True)
    freeze.add_argument('--out',type=Path,required=True)
    freeze.add_argument('--cognee-python',type=Path,required=True)
    freeze.add_argument('--host-reserve-mib',type=int,required=True)
    freeze.add_argument('--max-workers',type=int,default=500)
    freeze.add_argument('--native-limits',type=parse_native_limits)
    freeze.add_argument('--history-commit-limit-mib',type=int,required=True)
    freeze.add_argument('--native-mode',choices=('process','gluomem-shared-process'),default='process')
    freeze.add_argument('--gluomem-group-commit-limit-mib',type=int)
    freeze.add_argument('--shared-gluomem-qualification',type=Path)
    freeze.add_argument('--shared-gluomem-qualification-sha256')
    materialize = sub.add_parser('materialize',help='zero-inference allowed-source memory measurement')
    materialize.add_argument('--source',type=Path,required=True)
    materialize.add_argument('--source-sha256',required=True)
    materialize.add_argument('--out',type=Path,required=True)
    materialize.add_argument('--host-reserve-mib',type=int,required=True)
    materialize.add_argument('--commit-limit-mib',type=int,required=True)
    resources = sub.add_parser('resources',help='read-only OS physical, system commit and exact process tree')
    resources.add_argument('--pid',type=int,default=os.getpid())
    args = parser.parse_args(argv)
    if args.command == 'resources':
        print(json.dumps({'host':host_memory_status(),'tree':process_tree_memory(args.pid)}))
        return 0
    if args.command == 'materialize':
        report = materialize_source(source=args.source,source_sha256=args.source_sha256,out=args.out,
            host_reserve_mib=args.host_reserve_mib,commit_limit_mib=args.commit_limit_mib)
        print(json.dumps({'materialized':True,'source_histories':report['source_histories'],
            'generation_calls':0,'encoder_calls':0,'native_ingestion_complete':False}))
        return 0
    if args.command == 'worker':
        return worker_main(args.request)
    if args.command == 'group-worker':
        return group_worker_main(args.request)
    if args.command == 'freeze':
        result = freeze_candidate(plan_path=args.plan,plan_sha256=args.plan_sha256,
            budget_decision_path=args.budget_decision,budget_decision_sha256=args.budget_decision_sha256,
            qualification_path=args.qualification,qualification_sha256=args.qualification_sha256,
            output_root=args.output_root,out=args.out,cognee_python=args.cognee_python,
            host_reserve_mib=args.host_reserve_mib,max_workers=args.max_workers,
            native_limits=args.native_limits,
            history_commit_limit_mib=args.history_commit_limit_mib,native_mode=args.native_mode,
            gluomem_group_commit_limit_mib=args.gluomem_group_commit_limit_mib,
            shared_gluomem_qualification_path=args.shared_gluomem_qualification,
            shared_gluomem_qualification_sha256=args.shared_gluomem_qualification_sha256)
    elif args.command == 'verify':
        approval = json.loads(args.approval.read_bytes())
        candidate,plan,qualification = validate_candidate(args.candidate,approval,post_exit=True)
        result = verify_root(candidate,plan=plan,qualification=qualification)
    else:
        result = execute_candidate(args.candidate,args.approval)
    print(json.dumps({'command':args.command,'quality_complete':result.get('quality_complete',False),
                      **(result if args.command=='freeze' else {})}))
    return 0 if args.command=='freeze' or result.get('quality_complete') is True else 2


def run_native_job(*, job, context, cfg, adapter, sample, profile, judge_source,
                   embedder, cognee_settings=None, transport=None, require_real=True):
    """Concrete native factories and unchanged common reader + native scorer.

    Qualification may call this seam in its own fresh process. Operational
    execution ALWAYS calls it through the isolated interpreter below.
    """
    from gluomem_bench.bench.suite_campaign import run_history
    common = dict(job=job,context=context,cfg=cfg,adapter=adapter,sample=sample,
                  profile=profile,judge_source=judge_source,require_real=require_real)
    if job.system == 'cognee':
        if cognee_settings is None:
            raise ValueError('explicit isolated native Cognee settings required')
        from gluomem_bench.bench.cognee_driver import run_graph_job
        run_graph_job(**common, settings=cognee_settings,embedder=embedder,transport=transport)
    elif job.system in ('gluomem','mem0','no_memory'):
        run_history(cfg=cfg,adapter=adapter,samples=[sample],profile=profile,out=job.output,
                    system=job.system,judge_source=judge_source,embedder=embedder,
                    outer_budget=context,evidence_token_budget=4096,transport=transport)
    else:
        raise ValueError('unadmitted native system')
    return verify_native_job(**common)


def verify_native_job(*, job, context, cfg, adapter, sample, profile, judge_source,
                      require_real=True):
    """Independent full native leaf checks; no callback success can certify QA."""
    from dataclasses import asdict
    from gluomem_bench.bench.suite_campaign import verify_history
    if job.system == 'cognee':
        from gluomem_bench.bench.cognee_driver import reconcile_job
        from gluomem_bench.bench.cognee_retriever import session_documents
        conv = next(c for c in adapter.conversations() if c.conv_id == sample.conv_id)
        return reconcile_job(out=job.output,documents=session_documents(conv),sample=sample,
                             profile=profile,cfg=cfg,adapter=adapter,judge_source=judge_source,
                             expected_calls=context.calls,require_real=require_real)
    expected = {'dataset':adapter.name,'system':job.system,'sample_ids':[sample.id],
                'profile_sha256':hashlib.sha256(profile.model_dump_json().encode()).hexdigest(),
                'dataset_sha256':adapter.sha}
    from gluomem_bench.bench.rack_schema import production_expectation
    expected.update(production_expectation(job.system, profile, cfg))
    verification = {}
    if job.system == 'mem0':
        from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT
        from gluomem_bench.bench.mem0 import _mem0_digest, _session_messages, mem0_adapted_binding
        conv = next(c for c in adapter.conversations() if c.conv_id == sample.conv_id)
        adapted = cfg.benchmark.mem0.max_batch_input_tokens is not None
        sessions = sorted(conv.sessions, key=lambda s: (s.date_time, s.index)) if adapted else conv.sessions
        originals = [
            {'session_index':s.index,'session_date':s.date_time.isoformat(),'messages':_session_messages(s)}
            for s in sessions]
        expected.update(mem0_source_inputs_sha256=_mem0_digest(originals),
                        mem0_extractor_system_sha256=hashlib.sha256(ADDITIVE_EXTRACTION_PROMPT.encode()).hexdigest())
        if adapted:
            _, binding = mem0_adapted_binding(cfg, originals, scope=sample.conv_id)
            if binding['mem0_original_source_inputs_sha256'] != expected['mem0_source_inputs_sha256']:
                raise ValueError('adapted original source binding differs')
            expected.update(binding)
            verification = dict(mem0_cfg=cfg, mem0_original_inputs=originals, mem0_scope=sample.conv_id)
    summary = verify_history(job.output,expected_job=expected,require_real=require_real,
                             judge_source=judge_source, **verification)
    if json.loads((job.output/'effective-config.json').read_bytes()) != cfg.model_dump(mode='json'):
        raise ValueError('saved native configuration differs')
    rows = json.loads((job.output/'predictions.json').read_bytes())
    if rows[0]['sample'] != json.loads(json.dumps(asdict(sample))):
        raise ValueError('saved native sample/clock/scorer inputs differ')
    if job.system == 'gluomem':
        from gluomem_bench.bench.beam_ingest import _read_rows, _validate_rows
        from gluomem_bench.bench.suite_campaign import make_source_events
        conv = next(c for c in adapter.conversations() if c.conv_id == sample.conv_id)
        _validate_rows(_read_rows(job.output/'stores'/f'{conv.conv_id}.mind'),
                       make_source_events(conv,adapter.name))
    usage = json.loads((job.output/'usage.json').read_bytes())
    if usage['generation_calls'] != context.calls:
        raise ValueError('native leaf and global attempt accounting differ')
    return {'quality_status':'complete_verified','summary':summary,'expected_job':expected,
            'generation_calls':context.calls}


def chronological_tree(rows, pid, created_100ns):
    """Reject PID reuse and older false descendants before aggregating bytes."""
    by_pid = {r['pid']:r for r in rows}
    if pid not in by_pid or by_pid[pid]['created_100ns'] != created_100ns:
        raise OSError('original worker identity no longer live')
    selected = {pid:by_pid[pid]}
    while True:
        additions = {p:r for p,r in by_pid.items() if p not in selected
            and r['parent_pid'] in selected
            and r['created_100ns'] >= selected[r['parent_pid']]['created_100ns']}
        if not additions:
            return sorted(selected.values(),key=lambda r:(r['created_100ns'],r['pid']))
        selected.update(additions)


def process_creation_time(pid):
    import ctypes as c
    from ctypes import wintypes as w
    kernel = c.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes = [w.DWORD,w.BOOL,w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    kernel.GetProcessTimes.argtypes = [w.HANDLE] + [c.POINTER(w.FILETIME)]*4
    handle = kernel.OpenProcess(0x1000,False,pid)
    if not handle:
        raise OSError('process identity unavailable')
    try:
        times = [w.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle,*(c.byref(t) for t in times)):
            raise OSError('process creation time unavailable')
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


def process_tree_memory(pid, *, created_100ns=None):
    """Chronological exact-identity whole-tree RSS/private-commit; no VMS alias."""
    import ctypes as c
    from ctypes import wintypes as w
    kernel = c.WinDLL('kernel32',use_last_error=True)
    psapi = c.WinDLL('psapi',use_last_error=True)
    class Entry(c.Structure):
        _fields_ = [('size',w.DWORD),('usage',w.DWORD),('pid',w.DWORD),('heap',c.c_size_t),
                    ('module',w.DWORD),('threads',w.DWORD),('parent',w.DWORD),('priority',w.LONG),
                    ('flags',w.DWORD),('exe',w.WCHAR*260)]
    class Memory(c.Structure):
        _fields_ = [('cb',w.DWORD),('faults',w.DWORD)] + [(n,c.c_size_t) for n in
                    ('peak_rss','rss','paged_peak','paged','nonpaged_peak','nonpaged','pagefile','peak_pagefile','private')]
    kernel.CreateToolhelp32Snapshot.restype = w.HANDLE
    kernel.OpenProcess.restype = w.HANDLE
    kernel.OpenProcess.argtypes = [w.DWORD,w.BOOL,w.DWORD]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    kernel.WaitForSingleObject.argtypes = [w.HANDLE,w.DWORD]
    kernel.Process32FirstW.argtypes = [w.HANDLE,c.POINTER(Entry)]
    kernel.Process32NextW.argtypes = [w.HANDLE,c.POINTER(Entry)]
    psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE,c.POINTER(Memory),w.DWORD]
    snapshot = kernel.CreateToolhelp32Snapshot(2,0)
    if snapshot == c.c_void_p(-1).value:
        raise ResourceFailure('process-snapshot')
    parents = {}
    try:
        entry = Entry()
        entry.size = c.sizeof(entry)
        ok = kernel.Process32FirstW(snapshot,c.byref(entry))
        if not ok:
            raise ResourceFailure('process-snapshot')
        while ok:
            parents[entry.pid] = entry.parent
            ok = kernel.Process32NextW(snapshot,c.byref(entry))
            if not ok and c.get_last_error()!=18:
                raise ResourceFailure('process-snapshot')
    finally:
        kernel.CloseHandle(snapshot)
    try:
        created_100ns = process_creation_time(pid) if created_100ns is None else created_100ns
    except OSError:
        raise ResourceFailure('identity') from None
    members = discover_members(parents, pid, created_100ns, process_creation_time)
    rss = private = 0
    measured = []
    for row in members:
        identity = row['pid']
        handle = kernel.OpenProcess(0x100000 | 0x0400 | 0x0010,False,identity)
        if not handle:
            raise ResourceFailure('member-open',row)
        try:
            # A retained handle can expose zeroed counters AFTER exact exit.
            # That is a terminal sample loss, never a valid zero-RSS window.
            if kernel.WaitForSingleObject(handle,0) != 258:
                raise ResourceFailure('memory-query',row)
            memory = Memory()
            memory.cb = c.sizeof(memory)
            if not psapi.GetProcessMemoryInfo(handle,c.byref(memory),memory.cb):
                raise ResourceFailure('memory-query',row)
            try:
                same_identity = process_creation_time(identity) == row['created_100ns']
            except OSError:
                raise ResourceFailure('identity',row) from None
            if not same_identity:
                raise ResourceFailure('identity',row)
            if kernel.WaitForSingleObject(handle,0) != 258:
                raise ResourceFailure('memory-query',row)
            rss += memory.rss
            private += memory.private
            measured.append(dict(row,rss_mib=memory.rss/1048576,
                                 private_mib=memory.private/1048576,
                                 peak_rss_mib=memory.peak_rss/1048576))
        finally:
            kernel.CloseHandle(handle)
    return {'rss_mib':rss/1048576,'private_mib':private/1048576,'processes':len(measured),
            'root_pid':pid,'root_created_100ns':created_100ns,'members':measured,
            'sample_monotonic':time.monotonic(),'units':'MiB (1048576 bytes)',
            'private_metric':'PROCESS_MEMORY_COUNTERS_EX.PrivateUsage (committed private bytes)'}


class ProcessCommitLimit:
    """Explicit OS-enforced committed-byte ceiling for this worker and children.

    No working-set/VMS fiction, no kill-on-close, no termination of processes.
    Windows refuses further commit allocations at the bound. A native runtime
    may exit rather than raise MemoryError; the parent retains a failed job.
    Set this before native imports/descendant creation. It is an operator bound,
    NOT a claim that a complete history fits it. Never apply in shared threads.
    """
    def __init__(self, limit_mib):
        import ctypes as c
        from ctypes import wintypes as w
        if type(limit_mib) is not int or limit_mib < 1:
            raise ValueError('explicit positive worker commit ceiling required')
        class Basic(c.Structure):
            _fields_ = [('process_time',c.c_longlong),('job_time',c.c_longlong),
                ('flags',w.DWORD),('min_ws',c.c_size_t),('max_ws',c.c_size_t),
                ('active',w.DWORD),('affinity',c.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
        class IO(c.Structure):
            _fields_ = [(n,c.c_ulonglong) for n in
                ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
        class Extended(c.Structure):
            _fields_ = [('basic',Basic),('io',IO),('process_limit',c.c_size_t),
                ('job_limit',c.c_size_t),('peak_process',c.c_size_t),('peak_job',c.c_size_t)]
        self.kernel = c.WinDLL('kernel32',use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [c.c_void_p,w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.GetCurrentProcess.restype = w.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [w.HANDLE,c.c_int,c.c_void_p,w.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [w.HANDLE,w.HANDLE]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None,None)
        if not self.handle:
            raise OSError('worker memory job unavailable')
        limits = Extended()
        limits.basic.flags = 0x200  # JOB_OBJECT_LIMIT_JOB_MEMORY, entire inherited tree
        limits.job_limit = limit_mib*1048576
        if (not self.kernel.SetInformationJobObject(self.handle,9,c.byref(limits),c.sizeof(limits))
            or not self.kernel.AssignProcessToJobObject(self.handle,self.kernel.GetCurrentProcess())):
            self.close()
            raise OSError('worker commit ceiling could not be applied')
        self.limit_mib = limit_mib

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def launch_worker(request, *, broker, context, workdir, python_executable=sys.executable,
                  sample_memory=None, memory_failure=None, host_reserve_mib=None):
    """Fresh process per history: Mem0 observers and DSPy/import env cannot mix."""
    import subprocess
    workdir = Path(workdir)
    workdir.mkdir(parents=True,exist_ok=False)
    request = dict(request,address=broker.address,handle=broker.register(context),
                   job={'job_id':context.job.job_id,'history_id':context.job.history_id,
                        'output':str(context.job.output),'call_budget':context.job.call_budget,
                        'seconds_budget':context.job.seconds_budget},
                   campaign_caps={'max_calls':context.budget.max_calls,'max_seconds':context.budget.max_seconds})
    path = workdir/'request.json'
    path.write_bytes(json_bytes(request))
    env = os.environ.copy()
    for key in ('OPENAI_API_KEY','OPENROUTER_API_KEY','ANTHROPIC_API_KEY'):
        env.pop(key,None)
    env.update(MEM0_TELEMETRY='False',MEM0_DIR=str(workdir/'mem0-home'),
               LITELLM_LOCAL_MODEL_COST_MAP='True',
               GLUOMEM_RACK_API_KEY='local-endpoint-placeholder',PYTHONDONTWRITEBYTECODE='1',
               LME_RPC_KEY=broker.key.hex())
    # No captured arbitrary native stderr/stdout can leak through coordinator errors.
    process = subprocess.Popen([str(python_executable),'-B','-I',str(Path(__file__).resolve()),
                                'worker','--request',str(path)],cwd=str(Path(__file__).resolve().parents[2]),
                               env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    created = process_creation_time(process.pid)
    monitor(process,native=sys.modules[__name__],context=context,workdir=workdir,created=created,
            cap=request.get('commit_limit_mib'),sample_memory=sample_memory,
            memory_failure=memory_failure,reserve=host_reserve_mib)
    context.check()
    outcome = json.loads((workdir/'outcome.json').read_bytes())
    if process.returncode != 0 or outcome.get('status') != 'native-closed-verified':
        raise DispatchStopped('isolated native worker failed; categorical receipt retained')
    if outcome['calls'] != context.calls:
        context.budget.failed = True
        raise DispatchStopped('isolated accounting differs')
    return outcome


def probe_subprocess(broker,job,context,workdir):
    return launch_worker({'probe':True},broker=broker,context=context,workdir=Path(workdir)/'process')


class SharedGluomemWorker:
    """One owned, capped interpreter; coordinator/governor stay outside its job.

    This low-level lifecycle does not grant campaign admission. Only GluoMem may
    share it. One aggregate ceiling is installed BEFORE native imports; there is
    no per-thread Job Object or per-history memory claim. Local failed histories
    retain their spend; a dead group cannot corrupt other backends' interpreters.
    """
    def __init__(self, *, workdir, commit_limit_mib, max_workers, sample_memory=None,
                 memory_failure=None, host_reserve_mib=None, network_endpoints=None):
        if (type(commit_limit_mib) is not int or commit_limit_mib < 1
            or type(max_workers) is not int or not 1 <= max_workers <= 500):
            raise ValueError('explicit bounded shared group capacity required')
        self.workdir = Path(workdir)
        self.limit = commit_limit_mib
        self.max_workers = max_workers
        self.sample_memory = sample_memory
        self.memory_failure = memory_failure
        self.reserve = host_reserve_mib
        self.network_endpoints = network_endpoints
        self.key = secrets.token_bytes(32)
        self.lock = threading.RLock()
        self.failed = False
        self.contexts = set()


    def __enter__(self):
        import subprocess
        self.workdir.mkdir(parents=True,exist_ok=False)
        config = self.workdir/'group-request.json'
        config.write_bytes(json_bytes({'commit_limit_mib':self.limit,'max_workers':self.max_workers,
                                      'network_endpoints':self.network_endpoints}))
        env = os.environ.copy()
        for key in ('OPENAI_API_KEY','OPENROUTER_API_KEY','ANTHROPIC_API_KEY'):
            env.pop(key,None)
        env.update(LME_GROUP_KEY=self.key.hex(),LITELLM_LOCAL_MODEL_COST_MAP='True',
                   GLUOMEM_RACK_API_KEY='local-endpoint-placeholder',PYTHONDONTWRITEBYTECODE='1')
        self.process = subprocess.Popen([str(sys.executable),'-I',str(Path(__file__).resolve()),
            'group-worker','--request',str(config)],cwd=str(Path(__file__).resolve().parents[2]),
            env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        self.created = process_creation_time(self.process.pid)
        self.monitor = threading.Thread(target=self._monitor,daemon=True)
        self.monitor.start()
        deadline = time.monotonic()+30
        while not (self.workdir/'ready.json').exists():
            if self.process.poll() is not None or time.monotonic() >= deadline:
                self.failed = True
                raise DispatchStopped('owned shared worker did not become ready')
            time.sleep(.01)
        ready = json.loads((self.workdir/'ready.json').read_bytes())
        self.address = tuple(ready['address'])
        return self

    def _monitor(self):
        with (self.workdir/'resource-samples.jsonl').open('xb') as stream:
            while self.process.poll() is None:
                try:
                    row = process_tree_memory(self.process.pid,created_100ns=self.created)
                    host = host_memory_status() if self.reserve is not None else None
                    if host is not None:
                        row['host_status'] = host
                    stream.write(json_bytes(row))
                    stream.flush()
                    if self.sample_memory is not None:
                        self.sample_memory(row)
                    if host is not None:
                        if min(host['available_physical_mib'],host['commit_headroom_mib']) < self.reserve:
                            raise OSError('shared group host reserve pressure')
                except Exception:
                    if self.process.poll() is None:
                        with self.lock:
                            self.failed = True
                            for context in self.contexts:
                                context.failed = True
                        if self.memory_failure is not None:
                            self.memory_failure()
                time.sleep(.1)

    def run(self, request, *, broker, context, workdir):
        from multiprocessing.connection import Client
        if request.get('system') != 'gluomem':
            raise ValueError('only GluoMem may enter shared process')
        with self.lock:
            if self.failed or self.process.poll() is not None:
                raise DispatchStopped('shared GluoMem process unavailable')
            self.contexts.add(context)
        workdir = Path(workdir)
        try:
            workdir.mkdir(parents=True,exist_ok=False)
            request = dict(request,address=broker.address,handle=broker.register(context),
                job={'job_id':context.job.job_id,'history_id':context.job.history_id,
                     'output':str(context.job.output),'call_budget':context.job.call_budget,
                     'seconds_budget':context.job.seconds_budget},
                campaign_caps={'max_calls':context.budget.max_calls,'max_seconds':context.budget.max_seconds},
                group_commit_limit_mib=self.limit)
            path = workdir/'request.json'
            path.write_bytes(json_bytes(request))
            conn = Client(self.address,authkey=self.key)
            try:
                conn.send({'op':'run','request':str(path),'rpc_key':broker.key.hex()})
                while not conn.poll(.1):
                    if self.process.poll() is not None:
                        raise DispatchStopped('shared GluoMem process exited')
                reply = conn.recv()
            finally:
                conn.close()
            outcome = json.loads((workdir/'outcome.json').read_bytes())
            if reply.get('code') != 0 or outcome.get('status') != 'native-closed-verified':
                raise DispatchStopped('shared native history failed; categorical receipt retained')
            if outcome.get('calls') != context.calls:
                context.budget.failed = True
                raise DispatchStopped('shared native accounting differs')
            context.check()
            return outcome
        except BaseException:
            context.failed = True
            raise
        finally:
            with self.lock:
                self.contexts.discard(context)

    def __exit__(self, *args):
        from multiprocessing.connection import Client
        if self.process.poll() is None:
            conn = Client(self.address,authkey=self.key)
            try:
                conn.send({'op':'close'})
                conn.recv()
            finally:
                conn.close()
            # Cooperative native drain, never kill a process or unrelated tree.
            self.process.wait()
        self.monitor.join()
        (self.workdir/'process-exit.json').write_bytes(json_bytes({
            'root_pid':self.process.pid,'root_created_100ns':self.created,
            'returncode':self.process.returncode,'commit_limit_mib':self.limit,
            'measurement_failed':self.failed,'native_completion_claimed':False}))


def group_worker_main(request_path):
    """Owned group process entrypoint. No coordinator or other backend lives here."""
    from concurrent.futures import ThreadPoolExecutor
    from multiprocessing.connection import Listener
    path = Path(request_path)
    config = json.loads(path.read_bytes())
    key = bytes.fromhex(os.environ.pop('LME_GROUP_KEY'))
    cap = ProcessCommitLimit(config['commit_limit_mib'])
    listener = Listener(('127.0.0.1',0),backlog=512,authkey=key)
    network = None
    if config.get('network_endpoints') is not None:
        from gluomem_bench.bench.native_backend_qualification import SocketBoundary
        network = SocketBoundary(config['network_endpoints'])
        network.install()
    outcome = {'status':'failed'}
    count = 0
    try:
        ready = path.parent/'ready.pending.json'
        ready.write_bytes(json_bytes({'address':listener.address,'pid':os.getpid()}))
        os.replace(ready,path.parent/'ready.json')
        with ThreadPoolExecutor(max_workers=config['max_workers']) as pool:
            def run(conn,message):
                try:
                    code = worker_main(message['request'],shared_group={
                        'key':bytes.fromhex(message['rpc_key']),'limit_mib':cap.limit_mib})
                    conn.send({'code':code})
                except BaseException:
                    conn.send({'code':1})
                finally:
                    conn.close()
            while True:
                conn = listener.accept()
                message = conn.recv()
                if message.get('op') == 'close':
                    conn.send({'closed':True})
                    conn.close()
                    break
                if message.get('op') != 'run':
                    conn.close()
                    continue
                count += 1
                pool.submit(run,conn,message)
        outcome = {'status':'native-closed-verified','histories_submitted':count}
        if network is not None:
            outcome['network'] = network.counts
    finally:
        listener.close()
        outcome['resource'] = {'commit_limit_mib':cap.limit_mib,
            'enforcement':'Windows inherited job commit ceiling','pid':os.getpid(),
            'created_100ns':process_creation_time(os.getpid()),
            'scope':'shared-gluomem-group-not-per-history'}
        (path.parent/'outcome.json').write_bytes(json_bytes(outcome))
        cap.close()
    return 0 if outcome['status']=='native-closed-verified' else 1


def worker_main(request_path, *, shared_group=None):
    request_path = Path(request_path)
    request = json.loads(request_path.read_bytes())
    from gluomem_bench.bench.parallel_campaign import Job
    job = Job(**request['job'])
    rpc_key = shared_group['key'] if shared_group is not None else bytes.fromhex(os.environ.pop('LME_RPC_KEY'))
    context = RemoteContext(request['address'],rpc_key,
                            request['handle'],job,**request['campaign_caps'])
    outcome = {'status':'failed','failure':{'type':'native-worker-failure','message':'native job failed'}}
    code = 1
    phase = 'identity'
    memory_cap = None
    try:
        if shared_group is not None:
            if (request.get('system') != 'gluomem' or 'commit_limit_mib' in request
                or request.get('group_commit_limit_mib') != shared_group['limit_mib']):
                raise ValueError('shared group cannot install per-history ceilings or other backends')
        if 'commit_limit_mib' in request:
            phase = 'resource'
            memory_cap = ProcessCommitLimit(request['commit_limit_mib'])
        phase = 'identity'
        if request.get('probe') is True:
            phase = 'native'
            with context.lease(128):
                context.begin_call()
            result = {'scope':'accounting-process-fixture-only','pid':os.getpid()}
        else:
            with _WORKER_BOOTSTRAP_LOCK:
                candidate = json.loads(read_bound(request['candidate']['path'],request['candidate']['sha256']))
                plan = json.loads(read_bound(candidate['plan']['path'],candidate['plan']['sha256']))
                if candidate.get('protocol') == EXECUTION_PROTOCOL:
                    if shared_group is not None:
                        if (candidate.get('native_mode') != 'gluomem-shared-process'
                            or candidate.get('gluomem_group_commit_limit_mib') != shared_group['limit_mib']):
                            raise ValueError('shared group differs from approved candidate')
                    elif (memory_cap is None or memory_cap.limit_mib != candidate['history_commit_limits_mib'][request['system']]):
                        raise ValueError('worker OS commit ceiling differs from approved candidate')
                for item in candidate['files']:
                    # Dataset/inventory were authenticated by parent before setup; only local producer
                    # and runtime dependencies are checked here, avoiding 2000 full dataset rehashes.
                    if item.get('worker_check',True):
                        check_file(item['path'],item['sha256'])
                cfg,profile = effective_config(plan)
                adapter,sample = restore_input(request['input'],plan)
                native_job = SimpleNamespace(**request['job'],system=request['system'])
                native_job.output = Path(native_job.output)
                from gluomem_bench.bench.embedding_runtime import RemoteEmbedder, EmbeddingRecipe
                embedder = RemoteEmbedder(plan['embedding_url'],recipe=EmbeddingRecipe(**plan['embedding_recipe']))
                settings = None
                if request['system'] == 'cognee':
                    from gluomem_bench.bench.comparison_backends import CogneeSettings
                    study_clocks = plan.get('remediation', {}).get('cognee_clocks')
                    clock_kwargs = {}
                    if study_clocks is not None:
                        import math
                        if (type(study_clocks) is not dict or set(study_clocks) != {
                            'protocol', 'request_timeout_seconds', 'native_llm_timeout_seconds',
                            'cleanup_reserve_seconds'} or
                            study_clocks['protocol'] != 'cognee-clocks@v1' or
                            study_clocks['request_timeout_seconds'] != profile.timeout_seconds or
                            study_clocks['native_llm_timeout_seconds'] != 90 or
                            type(study_clocks['cleanup_reserve_seconds']) not in (int, float) or
                            not math.isfinite(study_clocks['cleanup_reserve_seconds']) or
                            study_clocks['cleanup_reserve_seconds'] <= 0):
                            raise ValueError('explicit Cognee study clocks required')
                        clock_origin = time.monotonic()
                        remaining = context.remaining()
                        if remaining <= study_clocks['cleanup_reserve_seconds']:
                            raise TimeoutError('cleanup reserve exhausts remaining allowance')
                        clock_kwargs = dict(request_timeout_seconds=profile.timeout_seconds,
                            operation_deadline=clock_origin + remaining,
                            cleanup_reserve_seconds=study_clocks['cleanup_reserve_seconds'],
                            native_llm_timeout_seconds=90)
                    else:
                        remaining = context.remaining()
                    settings = CogneeSettings(python_executable=candidate['cognee_python'],
                        root=job.output/'graph/native',dataset='lme-'+job.job_id,
                        llm_url=profile.base_url,llm_model=profile.model,
                        llm_api_key='local-endpoint-placeholder',embedding_url=plan['embedding_url'],
                        embedding_model=plan['embedding_recipe']['model'],
                        embedding_dimensions=plan['embedding_recipe']['dim'],
                        embedding_api_key='local-endpoint-placeholder',max_completion_tokens=profile.output_budget('cognify'),
                        timeout_seconds=min(profile.timeout_seconds,remaining), **clock_kwargs)
                judge_source = Path(plan['judge_source'])
                del candidate, plan
            phase = 'native'
            result = run_native_job(job=native_job,context=context,cfg=cfg,adapter=adapter,
                     sample=sample,profile=profile,judge_source=judge_source,
                     embedder=embedder,cognee_settings=settings)
        context.check()
        outcome = {'status':'native-closed-verified','calls':context.calls,'result':result}
        code = 0
    except BaseException as exc:
        from gluomem_bench.bench.rack import failure_origin
        if request.get('system') == 'cognee':
            try:
                from gluomem_bench.bench.cognee_native_boundary import clean_diagnostic
                outcome['failure']['native_diagnostic'] = clean_diagnostic(
                    getattr(exc, 'native_diagnostic', None))
            except BaseException:
                pass  # Diagnostic failure must not replace the initiating exception.
        outcome['failure']['origin'] = failure_origin(exc, job_id=job.job_id)
        try:
            context.failed = True
            if phase == 'identity':
                context.budget.failed = True
            outcome['calls'] = context.calls
        except BaseException:
            outcome['calls'] = None
    finally:
        try:
            context.close()
        finally:
            outcome['resource'] = {'commit_limit_mib':memory_cap.limit_mib if memory_cap else None,
                'enforcement':'Windows inherited job commit ceiling' if memory_cap else 'none (fixture or legacy caller only)',
                'pid':os.getpid(),'created_100ns':process_creation_time(os.getpid())}
            if shared_group is not None:
                outcome['resource'].update(commit_limit_mib=shared_group['limit_mib'],
                    enforcement='Windows inherited job commit ceiling',
                    scope='shared-gluomem-group-not-per-history')
            (request_path.parent/'outcome.json').write_bytes(json_bytes(outcome))
            if memory_cap is not None:
                memory_cap.close()
    return code


def effective_config(plan):
    from gluomem_bench.bench.rack import RackProfile, controlled_config
    profile = RackProfile.model_validate(plan['profile'])
    cfg = controlled_config(profile,data=Path(plan['dataset']['path']),sha256=plan['dataset']['sha256'])
    return cfg.with_overrides(plan['config_overrides']),profile


def restore_input(binding,plan):
    from datetime import datetime
    from gluomem.write.conversation import Conversation, Session, Turn
    from gluomem_bench.bench.types import Sample
    data = json.loads(read_bound(binding['path'],binding['sha256']))
    value = data['sample']
    value['evidence'] = tuple(value.get('evidence',()))
    sample = Sample(**value)
    value = data['conversation']
    conv = Conversation(conv_id=value['conv_id'],speakers=tuple(value['speakers']),sessions=[
        Session(index=s['index'],date_time=datetime.fromisoformat(s['date_time']),
                turns=[Turn(**t) for t in s['turns']]) for s in value['sessions']])
    def verify():
        read_bound(binding['path'],binding['sha256'])
    adapter = SimpleNamespace(name=plan['dataset']['name'],sha=plan['dataset']['sha256'],
                              path=Path(plan['dataset']['path']),verify=verify,
                              load=lambda:[sample],conversations=lambda:[conv])
    return adapter,sample


EXECUTION_PROTOCOL = 'lme-first-native-execution@v2'
RESOURCE_PROTOCOL = 'os-physical-system-commit-chronological-tree@v1'
QUALIFICATION_ASSERTIONS = ('native_factory_reader_scorer','all_generation_shared_accounting',
    'no_aux_sdk_bypass','privacy_categorical','source_clock_exact','mem0_execution_date_disclosed',
    'native_process_isolation')


def bound(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            digest.update(block)
    return {'path':str(path),'sha256':digest.hexdigest()}


def check_file(path,digest):
    import re
    if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):
        raise ValueError('literal lowercase SHA256 required')
    with Path(path).open('rb') as stream:
        actual = hashlib.file_digest(stream,'sha256').hexdigest()
    if actual != digest:
        raise ValueError('frozen file binding differs')


def validate_qualification_closure(files, qualification):
    """Bind native package/SQL/binary boundaries, not merely bench/*.py names."""
    rows = qualification.get('source_closure',[])
    closure = {str(Path(r['path']).resolve()):r['sha256'] for r in rows}
    if len(closure) != len(rows):
        raise ValueError('duplicate qualification source closure identity')
    for item in files:
        path = Path(item['path']).resolve()
        if path.suffix.lower() in ('.py','.sql','.mako','.pyd','.dll','.so') or 'site-packages' in path.parts:
            if closure.get(str(path)) != item['sha256']:
                raise ValueError('qualification native/backend/runtime source closure differs')


def verify_resource_measurement(proof, *, commit_limit_mib, measured, host_reserve_mib=None):
    """Replay bound OS samples, exact identities and applied worker cap receipts.

    This proves sampled measurements, not a maximum for unsampled ingestion.
    Positive numbers in a self-declared qualification summary are insufficient.
    """
    import math
    samples = [json.loads(line) for line in read_bound(**_binding_args(proof['samples'])).splitlines()]
    outcome = json.loads(read_bound(**_binding_args(proof['outcome'])))
    closed = json.loads(read_bound(**_binding_args(proof['exit'])))
    if (not samples or closed.get('returncode') != 0 or closed.get('measurement_failed') is not False
        or closed.get('commit_limit_mib') != commit_limit_mib
        or outcome.get('status') != 'native-closed-verified'
        or outcome.get('resource',{}).get('commit_limit_mib') != commit_limit_mib
        or outcome['resource'].get('enforcement') != 'Windows inherited job commit ceiling'):
        raise ValueError('resource applied-limit/closure evidence missing')
    identity = (closed['root_pid'],closed['root_created_100ns'])
    prior = -1
    seen = set()
    for row in samples:
        if host_reserve_mib is not None:
            host = row.get('host_status',{})
            if (host.get('source') != 'GlobalMemoryStatusEx + GetPerformanceInfo'
                or host.get('units') != 'MiB (1048576 bytes)'
                or any(type(host.get(k)) not in (int,float) or not math.isfinite(host[k])
                    or host[k] < host_reserve_mib for k in ('available_physical_mib','commit_headroom_mib'))):
                raise ValueError('resource host reserve/physical/system-commit evidence missing or pressured')
        members = row.get('members',[])
        if ((row.get('root_pid'),row.get('root_created_100ns')) != identity
            or row.get('units') != 'MiB (1048576 bytes)' or not members
            or type(row.get('sample_monotonic')) not in (int,float)
            or not math.isfinite(row['sample_monotonic']) or row['sample_monotonic'] <= prior
            or row.get('processes') != len(members)
            or len({m['pid'] for m in members}) != len(members)):
            raise ValueError('resource sample identity/units/chronology differs')
        prior = row['sample_monotonic']
        try:
            actual = chronological_tree(members,*identity)
        except OSError as exc:
            raise ValueError('resource worker identity differs') from exc
        if len(actual) != len(members):
            raise ValueError('resource recycled/non-chronological descendant')
        for key in ('rss_mib','private_mib'):
            if (any(type(m.get(key)) not in (float,int) or not math.isfinite(m[key]) or m[key] <= 0 for m in members)
                or row.get(key) != sum(m[key] for m in members)):
                raise ValueError('resource totals must equal actual per-process byte measurements')
        seen.update((m['pid'],m['created_100ns']) for m in members)
    if (outcome['resource']['pid'],outcome['resource']['created_100ns']) not in seen:
        raise ValueError('resource samples never observed native worker identity')
    if any(measured[k] != max(r[k] for r in samples) for k in ('rss_mib','private_mib')):
        raise ValueError('resource declared high-water differs from actual samples')
    return {'samples':len(samples),'identities':len(seen),'sampled_highwater_only':True}


def verify_qualification_resources(qualification, limits):
    evidence = qualification.get('resource_evidence',{})
    if set(evidence) != {'gluomem','mem0','cognee'}:
        raise ValueError('resource OS sample/outcome/exit bindings required for all native systems')
    for system in evidence:
        verify_resource_measurement(evidence[system],commit_limit_mib=limits[system],
                                    measured=qualification['memory_samples'][system],host_reserve_mib=8192)


SHARED_ASSERTIONS = ('three_parallel_native_ingest_store_retrieval', 'tenant_isolation',
    'exact_source_reference_and_reader_clocks', 'scoped_instance_history_and_traces',
    'healthy_after_local_failure', 'native_handles_closed', 'immutable_environment',
    'coordinator_governor_outside_group', 'one_aggregate_os_commit_cap')


def validate_shared_qualification(binding, *, commit_limit_mib, host_reserve_mib):
    """Replay authored native/group proof, not positive booleans or tiny RAM claims.

    This gate supplements (never substitutes for) the real all-three native
    qualification and independent exact-byte campaign review. Configured worker
    capacity is intentionally NOT inferred from the authored observed overlap.
    """
    from gluomem_bench.bench.lme_first import verify_journal
    import gluomem_bench.bench.lme_first as coordinator
    import gluomem_bench.bench.rack_trace as rack_boundary
    packet = json.loads(read_bound(**_binding_args(binding)))
    if (packet.get('protocol') != 'lme-first-shared-gluomem@v1'
        or packet.get('executor',{}).get('sha256') != bound(__file__)['sha256']
        or packet.get('coordinator',{}).get('sha256') != bound(coordinator.__file__)['sha256']
        or packet.get('rack_boundary',{}).get('sha256') != bound(rack_boundary.__file__)['sha256']
        or packet.get('group_commit_limit_mib') != commit_limit_mib
        or packet.get('host_reserve_mib') != host_reserve_mib or host_reserve_mib < 8192
        or any(packet.get('assertions',{}).get(k) is not True for k in SHARED_ASSERTIONS)):
        raise ValueError('final-byte shared GluoMem native isolation qualification required')
    read_bound(**_binding_args(packet['tests']))
    for key in ('executor','coordinator','rack_boundary'):
        read_bound(**_binding_args(packet[key]))
    plan = json.loads(read_bound(**_binding_args(packet['plan'])))
    from gluomem_bench.bench.comprehensive_driver import frozen_inventory
    closure = packet.get('source_closure',[])
    if len({item['path'] for item in closure}) != len(closure):
        raise ValueError('duplicate shared native source closure identity')
    for item in closure:
        read_bound(**_binding_args(item))
    pinned_bytes = {item['sha256'] for item in closure}
    # Exact bytes permit an independently frozen private copy; paths themselves
    # remain authenticated in the proof and the campaign's own full inventory.
    if any(item['sha256'] not in pinned_bytes for item in
           frozen_inventory(plan['dataset']['path'],plan['judge_source'])):
        raise ValueError('shared native/core/package/non-Python source closure differs')
    inventory = {str(Path(item['path']).resolve()):item['sha256'] for item in packet['evidence_files']}
    if len(inventory) != len(packet['evidence_files']):
        raise ValueError('duplicate shared evidence identity')
    for item in packet['evidence_files']:
        read_bound(**_binding_args(item))
    for item in [packet['plan'],*packet['inputs'].values(),*packet['resource_evidence'].values(),
                 *packet['worker_outcomes'].values()]:
        if inventory.get(str(Path(item['path']).resolve())) != item['sha256']:
            raise ValueError('shared native evidence inventory missing')
    verify_resource_measurement(packet['resource_evidence'],commit_limit_mib=commit_limit_mib,
                                measured=packet['memory_samples'],host_reserve_mib=host_reserve_mib)
    outcome = json.loads(read_bound(**_binding_args(packet['resource_evidence']['outcome'])))
    if (outcome.get('resource',{}).get('scope') != 'shared-gluomem-group-not-per-history'
        or outcome.get('network',{}).get('denied_connections') != 0):
        raise ValueError('shared group native resource/network proof differs')
    group_pid = outcome['resource']['pid']
    samples = [json.loads(line) for line in read_bound(**_binding_args(packet['resource_evidence']['samples'])).splitlines()]
    root_members = [member for sample in samples for member in sample['members']
                    if member['pid']==sample['root_pid']]
    # A closing Windows launcher can be absent from Toolhelp while its retained
    # process handle still supplies exact creation time/private bytes. The sampler
    # preserves parent=0 as unavailable, never invents ancestry. Require positive
    # observed parent identity and reject every contradictory available sample.
    if (group_pid == packet.get('coordinator_pid')
        or not any(member['parent_pid']==packet.get('coordinator_pid') for member in root_members)
        or any(member['parent_pid'] not in (0,packet.get('coordinator_pid')) for member in root_members)):
        raise ValueError('shared group must be outside coordinator/governor process')
    records = packet['records']
    report = verify_journal(packet['coordination'],records)
    if (len(records) < 5 or any(r['system'] != 'gluomem' or r.get('memory_group') != 'gluomem'
                              for r in records)
        or report['peak_active_history_jobs'] < 3 or report['limiter']['peak_in_flight'] < 3
        or [r['status'] for r in report['jobs'][:5]] != ['complete']*3+['failed','complete']):
        raise ValueError('three parallel native histories and healthy local-failure successor required')
    cfg,profile = effective_config(plan)
    verify_input_cache(Path(packet['coordination']).parent,plan,packet['inputs'])
    for record,row in zip(records,report['jobs'],strict=True):
        leaf_outcome = json.loads(read_bound(**_binding_args(packet['worker_outcomes'][record['job_id']])))
        if (leaf_outcome.get('calls') != row['calls']
            or leaf_outcome.get('resource',{}).get('pid') != group_pid
            or leaf_outcome.get('resource',{}).get('scope') != 'shared-gluomem-group-not-per-history'):
            raise ValueError('shared native worker identity/call accounting differs')
        if row['status'] != 'complete':
            continue
        adapter,sample = restore_input(packet['inputs'][record['sample_id']],plan)
        job = SimpleNamespace(**record)
        job.output = Path(job.output)
        verify_native_job(job=job,context=SimpleNamespace(calls=row['calls']),cfg=cfg,
            adapter=adapter,sample=sample,profile=profile,judge_source=Path(plan['judge_source']))
        for path in job.output.rglob('*'):
            if path.is_file() and inventory.get(str(path.resolve())) != bound(path)['sha256']:
                raise ValueError('complete shared native leaf artifact missing/different')
    return packet


def validate_resource_contract(candidate, qualification, *, local_memory_mib):
    """Tiny failed native observations cannot authorize a v2 physical policy."""
    import math
    if (candidate.get('resource_protocol') != RESOURCE_PROTOCOL
        or qualification.get('resource_protocol') != RESOURCE_PROTOCOL
        or candidate.get('native_mode') not in ('process','gluomem-shared-process')):
        raise ValueError('qualified physical resource protocol required; shared threads are not isolated')
    limits = candidate.get('history_commit_limits_mib',{})
    shared = candidate['native_mode'] == 'gluomem-shared-process'
    expected = {'mem0','cognee','no_memory'} if shared else {'gluomem','mem0','cognee','no_memory'}
    if set(limits) != expected:
        raise ValueError('all explicit per-history commit ceilings required')
    process_limits = dict(limits)
    if shared:
        # This bound belongs to the separate one-history real qualification,
        # never to histories resident in the shared GluoMem interpreter.
        process_limits['gluomem'] = candidate.get('gluomem_qualification_process_commit_limit_mib')
    reserve = candidate.get('host_reserve_mib')
    if type(reserve) is not int or not 8192 <= reserve < local_memory_mib:
        raise ValueError('explicit >=8192 MiB physical/commit host reserve must leave local capacity')
    for system,cap in process_limits.items():
        measured = qualification.get('memory_samples',{}).get(system if system!='no_memory' else 'gluomem',{})
        if (any(type(measured.get(k)) not in (int,float) or not math.isfinite(measured[k])
                or measured[k] <= 0 for k in ('rss_mib','private_mib'))
            or type(cap) is not int or not 1 <= cap <= local_memory_mib-reserve
            or cap < max(measured[k] for k in ('rss_mib','private_mib'))*1.5):
            raise ValueError('explicit bounded worker envelope must cover measured startup/resident high-water')
    if candidate['native_mode'] == 'gluomem-shared-process':
        cap = candidate.get('gluomem_group_commit_limit_mib')
        if (type(cap) is not int or not 1 <= cap <= local_memory_mib-reserve
            or not candidate.get('shared_gluomem_qualification')):
            raise ValueError('explicit qualified shared group envelope required')
        packet = validate_shared_qualification(candidate['shared_gluomem_qualification'],
            commit_limit_mib=cap,host_reserve_mib=reserve)
        if cap < max(packet['memory_samples'].values())*1.5:
            raise ValueError('shared group ceiling must cover sampled startup/resident high-water')
    return limits


def validate_candidate(candidate_path, approval, *, post_exit=False):
    """Parent operational approval + independent exact-byte review are distinct.

    Unknown historical spend never debits or blocks this separately approved
    prospective phase. Neither this producer nor a qualification claims approval.
    """
    material = Path(candidate_path).read_bytes()
    digest = hashlib.sha256(material).hexdigest()
    if (approval.get('approver_kind') != 'parent' or approval.get('approved') is not True
        or approval.get('operational_full_benchmark_execution_authorized') is not True
        or approval.get('candidate_sha256') != digest):
        raise ValueError('separate exact-byte parent operational approval required')
    candidate = json.loads(material)
    if candidate.get('protocol') != EXECUTION_PROTOCOL:
        raise ValueError('native execution protocol required')
    review = json.loads(read_bound(**_binding_args(approval['independent_review'])))
    if (review.get('approved') is not True or review.get('blockers') != []
        or review.get('candidate_sha256') != digest
        or review.get('reviewer_kind') not in ('human','AI-assisted')
        or not review.get('reviewer_id') or review['reviewer_id'] == candidate['producer_id']):
        raise ValueError('independent exact-byte review required')
    from gluomem_bench.bench.lme_first import validate_plan
    plan = json.loads(read_bound(**_binding_args(candidate['plan'])))
    validate_plan(plan)
    decision = json.loads(read_bound(**_binding_args(candidate['budget_decision'])))
    if (decision.get('approver_kind') != 'parent' or decision.get('prospective_budget_approved') is not True
        or decision['plan']['sha256'] != candidate['plan']['sha256']
        or Path(decision['plan']['path']).resolve() != Path(candidate['plan']['path']).resolve()
        or plan['future_budget_proposal']['max_calls'] != decision['future_generation_call_ceiling']
        or plan['future_budget_proposal']['max_seconds'] != decision['future_wall_seconds_ceiling']
        or plan['profile']['max_generation_calls'] != 16384
        or plan['profile']['max_run_seconds'] != 21600
        or plan['operating_point']['max_in_flight'] != 300
        or plan['operating_point']['max_in_flight_tokens'] != 39321600
        or plan['evidence_token_budget'] != 4096):
        raise ValueError('prospective budget/profile/global governor binding differs')
    frozen = {}
    for item in candidate['files']:
        path = str(Path(item['path']).resolve())
        if path in frozen:
            raise ValueError('duplicate inventory identity')
        check_file(path,item['sha256'])
        check_file(item['snapshot'],item['sha256'])
        frozen[path] = item['sha256']
    # Require actual imported source roots, not merely similarly named mirror paths.
    import gluomem
    roots = [Path(__file__).resolve().parents[1],Path(gluomem.__file__).resolve().parent]
    for root in roots:
        for path in root.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo'):
                if str(path.resolve()) not in frozen:
                    raise ValueError('actual source/non-Python runtime closure missing')
    for path in [Path(plan['dataset']['path']),Path(plan['judge_source']),Path(sys.executable),
                 Path(candidate['cognee_python'])]:
        if str(path.resolve()) not in frozen:
            raise ValueError('dataset/scorer/interpreter inventory missing')
    qualification = json.loads(read_bound(**_binding_args(candidate['qualification'])))
    if (qualification.get('qualified') is not True
        or qualification.get('candidate_code_sha256') != frozen[str(Path(__file__).resolve())]
        or qualification.get('systems') != ['gluomem','mem0','cognee']
        or not qualification.get('evidence_files')
        or any(qualification.get('assertions',{}).get(k) is not True for k in QUALIFICATION_ASSERTIONS)):
        raise ValueError('final-byte actual all-native qualification incomplete')
    for item in qualification['evidence_files']:
        read_bound(**_binding_args(item))
    qualified_sources = {str(Path(r['path']).resolve()):r['sha256']
                         for r in qualification.get('executor_sources',[])}
    for path in Path(__file__).resolve().parent.glob('*.py'):
        key = str(path.resolve())
        if qualified_sources.get(key) != frozen.get(key):
            raise ValueError('qualification and final native producer source bytes differ')
    validate_qualification_closure(candidate['files'],qualification)
    if review.get('qualification_sha256') != candidate['qualification']['sha256']:
        raise ValueError('review must bind actual qualification evidence')
    limits = candidate['native_limits']
    if (set(limits) != {'gluomem','mem0','cognee','no_memory'}
        or any(type(v) is not int or not 1 <= v <= 500 for v in limits.values())
        or limits['cognee'] > 2 or not 1 <= candidate['max_workers'] <= 500):
        raise ValueError('bounded native isolation limits required')
    for system in limits:
        measured = qualification['memory_samples'][system if system != 'no_memory' else 'gluomem']
        if any(type(measured[k]) not in (float,int) or measured[k] <= 0 for k in ('rss_mib','private_mib')):
            raise ValueError('measured native whole-tree memory required')
    validate_resource_contract(candidate,qualification,local_memory_mib=plan['native_constraints']['local_memory_mib'])
    qualification_limits = dict(candidate['history_commit_limits_mib'])
    if candidate['native_mode'] == 'gluomem-shared-process':
        qualification_limits['gluomem'] = candidate['gluomem_qualification_process_commit_limit_mib']
    verify_qualification_resources(qualification,qualification_limits)
    root = Path(candidate['output_root'])
    project = Path(__file__).resolve().parents[2]
    if (not root.is_absolute() or root.is_symlink()
        or root.resolve().is_relative_to(project.parent)
        or (not post_exit and root.exists())):
        raise ValueError('fresh external execution root required')
    from gluomem_bench.bench.upstream_scoring import load_longmemeval_prompt
    load_longmemeval_prompt(Path(plan['judge_source']))  # Actual byte-pinned loader, before clients.
    return candidate,plan,qualification


def _binding_args(item):
    return {'path':item['path'],'digest':item['sha256']}


def native_records(candidate,plan,qualification):
    import math
    root = Path(candidate['output_root'])
    rows = []
    for record in plan['jobs']:
        sample = qualification['memory_samples'][record['system'] if record['system'] != 'no_memory' else 'gluomem']
        rows.append(dict(record,output=str(root/'jobs'/record['job_id']),
                         memory_mib=max(candidate.get('history_commit_limits_mib',{}).get(record['system'],0),
                             math.ceil(max(sample['rss_mib'],sample['private_mib'])*1.5))))
        if candidate.get('native_mode') == 'gluomem-shared-process' and record['system'] == 'gluomem':
            rows[-1].update(memory_group='gluomem',memory_mib=candidate['gluomem_group_commit_limit_mib'])
    return rows


def build_input_cache(root,plan,budget):
    """Construct exact projection ONCE; lightweight workers open only their row."""
    from dataclasses import asdict
    from gluomem_bench.bench.lme_projection import ExactLongMemEvalAdapter
    cfg,_ = effective_config(plan)
    adapter = ExactLongMemEvalAdapter(cfg,plan['dataset']['name'])
    adapter.verify()
    samples = {s.id:s for s in adapter.load()}
    conversations = {c.conv_id:c for c in adapter.conversations()}
    if list(samples) != plan['sample_ids']:
        raise ValueError('released500 ordered identities differ')
    target = Path(root)/'input-cache'
    target.mkdir()
    bindings = {}
    for index,sid in enumerate(plan['sample_ids']):
        budget.check()
        sample = samples[sid]
        data = {'sample':asdict(sample),'conversation':asdict(conversations[sample.conv_id])}
        # Native datetime objects serialize once without changing source text.
        for s in data['conversation']['sessions']:
            s['date_time'] = s['date_time'].isoformat()
        path = target/f'{index:04d}.json'
        path.write_bytes(json_bytes(data))
        bindings[sid] = bound(path)
    (Path(root)/'input-cache.json').write_bytes(json_bytes(bindings))
    return bindings


def verify_input_cache(root,plan,inputs,*,budget=None):
    """Reconstruct from source; self-rehashed cached question/gold/history is not truth."""
    from dataclasses import asdict
    from gluomem_bench.bench.lme_projection import ExactLongMemEvalAdapter
    cfg,_ = effective_config(plan)
    adapter = ExactLongMemEvalAdapter(cfg,plan['dataset']['name'])
    adapter.verify()
    samples = {s.id:s for s in adapter.load()}
    conversations = {c.conv_id:c for c in adapter.conversations()}
    if list(samples) != plan['sample_ids'] or set(inputs) != set(samples):
        raise ValueError('source/cache identity coverage differs')
    for index,sid in enumerate(plan['sample_ids']):
        if budget is not None:
            budget.check()
        expected_path = Path(root)/'input-cache'/f'{index:04d}.json'
        if Path(inputs[sid]['path']).resolve() != expected_path.resolve():
            raise ValueError('input cache identity/path differs')
        sample = samples[sid]
        data = {'sample':asdict(sample),'conversation':asdict(conversations[sample.conv_id])}
        for session in data['conversation']['sessions']:
            session['date_time'] = session['date_time'].isoformat()
        if read_bound(**_binding_args(inputs[sid])) != json_bytes(data):
            raise ValueError('exact source projection/clock/gold differs')


def verify_root(candidate, *, plan, candidate_path=None, budget=None, qualification=None):
    """Full root verifier, replays journal AND every claimed-complete native leaf.

    Missing/failed/unrun rows cannot be manufactured into scores or completion.
    The same frozen source projection/scorer/profile authenticates leaf checks.
    """
    from gluomem_bench.bench.lme_first import verify_journal
    root = Path(candidate['output_root'])
    if (root/'incomplete.json').exists():
        raise ValueError('incomplete revocation overrides root authority')
    if qualification is None:
        qualification = json.loads(read_bound(**_binding_args(candidate['qualification'])))
    records = native_records(candidate,plan,qualification)
    coordination = verify_journal(root/'coordination',records)
    inputs = json.loads((root/'input-cache.json').read_bytes())
    report = json.loads((Path(candidate_path) if candidate_path else root/'root-status.json').read_bytes())
    if candidate_path is None:
        closure = json.loads((root/'completion-receipt.json').read_bytes())
        if (closure.get('root_sha256') != bound(root/'root-status.json')['sha256']
            or closure.get('calls') != coordination['accounting']['calls']
            or not coordination['accounting']['elapsed_seconds'] <= closure.get('elapsed_seconds',-1)
                   < plan['future_budget_proposal']['max_seconds']):
            raise ValueError('final publication/outer clock closure differs')
    if (report.get('protocol') != EXECUTION_PROTOCOL or report.get('jobs') != coordination['jobs']
        or report.get('accounting') != coordination['accounting']
        or report.get('quality_complete') != all(r['status']=='complete' for r in coordination['jobs'])
        or report.get('configured_remote_request_slots') != 300
        or report.get('observed_leased_request_peak') != coordination['limiter']['peak_in_flight']
        or report.get('peak_active_history_jobs') != coordination['peak_active_history_jobs']
        or report.get('planned') != len(plan['jobs']) or len(records) != len(plan['jobs']) or set(inputs) != set(plan['sample_ids'])):
        raise ValueError('root coverage/completion/accounting differs')
    if candidate.get('native_mode') == 'gluomem-shared-process':
        resource = report.get('shared_gluomem_resources',{})
        expected = {key:bound(root/'workers'/'gluomem-group'/filename) for key,filename in
            (('samples','resource-samples.jsonl'),('outcome','outcome.json'),('exit','process-exit.json'))}
        if (resource.get('evidence') != expected
            or resource.get('commit_limit_mib') != candidate['gluomem_group_commit_limit_mib']
            or report.get('native_mode') != candidate['native_mode']
            or report.get('configured_gluomem_history_capacity') != candidate['native_limits']['gluomem']
            or report.get('native_300_parallelism_certified') is not False):
            raise ValueError('root shared group resource scope/capacity differs')
        verify_resource_measurement(resource['evidence'],commit_limit_mib=resource['commit_limit_mib'],
                                    measured=resource['memory_samples'],host_reserve_mib=candidate['host_reserve_mib'])
    verify_input_cache(root,plan,inputs,budget=budget)
    cfg,profile = effective_config(plan)
    for record,row in zip(records,coordination['jobs'],strict=True):
        if budget is not None:
            budget.check()
        if row['status'] != 'complete':
            continue
        adapter,sample = restore_input(inputs[record['sample_id']],plan)
        job = SimpleNamespace(**record)
        job.output = Path(job.output)
        verify_native_job(job=job,context=SimpleNamespace(calls=row['calls']),cfg=cfg,adapter=adapter,
                          sample=sample,profile=profile,judge_source=Path(plan['judge_source']))
    if report.get('systems') != _system_summaries(plan,coordination,root,budget=budget):
        raise ValueError('root native score aggregation differs')
    if budget is not None:
        budget.check()
    return report


def _system_summaries(plan,coordination,root,*,budget=None):
    """Fold native system totals only; callers authenticate evidence separately."""
    summaries = {s:{'planned':sum(r['system']==s for r in plan['jobs']),'complete':0,'failed':0,'unrun':0,'score_sum':0,'scores':0}
                 for s in ('gluomem','mem0','cognee','no_memory')}
    for record,row in zip(plan['jobs'],coordination['jobs'],strict=True):
        if budget is not None:
            budget.check()
        summary = summaries[record['system']]
        summary[row['status']] += 1
        if row['status']=='complete':
            reader = root/'jobs'/record['job_id']
            if record['system']=='cognee':
                reader /= 'reader'
            predictions = json.loads((reader/'predictions.json').read_bytes())
            summary['scores'] += len(predictions)
            summary['score_sum'] += sum(r['score'] for r in predictions)
    return summaries


def _root_report(candidate,plan,coordination):
    summaries = _system_summaries(plan,coordination,Path(candidate['output_root']))
    return {'protocol':EXECUTION_PROTOCOL,'quality_complete':all(r['status']=='complete' for r in coordination['jobs']),
            'planned':len(plan['jobs']),'jobs':coordination['jobs'],'accounting':coordination['accounting'],
            'systems':summaries,'historical_quality_pooled':False,
            'completion_authority':'root-status.json + hash-bound completion-receipt.json; incomplete.json revokes both',
            'mem0_observation_date':'native execution-day observation date; declared method difference',
            'native_300_parallelism_certified':False,
            'native_mode':candidate.get('native_mode','process'),
            'configured_gluomem_history_capacity':candidate.get('native_limits',{}).get('gluomem'),
            'configured_remote_request_slots':300,
            'observed_leased_request_peak':coordination['limiter']['peak_in_flight'],
            'peak_active_history_jobs':coordination['peak_active_history_jobs']}


def freeze_candidate(*, plan_path, plan_sha256, budget_decision_path, budget_decision_sha256,
                     qualification_path, qualification_sha256, output_root, out,
                     cognee_python, host_reserve_mib, max_workers=500,
                     native_limits=None, producer_id='lme-native-executor', history_commit_limit_mib=None,
                     native_mode='process', gluomem_group_commit_limit_mib=None,
                     shared_gluomem_qualification_path=None,shared_gluomem_qualification_sha256=None):
    """Freeze recoverable source/runtime bytes; never grant execution authority.

    A qualification must bind final executor bytes. Candidate review and parent
    execution approval happen AFTER this function; neither is self-generated.
    """
    if type(history_commit_limit_mib) is not int or history_commit_limit_mib < 1:
        raise ValueError('explicit operator per-history commit ceiling required; tiny samples are not bounds')
    shared_binding = None
    shared_packet = None
    if native_mode == 'gluomem-shared-process':
        if not shared_gluomem_qualification_path or not shared_gluomem_qualification_sha256:
            raise ValueError('shared mode requires final-byte actual native group qualification')
        shared_binding = {'path':str(Path(shared_gluomem_qualification_path).resolve()),
                          'sha256':shared_gluomem_qualification_sha256}
        shared_packet = validate_shared_qualification(shared_binding,
            commit_limit_mib=gluomem_group_commit_limit_mib,host_reserve_mib=host_reserve_mib)
    elif native_mode != 'process' or any(v is not None for v in (gluomem_group_commit_limit_mib,
            shared_gluomem_qualification_path,shared_gluomem_qualification_sha256)):
        raise ValueError('explicit supported native isolation mode required')
    import shutil
    import subprocess
    import gluomem
    from gluomem_bench.bench.comprehensive_driver import frozen_inventory
    from gluomem_bench.bench.lme_first import validate_plan
    plan = json.loads(read_bound(plan_path,plan_sha256))
    validate_plan(plan)
    qualification = json.loads(read_bound(qualification_path,qualification_sha256))
    if (qualification.get('qualified') is not True
        or qualification.get('candidate_code_sha256') != bound(__file__)['sha256']):
        raise ValueError('final-byte native qualification required before freeze')
    verify_qualification_resources(qualification,{s:history_commit_limit_mib for s in ('gluomem','mem0','cognee')})
    decision = json.loads(read_bound(budget_decision_path,budget_decision_sha256))
    if decision.get('prospective_budget_approved') is not True:
        raise ValueError('parent prospective budget decision required')
    out = Path(out).resolve()
    if out.exists() or out.with_suffix('.objects').exists():
        raise FileExistsError('fresh candidate and recoverable object directory required')
    files = {r['path']:r['sha256'] for r in frozen_inventory(plan['dataset']['path'],plan['judge_source'])}
    for root in (Path(__file__).resolve().parents[1],Path(gluomem.__file__).resolve().parent):
        for p in root.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo'):
                files[str(p.resolve())] = bound(p)['sha256']
    # Include actual installed Cognee package data, not only its Python adapter.
    code = "import importlib.util,json;from pathlib import Path;p=Path(importlib.util.find_spec('cognee').origin).parent;print(json.dumps([str(x) for x in p.rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix not in ('.pyc','.pyo')]))"
    process = subprocess.run([str(cognee_python),'-I','-c',code],capture_output=True,timeout=30)
    if process.returncode != 0:
        raise ValueError('native Cognee installed source inventory unavailable')
    for p in json.loads(process.stdout):
        files[str(Path(p).resolve())] = bound(p)['sha256']
    for p in (cognee_python,plan_path,budget_decision_path,qualification_path):
        files[str(Path(p).resolve())] = bound(p)['sha256']
    for item in qualification.get('evidence_files',[]):
        read_bound(**_binding_args(item))
        files[str(Path(item['path']).resolve())] = item['sha256']
    for proof in qualification['resource_evidence'].values():
        for item in proof.values():
            read_bound(**_binding_args(item))
            files[str(Path(item['path']).resolve())] = item['sha256']
    if shared_packet is not None:
        for item in [shared_binding,*shared_packet['evidence_files']]:
            read_bound(**_binding_args(item))
            files[str(Path(item['path']).resolve())] = item['sha256']
    validate_qualification_closure([{'path':p,'sha256':h} for p,h in files.items()],qualification)
    objects = out.with_suffix('.objects')
    objects.mkdir(parents=True,exist_ok=False)
    inventory = []
    dataset_path = str(Path(plan['dataset']['path']).resolve())
    for path,digest in sorted(files.items()):
        target = objects/digest
        if not target.exists():
            shutil.copyfile(path,target)
        check_file(target,digest)
        inventory.append({'path':path,'sha256':digest,'snapshot':str(target),
                          'worker_check':path != dataset_path})
    candidate = {'protocol':EXECUTION_PROTOCOL,'producer_id':producer_id,
        'plan':{'path':str(Path(plan_path).resolve()),'sha256':plan_sha256},
        'budget_decision':{'path':str(Path(budget_decision_path).resolve()),'sha256':budget_decision_sha256},
        'qualification':{'path':str(Path(qualification_path).resolve()),'sha256':qualification_sha256},
        'output_root':str(Path(output_root).resolve()),'files':inventory,
        'cognee_python':str(Path(cognee_python).resolve()),'max_workers':max_workers,
        'native_limits':native_limits or {'gluomem':500,'mem0':4,'cognee':2,'no_memory':500},
        'host_reserve_mib':host_reserve_mib,'execution_authorized':False,
        'resource_protocol':RESOURCE_PROTOCOL,'native_mode':native_mode,
        'history_commit_limits_mib':{s:history_commit_limit_mib for s in ('gluomem','mem0','cognee','no_memory')},
        'native_300_parallelism_certified':False}
    if shared_binding is not None:
        candidate.update(shared_gluomem_qualification=shared_binding,
                         gluomem_group_commit_limit_mib=gluomem_group_commit_limit_mib,
                         gluomem_qualification_process_commit_limit_mib=history_commit_limit_mib)
        del candidate['history_commit_limits_mib']['gluomem']
    validate_resource_contract(candidate,qualification,local_memory_mib=plan['native_constraints']['local_memory_mib'])
    with out.open('xb') as stream:
        stream.write(json_bytes(candidate))
    return bound(out)


def run_authored_qualification(*, plan_path, plan_sha256, budget_decision_path,
        budget_decision_sha256, source_path, out, cognee_python, host_reserve_mib,
        history_commit_limit_mib=None, post_recovery_permit_path=None,
        post_recovery_permit_sha256=None, budget=None):
    """Qualification-owner ONLY: one fresh authored all-three pass, 192/1200.

    This is an executable final-path entrypoint, not a qualification certificate.
    It uses the unchanged approved role/model profile, common native projection,
    scorer and reader. Three 64-call/360s envelopes run serially for measured
    bootstrap safety; this does not certify native300 or full-history RAM.
    The caller must retain the output even when failed; no internal retry exists.
    """
    if type(history_commit_limit_mib) is not int or history_commit_limit_mib < 1:
        raise ValueError('explicit bounded qualification worker commit ceiling required')
    if budget is not None:
        budget.check()
        if budget.max_calls != 192 or budget.max_seconds != 1200:
            raise ValueError('qualification budget must retain declared bounded caps')
    import copy
    from gluomem_bench.bench.lme_first import run_isolated_jobs
    from gluomem_bench.bench.parallel_campaign import CampaignBudget
    plan = json.loads(read_bound(plan_path,plan_sha256))
    decision = json.loads(read_bound(budget_decision_path,budget_decision_sha256))
    permit = decision.get('bounded_authored_real_backend_qualification_permitted',{})
    if (decision.get('approver_kind')!='parent' or decision.get('plan',{}).get('sha256')!=plan_sha256
        or permit.get('generation_admission_cap')!=192 or permit.get('wall_seconds_cap')!=1200
        or permit.get('retries')!=0):
        raise ValueError('exact parent authored qualification permit required')
    recovery_binding = None
    if post_recovery_permit_path is not None or post_recovery_permit_sha256 is not None:
        recovery = json.loads(read_bound(post_recovery_permit_path,post_recovery_permit_sha256))
        if (recovery.get('protocol') != 'lme-first-parent-post-recovery-qualification-permit@v1'
            or recovery.get('approver_kind') != 'parent' or recovery.get('approved') is not True
            or Path(recovery.get('base_budget_decision','')).resolve() != Path(budget_decision_path).resolve()
            or recovery.get('new_generation_admission_cap') != 192
            or recovery.get('new_wall_seconds_cap') != 1200
            or recovery.get('new_attempts_per_backend') != 1 or recovery.get('generation_retries') != 0
            or recovery.get('encoder_repeat_dispatch_latch_required') is not True
            or recovery.get('history_commit_limit_mib') != history_commit_limit_mib
            or recovery.get('host_physical_and_commit_reserve_mib') != host_reserve_mib
            or host_reserve_mib != 8192 or recovery.get('bootstrap_resident_histories') != 1
            or recovery.get('benchmark_released_qa_authorized') is not False
            or recovery.get('full_campaign_execution_authorized') is not False):
            raise ValueError('exact bounded post-recovery parent permit required')
        recovery_binding = {'path':str(Path(post_recovery_permit_path).resolve()),
                            'sha256':post_recovery_permit_sha256}
    source_path = Path(source_path).resolve()
    raw = json.loads(source_path.read_bytes())
    if (len(raw)!=1 or raw[0].get('question_id') not in
            ('authored-lme-native-v1','authored-clock-whitespace-empty-01')
        or raw[0].get('question_date') is None):
        raise ValueError('exact single authored qualification identity/clock required')
    authored_id = raw[0]['question_id']
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('one fresh qualification destination required')
    # SAME parent clock across source/cache/native/reader/scorer/publication.
    budget = CampaignBudget(max_calls=192,max_seconds=1200) if budget is None else budget
    budget.check()
    out.mkdir(parents=True,exist_ok=False)
    qplan = copy.deepcopy(plan)
    qplan['dataset'] = dict(plan['dataset'],path=str(source_path),sha256=bound(source_path)['sha256'])
    qplan['sample_ids'] = [authored_id]
    qplan['config_overrides'].update({'benchmark.datasets_dir':str(source_path.parent),
        'benchmark.datasets.longmemeval_s':{'url':'https://authored.invalid/qualification.json',
                'file':source_path.name,'sha256':qplan['dataset']['sha256']}})
    qpath = out/'qualification-plan.json'
    qpath.write_bytes(json_bytes(qplan))
    cpath = out/'qualification-producer.json'
    package = Path(__file__).resolve().parent
    source_files = [bound(p) for p in sorted(package.glob('*.py'))]
    cpath.write_bytes(json_bytes({'plan':bound(qpath),'cognee_python':str(Path(cognee_python).resolve()),
                                'files':source_files}))
    inputs = build_input_cache(out,qplan,budget)
    bootstrap_capacity = history_commit_limit_mib
    host = host_memory_status()
    if min(host['available_physical_mib'],host['commit_headroom_mib']) < bootstrap_capacity+host_reserve_mib:
        raise DispatchStopped('host reserve leaves no authored bootstrap admission')
    rows = [dict(job_id='authored--'+system,history_id=system+':authored',system=system,
                 sample_id=authored_id,output=str(out/'jobs'/system),
                 memory_mib=bootstrap_capacity,call_budget=64,seconds_budget=360) for system in ('gluomem','mem0','cognee')]
    measurements = {}
    from gluomem_bench.bench.native_backend_qualification import EncoderProxy
    with EncoderProxy(qplan['embedding_url'],receipt=out/'encoder-proxy.jsonl',
            timeout=lambda:budget.deadline-budget.clock()) as encoder_proxy, ContextBroker() as broker:
        # A single shared failure latch covers native SDK retries across all
        # backends. No discovery/inference is performed by opening this proxy.
        qplan['embedding_url'] = encoder_proxy.base_url
        qpath.write_bytes(json_bytes(qplan))
        cpath.write_bytes(json_bytes({'plan':bound(qpath),'cognee_python':str(Path(cognee_python).resolve()),
                                    'files':source_files}))
        def run(job,context):
            system = job.job_id.removeprefix('authored--')
            def measure(sample):
                old = measurements.setdefault(system,{'rss_mib':0,'private_mib':0,'processes':0})
                for key in old:
                    old[key] = max(old[key],sample[key])
            return launch_worker({'candidate':bound(cpath),'input':inputs[authored_id],
                    'system':system,'commit_limit_mib':history_commit_limit_mib},
                    broker=broker,context=context,workdir=out/'workers'/system,
                    sample_memory=measure,host_reserve_mib=host_reserve_mib)
        report = run_isolated_jobs(rows,out=out/'coordination',run_job=run,max_calls=192,max_seconds=1200,
            max_workers=1,native_limits={'gluomem':1,'mem0':1,'cognee':1},memory_limit_mib=bootstrap_capacity,
            max_in_flight_tokens=39321600,budget=budget,
            admit=lambda record:min(host_memory_status()[k] for k in
                ('available_physical_mib','commit_headroom_mib'))>=host_reserve_mib+history_commit_limit_mib)
    resource_evidence = {}
    for system in ('gluomem','mem0','cognee'):
        paths = {'samples':out/'workers'/system/'resource-samples.jsonl',
                 'outcome':out/'workers'/system/'outcome.json','exit':out/'workers'/system/'process-exit.json'}
        if all(p.is_file() for p in paths.values()):
            resource_evidence[system] = {k:bound(p) for k,p in paths.items()}
    report.update(scope='authored actual native qualification ONLY; not released QA',
                  post_recovery_permit=recovery_binding,
                  encoder_repeat_dispatch_latch=True,
                  encoder_proxy_evidence=bound(out/'encoder-proxy.jsonl') if (out/'encoder-proxy.jsonl').exists() else None,
                  resource_evidence=resource_evidence,
                  memory_samples=measurements,executor_sources=source_files,
                  resource_protocol=RESOURCE_PROTOCOL,
                  host_reserve_mib=host_reserve_mib,
                  bootstrap_memory_policy='explicit OS commit envelope plus host physical/commit reserve; serial authored bootstrap only; not a full-history maximum',
                  generation_admissions=report['accounting']['calls'])
    budget.check()
    (out/'qualification-execution.json').write_bytes(json_bytes(report))
    budget.check()
    return report


def execute_candidate(candidate_path, approval_path):
    from gluomem_bench.bench.lme_first import run_isolated_jobs
    from gluomem_bench.bench.parallel_campaign import CampaignBudget
    # Approval parsing does not create state. The prospective clock starts before
    # inventory verification, dataset/scorer load, setup, clients and publication.
    approval = json.loads(Path(approval_path).read_bytes())
    raw = json.loads(Path(candidate_path).read_bytes())
    raw_plan = json.loads(read_bound(**_binding_args(raw['plan'])))
    caps = raw_plan['future_budget_proposal']
    budget = CampaignBudget(max_calls=caps['max_calls'],max_seconds=caps['max_seconds'])
    candidate,plan,qualification = validate_candidate(candidate_path,approval)
    budget.check()
    root = Path(candidate['output_root'])
    root.mkdir(parents=True,exist_ok=False)
    try:
        (root/'candidate.json').write_bytes(Path(candidate_path).read_bytes())
        (root/'execution-approval.json').write_bytes(Path(approval_path).read_bytes())
        inputs = build_input_cache(root,plan,budget)
        records = native_records(candidate,plan,qualification)
        bindings = {r['job_id']:r for r in records}
        memory = MemoryAdmission(host_reserve_mib=candidate['host_reserve_mib'],
                                 commit_limits=candidate['history_commit_limits_mib'])
        for system in candidate['native_limits']:
            sample = qualification['memory_samples'][system if system!='no_memory' else 'gluomem']
            memory.observe(system,**{k:sample[k] for k in ('rss_mib','private_mib')})
        from contextlib import ExitStack
        shared_mode = candidate['native_mode'] == 'gluomem-shared-process'
        group_identity = 'owned-gluomem-process-group'
        if shared_mode:
            packet = json.loads(read_bound(**_binding_args(candidate['shared_gluomem_qualification'])))
            memory.commit_limits['gluomem-group'] = candidate['gluomem_group_commit_limit_mib']
            memory.observe('gluomem-group',**packet['memory_samples'])
            if not memory.acquire(group_identity,'gluomem-group'):
                raise DispatchStopped('whole GluoMem group exceeds physical/commit host reserve')
        with ContextBroker() as broker, ExitStack() as lifecycle:
            group = None
            if shared_mode:
                from urllib.parse import urlsplit
                endpoints = [broker.address]
                for url in (plan['profile']['base_url'],plan['embedding_url']):
                    endpoint = urlsplit(url)
                    endpoints.append((endpoint.hostname,endpoint.port or (443 if endpoint.scheme=='https' else 80)))
                group = lifecycle.enter_context(SharedGluomemWorker(
                    workdir=root/'workers'/'gluomem-group',
                    commit_limit_mib=candidate['gluomem_group_commit_limit_mib'],
                    max_workers=candidate['native_limits']['gluomem'],
                    host_reserve_mib=candidate['host_reserve_mib'],network_endpoints=endpoints,
                    sample_memory=lambda sample:memory.update(group_identity,
                        rss_mib=sample['rss_mib'],private_mib=sample['private_mib']),
                    memory_failure=memory.latch_pressure))
            def admit(record):
                if group is not None and record['system']=='gluomem':
                    return not group.failed and group.process.poll() is None and memory.admit_existing(group_identity)
                return memory.acquire(record['job_id'],record['system'])
            def run(job,context):
                record = bindings[job.job_id]
                system = record['system']
                def sample_memory(sample):
                    memory.update(job.job_id,rss_mib=sample['rss_mib'],private_mib=sample['private_mib'])
                try:
                    request = {'candidate':bound(candidate_path),'input':inputs[record['sample_id']], 'system':system}
                    if group is not None and system=='gluomem':
                        group.run(request,broker=broker,context=context,workdir=root/'workers'/job.job_id)
                    else:
                        launch_worker(dict(request,commit_limit_mib=candidate['history_commit_limits_mib'][system]),
                                      broker=broker,context=context,workdir=root/'workers'/job.job_id,
                                      sample_memory=sample_memory,memory_failure=memory.latch_pressure,
                                      host_reserve_mib=candidate['host_reserve_mib'])
                    adapter,sample = restore_input(inputs[record['sample_id']],plan)
                    native_job = SimpleNamespace(**record)
                    native_job.output = Path(native_job.output)
                    cfg,profile = effective_config(plan)
                    return verify_native_job(job=native_job,context=context,cfg=cfg,adapter=adapter,
                           sample=sample,profile=profile,judge_source=Path(plan['judge_source']))
                finally:
                    memory.release(job.job_id)
            coordination = run_isolated_jobs(records,out=root/'coordination',run_job=run,
                max_calls=budget.max_calls,max_seconds=budget.max_seconds,max_workers=candidate['max_workers'],
                native_limits=candidate['native_limits'],memory_limit_mib=plan['native_constraints']['local_memory_mib'],
                max_in_flight_tokens=39321600,budget=budget,admit=admit,
                release_admission=lambda record: memory.release(record['job_id']))
        if shared_mode:
            memory.release(group_identity)
        budget.check()
        report = _root_report(candidate,plan,coordination)
        report['memory_admission'] = memory.snapshot()
        if shared_mode:
            evidence = {key:bound(root/'workers'/'gluomem-group'/filename) for key,filename in
                (('samples','resource-samples.jsonl'),('outcome','outcome.json'),('exit','process-exit.json'))}
            samples = [json.loads(line) for line in read_bound(**_binding_args(evidence['samples'])).splitlines()]
            report['shared_gluomem_resources'] = {'commit_limit_mib':candidate['gluomem_group_commit_limit_mib'],
                'evidence':evidence,'memory_samples':{k:max(row[k] for row in samples)
                    for k in ('rss_mib','private_mib')},'full_history_ingestion_peak_mib':None,
                'scope':'one whole GluoMem inherited process group; not per-thread/history limits'}
        staged = root/'root-status.pending.json'
        with staged.open('xb') as stream:
            stream.write(json_bytes(report))
            stream.flush()
            os.fsync(stream.fileno())
        verify_root(candidate,plan=plan,candidate_path=staged,budget=budget,qualification=qualification)
        material = staged.read_bytes()
        budget.check()
        os.replace(staged,root/'root-status.json')
        if (root/'root-status.json').read_bytes() != material:
            raise ValueError('root publication readback differs')
        budget.check()
        closure = dict(root_sha256=bound(root/'root-status.json')['sha256'],**budget.snapshot())
        with (root/'completion-receipt.json').open('xb') as stream:
            stream.write(json_bytes(closure))
            stream.flush()
            os.fsync(stream.fileno())
        if json.loads((root/'completion-receipt.json').read_bytes()) != closure:
            raise ValueError('completion receipt readback differs')
        budget.check()
        return report
    except BaseException:
        # A separate revocation wins even if a stale complete root cannot be removed.
        failure = {'quality_complete':False,'status':'incomplete','accounting':budget.snapshot(),
                   'failure':{'type':'native-execution-failed','message':'execution or finalization failed'}}
        (root/'incomplete.json').write_bytes(json_bytes(failure))
        for name in ('root-status.pending.json','root-status.json'):
            try:
                (root/name).unlink(missing_ok=True)
            except OSError:
                pass
        raise



if __name__ == '__main__':
    raise SystemExit(main())
