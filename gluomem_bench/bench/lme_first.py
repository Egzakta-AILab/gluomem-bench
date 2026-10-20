"""Prospective LongMemEval-S preparation; no inference/approval CLI.

The fair governor is compatible with JobContext. One instance must be supplied
for ALL extraction/consolidation, reader and judge transports, not per backend.
The token cap is an aggregate *reservation*, not measured KV memory or usage.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from threading import Condition

from gluomem_bench.bench.parallel_campaign import guard_stopped


class DispatchStopped(RuntimeError):
    """Queue cancellation/deadline; never a score."""


class FairRequestLimiter:
    """FIFO requests, bounded globally by count AND served-token reservations.

    Head-of-line blocking is intentional: small requests cannot starve a large
    admitted request. Dead/cancelled tickets are removed, including non-heads.
    No mutex is held over model work. The public protocol matches RequestLimiter.
    """

    def __init__(self, *, max_in_flight=300, max_in_flight_tokens):
        if type(max_in_flight) is not int or not 1 <= max_in_flight <= 300:
            raise ValueError("global request ceiling must be within 1..300")
        if type(max_in_flight_tokens) is not int or max_in_flight_tokens < 1:
            raise ValueError("positive aggregate token reservation required")
        self.max_in_flight = max_in_flight
        self.max_in_flight_tokens = max_in_flight_tokens
        self._condition = Condition()
        self._queue = deque()
        self._requests = self._tokens = self._peak = self._peak_tokens = 0

    @contextmanager
    def request(self, tokens, *, deadline=None, cancel=None, clock=time.monotonic):
        if type(tokens) is not int or not 1 <= tokens <= self.max_in_flight_tokens:
            raise ValueError("invalid aggregate token reservation")
        ticket = object()
        with self._condition:
            self._queue.append(ticket)
            try:
                while True:
                    remaining = None if deadline is None else deadline - clock()
                    if cancel is not None and cancel.is_set():
                        raise guard_stopped(DispatchStopped("request cancelled"), 'cancelled')
                    if remaining is not None and remaining <= 0:
                        raise guard_stopped(DispatchStopped("request deadline exhausted"), 'job-deadline')
                    if (
                        self._queue[0] is ticket
                        and self._requests < self.max_in_flight
                        and self._tokens + tokens <= self.max_in_flight_tokens
                    ):
                        self._queue.popleft()
                        self._requests += 1
                        self._tokens += tokens
                        self._peak = max(self._peak, self._requests)
                        self._peak_tokens = max(self._peak_tokens, self._tokens)
                        self._condition.notify_all()
                        break
                    self._condition.wait(0.02 if remaining is None else min(0.02, remaining))
            except BaseException:
                self._queue.remove(ticket)
                self._condition.notify_all()
                raise
        try:
            yield
        finally:
            with self._condition:
                self._requests -= 1
                self._tokens -= tokens
                self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            return {
                "in_flight": self._requests,
                "in_flight_tokens": self._tokens,
                "peak_in_flight": self._peak,
                "peak_in_flight_tokens": self._peak_tokens,
                "waiting": len(self._queue),
            }


PROTOCOL = "longmemeval-s-first-prospective@v1"
PRIORITY_SYSTEMS = ("gluomem", "mem0", "cognee")
OLD_SYSTEMS = ("gluomem", "mem0", "no_memory", "bm25", "dense", "hybrid", "full_context")
BLOCKERS = [
    "parent-future-budget-review",
    "independent-byte-bound-review",
    "native-concurrency-qualification",
    "full-native-projection-and-clock-qualification",
    "shared-served-identity-tokenizer-accounting-qualification",
    "native-executor-integration",
]


def json_bytes(value):
    return (
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def read_bound(path, digest):
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("literal lowercase SHA256 required")
    material = Path(path).read_bytes()
    if hashlib.sha256(material).hexdigest() != digest:
        raise ValueError("source binding differs")
    return material


def _assemble_plan(source, binding, future_calls, future_seconds, memory_mib,
                   worker_proposal=None):
    for value in (future_calls, future_seconds, memory_mib):
        if type(value) is not int or value < 1:
            raise ValueError("explicit positive prospective caps required")
    ids = source["sample_ids"]
    if (
        len(ids) != 500
        or len(set(ids)) != 500
        or any(not isinstance(i, str) or not i for i in ids)
        or source["dataset"]["name"] != "longmemeval_s"
    ):
        raise ValueError("all 500 published LME-S identities required")
    rows = source["jobs"]
    if (
        [(r["sample_id"], r["system"]) for r in rows]
        != [(id_, system) for id_ in ids for system in source["systems"]]
        or set(source["systems"]) != set(OLD_SYSTEMS)
        or len(source["systems"]) != 7
        or len({r["job_id"] for r in rows}) != 3500
    ):
        raise ValueError("lossless seven-method prior matrix required")
    cognee = source["cognee_unrun"]
    if [(r["sample_id"], r["system"]) for r in cognee] != [(i, "cognee") for i in ids]:
        raise ValueError("all500 prior Cognee unrun rows required")
    profile = copy.deepcopy(source["profile"])
    envelope = profile["max_generation_calls"]
    seconds = profile["max_run_seconds"]
    if any(type(v) is not int or v < 1 for v in (envelope, seconds, profile["context_window"])):
        raise ValueError("pinned whole-history bounds required")
    point = {
        "max_in_flight": 300,
        "max_in_flight_tokens": 300 * profile["context_window"],
        "token_policy": "served prompt + unchanged output + schema reserve; one global FIFO",
        "history_workers": 30,
        "native_pipeline_parallelism": "existing admitted settings only",
        "stop_on_local_failure": False,
    }
    workers: dict[str, dict] = {
        "gluomem": {"workers": 24, "memory_reservation_mib": 256},
        "mem0": {"workers": 4, "memory_reservation_mib": 512},
        "cognee": {"workers": 2, "memory_reservation_mib": 1536},
    }
    if worker_proposal is not None:
        if (type(worker_proposal) is not dict
            or set(worker_proposal) != {"protocol", "gluomem_workers", "group_memory_mib"}
            or worker_proposal["protocol"] != "lme-first-shared-worker-proposal@v1"
            or type(worker_proposal["gluomem_workers"]) is not int
            or not 1 <= worker_proposal["gluomem_workers"] <= 300
            or type(worker_proposal["group_memory_mib"]) is not int
            or worker_proposal["group_memory_mib"] < 1):
            raise ValueError("explicit bounded shared GluoMem worker proposal required")
        workers["gluomem"] = {
            "workers": worker_proposal["gluomem_workers"],
            "memory_reservation_mib": worker_proposal["group_memory_mib"],
            "reservation_scope": "one-shared-process-group-not-per-thread",
        }
        point["history_workers"] = sum(v["workers"] for v in workers.values())
    reserved_memory = 1024 + sum(
        (1 if "reservation_scope" in v else v["workers"]) * v["memory_reservation_mib"]
        for v in workers.values()
    )
    if memory_mib < reserved_memory:
        raise ValueError("proposed native workers exceed explicit local memory reservations")
    pairs = [(i, system) for i in ids for system in PRIORITY_SYSTEMS]
    pairs += [(i, "no_memory") for i in ids]
    jobs = [
        {
            "job_id": f"lmes-new-{index:04d}--{system}",
            "sample_id": id_,
            "history_id": f"{system}:{id_}",
            "system": system,
            "phase": "primary" if system in PRIORITY_SYSTEMS else "control-after-primary",
            "call_budget": envelope,
            "seconds_budget": seconds,
            "operating_point": "shared-300-fifo@v1",
            "status": "unrun",
        }
        for index, (id_, system) in enumerate(pairs)
    ]
    plan = {
        "protocol": PROTOCOL,
        "execution_authorized": False,
        "new_experiment": True,
        "scope": "fresh full500 per primary system; old quality cohorts never replaced or pooled",
        "source_spec_binding": binding,
        "sample_ids": list(ids),
        **{
            key: copy.deepcopy(source[key])
            for key in (
                "dataset",
                "profile",
                "config_overrides",
                "judge_source",
                "embedding_url",
                "embedding_recipe",
                "source_identifier_policy",
                "evidence_token_budget",
            )
        },
        "operating_point": point,
        "jobs": jobs,
        "history_execution_contract": {
            "question_clock": (
                "exact native question_date in each framework reader; never wall clock"
            ),
            "session_commit_order": "chronological; stable original-index tie break",
            "ingestion": "await committed session before next; independent histories overlap",
            "within_session_parallelism": "unchanged already-admitted native pipeline only",
            "gold_and_source_labels": "scorer/audit only; opaque ordinal model-visible IDs",
            "qualification": "pending full native projection/commit/readback evidence",
        },
        "retained_old_matrix": copy.deepcopy(rows + cognee),
        "historical_accounting": {
            "observed_generation_calls": 8225,
            "observed_calls_basis": (
                "operator supplied prior ledger, not independently certified here"
            ),
            "unreceipted_calls": None,
            "pre_clock_setup_seconds": None,
            "cumulative_historical_bound_certified": False,
            "policy": "separate historical ledger; no retroactive reserve or reset",
        },
        "future_budget_proposal": {
            "max_calls": future_calls,
            "max_seconds": future_seconds,
            "primary_full_call_envelopes": 1500 * envelope,
            "control_full_call_envelopes": 500 * envelope,
            "full_call_envelopes": 2000 * envelope,
            "scope": "future phase only: clock before setup through verification/publication",
            "historical_debit_included": False,
            "approved": False,
            "may_leave_unrun": True,
            "call_budget_covers_all_full_envelopes": future_calls >= 2000 * envelope,
            "walltime_is_proposal_not_ETA": True,
        },
        "native_constraints": {
            "workers": workers,
            "local_memory_mib": memory_mib,
            "reserved_memory_mib": reserved_memory,
            "coordinator_memory_mib": 1024,
            "memory_values": "unqualified admission reservations, not measured RSS bounds",
            "serial_native_request_upper_bound": 30,
            "target_300_native_realization_certified": False,
            "cognee_process_ceiling": 2,
            "isolation_required": [
                "per-job stores, native homes, caches and immutable environment",
                "process-isolated Mem0 warning namespace or qualified scoped observer",
                "DSPy task/thread contexts and scoped logger sinks",
                "one coordinator-owned limiter; worker proxy must not create private limiters",
            ],
            "pipeline_note": (
                "Do not raise native extraction/consolidation parallelism without admission. "
                "30 serial native workers cannot realize300; qualify existing native pipeline "
                "overlap and measured local memory before proposing more workers."
            ),
        },
        "blockers": list(BLOCKERS),
    }
    if worker_proposal is not None:
        plan["worker_proposal"] = copy.deepcopy(worker_proposal)
        plan["native_constraints"].update(
            serial_native_request_upper_bound=min(300, point["history_workers"]),
            pipeline_note=("Configured shared-history ceilings are not observed native overlap; "
                           "the qualified aggregate process cap and OS admission remain required."),
        )
    return plan


def build_plan(
    *, source_spec, source_spec_sha256, future_max_calls, future_max_seconds, local_memory_mib,
    gluomem_workers=None, gluomem_group_memory_mib=4096
):
    """Small offline preparation: read pinned spec, not the ~1GB dataset/archive.

    This is an explicit new matched experiment proposal, NOT the old continuation
    API with its caps reset. Preserved source rows retain every old disposition.
    Projection/native readiness remains a separately byte-bound required gate.
    """
    binding = {"path": str(Path(source_spec).resolve()), "sha256": source_spec_sha256}
    source = json.loads(read_bound(binding["path"], binding["sha256"]))
    proposal = None if gluomem_workers is None else {
        "protocol": "lme-first-shared-worker-proposal@v1",
        "gluomem_workers": gluomem_workers,
        "group_memory_mib": gluomem_group_memory_mib,
    }
    return _assemble_plan(source, binding, future_max_calls, future_max_seconds,
                          local_memory_mib, proposal)


def validate_plan(plan):
    """Exact reconstruction; no dataset, backend, client or run root is opened.

    An external plan-byte hash is additionally required at CLI verification.
    Internal equality is not approval or an independent source trust anchor.
    """
    binding = plan["source_spec_binding"]
    source = json.loads(read_bound(binding["path"], binding["sha256"]))
    budget = plan["future_budget_proposal"]
    expected = _assemble_plan(
        source,
        binding,
        budget["max_calls"],
        budget["max_seconds"],
        plan["native_constraints"]["local_memory_mib"],
        plan.get("worker_proposal"),
    )
    if plan != expected:
        raise ValueError("prospective plan differs from exact full-scope reconstruction")
    return len(plan["jobs"])


def run_isolated_jobs(
    records,
    *,
    out,
    run_job,
    max_calls,
    max_seconds,
    max_workers,
    native_limits,
    memory_limit_mib,
    max_in_flight_tokens,
    budget=None,
    admit=None,
    release_admission=None,
):
    """Executable coordinator seam, not an authorized native campaign entrypoint.

    Callbacks own isolated native lifecycle, ordered ingestion and leaf validation.
    This function does not construct backends, alter environment/DSPy globals or
    spawn processes. Returning proves scheduler closure ONLY, never QA completion.
    The production paths' existing lease+begin_call protocol counts calls ONCE.
    Journal appends are O(1), process-flushed before dispatch; fsync is performed
    at initial/final checkpoints. Power-loss durability is NOT certified: an
    interrupted journal is incomplete and unreceipted spend remains unknown.
    """
    import os
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    from threading import Event, local

    from gluomem_bench.bench.embedding_runtime import safe_failure_diagnostic
    from gluomem_bench.bench.parallel_campaign import CampaignBudget, Job, JobContext

    records = copy.deepcopy(list(records))
    out = Path(out).resolve()
    if type(max_workers) is not int or not 1 <= max_workers <= 500:
        raise ValueError("bounded history workers required")
    if type(memory_limit_mib) is not int or memory_limit_mib < 1:
        raise ValueError("explicit native memory reservations required")
    if not native_limits or any(type(v) is not int or v < 1 for v in native_limits.values()):
        raise ValueError("explicit per-backend worker limits required")
    jobs = [
        Job(
            **{k: r[k] for k in ("job_id", "history_id", "output", "call_budget", "seconds_budget")}
        )
        for r in records
    ]
    paths = [j.output for j in jobs]
    grouped_caps = set()
    for record in records:
        if 'memory_group' in record:
            if record['memory_group'] != 'gluomem' or record['system'] != 'gluomem':
                raise ValueError('only explicit GluoMem process group may share memory reservation')
            grouped_caps.add(record['memory_mib'])
    if len(grouped_caps) > 1:
        raise ValueError('shared process group must have one aggregate memory ceiling')
    if len({j.job_id for j in jobs}) != len(jobs):
        raise ValueError("duplicate QA/job identity")
    for index, record in enumerate(records):
        if (
            record["system"] not in native_limits
            or type(record["memory_mib"]) is not int
            or not 1 <= record["memory_mib"] <= memory_limit_mib
        ):
            raise ValueError("unadmitted native resource request")
        path = paths[index]
        if path == out or path.is_relative_to(out) or out.is_relative_to(path):
            raise ValueError("job/coordinator output overlap")
        if any(path.is_relative_to(p) or p.is_relative_to(path) for p in paths[index + 1 :]):
            raise ValueError("job output overlap")
    if any(p.exists() or p.is_symlink() for p in [out, *paths]):
        raise FileExistsError("fresh output paths required")
    # Starts before first mkdir/plan/cache activity; not the legacy post-setup clock.
    if budget is None:
        budget = CampaignBudget(max_calls=max_calls, max_seconds=max_seconds)
    elif budget.max_calls != max_calls or budget.max_seconds != max_seconds:
        raise ValueError("external prospective budget caps differ")
    budget.check()
    cancel = Event()
    limiter = FairRequestLimiter(max_in_flight=300, max_in_flight_tokens=max_in_flight_tokens)
    out.mkdir(parents=True, exist_ok=False)
    policy = dict(
        max_calls=max_calls,
        max_seconds=max_seconds,
        max_workers=max_workers,
        native_limits=native_limits,
        memory_limit_mib=memory_limit_mib,
        max_in_flight=300,
        max_in_flight_tokens=max_in_flight_tokens,
    )
    plan = {"protocol": "lme-first-coordinator@v1", "jobs": records, "policy": policy}
    (out / "plan.json").write_bytes(json_bytes(plan))
    rows = [
        {"job_id": j.job_id, "history_id": j.history_id, "status": "queued", "calls": 0}
        for j in jobs
    ]
    sequence = 0
    memory = peak_memory = peak_active_jobs = 0
    memory_groups = {}
    journal = (out / "journal.jsonl").open("xb")

    def append(event, **fields):
        nonlocal sequence
        with budget.lock:
            sequence += 1
            try:
                journal.write(json_bytes(dict(seq=sequence, event=event, **fields)))
                journal.flush()
                if event == "dispatch":
                    os.fsync(journal.fileno())
            except BaseException:
                budget.failed = True
                cancel.set()
                raise

    def checkpoint(value):
        budget.check()
        temp = out / "progress.pending.json"
        with temp.open("xb") as stream:
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, out / "progress.json")
        journal.flush()
        os.fsync(journal.fileno())

    class Context(JobContext):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.thread = local()

        def begin_call(self):
            if not getattr(self.thread, "leased", False) or self.thread.charged:
                self.failure = "invalid or duplicate call accounting"
                self.budget.failed = True
                cancel.set()
                raise guard_stopped(DispatchStopped("call requires exactly one leased admission"), 'accounting-invalid')
            with budget.lock:
                super().begin_call()
                self.thread.charged = True
                self.thread.ordinal = self.calls
                append(
                    "dispatch",
                    job_id=self.job.job_id,
                    ordinal=self.calls,
                    total_calls=budget.calls,
                    tokens=self.thread.tokens,
                    dispatched=budget.clock(),
                )

        @contextmanager
        def lease(self, tokens):
            if getattr(self.thread, "leased", False):
                raise guard_stopped(DispatchStopped("nested request lease forbidden"), 'accounting-invalid')
            try:
                with self.limiter.request(
                    tokens, deadline=self.deadline, cancel=self.cancel, clock=self.budget.clock
                ):
                    self.check()
                    self.thread.leased = True
                    self.thread.charged = False
                    self.thread.tokens = tokens
                    try:
                        yield
                    finally:
                        if self.thread.charged:
                            append("release", job_id=self.job.job_id, ordinal=self.thread.ordinal)
                        self.thread.leased = False
            except BaseException:
                self.failure = self.failure or "local request failed"
                raise

    checkpoint({"quality_complete": False, "jobs": rows, "accounting": budget.snapshot()})
    pending = list(range(len(jobs)))
    primary_remaining = sum(r.get("phase", "primary") == "primary" for r in records)
    running = {}
    active_histories = set()
    native_active = {key: 0 for key in native_limits}
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while pending or running:
                seen = set(active_histories)
                for index in pending[:]:
                    job, record = jobs[index], records[index]
                    if budget.failed or cancel.is_set() or budget.clock() >= budget.deadline:
                        pending.remove(index)
                        rows[index]["status"] = "unrun"
                        primary_remaining -= record.get("phase", "primary") == "primary"
                        append("end", job_id=job.job_id, status="unrun", calls=0)
                        continue
                    if (
                        record.get("phase", "primary") == "control-after-primary"
                        and primary_remaining
                    ):
                        continue
                    if len(running) >= max_workers:
                        break
                    if job.history_id in seen:
                        continue
                    seen.add(job.history_id)
                    if (
                        native_active[record["system"]] >= native_limits[record["system"]]
                        or memory + (0 if record.get('memory_group') in memory_groups else record["memory_mib"]) > memory_limit_mib
                    ):
                        continue
                    if admit is not None and not admit(record):
                        if not running:
                            pending.remove(index)
                            rows[index]["status"] = "unrun"
                            primary_remaining -= record.get("phase", "primary") == "primary"
                            append("end", job_id=job.job_id, status="unrun", calls=0)
                        continue
                    if budget.clock() + job.seconds_budget > budget.deadline or not budget.reserve(
                        job
                    ):
                        if release_admission is not None:
                            release_admission(record)
                        if running:
                            continue
                        pending.remove(index)
                        rows[index]["status"] = "unrun"
                        primary_remaining -= record.get("phase", "primary") == "primary"
                        append("end", job_id=job.job_id, status="unrun", calls=0)
                        continue
                    pending.remove(index)
                    context = Context(job, budget, limiter, cancel=cancel)
                    rows[index]["status"] = "running"
                    active_histories.add(job.history_id)
                    native_active[record["system"]] += 1
                    group = record.get('memory_group')
                    if group is None or group not in memory_groups:
                        memory += record["memory_mib"]
                    if group is not None:
                        memory_groups[group] = memory_groups.get(group,0)+1
                    peak_memory = max(peak_memory, memory)
                    peak_active_jobs = max(peak_active_jobs,len(active_histories))
                    append(
                        "start",
                        job_id=job.job_id,
                        deadline=context.deadline,
                        started=budget.clock(),
                    )
                    running[pool.submit(run_job, job, context)] = (index, context)
                if running:
                    done, _ = wait(running, timeout=0.25, return_when=FIRST_COMPLETED)
                    for future in done:
                        index, context = running.pop(future)
                        record, job = records[index], jobs[index]
                        active_histories.remove(job.history_id)
                        native_active[record["system"]] -= 1
                        group = record.get('memory_group')
                        if group is not None:
                            memory_groups[group] -= 1
                            if not memory_groups[group]:
                                del memory_groups[group]
                        if group is None or group not in memory_groups:
                            memory -= record["memory_mib"]
                        budget.release(context)
                        row = rows[index]
                        row["calls"] = context.calls
                        try:
                            value = future.result()
                            context.check()
                            json_bytes(value)
                            row["status"] = "complete"
                            # Results belong to leaves; no arbitrary callback text in coordinator.
                        except Exception as exc:
                            row.update(status="failed", failure=safe_failure_diagnostic(exc))
                            if budget.failed:
                                cancel.set()
                        primary_remaining -= record.get("phase", "primary") == "primary"
                        append(
                            "end",
                            job_id=job.job_id,
                            status=row["status"],
                            calls=context.calls,
                            **({"failure": row["failure"]} if "failure" in row else {}),
                        )
        report = {
            "quality_complete": False,
            "completion_scope": "coordinator only; native leaf gates separate",
            "jobs": rows,
            "accounting": budget.snapshot(),
            "limiter": limiter.snapshot(),
            "peak_memory_mib": peak_memory,
            "memory_metric":"scheduler-reservations-not-physical-measurement",
            "peak_active_history_jobs":peak_active_jobs,
            "checkpoint_writes": 2,
            "policy": policy,
            "journal_durability": "dispatch fsync before wire; hardware power-loss guarantees unspecified",
        }
        append("finish", report=report)
        journal.flush()
        os.fsync(journal.fileno())
        # Replay exact persisted bytes before final publication; one linear scan.
        replayed = replay_journal(
            [json.loads(line) for line in (out / "journal.jsonl").read_bytes().splitlines()],
            records,
        )
        if replayed != report:
            raise ValueError("journal replay differs")
        # Failed shared budgets still need terminal incomplete accounting, not a false success.
        material = json_bytes(report)
        with (out / "progress.pending.json").open("xb") as stream:
            stream.write(material)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(out / "progress.pending.json", out / "progress.json")
        if (out / "progress.json").read_bytes() != material:
            raise ValueError("coordinator final readback differs")
        journal.close()
        if budget.clock() >= budget.deadline:
            raise DispatchStopped("future-phase deadline exhausted during finalization")
        return report
    except BaseException as exc:
        # Preserve the originating exception and actual spend even on disk failure.
        exc.__dict__["accounting"] = budget.snapshot()
        try:
            failure = {
                "quality_complete": False,
                "coordinator_closed_verified": False,
                "accounting": budget.snapshot(),
                "failure": safe_failure_diagnostic(exc),
            }
            with (out / "coordinator-incomplete.json").open("xb") as stream:
                stream.write(json_bytes(failure))
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            exc.add_note(
                "coordinator incomplete receipt unavailable; in-memory accounting retained"
            )
        raise
    finally:
        journal.close()


def replay_journal(events, records):
    """Linear replay rejects duplicate/missing calls, QA rows and open attempts.

    External source/plan/leaf bindings still establish native truth; this is not
    a replacement for pinned native scorer/receipt verifiers.
    """
    records = list(records)
    by_id = {r["job_id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate expected QA")
    states = {key: "queued" for key in by_id}
    calls = dict.fromkeys(by_id, 0)
    active_histories = set()
    active_calls = {}
    deadlines = {}
    history_queue = {}
    native = {}
    native_peak = {}
    memory = peak_memory = peak_requests = tokens = peak_tokens = peak_workers = 0
    memory_groups = {}
    group_caps = set()
    for row in records:
        if 'memory_group' in row:
            if row['memory_group'] != 'gluomem' or row['system'] != 'gluomem':
                raise ValueError('invalid shared process memory group')
            group_caps.add(row['memory_mib'])
    if len(group_caps) > 1:
        raise ValueError('inconsistent shared process aggregate ceiling')
    for r in records:
        history_queue.setdefault(r["history_id"], deque()).append(r["job_id"])
    total = 0
    report = None
    ends = {}
    for expected_seq, event in enumerate(events, 1):
        if event["seq"] != expected_seq or report is not None:
            raise ValueError("journal sequence/drop/duplicate")
        kind = event["event"]
        if kind == "finish":
            if (
                any(state not in {"complete", "failed", "unrun"} for state in states.values())
                or active_calls
            ):
                raise ValueError("incomplete journal cannot certify closure")
            report = event["report"]
            continue
        key = event["job_id"]
        if key not in by_id:
            raise ValueError("unknown job receipt")
        row = by_id[key]
        if kind == "start":
            if (
                states[key] != "queued"
                or row["history_id"] in active_histories
                or history_queue[row["history_id"]][0] != key
            ):
                raise ValueError("duplicate/out-of-order history start")
            if not event["started"] < event["deadline"] <= event["started"] + row["seconds_budget"]:
                raise ValueError("history clock differs")
            states[key] = "running"
            active_histories.add(row["history_id"])
            deadlines[key] = (event["started"], event["deadline"])
            native[row["system"]] = native.get(row["system"], 0) + 1
            native_peak[row["system"]] = max(
                native_peak.get(row["system"], 0), native[row["system"]]
            )
            group = row.get('memory_group')
            if group is None or group not in memory_groups:
                memory += row["memory_mib"]
            if group is not None:
                memory_groups[group] = memory_groups.get(group,0)+1
            peak_memory = max(peak_memory, memory)
            peak_workers = max(peak_workers, len(active_histories))
        elif kind == "dispatch":
            if states[key] != "running" or event["ordinal"] != calls[key] + 1:
                raise ValueError("duplicate/out-of-order/unadmitted dispatch")
            calls[key] += 1
            total += 1
            if calls[key] > row["call_budget"] or event["total_calls"] != total:
                raise ValueError("shared call accounting differs")
            if (
                type(event["tokens"]) is not int
                or event["tokens"] < 1
                or not deadlines[key][0] <= event["dispatched"] < deadlines[key][1]
            ):
                raise ValueError("invalid tokenizer reservation/dispatch clock")
            active_calls[(key, event["ordinal"])] = event["tokens"]
            tokens += event["tokens"]
            peak_tokens = max(peak_tokens, tokens)
            peak_requests = max(peak_requests, len(active_calls))
        elif kind == "release":
            identity = (key, event["ordinal"])
            if identity not in active_calls:
                raise ValueError("unknown/duplicate release")
            tokens -= active_calls.pop(identity)
        elif kind == "end":
            status = event["status"]
            if (
                status not in {"complete", "failed", "unrun"}
                or states[key] != ("queued" if status == "unrun" else "running")
                or event["calls"] != calls[key]
                or any(k == key for k, _ in active_calls)
            ):
                raise ValueError("invalid terminal state/coverage")
            if status != "unrun":
                active_histories.remove(row["history_id"])
                group = row.get('memory_group')
                if group is not None:
                    memory_groups[group] -= 1
                    if not memory_groups[group]:
                        del memory_groups[group]
                if group is None or group not in memory_groups:
                    memory -= row["memory_mib"]
                native[row["system"]] -= 1
            if status == "unrun":
                history_queue[row["history_id"]].remove(key)
            else:
                if history_queue[row["history_id"]][0] != key:
                    raise ValueError("out-of-order history end")
                history_queue[row["history_id"]].popleft()
            states[key] = status
            ends[key] = event
        else:
            raise ValueError("unknown journal record")
    if (
        report is None
        or report["quality_complete"] is not False
        or report["accounting"]["calls"] != total
    ):
        raise ValueError("missing/forged final accounting")
    expected_rows = [
        {
            "job_id": r["job_id"],
            "history_id": r["history_id"],
            "status": states[r["job_id"]],
            "calls": calls[r["job_id"]],
            **({"failure": ends[r["job_id"]]["failure"]} if "failure" in ends[r["job_id"]] else {}),
        }
        for r in records
    ]
    if report["jobs"] != expected_rows:
        raise ValueError("forged QA coverage/summary")
    policy, limits = report["policy"], report["limiter"]
    if (
        policy["max_in_flight"] != 300
        or total > policy["max_calls"]
        or not peak_requests <= limits["peak_in_flight"] <= 300
        or not peak_tokens <= limits["peak_in_flight_tokens"] <= policy["max_in_flight_tokens"]
        or limits["in_flight"] != 0
        or limits["in_flight_tokens"] != 0
        or limits["waiting"] != 0
        or report.get("peak_active_history_jobs") != peak_workers
        or report.get("memory_metric") != "scheduler-reservations-not-physical-measurement"
        or report["peak_memory_mib"] != peak_memory
        or peak_memory > policy["memory_limit_mib"]
        or peak_workers > policy["max_workers"]
        or any(value > policy["native_limits"].get(key, 0) for key, value in native_peak.items())
    ):
        raise ValueError("forged global token/request/native resource accounting")
    return report


def verify_journal(out, records):
    out = Path(out)
    if (out / "coordinator-incomplete.json").exists():
        raise ValueError("coordinator terminal failure overrides stale progress")
    plan = json.loads((out / "plan.json").read_bytes())
    if plan["jobs"] != list(records):
        raise ValueError("external frozen job selection differs")
    events = [json.loads(line) for line in (out / "journal.jsonl").read_bytes().splitlines()]
    report = replay_journal(events, records)
    if report["policy"] != plan["policy"] or report != json.loads(
        (out / "progress.json").read_bytes()
    ):
        raise ValueError("checkpoint differs from journal")
    return report


def main(argv=None):
    """Offline-only CLI. No execute subcommand exists until native gates align."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source-spec", type=Path, required=True)
    prepare.add_argument("--source-spec-sha256", required=True)
    prepare.add_argument("--future-max-calls", type=int, required=True)
    prepare.add_argument("--future-max-seconds", type=int, required=True)
    prepare.add_argument("--local-memory-mib", type=int, required=True)
    prepare.add_argument("--gluomem-workers", type=int)
    prepare.add_argument("--gluomem-group-memory-mib", type=int, default=4096)
    prepare.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        plan = build_plan(
            source_spec=args.source_spec,
            source_spec_sha256=args.source_spec_sha256,
            future_max_calls=args.future_max_calls,
            future_max_seconds=args.future_max_seconds,
            local_memory_mib=args.local_memory_mib,
            gluomem_workers=args.gluomem_workers,
            gluomem_group_memory_mib=args.gluomem_group_memory_mib,
        )
        validate_plan(plan)
        material = json_bytes(plan)
        # Caller creates its own review namespace; never mkdir/copy caches in old roots.
        with args.out.open("xb") as stream:
            stream.write(material)
        if args.out.read_bytes() != material:
            raise ValueError("prepared plan readback differs")
        print(
            json.dumps(
                {
                    "plan_sha256": hashlib.sha256(material).hexdigest(),
                    "jobs": len(plan["jobs"]),
                    "execution_authorized": False,
                }
            )
        )
    else:
        plan = json.loads(read_bound(args.plan, args.sha256))
        count = validate_plan(plan)
        print(
            json.dumps(
                {
                    "verified_jobs": count,
                    "execution_authorized": False,
                    "blockers": plan["blockers"],
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
