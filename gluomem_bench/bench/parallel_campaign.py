"""Explicit, provider-independent concurrency and campaign accounting.

Transports must use ``ctx.request(prompt_tokens + output_reserve)`` for combined
accounting/lease, or ``ctx.lease(tokens)`` around a native path which already calls
``ctx.begin_call()``. Never combine both counting paths. Hold the lease across the
actual HTTP/DSPy call and validation; use ``ctx.remaining()`` to bound transport
timeout. Token reservations are conservative bounds, NOT measured usage.

``run_jobs(jobs, policy=..., out=..., run_job=lambda job, ctx: ...)`` snapshots only
job IDs, history keys, output locations and full call/time envelopes. Close over
operator configuration and samples in the supplied callable. Use ctx as the outer
workload-budget protocol for ingestion, readers and judges of the SAME job. Every
job receives a distinct context backed by ONE campaign budget/limiter. Native
clients remain per-job concurrency 1; this module never changes RackProfile.

The callable must join its work and close stores before returning a JSON receipt.
A failure never produces a result or substitute score. Queued IDs remain in frozen
order as cancelled/unrun. Receipt I/O errors stop admission and expose actual
conservative attempted-call accounting in CampaignReceiptError.receipt. Calls are
charged immediately before dispatch; a failed pre-dispatch receipt write is also
charged conservatively. No retries, resampling, scoring or provider/dataset loading
occurs here. Python cannot forcibly cancel an already executing transport: callers
MUST honor remaining timeout; cancellation prevents subsequent dispatches.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Condition, Event, RLock

from gluomem_bench.bench.embedding_runtime import safe_failure_diagnostic


def guard_stopped(error, reason):
    """Attach the static originating boundary without changing exception identity."""
    error.guard_reason = reason
    return error


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class RequestLimiter:
    """Share one instance across every transport in a campaign."""

    def __init__(self, *, max_in_flight, max_in_flight_tokens):
        _positive_int(max_in_flight, "max_in_flight")
        _positive_int(max_in_flight_tokens, "max_in_flight_tokens")
        if max_in_flight > 500:
            raise ValueError("max_in_flight cannot exceed 500")
        self.max_in_flight = max_in_flight
        self.max_in_flight_tokens = max_in_flight_tokens
        self._condition = Condition()
        self._requests = self._tokens = 0

    @contextmanager
    def request(self, tokens, *, deadline=None, cancel=None, clock=time.monotonic):
        _positive_int(tokens, "tokens")
        if tokens > self.max_in_flight_tokens:
            raise ValueError("request exceeds aggregate token cap")
        with self._condition:
            while True:
                if cancel is not None and cancel.is_set():
                    raise guard_stopped(CampaignStopped("request cancelled"), 'cancelled')
                remaining = None if deadline is None else deadline - clock()
                if remaining is not None and remaining <= 0:
                    raise guard_stopped(CampaignStopped("request deadline exhausted"), 'job-deadline')
                if (
                    self._requests < self.max_in_flight
                    and self._tokens + tokens <= self.max_in_flight_tokens
                ):
                    break
                self._condition.wait(timeout=0.05 if remaining is None else min(0.05, remaining))
            self._requests += 1
            self._tokens += tokens
        try:
            yield
        finally:
            with self._condition:
                self._requests -= 1
                self._tokens -= tokens
                self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            return {"in_flight": self._requests, "in_flight_tokens": self._tokens}


@dataclass(frozen=True)
class CampaignPolicy:
    max_workers: int
    max_in_flight: int
    max_in_flight_tokens: int
    max_calls: int
    max_seconds: float
    stop_on_failure: bool

    def __post_init__(self):
        for name in ("max_workers", "max_in_flight", "max_in_flight_tokens", "max_calls"):
            _positive_int(getattr(self, name), name)
        if self.max_workers > 500 or self.max_in_flight > 500:
            raise ValueError("worker/request concurrency cannot exceed 500")
        _positive_seconds(self.max_seconds, "max_seconds")
        if type(self.stop_on_failure) is not bool:
            raise ValueError("stop_on_failure must be boolean")


def _positive_seconds(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class Job:
    """Metadata only; caller closes over samples/config, never stores them here."""

    job_id: str
    history_id: str
    output: Path
    call_budget: int
    seconds_budget: float

    def __post_init__(self):
        for name in ("job_id", "history_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        _positive_int(self.call_budget, "call_budget")
        _positive_seconds(self.seconds_budget, "seconds_budget")
        object.__setattr__(self, "output", Path(self.output).resolve())


class CampaignBudget:
    """Original total caps, optionally debited by retained spend.

    ``started`` is this instance's actual start; prior elapsed time is subtracted
    from its deadline exactly once. ``calls`` and snapshot ``elapsed_seconds``
    include retained spend. Continuation snapshots additionally split prior/new
    spend; zero-prior snapshots keep the legacy two-key shape.
    """

    def __init__(
        self,
        *,
        max_calls,
        max_seconds,
        clock=time.monotonic,
        prior_calls=0,
        prior_elapsed_seconds=0,
        started_monotonic=None,
    ):
        _positive_int(max_calls, "max_calls")
        _positive_seconds(max_seconds, "max_seconds")
        if type(prior_calls) is not int or not 0 <= prior_calls <= max_calls:
            raise ValueError("prior_calls must be a nonnegative integer within max_calls")
        if (
            type(prior_elapsed_seconds) not in (float, int)
            or not math.isfinite(prior_elapsed_seconds)
            or not 0 <= prior_elapsed_seconds <= max_seconds
        ):
            raise ValueError(
                "prior_elapsed_seconds must be finite, nonnegative and within max_seconds"
            )
        self.max_calls = max_calls
        self.max_seconds = max_seconds
        self.prior_calls = prior_calls
        self.prior_elapsed_seconds = prior_elapsed_seconds
        self.failed = False
        self.clock = clock
        now = clock()
        if started_monotonic is not None and (
            type(started_monotonic) not in (int, float)
            or not math.isfinite(started_monotonic) or not 0 <= started_monotonic <= now
        ):
            raise ValueError("started_monotonic must be finite, nonnegative and no later than now")
        self.started = now if started_monotonic is None else started_monotonic
        self.deadline = self.started + (max_seconds - prior_elapsed_seconds)
        self.calls = prior_calls
        self.reserved_calls = 0
        self.lock = RLock()

    def check(self):
        with self.lock:
            if self.failed or self.clock() >= self.deadline:
                reason = 'campaign-failed' if self.failed else 'campaign-deadline'
                self.failed = True
                raise guard_stopped(CampaignStopped("campaign failed or deadline exhausted"), reason)

    def remaining(self):
        with self.lock:
            self.check()
            return self.deadline - self.clock()

    def begin_call(self):
        """Standalone counter protocol; job workers should use their JobContext."""
        with self.lock:
            self.check()
            if self.calls + self.reserved_calls >= self.max_calls:
                self.failed = True
                raise guard_stopped(CampaignStopped("campaign call allowance exhausted"), 'campaign-call-cap')
            self.calls += 1

    def reserve(self, job):
        with self.lock:
            if self.calls + self.reserved_calls + job.call_budget > self.max_calls:
                return False
            self.reserved_calls += job.call_budget
            return True

    def release(self, context):
        with self.lock:
            self.reserved_calls -= context.job.call_budget - context.calls

    def snapshot(self):
        with self.lock:
            elapsed = self.clock() - self.started
            result = {"calls": self.calls, "elapsed_seconds": self.prior_elapsed_seconds + elapsed}
            if self.prior_calls or self.prior_elapsed_seconds:
                result.update(
                    prior_calls=self.prior_calls,
                    new_calls=self.calls - self.prior_calls,
                    prior_elapsed_seconds=self.prior_elapsed_seconds,
                    new_elapsed_seconds=elapsed,
                )
            return result


class CampaignStopped(RuntimeError):
    """A guard refusal, never a scoreable quality outcome."""


class CampaignReceiptError(RuntimeError):
    """Persistence failed; receipt carries actual spend when disk is unavailable."""

    def __init__(self, receipt):
        super().__init__("campaign receipt unavailable; further admission stopped")
        self.receipt = receipt


class JobContext:
    def __init__(self, job, budget, limiter, on_spend=None, cancel=None):
        self.cancel = cancel
        self.on_spend = on_spend
        self.job = job
        self.budget = budget
        self.limiter = limiter
        self.calls = 0
        self.failure = None
        self.deadline = min(budget.deadline, budget.clock() + job.seconds_budget)

    @property
    def failed(self):
        return self.failure is not None

    @failed.setter
    def failed(self, value):
        if value:
            self.failure = self.failure or "job failure latched by transport"

    def check(self):
        if self.cancel is not None and self.cancel.is_set():
            self.failure = "job cancelled"
            raise guard_stopped(CampaignStopped(self.failure), 'cancelled')
        self.budget.check()
        if self.budget.clock() >= self.deadline:
            self.failure = "job/campaign deadline exhausted"
            raise guard_stopped(CampaignStopped(self.failure), 'job-deadline')
        if self.failure:
            raise guard_stopped(CampaignStopped(self.failure), 'job-failed')

    def remaining(self):
        self.check()
        return self.deadline - self.budget.clock()

    def begin_call(self):
        with self.budget.lock:
            self.check()
            if self.calls >= self.job.call_budget:
                self.failure = "job call allowance exhausted"
                raise guard_stopped(CampaignStopped(self.failure), 'job-call-cap')
            self.calls += 1
            self.budget.calls += 1
            self.budget.reserved_calls -= 1
            if self.on_spend is not None:
                self.on_spend(self)

    @contextmanager
    def lease(self, tokens):
        """Concurrency only: use when the native path already calls begin_call()."""
        try:
            with self.limiter.request(
                tokens, deadline=self.deadline, cancel=self.cancel, clock=self.budget.clock
            ):
                self.check()
                yield
        except Exception:
            self.failure = self.failure or "local request failed"
            raise

    @contextmanager
    def request(self, tokens):
        """Combined lease + begin_call; do NOT also call begin_call in the body."""
        with self.lease(tokens):
            self.begin_call()
            yield self.remaining()


def _json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")


def _atomic_write(path, value):
    material = _json_bytes(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(material)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_jobs(jobs, *, policy, out, run_job, cancel=None, clock=None, budget=None):
    """Run callable(job, context); no network clients are constructed here.

    A result is an explicit JSON value, never an inferred score. This coordinator
    serializes jobs sharing a history key while permitting unrelated histories.
    Pass a driver-created ``budget`` before setup to retain that setup's spend
    and deadline. Policy caps must be the original totals, not remaining caps.
    An omitted clock uses that budget's clock; an explicit clock must be the
    identical callable. Without a budget, the legacy fresh monotonic guard is
    created after plan publication.
    """
    if budget is not None:
        if not isinstance(budget, CampaignBudget):
            raise ValueError("budget must be a CampaignBudget")
        if budget.max_calls != policy.max_calls or budget.max_seconds != policy.max_seconds:
            raise ValueError("budget original total caps conflict with policy")
        if clock is not None and clock is not budget.clock:
            raise ValueError("clock must be identical to budget.clock")
        with budget.lock:
            budget.check()
            if budget.calls >= budget.max_calls:
                raise CampaignStopped("campaign call allowance exhausted")
    cancel = cancel if cancel is not None else Event()
    jobs = tuple(jobs)
    out = Path(out).resolve()
    paths = [Path(job.output).resolve() for job in jobs]
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("duplicate planned job IDs")
    for index, path in enumerate(paths):
        if path == out or out.is_relative_to(path):
            raise ValueError("job output overlaps campaign receipts")
        for other in paths[index + 1 :]:
            if path.is_relative_to(other) or other.is_relative_to(path):
                raise ValueError("job outputs overlap")
    for path in [out, *paths]:
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    if budget is not None:
        budget.check()
    out.mkdir(parents=True)
    plan = {
        "version": 1,
        "policy": asdict(policy),
        "jobs": [{**asdict(job), "output": str(job.output)} for job in jobs],
    }
    _atomic_write(out / "plan.json", plan)
    if budget is None:
        budget = CampaignBudget(
            max_calls=policy.max_calls,
            max_seconds=policy.max_seconds,
            clock=time.monotonic if clock is None else clock,
        )
    limiter = RequestLimiter(
        max_in_flight=policy.max_in_flight, max_in_flight_tokens=policy.max_in_flight_tokens
    )
    rows = [{"job_id": j.job_id, "history_id": j.history_id, "status": "queued"} for j in jobs]
    result = {
        "version": 1,
        "plan_sha256": hashlib.sha256(_json_bytes(plan)).hexdigest(),
        "jobs": rows,
    }

    def save():
        with budget.lock:
            result["accounting"] = budget.snapshot()
            if "receipt_failure" in result:
                raise CampaignReceiptError(result)
            try:
                _atomic_write(out / "progress.json", result)
            except Exception as exc:
                budget.failed = True
                result["receipt_failure"] = safe_failure_diagnostic(exc)
                for row in rows:
                    if row["status"] == "queued":
                        row.update(status="unrun", reason="accounting persistence unavailable")
                raise CampaignReceiptError(result) from exc

    def record_spend(context):
        rows[jobs.index(context.job)]["calls"] = context.calls
        save()

    save()
    pending = list(range(len(jobs)))
    running = {}
    active = set()
    stopped = False
    with ThreadPoolExecutor(max_workers=policy.max_workers) as pool:
        while pending or running:
            seen_histories = set(active)
            for index in pending[:]:
                job = jobs[index]
                if cancel.is_set() or stopped:
                    pending.remove(index)
                    rows[index].update(
                        status="cancelled" if cancel.is_set() else "unrun",
                        reason="cancelled" if cancel.is_set() else "prior job failed",
                    )
                    save()
                    continue
                if len(running) >= policy.max_workers:
                    break
                if job.history_id in seen_histories:
                    continue
                seen_histories.add(job.history_id)
                if budget.clock() + job.seconds_budget > budget.deadline:
                    pending.remove(index)
                    rows[index].update(status="unrun", reason="campaign deadline envelope")
                    save()
                    continue
                if not budget.reserve(job):
                    if not running:
                        pending.remove(index)
                        rows[index].update(status="unrun", reason="campaign call allowance")
                        save()
                    continue
                pending.remove(index)
                active.add(job.history_id)
                rows[index]["status"] = "running"
                ctx = JobContext(job, budget, limiter, on_spend=record_spend, cancel=cancel)
                save()
                running[pool.submit(run_job, job, ctx)] = (index, ctx)
            if running:
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    index, ctx = running.pop(future)
                    active.remove(jobs[index].history_id)
                    budget.release(ctx)
                    rows[index]["calls"] = ctx.calls
                    try:
                        value = future.result()
                        if ctx.failure or budget.failed or budget.clock() >= ctx.deadline:
                            raise CampaignStopped(
                                ctx.failure or "finalization deadline/guard exhausted"
                            )
                        _json_bytes(value)
                        rows[index].update(status="complete", result=value)
                    except Exception as exc:
                        stopped = stopped or policy.stop_on_failure
                        status = (
                            "cancelled"
                            if cancel.is_set() and isinstance(exc, CampaignStopped)
                            else "failed"
                        )
                        rows[index].update(
                            status=status, failure=safe_failure_diagnostic(exc)
                        )
                    save()
    save()
    return result
