"""Offline scheduler contracts; synthetic work is not benchmark inference."""

import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from threading import Barrier, Event, Lock

import pytest


def policy(**changes):
    values = dict(
        max_workers=3,
        max_in_flight=2,
        max_in_flight_tokens=20,
        max_calls=20,
        max_seconds=10,
        stop_on_failure=False,
    )
    values.update(changes)
    return api().CampaignPolicy(**values)


def test_run_jobs_serializes_histories_and_freezes_plan(tmp_path):
    m = api()
    p = policy()
    with pytest.raises(FrozenInstanceError):
        p.max_calls = 100
    jobs = [
        m.Job("a1", "a", tmp_path / "a1", 2, 2),
        m.Job("a2", "a", tmp_path / "a2", 2, 2),
        m.Job("b1", "b", tmp_path / "b1", 2, 2),
    ]
    barrier = Barrier(2)
    active = set()
    lock = Lock()
    observations = []

    def run(job, ctx):
        with lock:
            assert job.history_id not in active
            active.add(job.history_id)
            observations.append(job.job_id)
        if job.job_id in ("a1", "b1"):
            barrier.wait(timeout=2)
        with ctx.request(4) as timeout:
            assert 0 < timeout <= 2
        with lock:
            active.remove(job.history_id)
        return {"artifact": job.job_id}

    result = m.run_jobs(jobs, policy=p, out=tmp_path / "campaign", run_job=run)
    assert [r["status"] for r in result["jobs"]] == ["complete"] * 3
    assert result["accounting"]["calls"] == 3
    assert observations.index("a1") < observations.index("a2")
    plan = json.loads((tmp_path / "campaign" / "plan.json").read_bytes())
    assert [j["job_id"] for j in plan["jobs"]] == ["a1", "a2", "b1"]
    assert plan["policy"]["max_calls"] == 20
    assert json.loads((tmp_path / "campaign" / "progress.json").read_bytes()) == result
    assert result["jobs"][0]["result"] == {"artifact": "a1"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_workers", 0),
        ("max_workers", 501),
        ("max_calls", True),
        ("max_calls", 0),
        ("max_seconds", float("nan")),
        ("max_seconds", float("inf")),
        ("max_seconds", -1),
        ("max_in_flight", 501),
        ("max_in_flight_tokens", 0),
        ("stop_on_failure", 1),
    ],
)
def test_policy_rejects_invalid_bounds(field, value):
    with pytest.raises(ValueError):
        policy(**{field: value})


def test_freshness_checks_all_targets_before_any_work(tmp_path):
    m = api()
    old = tmp_path / "second" / "calls.jsonl"
    old.parent.mkdir()
    old.write_bytes(b"preserve me")
    jobs = [m.Job("a", "a", tmp_path / "first", 1, 1), m.Job("b", "b", old.parent, 1, 1)]
    invoked = []
    with pytest.raises(FileExistsError):
        m.run_jobs(
            jobs,
            policy=policy(),
            out=tmp_path / "campaign",
            run_job=lambda *args: invoked.append(args),
        )
    assert invoked == []
    assert old.read_bytes() == b"preserve me"
    assert not (tmp_path / "campaign").exists()
    assert not (tmp_path / "first").exists()
    old.unlink()
    old.parent.rmdir()
    for bad_jobs in (
        [jobs[0], jobs[0]],
        [jobs[0], m.Job("b", "b", tmp_path / "first" / "nested", 1, 1)],
    ):
        with pytest.raises(ValueError):
            m.run_jobs(
                bad_jobs,
                policy=policy(),
                out=tmp_path / "campaign",
                run_job=lambda *args: invoked.append(args),
            )
    (tmp_path / "campaign").mkdir()
    with pytest.raises(FileExistsError):
        m.run_jobs(
            [jobs[0]],
            policy=policy(),
            out=tmp_path / "campaign",
            run_job=lambda *args: invoked.append(args),
        )
    assert invoked == []


def test_failed_requests_keep_campaign_spend_and_unrun_jobs(tmp_path):
    m = api()
    jobs = [m.Job(str(i), str(i), tmp_path / str(i), 1, 1) for i in range(3)]
    attempted = []

    def run(job, ctx):
        attempted.append(job.job_id)
        with ctx.request(3):
            raise OSError("actual request failed")

    result = m.run_jobs(
        jobs, policy=policy(max_workers=1, max_calls=2), out=tmp_path / "campaign", run_job=run
    )
    assert attempted == ["0", "1"]
    assert result["accounting"]["calls"] == 2
    assert [r["status"] for r in result["jobs"]] == ["failed", "failed", "unrun"]
    assert result["jobs"][0]["failure"]["type"] == "OSError"
    assert result["jobs"][0]["calls"] == 1
    assert all("result" not in row and "score" not in row for row in result["jobs"])
    assert "call" in result["jobs"][2]["reason"]
    assert json.loads((tmp_path / "campaign" / "progress.json").read_bytes()) == result


def test_dispatch_guard_latches_even_if_job_catches_exhaustion(tmp_path):
    m = api()
    dispatched = []

    def run(job, ctx):
        for _ in range(3):
            try:
                with ctx.request(2):
                    dispatched.append(job.job_id)
            except m.CampaignStopped:
                pass
        return {"must_not_be_published": True}

    result = m.run_jobs(
        [m.Job("one", "h", tmp_path / "job", 1, 1)],
        policy=policy(),
        out=tmp_path / "campaign",
        run_job=run,
    )
    assert dispatched == ["one"]
    assert result["accounting"]["calls"] == 1
    assert result["jobs"][0]["status"] == "failed"
    assert "result" not in result["jobs"][0]


def test_deadline_shared_across_failed_histories_and_checked_at_dispatch(tmp_path):
    m = api()
    now = [100.0]
    attempted = []
    jobs = [m.Job(str(i), str(i), tmp_path / str(i), 2, 4) for i in range(3)]

    def run(job, ctx):
        with ctx.request(2) as timeout:
            assert 0 < timeout <= 4
            attempted.append(job.job_id)
            now[0] += 4
            raise OSError("spent time on failed transport")

    result = m.run_jobs(
        jobs,
        policy=policy(max_workers=1, max_seconds=6),
        out=tmp_path / "campaign",
        run_job=run,
        clock=lambda: now[0],
    )
    assert attempted == ["0"]
    assert result["accounting"]["calls"] == 1
    assert [r["status"] for r in result["jobs"]] == ["failed", "unrun", "unrun"]
    assert "deadline" in result["jobs"][1]["reason"]

    def late(job, ctx):
        now[0] += 7
        with ctx.request(2):
            pytest.fail("must not dispatch after deadline")

    result = m.run_jobs(
        [m.Job("late", "h", tmp_path / "late", 1, 4)],
        policy=policy(max_seconds=6),
        out=tmp_path / "late-campaign",
        run_job=late,
        clock=lambda: now[0],
    )
    assert result["jobs"][0]["status"] == "failed"
    assert result["accounting"]["calls"] == 0


@pytest.mark.parametrize("cancel_before", [True, False])
def test_cancellation_retains_every_planned_id(tmp_path, cancel_before):
    m = api()
    cancel = Event()
    if cancel_before:
        cancel.set()
    jobs = [m.Job(str(i), "same", tmp_path / str(i), 1, 1) for i in range(3)]

    def run(job, ctx):
        with ctx.request(1):
            cancel.set()
        return {"actual": True}

    result = m.run_jobs(
        jobs, policy=policy(), out=tmp_path / "campaign", run_job=run, cancel=cancel
    )
    assert [r["job_id"] for r in result["jobs"]] == ["0", "1", "2"]
    expected = ["cancelled"] * 3 if cancel_before else ["complete", "cancelled", "cancelled"]
    assert [r["status"] for r in result["jobs"]] == expected
    assert result["accounting"]["calls"] == (0 if cancel_before else 1)


def test_stop_on_failure_does_not_admit_next_history(tmp_path):
    m = api()
    jobs = [m.Job(str(i), "same", tmp_path / str(i), 1, 1) for i in range(3)]

    def run(job, ctx):
        with ctx.request(1):
            raise OSError("transport failed")

    result = m.run_jobs(
        jobs, policy=policy(stop_on_failure=True), out=tmp_path / "campaign", run_job=run
    )
    assert [r["status"] for r in result["jobs"]] == ["failed", "unrun", "unrun"]
    assert result["accounting"]["calls"] == 1


def test_limiter_wait_is_cancellable_and_deadline_bounded():
    m = api()
    limiter = m.RequestLimiter(max_in_flight=1, max_in_flight_tokens=3)
    cancel = Event()
    cancel.set()
    with limiter.request(3):
        with pytest.raises(m.CampaignStopped, match="cancel"):
            with limiter.request(1, cancel=cancel):
                pytest.fail("cancelled waiter acquired")
        with pytest.raises(m.CampaignStopped, match="deadline"):
            with limiter.request(1, deadline=10, clock=lambda: 11):
                pytest.fail("expired waiter acquired")
    assert limiter.snapshot() == {"in_flight": 0, "in_flight_tokens": 0}


def test_running_receipt_precedes_callable_and_dispatch_spend_is_durable(tmp_path):
    m = api()
    progress = tmp_path / "campaign" / "progress.json"

    def run(job, ctx):
        assert json.loads(progress.read_bytes())["jobs"][0]["status"] == "running"
        with ctx.request(2):
            saved = json.loads(progress.read_bytes())
            assert saved["accounting"]["calls"] == 1
            assert saved["jobs"][0]["calls"] == 1
        return {"ok": True}

    result = m.run_jobs(
        [m.Job("j", "h", tmp_path / "job", 1, 1)], policy=policy(), out=progress.parent, run_job=run
    )
    assert result["jobs"][0]["status"] == "complete"


def test_budget_protocol_supports_outer_rack_accounting_without_double_count(tmp_path):
    m = api()
    now = [0.0]
    budget = m.CampaignBudget(max_calls=100000, max_seconds=50000, clock=lambda: now[0])
    budget.check()
    assert budget.remaining() == 50000
    budget.begin_call()
    assert budget.calls == 1
    budget.failed = True
    with pytest.raises(m.CampaignStopped):
        budget.begin_call()
    assert budget.calls == 1

    def run(job, ctx):
        assert ctx.remaining() > 0
        # Native Rack paths already call begin_call; acquire lease separately.
        with ctx.lease(2):
            ctx.begin_call()
            ctx.check()  # Last allowed call is still a valid completed response.
        assert ctx.calls == 1
        assert ctx.failed is False
        return {"artifact": "native"}

    result = m.run_jobs(
        [m.Job("j", "h", tmp_path / "job", 1, 1)],
        policy=policy(),
        out=tmp_path / "campaign",
        run_job=run,
    )
    assert result["jobs"][0]["status"] == "complete"
    assert result["accounting"]["calls"] == 1


@pytest.mark.parametrize("external", [False, True])
def test_receipt_write_failure_stops_before_network_and_preserves_accounting(
    tmp_path, monkeypatch, external
):
    m = api()
    original = m._atomic_write
    dispatched = []
    budget = m.CampaignBudget(max_calls=20, max_seconds=10, prior_calls=3) if external else None
    attempted_total = 4 if external else 1

    def fail_spend(path, value):
        if (
            path.name == "progress.json"
            and value.get("accounting", {}).get("calls") == attempted_total
        ):
            raise OSError("disk unavailable")
        return original(path, value)

    monkeypatch.setattr(m, "_atomic_write", fail_spend)

    def run(job, ctx):
        with ctx.request(1):
            dispatched.append(job.job_id)
        return {}

    jobs = [m.Job(str(i), "same", tmp_path / str(i), 1, 1) for i in range(2)]
    with pytest.raises(m.CampaignReceiptError) as caught:
        m.run_jobs(jobs, policy=policy(), out=tmp_path / "campaign", run_job=run, budget=budget)
    assert dispatched == []
    assert caught.value.receipt["accounting"]["calls"] == attempted_total
    if budget is not None:
        assert budget.failed is True
        assert budget.calls == 4
        assert caught.value.receipt["accounting"]["prior_calls"] == 3
        assert caught.value.receipt["accounting"]["new_calls"] == 1
    assert [r["job_id"] for r in caught.value.receipt["jobs"]] == ["0", "1"]
    assert caught.value.receipt["jobs"][1]["status"] == "unrun"
    # The last valid on-disk receipt remains parseable and is never scored.
    saved = json.loads((tmp_path / "campaign" / "progress.json").read_bytes())
    assert saved["jobs"][0]["status"] == "running"
    assert all("score" not in row for row in saved["jobs"])


def test_swallowed_transport_failure_cannot_publish_success(tmp_path):
    m = api()

    def run(job, ctx):
        try:
            with ctx.request(2):
                raise OSError("raw transport failure")
        except OSError:
            pass
        return {"score": 1}

    result = m.run_jobs(
        [m.Job("j", "h", tmp_path / "job", 1, 1)],
        policy=policy(),
        out=tmp_path / "campaign",
        run_job=run,
    )
    assert result["jobs"][0]["status"] == "failed"
    assert "result" not in result["jobs"][0]
    assert result["accounting"]["calls"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"job_id": ""},
        {"job_id": 1},
        {"history_id": ""},
        {"call_budget": 0},
        {"call_budget": True},
        {"seconds_budget": float("nan")},
        {"seconds_budget": -1},
    ],
)
def test_job_metadata_rejects_invalid_envelopes(tmp_path, changes):
    values = dict(
        job_id="j", history_id="h", output=tmp_path / "job", call_budget=1, seconds_budget=1
    )
    values.update(changes)
    with pytest.raises(ValueError):
        api().Job(**values)


def test_cancellation_blocks_later_request_in_running_job(tmp_path):
    m = api()
    cancel = Event()
    dispatched = []

    def run(job, ctx):
        with ctx.request(1):
            dispatched.append(1)
        cancel.set()
        with ctx.request(1):
            dispatched.append(2)
        return {}

    result = m.run_jobs(
        [m.Job("j", "h", tmp_path / "job", 2, 1)],
        policy=policy(),
        out=tmp_path / "campaign",
        run_job=run,
        cancel=cancel,
    )
    assert dispatched == [1]
    assert result["accounting"]["calls"] == 1
    assert result["jobs"][0]["status"] == "cancelled"
    assert result["jobs"][0]["calls"] == 1


def test_late_finalization_cannot_be_marked_complete(tmp_path):
    m = api()
    now = [0.0]

    def run(job, ctx):
        with ctx.request(1):
            pass
        now[0] = 3
        return {"finished_after_envelope": True}

    result = m.run_jobs(
        [m.Job("j", "h", tmp_path / "job", 1, 2)],
        policy=policy(),
        out=tmp_path / "campaign",
        run_job=run,
        clock=lambda: now[0],
    )
    assert result["jobs"][0]["status"] == "failed"
    assert result["accounting"]["calls"] == 1
    assert "result" not in result["jobs"][0]


def test_call_reservation_never_reorders_same_history(tmp_path):
    m = api()
    first_started = Event()
    release_first = Event()
    order = []
    jobs = [
        m.Job("busy", "x", tmp_path / "busy", 2, 2),
        m.Job("a1", "a", tmp_path / "a1", 2, 2),
        m.Job("a2", "a", tmp_path / "a2", 1, 2),
    ]

    def run(job, ctx):
        if job.job_id == "busy":
            first_started.set()
            assert release_first.wait(2)
        order.append(job.job_id)
        return {}

    def release():
        assert first_started.wait(2)
        # a2 must not pass the blocked a1 while busy holds a reservation.
        assert not Event().wait(0.05)
        release_first.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        releaser = pool.submit(release)
        result = m.run_jobs(
            jobs, policy=policy(max_calls=3), out=tmp_path / "campaign", run_job=run
        )
        releaser.result()
    assert [r["status"] for r in result["jobs"]] == ["complete"] * 3
    assert order.index("a1") < order.index("a2")


@pytest.mark.parametrize("max_requests,token_cap", [(1, 20), (3, 5)])
def test_shared_limiter_blocks_request_and_token_oversubscription(max_requests, token_cap):
    m = api()
    limiter = m.RequestLimiter(max_in_flight=max_requests, max_in_flight_tokens=token_cap)
    waiting = Event()
    acquired = Event()

    def contender():
        waiting.set()
        with limiter.request(3):
            acquired.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with limiter.request(3):
            future = pool.submit(contender)
            assert waiting.wait(2)
            assert not acquired.wait(0.05)
            assert limiter.snapshot() == {"in_flight": 1, "in_flight_tokens": 3}
        future.result(timeout=2)
    assert acquired.is_set()
    assert limiter.snapshot() == {"in_flight": 0, "in_flight_tokens": 0}


def test_prior_spend_debits_original_caps_once_and_keeps_legacy_snapshot():
    m = api()
    now = [100.0]

    def clock():
        return now[0]

    legacy = m.CampaignBudget(max_calls=5, max_seconds=20, clock=clock)
    assert legacy.snapshot() == {"calls": 0, "elapsed_seconds": 0}
    budget = m.CampaignBudget(
        max_calls=5, max_seconds=20, prior_calls=2, prior_elapsed_seconds=7, clock=clock
    )
    assert budget.max_seconds == 20
    assert budget.calls == 2
    assert budget.deadline == 113
    assert budget.remaining() == 13
    now[0] += 3
    budget.begin_call()
    assert budget.remaining() == 10
    assert budget.snapshot() == {
        "calls": 3,
        "elapsed_seconds": 10,
        "prior_calls": 2,
        "new_calls": 1,
        "prior_elapsed_seconds": 7,
        "new_elapsed_seconds": 3,
    }
    now[0] = 113
    with pytest.raises(m.CampaignStopped):
        budget.check()


@pytest.mark.parametrize(
    "field,value",
    [
        ("prior_calls", -1),
        ("prior_calls", True),
        ("prior_calls", 1.5),
        ("prior_calls", float("nan")),
        ("prior_calls", float("inf")),
        ("prior_calls", 6),
        ("prior_elapsed_seconds", -1),
        ("prior_elapsed_seconds", True),
        ("prior_elapsed_seconds", float("nan")),
        ("prior_elapsed_seconds", float("inf")),
        ("prior_elapsed_seconds", 21),
    ],
)
def test_prior_spend_rejects_invalid_or_over_cap_values(field, value):
    with pytest.raises(ValueError, match=field):
        api().CampaignBudget(max_calls=5, max_seconds=20, **{field: value})


def test_prior_calls_at_cap_block_further_calls_without_charging():
    m = api()
    budget = m.CampaignBudget(max_calls=5, max_seconds=20, prior_calls=5)
    budget.check()  # Checking a completed last response remains permitted.
    with pytest.raises(m.CampaignStopped, match="call allowance"):
        budget.begin_call()
    assert budget.calls == 5


def test_external_budget_preserves_setup_spend_and_shared_guard_after_local_fault(tmp_path):
    m = api()
    now = [100.0]

    def clock():
        return now[0]

    budget = m.CampaignBudget(
        max_calls=6, max_seconds=20, prior_calls=2, prior_elapsed_seconds=5, clock=clock
    )
    budget.begin_call()  # Driver setup uses the SAME guard before scheduling.
    now[0] += 3
    original_deadline = budget.deadline
    seen = []

    def run(job, ctx):
        assert ctx.budget is budget
        assert ctx.budget.deadline == original_deadline
        seen.append(job.job_id)
        with ctx.request(1) as remaining:
            assert remaining == 2
            now[0] += 1
            if job.job_id == "bad":
                raise ValueError("local response schema fault")
        return {"ok": True}

    result = m.run_jobs(
        [m.Job(name, name, tmp_path / name, 2, 2) for name in ("bad", "healthy", "unrun")],
        policy=policy(max_workers=1, max_calls=6, max_seconds=20),
        out=tmp_path / "campaign",
        run_job=run,
        budget=budget,
    )
    assert seen == ["bad", "healthy"]
    assert [row["status"] for row in result["jobs"]] == ["failed", "complete", "unrun"]
    assert budget.failed is False
    assert budget.reserved_calls == 0
    assert budget.calls == 5
    assert result["accounting"] == {
        "calls": 5,
        "elapsed_seconds": 10,
        "prior_calls": 2,
        "new_calls": 3,
        "prior_elapsed_seconds": 5,
        "new_elapsed_seconds": 5,
    }
    assert json.loads((tmp_path / "campaign" / "progress.json").read_bytes()) == result


@pytest.mark.parametrize(
    "conflict", ["calls", "seconds", "clock", "type", "expired", "failed", "exhausted"]
)
def test_external_budget_refuses_conflicts_and_exhaustion_before_outputs(tmp_path, conflict):
    m = api()
    now = [0.0]

    def clock():
        return now[0]

    budget = m.CampaignBudget(max_calls=5, max_seconds=10, clock=clock)
    options = {}
    p = policy(max_calls=5, max_seconds=10)
    if conflict == "calls":
        p = policy(max_calls=4)
    elif conflict == "seconds":
        p = policy(max_calls=5, max_seconds=9)
    elif conflict == "clock":
        options["clock"] = lambda: now[0]
    elif conflict == "type":
        budget = object()
    elif conflict == "expired":
        now[0] = 10
    elif conflict == "failed":
        budget.failed = True
    elif conflict == "exhausted":
        budget = m.CampaignBudget(max_calls=5, max_seconds=10, prior_calls=5, clock=clock)
    expected = (
        ValueError if conflict in ("calls", "seconds", "clock", "type") else m.CampaignStopped
    )
    with pytest.raises(expected):
        m.run_jobs(
            [m.Job("j", "h", tmp_path / "job", 1, 1)],
            policy=p,
            out=tmp_path / "campaign",
            run_job=lambda *_: pytest.fail("must not execute"),
            budget=budget,
            **options,
        )
    assert list(tmp_path.iterdir()) == []


def test_external_budget_counts_setup_time_and_checks_own_clock_at_dispatch(tmp_path):
    m = api()
    now = [50.0]

    def clock():
        return now[0]

    budget = m.CampaignBudget(max_calls=5, max_seconds=10, prior_elapsed_seconds=3, clock=clock)
    now[0] += 2

    def run(job, ctx):
        assert ctx.budget is budget
        assert ctx.deadline == 57
        now[0] = 57
        with ctx.request(1):
            pytest.fail("expired dispatch")

    result = m.run_jobs(
        [m.Job("j", "h", tmp_path / "job", 1, 5)],
        policy=policy(max_calls=5, max_seconds=10),
        out=tmp_path / "campaign",
        run_job=run,
        budget=budget,
        clock=clock,
    )
    assert result["jobs"][0]["status"] == "failed"
    assert result["accounting"]["calls"] == 0
    assert result["accounting"]["elapsed_seconds"] == 10


def test_external_budget_rechecks_deadline_after_planning_before_outputs(tmp_path):
    m = api()
    now = [0.0]
    budget = m.CampaignBudget(max_calls=5, max_seconds=10, clock=lambda: now[0])

    def jobs():
        yield m.Job("j", "h", tmp_path / "job", 1, 1)
        now[0] = 10

    with pytest.raises(m.CampaignStopped):
        m.run_jobs(
            jobs(),
            policy=policy(max_calls=5),
            out=tmp_path / "campaign",
            run_job=lambda *_: pytest.fail("must not execute"),
            budget=budget,
        )
    assert list(tmp_path.iterdir()) == []


def api():
    return importlib.import_module("gluomem_bench.bench.parallel_campaign")


def test_request_limiter_bounds_tokens_and_releases_after_failure():
    m = api()
    for requests, tokens in [(0, 1), (501, 1), (True, 1), (1, 0)]:
        with pytest.raises(ValueError):
            m.RequestLimiter(max_in_flight=requests, max_in_flight_tokens=tokens)
    limiter = m.RequestLimiter(max_in_flight=2, max_in_flight_tokens=6)
    with pytest.raises(RuntimeError):
        with limiter.request(4):
            raise RuntimeError("transport failure")
    gate = Barrier(2)
    lock = Lock()
    active = peak = 0

    def work():
        nonlocal active, peak
        with limiter.request(3):
            with lock:
                active += 1
                peak = max(peak, active)
            gate.wait(timeout=2)
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: work(), range(2)))
    assert peak == 2
    assert limiter.snapshot()["in_flight"] == 0
    assert limiter.snapshot()["in_flight_tokens"] == 0
    for tokens in (0, -1, 7, True):
        with pytest.raises(ValueError):
            with limiter.request(tokens):
                pass
