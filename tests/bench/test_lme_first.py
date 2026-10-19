"""Offline readiness contracts: authored inputs only; no quality claims."""

import copy
import hashlib
import importlib
import importlib.util
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.fixture(autouse=True)
def offline_boundary(monkeypatch):
    """Synthetic readiness may never reach the real model/shared encoder."""
    import socket
    import subprocess

    attempts = []

    def deny(*args, **kwargs):
        attempts.append("denied")
        raise OSError("offline readiness denies all sockets and child processes")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(subprocess, "Popen", deny)
    yield
    assert attempts == [], "test attempted network/process escape"


def module():
    name = "gluomem_bench.bench.lme_first"
    assert importlib.util.find_spec(name) is not None, "LME-first preparation is missing"
    return importlib.import_module(name)


def test_global_300_not_per_framework_and_real_token_reservations():
    m = module()
    governor = m.FairRequestLimiter(max_in_flight=300, max_in_flight_tokens=300 * 131072)
    release = threading.Event()
    reached = threading.Event()
    lock = threading.Lock()
    entered = 0
    peak = 0

    def work(i):
        nonlocal entered, peak
        with governor.request(131072):
            with lock:
                entered += 1
                peak = max(peak, governor.snapshot()["in_flight"])
                if entered == 300:
                    reached.set()
            assert release.wait(15)

    with ThreadPoolExecutor(max_workers=303) as pool:
        futures = [pool.submit(work, i) for i in range(303)]
        try:
            assert reached.wait(15), governor.snapshot()
            time.sleep(0.05)
            assert entered == 300
            assert governor.snapshot()["in_flight_tokens"] == 39321600
        finally:
            release.set()
        for future in futures:
            future.result()
    assert entered == 303
    assert peak == 300
    assert governor.snapshot()["in_flight"] == 0
    assert governor.snapshot()["peak_in_flight"] == 300


def test_fifo_prevents_small_request_barging_and_cancelled_head_releases():
    m = module()
    limiter = m.FairRequestLimiter(max_in_flight=300, max_in_flight_tokens=10)
    order = []
    cancel = threading.Event()

    def run(label, tokens, cancelled=None):
        try:
            with limiter.request(tokens, cancel=cancelled):
                order.append(label)
        except m.DispatchStopped:
            order.append("cancelled")

    with ThreadPoolExecutor(max_workers=2) as pool:
        with limiter.request(8):
            large = pool.submit(run, "large", 9, cancel)
            limit = time.monotonic() + 3
            while limiter.snapshot()["waiting"] != 1 and time.monotonic() < limit:
                time.sleep(0.001)
            small = pool.submit(run, "small", 1)
            time.sleep(0.03)
            assert order == []
            cancel.set()
            large.result(timeout=3)
            small.result(timeout=3)
    assert order == ["cancelled", "small"]


def source_spec(tmp_path):
    ids = [f"authored-{i}" for i in range(500)]
    s = {
        "dataset": {"name": "longmemeval_s", "path": "not-opened.json", "sha256": "d" * 64},
        "sample_ids": ids,
        "profile": {
            "context_window": 131072,
            "max_generation_calls": 16384,
            "max_run_seconds": 21600,
            "timeout_seconds": 600,
            "model": "authored/model",
            "base_url": "http://example.invalid/v1",
            "generation_budgets": {"extraction": 16384, "reader": 4096, "judge": 4096},
        },
        "systems": ["gluomem", "mem0", "no_memory", "bm25", "dense", "hybrid", "full_context"],
        "jobs": [
            {
                "job_id": f"old-{i}-{sys}",
                "sample_id": id_,
                "system": sys,
                "prior_row": {"status": "unrun"},
            }
            for i, id_ in enumerate(ids)
            for sys in ["gluomem", "mem0", "no_memory", "bm25", "dense", "hybrid", "full_context"]
        ],
        "cognee_unrun": [{"sample_id": id_, "system": "cognee", "status": "unrun"} for id_ in ids],
        "source_identifier_policy": "opaque-ordinal-turns@v1",
        "evidence_token_budget": 4096,
        "config_overrides": {},
        "judge_source": "authored-scorer.py",
        "embedding_url": "http://example.invalid/encoder",
        "embedding_recipe": {"model": "authored"},
    }
    p = tmp_path / "source.json"
    p.write_bytes(json.dumps(s).encode())
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def test_coordinator_300_overlap_exact_journal_and_local_failure(tmp_path):
    m = module()
    jobs = [
        {
            "job_id": f"j{i}",
            "history_id": f"h{i}",
            "system": ("gluomem", "mem0", "cognee")[i % 3],
            "output": str(tmp_path / "leaves" / str(i)),
            "call_budget": 3,
            "seconds_budget": 30,
            "memory_mib": 1,
        }
        for i in range(303)
    ]
    active = 0
    peak = 0
    lock = threading.Lock()
    barrier = threading.Barrier(300)

    def work(job, context):
        nonlocal active, peak
        # Native path owns begin_call; never combine context.request and it.
        with context.lease(131072):
            context.begin_call()
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                if int(job.job_id[1:]) < 300:
                    barrier.wait(timeout=15)
                if job.job_id == "j1":
                    raise ValueError("PRIVATE-BODY-DO-NOT-PERSIST")
            finally:
                with lock:
                    active -= 1
        # Plain reader and judge, each charged once too.
        for _ in range(2):
            with context.request(131072):
                pass
        return {"saved": job.job_id}

    report = m.run_isolated_jobs(
        jobs,
        out=tmp_path / "coordination",
        run_job=work,
        max_calls=909,
        max_seconds=60,
        max_workers=303,
        native_limits={"gluomem": 101, "mem0": 101, "cognee": 101},
        memory_limit_mib=303,
        max_in_flight_tokens=39321600,
    )
    assert peak == 300
    assert report["limiter"]["peak_in_flight"] == 300
    assert report["accounting"]["calls"] == 907
    assert report["jobs"][1]["status"] == "failed"
    assert report["jobs"][-1]["status"] == "complete"
    assert report["quality_complete"] is False
    assert report["checkpoint_writes"] == 2  # no per-request whole matrix writes
    assert m.verify_journal(tmp_path / "coordination", jobs) == report
    for p in (tmp_path / "coordination").glob("*"):
        assert b"PRIVATE-BODY" not in p.read_bytes()
    material = (tmp_path / "coordination" / "journal.jsonl").read_bytes()
    records = [json.loads(line) for line in material.splitlines()]
    assert sum(r["event"] == "dispatch" for r in records) == 907
    # Rehashed forgery still fails structural/accounting replay.
    for mutate in [
        lambda r: r.pop(5),
        lambda r: r.insert(5, copy.deepcopy(r[5])),
        lambda r: r[5].update(job_id="unknown-job"),
    ]:
        forged = copy.deepcopy(records)
        mutate(forged)
        with pytest.raises(ValueError):
            m.replay_journal(forged, jobs)


def test_three_bounded_history_workers_can_share_300_pipeline_leases(tmp_path):
    m = module()
    barrier = threading.Barrier(300)
    jobs = [
        {
            "job_id": f"j{i}",
            "history_id": f"h{i}",
            "system": ("gluomem", "mem0", "cognee")[i],
            "output": str(tmp_path / "leaves" / str(i)),
            "call_budget": 101,
            "seconds_budget": 30,
            "memory_mib": 1,
        }
        for i in range(3)
    ]

    def run(job, context):
        def pipeline(i):
            with context.request(131072):
                if i < 100:
                    barrier.wait(timeout=15)

        with ThreadPoolExecutor(max_workers=101) as pool:
            list(pool.map(pipeline, range(101)))
        return {}

    result = m.run_isolated_jobs(
        jobs,
        out=tmp_path / "coord",
        run_job=run,
        max_calls=303,
        max_seconds=60,
        max_workers=3,
        native_limits={"gluomem": 1, "mem0": 1, "cognee": 1},
        memory_limit_mib=3,
        max_in_flight_tokens=39321600,
    )
    assert result["accounting"]["calls"] == 303
    assert result["limiter"]["peak_in_flight"] == 300
    assert result["peak_memory_mib"] == 3
    assert [row["calls"] for row in result["jobs"]] == [101, 101, 101]
    assert [row["status"] for row in result["jobs"]] == ["complete"] * 3


def test_coordinator_history_order_clock_local_recovery_and_memory_slots(tmp_path):
    m = module()
    jobs = [
        {
            "job_id": f"j{i}",
            "history_id": "same" if i < 2 else "other",
            "system": "mem0",
            "output": str(tmp_path / "leaves" / str(i)),
            "call_budget": 1,
            "seconds_budget": 5,
            "memory_mib": 4,
        }
        for i in range(3)
    ]
    seen = []

    def run(job, context):
        seen.append((job.job_id, "begin", context.deadline))
        with context.request(1):
            time.sleep(0.01)
        seen.append((job.job_id, "end", context.deadline))
        return {}

    result = m.run_isolated_jobs(
        jobs,
        out=tmp_path / "coord",
        run_job=run,
        max_calls=3,
        max_seconds=10,
        max_workers=3,
        native_limits={"mem0": 3},
        memory_limit_mib=8,
        max_in_flight_tokens=3,
    )
    assert result["accounting"]["calls"] == 3
    assert result["peak_memory_mib"] == 8
    assert [(s[0], s[1]) for s in seen].index(("j0", "end")) < [(s[0], s[1]) for s in seen].index(
        ("j1", "begin")
    )
    assert all(seen[i][2] == seen[i + 1][2] for i in [0] if seen[i][0] == seen[i + 1][0])


@pytest.mark.parametrize("fault", ["identity", "tokenizer", "double-count"])
def test_shared_fault_stops_queued_jobs_without_reset(tmp_path, fault):
    m = module()
    jobs = [
        {
            "job_id": f"j{i}",
            "history_id": f"h{i}",
            "system": "mem0",
            "output": str(tmp_path / "leaves" / str(i)),
            "call_budget": 2,
            "seconds_budget": 5,
            "memory_mib": 1,
        }
        for i in range(3)
    ]
    dispatched = []

    def run(job, context):
        dispatched.append(job.job_id)
        if fault == "double-count":
            with context.request(1):
                context.begin_call()
        else:
            import httpx

            from gluomem_bench.bench.rack import RackClient, RackProfile

            p = RackProfile(
                base_url="http://example.invalid/v1",
                model="authored/model",
                profile_version="role-budgets-v6",
                generation_budgets={
                    "extraction": 512,
                    "consolidation": 512,
                    "reader": 512,
                    "judge": 512,
                },
            )

            def serve(request):
                if fault == "identity":
                    return httpx.Response(200, json={"data": [{"id": "wrong-model"}]})
                return httpx.Response(200, json={"count": -1})

            client = RackClient(
                p,
                receipt=tmp_path / "wire.jsonl",
                budget=context,
                transport=httpx.MockTransport(serve),
            )
            try:
                if fault == "identity":
                    client.preflight()
                else:
                    client.prompt_token_count([{"role": "user", "content": "authored"}])
            finally:
                client.close()

    report = m.run_isolated_jobs(
        jobs,
        out=tmp_path / "coord",
        run_job=run,
        max_calls=6,
        max_seconds=10,
        max_workers=1,
        native_limits={"mem0": 1},
        memory_limit_mib=1,
        max_in_flight_tokens=3,
    )
    assert dispatched == ["j0"]
    assert [r["status"] for r in report["jobs"]] == ["failed", "unrun", "unrun"]
    assert report["accounting"]["calls"] == (1 if fault == "double-count" else 0)


def test_native_plain_structured_production_share_actual_outer_once(tmp_path, monkeypatch):
    m = module()
    import dspy
    import httpx

    from gluomem_bench.bench.rack import RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import RackLM, WorkloadBudget

    jobs = [
        {
            "job_id": f"j{i}",
            "history_id": f"h{i}",
            "system": "gluomem",
            "output": str(tmp_path / "leaves" / str(i)),
            "call_budget": 3,
            "seconds_budget": 30,
            "memory_mib": 1,
        }
        for i in range(3)
    ]
    p = RackProfile(base_url="http://example.invalid/v1", model="authored/model")

    def provider(lm, *args, **kwargs):
        assert lm.budget.outer_budget.limiter.snapshot()["in_flight"] > 0
        lm.history.append(
            {
                "response": {
                    "model": p.model,
                    "choices": [{"finish_reason": "stop", "message": {"content": "ready"}}],
                }
            }
        )
        return ["ready"]

    monkeypatch.setattr(dspy.LM, "__call__", provider)

    def work(job, context):
        budget = WorkloadBudget(outer_budget=context)

        def serve(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 20})
            assert context.limiter.snapshot()["in_flight"] > 0
            return httpx.Response(
                200,
                json={
                    "model": p.model,
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": '{"answer":"ready"}'}}
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5},
                },
            )

        client = RackClient(
            p,
            receipt=tmp_path / (job.job_id + "-wire.jsonl"),
            budget=budget,
            transport=httpx.MockTransport(serve),
        )
        try:
            client.generate_messages("answer", [{"role": "user", "content": "authored"}])
            client.complete("answer", {"question": "authored"}, {"answer": {"type": "string"}})
            lm = RackLM(
                "openai/" + p.model,
                client=client,
                task="event_claims",
                trace=tmp_path / (job.job_id + "-raw.jsonl"),
                budget=budget,
                api_base=p.base_url,
                max_tokens=512,
                temperature=0,
                num_retries=0,
                cache=False,
            )
            lm(messages=[{"role": "user", "content": "authored"}])
            assert budget.calls == context.calls == 3
            return {}
        finally:
            client.close()

    report = m.run_isolated_jobs(
        jobs,
        out=tmp_path / "coord",
        run_job=work,
        max_calls=9,
        max_seconds=60,
        max_workers=3,
        native_limits={"gluomem": 3},
        memory_limit_mib=3,
        max_in_flight_tokens=39321600,
    )
    assert [r["status"] for r in report["jobs"]] == ["complete"] * 3
    assert report["accounting"]["calls"] == 9


def test_journal_rejects_resequenced_tokens_summary_and_clock_forgeries(tmp_path):
    m = module()
    jobs = [
        {
            "job_id": "j",
            "history_id": "h",
            "system": "mem0",
            "output": str(tmp_path / "leaf"),
            "call_budget": 1,
            "seconds_budget": 5,
            "memory_mib": 1,
        }
    ]

    def run(job, context):
        with context.request(2):
            pass
        return {}

    m.run_isolated_jobs(
        jobs,
        out=tmp_path / "coord",
        run_job=run,
        max_calls=1,
        max_seconds=10,
        max_workers=1,
        native_limits={"mem0": 1},
        memory_limit_mib=1,
        max_in_flight_tokens=3,
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "coord" / "journal.jsonl").read_bytes().splitlines()
    ]
    mutations = [
        lambda e: e[1].update(tokens=4),
        lambda e: e[-1]["report"]["limiter"].update(peak_in_flight=301),
        lambda e: e[0].update(deadline=e[0]["started"] + 6),
        lambda e: e[-1]["report"]["jobs"].append(copy.deepcopy(e[-1]["report"]["jobs"][0])),
    ]
    for mutate in mutations:
        forged = copy.deepcopy(events)
        mutate(forged)
        with pytest.raises(ValueError):
            m.replay_journal(forged, jobs)


def test_future_clock_includes_final_publication(tmp_path, monkeypatch):
    m = module()
    import os

    from gluomem_bench.bench import parallel_campaign

    clock = [0.0]
    real_budget = parallel_campaign.CampaignBudget
    monkeypatch.setattr(
        parallel_campaign, "CampaignBudget", lambda **kw: real_budget(**kw, clock=lambda: clock[0])
    )
    real_replace = os.replace
    writes = [0]

    def delayed_publication(source, destination):
        real_replace(source, destination)
        writes[0] += 1
        if writes[0] == 2:
            clock[0] = 11.0

    monkeypatch.setattr(os, "replace", delayed_publication)
    jobs = [
        {
            "job_id": "j",
            "history_id": "h",
            "system": "mem0",
            "output": str(tmp_path / "leaf"),
            "call_budget": 1,
            "seconds_budget": 5,
            "memory_mib": 1,
        }
    ]

    def work(job, context):
        with context.request(1):
            pass
        return {}

    with pytest.raises(m.DispatchStopped) as caught:
        m.run_isolated_jobs(
            jobs,
            out=tmp_path / "coord",
            run_job=work,
            max_calls=1,
            max_seconds=10,
            max_workers=1,
            native_limits={"mem0": 1},
            memory_limit_mib=1,
            max_in_flight_tokens=3,
        )
    assert caught.value.accounting["calls"] == 1
    assert caught.value.accounting["elapsed_seconds"] == 11
    incomplete = json.loads((tmp_path / "coord" / "coordinator-incomplete.json").read_bytes())
    assert incomplete["accounting"]["calls"] == 1
    assert incomplete["accounting"]["elapsed_seconds"] == 11
    with pytest.raises(ValueError):
        m.verify_journal(tmp_path / "coord", jobs)
    assert (
        json.loads((tmp_path / "coord" / "progress.json").read_bytes())["quality_complete"] is False
    )


def test_cli_freezes_verifies_and_refuses_inference_or_overwrite(tmp_path, capsys):
    m = module()
    source, digest = source_spec(tmp_path)
    out = tmp_path / "proposal.json"
    args = [
        "prepare",
        "--source-spec",
        str(source),
        "--source-spec-sha256",
        digest,
        "--future-max-calls",
        "32768000",
        "--future-max-seconds",
        "172800",
        "--local-memory-mib",
        "12288",
        "--out",
        str(out),
    ]
    assert hasattr(m, "main"), "preparation CLI missing"
    assert m.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["execution_authorized"] is False
    assert result["plan_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert m.main(["verify", "--plan", str(out), "--sha256", result["plan_sha256"]]) == 0
    capsys.readouterr()
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        m.main(args)
    assert out.read_bytes() == before
    with pytest.raises(SystemExit):
        m.main(["execute", "--plan", str(out)])


def test_primary_phase_drains_before_control(tmp_path):
    m = module()
    jobs = [
        {
            "job_id": f"j{i}",
            "history_id": f"h{i}",
            "system": "mem0" if i < 2 else "no_memory",
            "phase": "primary" if i < 2 else "control-after-primary",
            "output": str(tmp_path / "leaves" / str(i)),
            "call_budget": 1,
            "seconds_budget": 5,
            "memory_mib": 1,
        }
        for i in range(3)
    ]
    seen = []

    def run(job, context):
        seen.append(job.job_id + "-start")
        with context.request(1):
            time.sleep(0.01)
        seen.append(job.job_id + "-end")
        return {}

    m.run_isolated_jobs(
        jobs,
        out=tmp_path / "coord",
        run_job=run,
        max_calls=3,
        max_seconds=10,
        max_workers=3,
        native_limits={"mem0": 1, "no_memory": 1},
        memory_limit_mib=3,
        max_in_flight_tokens=3,
    )
    assert seen.index("j1-end") < seen.index("j2-start")


def test_fresh_full500_matched_proposal_never_resets_historical_budget(tmp_path):
    m = module()
    source, digest = source_spec(tmp_path)
    before = source.read_bytes()
    plan = m.build_plan(
        source_spec=source,
        source_spec_sha256=digest,
        future_max_calls=32768000,
        future_max_seconds=172800,
        local_memory_mib=12288,
    )
    assert plan["execution_authorized"] is False
    assert plan["new_experiment"] is True
    assert plan["operating_point"]["max_in_flight"] == 300
    assert plan["operating_point"]["max_in_flight_tokens"] == 39321600
    assert len(plan["jobs"]) == 2000
    assert [r["system"] for r in plan["jobs"][:3]] == ["gluomem", "mem0", "cognee"]
    assert all(r["system"] == "no_memory" for r in plan["jobs"][1500:])
    assert plan["future_budget_proposal"]["full_call_envelopes"] == 32768000
    assert plan["future_budget_proposal"]["may_leave_unrun"] is True
    assert plan["future_budget_proposal"]["call_budget_covers_all_full_envelopes"] is True
    assert plan["historical_accounting"]["observed_generation_calls"] == 8225
    assert plan["historical_accounting"]["unreceipted_calls"] is None
    assert plan["historical_accounting"]["pre_clock_setup_seconds"] is None
    assert len(plan["retained_old_matrix"]) == 4000
    assert source.read_bytes() == before
    assert m.validate_plan(plan) == 2000
    assert "native-concurrency-qualification" in plan["blockers"]
    assert plan["native_constraints"]["serial_native_request_upper_bound"] == 30
    assert (
        plan["history_execution_contract"]["session_commit_order"]
        == "chronological; stable original-index tie break"
    )
    bad = copy.deepcopy(plan)
    bad["jobs"][0]["sample_id"] = bad["jobs"][3]["sample_id"]
    with pytest.raises(ValueError):
        m.validate_plan(bad)
    for change in [
        lambda p: p["jobs"].pop(),
        lambda p: p["operating_point"].update(max_in_flight_tokens=262144),
        lambda p: p.update(execution_authorized=True),
        lambda p: p["profile"].update(timeout_seconds=601),
    ]:
        bad = copy.deepcopy(plan)
        change(bad)
        with pytest.raises(ValueError):
            m.validate_plan(bad)
    with pytest.raises(ValueError):
        m.build_plan(
            source_spec=source,
            source_spec_sha256="0" * 64,
            future_max_calls=32768000,
            future_max_seconds=172800,
            local_memory_mib=12288,
        )
