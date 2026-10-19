"""Offline versioned NEVER-RUN continuation contracts; no live dispatch."""

import copy
import json
from pathlib import Path

import pytest

from gluomem_bench.bench import arena_travel_driver as travel

ROOT = Path('tests/fixtures/native/operator-supplied')
LEDGER = ROOT / "analysis/travel-v3-poststop-ledger-v1.json"
LEDGER_SHA = "2b7bf7a16614df23a1230d97d2f06db8387188b2251bbbe7867002e88015a4f3"


def ledger():
    return json.loads(LEDGER.read_bytes())


def original():
    return json.loads((Path(ledger()["original_root"]) / "execution-spec.json").read_bytes())


@pytest.mark.parametrize(
    "attack", ["missing", "duplicate", "escalate", "fake_calls", "forged_spend", "overflow"]
)
def test_ledger_forgery_is_rejected(attack):
    d = ledger()
    if attack == "missing":
        d["rows"].pop()
    elif attack == "duplicate":
        d["rows"][110] = copy.deepcopy(d["rows"][109])
    elif attack == "escalate":
        d["rows"][0]["disposition"] = travel.NEVER_RUN
    elif attack == "fake_calls":
        d["rows"][109]["spend"]["calls"] = 1
    elif attack == "overflow":
        d["rows"][0]["spend"]["calls"] = 513
    else:
        d["spend"]["calls"] -= 1
    with pytest.raises(ValueError):
        travel.continuation_plan(d, original())


def candidate(tmp_path):
    runtime = original()
    runtime["output_root"] = str(tmp_path / "new-root")
    runtime["files"] = travel.formal._code_inventory()
    return travel.build_continuation_spec(
        runtime=runtime,
        ledger_path=LEDGER,
        ledger_sha256=LEDGER_SHA,
        reserve_proposal_path=ROOT
        / "analysis/travel-continuation-conservative-reserve-proposal-v1.json",
    )


def save_approvals(tmp_path, spec):
    sp = tmp_path / "spec.json"
    sp.write_bytes(travel.common._json(spec))
    reserve = {
        "approved": True,
        "reviewer_kind": "parent",
        "spec_sha256": travel.file_record(sp)["sha256"],
        "ledger_sha256": spec["ledger"]["sha256"],
        "reserve_proposal_sha256": spec["reserve_proposal"]["sha256"],
        "allowance_debit": spec["plan"]["allowance_debit"],
        "original_guards_verified": True,
        "no_transport_bypass": True,
        "unknown_attempts_bounded_by_original_guards": True,
        "semantic_bridge_verified": True,
        "semantic_bridge_sha256": travel.digest(spec["semantic_bridge"]),
    }
    rp = tmp_path / "reserve-approval.json"
    rp.write_bytes(travel.common._json(reserve))
    ap = tmp_path / "execution-approval.json"
    ap.write_bytes(
        travel.common._json(
            {
                "approved": True,
                "reviewer_kind": "parent",
                "spec_sha256": travel.file_record(sp)["sha256"],
                "reserve_approval_sha256": travel.file_record(rp)["sha256"],
            }
        )
    )
    return sp, ap, rp


def test_candidate_requires_separate_byte_bound_parent_reserve(tmp_path):
    spec = candidate(tmp_path)
    sp, ap, rp = save_approvals(tmp_path, spec)
    rp.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        travel.verify_continuation_approval(sp, ap, rp)
    assert not Path(spec["runtime"]["output_root"]).exists()


def test_separate_reserve_refuses_unguarded_unknown_attempts(tmp_path):
    spec = candidate(tmp_path)
    sp, ap, rp = save_approvals(tmp_path, spec)
    reserve = json.loads(rp.read_bytes())
    reserve["no_transport_bypass"] = False
    rp.write_bytes(travel.common._json(reserve))
    approval = json.loads(ap.read_bytes())
    approval["reserve_approval_sha256"] = travel.file_record(rp)["sha256"]
    ap.write_bytes(travel.common._json(approval))
    with pytest.raises(ValueError, match="reserve"):
        travel.verify_continuation_approval(sp, ap, rp)


def test_bound_spec_rebuild_refuses_forged_ledger_and_replays_old_leaf(tmp_path, monkeypatch):
    spec = candidate(tmp_path)
    sp, ap, rp = save_approvals(tmp_path, spec)
    assert travel.verify_continuation_spec(sp, ap, rp, verify_runtime=False) == spec
    forged = copy.deepcopy(spec)
    forged["plan"]["dispatch_indices"].append(0)
    sp, ap, rp = save_approvals(tmp_path, forged)
    with pytest.raises(ValueError, match="rebuild"):
        travel.verify_continuation_spec(sp, ap, rp, verify_runtime=False)


def test_executable_subset_local_failure_then_healthy_then_global_stop(tmp_path):
    from gluomem_bench.bench.parallel_campaign import CampaignBudget, CampaignStopped

    spec = candidate(tmp_path)
    sp, ap, rp = save_approvals(tmp_path, spec)
    rows = travel.load_rows(spec["runtime"]["data"]["path"], spec["runtime"]["data"]["sha256"])
    budget = CampaignBudget(max_calls=773632, max_seconds=32637600)
    seen = []

    def run(raw, system, ctx, target):
        assert ctx.budget is budget
        assert ctx.budget.max_calls == 773632
        assert ctx.budget.deadline == budget.deadline
        ctx.begin_call()
        seen.append((raw["id"], system))
        if system == "full_context":
            target.mkdir(parents=True)
            (target / "input-task.json").write_bytes(travel.common._json(raw))
            (target / "effective-profile.json").write_bytes(
                travel.common._json(spec["runtime"]["profile"])
            )
            (target / "http.jsonl").write_bytes(
                json.dumps(
                    {
                        "url": "https://authored.invalid/tokenize",
                        "status": 200,
                        "response": {"count": 131072, "max_model_len": 131072},
                    }
                ).encode()
                + b"\n"
            )
            (target / "failure.json").write_bytes(
                travel.common._json(
                    {
                        "status": "not_applicable",
                        "calls": ctx.calls,
                        "native_closed": True,
                        "completed_sessions": 0,
                        "expected_sessions": len(raw["questions"]),
                    }
                )
            )
            ctx.failed = True  # The actual tokenizer overflow latches the local transport.
            return {"status": "not_applicable"}
        if system == "gluomem":
            budget.failed = True
            raise CampaignStopped("authored true shared identity failure")
        raise ValueError("authored local quality fault")

    result = travel._execute_continuation_matrix(
        spec, rows, budget=budget, run_one=run, spec_paths=(sp, ap, rp)
    )
    assert seen == [(19, s) for s in ("bm25", "dense", "hybrid", "full_context", "gluomem")]
    assert len(result["tasks"]) == 1620
    assert result["expected_sessions"] == 11214
    assert result["calls"] == 1016
    assert result["accounting"]["allowance_consumed_calls"] == 55813
    assert result["accounting"]["observed_prior"]["calls"] == 1011
    assert result["accounting"]["new_actual"]["calls"] == 5
    assert result["accounting"]["unknown_prior_attempts"] is None
    assert result["quality_complete"] is False
    assert result["tasks"][0]["disposition"] == "quality_failed_latched_no_retry"
    assert result["tasks"][0]["calls"] == 30
    assert result["tasks"][112]["status"] == "not_applicable"
    assert all(r["status"] == "unrun" for r in result["tasks"][114:])
    root = Path(spec["runtime"]["output_root"])
    assert travel.verify_continuation_root(root)["verified"] is True
    assert result["metrics"]["full_context"]["complete_tasks"] == 1
    assert result["metrics"]["full_context"]["full_metrics"] is None


def test_live_entry_refuses_missing_reserve_before_any_setup(tmp_path, monkeypatch):
    spec = candidate(tmp_path)
    sp, ap, rp = save_approvals(tmp_path, spec)
    rp.unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("setup/client/tool constructed without reserve approval")

    monkeypatch.setattr(travel, "sources", forbidden)
    monkeypatch.setattr(travel, "NativeTools", forbidden)
    with pytest.raises(FileNotFoundError):
        travel.execute_continuation(sp, ap, rp)
    assert not Path(spec["runtime"]["output_root"]).exists()


def test_live_entry_setup_is_inside_reduced_outer_guard_and_retains_failure(tmp_path, monkeypatch):
    spec = candidate(tmp_path)
    sp, ap, rp = save_approvals(tmp_path, spec)

    def failed_setup(*args, **kwargs):
        b = kwargs["budget"]
        assert (b.max_calls, b.max_seconds, b.calls) == (773632, 32637600, 0)
        assert b.prior_calls == 0
        raise ValueError("authored setup failure")

    monkeypatch.setattr(travel, "verify_continuation_spec", failed_setup)
    with pytest.raises(ValueError, match="authored setup"):
        travel.execute_continuation(sp, ap, rp)
    root = Path(spec["runtime"]["output_root"])
    receipt = json.loads((root / "campaign-failure.json").read_bytes())
    assert receipt["calls"] == 1011
    assert receipt["accounting"]["reserved_prior_allowance"]["calls"] == 55808
    assert receipt["accounting"]["new_actual"]["calls"] == 0
    assert len(receipt["tasks"]) == 1620
    assert not (root / "campaign-result.json").exists()


def test_continuation_permit_refuses_held_identity_and_rechecks_reserve(tmp_path):
    from gluomem_bench.bench.rack import RackProfile

    spec = candidate(tmp_path)
    sp, ap, rp = save_approvals(tmp_path, spec)
    rows = travel.load_rows(spec["runtime"]["data"]["path"], spec["runtime"]["data"]["sha256"])
    kwargs = dict(
        spec_path=sp,
        approval_path=ap,
        reserve_approval_path=rp,
        expected_sha256=travel.file_record(sp)["sha256"],
        approval_sha256=travel.file_record(ap)["sha256"],
        reserve_approval_sha256=travel.file_record(rp)["sha256"],
    )
    held = travel.FrozenTravelContinuationTask(**kwargs, system="no_memory")
    with pytest.raises(ValueError, match="never-run"):
        held.validate(rows[0], RackProfile.model_validate(spec["runtime"]["profile"]), 30)
    good = travel.FrozenTravelContinuationTask(**kwargs, system="bm25")
    assert (
        good.validate(rows[18], RackProfile.model_validate(spec["runtime"]["profile"]), 30)
        == spec["runtime"]
    )
    rp.write_bytes(b"{}")
    with pytest.raises(ValueError):
        good.check_approval()


def test_new_exact_permit_admits_real_transport_and_revocation_prevents_dispatch(tmp_path):
    import httpx

    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )
    from gluomem_bench.bench.rack import RackProfile

    spec = candidate(tmp_path)
    sp, ap, rp = save_approvals(tmp_path, spec)
    rows = travel.load_rows(spec["runtime"]["data"]["path"], spec["runtime"]["data"]["sha256"])
    permit = travel.FrozenTravelContinuationTask(
        sp,
        ap,
        rp,
        travel.file_record(sp)["sha256"],
        travel.file_record(ap)["sha256"],
        travel.file_record(rp)["sha256"],
        system="bm25",
    )
    b = CampaignBudget(max_calls=773632, max_seconds=32637600)
    j = Job("authored", "authored", tmp_path / "leaf", 512, 21600)
    assert b.reserve(j)
    ctx = JobContext(j, b, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))
    requests = []

    def http(request):
        requests.append(request)
        pytest.fail("revoked permit dispatched HTTP")

    client = travel.TravelTransport(
        RackProfile.model_validate(spec["runtime"]["profile"]),
        receipt=tmp_path / "http.jsonl",
        budget=ctx,
        tools=[],
        task=rows[18],
        max_steps=30,
        admission=permit,
        transport=httpx.MockTransport(http),
    )
    b.max_calls += 1
    with pytest.raises(ValueError, match="reduced"):
        client.check()
    b.max_calls -= 1
    rp.write_bytes(b"{}")
    with pytest.raises(ValueError):
        client.check()
    client.close()
    assert requests == []
    assert ctx.calls == 0


@pytest.mark.parametrize("caps", [(829440, 34995600), (773633, 32637600), (773632, 32637601)])
def test_continuation_transport_rejects_undebited_outer_guard(tmp_path, caps):
    import httpx

    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )
    from gluomem_bench.bench.rack import RackProfile

    spec = candidate(tmp_path)
    sp, ap, rp = save_approvals(tmp_path, spec)
    rows = travel.load_rows(spec["runtime"]["data"]["path"], spec["runtime"]["data"]["sha256"])
    permit = travel.FrozenTravelContinuationTask(
        sp,
        ap,
        rp,
        travel.file_record(sp)["sha256"],
        travel.file_record(ap)["sha256"],
        travel.file_record(rp)["sha256"],
        system="bm25",
    )
    b = CampaignBudget(max_calls=caps[0], max_seconds=caps[1])
    job = Job("bad-outer", "authored", tmp_path / "leaf", 512, 21600)
    assert b.reserve(job)
    context = JobContext(job, b, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))
    with pytest.raises(ValueError, match="reduced"):
        travel.TravelTransport(
            RackProfile.model_validate(spec["runtime"]["profile"]),
            receipt=tmp_path / "http.jsonl",
            budget=context,
            tools=[],
            task=rows[18],
            max_steps=30,
            admission=permit,
            transport=httpx.MockTransport(lambda r: httpx.Response(500)),
        )


def test_rehashed_forged_checkpoint_cannot_escalate_to_never_run(tmp_path):
    d = ledger()
    d["rows"][0].update(
        disposition=travel.NEVER_RUN,
        checkpoint_row=copy.deepcopy(d["rows"][0]["original_frozen_row"]),
        failure=None,
        retained_leaf_path=None,
        checkpoint_call_delta=0,
        spend={"calls": 0},
    )
    d["rows"][109]["disposition"] = "running_interrupted_infrastructure_hold"
    for k in (
        "calls",
        "known_usage_calls",
        "unknown_usage_calls",
        "prompt_tokens",
        "completion_tokens",
    ):
        d["spend"][k] = sum(r["spend"].get(k, 0) for r in d["rows"])
    lp = tmp_path / "forged-ledger.json"
    lp.write_bytes(travel.common._json(d))
    runtime = original()
    runtime["output_root"] = str(tmp_path / "fresh")
    with pytest.raises(ValueError, match="checkpoint"):
        travel.build_continuation_spec(
            runtime=runtime,
            ledger_path=lp,
            ledger_sha256=travel.file_record(lp)["sha256"],
            reserve_proposal_path=ROOT
            / "analysis/travel-continuation-conservative-reserve-proposal-v1.json",
        )


@pytest.mark.parametrize(
    "fault",
    [
        "aggregate",
        "staged_write",
        "staged_read",
        "verify",
        "promote",
        "final_read",
        "deadline_aggregate",
        "deadline_verified",
        "deadline_final",
        "revoke",
    ],
)
def test_publication_fault_preserves_actual_and_reserved_accounting(tmp_path, monkeypatch, fault):
    from gluomem_bench.bench.parallel_campaign import CampaignBudget

    spec = candidate(tmp_path)
    sp, ap, rp = save_approvals(tmp_path, spec)
    rows = travel.load_rows(spec["runtime"]["data"]["path"], spec["runtime"]["data"]["sha256"])
    now = [0.0]
    b = CampaignBudget(max_calls=773632, max_seconds=32637600, clock=lambda: now[0])
    root = Path(spec["runtime"]["output_root"])

    def run(raw, system, ctx, target):
        ctx.begin_call()
        target.mkdir(parents=True)
        (target / "http.jsonl").write_bytes(
            travel.common._json(
                {
                    "url": "https://authored.invalid/chat/completions",
                    "response": {"usage": {"prompt_tokens": 7, "completion_tokens": 3}},
                }
            ).replace(b"\n", b"")
            + b"\n"
        )
        b.failed = True
        raise ValueError("local authored fault followed by global stop")

    original_metrics = travel.campaign_metrics

    def metrics(*args, **kwargs):
        if fault == "aggregate":
            raise OSError("authored aggregate fault")
        answer = original_metrics(*args, **kwargs)
        if fault == "deadline_aggregate":
            now[0] = b.deadline + 1
        return answer

    monkeypatch.setattr(travel, "campaign_metrics", metrics)
    real_write, real_read, real_replace, real_unlink = (
        Path.write_bytes,
        Path.read_bytes,
        Path.replace,
        Path.unlink,
    )

    def write(path, data):
        if path == root / "campaign-result.pending.json" and fault == "staged_write":
            raise OSError("authored staged write fault")
        return real_write(path, data)

    def read(path):
        data = real_read(path)
        if path == root / "campaign-result.pending.json" and fault == "staged_read":
            return b"corrupt"
        if path == root / "campaign-result.json":
            if fault in {"final_read", "revoke"}:
                return b"corrupt"
            if fault == "deadline_final":
                now[0] = b.deadline + 1
        return data

    def replace(path, target):
        if path == root / "campaign-result.pending.json" and fault == "promote":
            raise OSError("authored promotion fault")
        return real_replace(path, target)

    def unlink(path, *args, **kwargs):
        if path == root / "campaign-result.json" and fault == "revoke":
            raise OSError("authored locked complete file")
        return real_unlink(path, *args, **kwargs)

    real_verify = travel.verify_continuation_root

    def verify(*args, **kwargs):
        if fault == "verify":
            raise ValueError("authored full root verify fault")
        answer = real_verify(*args, **kwargs)
        if fault == "deadline_verified":
            now[0] = b.deadline + 1
        return answer

    monkeypatch.setattr(Path, "write_bytes", write)
    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(travel, "verify_continuation_root", verify)
    with pytest.raises((ValueError, OSError, RuntimeError)) as failure:
        travel._execute_continuation_matrix(
            spec, rows, budget=b, run_one=run, spec_paths=(sp, ap, rp)
        )
    receipt = json.loads(real_read(root / "campaign-failure.json"))
    assert receipt["finalized"] is False and receipt["quality_complete"] is False
    assert receipt["calls"] == 1012
    assert receipt["accounting"]["allowance_consumed_calls"] == 55809
    assert receipt["accounting"]["reserved_prior_allowance"] == {"calls": 55808, "seconds": 2358000}
    assert receipt["accounting"]["new_actual"]["prompt_tokens"] == 7
    assert receipt["accounting"]["observed_total_lower_bound"]["completion_tokens"] == 94868
    assert receipt["tasks"][109]["quality_metrics"] is None
    assert all(r["status"] == "unrun" for r in receipt["tasks"][110:])
    assert not (root / "campaign-result.pending.json").exists()
    if fault != "revoke":
        assert not (root / "campaign-result.json").exists()
    assert failure.value.continuation_receipt["calls"] == 1012


def test_semantic_bridge_executes_byte_bound_old_archive_projection(tmp_path):
    spec = candidate(tmp_path)
    bridge = spec["semantic_bridge"]["native_execution_projection"]
    assert bridge["normal_path_equal"] is True
    assert bridge["native_source_pins_equal"] is True
    assert bridge["unchanged_memory_classes"] == [
        "RawExperienceMemory",
        "GluoExperienceMemory",
        "ArenaProductionRouter",
        "BudgetedArenaEmbedder",
    ]
    assert bridge["cleanup_and_failure_handlers_require_independent_review"] is True
    old = (
        b'def run_task(raw):\n    try:\n        return raw["questions"]\n'
        b"    finally:\n        close()\n"
    )
    changed = old.replace(b'raw["questions"]', b'raw["answers"]')
    assert travel._travel_normal_path(old, "run_task") != travel._travel_normal_path(
        changed, "run_task"
    )


def test_real_closed_leaf_replay_denies_network_database_dataset_and_writes(tmp_path, monkeypatch):
    import io
    import socket
    import sqlite3
    import urllib.request

    import httpx

    spec = candidate(tmp_path)
    sp, ap, rp = save_approvals(tmp_path, spec)
    dataset = Path(spec["runtime"]["data"]["path"]).resolve()
    oldroot = Path(ledger()["original_root"]).resolve()
    native = oldroot / "3/full_context"
    before = {str(p): travel.file_record(p)["sha256"] for p in native.rglob("*") if p.is_file()}

    def denied(*args, **kwargs):
        pytest.fail("saved native replay attempted network or database")

    monkeypatch.setattr(socket, "socket", denied)
    monkeypatch.setattr(sqlite3, "connect", denied)
    monkeypatch.setattr(urllib.request, "urlopen", denied)
    monkeypatch.setattr(httpx.Client, "send", denied)
    real_open = io.open

    def checked_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, Path)):
            p = Path(file).resolve()
            assert p != dataset, "saved replay reopened original dataset"
            assert not (p.is_relative_to(oldroot) and any(x in mode for x in "wax+")), (
                "old evidence write"
            )
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(io, "open", checked_open)
    assert travel.verify_continuation_spec(sp, ap, rp, verify_runtime=False) == spec
    assert before == {
        str(p): travel.file_record(p)["sha256"] for p in native.rglob("*") if p.is_file()
    }


def test_setup_time_does_not_extend_any_whole_task_envelope(tmp_path):
    from gluomem_bench.bench.parallel_campaign import CampaignBudget

    spec = candidate(tmp_path)
    sp, ap, rp = save_approvals(tmp_path, spec)
    rows = travel.load_rows(spec["runtime"]["data"]["path"], spec["runtime"]["data"]["sha256"])
    now = [0.0]
    b = CampaignBudget(max_calls=773632, max_seconds=32637600, clock=lambda: now[0])

    def setup(output):
        now[0] = b.deadline - 21599

    def forbidden(*args):
        pytest.fail("unfit whole envelope dispatched")

    result = travel._execute_continuation_matrix(
        spec, rows, budget=b, run_one=forbidden, spec_paths=(sp, ap, rp), initialize=setup
    )
    assert result["calls"] == 1011
    assert result["accounting"]["new_actual"]["calls"] == 0
    assert all(result["tasks"][i]["status"] == "unrun" for i in spec["plan"]["dispatch_indices"])
    assert result["accounting"]["reserved_prior_allowance"]["seconds"] == 2358000
    assert result["quality_complete"] is False


def test_public_native_entry_real_rebuild_all_rows_unrun_without_clients(tmp_path, monkeypatch):
    from gluomem_bench.bench import parallel_campaign
    from gluomem_bench.bench.rack import RackProfile

    old = original()
    runtime = travel.build_spec(
        data=old["data"]["path"],
        data_sha256=old["data"]["sha256"],
        native_root=old["native_root"],
        profile=RackProfile.model_validate(old["profile"]),
        output_root=tmp_path / "new-root",
        embedding_url=old["embedding_url"],
        readiness=old["readiness"],
        tool_python=old["tool_python"]["path"],
        tool_helper=old["tool_helper"]["path"],
        max_steps=30,
        campaign_seconds=old["bounds"]["campaign_seconds"],
        production_memory_qualification=old["production_memory_qualification"]["path"],
    )
    spec = travel.build_continuation_spec(
        runtime=runtime,
        ledger_path=LEDGER,
        ledger_sha256=LEDGER_SHA,
        reserve_proposal_path=ROOT
        / "analysis/travel-continuation-conservative-reserve-proposal-v1.json",
    )
    sp, ap, rp = save_approvals(tmp_path, spec)
    now = [0.0]
    real_budget = parallel_campaign.CampaignBudget
    guards = []

    class AuthoredClockBudget(real_budget):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, clock=lambda: now[0])
            guards.append(self)

    monkeypatch.setattr(parallel_campaign, "CampaignBudget", AuthoredClockBudget)
    real_qualification = travel.formal.verify_memory_qualification

    def qualification(*args, **kwargs):
        answer = real_qualification(*args, **kwargs)
        now[0] = guards[0].deadline - 21599
        return answer

    monkeypatch.setattr(travel.formal, "verify_memory_qualification", qualification)

    def forbidden(*args, **kwargs):
        pytest.fail("no whole task fits: native client or worker must not be constructed")

    monkeypatch.setattr(travel, "TravelTransport", forbidden)
    monkeypatch.setattr(travel, "NativeTools", forbidden)
    result = travel.execute_continuation(sp, ap, rp)
    assert len(guards) == 1
    assert result["calls"] == 1011
    assert result["accounting"]["new_actual"]["calls"] == 0
    assert all(result["tasks"][i]["status"] == "unrun" for i in spec["plan"]["dispatch_indices"])
    assert len(result["tasks"]) == 1620 and result["expected_sessions"] == 11214
    assert travel.verify_continuation_root(runtime["output_root"])["verified"] is True


def test_dataset_setup_fault_retains_full_matrix_and_debit_without_dispatch(tmp_path, monkeypatch):
    spec = candidate(tmp_path)
    sp, ap, rp = save_approvals(tmp_path, spec)

    def failed(*args, **kwargs):
        raise ValueError("authored dataset setup fault")

    monkeypatch.setattr(travel, "load_rows", failed)
    with pytest.raises(ValueError, match="dataset setup") as exc:
        travel.execute_continuation(sp, ap, rp)
    receipt = json.loads(
        (Path(spec["runtime"]["output_root"]) / "campaign-failure.json").read_bytes()
    )
    assert len(receipt["tasks"]) == 1620
    assert receipt["calls"] == 1011
    assert receipt["accounting"]["reserved_prior_allowance"]["calls"] == 55808
    assert exc.value.continuation_receipt["accounting"]["new_actual"]["calls"] == 0


def test_cli_builds_executable_protocol_but_refuses_run_without_reserve(tmp_path, capsys):
    runtime = original()
    runtime["output_root"] = str(tmp_path / "new-root")
    runtime["files"] = travel.formal._code_inventory()
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_bytes(travel.common._json(runtime))
    spec_path = tmp_path / "continuation.json"
    assert (
        travel.main(
            [
                "plan-continuation",
                "--runtime-spec",
                str(runtime_path),
                "--ledger",
                str(LEDGER),
                "--ledger-sha256",
                LEDGER_SHA,
                "--reserve-proposal",
                str(ROOT / "analysis/travel-continuation-conservative-reserve-proposal-v1.json"),
                "--spec-out",
                str(spec_path),
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["dispatch_rows"] == 1511 and printed["retained_rows"] == 109
    spec = json.loads(spec_path.read_bytes())
    assert spec["protocol"] == travel.CONTINUATION_PROTOCOL
    with pytest.raises(FileNotFoundError):
        travel.main(
            [
                "run-never-run",
                "--spec",
                str(spec_path),
                "--approval",
                str(tmp_path / "missing"),
                "--reserve-approval",
                str(tmp_path / "missing-reserve"),
            ]
        )
    assert not Path(runtime["output_root"]).exists()


def test_public_native_dispatch_uses_one_reduced_guard_and_real_mock_transport(
    tmp_path, monkeypatch
):
    import httpx

    from gluomem_bench.bench.rack import RackProfile

    old = original()
    runtime = travel.build_spec(
        data=old["data"]["path"],
        data_sha256=old["data"]["sha256"],
        native_root=old["native_root"],
        profile=RackProfile.model_validate(old["profile"]),
        output_root=tmp_path / "new-root",
        embedding_url=old["embedding_url"],
        readiness=old["readiness"],
        tool_python=old["tool_python"]["path"],
        tool_helper=old["tool_helper"]["path"],
        max_steps=30,
        campaign_seconds=old["bounds"]["campaign_seconds"],
        production_memory_qualification=old["production_memory_qualification"]["path"],
    )
    spec = travel.build_continuation_spec(
        runtime=runtime,
        ledger_path=LEDGER,
        ledger_sha256=LEDGER_SHA,
        reserve_proposal_path=ROOT
        / "analysis/travel-continuation-conservative-reserve-proposal-v1.json",
    )
    sp, ap, rp = save_approvals(tmp_path, spec)
    guards, dispatched, closed = [], [], []
    real_transport = travel.TravelTransport

    def transport(profile, **kwargs):
        ctx = kwargs["budget"]
        guards.append(ctx.budget)

        def http(request):
            if request.url.path.endswith("/models"):
                return httpx.Response(
                    200,
                    json={"data": [{"id": profile.model, "max_model_len": profile.context_window}]},
                )
            if request.url.path.endswith("/tokenize"):
                return httpx.Response(
                    200, json={"count": 100, "max_model_len": profile.context_window}
                )
            assert request.url.path.endswith("/chat/completions")
            assert ctx.budget.max_calls == 773632 and ctx.budget.max_seconds == 32637600
            assert ctx.calls == 1 and ctx.budget.calls == len(dispatched) + 1
            dispatched.append(ctx.job.job_id)
            return httpx.Response(
                200,
                json={
                    "model": profile.model,
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "Authored offline response",
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 2},
                },
            )

        return real_transport(profile, **kwargs, transport=httpx.MockTransport(http))

    class Tools:
        def __init__(self, **kwargs):
            self.budget = kwargs["budget"]

        def close(self):
            closed.append(self.budget.job.job_id)

    def native_run(raw, **kwargs):
        client = kwargs["client"]
        response = client.create(
            "actor",
            model=client.profile.model,
            temperature=0,
            max_tokens=client.profile.output_budget("answer"),
            messages=[{"role": "user", "content": "Authored"}],
            tools=client.tools,
            tool_choice="auto",
        )
        assert response.choices[0].message.content == "Authored offline response"
        if kwargs["system"] == "dense":
            client.budget.budget.failed = True
        raise ValueError("authored local failure; unrelated second task must dispatch")

    monkeypatch.setattr(travel, "TravelTransport", transport)
    monkeypatch.setattr(travel, "NativeTools", Tools)
    monkeypatch.setattr(travel, "run_task", native_run)
    result = travel.execute_continuation(sp, ap, rp)
    assert dispatched == ["travel-continuation:19:bm25", "travel-continuation:19:dense"]
    assert closed == dispatched
    assert guards[0] is guards[1]
    assert result["calls"] == 1013
    assert result["accounting"]["allowance_consumed_calls"] == 55810
    assert result["accounting"]["new_actual"]["known_usage_calls"] == 2
    assert result["accounting"]["new_actual"]["prompt_tokens"] == 200
    assert all(r["status"] == "unrun" for r in result["tasks"][111:])


@pytest.mark.parametrize(
    "attack",
    [
        "missing",
        "duplicate",
        "retry",
        "calls",
        "metrics",
        "unrun_complete",
        "partial_inflation",
        "failed_quality",
    ],
)
def test_full_root_rejects_rehashed_coverage_and_accounting_forgery(tmp_path, attack):
    from gluomem_bench.bench.parallel_campaign import CampaignBudget

    spec = candidate(tmp_path)
    sp, ap, rp = save_approvals(tmp_path, spec)
    rows = travel.load_rows(spec["runtime"]["data"]["path"], spec["runtime"]["data"]["sha256"])
    b = CampaignBudget(max_calls=773632, max_seconds=32637600)
    b.failed = True
    result = travel._execute_continuation_matrix(
        spec, rows, budget=b, run_one=lambda *args: pytest.fail("dispatch"), spec_paths=(sp, ap, rp)
    )
    if attack == "missing":
        result["tasks"].pop()
    elif attack == "duplicate":
        result["tasks"][110] = copy.deepcopy(result["tasks"][109])
    elif attack == "retry":
        result["tasks"][0]["disposition"] = travel.NEVER_RUN
    elif attack == "calls":
        result["calls"] += 1
    elif attack == "metrics":
        result["metrics"]["no_memory"]["full_metrics"] = {"ps": 100, "sps": 100, "sr": 100}
    elif attack == "unrun_complete":
        result["tasks"][109]["status"] = "complete"
    elif attack == "partial_inflation":
        result["tasks"][109].update(status="failed", completed_sessions=999)
        result["completed_sessions"] += 999
    else:
        result["tasks"][109].update(status="failed", quality_metrics={"ps": 0, "sr": 0})
    root = Path(spec["runtime"]["output_root"])
    (root / "task-statuses.json").write_bytes(travel.common._json(result["tasks"]))
    (root / "campaign-result.json").write_bytes(travel.common._json(result))
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        travel.verify_continuation_root(root)


def test_old_root_added_dispatch_evidence_fails_closed_without_mutating_old_tree(
    tmp_path, monkeypatch
):
    spec = candidate(tmp_path)
    sp, ap, rp = save_approvals(tmp_path, spec)
    oldroot = Path(ledger()["original_root"]).resolve()
    extra = oldroot / "19/bm25/http.jsonl"
    real_glob, real_file = Path.rglob, Path.is_file

    def glob(path, pattern, *args, **kwargs):
        yield from real_glob(path, pattern, *args, **kwargs)
        if path.resolve() == oldroot:
            yield extra

    def is_file(path, *args, **kwargs):
        return True if path == extra else real_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "rglob", glob)
    monkeypatch.setattr(Path, "is_file", is_file)
    with pytest.raises(ValueError, match="inventory"):
        travel.verify_continuation_spec(sp, ap, rp, verify_runtime=False)


def test_exact_never_run_selection_and_reserve_are_not_actual_spend():
    plan = travel.continuation_plan(ledger(), original())
    assert plan["scope"] == {"tasks": 270, "ordered_sessions_per_system": 1869, "rows": 1620}
    assert len(plan["dispatch_indices"]) == 1511
    assert len(plan["retained_indices"]) == 109
    assert plan["allowance_debit"] == {"calls": 55808, "seconds": 2358000}
    assert plan["observed_prior"]["calls"] == 1011
    assert plan["observed_prior"]["prompt_tokens"] == 13774852
    assert plan["observed_prior"]["completion_tokens"] == 94865
    assert plan["unknown_prior_attempts"] is None
    assert plan["measured_prior_elapsed_seconds"] is None
    assert plan["remaining_allowance"] == {"calls": 773632, "seconds": 32637600}
