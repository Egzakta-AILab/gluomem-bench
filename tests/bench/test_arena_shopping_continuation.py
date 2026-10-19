"""Offline Shopping NEVER-RUN continuation: no endpoint/native startup allowed."""

import copy
import json
from pathlib import Path

import pytest

ROOT = Path('tests/fixtures/native/operator-supplied')
LEDGER = ROOT / "analysis/shopping-v2-poststop-ledger-v1.json"
LEDGER_SHA = "add32aefe143ff33591fb388ea0a504a669592869f90f6d98180a103c57ff322"
PROPOSAL = ROOT / "analysis/shopping-continuation-conservative-reserve-proposal-v1.json"


def driver():
    from gluomem_bench.bench import arena_shopping_driver as d

    return d


@pytest.fixture
def prior():
    ledger = json.loads(LEDGER.read_bytes())
    original = json.loads((Path(ledger["original_root"]) / "execution-spec.json").read_bytes())
    return ledger, original


def test_continuation_plan_exact_original_scope_reserved_not_actual(prior):
    ledger, original = prior
    plan = driver().continuation_plan(ledger, original)
    assert len(plan["dispatch_indices"]) == 849
    assert len(plan["retained_indices"]) == 51
    assert plan["dispatch_indices"] == list(range(51, 900))
    assert plan["allowance_debit"] == {"calls": 26112, "seconds": 1105200}
    assert plan["remaining_allowance"] == {"calls": 434688, "seconds": 18338400}
    assert plan["observed_prior"]["calls_lower_bound"] == 1175
    assert plan["observed_prior"]["prompt_tokens"] == 8818579
    assert plan["unknown_prior_attempts"] is None
    assert plan["measured_prior_elapsed_seconds"] is None


@pytest.mark.parametrize("attack", ["drop", "duplicate", "promote", "cost", "negative", "reorder"])
def test_continuation_rejects_forged_ledger_projection(prior, attack):
    ledger, original = prior
    if attack == "drop":
        ledger["rows"].pop()
    if attack == "duplicate":
        ledger["rows"][52] = copy.deepcopy(ledger["rows"][51])
    if attack == "promote":
        ledger["rows"][1]["disposition"], ledger["rows"][51]["disposition"] = (
            ledger["rows"][51]["disposition"],
            ledger["rows"][1]["disposition"],
        )
    if attack == "cost":
        ledger["spend"]["prompt_tokens"] += 1
    if attack == "negative":
        ledger["rows"][0]["spend"]["prompt_tokens"] = -1
    if attack == "reorder":
        ledger["rows"][51:53] = reversed(ledger["rows"][51:53])
    with pytest.raises(ValueError):
        driver().continuation_plan(ledger, original)


def build(tmp_path, prior):
    d = driver()
    runtime = copy.deepcopy(prior[1])
    runtime["output_root"] = str(tmp_path / "new")
    runtime["files"] = d.common._code_inventory()
    return d.build_continuation_spec(
        runtime=runtime,
        ledger_path=LEDGER,
        ledger_sha256=LEDGER_SHA,
        reserve_proposal_path=PROPOSAL,
    )


def test_continuation_builder_and_actual_eight_pinned_readonly_replays(
    tmp_path, prior, monkeypatch
):
    import socket

    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: pytest.fail("network forbidden")
    )
    d = driver()
    spec = build(tmp_path, prior)
    assert spec["protocol"] == d.CONTINUATION_PROTOCOL
    assert spec["semantic_bridge"]["privacy_guard_changed_requires_review"] is True
    assert spec["reserve_proposal"]["sha256"] == d.sha(PROPOSAL)
    proof = d.verify_continuation_origin(spec)
    assert proof["reused_tasks"] == 8
    assert proof["reused_sessions"] == 48
    assert proof["database_opens"] == 0


def test_continuation_no_reserve_approval_refuses_before_setup(tmp_path, prior, monkeypatch):
    d = driver()
    spec = build(tmp_path, prior)
    sp = tmp_path / "spec.json"
    sp.write_bytes(d.native._json(spec))
    approval = tmp_path / "approval.json"
    reserve = tmp_path / "reserve.json"
    reserve.write_bytes(d.native._json({"approved": False}))
    approval.write_bytes(
        d.native._json(
            {
                "approved": True,
                "reviewer_kind": "parent",
                "spec_sha256": d.sha(sp),
                "reserve_approval_sha256": d.sha(reserve),
            }
        )
    )
    monkeypatch.setattr(
        d, "ShoppingRuntime", lambda *a, **k: pytest.fail("startup before approval")
    )
    with pytest.raises(ValueError, match="reserve"):
        d.execute_continuation(sp, approval, reserve)
    assert not Path(spec["runtime"]["output_root"]).exists()


def test_continuation_guarded_matrix_local_failure_does_not_cancel_healthy(
    tmp_path, prior, monkeypatch
):
    d = driver()
    spec = build(tmp_path, prior)
    from gluomem_bench.bench.parallel_campaign import CampaignBudget

    allowance = spec["plan"]["remaining_allowance"]
    budget = CampaignBudget(max_calls=allowance["calls"], max_seconds=allowance["seconds"])
    visited = []

    def run(task, system, context, target):
        visited.append((task.id, system))
        context.begin_call()
        if len(visited) == 1:
            context.failed = True
            raise ValueError("local authored quality failure; discovery healthy")
        # Cancel genuine shared infrastructure only after an unrelated job admitted.
        context.budget.failed = True
        raise ValueError("authored shared identity failure")

    tasks = d.load_arena(
        Path(spec["runtime"]["data_root"]), domain=d.DOMAIN, pins=spec["runtime"]["data_pins"]
    )
    result = d._execute_continuation_matrix(
        spec,
        tasks,
        budget=budget,
        run_one=run,
        initialize=lambda root: None,
        verify_saved=lambda root, **kwargs: kwargs["payload"],
    )
    assert visited == [(8, "hybrid"), (8, "full_context")]
    assert len(result["tasks"]) == 900
    assert result["tasks"][51]["status"] == "failed"
    assert result["tasks"][52]["status"] == "failed"
    assert result["tasks"][53]["status"] == "unrun"
    assert result["accounting"]["new_actual_calls"] == 2
    assert result["accounting"]["allowance_consumed_calls"] == 26114
    assert result["accounting"]["observed_total_calls_lower_bound"] == 1177
    assert result["quality_complete"] is False
    assert all(r.get("metrics") is None for r in result["tasks"][51:])


def authored_authority(tmp_path, spec):
    """Only authored test authority in scratch, never campaign admission files."""
    d = driver()
    sp, ap, rp = (tmp_path / n for n in ("spec.json", "approval.json", "reserve.json"))
    sp.write_bytes(d.native._json(spec))
    proof = tmp_path / "authored-guard-proof.json"
    proof.write_bytes(
        d.native._json(
            {
                "original_spec_sha256": prior_spec_sha(spec),
                "original_source_archive_sha256": json.loads(
                    Path(spec["ledger"]["path"]).read_bytes()
                )["source_archive_sha256"],
                "original_bounds": spec["runtime"]["bounds"],
                "original_guards_verified": True,
                "no_transport_bypass": True,
                "unknown_attempts_bounded_by_original_guards": True,
            }
        )
    )
    review = tmp_path / "authored-independent-review.json"
    review.write_bytes(
        d.native._json(
            {
                "approved": True,
                "reviewer_kind": "independent",
                "blockers": [],
                "spec_sha256": d.sha(sp),
                "ledger_sha256": spec["ledger"]["sha256"],
                "reserve_proposal_sha256": spec["reserve_proposal"]["sha256"],
                "semantic_bridge_sha256": __import__("hashlib")
                .sha256(d.native._json(spec["semantic_bridge"]))
                .hexdigest(),
                "guard_proof_sha256": d.sha(proof),
            }
        )
    )
    reserve = {
        "approved": True,
        "reviewer_kind": "parent",
        "spec_sha256": d.sha(sp),
        "ledger_sha256": spec["ledger"]["sha256"],
        "reserve_proposal_sha256": spec["reserve_proposal"]["sha256"],
        "allowance_debit": spec["plan"]["allowance_debit"],
        "semantic_bridge_sha256": __import__("hashlib")
        .sha256(d.native._json(spec["semantic_bridge"]))
        .hexdigest(),
        "guard_proof": d._shopping_record(proof),
        "independent_review": d._shopping_record(review),
    }
    reserve.update(
        {
            k: True
            for k in (
                "original_guards_verified",
                "no_transport_bypass",
                "unknown_attempts_bounded_by_original_guards",
                "semantic_bridge_verified",
                "privacy_guard_repair_reviewed",
            )
        }
    )
    rp.write_bytes(d.native._json(reserve))
    ap.write_bytes(
        d.native._json(
            {
                "approved": True,
                "reviewer_kind": "parent",
                "spec_sha256": d.sha(sp),
                "reserve_approval_sha256": d.sha(rp),
            }
        )
    )
    return sp, ap, rp


def prior_spec_sha(spec):
    return json.loads(Path(spec["ledger"]["path"]).read_bytes())["spec_sha256"]


def test_continuation_typed_permit_cannot_admit_prior_held_or_altered_authority(tmp_path, prior):
    d = driver()
    spec = build(tmp_path, prior)
    paths = authored_authority(tmp_path, spec)
    assert d.verify_continuation_spec(*paths, verify_runtime=False) == spec
    # Actual native task reconstruction, not fabricated task identities.
    from gluomem_bench.bench.rack import RackProfile

    profile = RackProfile.model_validate(spec["runtime"]["profile"])
    tasks = d.load_arena(
        Path(spec["runtime"]["data_root"]), domain=d.DOMAIN, pins=spec["runtime"]["data_pins"]
    )
    permit = d.ShoppingContinuationPermit(*paths, system="hybrid")
    assert permit.validate(tasks[8], profile, 25) == spec["runtime"]
    with pytest.raises(ValueError, match="never-run"):
        permit.validate(tasks[0], profile, 25)
    paths[1].write_bytes(paths[1].read_bytes() + b" ")
    with pytest.raises(ValueError, match="authority"):
        permit.check()


def test_continuation_refuses_missing_independent_guard_review_even_with_parent_flags(
    tmp_path, prior
):
    d = driver()
    spec = build(tmp_path, prior)
    sp, ap, rp = authored_authority(tmp_path, spec)
    reserve = json.loads(rp.read_bytes())
    reserve.pop("independent_review")
    rp.write_bytes(d.native._json(reserve))
    approval = json.loads(ap.read_bytes())
    approval["reserve_approval_sha256"] = d.sha(rp)
    ap.write_bytes(d.native._json(approval))
    with pytest.raises(ValueError, match="independent"):
        d.verify_continuation_approval(sp, ap, rp)


def saved_inputs(root, paths, spec):
    d = driver()
    for path, name in zip(
        paths,
        ("continuation-spec.json", "execution-approval.json", "reserve-approval.json"),
        strict=True,
    ):
        (root / name).write_bytes(path.read_bytes())
    (root / "execution-spec.json").write_bytes(d.native._json(spec["runtime"]))
    d._shopping_archive(root, spec["runtime"])


def test_continuation_full_root_replay_without_dispatch_rejects_digest_and_row_promotion(
    tmp_path, prior
):
    d = driver()
    spec = build(tmp_path, prior)
    paths = authored_authority(tmp_path, spec)
    from gluomem_bench.bench.parallel_campaign import CampaignBudget

    clock = [0.0]
    remaining = spec["plan"]["remaining_allowance"]
    budget = CampaignBudget(
        max_calls=remaining["calls"], max_seconds=remaining["seconds"], clock=lambda: clock[0]
    )

    def init(root):
        saved_inputs(root, paths, spec)
        clock[0] = remaining["seconds"] - 100  # Insufficient whole-task time.

    tasks = d.load_arena(
        Path(spec["runtime"]["data_root"]), domain=d.DOMAIN, pins=spec["runtime"]["data_pins"]
    )
    result = d._execute_continuation_matrix(
        spec,
        tasks,
        budget=budget,
        run_one=lambda *a: pytest.fail("whole task must not dispatch"),
        initialize=init,
    )
    root = Path(spec["runtime"]["output_root"])
    assert d.verify_continuation_root(root) == result
    assert result["calls"] == 0
    assert result["completed_sessions"] == 48
    assert result["quality_complete"] is False
    raw = (root / "campaign-result.json").read_bytes()
    result["tasks"][0]["result_sha256"] = "f" * 64
    (root / "campaign-result.json").write_bytes(d.native._json(result))
    with pytest.raises(ValueError):
        d.verify_continuation_root(root)
    (root / "campaign-result.json").write_bytes(raw)
    statuses = json.loads((root / "task-statuses.json").read_bytes())
    statuses[1]["status"] = "complete"
    (root / "task-statuses.json").write_bytes(d.native._json(statuses))
    result = json.loads(raw)
    result["tasks"] = statuses
    (root / "campaign-result.json").write_bytes(d.native._json(result))
    with pytest.raises(ValueError):
        d.verify_continuation_root(root)


@pytest.mark.parametrize("cleanup_fault", [False, True])
def test_continuation_public_executor_uses_native_runner_and_same_outer_guard(
    tmp_path, prior, monkeypatch, cleanup_fault
):
    d = driver()
    spec = build(tmp_path, prior)
    paths = authored_authority(tmp_path, spec)
    import socket

    monkeypatch.setattr(
        socket.socket, "connect", lambda *a, **k: pytest.fail("real endpoint forbidden")
    )
    original_verify = d.verify_continuation_spec
    seen = []

    def verify(*args, **kwargs):
        if kwargs.get("verify_runtime", True):
            assert kwargs["budget"].calls == 0
        return original_verify(*args, **{**kwargs, "verify_runtime": False})

    monkeypatch.setattr(d, "verify_continuation_spec", verify)
    monkeypatch.setattr(d.common, "verify_memory_qualification", lambda *a: None)
    monkeypatch.setattr(d.common, "BudgetedArenaEmbedder", lambda *a, **k: object())

    def native_run(row, *, client, out, **kwargs):
        from gluomem_bench.bench.rack import SharedRackError

        client.permit.validate(
            d.load_arena(
                Path(spec["runtime"]["data_root"]),
                domain=d.DOMAIN,
                pins=spec["runtime"]["data_pins"],
            )[row["id"]],
            client.profile,
            25,
        )
        seen.append((row["id"], client.permit.system, client.budget.budget))
        client.budget.begin_call()
        if len(seen) == 1:
            raise ValueError("authored local failure")
        client._latch(SharedRackError("authored shared identity failure"))
        raise SharedRackError("authored shared identity failure")

    monkeypatch.setattr(d, "run_task", native_run)
    if cleanup_fault:
        from gluomem_bench.bench.rack import RackClient

        monkeypatch.setattr(
            RackClient,
            "close",
            lambda self: (_ for _ in ()).throw(OSError("authored client cleanup fault")),
        )
    result = d.execute_continuation(*paths)
    assert "authored local failure" in result["tasks"][51]["error"]
    if cleanup_fault:
        assert "authored client cleanup fault" in str(result["tasks"][51]["cleanup_diagnostics"])
    assert [(t, s) for t, s, _ in seen] == [(8, "hybrid"), (8, "full_context")]
    assert seen[0][2] is seen[1][2]
    assert result["calls"] == 2
    assert d.verify_continuation_root(Path(spec["runtime"]["output_root"])) == result


def test_continuation_builder_rejects_native_runtime_semantic_drift(tmp_path, prior):
    d = driver()
    runtime = copy.deepcopy(prior[1])
    runtime["output_root"] = str(tmp_path / "new")
    runtime["files"] = d.common._code_inventory()
    runtime["runtime"]["files"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="native runtime"):
        d.build_continuation_spec(
            runtime=runtime,
            ledger_path=LEDGER,
            ledger_sha256=LEDGER_SHA,
            reserve_proposal_path=PROPOSAL,
        )


@pytest.mark.parametrize(
    "phase",
    [
        "pending-write",
        "pending-read",
        "verify",
        "replace",
        "final-read",
        "setup-deadline",
        "verify-deadline",
    ],
)
def test_continuation_publication_faults_revoke_authority_keep_spend_and_fresh_recovery(
    tmp_path, prior, monkeypatch, phase
):
    d = driver()
    spec = build(tmp_path, prior)
    from gluomem_bench.bench.parallel_campaign import CampaignBudget

    allowance = spec["plan"]["remaining_allowance"]
    clock = [0.0]

    def budget():
        return CampaignBudget(
            max_calls=allowance["calls"], max_seconds=allowance["seconds"], clock=lambda: clock[0]
        )

    guard = budget()
    root = Path(spec["runtime"]["output_root"])
    tasks = d.load_arena(
        Path(spec["runtime"]["data_root"]), domain=d.DOMAIN, pins=spec["runtime"]["data_pins"]
    )

    def run(task, system, context, target):
        context.begin_call()
        context.budget.failed = True
        raise ValueError("originating authored local failure")

    def init(out):
        if phase == "setup-deadline":
            clock[0] = guard.deadline

    def verifier(out, **kwargs):
        if phase == "verify":
            raise OSError("authored verifier fault")
        if phase == "verify-deadline":
            clock[0] = guard.deadline
        return kwargs["payload"]

    original_write, original_read, original_replace = (
        Path.write_bytes,
        Path.read_bytes,
        Path.replace,
    )

    def write(path, data):
        if (
            path.parent == root
            and path.name == "campaign-result.pending.json"
            and phase == "pending-write"
        ):
            original_write(path, b"partial")
            raise OSError("authored pending write fault")
        return original_write(path, data)

    def read(path):
        if path.parent == root and (
            (path.name == "campaign-result.pending.json" and phase == "pending-read")
            or (path.name == "campaign-result.json" and phase == "final-read")
        ):
            raise OSError("authored readback fault")
        return original_read(path)

    def replace(path, target):
        if (
            path.parent == root
            and path.name == "campaign-result.pending.json"
            and phase == "replace"
        ):
            raise OSError("authored promotion fault")
        return original_replace(path, target)

    with monkeypatch.context() as m:
        m.setattr(Path, "write_bytes", write)
        m.setattr(Path, "read_bytes", read)
        m.setattr(Path, "replace", replace)
        with pytest.raises((OSError, RuntimeError)):
            d._execute_continuation_matrix(
                spec, tasks, budget=guard, run_one=run, initialize=init, verify_saved=verifier
            )
    assert not (root / "campaign-result.json").exists()
    assert not (root / "campaign-result.pending.json").exists()
    failure = json.loads((root / "campaign-failure.json").read_bytes())
    assert failure["method_metrics"] is None and failure["quality_complete"] is False
    assert failure["calls"] == (0 if phase == "setup-deadline" else 1)
    assert failure["accounting"]["reserved_prior_allowance"] == {"calls": 26112, "seconds": 1105200}
    spec["runtime"]["output_root"] = str(tmp_path / "fresh-recovery")
    clock[0] = 0.0
    recovered = d._execute_continuation_matrix(
        spec,
        tasks,
        budget=budget(),
        run_one=run,
        initialize=lambda root: None,
        verify_saved=lambda root, **kwargs: kwargs["payload"],
    )
    assert recovered["calls"] == 1
    assert Path(spec["runtime"]["output_root"], "campaign-result.json").exists()


def test_continuation_cli_plan_and_unapproved_run_are_offline(tmp_path, prior, capsys):
    d = driver()
    runtime = copy.deepcopy(prior[1])
    runtime["files"] = d.common._code_inventory()
    runtime["output_root"] = str(tmp_path / "new")
    runtime_path = tmp_path / "runtime-spec.json"
    runtime_path.write_bytes(d.native._json(runtime))
    spec_path = tmp_path / "cli-spec.json"
    assert (
        d.main(
            [
                "plan-continuation",
                "--runtime-spec",
                str(runtime_path),
                "--ledger",
                str(LEDGER),
                "--ledger-sha256",
                LEDGER_SHA,
                "--reserve-proposal",
                str(PROPOSAL),
                "--spec-out",
                str(spec_path),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["dispatch_rows"] == 849 and report["quality_calls"] == 0
    approval = tmp_path / "unapproved.json"
    approval.write_bytes(b'{"approved":false}')
    with pytest.raises(ValueError, match="reserve"):
        d.main(
            [
                "run-never-run",
                "--spec",
                str(spec_path),
                "--approval",
                str(approval),
                "--reserve-approval",
                str(approval),
            ]
        )
    assert not Path(runtime["output_root"]).exists()


def test_continuation_native_cleanup_does_not_hide_origin(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    d = driver()
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )
    from gluomem_bench.bench.rack import RackProfile

    outer = CampaignBudget(max_calls=512, max_seconds=21600)
    job = Job("authored", "authored", tmp_path, 512, 21600)
    outer.reserve(job)
    budget = JobContext(job, outer, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))
    runtime = tmp_path / "runtime.json"
    runtime.write_bytes(b'{"files":[]}')

    class Worker:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def request(self, op, **kwargs):
            if op == "close_session":
                raise OSError("authored cleanup failure")
            return {
                "conversation_start": [],
                "public_observation": {"state": "authored public state"},
            }

    @contextmanager
    def actor(*args):
        def act(inputs):
            budget.begin_call()
            raise ValueError("authored originating actor failure")

        yield SimpleNamespace(reset=lambda: None, act_with_messages=act)

    monkeypatch.setattr(d, "ShoppingRuntime", Worker)
    monkeypatch.setattr(d, "native_actor", actor)
    client = SimpleNamespace(
        live=False,
        profile=RackProfile(base_url="http://localhost:1/v1", model="Qwen/authored"),
        budget=budget,
        check=budget.check,
        close=lambda: None,
        preflight=lambda: None,
    )
    memory = SimpleNamespace(budget=budget, query=lambda text: "", close=lambda: None)
    with pytest.raises(RuntimeError):
        d.run_task(
            {"id": -1, "questions": ["authored"], "answers": [{}]},
            runtime=runtime,
            client=client,
            memory=memory,
            out=tmp_path / "leaf",
            max_rounds=25,
        )
    failure = json.loads((tmp_path / "leaf/failure.json").read_bytes())
    assert failure["error"] == "ValueError: authored originating actor failure"
    assert "authored cleanup failure" in json.dumps(failure["cleanup_diagnostics"])
    assert failure["calls"] == 1


def test_continuation_runtime_cleanup_uses_same_deadline_and_preserves_origin(tmp_path):
    import queue
    from types import SimpleNamespace

    d = driver()
    observed = []
    runtime = d.ShoppingRuntime.__new__(d.ShoppingRuntime)
    runtime.budget = SimpleNamespace(deadline=10, budget=SimpleNamespace(clock=lambda: 10))
    runtime.sequence = 0
    runtime.out = tmp_path
    runtime.process = SimpleNamespace(
        stdin=SimpleNamespace(write=lambda b: None, flush=lambda: None), kill=lambda: None
    )

    def get(*, timeout):
        observed.append(timeout)
        raise queue.Empty

    runtime.lines = SimpleNamespace(get=get)
    with pytest.raises(TimeoutError):
        runtime.request("close")
    assert observed[0] <= 0.01
    origin = ValueError("authored origin")
    runtime.request = lambda *a, **k: (_ for _ in ()).throw(
        OSError("authored native cleanup fault")
    )
    runtime.process = SimpleNamespace(poll=lambda: 0, stdout=SimpleNamespace(close=lambda: None))
    runtime.log = SimpleNamespace(close=lambda: None)
    runtime.thread = SimpleNamespace(join=lambda **k: None)
    runtime.__exit__(ValueError, origin, None)
    assert "authored native cleanup fault" in str(origin.__notes__)


def test_continuation_origin_rejects_rehashed_costs_against_actual_old_receipts(
    tmp_path, prior, monkeypatch
):
    d = driver()
    spec = build(tmp_path, prior)
    forged = copy.deepcopy(prior[0])
    forged["rows"][0]["spend"]["prompt_tokens"] += 1
    forged["spend"]["prompt_tokens"] += 1
    original_bound = d._shopping_bound
    monkeypatch.setattr(
        d,
        "_shopping_bound",
        lambda rec: d.native._json(forged) if Path(rec["path"]) == LEDGER else original_bound(rec),
    )
    with pytest.raises(ValueError, match="spend"):
        d.verify_continuation_origin(spec)


def test_continuation_failed_publication_latch_overrides_surviving_result(tmp_path):
    d = driver()
    (tmp_path / "campaign-failure.json").write_bytes(b'{"quality_complete":false}')
    (tmp_path / "campaign-result.json").write_bytes(b'{"quality_complete":true}')
    with pytest.raises(ValueError, match="failed publication"):
        d.verify_continuation_root(tmp_path)


def test_continuation_expired_runtime_refuses_before_native_process_start(tmp_path, monkeypatch):
    from types import SimpleNamespace

    d = driver()
    from gluomem_bench.bench.parallel_campaign import CampaignStopped

    def expired():
        raise CampaignStopped("authored setup deadline")

    monkeypatch.setattr(
        d.subprocess, "Popen", lambda *a, **k: pytest.fail("JVM startup after deadline")
    )
    with pytest.raises(CampaignStopped):
        d.ShoppingRuntime(
            tmp_path / "missing-manifest.json",
            budget=SimpleNamespace(check=expired),
            out=tmp_path / "out",
        )


@pytest.mark.parametrize("field", ["original_source_archive_sha256", "original_bounds"])
def test_continuation_guard_review_must_bind_original_archive_and_caps(tmp_path, prior, field):
    d = driver()
    spec = build(tmp_path, prior)
    sp, ap, rp = authored_authority(tmp_path, spec)
    reserve = json.loads(rp.read_bytes())
    proof_path = Path(reserve["guard_proof"]["path"])
    proof = json.loads(proof_path.read_bytes())
    proof[field] = "forged-but-rehashed"
    proof_path.write_bytes(d.native._json(proof))
    reserve["guard_proof"] = d._shopping_record(proof_path)
    review_path = Path(reserve["independent_review"]["path"])
    review = json.loads(review_path.read_bytes())
    review["guard_proof_sha256"] = d.sha(proof_path)
    review_path.write_bytes(d.native._json(review))
    reserve["independent_review"] = d._shopping_record(review_path)
    rp.write_bytes(d.native._json(reserve))
    approval = json.loads(ap.read_bytes())
    approval["reserve_approval_sha256"] = d.sha(rp)
    ap.write_bytes(d.native._json(approval))
    with pytest.raises(ValueError, match="original-guard"):
        d.verify_continuation_approval(sp, ap, rp)


def test_continuation_semantic_bridge_rejects_changed_native_memory_wiring(prior, monkeypatch):
    d = driver()
    path = Path(d.__file__)
    raw = path.read_bytes()
    changed = raw.replace(
        b"system, budget=context, embedder=encoder, top_k=10",
        b"system, budget=context, embedder=encoder, top_k=9",
    )
    assert changed != raw
    original_read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda p: changed if p == path else original_read(p))
    with pytest.raises(ValueError, match="wiring"):
        d._shopping_bridge(prior[1])


def test_continuation_revocation_cleanup_fault_preserves_origin_and_actual_counter(
    tmp_path, prior, monkeypatch
):
    d = driver()
    spec = build(tmp_path, prior)
    from gluomem_bench.bench.parallel_campaign import CampaignBudget

    remaining = spec["plan"]["remaining_allowance"]
    budget = CampaignBudget(max_calls=remaining["calls"], max_seconds=remaining["seconds"])
    tasks = d.load_arena(
        Path(spec["runtime"]["data_root"]), domain=d.DOMAIN, pins=spec["runtime"]["data_pins"]
    )
    root = Path(spec["runtime"]["output_root"])

    def run(task, system, context, target):
        context.begin_call()
        context.budget.failed = True
        raise ValueError("authored local failure")

    def verify(*args, **kwargs):
        raise ValueError("authored originating verifier failure")

    real_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if path.parent == root and path.name == "campaign-result.pending.json":
            raise PermissionError("authored revocation cleanup fault")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(ValueError, match="originating verifier") as error:
        d._execute_continuation_matrix(
            spec,
            tasks,
            budget=budget,
            run_one=run,
            initialize=lambda root: None,
            verify_saved=verify,
        )
    assert "revocation cleanup fault" in str(error.value.__notes__)
    failure = json.loads((root / "campaign-failure.json").read_bytes())
    assert failure["calls"] == 1
    assert failure["accounting"]["allowance_consumed_calls"] == 26113
    assert not (root / "campaign-result.pending.json").exists()
    assert not (root / "campaign-result.json").exists()


def test_continuation_review_byte_tamper_is_shared_authority_cancellation(tmp_path, prior):
    d = driver()
    spec = build(tmp_path, prior)
    paths = authored_authority(tmp_path, spec)
    permit = d.ShoppingContinuationPermit(*paths, system="hybrid")
    review = json.loads(paths[2].read_bytes())["independent_review"]
    path = Path(review["path"])
    path.write_bytes(path.read_bytes() + b" ")
    from gluomem_bench.bench.rack import SharedRackError

    with pytest.raises(SharedRackError):
        permit.check()
