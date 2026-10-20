"""Offline full-matrix continuation and enclosing publication authority."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from gluomem_bench.bench import beam_driver as driver


def full_spec(root):
    selections = [dict(split=split, history_id=f"beam-{split}-{i}", questions=20,
                       case_ids=[f"beam-{split}-{i}-q{q}" for q in range(20)])
                  for split, count in (("100K", 20), ("500K", 35), ("1M", 35), ("10M", 10))
                  for i in range(1, count + 1)]
    return dict(protocol=driver.FULL_PROTOCOL, selections=selections,
                systems=list(driver.SUPPORTED_SYSTEMS), output_root=str(root),
                jobs=driver.full_job_matrix(selections, list(driver.SUPPORTED_SYSTEMS), root, 100, 120),
                campaign_policy=dict(max_workers=1, max_in_flight=1, max_in_flight_tokens=200000,
                                     max_calls=100000, max_seconds=100000, stop_on_failure=False))


def root_fixture(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    spec = full_spec(root)
    driver.write(root / "execution-spec.json", spec)
    sha = hashlib.sha256((root / "execution-spec.json").read_bytes()).hexdigest()
    driver.write(root / "independent-approval.json", dict(approved=True, spec_sha256=sha))
    state = dict(complete=False, status="incomplete", spec_sha256=sha,
                 accounting=dict(calls=0, elapsed_seconds=0),
                 jobs=[dict(job_id=j["job_id"], status="unrun", calls=0, artifact_sha256={})
                       for j in spec["jobs"]])
    driver.write(root / "root-status.json", state)
    return root, spec, state


def test_continuation_freezes_held_failed_and_selects_only_never_run(tmp_path):
    root, old, state = root_fixture(tmp_path)
    old["campaign_policy"].update(max_calls=11468800, max_seconds=15123600)
    driver.write(root / "execution-spec.json", old)
    state["spec_sha256"] = hashlib.sha256((root / "execution-spec.json").read_bytes()).hexdigest()
    state["accounting"] = dict(calls=283, elapsed_seconds=214.23125769998296)
    for i, row in enumerate(state["jobs"]):
        row.update(status="failed" if i < 8 else "cancelled", calls=283 if i == 0 else 0)
    driver.write(root / "root-status.json", state)
    archive = tmp_path / "old.zip"
    archive.write_bytes(b"authored preserved producer archive")
    fresh = copy.deepcopy(old)
    fresh["output_root"] = str(tmp_path / "fresh")
    fresh["jobs"] = driver.full_job_matrix(fresh["selections"], fresh["systems"], fresh["output_root"], 100, 120)
    bound = driver.bind_continuation(fresh, prior_root=root,
        prior_root_sha256=hashlib.sha256((root / "root-status.json").read_bytes()).hexdigest(),
        prior_spec_sha256=state["spec_sha256"], prior_archive_path=archive,
        prior_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    assert bound["protocol"] == driver.CONTINUATION_PROTOCOL
    assert driver.validate_full_matrix(bound) == 14000
    plan = driver.continuation_plan(bound)
    assert len(plan["held"]) == 8 and len(plan["dispatch_job_ids"]) == 692
    assert plan["prior_calls"] == 283
    assert plan["prior_elapsed_seconds"] == 214.23125769998296
    assert bound["campaign_policy"]["max_calls"] == 11468800
    assert bound["campaign_policy"]["max_seconds"] == 15123600
    assert bound["continuation"]["cognee"] == dict(status="unrun", qualified=False)
    bad = copy.deepcopy(bound)
    bad["continuation"]["dispatch_job_ids"].append(plan["held"][0]["job_id"])
    with pytest.raises(ValueError, match="dispatch"):
        driver.continuation_plan(bad)
    (root / "root-status.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="prior.*hash"):
        driver.continuation_plan(bound)


@pytest.mark.parametrize("fault", [None, "deadline", "verifier", "write", "replace", "staged_readback", "final_readback"])
def test_publication_faults_revoke_authority_retain_spend_and_recover(tmp_path, monkeypatch, fault):
    from gluomem_bench.bench.parallel_campaign import CampaignBudget, CampaignReceiptError
    import os
    root, _, state = root_fixture(tmp_path)
    clock = [0.0]
    budget = CampaignBudget(max_calls=100000, max_seconds=100000, clock=lambda: clock[0])
    budget.begin_call()
    state["jobs"][0].update(status="failed", calls=1)
    pending = root / "root-status.json.pending"
    original_verify, original_write = driver.verify_campaign, driver.write
    original_replace, original_read = os.replace, Path.read_bytes
    with monkeypatch.context() as m:
        if fault == "deadline":
            def verify(*a, **kw):
                result = original_verify(*a, **kw)
                clock[0] = 100001
                return result
            m.setattr(driver, "verify_campaign", verify)
        if fault == "verifier":
            m.setattr(driver, "verify_campaign", lambda *a, **kw: (_ for _ in ()).throw(ValueError("authored verifier fault")))
        if fault == "write":
            def write(path, value):
                if path == pending:
                    raise OSError("authored staged write fault")
                return original_write(path, value)
            m.setattr(driver, "write", write)
        if fault == "replace":
            def replace(src, dst):
                raise OSError("authored atomic replace fault")
            m.setattr(os, "replace", replace)
        if fault in {"staged_readback", "final_readback"}:
            def read(path):
                if (fault == "staged_readback" and path == pending) or (
                        fault == "final_readback" and path == root / "root-status.json" and not pending.exists()):
                    raise OSError("authored readback fault")
                return original_read(path)
            m.setattr(Path, "read_bytes", read)
        if fault is None:
            assert driver._publish_campaign(root, state, budget)["accounting"]["calls"] == 1
        else:
            with pytest.raises(CampaignReceiptError) as raised:
                driver._publish_campaign(root, state, budget)
            assert raised.value.receipt["accounting"]["calls"] == 1
    saved = json.loads((root / "root-status.json").read_bytes())
    assert saved["complete"] is False
    assert saved["accounting"]["calls"] == 1
    assert not pending.exists()
    if fault:
        assert saved["status"] == "incomplete" and "publication_failure" in saved
    fresh = tmp_path / "recovery"
    fresh.mkdir()
    newroot, _, healthy = root_fixture(fresh)
    healthy_budget = CampaignBudget(max_calls=100000, max_seconds=100000)
    assert driver._publish_campaign(newroot, healthy, healthy_budget) == driver.verify_campaign(newroot)


@pytest.mark.parametrize("receipt_fault", [False, True])
def test_executor_owns_original_debited_budget_before_setup_and_publication(tmp_path, monkeypatch, receipt_fault):
    from gluomem_bench.bench import parallel_campaign as scheduler
    from gluomem_bench.bench import comprehensive_driver, embedding_runtime
    from gluomem_bench.bench.rack import RackProfile
    root, old, previous = root_fixture(tmp_path)
    previous["accounting"] = dict(calls=283, elapsed_seconds=214.23125769998296)
    for i, row in enumerate(previous["jobs"]):
        row.update(status="failed" if i < 8 else "cancelled", calls=283 if i == 0 else 0)
    driver.write(root / "root-status.json", previous)
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"authored archive")
    spec = full_spec(tmp_path / "continuation")
    spec.update(profile=RackProfile(base_url="http://localhost:1/v1", model="authored/Qwen",
        profile_version="role-budgets-v6", generation_budgets=dict(extraction=4096,
        consolidation=4096, reader=4096, judge=4096)).model_dump(mode="json"),
        embedding_url="http://localhost:1/v1", config_overrides={})
    spec = driver.bind_continuation(spec, prior_root=root,
        prior_root_sha256=hashlib.sha256((root / "root-status.json").read_bytes()).hexdigest(),
        prior_spec_sha256=previous["spec_sha256"], prior_archive_path=archive,
        prior_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    specpath, approval = tmp_path / "new-spec.json", tmp_path / "approval.json"
    driver.write(specpath, spec)
    driver.write(approval, dict(approved=True, spec_sha256=hashlib.sha256(specpath.read_bytes()).hexdigest()))
    monkeypatch.setattr(driver, "verify_execution_spec", lambda *a: spec)
    observed = {}
    original_budget = scheduler.CampaignBudget
    def budget(**kw):
        value = original_budget(**kw)
        observed["budget"] = value
        assert kw["prior_calls"] == 283 and kw["prior_elapsed_seconds"] == 214.23125769998296
        assert kw["max_calls"] == 100000 and kw["max_seconds"] == 100000
        return value
    monkeypatch.setattr(scheduler, "CampaignBudget", budget)
    def preserve(*a):
        assert observed["budget"].calls == 283  # The deadline has started BEFORE setup.
        return []
    monkeypatch.setattr(comprehensive_driver, "preserve_frozen_inputs", preserve)
    monkeypatch.setattr(embedding_runtime, "RemoteEmbedder", lambda *a, **kw: object())
    def run_jobs(jobs, *, budget, **kw):
        assert budget is observed["budget"]
        assert len(jobs) == 692
        assert jobs[0].job_id == spec["jobs"][8]["job_id"]
        budget.begin_call()
        receipt = dict(accounting=budget.snapshot(), jobs=[dict(job_id=j.job_id,
            history_id=j.history_id, status="failed" if i == 0 else "unrun",
            calls=1 if i == 0 else 0) for i, j in enumerate(jobs)])
        if receipt_fault:
            raise scheduler.CampaignReceiptError(receipt)
        return receipt
    monkeypatch.setattr(scheduler, "run_jobs", run_jobs)
    original_publish = driver._publish_campaign
    def publish(root, state, budget):
        assert budget is observed["budget"]
        assert len(state["jobs"]) == 700
        assert [r["status"] for r in state["jobs"][:8]] == ["held"] * 8
        return original_publish(root, state, budget)
    monkeypatch.setattr(driver, "_publish_campaign", publish)
    if receipt_fault:
        with pytest.raises(scheduler.CampaignReceiptError) as raised:
            driver.execute(specpath, approval)
        result = json.loads((Path(spec["output_root"]) / "root-status.json").read_bytes())
        assert result["complete"] is False and result["status"] == "incomplete"
        assert raised.value.receipt["accounting"]["calls"] == 284
    else:
        result = driver.execute(specpath, approval)
    assert result["accounting"]["calls"] == 284
    assert result["accounting"]["prior_calls"] == 283 and result["accounting"]["new_calls"] == 1
    assert result["accounting"]["elapsed_seconds"] >= 214.23125769998296
    assert driver.verify_campaign(spec["output_root"]) == result
    assert not result["complete"]


@pytest.mark.parametrize("tamper", ["held_status", "prior_calls", "prior_seconds", "new_calls", "elapsed", "orphan", "digest_omission"])
def test_continuation_root_reconciles_authority_accounting_and_leaf_inventory(tmp_path, tamper):
    oldroot, _, oldstate = root_fixture(tmp_path)
    oldstate["accounting"] = dict(calls=283, elapsed_seconds=214.23125769998296)
    for i, row in enumerate(oldstate["jobs"]):
        row.update(status="failed" if i < 8 else "cancelled", calls=283 if i == 0 else 0)
    driver.write(oldroot / "root-status.json", oldstate)
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"authored archive")
    root = tmp_path / "new"
    root.mkdir()
    spec = driver.bind_continuation(full_spec(root), prior_root=oldroot,
        prior_root_sha256=hashlib.sha256((oldroot / "root-status.json").read_bytes()).hexdigest(),
        prior_spec_sha256=oldstate["spec_sha256"], prior_archive_path=archive,
        prior_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    driver.write(root / "execution-spec.json", spec)
    sha = hashlib.sha256((root / "execution-spec.json").read_bytes()).hexdigest()
    driver.write(root / "independent-approval.json", dict(approved=True, spec_sha256=sha))
    held = {r["job_id"]: r for r in driver.continuation_plan(spec)["held"]}
    state = dict(complete=False, status="incomplete", spec_sha256=sha,
        jobs=[held.get(j["job_id"], dict(job_id=j["job_id"], status="unrun", calls=0,
              artifact_sha256={})) for j in spec["jobs"]],
        accounting=dict(calls=283, prior_calls=283, new_calls=0,
            prior_elapsed_seconds=214.23125769998296, new_elapsed_seconds=1,
            elapsed_seconds=215.23125769998296))
    driver.write(root / "root-status.json", state)
    driver.verify_campaign(root)
    if tamper == "held_status":
        state["jobs"][0]["status"] = "unrun"
    elif tamper == "prior_calls":
        state["accounting"].update(calls=0, prior_calls=0)
    elif tamper == "prior_seconds":
        state["accounting"]["prior_elapsed_seconds"] = 0
    elif tamper == "new_calls":
        state["accounting"]["new_calls"] = 99
    elif tamper == "elapsed":
        state["accounting"]["elapsed_seconds"] = 0
    else:
        output = Path(spec["jobs"][8]["output"])
        output.mkdir(parents=True)
        driver.write(output / "root-status.json", dict(complete=True, status="complete_verified"))
        if tamper == "digest_omission":
            state["jobs"][8].update(status="failed")
    driver.write(root / "root-status.json", state)
    with pytest.raises(ValueError, match="held|accounting|orphan|inventory"):
        driver.verify_campaign(root)


def test_new_protocol_routes_full_preflight_without_legacy_or_pilot_bypass(tmp_path, monkeypatch):
    spec = full_spec(tmp_path / "output")
    spec.update(protocol="beam-full-continuation@v2", data_dir="unused", upstream_dir="unused")
    source = tmp_path / "pinned.py"
    source.write_bytes(b"# authored byte pin\n")
    spec["files"] = [dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())]
    specpath, approval = tmp_path / "spec.json", tmp_path / "approval.json"
    driver.write(specpath, spec)
    driver.write(approval, dict(approved=True, reviewer="authored-independent", reviewer_kind="AI-assisted",
        spec_sha256=hashlib.sha256(specpath.read_bytes()).hexdigest()))
    seen = []
    monkeypatch.setattr(driver, "frozen_inventory", lambda *a: spec["files"])
    monkeypatch.setattr(driver, "full_dependency_preflight", lambda s: seen.append("full"))
    monkeypatch.setattr(driver, "continuation_plan", lambda s: seen.append("continuation"))
    monkeypatch.setattr(driver, "dependency_preflight", lambda *a: pytest.fail("pilot bypass"))
    assert driver.verify_execution_spec(specpath, approval) == spec
    assert seen == ["full", "continuation"]


@pytest.mark.parametrize("tamper", ["quality_status", "status", "na_system", "leaf_calls", "case_plan", "false_fit"])
def test_root_binds_leaf_na_policy_calls_and_cases(tmp_path, monkeypatch, tamper):
    root, spec, state = root_fixture(tmp_path)
    record = spec["jobs"][6]  # full_context, not a typed/native input admission.
    output = Path(record["output"])
    output.mkdir(parents=True)
    plan = [dict(case_id=q, system="full_context", conversation_id=record["history_id"],
                 storage_split=record["split"]) for q in record["case_ids"]]
    driver.write(output / "plan.json", plan)
    admission = [dict(sample_id=q, prompt_tokens=5000, fits=False) for q in record["case_ids"]]
    driver.write(output / "full-context-admission.json", admission)
    leaf = dict(complete=False, status="not_applicable_context_overflow", system="full_context",
        case_ids=record["case_ids"], generation_calls=0, plan_sha256=driver.digest(plan),
        artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()})
    driver.write(output / "root-status.json", leaf)
    saved = dict(quality_status=leaf["status"], result=leaf,
        root_status_sha256=hashlib.sha256((output / "root-status.json").read_bytes()).hexdigest())
    row = state["jobs"][6]
    row.update(status="complete", result=saved)
    def seal():
        row["artifact_sha256"] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in output.iterdir()}
        driver.write(root / "root-status.json", state)
    seal()
    driver.verify_campaign(root)
    if tamper == "quality_status":
        saved["quality_status"] = "invented_na"
    elif tamper == "status":
        leaf["status"] = "invented_na"
    elif tamper == "na_system":
        leaf["status"] = saved["quality_status"] = "input_not_admitted_missing_source_clock"
    elif tamper == "leaf_calls":
        leaf["generation_calls"] = 1
    elif tamper == "false_fit":
        for item in admission:
            item["prompt_tokens"] = 100
        driver.write(output / "full-context-admission.json", admission)
        leaf["artifact_sha256"]["full-context-admission.json"] = hashlib.sha256((output / "full-context-admission.json").read_bytes()).hexdigest()
    elif tamper == "case_plan":
        plan[0]["system"] = "gluomem"
        driver.write(output / "plan.json", plan)
        leaf["plan_sha256"] = driver.digest(plan)
        leaf["artifact_sha256"]["plan.json"] = hashlib.sha256((output / "plan.json").read_bytes()).hexdigest()
    driver.write(output / "root-status.json", leaf)
    saved["root_status_sha256"] = hashlib.sha256((output / "root-status.json").read_bytes()).hexdigest()
    seal()
    with pytest.raises(ValueError, match="status|policy|calls|case|system"):
        driver.verify_campaign(root)


def test_full_root_aggregation_keeps_all_planned_coverage_and_rejects_forgery(tmp_path):
    root, spec, state = root_fixture(tmp_path)
    summary = driver._campaign_summary(spec, state)
    assert summary["coverage"] == dict(histories=100, questions=2000, conditions=700,
        question_system_slots=14000, statuses=dict(unrun=700), quality_statuses={})
    assert summary["completed_native_report"] is None
    state.update(summary)
    driver.write(root / "root-status.json", state)
    driver.verify_campaign(root)
    state["coverage"]["conditions"] = 699
    driver.write(root / "root-status.json", state)
    with pytest.raises(ValueError, match="aggregate|coverage"):
        driver.verify_campaign(root)


def test_executor_refuses_existing_root_without_mutating_a_single_byte(tmp_path):
    from gluomem_bench.bench.rack import RackProfile
    root, spec, state = root_fixture(tmp_path)
    spec["profile"] = RackProfile(base_url="http://localhost:1/v1", model="authored/Qwen",
        profile_version="role-budgets-v6", generation_budgets=dict(extraction=4096,
        consolidation=4096, reader=4096, judge=4096)).model_dump(mode="json")
    before = {p: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(FileExistsError):
        driver._execute_full(spec, root / "execution-spec.json", root / "independent-approval.json")
    assert {p: p.read_bytes() for p in root.iterdir()} == before


def test_root_refuses_unknown_status_and_unaccounted_calls(tmp_path):
    root, _, state = root_fixture(tmp_path)
    driver.verify_campaign(root)
    state["jobs"][0]["status"] = "mystery"
    driver.write(root / "root-status.json", state)
    with pytest.raises(ValueError, match="status"):
        driver.verify_campaign(root)
    state["jobs"][0].update(status="failed", calls=2)
    driver.write(root / "root-status.json", state)
    with pytest.raises(ValueError, match="accounting"):
        driver.verify_campaign(root)
