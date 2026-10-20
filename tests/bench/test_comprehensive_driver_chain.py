"""Offline authored chain contracts. No live endpoint or old-root writes."""

import copy
import hashlib
import json
from pathlib import Path

import pytest


def test_reviewed_delta_keeps_normal_path_ast_exact(tmp_path):
    import hashlib

    from gluomem_bench.bench import comprehensive_driver as d

    old = tmp_path / "old.py"
    new = tmp_path / "new.py"
    old.write_bytes(b"def run(x):\n try:\n  return reader(x)\n except OSError:\n  raise\n")
    new.write_bytes(
        b"def run(x):\n try:\n  return reader(x)\n"
        b" except OSError as error:\n  record(error)\n  raise\n"
    )
    entry = dict(
        old_path=str(old),
        old_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
        new_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),
        exception_symbols=["run"],
        telemetry_symbols=[],
    )
    d._verify_guard_delta(new, entry)
    # Wrapping unchanged normal instructions and adding exception-only imports
    # are guard changes, not a vector/reader algorithm exemption.
    new.write_bytes(
        b"from urllib.error import HTTPError\ndef run(x):\n try:\n"
        b"  try:\n   return reader(x)\n  except HTTPError:\n   raise\n"
        b" except OSError as error:\n  record(error)\n  raise\n"
    )
    entry["new_sha256"] = hashlib.sha256(new.read_bytes()).hexdigest()
    entry["telemetry_imports"] = ["from urllib.error import HTTPError"]
    d._verify_guard_delta(new, entry)
    # Cleanup structure is normal-path semantics: do not flatten a finally into
    # an unreachable statement after return.
    old.write_bytes(b"def run(x):\n try:\n  return reader(x)\n finally:\n  cleanup()\n")
    new.write_bytes(b"def run(x):\n return reader(x)\n cleanup()\n")
    entry.update(
        old_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
        new_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),
        telemetry_imports=[],
    )
    with pytest.raises(ValueError, match="normal-path"):
        d._verify_guard_delta(new, entry)
    new.write_bytes(new.read_bytes().replace(b"reader(x)", b"reader(0)"))
    entry["new_sha256"] = hashlib.sha256(new.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="normal-path"):
        d._verify_guard_delta(new, entry)


def _authored_previous(tmp_path):
    from gluomem_bench.bench import comprehensive_driver as d

    jobs = [
        dict(
            job_id=f"j{i}",
            sample_id="q",
            history_id="q",
            system="no_memory",
            output=str(tmp_path / "old" / "jobs" / f"j{i}"),
            disposition="dispatch",
            prior_row=dict(job_id=f"j{i}", status="cancelled"),
        )
        for i in range(2)
    ]
    spec = dict(
        protocol=d.CONTINUATION_PROTOCOL,
        output_root=str(tmp_path / "old"),
        jobs=jobs,
        sample_ids=["q"],
        systems=["no_memory"],
        files=d.frozen_inventory(str(Path(d.__file__)), str(Path(d.__file__))),
        dataset=dict(path=str(Path(d.__file__)), sha256=hashlib.sha256(Path(d.__file__).read_bytes()).hexdigest()),
        judge_source=str(Path(d.__file__)),
        profile=dict(timeout_seconds=600),
        campaign_policy=dict(max_calls=10000, max_seconds=20000, max_workers=1, max_in_flight=1),
        cohorts=dict(legacy="timeout120_parallel8", fresh="timeout600_serial1", pool_quality=False),
        cognee_unrun=[dict(sample_id="q", system="cognee", status="unrun", admitted=False)],
    )
    prior = dict(
        jobs=[
            dict(job_id="j0", status="failed", calls=8225, cohort="fresh", artifact_sha256={}),
            dict(job_id="j1", status="cancelled", calls=0, cohort="fresh", artifact_sha256={}),
        ],
        accounting=dict(calls=8225, elapsed_seconds=14937.567625900003),
    )
    return spec, prior, dict(extra_generation_attempts=None, extra_tokens=None)


def test_chain_builder_binds_immediate_predecessor_and_exact_cumulative_debit(
    tmp_path, monkeypatch
):
    from gluomem_bench.bench import comprehensive_driver as d

    previous = _authored_previous(tmp_path)
    monkeypatch.setattr(d, "_load_chained_predecessor", lambda b, **kw: copy.deepcopy(previous))
    monkeypatch.setattr(d, "_chain_semantic_bridge", lambda *a: [])
    spec = d.build_chained_continuation_execution_spec(
        predecessor_binding=dict(root=str(tmp_path / "old")),
        output_root=tmp_path / "new",
        semantic_delta_manifest=dict(path="reviewed.json", sha256="a" * 64),
        repaired_cohort="timeout600_serial1_repaired_guard_v1",
    )
    assert spec["protocol"] == d.CHAINED_PROTOCOL
    assert spec["dispatch_job_ids"] == ["j1"]
    assert spec["prior_accounting"] == previous[1]["accounting"]
    assert spec["prior_spend_uncertainty"]["extra_generation_attempts"] is None
    assert spec["cohorts"] == dict(
        previous[0]["cohorts"], repaired="timeout600_serial1_repaired_guard_v1"
    )
    assert d.validate_chained_continuation_spec(spec) == previous[1]
    for mutation in ("drop", "duplicate", "spend", "retry", "root", "cohort", "cap", "source"):
        bad = copy.deepcopy(spec)
        if mutation == "drop":
            bad["jobs"].pop()
        if mutation == "duplicate":
            bad["jobs"][-1] = bad["jobs"][0]
        if mutation == "spend":
            bad["prior_accounting"]["calls"] = 2367
        if mutation == "retry":
            bad["dispatch_job_ids"].insert(0, "j0")
        if mutation == "root":
            bad["jobs"][-1]["output"] = previous[0]["jobs"][-1]["output"]
        if mutation == "cohort":
            bad["cohorts"]["fresh"] = "relabelled"
        if mutation == "cap":
            bad["campaign_policy"]["max_calls"] += 1
        if mutation == "source":
            bad["dataset"]["path"] = "different-source.json"
        with pytest.raises(ValueError):
            d.validate_chained_continuation_spec(bad)
    assert not (tmp_path / "new").exists()


def test_chain_execution_uses_one_outer_budget_and_verifies_incomplete_abort(tmp_path, monkeypatch):
    from test_comprehensive_driver import (
        test_authored_end_to_end_driver_accounts_and_offline_verifies,
    )

    test_authored_end_to_end_driver_accounts_and_offline_verifies(
        tmp_path, monkeypatch, "role-budgets-v6", False, "timeout", "chain"
    )
    saved = json.loads((tmp_path / "out/root-status.json").read_bytes())
    assert saved["prior_spend_uncertainty"] == dict(extra_generation_attempts=None)


@pytest.mark.parametrize("fault", ["summary", True])
def test_chain_abort_has_readonly_verifiable_incomplete_authority(tmp_path, monkeypatch, fault):
    from test_comprehensive_driver import (
        test_authored_end_to_end_driver_accounts_and_offline_verifies,
    )

    test_authored_end_to_end_driver_accounts_and_offline_verifies(
        tmp_path, monkeypatch, "role-budgets-v6", fault, False, "chain"
    )
    from gluomem_bench.bench import comprehensive_driver as d

    saved = json.loads((tmp_path / "out/root-status.json").read_bytes())
    assert d.verify_campaign(tmp_path / "out") == saved
    assert saved["publication"] == "incomplete" and saved["complete"] is False
    assert saved["accounting"]["calls"] == 9


def _real_binding(root):
    import hashlib

    base = root.parent
    receipt_path = base / "analysis/longmemeval-continuation-v3-postexit-audit.hashreceipt.json"
    receipt = json.loads(receipt_path.read_bytes())

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    binding = dict(
        root=str(root),
        spec_sha256="9eb49d8d625c08e5da4bfc9235433f2b28dfb39970d592e67b61c00b0ef9c7e2",
        root_sha256="87d4ea6a9b4248ce0e22779af815ecc57fef1425a4bcb4e33042062a822a46f4",
        archive_path=str(base / "longmemeval-continuation-runtime-v3/candidate-producers.zip"),
        archive_sha256="855cbdd837a9cb2dcc4c4e437205afbe2c344a30dcde147e65830a7ff8288421",
    )
    for key, name in [
        ("review", "review/longmemeval-continuation-review-v3.json"),
        ("approval", "review/longmemeval-continuation-parent-approval-v3.json"),
        ("auditor_code", "analysis/longmemeval-continuation-v3-postexit-verify-v1.py"),
        ("audit_result", "analysis/longmemeval-continuation-v3-postexit-audit.json"),
        ("audit_receipt", "analysis/longmemeval-continuation-v3-postexit-audit.hashreceipt.json"),
    ]:
        path = base / name
        binding[key + "_path"] = str(path)
        binding[key + "_sha256"] = receipt["files"].get(str(path), sha(path))
    return binding


def test_real_v3_readonly_replay_all154_qa30_na_and_only3288_eligible(monkeypatch):
    import os

    from gluomem_bench.bench import comprehensive_driver as d

    location = os.environ.get("LME_CHAIN_REVIEW_ROOT")
    if not location:
        pytest.skip("set LME_CHAIN_REVIEW_ROOT to quiescent v3 root")
    root = Path(location)

    def no_db(*a, **kw):
        raise AssertionError("saved-leaf replay must not open DB")

    import sqlite3

    monkeypatch.setattr(sqlite3, "connect", no_db)
    spec, result, uncertainty = d._load_chained_predecessor(_real_binding(root))
    rows = d._chained_rows(spec, result, root.parent / "authored-not-created-chain-target")
    from collections import Counter

    assert Counter(r["disposition"] for r in rows) == dict(
        reuse_qa=154, reuse_na=30, retained_failure=28, dispatch=3288
    )
    assert result["accounting"]["calls"] == 8225
    assert uncertainty["extra_generation_attempts"] is None
    assert not (root.parent / "authored-not-created-chain-target").exists()


def test_chain_cycle_is_rejected_before_any_file_read(tmp_path):
    from gluomem_bench.bench import comprehensive_driver as d

    with pytest.raises(ValueError, match="cycle"):
        d._load_chained_predecessor(dict(root=str(tmp_path)), ancestry=(str(tmp_path.resolve()),))


def test_real_repaired_three_file_delta_preserves_algorithms(tmp_path, monkeypatch):
    import hashlib
    import os

    from gluomem_bench.bench import comprehensive_driver as d

    location = os.environ.get("LME_CHAIN_REVIEW_ROOT")
    if not location:
        pytest.skip("set LME_CHAIN_REVIEW_ROOT to quiescent v3 root")
    root = Path(location)
    previous = json.loads((root / "execution-spec.json").read_bytes())
    entries = []
    for name, exceptions, telemetry, imports in [
        ("suite_campaign.py", ["_run_history", "run_history"], [], [
            "from gluomem_bench.bench.embedding_runtime import safe_failure_diagnostic"]),
        ("parallel_campaign.py", ["run_jobs"], [], [
            "from gluomem_bench.bench.embedding_runtime import safe_failure_diagnostic"]),
        ("mem0.py", ["GuardedMem0SDK.create"], ["_capture_native_mem0_log",
            "NativeMem0Diagnostics.__init__", "NativeMem0Diagnostics.emit", "NativeMem0Diagnostics.close"], []),
        ("rack.py", ["RackClient._request"], ["RackClient._latch"], []),
        (
            "embedding_runtime.py",
            ["RemoteEmbedder._encode", "make_server.Handler.do_POST"],
            ["safe_failure_diagnostic"],
            [
                "from urllib.error import HTTPError",
                "from urllib.parse import parse_qsl, urlsplit, urlunsplit",
            ],
        ),
    ]:
        pin = next(
            f
            for f in previous["files"]
            if f["path"].replace(chr(92), "/").endswith("/bench/" + name)
        )
        actual = Path(d.__file__).parent / name
        statements = ["NativeMem0Diagnostics." + value for value in
            ("__doc__", "_capture_lock", "_captures", "_original_dispatch")] if name == "mem0.py" else []
        entries.append(
            dict(
                contract="longmemeval-privacy-guard-symbols@v2",
                relative_path=name,
                telemetry_statements=statements,
                symbol_bindings=d.guard_delta_symbol_bindings(
                    (root / "frozen-inputs" / pin["sha256"]).read_bytes(), actual.read_bytes(),
                    set(exceptions) | set(telemetry) | set(statements)),
                old_path=str(root / "frozen-inputs" / pin["sha256"]),
                old_sha256=pin["sha256"],
                new_sha256=hashlib.sha256(actual.read_bytes()).hexdigest(),
                exception_symbols=exceptions,
                telemetry_symbols=telemetry,
                telemetry_imports=imports,
            )
        )
    manifest = tmp_path / "authored-review-delta-NOT-APPROVAL.json"
    manifest.write_bytes(
        json.dumps(dict(protocol="longmemeval-reviewed-guard-delta@v2", files=entries)).encode()
    )
    manifest_binding = dict(
        path=str(manifest), sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    bridge = d._chain_semantic_bridge(previous, manifest_binding)
    assert bridge
    # Launch admission now enumerates actual private imports and runtime bytes.
    spec = d.build_chained_continuation_execution_spec(
        predecessor_binding=_real_binding(root),
        output_root=tmp_path / "not-launched",
        semantic_delta_manifest=manifest_binding,
        repaired_cohort="timeout600_serial1_repaired_guard_v1",
    )
    assert len(spec["jobs"]) == 3500 and len(spec["dispatch_job_ids"]) == 3288
    assert len(spec["cognee_unrun"]) == 500
    assert spec["prior_accounting"]["calls"] == 8225
    assert not (tmp_path / "not-launched").exists()
    # A later chain hop compares to its immediate predecessor's repaired bytes,
    # not v1 guard bytes, while retaining original core/prompt semantics.
    manifest.write_bytes(
        json.dumps(dict(protocol="longmemeval-reviewed-guard-delta@v2", files=[])).encode()
    )
    d._chain_semantic_bridge(
        spec, dict(path=str(manifest), sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    )


def test_chain_locked_authority_retains_independent_incomplete_revocation(tmp_path, monkeypatch):
    import os

    from gluomem_bench.bench import comprehensive_driver as d
    from gluomem_bench.bench.parallel_campaign import CampaignBudget, CampaignReceiptError

    budget = CampaignBudget(
        max_calls=10000,
        max_seconds=20000,
        prior_calls=8225,
        prior_elapsed_seconds=14937.567625900003,
    )
    authority = tmp_path / "root-status.json"
    original_write, original_unlink, original_read = Path.write_bytes, Path.unlink, Path.read_bytes
    promoted = [False]
    replace = os.replace

    def promote(a, b):
        replace(a, b)
        promoted[0] = True

    def locked_write(p, data):
        if promoted[0] and p == authority:
            raise PermissionError("authored locked authority")
        return original_write(p, data)

    def locked_unlink(p, *a, **kw):
        if promoted[0] and p == authority:
            raise PermissionError("authored locked authority")
        return original_unlink(p, *a, **kw)

    def broken_read(p):
        if promoted[0] and p == authority:
            raise OSError("authored final readback fault")
        return original_read(p)

    monkeypatch.setattr(os, "replace", promote)
    monkeypatch.setattr(Path, "write_bytes", locked_write)
    monkeypatch.setattr(Path, "unlink", locked_unlink)
    monkeypatch.setattr(Path, "read_bytes", broken_read)
    monkeypatch.setattr(d, "verify_campaign", lambda *a, **kw: None)
    with pytest.raises(CampaignReceiptError) as raised:
        d._publish_continuation(
            tmp_path, dict(complete=False, execution_complete=True, jobs=[]), budget
        )
    revoked = json.loads((tmp_path / "root-incomplete.json").read_bytes())
    assert revoked["publication"] == "incomplete" and revoked["execution_complete"] is False
    assert revoked["accounting"]["calls"] == raised.value.receipt["accounting"]["calls"] == 8225


def test_chain_e2e_rejects_rehashed_qa_summary_and_cancelled_quality(tmp_path, monkeypatch):
    from test_comprehensive_driver import (
        test_authored_end_to_end_driver_accounts_and_offline_verifies,
    )

    from gluomem_bench.bench import comprehensive_driver as d

    test_authored_end_to_end_driver_accounts_and_offline_verifies(
        tmp_path, monkeypatch, "role-budgets-v6", "summary", False, "chain"
    )
    root = tmp_path / "out"
    status = root / "root-status.json"
    saved = json.loads(status.read_bytes())
    forged = copy.deepcopy(saved)
    forged["jobs"][1]["result"]["summary"]["accuracy"] = 123
    status.write_bytes(json.dumps(forged).encode())
    progress_path = root / "coordination/progress.json"
    progress = json.loads(progress_path.read_bytes())
    progress["jobs"][0]["result"] = forged["jobs"][1]["result"]
    progress_path.write_bytes(json.dumps(progress).encode())
    with pytest.raises(ValueError, match="QA"):
        d.verify_campaign(root)


@pytest.mark.parametrize("revoked", [False, True])
def test_authored_self_chain_cycle_is_rejected_recursively(tmp_path, revoked):
    import hashlib

    from gluomem_bench.bench import comprehensive_driver as d

    def save(name, value):
        path = tmp_path / name
        path.write_bytes(json.dumps(value).encode())
        return str(path), hashlib.sha256(path.read_bytes()).hexdigest()

    binding = dict(root=str(tmp_path))
    _, binding["spec_sha256"] = save(
        "execution-spec.json",
        dict(protocol=d.CHAINED_PROTOCOL, predecessor_binding=dict(root=str(tmp_path))),
    )
    _, binding["root_sha256"] = save("root-incomplete.json" if revoked else "root-status.json", {})
    if revoked:
        (tmp_path / "root-status.json").write_bytes(b"locked or torn former authority")
    binding["archive_path"], binding["archive_sha256"] = save("archive.json", {})
    shared = dict(
        approved=True,
        blockers=[],
        spec_sha256=binding["spec_sha256"],
        archive_sha256=binding["archive_sha256"],
    )
    binding["review_path"], binding["review_sha256"] = save("review.json", shared)
    binding["approval_path"], binding["approval_sha256"] = save(
        "independent-approval.json",
        dict(
            shared,
            parent_execution_authorized=True,
            independent_review_sha256=binding["review_sha256"],
        ),
    )
    with pytest.raises(ValueError, match="cycle"):
        d._load_chained_predecessor(binding)


def test_chain_root_requires_delta_digest_in_saved_approval(tmp_path, monkeypatch):
    from test_comprehensive_driver import (
        test_authored_end_to_end_driver_accounts_and_offline_verifies,
    )

    from gluomem_bench.bench import comprehensive_driver as d

    test_authored_end_to_end_driver_accounts_and_offline_verifies(
        tmp_path, monkeypatch, "role-budgets-v6", "summary", False, "chain"
    )
    root = tmp_path / "out"
    approval_path = root / "independent-approval.json"
    approval = json.loads(approval_path.read_bytes())
    approval["semantic_delta_manifest_sha256"] = "b" * 64
    approval_path.write_bytes(json.dumps(approval).encode())
    with pytest.raises(ValueError, match="delta"):
        d.verify_campaign(root)


def test_delta_cannot_relabel_normal_reader_as_telemetry(tmp_path):
    import hashlib

    from gluomem_bench.bench import comprehensive_driver as d

    old, new = tmp_path / "old.py", tmp_path / "new.py"
    old.write_bytes(b"def run(x):\n return reader(x)\n")
    new.write_bytes(b"def run(x):\n return reader(0)\n")
    entry = dict(
        old_path=str(old),
        old_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
        new_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),
        exception_symbols=[],
        telemetry_symbols=["run"],
    )
    with pytest.raises(ValueError, match="telemetry"):
        d._verify_guard_delta(new, entry)


def test_chain_unknown_prior_attempts_cannot_be_rehashed_to_zero(tmp_path, monkeypatch):
    from test_comprehensive_driver import (
        test_authored_end_to_end_driver_accounts_and_offline_verifies,
    )

    from gluomem_bench.bench import comprehensive_driver as d

    test_authored_end_to_end_driver_accounts_and_offline_verifies(
        tmp_path, monkeypatch, "role-budgets-v6", "summary", False, "chain"
    )
    status = tmp_path / "out/root-status.json"
    result = json.loads(status.read_bytes())
    result["prior_spend_uncertainty"]["extra_generation_attempts"] = 0
    status.write_bytes(json.dumps(result).encode())
    with pytest.raises(ValueError, match="uncertainty"):
        d.verify_campaign(tmp_path / "out")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket

    def denied(*a, **kw):
        raise OSError("chain tests deny all sockets")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def test_chain_rows_dispatch_only_latest_and_original_never_attempted(tmp_path):
    from gluomem_bench.bench import comprehensive_driver as d

    old_jobs = [
        dict(
            job_id=f"j{i}",
            sample_id="q",
            history_id="q",
            system="no_memory",
            output=str(tmp_path / "old" / "jobs" / f"j{i}"),
            disposition="dispatch",
            prior_row=dict(job_id=f"j{i}", status="cancelled"),
        )
        for i in range(4)
    ]
    previous = dict(
        jobs=[
            dict(
                job_id="j0",
                status="complete",
                calls=2,
                result=dict(quality_status="complete_verified"),
                cohort="fresh",
            ),
            dict(
                job_id="j1",
                status="complete",
                calls=0,
                result=dict(quality_status="not_applicable_context_overflow"),
                cohort="fresh",
            ),
            dict(job_id="j2", status="failed", calls=1, cohort="fresh"),
            dict(job_id="j3", status="cancelled", calls=0, cohort="fresh"),
        ]
    )
    rows = d._chained_rows(dict(jobs=old_jobs), previous, tmp_path / "next")
    assert [r["disposition"] for r in rows] == [
        "reuse_qa",
        "reuse_na",
        "retained_failure",
        "dispatch",
    ]
    assert [r["prior_cohort"] for r in rows] == ["fresh"] * 4
    assert rows[0]["prior_row"] == previous["jobs"][0]
    assert rows[3]["output"] == str(tmp_path / "next" / "jobs" / "j3")
    assert rows[0]["output"] == old_jobs[0]["output"]
    for bad in ("spent", "original-attempt", "old-directory", "duplicate", "cancelled-qa"):
        jobs, prior = copy.deepcopy(old_jobs), copy.deepcopy(previous)
        if bad == "spent":
            prior["jobs"][-1]["calls"] = 1
        if bad == "original-attempt":
            jobs[-1]["prior_row"]["status"] = "failed"
        if bad == "old-directory":
            Path(jobs[-1]["output"]).mkdir(parents=True)
        if bad == "duplicate":
            prior["jobs"][-1] = prior["jobs"][0]
        if bad == "cancelled-qa":
            prior["jobs"][-1]["result"] = dict(quality_status="complete_verified", score=0)
        with pytest.raises(ValueError):
            d._chained_rows(dict(jobs=jobs), prior, tmp_path / "next")
        if bad == "old-directory":
            Path(jobs[-1]["output"]).rmdir()
