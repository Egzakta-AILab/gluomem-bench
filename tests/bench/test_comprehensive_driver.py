"""Execution admission is byte-bound and happens before model clients exist."""

import hashlib
import json

import pytest


@pytest.fixture(autouse=True)
def deny_external_sockets(monkeypatch):
    import socket

    def denied(*args, **kwargs):
        raise OSError("offline driver tests forbid real sockets")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def test_continuation_plan_retains_failed_spend_and_dispatches_only_never_run(tmp_path):
    import gluomem_bench.bench.comprehensive_driver as d

    old_spec = {
        "jobs": [
            dict(
                job_id=f"j{i}",
                sample_id="q",
                history_id="q",
                system=s,
                output=str(tmp_path / "old" / f"j{i}"),
                call_budget=10,
                seconds_budget=20,
            )
            for i, s in enumerate(("mem0", "full_context", "gluomem", "bm25"))
        ]
    }
    old = {
        "jobs": [
            dict(
                job_id="j0",
                status="complete",
                calls=2,
                result={"quality_status": "complete_verified"},
            ),
            dict(
                job_id="j1",
                status="complete",
                calls=0,
                result={"quality_status": "not_applicable_context_overflow"},
            ),
            dict(job_id="j2", status="failed", calls=7, failure={"type": "quality"}),
            dict(job_id="j3", status="cancelled"),
        ]
    }
    reuse = dict(reuse_qa_job_ids=["j0"], not_applicable_job_ids=["j1"])
    rows = d._continuation_rows(old_spec, old, reuse, ["j2"], tmp_path / "new")
    assert [r["disposition"] for r in rows] == [
        "reuse_qa",
        "reuse_na",
        "retained_quality_failure",
        "dispatch",
    ]
    assert rows[2]["prior_row"] == old["jobs"][2]
    assert rows[3]["output"] == str(tmp_path / "new" / "jobs" / "j3")
    assert rows[0]["output"] == old_spec["jobs"][0]["output"]
    old["jobs"][3]["calls"] = 1
    with pytest.raises(ValueError, match="never-run"):
        d._continuation_rows(old_spec, old, reuse, ["j2"], tmp_path / "new")


@pytest.fixture
def immutable_lme_root():
    """Optional real, quiescent full500 evidence; never generate or edit it."""
    import os
    from pathlib import Path

    root = os.environ.get("LME_REVIEW_ROOT")
    if not root:
        pytest.skip("set LME_REVIEW_ROOT to immutable post-exit full500 evidence")
    return Path(root)


def test_continuation_public_builder_replays_pinned_real_old_root(
    immutable_lme_root, tmp_path, monkeypatch
):
    import gluomem_bench.bench.comprehensive_driver as d

    candidate = immutable_lme_root.parent / "longmemeval-continuation-candidate-v2"
    prepared = json.loads((candidate / "candidate-spec.json").read_bytes())
    conditions = json.loads((candidate / "original-conditions-3500.json").read_bytes())
    binding = dict(
        original_root=str(immutable_lme_root),
        spec_sha256=prepared["original_pins"]["spec"],
        root_sha256=prepared["original_pins"]["root"],
        archive_path=prepared["original_archive"],
        archive_sha256=prepared["original_pins"]["archive"],
        verifier_path=str(
            candidate / "runtime/gluomem-bench/gluomem_bench/bench/comprehensive_driver.py"
        ),
        verifier_sha256=prepared["original_pins"]["verifier"],
    )
    quality = [
        r["job_id"]
        for r in conditions
        if r["continuation_disposition"] == "retained_quality_failure_no_retry"
    ]
    # Exercise the historical v3 producer with its actual original suite guard.
    # The repaired live builder MUST still reject that guard mismatch; do not
    # weaken its legacy bridge or call the regression obsolete.
    import types
    import zipfile
    from pathlib import Path

    with pytest.raises(ValueError, match="original binding hash differs"):
        d.build_continuation_execution_spec(
            original_binding=binding, quality_failure_job_ids=quality, output_root=tmp_path / "rejected"
        )
    archive = immutable_lme_root.parent / "longmemeval-continuation-runtime-v3/candidate-producers.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == "855cbdd837a9cb2dcc4c4e437205afbe2c344a30dcde147e65830a7ff8288421"
    historical_path = archive.parent / "runtime/gluomem-bench/gluomem_bench/bench/comprehensive_driver.py"
    with zipfile.ZipFile(archive) as frozen:
        material = frozen.read("runtime/gluomem-bench/gluomem_bench/bench/comprehensive_driver.py")
    assert material == historical_path.read_bytes()
    historical = types.ModuleType("_pinned_original_v3_builder")
    historical.__file__ = str(historical_path)
    exec(compile(material, str(historical_path), "exec"), historical.__dict__)
    d = historical
    # Inventory admission is tested separately; replay uses actual unchanged code.
    monkeypatch.setattr(d, "frozen_inventory", lambda *a: [])
    spec = d.build_continuation_execution_spec(
        original_binding=binding, quality_failure_job_ids=quality, output_root=tmp_path / "new"
    )
    assert spec["protocol"] == d.CONTINUATION_PROTOCOL
    assert len(spec["jobs"]) == 3500 and len(spec["cognee_unrun"]) == 500
    assert len(spec["dispatch_job_ids"]) == 3457
    assert spec["prior_accounting"] == {"calls": 2367, "elapsed_seconds": 1013.4239649000228}
    assert spec["profile"]["timeout_seconds"] == 600
    assert (
        d.validate_continuation_spec(spec)["reuse_qa_job_ids"]
        == spec["reuse_receipt"]["reuse_qa_job_ids"]
    )
    for mutation in ("missing", "duplicate", "prior", "promote", "timeout", "reader"):
        bad = json.loads(json.dumps(spec))
        if mutation == "missing":
            bad["jobs"].pop()
        if mutation == "duplicate":
            bad["jobs"][-1] = bad["jobs"][0]
        if mutation == "prior":
            bad["prior_accounting"]["calls"] -= 1
        if mutation == "promote":
            bad["jobs"][0]["disposition"] = "dispatch"
        if mutation == "timeout":
            bad["profile"]["timeout_seconds"] = 120
        if mutation == "reader":
            bad["profile"]["generation_budgets"]["reader"] += 1
        with pytest.raises(ValueError):
            d.validate_continuation_spec(bad)
    assert not (tmp_path / "new").exists()


def test_continuation_bridge_checks_actual_runtime_not_original_live_paths(
    immutable_lme_root, tmp_path, monkeypatch
):
    import shutil

    import gluomem_bench.bench.comprehensive_driver as d

    original = json.loads((immutable_lme_root / "execution-spec.json").read_bytes())
    candidate = immutable_lme_root.parent / "longmemeval-continuation-candidate-v2"
    verifier = candidate / "runtime/gluomem-bench/gluomem_bench/bench/comprehensive_driver.py"
    actual = tmp_path / "gluomem_bench/bench"
    actual.mkdir(parents=True)
    for pin in original["files"]:
        name = pin["path"].replace("\\", "/")
        marker = "/gluomem_bench/bench/"
        if marker in name:
            target = actual / name.split(marker, 1)[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pin["path"], target)
    shutil.copyfile(d.__file__, actual / "comprehensive_driver.py")
    (actual / "baselines.py").write_bytes(b"# forged reader algorithm")
    monkeypatch.setattr(d, "__file__", str(actual / "comprehensive_driver.py"))
    with pytest.raises(ValueError, match="hash"):
        d._semantic_bridge(original, verifier)


@pytest.mark.parametrize("relative", ["__init__.py", "../artifacts/dspy/event_claims.json"])
def test_continuation_bridge_rejects_changed_core_algorithm(
    immutable_lme_root, monkeypatch, relative
):
    from pathlib import Path

    import gluomem

    import gluomem_bench.bench.comprehensive_driver as d

    original = json.loads((immutable_lme_root / "execution-spec.json").read_bytes())
    core = Path(gluomem.__file__).resolve().parent
    target = (core / relative).resolve()
    read = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda p: read(p) + b"\n# forged core" if p.resolve() == target else read(p),
    )
    verifier = (
        immutable_lme_root.parent
        / "longmemeval-continuation-candidate-v2/runtime/gluomem-bench"
        / "gluomem_bench/bench/comprehensive_driver.py"
    )
    with pytest.raises(ValueError, match="hash"):
        d._semantic_bridge(original, verifier)


def test_actual_complete_mem0_leaf_and_full_root_use_reconstructed_bindings(immutable_lme_root):
    from gluomem_bench.bench.comprehensive_driver import verify_campaign

    result = verify_campaign(immutable_lme_root)
    qa = [
        r
        for r in result["jobs"]
        if r["status"] == "complete" and r["result"]["quality_status"] == "complete_verified"
    ]
    assert len(result["jobs"]) == 3500
    assert len(qa) == 25
    assert sum(r["job_id"].endswith("--mem0") for r in qa) == 5
    assert result["complete"] is False


def test_versioned_readonly_reuse_binds_originals_and_excludes_failed_and_na(
    immutable_lme_root, monkeypatch
):
    from pathlib import Path

    import gluomem_bench.bench.comprehensive_driver as driver

    root = immutable_lme_root
    archive = root.parent / "longmemeval-full-producers-v1.zip"
    pins = dict(
        spec_sha256=hashlib.sha256((root / "execution-spec.json").read_bytes()).hexdigest(),
        root_sha256=hashlib.sha256((root / "root-status.json").read_bytes()).hexdigest(),
        archive_path=archive,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(Path, "write_bytes", lambda *a, **kw: pytest.fail("read-only verifier"))
    report = driver.verify_campaign_reuse_v2(root, **pins)
    original = json.loads((root / "root-status.json").read_bytes())
    assert report["verification_protocol"] == "longmemeval-readonly-reuse@v2"
    assert (
        report["verifier_sha256"] == hashlib.sha256(Path(driver.__file__).read_bytes()).hexdigest()
    )
    assert report["original_spec_sha256"] == pins["spec_sha256"]
    assert report["original_root_sha256"] == pins["root_sha256"]
    assert report["original_archive_sha256"] == pins["archive_sha256"]
    assert report["planned_job_ids"] == [r["job_id"] for r in original["jobs"]]
    assert len(report["reuse_qa_job_ids"]) == 25
    assert len(report["not_applicable_job_ids"]) == 5
    assert report["campaign_complete"] is False
    assert not set(report["reuse_qa_job_ids"]) & {
        r["job_id"] for r in original["jobs"] if r["status"] != "complete"
    }
    for name in ("spec_sha256", "root_sha256", "archive_sha256"):
        with pytest.raises(ValueError, match="original.*hash"):
            driver.verify_campaign_reuse_v2(root, **dict(pins, **{name: "0" * 64}))


@pytest.mark.parametrize(
    "field",
    [
        "mem0_source_inputs_sha256",
        "mem0_extractor_system_sha256",
        "profile_sha256",
        "dataset_sha256",
    ],
)
@pytest.mark.parametrize("forge_expected", [False, True])
def test_rehashed_mem0_leaf_and_self_declared_expected_cannot_override_native_pin(
    immutable_lme_root, monkeypatch, field, forge_expected
):
    """Virtual read overlay exercises real leaf/root bytes without editing originals."""
    from pathlib import Path

    from gluomem_bench.bench.comprehensive_driver import verify_campaign

    root = immutable_lme_root
    original_read = Path.read_bytes
    result = json.loads(original_read(root / "root-status.json"))
    row = next(r for r in result["jobs"] if r["job_id"] == "lme-000--mem0")
    leaf = root / "jobs" / row["job_id"]
    overlay = {}

    def save(path, value):
        overlay[path] = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        name = path.relative_to(root).as_posix()
        if name in row["artifact_sha256"]:
            row["artifact_sha256"][name] = hashlib.sha256(overlay[path]).hexdigest()

    protocol = json.loads(original_read(leaf / "protocol.json"))
    if field == "mem0_source_inputs_sha256":
        inputs = json.loads(original_read(leaf / "mem0-source-inputs.json"))
        inputs[0]["messages"][0]["content"] += " tampered"
        save(leaf / "mem0-source-inputs.json", inputs)
        forged = hashlib.sha256(
            json.dumps(inputs, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
    elif field == "mem0_extractor_system_sha256":
        source = json.loads(original_read(leaf / "mem0-source.json"))
        source["extractor_system"] += " tampered"
        save(leaf / "mem0-source.json", source)
        forged = hashlib.sha256(source["extractor_system"].encode()).hexdigest()
    elif field == "profile_sha256":
        profile = json.loads(original_read(leaf / "profile.json"))
        profile["max_generation_calls"] -= 1
        save(leaf / "profile.json", profile)
        from gluomem_bench.bench.rack import RackProfile

        forged = hashlib.sha256(
            RackProfile.model_validate(profile).model_dump_json().encode()
        ).hexdigest()
    else:
        forged = "0" * 64
    if field != "mem0_extractor_system_sha256":
        protocol[field] = forged
    save(leaf / "protocol.json", protocol)
    if forge_expected:
        row["result"]["expected_job"][field] = forged
    manifest = json.loads(original_read(leaf / "manifest.json"))
    for path, material in overlay.items():
        if path.parent == leaf:
            manifest["artifact_sha256"][path.name] = hashlib.sha256(material).hexdigest()
    save(leaf / "manifest.json", manifest)
    overlay[root / "root-status.json"] = json.dumps(result).encode()
    monkeypatch.setattr(
        Path, "read_bytes", lambda p: overlay[p] if p in overlay else original_read(p)
    )
    with pytest.raises(ValueError, match="binding|profile hash"):
        verify_campaign(root)


def test_failed_root_row_with_real_complete_leaf_is_never_reuse_qa(immutable_lme_root, monkeypatch):
    from pathlib import Path

    from gluomem_bench.bench.comprehensive_driver import verify_campaign_reuse_v2

    root = immutable_lme_root
    original_read = Path.read_bytes
    result = json.loads(original_read(root / "root-status.json"))
    row = next(r for r in result["jobs"] if r["job_id"] == "lme-000--mem0")
    assert (root / "jobs" / row["job_id"] / "manifest.json").exists()
    row["status"] = "failed"
    changed = json.dumps(result).encode()
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda p: changed if p == root / "root-status.json" else original_read(p),
    )
    archive = root.parent / "longmemeval-full-producers-v1.zip"
    report = verify_campaign_reuse_v2(
        root,
        spec_sha256=hashlib.sha256(original_read(root / "execution-spec.json")).hexdigest(),
        root_sha256=hashlib.sha256(changed).hexdigest(),
        archive_path=archive,
        archive_sha256=hashlib.sha256(original_read(archive)).hexdigest(),
    )
    assert row["job_id"] not in report["reuse_qa_job_ids"]
    assert len(report["reuse_qa_job_ids"]) == 24
    assert len(report["not_applicable_job_ids"]) == 5
    assert report["campaign_complete"] is False


@pytest.mark.parametrize("mutation", ["drop", "swap-native-order"])
def test_full_root_rehashed_spec_cannot_change_native_500_matrix(
    immutable_lme_root, monkeypatch, mutation
):
    from pathlib import Path

    from gluomem_bench.bench.comprehensive_driver import verify_campaign

    root = immutable_lme_root
    original_read = Path.read_bytes
    spec = json.loads(original_read(root / "execution-spec.json"))
    result = json.loads(original_read(root / "root-status.json"))
    approval = json.loads(original_read(root / "independent-approval.json"))
    if mutation == "drop":
        spec["jobs"].pop()
        result["jobs"].pop()
    else:
        spec["sample_ids"][:2] = reversed(spec["sample_ids"][:2])
        spec["jobs"][:14] = spec["jobs"][7:14] + spec["jobs"][:7]
        result["jobs"][:14] = result["jobs"][7:14] + result["jobs"][:7]
    raw = json.dumps(spec).encode()
    approval["spec_sha256"] = result["spec_sha256"] = hashlib.sha256(raw).hexdigest()
    overlay = {
        root / "execution-spec.json": raw,
        root / "root-status.json": json.dumps(result).encode(),
        root / "independent-approval.json": json.dumps(approval).encode(),
    }
    monkeypatch.setattr(
        Path, "read_bytes", lambda p: overlay[p] if p in overlay else original_read(p)
    )
    with pytest.raises(ValueError, match="matrix|dataset-ordered"):
        verify_campaign(root)


def test_full_builder_constructs_500_native_order_histories_and_seven_conditions(
    tmp_path, monkeypatch
):
    import gluomem_bench.bench.comprehensive_driver as driver

    source = tmp_path / "full-authored.json"
    rows = [
        dict(
            question_id=f"authored-{i}",
            question="Which editor?",
            answer="Neovim",
            question_type="single-session-user",
            question_date="2026/01/02",
            haystack_dates=["2026/01/01 (Thu) 10:00"],
            haystack_session_ids=["answer_s"],
            haystack_sessions=[[dict(role="user", content="I use Neovim.")]],
            answer_session_ids=["answer_s"],
        )
        for i in range(500)
    ]
    source.write_bytes(json.dumps(rows).encode())
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    judge = tmp_path / "authored-judge.py"
    judge.write_bytes(b"# authored scorer admission seam")
    settings = dict(
        profile=dict(
            base_url="http://localhost:1/v1",
            model="authored/model",
            profile_version="role-budgets-v6",
            max_generation_calls=16384,
            max_run_seconds=21600,
            generation_budgets=dict(extraction=16384, consolidation=4096, reader=4096, judge=1024),
        ),
        config_overrides={
            "benchmark.datasets.longmemeval_s": dict(
                url="https://example.invalid/data", file=source.name, sha256=sha
            ),
            "benchmark.datasets_dir": str(tmp_path),
        },
        embedding_url="http://localhost:2/v1",
        embedding_recipe={},
        raw_chunk_chars=1000,
        judge_source=str(judge),
        campaign_policy=dict(
            max_workers=8,
            max_in_flight=16,
            max_in_flight_tokens=2000000,
            max_calls=60000000,
            max_seconds=999999,
            stop_on_failure=False,
        ),
    )
    monkeypatch.setattr(
        "gluomem_bench.bench.upstream_scoring.load_longmemeval_prompt", lambda p: None
    )
    monkeypatch.setattr("httpx.Client", lambda *a, **kw: pytest.fail("offline preparation"))
    spec = driver.build_full_execution_spec(
        dataset=dict(name="longmemeval_s", path=str(source), sha256=sha),
        output_root=tmp_path / "full",
        settings=settings,
    )
    assert spec["sample_ids"] == [r["question_id"] for r in rows]
    assert len(spec["jobs"]) == 3500
    assert {j["system"] for j in spec["jobs"]} == set(driver.SUPPORTED_SYSTEMS)
    assert spec["scope"] == "all published development; no heldout claim"
    assert not (tmp_path / "full").exists()
    spec_path, approval = tmp_path / "spec.json", tmp_path / "approval.json"
    spec_path.write_bytes(json.dumps(spec).encode())
    approval.write_bytes(
        json.dumps(
            dict(
                approved=True,
                reviewer_kind="AI-assisted",
                spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            )
        ).encode()
    )
    assert driver.verify_execution_spec(spec_path, approval) == spec
    for change in ("remove-job", "duplicate-job", "reorder-dataset"):
        bad = json.loads(json.dumps(spec))
        if change == "remove-job":
            bad["jobs"].pop()
        if change == "duplicate-job":
            bad["jobs"][-1] = bad["jobs"][0]
        if change == "reorder-dataset":
            bad["sample_ids"].reverse()
        with pytest.raises(ValueError):
            driver.validate_full_spec(bad, [r["question_id"] for r in rows])


def test_full_context_common_ceiling_is_na_without_truncation_but_legacy_remains():
    from types import SimpleNamespace

    from gluomem_bench.bench.comprehensive_driver import full_context_fits

    profile = SimpleNamespace(
        output_budget=lambda task: 4096,
        context_window=131072,
        schema_reserve_tokens=2048,
        max_input_chars=1000000,
    )
    assert full_context_fits(
        dict(protocol="longmemeval-full-development@v1", evidence_token_budget=4096),
        profile,
        4096,
        100000,
    )
    assert not full_context_fits(
        dict(protocol="longmemeval-full-development@v1", evidence_token_budget=4096),
        profile,
        4097,
        100000,
    )
    assert full_context_fits(
        dict(protocol="local-qwen-development-pilot@v1"), profile, 4097, 100000
    )


def test_execution_gate_requires_exact_approval_and_unchanged_inputs(tmp_path):
    from gluomem_bench.bench.comprehensive_driver import verify_execution_spec

    source = tmp_path / "producer.py"
    source.write_bytes(b"# authored producer\n")
    spec = {
        "protocol": "local-qwen-development-pilot@v1",
        "files": [{"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
        "output_root": str(tmp_path / "fresh-output"),
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(json.dumps(spec).encode())
    approval = tmp_path / "approval.json"
    approval.write_bytes(
        json.dumps(
            {"approved": True, "reviewer_kind": "AI-assisted", "spec_sha256": "wrong"}
        ).encode()
    )
    with pytest.raises(ValueError, match="approval"):
        verify_execution_spec(spec_path, approval)
    approval.write_bytes(
        json.dumps(
            {
                "approved": True,
                "reviewer_kind": "AI-assisted",
                "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            }
        ).encode()
    )
    assert verify_execution_spec(spec_path, approval) == spec
    source.write_bytes(b"# changed after approval\n")
    with pytest.raises(ValueError, match="frozen file"):
        verify_execution_spec(spec_path, approval)


def test_execution_refuses_before_any_client_or_output_without_approval(tmp_path, monkeypatch):
    from gluomem_bench.bench.comprehensive_driver import execute_longmemeval

    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(
        json.dumps(
            {"protocol": "local-qwen-development-pilot@v1", "output_root": str(tmp_path / "out")}
        ).encode()
    )
    approval = tmp_path / "approval.json"
    approval.write_bytes(b'{"approved": false}')
    calls = []
    monkeypatch.setattr(
        "gluomem_bench.bench.rack.RackClient.__init__", lambda *args, **kwargs: calls.append(True)
    )
    with pytest.raises(ValueError, match="approval"):
        execute_longmemeval(spec_path, approval)
    assert calls == []
    assert not (tmp_path / "out").exists()


def test_native_scorer_bytes_are_admitted_before_job_scheduling(tmp_path):
    from gluomem_bench.bench.comprehensive_driver import execute_longmemeval

    scorer = tmp_path / "wrong-native-scorer.py"
    scorer.write_bytes(b"# authored wrong bytes\r\n")
    spec = {
        "protocol": "local-qwen-development-pilot@v1",
        "files": [{"path": str(scorer), "sha256": hashlib.sha256(scorer.read_bytes()).hexdigest()}],
        "judge_source": str(scorer),
        "output_root": str(tmp_path / "out"),
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(json.dumps(spec).encode())
    approval = tmp_path / "authored-approval.json"
    approval.write_bytes(
        json.dumps(
            {
                "approved": True,
                "reviewer_kind": "AI-assisted",
                "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            }
        ).encode()
    )
    with pytest.raises(ValueError, match="scorer byte hash mismatch"):
        execute_longmemeval(spec_path, approval)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("middle_failure", [False, True, "shared", "timeout"])
@pytest.mark.parametrize("root_fault", [False, True])
@pytest.mark.parametrize("version", ["role-budgets-v5", "role-budgets-v6"])
@pytest.mark.parametrize("continuation", [False, True])
def test_authored_end_to_end_driver_accounts_and_offline_verifies(
    tmp_path, monkeypatch, version, root_fault, middle_failure, continuation
):
    import httpx

    from gluomem_bench.bench.comprehensive_driver import execute_longmemeval
    from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe
    from gluomem_bench.bench.rack import GenerationBudgets, RackClient, RackProfile

    source = tmp_path / "authored.json"
    source.write_bytes(
        json.dumps(
            [
                {
                    "question_id": "authored-q",
                    "question": "What editor?",
                    "answer": "right",
                    "question_type": "single-session-user",
                    "question_date": "2026/01/02",
                    "haystack_dates": ["2026/01/01 (Thu) 10:00"],
                    "haystack_session_ids": ["s1"],
                    "haystack_sessions": [[{"role": "user", "content": "I use Neovim."}]],
                    "answer_session_ids": ["s1"],
                }
            ]
        ).encode()
    )
    if middle_failure:
        template = json.loads(source.read_bytes())[0]
        source.write_bytes(
            json.dumps(
                [
                    dict(
                        template,
                        question_id=f"authored-q{i}",
                        question="fail-middle" if i == 1 else "What editor?",
                    )
                    for i in range(3)
                ]
            ).encode()
        )
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    judge = tmp_path / "authored-judge.py"
    judge.write_bytes(b"# synthetic scorer seam, not native metric evidence\n")
    profile = RackProfile(
        base_url="http://localhost:8000/v1",
        model="example/served-model",
        profile_version=version,
        max_generation_calls=32,
        max_run_seconds=120,
        generation_budgets=GenerationBudgets(extraction=32, consolidation=32, reader=32, judge=32),
    )
    root = tmp_path / "out"
    recipe = EmbeddingRecipe()
    spec = {
        "protocol": "local-qwen-development-pilot@v1",
        "output_root": str(root),
        "files": [{"path": str(source), "sha256": sha}],
        "profile": profile.model_dump(mode="json"),
        "dataset": {"name": "longmemeval_s", "path": str(source), "sha256": sha},
        "sample_ids": ["authored-q"],
        "systems": ["no_memory"],
        "judge_source": str(judge),
        "embedding_url": "http://localhost:8766/v1",
        "embedding_recipe": recipe.as_dict(),
        "evidence_token_budget": 4096,
        "raw_chunk_chars": 1000,
        "config_overrides": {
            "benchmark.datasets.longmemeval_s": {
                "url": "https://example.invalid/authored.json",
                "file": source.name,
                "sha256": sha,
            },
            "benchmark.datasets_dir": str(tmp_path),
            "models.shared": {
                "provider": "shared-local",
                "provider_model": recipe.model,
                "revision": recipe.revision,
                "dim": recipe.dim,
            },
            "embedding.model": "shared",
            "benchmark.mem0.embedder_model": "shared",
            "storage.vector_dim": 768,
        },
        "jobs": [
            {
                "job_id": "authored-job",
                "history_id": "authored-q",
                "output": str(root / "jobs" / "authored-job"),
                "system": "no_memory",
                "sample_id": "authored-q",
                "call_budget": 32,
                "seconds_budget": 120,
            }
        ],
        "campaign_policy": {
            "max_workers": 1,
            "max_in_flight": 1,
            "max_in_flight_tokens": 10000,
            "max_calls": 32,
            "max_seconds": 300,
            "stop_on_failure": False,
        },
    }
    if middle_failure:
        spec["sample_ids"] = [f"authored-q{i}" for i in range(3)]
        spec["jobs"] = [
            dict(
                spec["jobs"][0],
                job_id=f"j{i}",
                sample_id=f"authored-q{i}",
                history_id=f"authored-q{i}",
                output=str(root / "jobs" / f"j{i}"),
            )
            for i in range(3)
        ]
        spec["campaign_policy"]["max_calls"] = 96
    global_abort = middle_failure == "shared" and version == "role-budgets-v6"
    expected_calls = 3 if global_abort else (5 if middle_failure else 2)
    if continuation:
        import gluomem_bench.bench.comprehensive_driver as d

        # Authored execution fixture; the exact3500 gate is tested on real evidence.
        monkeypatch.setattr(d, "validate_continuation_spec", lambda s: s["reuse_receipt"])
        monkeypatch.setattr(d, "frozen_inventory", lambda *a: spec["files"])
        spec["files"].append(
            dict(path=str(judge), sha256=hashlib.sha256(judge.read_bytes()).hexdigest())
        )
        if continuation == "chain":
            monkeypatch.setattr(d, "validate_chained_continuation_spec", lambda s, **kw: {})
        spec.update(
            protocol=d.CHAINED_PROTOCOL if continuation == "chain" else d.CONTINUATION_PROTOCOL,
            prior_accounting=dict(calls=7, elapsed_seconds=3.5),
            reuse_receipt={},
            cognee_unrun=[
                dict(sample_id="authored-q", system="cognee", status="unrun", admitted=False)
            ],
            cohorts=dict(
                legacy="timeout120_parallel8", fresh="timeout600_serial1", pool_quality=False
            ),
        )
        for row in spec["jobs"]:
            row.update(
                disposition="dispatch", prior_row=dict(job_id=row["job_id"], status="cancelled")
            )
        spec["dispatch_job_ids"] = [r["job_id"] for r in spec["jobs"]]
        spec["campaign_policy"]["max_calls"] += 7
        spec["jobs"].insert(
            0,
            dict(
                spec["jobs"][0],
                job_id="retained",
                disposition="retained_quality_failure",
                prior_row=dict(
                    job_id="retained",
                    history_id="old",
                    status="failed",
                    calls=7,
                    artifact_sha256={},
                    failure=dict(type="quality"),
                ),
            ),
        )
        if continuation == "chain":
            spec.update(
                new_cohort="repaired",
                semantic_delta_manifest=dict(sha256="a" * 64),
                prior_spend_uncertainty=dict(extra_generation_attempts=None),
            )
            spec["cohorts"]["repaired"] = "timeout600_serial1_repaired_guard_v1"
            for record in spec["jobs"]:
                record["prior_cohort"] = "legacy"
            from test_comprehensive_driver_resource import authored_allowance
            spec["predecessor_binding"] = dict(root=str(tmp_path / "authored-old"))
            spec["prior_resource_allowance"] = authored_allowance(spec, tmp_path, calls=7, seconds=3.5)
        expected_calls += 7
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(json.dumps(spec).encode())
    approval = tmp_path / "authored-approval.json"
    approval.write_bytes(
        json.dumps(
            {
                "approved": True,
                "reviewer_kind": "AI-assisted",
                "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                **({"semantic_delta_manifest_sha256": "a" * 64,
                    "prior_resource_allowance_sha256": hashlib.sha256(json.dumps(spec["prior_resource_allowance"],sort_keys=True,separators=(",", ":")).encode()).hexdigest(),
                    "parent_reserve_approval_sha256": spec["prior_resource_allowance"]["parent_reserve_approval"]["sha256"]} if continuation == "chain" else {}),
            }
        ).encode()
    )

    def serve(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"data": [{"id": profile.model, "max_model_len": 131072}]}
            )
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 10})
        payload = json.loads(request.content)
        if middle_failure and "fail-middle" in json.dumps(payload["messages"]):
            if middle_failure == "shared":
                raise httpx.ConnectError(
                    "authored shared connection-health failure", request=request
                )
            if middle_failure == "timeout":
                raise httpx.ReadTimeout("authored generation deadline", request=request)
            return httpx.Response(
                200,
                json={
                    "model": profile.model,
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "authored truncated response"},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1},
                },
            )
        answer = "right" if payload["messages"][0]["role"] == "system" else "yes"
        return httpx.Response(
            200,
            json={
                "model": profile.model,
                "choices": [{"finish_reason": "stop", "message": {"content": answer}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            },
        )

    original_init = RackClient.__init__

    def init(client, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(serve)
        original_init(client, *args, **kwargs)

    monkeypatch.setattr(RackClient, "__init__", init)

    def synthetic_prompt(*args, **kwargs):
        return "authored judging instruction"

    monkeypatch.setattr(
        "gluomem_bench.bench.suite_campaign.load_longmemeval_prompt", lambda path: synthetic_prompt
    )
    monkeypatch.setattr(
        "gluomem_bench.bench.suite_evidence.load_longmemeval_prompt", lambda path: synthetic_prompt
    )
    monkeypatch.setattr(
        "gluomem_bench.bench.upstream_scoring.load_longmemeval_prompt",
        lambda path: synthetic_prompt,
    )
    if root_fault or (continuation and global_abort):
        from pathlib import Path

        from gluomem_bench.bench.parallel_campaign import CampaignReceiptError

        original_write = Path.write_bytes

        def fail_root(path, data):
            if path == root / "campaign-result.json":
                raise OSError("authored root finalization fault")
            return original_write(path, data)

        if root_fault == "summary":
            import gluomem_bench.bench.comprehensive_driver as d

            def broken_summary(*args):
                raise OSError("authored summary fault")

            monkeypatch.setattr(d, "_continuation_summary", broken_summary)
        elif root_fault:
            monkeypatch.setattr(Path, "write_bytes", fail_root)
        with pytest.raises(CampaignReceiptError) as raised:
            execute_longmemeval(spec_path, approval)
        assert raised.value.receipt["accounting"]["calls"] == expected_calls
        assert json.loads((root / "root-status.json").read_bytes())["complete"] is False
        assert (
            json.loads((root / "coordination/progress.json").read_bytes())["accounting"]["calls"]
            == expected_calls
        )
        if continuation:
            assert not (root / "root-status.pending.json").exists()
            assert not (root / "campaign-result.json").exists()
            assert (
                json.loads((root / "root-status.json").read_bytes())["publication"] == "incomplete"
            )
        return
    result = execute_longmemeval(spec_path, approval)
    assert result["accounting"]["calls"] == expected_calls
    if continuation:
        assert result["jobs"][0]["status"] == "failed"
        assert result["jobs"][0]["calls"] == 7
        assert result["complete"] is False
        assert result["accounting"]["prior_calls"] == 7
        assert result["accounting"]["new_calls"] == expected_calls - 7
        result_jobs = result["jobs"][1:]
    else:
        result_jobs = result["jobs"]
    assert result_jobs[0]["status"] == "complete"
    assert result_jobs[0]["result"]["quality_status"] == "complete_verified"
    if middle_failure:
        assert [r["status"] for r in result_jobs] == [
            "complete",
            "failed",
            "cancelled" if global_abort else "complete",
        ]
        assert result_jobs[1]["calls"] == 1
        assert result["complete"] is False
        assert "result" not in result_jobs[1]
        if middle_failure == "timeout":
            assert not (root / "jobs/j1/manifest.json").exists()
            assert (root / "jobs/j1/failure.json").exists()
    saved_root = json.loads((root / "root-status.json").read_bytes())
    assert saved_root == result
    assert (root / "campaign-result.json").is_file()
    from gluomem_bench.bench.comprehensive_driver import verify_campaign

    assert verify_campaign(root) == result
    if continuation:
        from pathlib import Path

        original_read = Path.read_bytes
        for mutation in (
            "missing",
            "duplicate",
            "prior",
            "cost",
            "elapsed",
            "quality",
            "scores",
            "missing-inventory",
            "missing-scorer-snapshot",
        ):
            forged = json.loads(json.dumps(result))
            if mutation == "missing":
                forged["jobs"].pop()
            if mutation == "duplicate":
                forged["jobs"][-1] = forged["jobs"][0]
            if mutation == "prior":
                forged["jobs"][0]["calls"] -= 1
            if mutation == "elapsed":
                forged["accounting"]["new_elapsed_seconds"] += 1
            if mutation == "quality":
                forged["jobs"][0]["status"] = "complete"
            if mutation == "scores":
                forged["summary"]["quality_by_cohort"]["legacy"] = forged["summary"][
                    "quality_by_cohort"
                ]["repaired" if continuation == "chain" else "fresh"]
            if mutation == "missing-inventory":
                forged["jobs"][1]["artifact_sha256"] = {}
            if mutation == "missing-scorer-snapshot":
                forged["frozen_inputs"] = [
                    r for r in forged["frozen_inputs"] if r["path"] != str(judge)
                ]
            overlay = {}
            if mutation == "cost":
                failed = next(
                    (r for r in forged["jobs"][1:] if r["status"] == "failed"), forged["jobs"][1]
                )
                failed["calls"] += 1
                forged["accounting"]["calls"] += 1
                forged["accounting"]["new_calls"] += 1
                coord = json.loads(original_read(root / "coordination/progress.json"))
                next(r for r in coord["jobs"] if r["job_id"] == failed["job_id"])["calls"] += 1
                coord["accounting"]["calls"] += 1
                coord["accounting"]["new_calls"] += 1
                overlay[root / "coordination/progress.json"] = json.dumps(coord).encode()
            overlay[root / "root-status.json"] = json.dumps(forged).encode()
            with monkeypatch.context() as patcher:
                patcher.setattr(
                    Path,
                    "read_bytes",
                    lambda p, overlay=overlay: overlay[p] if p in overlay else original_read(p),
                )
                with pytest.raises(ValueError):
                    verify_campaign(root)
    saved_source = next(item for item in result["frozen_inputs"] if item["path"] == str(source))
    assert (root / saved_source["snapshot"]).read_bytes() == source.read_bytes()
    leaf = root / "jobs" / ("j0" if middle_failure else "authored-job") / "predictions.json"
    leaf.write_bytes(leaf.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash"):
        verify_campaign(root)


def test_continuation_summary_fault_retains_actual_calls_and_incomplete_root(tmp_path, monkeypatch):
    test_authored_end_to_end_driver_accounts_and_offline_verifies(
        tmp_path, monkeypatch, "role-budgets-v6", "summary", False, True
    )


def test_continuation_policy_na_replays_exact_request_identity(tmp_path):
    from types import SimpleNamespace

    import gluomem_bench.bench.comprehensive_driver as d
    from gluomem_bench.bench.rack import RackProfile
    from gluomem_bench.bench.suite_campaign import reader_messages

    profile = RackProfile(base_url="http://example.invalid/v1", model="authored/model")
    messages = reader_messages("Current Date: 2026/01/02\nQuestion: Editor?", "")
    sample = SimpleNamespace(
        reader_question="Current Date: 2026/01/02\nQuestion: Editor?", conv_id="q"
    )
    native = (None, profile, {"q": sample}, {"q": SimpleNamespace(sessions=[])}, None)
    folder = tmp_path / "context-admission"
    folder.mkdir()
    record = dict(job_id="j", sample_id="q", system="full_context")
    row = dict(
        calls=0, result=dict(quality_status="not_applicable_context_overflow", prompt_tokens=5000)
    )
    trace = dict(
        method="POST",
        url="http://example.invalid/tokenize",
        status=200,
        request=dict(model=profile.model, messages=messages),
        response=dict(count=5000),
    )
    spec = dict(protocol=d.CONTINUATION_PROTOCOL, evidence_token_budget=4096)
    (folder / "j.jsonl").write_bytes(json.dumps(trace).encode())
    d._verify_continuation_na(tmp_path, spec, record, row, native)
    trace["request"]["model"] = "forged/model"
    (folder / "j.jsonl").write_bytes(json.dumps(trace).encode())
    with pytest.raises(ValueError, match="N/A"):
        d._verify_continuation_na(tmp_path, spec, record, row, native)


def test_continuation_full_verifier_receives_the_identical_outer_budget(tmp_path, monkeypatch):
    import gluomem_bench.bench.comprehensive_driver as d
    from gluomem_bench.bench.parallel_campaign import CampaignBudget

    budget = CampaignBudget(max_calls=20, max_seconds=10, prior_calls=7, prior_elapsed_seconds=3.5)
    seen = []

    def verify(root, *, candidate_path, budget):
        seen.append(budget)
        budget.check()

    monkeypatch.setattr(d, "verify_campaign", verify)
    result = dict(complete=False, execution_complete=True, jobs=[])
    assert d._publish_continuation(tmp_path, result, budget) == result
    assert seen == [budget]


def test_continuation_hashing_stops_at_the_original_outer_deadline(tmp_path, monkeypatch):
    from pathlib import Path

    import gluomem_bench.bench.comprehensive_driver as d
    from gluomem_bench.bench.parallel_campaign import CampaignBudget, CampaignStopped

    now = [0.0]
    budget = CampaignBudget(
        max_calls=20, max_seconds=10, prior_calls=7, prior_elapsed_seconds=3.5, clock=lambda: now[0]
    )
    leaf = tmp_path / "jobs/j"
    leaf.mkdir(parents=True)
    (leaf / "a.json").write_bytes(b"a")
    (leaf / "b.json").write_bytes(b"b")
    read = Path.read_bytes
    seen = []

    def slow_read(path):
        seen.append(path)
        now[0] = 20.0
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", slow_read)
    with pytest.raises(CampaignStopped):
        d._job_artifacts(tmp_path, dict(job_id="j", output=str(leaf)), budget=budget)
    assert len(seen) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "stage-write",
        "verify",
        "stage-read",
        "report-write",
        "report-read",
        "promote",
        "final-read",
        "deadline-verify",
        "deadline-promote",
        "interrupt-promote",
    ],
)
def test_continuation_publication_faults_revoke_authority_keep_spend(tmp_path, monkeypatch, fault):
    import os
    from pathlib import Path

    import gluomem_bench.bench.comprehensive_driver as d
    from gluomem_bench.bench.parallel_campaign import CampaignBudget, CampaignReceiptError

    now = [0.0]
    budget = CampaignBudget(
        max_calls=20, max_seconds=10, prior_calls=7, prior_elapsed_seconds=3.5, clock=lambda: now[0]
    )
    budget.begin_call()
    result = dict(
        jobs=[dict(job_id="authored", status="complete", calls=1)],
        complete=False,
        execution_complete=True,
        accounting=budget.snapshot(),
    )
    leaf = tmp_path / "leaf.txt"
    leaf.write_bytes(b"retained closed leaf")
    authority = tmp_path / "root-status.json"
    stage = tmp_path / "root-status.pending.json"
    report = tmp_path / "campaign-result.json"
    original_read, original_write, replace = Path.read_bytes, Path.write_bytes, os.replace
    promoted = [False]

    class InjectedAbort(BaseException):
        pass

    def verify(root, *, candidate_path, budget):
        assert candidate_path == stage
        assert json.loads(original_read(authority))["execution_complete"] is False
        if fault == "verify":
            raise ValueError("authored full verifier fault")
        if fault == "deadline-verify":
            now[0] = 20.0

    def write(path, data):
        if (fault == "stage-write" and path == stage) or (
            fault == "report-write" and path == report
        ):
            raise OSError("authored write fault")
        return original_write(path, data)

    def read(path):
        if (
            (fault == "stage-read" and path == stage)
            or (fault == "report-read" and path == report)
            or (fault == "final-read" and path == authority and promoted[0])
        ):
            raise OSError("authored readback fault")
        return original_read(path)

    def promote(source, target):
        if fault == "promote":
            raise OSError("authored promotion fault")
        replace(source, target)
        promoted[0] = True
        if fault == "interrupt-promote":
            raise InjectedAbort("authored interrupted promotion")
        if fault == "deadline-promote":
            now[0] = 20.0

    monkeypatch.setattr(d, "verify_campaign", verify)
    monkeypatch.setattr(Path, "write_bytes", write)
    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(os, "replace", promote)
    with pytest.raises(CampaignReceiptError) as raised:
        d._publish_continuation(tmp_path, result, budget)
    saved = json.loads(original_read(authority))
    assert saved["publication"] == "incomplete" and saved["execution_complete"] is False
    assert saved["accounting"]["calls"] == 8 == raised.value.receipt["accounting"]["calls"]
    assert not stage.exists() and not report.exists()
    assert original_read(leaf) == b"retained closed leaf"
