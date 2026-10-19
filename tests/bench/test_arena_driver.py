"""Full dataset driver contracts. Offline fixtures are never live quality evidence."""

import hashlib
import json
from pathlib import Path

import pytest

from gluomem_bench.bench import arena_campaign as native

ROOT = Path(__file__).resolve().parents[2] / 'tests/fixtures/native/operator-supplied'


def test_full_plan_has_every_formal_task_and_session_without_generation():
    from gluomem_bench.bench import arena_driver as full

    pins = json.loads((ROOT / "memoryarena-pins.json").read_bytes())
    plan = full.plan_dataset(ROOT / "memoryarena-data", pins)
    assert set(plan["inventory"]) == set(native.ARENA_DOMAINS)
    for domain in full.FORMAL_DOMAINS:
        from gluomem_bench.bench.trajectory_suite import load_arena

        tasks = load_arena(ROOT / "memoryarena-data", domain=domain, pins=pins)
        assert plan["tasks"][domain] == [
            {"id": t.id, "sessions": len(t.sessions), "input_sha256": full.task_digest(t)}
            for t in tasks
        ]
        assert plan["inventory"][domain]["sessions"] == sum(len(t.sessions) for t in tasks)
    assert plan["quality_calls"] == 0


def test_frozen_approval_binds_dataset_ids_roles_profile_and_code_before_any_output(tmp_path):
    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.rack import GenerationBudgets, RackProfile

    profile = RackProfile(
        base_url="https://example.invalid/v1",
        model="authored-offline",
        profile_version="role-budgets-v6",
        max_generation_calls=100,
        max_run_seconds=300,
        generation_budgets=GenerationBudgets(
            extraction=4096, consolidation=2048, reader=4096, judge=4096
        ),
    )
    pins = json.loads((ROOT / "memoryarena-pins.json").read_bytes())
    spec = full.build_spec(
        data_root=ROOT / "memoryarena-data",
        pins=pins,
        native_root=ROOT / "MemoryArena",
        profile=profile,
        output_root=tmp_path / "run",
        embedding_url="http://127.0.0.1:8766/v1",
        call_budget=100,
        seconds_budget=300,
        max_tool_iterations=3,
        campaign_seconds=21600,
    )
    sp, ap = tmp_path / "spec.json", tmp_path / "approval.json"
    sp.write_bytes(native._json(spec))
    ap.write_bytes(
        native._json(
            {
                "approved": True,
                "reviewer_kind": "parent",
                "spec_sha256": hashlib.sha256(sp.read_bytes()).hexdigest(),
            }
        )
    )
    assert full.verify_spec(sp, ap)["tasks"] == spec["tasks"]
    assert any(item["path"].endswith("schema.sql") for item in spec["files"])
    assert spec["roles"] == {
        r: profile.model for r in ("actor", "reasoning", "nativejudge", "extraction")
    }
    for mutate in ("ids", "roles", "profile", "files"):
        changed = json.loads(json.dumps(spec))
        if mutate == "ids":
            changed["tasks"][full.FORMAL_DOMAINS[0]].pop()
        if mutate == "roles":
            changed["roles"]["actor"] = "other"
        if mutate == "profile":
            changed["profile"]["generation_budgets"]["reader"] = 32
        if mutate == "files":
            changed["files"][0]["sha256"] = "0" * 64
        sp.write_bytes(native._json(changed))
        # Even re-approved malformed scopes must be refused before output creation.
        ap.write_bytes(
            native._json(
                {
                    "approved": True,
                    "reviewer_kind": "parent",
                    "spec_sha256": hashlib.sha256(sp.read_bytes()).hexdigest(),
                }
            )
        )
        with pytest.raises(ValueError):
            full.verify_spec(sp, ap)
        assert not (tmp_path / "run").exists()


def test_native_capability_requires_actual_exact_declared_sdk_tool_not_similar_schema(tmp_path):
    from test_arena_campaign import frozen_capability

    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.rack import RackProfile

    legacy = frozen_capability(native, tmp_path)
    profile = RackProfile(
        base_url="https://example.invalid/v1", model="authored-offline", max_tokens=128
    )
    cap = full.FrozenNativeToolCapability(
        legacy.receipt_path, legacy.receipt_sha256, legacy.wire_path, legacy.wire_sha256
    )
    with pytest.raises(ValueError, match="native.*tool"):
        cap.validate(profile)
    rows = [json.loads(line) for line in cap.wire_path.read_bytes().splitlines()]
    for row in rows[1:]:
        row["request"]["tools"] = native.NATIVE_REASONING_TOOLS
    cap.wire_path.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in rows))
    receipt = json.loads(cap.receipt_path.read_bytes())
    receipt["jobs"][0]["result"]["tools_sha256"] = hashlib.sha256(
        json.dumps(native.NATIVE_REASONING_TOOLS, sort_keys=True).encode()
    ).hexdigest()
    cap.receipt_path.write_bytes(native._json(receipt))
    cap = full.FrozenNativeToolCapability(
        cap.receipt_path,
        hashlib.sha256(cap.receipt_path.read_bytes()).hexdigest(),
        cap.wire_path,
        hashlib.sha256(cap.wire_path.read_bytes()).hexdigest(),
    )
    assert cap.validate(profile)["tool_names"] == ["reasoning"]


def test_full_native_phys_runs_all_sessions_stores_native_entries_and_rescores(tmp_path):
    import httpx

    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )
    from gluomem_bench.bench.rack import RackProfile
    from gluomem_bench.bench.trajectory_suite import ArenaSession, ArenaTask

    out = tmp_path / "phys"
    job = Job("phys", "phys", out, 20, 120)
    outer = CampaignBudget(max_calls=20, max_seconds=120)
    outer.reserve(job)
    budget = JobContext(job, outer, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))
    profile = RackProfile(
        base_url="https://example.invalid/v1", model="authored-offline", max_tokens=128
    )

    def wire(req):
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": profile.model}]})
        if req.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100})
        payload = json.loads(req.content)
        judge = "equivalence" in payload["messages"][0]["content"]
        return httpx.Response(
            200,
            json={
                "model": profile.model,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "yes" if judge else "Authored physics solution 5",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    client = native.ArenaTransport(
        profile,
        receipt=out / "http.jsonl",
        budget=budget,
        transport=httpx.MockTransport(wire),
        tool_capability="offline-authored",
    )

    def memory(context, output):
        return full.RawExperienceMemory("full_context", budget=context)

    task = ArenaTask(
        7,
        "formal_reasoning_phys",
        (ArenaSession("Authored physics A"), ArenaSession("Use the preceding solution.")),
        ("5", "5"),
    )
    try:
        result = full.run_native_task(
            task,
            source=native._sources(ROOT / "MemoryArena"),
            transport=client,
            memory=memory,
            out=out,
            max_tool_iterations=3,
        )
    finally:
        client.close()
    assert result["status"] == "complete"
    assert result["completed_sessions"] == 2
    rows = [json.loads(r) for r in (out / "steps.jsonl").read_bytes().splitlines()]
    assert rows[0]["memory_context"] == ""
    assert "## solution: Authored physics solution" in rows[1]["memory_context"]
    assert "## solution: Authored physics solution" in rows[0]["post_insert_readback"]
    assert rows[0]["experience"]["authorship"] == "agent"
    assert "ground_truth" not in rows[0]["experience"]
    assert full.verify_task(out)["avg_progress_score"] == 1
    saved = json.loads((out / "result.json").read_bytes())
    saved["calls"] += 1
    (out / "result.json").write_bytes(native._json(saved))
    with pytest.raises(ValueError, match="accounting"):
        full.verify_task(out)
    saved["calls"] -= 1
    (out / "result.json").write_bytes(native._json(saved))
    rows[0]["experience"]["content"] = "tampered"
    (out / "steps.jsonl").write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in rows))
    with pytest.raises(ValueError, match="hash"):
        full.verify_task(out)


@pytest.mark.parametrize("mode", ["no_memory", "bm25", "dense", "hybrid", "full_context"])
def test_online_controls_retrieve_only_past_native_experience_with_real_indexes(mode):
    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.parallel_campaign import CampaignBudget

    # Tiny deterministic encoder is an explicit test double, not a production backend.
    class Encoder:
        def embed(self, text):
            raise AssertionError("query must use shared query role")

        def embed_query(self, text):
            return [float("apple" in text), float("banana" in text)]

        def embed_batch(self, texts):
            return [self.embed_query(t) for t in texts]

    memory = full.RawExperienceMemory(
        mode, budget=CampaignBudget(max_calls=1, max_seconds=60), embedder=Encoder(), top_k=1
    )
    assert memory.query("apple") == ""
    first = {"content": "apple solution", "source_ref": "session/0", "authorship": "agent"}
    assert memory.insert(first)["readback_verified"]
    assert memory.insert(
        {"content": "banana solution", "source_ref": "session/1", "authorship": "agent"}
    )["readback_verified"]
    context = memory.query("apple")
    if mode == "no_memory":
        assert context == ""
    else:
        assert "apple solution" in context
    if mode not in {"no_memory", "full_context"}:
        assert "banana solution" not in context
    memory.close()


def test_gluomem_online_document_uses_production_source_event_commit_and_provenance(
    tmp_path, monkeypatch
):
    from gluomem.embedding.providers import MockEmbedder
    from gluomem.llm.modules import DspyExtractors
    from gluomem.llm.outputs import ExtractedClaimSpec

    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.parallel_campaign import CampaignBudget
    from gluomem_bench.bench.rack import RackProfile, controlled_config

    # Boundary fixture only: real core extraction envelope/admission/DB/provenance/search run.
    extractors = DspyExtractors(
        controlled_config(
            RackProfile(base_url="https://example.invalid/v1", model="authored-offline"),
            data=tmp_path / "authored.txt",
            sha256="0" * 64,
        )
    )
    monkeypatch.setattr(
        extractors,
        "extract_event_claims",
        lambda *a, **k: [
            ExtractedClaimSpec(text="The agent computed the authored sum as five.", kind="fact")
        ],
    )
    cfg = (
        extractors.cfg
        if hasattr(extractors, "cfg")
        else controlled_config(
            RackProfile(base_url="https://example.invalid/v1", model="authored-offline"),
            data=tmp_path / "authored.txt",
            sha256="0" * 64,
        )
    )
    cfg = cfg.with_overrides({"storage.vector_dim": 384})
    budget = CampaignBudget(max_calls=8, max_seconds=120)
    memory = full.GluoExperienceMemory(
        cfg=cfg,
        extractors=extractors,
        embedder=MockEmbedder(dim=384),
        budget=budget,
        db_path=tmp_path / "memory.mind",
        evidence_path=tmp_path / "memory-events.jsonl",
    )
    experience = {
        "content": "## Task: Compute authored 2+3\n## solution: 5\n",
        "authorship": "agent",
        "source_ref": "memoryarena/authored/0/session/0",
        "session_index": 0,
        "feedback_in_memory": False,
        "native_source_commit": native.PIN,
    }
    assert memory.query("authored sum") == ""
    receipt = memory.insert(experience)
    assert receipt["readback_verified"] is True
    assert receipt["source_type"] == "document"
    assert receipt["provenance_count"] >= 1
    assert "authored sum" in memory.query("authored sum")
    memory.close()
    events = [json.loads(r) for r in (tmp_path / "memory-events.jsonl").read_bytes().splitlines()]
    assert events[0]["event"]["source_type"] == "document"
    assert events[0]["event"]["messages"] == []
    assert events[0]["event"]["kind"] is None


def test_full_dataset_permit_admits_large_explicit_scope_not_legacy_and_refuses_task_mutation(
    tmp_path,
):
    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.rack import GenerationBudgets, RackProfile
    from gluomem_bench.bench.trajectory_suite import ArenaTask, load_arena

    profile = RackProfile(
        base_url="https://example.invalid/v1",
        model="authored-offline",
        profile_version="role-budgets-v6",
        max_generation_calls=1000,
        max_run_seconds=1800,
        generation_budgets=GenerationBudgets(
            extraction=4096, consolidation=2048, reader=4096, judge=4096
        ),
    )
    pins = json.loads((ROOT / "memoryarena-pins.json").read_bytes())
    spec = full.build_spec(
        data_root=ROOT / "memoryarena-data",
        pins=pins,
        native_root=ROOT / "MemoryArena",
        profile=profile,
        output_root=tmp_path / "run",
        embedding_url="http://127.0.0.1:8766/v1",
        call_budget=1000,
        seconds_budget=1800,
        max_tool_iterations=3,
        campaign_seconds=86400,
    )
    sp, ap = tmp_path / "spec.json", tmp_path / "approval.json"
    sp.write_bytes(native._json(spec))
    ap.write_bytes(
        native._json(
            {
                "approved": True,
                "reviewer_kind": "parent",
                "spec_sha256": hashlib.sha256(sp.read_bytes()).hexdigest(),
            }
        )
    )
    permit = full.FrozenDatasetTask(sp, ap)
    task = load_arena(ROOT / "memoryarena-data", domain=full.FORMAL_DOMAINS[0], pins=pins)[0]
    permit.validate(task, profile, 3)
    with pytest.raises(ValueError, match="task"):
        permit.validate(
            ArenaTask(task.id, task.domain, task.sessions, ("changed",) * len(task.gold)),
            profile,
            3,
        )
    with pytest.raises(ValueError, match="profile"):
        permit.validate(task, profile.model_copy(update={"model": "other"}), 3)
    with pytest.raises(ValueError, match="iteration"):
        permit.validate(task, profile, 1)


def test_full_transport_does_not_reuse_legacy_eight_call_gate_or_accept_undeclared_code(tmp_path):
    import httpx

    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )
    from gluomem_bench.bench.rack import GenerationBudgets, RackProfile

    profile = RackProfile(
        base_url="https://example.invalid/v1",
        model="authored-offline",
        profile_version="role-budgets-v6",
        max_generation_calls=100,
        max_run_seconds=300,
        generation_budgets=GenerationBudgets(
            extraction=4096, consolidation=2048, reader=4096, judge=4096
        ),
    )
    # Authored constructor has its own <=8/300s envelope; cannot authorize dataset IDs.
    job = Job("qualification", "qualification", tmp_path / "run", 8, 300)
    outer = CampaignBudget(max_calls=8, max_seconds=300)
    outer.reserve(job)
    budget = JobContext(job, outer, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))

    def wire(req):
        if req.url.path == "/v1/models":
            return httpx.Response(
                200, json={"data": [{"id": profile.model, "max_model_len": 131072}]}
            )
        if req.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100, "max_model_len": 131072})
        return httpx.Response(
            200,
            json={
                "model": profile.model,
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "bad",
                                    "type": "function",
                                    "function": {
                                        "name": "coding",
                                        "arguments": '{"input":"print(5)"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    client = full.FullArenaTransport.authored(
        profile, receipt=tmp_path / "http.jsonl", budget=budget, transport=httpx.MockTransport(wire)
    )
    try:
        with pytest.raises(native.NativeAbort, match="undeclared"):
            client.create(
                "actor",
                model=profile.model,
                messages=[{"role": "user", "content": "Authored sum"}],
                tools=native.NATIVE_REASONING_TOOLS,
                tool_choice="auto",
                temperature=0,
            )
    finally:
        client.close()
    assert budget.calls == 1
    with pytest.raises(ValueError, match="authored"):
        full.FullArenaTransport.authored(
            profile,
            receipt=tmp_path / "other.jsonl",
            budget=JobContext(
                Job("too-large", "too-large", tmp_path / "other", 9, 300),
                outer,
                RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072),
            ),
        )


def test_paired_full_matrix_continues_after_failed_task_and_preserves_every_denominator(tmp_path):
    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.trajectory_suite import ArenaSession, ArenaTask

    tasks = [
        ArenaTask(i, "formal_reasoning_math", (ArenaSession("authored"),), ("5",)) for i in (1, 2)
    ]
    seen = []

    def run_one(task, system, context, out):
        seen.append((task.id, system))
        if task.id == 1 and system == "bm25":
            raise RuntimeError("authored infrastructure failure")
        return {"status": "complete", "expected_sessions": 1, "completed_sessions": 1, "calls": 0}

    result = full.execute_matrix(
        tasks,
        systems=["no_memory", "bm25"],
        out=tmp_path / "matrix",
        call_budget=8,
        seconds_budget=120,
        campaign_seconds=300,
        run_one=run_one,
    )
    assert seen == [(1, "no_memory"), (1, "bm25"), (2, "no_memory"), (2, "bm25")]
    assert len(result["tasks"]) == 4
    assert [r["status"] for r in result["tasks"]] == ["complete", "failed", "complete", "complete"]
    assert result["expected_sessions"] == 4
    assert result["completed_sessions"] == 3
    assert result["quality_complete"] is False
    assert json.loads((tmp_path / "matrix" / "campaign-result.json").read_bytes()) == result


def test_matrix_completion_owner_defers_only_when_formal_aggregation_owns_authority(tmp_path):
    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.trajectory_suite import ArenaSession, ArenaTask

    task = ArenaTask(1, full.FORMAL_DOMAINS[0], (ArenaSession("authored"),), ("5",))
    for owner in ("matrix", "formal"):
        out = tmp_path / owner
        result = full.execute_matrix(
            [task],
            systems=["no_memory"],
            out=out,
            call_budget=1,
            seconds_budget=1,
            campaign_seconds=10,
            completion_owner=owner,
            run_one=lambda *a: {"status": "complete", "completed_sessions": 1},
        )
        assert result["matrix_complete"] is True
        assert result["completion_owner"] == owner
        assert result["authoritative"] is (owner == "matrix")
        assert result["quality_complete"] is (owner == "matrix")
        assert json.loads((out / "campaign-result.json").read_bytes()) == result


@pytest.mark.parametrize("fault", ["aggregate", "final-write", "final-replace", "final-readback"])
def test_formal_finalization_faults_never_leave_complete_authority_and_recover(
    tmp_path, monkeypatch, fault
):
    import httpx

    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.rack import GenerationBudgets, RackProfile
    from gluomem_bench.bench.trajectory_suite import ArenaSession, ArenaTask

    profile = RackProfile(
        base_url="https://example.invalid/v1",
        model="authored-offline",
        profile_version="role-budgets-v6",
        max_generation_calls=8,
        max_run_seconds=120,
        generation_budgets=GenerationBudgets(
            extraction=4096, consolidation=2048, reader=4096, judge=4096
        ),
    )
    tasks = [
        ArenaTask(i, domain, (ArenaSession("authored"),), ("5",))
        for i, domain in enumerate(full.FORMAL_DOMAINS)
    ]
    archived_source = tmp_path / "schema.sql"
    archived_source.write_bytes(b"-- offline source inventory fixture outside egzakta-org\n")
    spec = {
        "profile": profile.model_dump(mode="json"),
        "output_root": str(tmp_path / "failed"),
        "data_root": str(tmp_path),
        "data_pins": {},
        "native_root": str(ROOT / "MemoryArena"),
        "systems": ["no_memory"],
        "embedding_url": "https://example.invalid/v1",
        "bounds": {
            "calls_per_task_method": 8,
            "seconds_per_task_method": 120,
            "campaign_seconds": 600,
            "max_tool_iterations": 3,
        },
        "inventory": {domain: {"tasks": 1, "sessions": 1} for domain in full.FORMAL_DOMAINS},
        "tasks": {
            t.domain: [{"id": t.id, "sessions": 1, "input_sha256": full.task_digest(t)}]
            for t in tasks
        },
        "files": [
            {
                "path": str(archived_source),
                "sha256": hashlib.sha256(archived_source.read_bytes()).hexdigest(),
            }
        ],
    }
    sp, ap = tmp_path / "spec.json", tmp_path / "approval.json"
    sp.write_bytes(native._json(spec))
    ap.write_bytes(b'{"fixture": "offline admission seam only"}')
    monkeypatch.setattr(full, "verify_spec", lambda *a, **k: json.loads(sp.read_bytes()))
    monkeypatch.setattr(
        full,
        "verify_memory_qualification",
        lambda *a: {
            "capability": {
                "receipt_path": str(tmp_path / "receipt"),
                "receipt_sha256": "",
                "wire_path": str(tmp_path / "wire"),
                "wire_sha256": "",
            }
        },
    )
    monkeypatch.setattr(full.FrozenNativeToolCapability, "validate", lambda *a: {})
    monkeypatch.setattr(
        full, "load_arena", lambda *a, domain, **k: [t for t in tasks if t.domain == domain]
    )
    original_init = full.FullArenaTransport.__init__

    def wire(req):
        if req.url.path == "/v1/models":
            return httpx.Response(
                200, json={"data": [{"id": profile.model, "max_model_len": 131072}]}
            )
        if req.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100, "max_model_len": 131072})
        payload = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "model": profile.model,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "yes"
                            if "equivalence" in payload["messages"][0]["content"]
                            else "5"
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    def injected_init(self, *args, **kwargs):
        original_init(self, *args, transport=httpx.MockTransport(wire), **kwargs)

    monkeypatch.setattr(full.FullArenaTransport, "__init__", injected_init)
    score, write, replace, read = (
        full.official_metrics,
        Path.write_bytes,
        Path.replace,
        Path.read_bytes,
    )
    root = Path(spec["output_root"])
    observed = []

    def assert_intermediate():
        current = json.loads(read(root / "campaign-result.json"))
        observed.append(current)
        assert current["quality_complete"] is False
        assert current["authoritative"] is False
        assert current["calls"] == 4

    def injected_score(source, logs, out):
        if Path(out).is_relative_to(root / "scores"):
            assert_intermediate()
            if fault == "aggregate":
                raise RuntimeError("injected aggregate scorer failure")
        return score(source, logs, out)

    def injected_write(path, data):
        if (
            fault == "final-write"
            and path.parent == root
            and path.name in {"campaign-result.json", "campaign-result.pending.json"}
            and json.loads(data).get("quality_complete") is True
        ):
            assert_intermediate()
            raise OSError("injected final root write failure")
        return write(path, data)

    def injected_replace(path, target):
        if fault == "final-replace" and Path(target) == root / "campaign-result.json":
            assert_intermediate()
            raise OSError("injected final root replace failure")
        return replace(path, target)

    def injected_read(path):
        data = read(path)
        if (
            fault == "final-readback"
            and path == root / "campaign-result.json"
            and json.loads(data).get("quality_complete") is True
        ):
            raise OSError("injected final root readback failure")
        return data

    with monkeypatch.context() as faults:
        faults.setattr(full, "official_metrics", injected_score)
        faults.setattr(Path, "write_bytes", injected_write)
        faults.setattr(Path, "replace", injected_replace)
        faults.setattr(Path, "read_bytes", injected_read)
        with pytest.raises((RuntimeError, OSError), match="injected"):
            full.execute_formal(sp, ap)
    assert observed
    if (root / "campaign-result.json").exists():
        assert json.loads(read(root / "campaign-result.json"))["quality_complete"] is False
    coordination = json.loads(read(root / "campaign-failure.json"))
    assert coordination["quality_complete"] is False
    assert coordination["calls"] == 4
    assert all(r["status"] == "complete" for r in coordination["tasks"])
    assert not (root / "campaign-result.pending.json").exists()
    # Actual successful leaf evidence remains intact; it is never promoted to root.
    for row in coordination["tasks"]:
        assert full.verify_task(row["output"])["avg_progress_score"] == 1
    with pytest.raises(FileExistsError):
        full.execute_formal(sp, ap)
    # Recovery uses a fresh explicitly selected root, never repeats into old spend.
    spec["output_root"] = str(tmp_path / "recovery")
    sp.write_bytes(native._json(spec))
    recovered = full.execute_formal(sp, ap)
    assert recovered["authoritative"] is True
    assert recovered["quality_complete"] is True
    assert recovered["calls"] == 4
    assert recovered["spec_sha256"] == hashlib.sha256(sp.read_bytes()).hexdigest()
    final_root = Path(spec["output_root"])
    assert json.loads(read(final_root / "campaign-result.json")) == recovered
    assert all(
        recovered["metrics"][domain]["no_memory"]["status"] == "complete"
        for domain in full.FORMAL_DOMAINS
    )
    assert recovered["artifact_sha256"]["execution-spec.json"] == recovered["spec_sha256"]


def test_authored_qualification_requires_full_role_profile_before_output_or_calls(tmp_path):
    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.rack import RackProfile

    with pytest.raises(ValueError, match="role-budgets-v6"):
        full.qualify_authored_memory(
            native_root=ROOT / "MemoryArena",
            out=tmp_path / "no-write",
            profile=RackProfile(base_url="https://example.invalid/v1", model="authored-offline"),
            embedding_url="http://127.0.0.1:8766/v1",
        )
    assert not (tmp_path / "no-write").exists()


def test_real_saved_authored_memory_qualification_replays_offline_before_quality():
    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.rack import RackProfile

    path = ROOT / "arena-full-authored-memory" / "qualification.json"
    if not path.exists():
        pytest.skip("real authored qualification is external; never synthesize it")
    record = json.loads(path.read_bytes())
    proof = full.verify_memory_qualification(
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        RackProfile.model_validate(record["profile"]),
    )
    assert proof["complete"] is True
    assert proof["production_readback"]["provenance_count"] >= 1
    assert 1 <= proof["calls"] <= 8


def test_full_cli_plan_produces_every_task_and_run_refuses_missing_production_proof(tmp_path):
    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.rack import GenerationBudgets, RackProfile

    profile = RackProfile(
        base_url="https://example.invalid/v1",
        model="authored-offline",
        profile_version="role-budgets-v6",
        max_generation_calls=1000,
        max_run_seconds=1800,
        generation_budgets=GenerationBudgets(
            extraction=4096, consolidation=2048, reader=4096, judge=4096
        ),
    )
    pp = tmp_path / "profile.json"
    pp.write_bytes(native._json(profile.model_dump(mode="json")))
    sp, ap = tmp_path / "spec.json", tmp_path / "approval.json"
    assert (
        full.main(
            [
                "plan",
                "--data-root",
                str(ROOT / "memoryarena-data"),
                "--pins",
                str(ROOT / "memoryarena-pins.json"),
                "--native-root",
                str(ROOT / "MemoryArena"),
                "--profile",
                str(pp),
                "--output-root",
                str(tmp_path / "full"),
                "--embedding-url",
                "http://127.0.0.1:8766/v1",
                "--spec-out",
                str(sp),
                "--calls-per-task",
                "1000",
                "--seconds-per-task",
                "1800",
                "--campaign-seconds",
                "86400",
                "--tool-iterations",
                "3",
            ]
        )
        == 0
    )
    spec = json.loads(sp.read_bytes())
    assert spec["systems"] == list(full.SYSTEMS)
    ap.write_bytes(
        native._json(
            {
                "approved": True,
                "reviewer_kind": "parent",
                "spec_sha256": hashlib.sha256(sp.read_bytes()).hexdigest(),
            }
        )
    )
    with pytest.raises(ValueError, match="production memory qualification"):
        full.execute_formal(sp, ap)
    assert not (tmp_path / "full").exists()


def test_native_task_failure_distinguishes_attempted_failed_from_unrun_subsessions(tmp_path):
    import httpx

    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )
    from gluomem_bench.bench.rack import RackProfile
    from gluomem_bench.bench.trajectory_suite import ArenaSession, ArenaTask

    out = tmp_path / "failed-task"
    profile = RackProfile(
        base_url="https://example.invalid/v1", model="authored-offline", max_tokens=128
    )
    job = Job("failed", "failed", out, 8, 120)
    outer = CampaignBudget(max_calls=8, max_seconds=120)
    outer.reserve(job)
    ctx = JobContext(job, outer, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))

    def wire(req):
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": profile.model}]})
        if req.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100})
        return httpx.Response(503, json={"error": "authored infrastructure failure"})

    client = native.ArenaTransport(
        profile,
        receipt=out / "http.jsonl",
        budget=ctx,
        transport=httpx.MockTransport(wire),
        tool_capability="offline-authored",
    )
    try:
        with pytest.raises(RuntimeError, match="full native task failed"):
            full.run_native_task(
                ArenaTask(
                    1,
                    "formal_reasoning_math",
                    (ArenaSession("Authored 1"), ArenaSession("Authored 2")),
                    ("5", "5"),
                ),
                source=native._sources(ROOT / "MemoryArena"),
                transport=client,
                memory=full.RawExperienceMemory("no_memory", budget=ctx),
                out=out,
                max_tool_iterations=3,
            )
    finally:
        client.close()
    failure = json.loads((out / "failure.json").read_bytes())
    assert [s["status"] for s in failure["sessions"]] == ["failed", "unrun"]
    assert failure["calls"] == 1
    assert not (out / "result.json").exists()


def test_optional_native_document_factories_require_frozen_real_backend_evidence(tmp_path):
    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.rack import GenerationBudgets, RackProfile

    profile = RackProfile(
        base_url="https://example.invalid/v1",
        model="authored-offline",
        profile_version="role-budgets-v6",
        max_generation_calls=100,
        max_run_seconds=300,
        generation_budgets=GenerationBudgets(
            extraction=4096, consolidation=2048, reader=4096, judge=4096
        ),
    )
    pins = json.loads((ROOT / "memoryarena-pins.json").read_bytes())
    spec = full.build_spec(
        data_root=ROOT / "memoryarena-data",
        pins=pins,
        native_root=ROOT / "MemoryArena",
        profile=profile,
        output_root=tmp_path / "run",
        embedding_url="http://127.0.0.1:8766/v1",
        call_budget=100,
        seconds_budget=300,
        max_tool_iterations=3,
        campaign_seconds=21600,
    )
    sp, ap = tmp_path / "spec.json", tmp_path / "approval.json"
    spec["systems"].append("mem0")
    spec["native_document_factories"] = {"mem0": {"kind": "fake"}}
    sp.write_bytes(native._json(spec))
    ap.write_bytes(
        native._json(
            {
                "approved": True,
                "reviewer_kind": "parent",
                "spec_sha256": hashlib.sha256(sp.read_bytes()).hexdigest(),
            }
        )
    )
    with pytest.raises(ValueError, match="native document factory"):
        full.verify_spec(sp, ap)
    assert not (tmp_path / "run").exists()


def test_shared_encoder_requests_obey_same_whole_task_deadline(monkeypatch):
    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.embedding_runtime import RemoteEmbedder
    from gluomem_bench.bench.parallel_campaign import CampaignBudget, CampaignStopped

    clock = [0.0]
    budget = CampaignBudget(max_calls=8, max_seconds=300, clock=lambda: clock[0])
    observed = []

    def encode(self, texts, role):
        observed.append((self.timeout, role))
        return [[1.0]]

    monkeypatch.setattr(RemoteEmbedder, "_encode", encode)
    encoder = full.BudgetedArenaEmbedder("http://127.0.0.1:8766/v1", budget=budget)
    clock[0] = 299.0
    encoder.embed_query("authored")
    assert observed == [(1.0, "query")]
    clock[0] = 301.0
    with pytest.raises(CampaignStopped):
        encoder.embed_query("authored")
    assert len(observed) == 1


def test_production_document_router_has_local_placeholder_without_ambient_credentials(
    tmp_path, monkeypatch
):
    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.parallel_campaign import CampaignBudget
    from gluomem_bench.bench.rack import RackClient, RackProfile, controlled_config

    monkeypatch.delenv("GLUOMEM_RACK_API_KEY", raising=False)
    profile = RackProfile(base_url="https://example.invalid/v1", model="authored-offline")
    budget = CampaignBudget(max_calls=8, max_seconds=300)
    cfg = controlled_config(profile, data=tmp_path / "authored.txt", sha256="0" * 64)
    client = RackClient(profile, receipt=tmp_path / "http.jsonl", budget=budget)
    try:
        router = full.ArenaProductionRouter(
            cfg, client=client, trace=tmp_path / "trace.jsonl", budget=budget
        )
        with router.bind("event_claims"):
            assert router.lm_for("event_claims").kwargs["api_key"] == "local-endpoint-placeholder"
        assert "GLUOMEM_RACK_API_KEY" not in __import__("os").environ
        assert budget.calls == 0
    finally:
        client.close()


def test_full_transport_requires_production_proof_even_when_called_outside_driver(tmp_path):
    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )
    from gluomem_bench.bench.rack import RackProfile
    from gluomem_bench.bench.trajectory_suite import load_arena

    proof_path = ROOT / "arena-full-authored-memory/qualification.json"
    if not proof_path.exists():
        pytest.skip("external real qualification required; no substitute")
    proof = json.loads(proof_path.read_bytes())
    profile = RackProfile.model_validate(
        {**proof["profile"], "max_generation_calls": 100, "max_run_seconds": 300}
    )
    pins = json.loads((ROOT / "memoryarena-pins.json").read_bytes())
    spec = full.build_spec(
        data_root=ROOT / "memoryarena-data",
        pins=pins,
        native_root=ROOT / "MemoryArena",
        profile=profile,
        output_root=tmp_path / "run",
        embedding_url="http://127.0.0.1:8766/v1",
        call_budget=100,
        seconds_budget=300,
        max_tool_iterations=3,
        campaign_seconds=21600,
    )
    sp, ap = tmp_path / "spec.json", tmp_path / "approval.json"
    sp.write_bytes(native._json(spec))
    ap.write_bytes(
        native._json(
            {
                "approved": True,
                "reviewer_kind": "parent",
                "spec_sha256": hashlib.sha256(sp.read_bytes()).hexdigest(),
            }
        )
    )
    task = load_arena(ROOT / "memoryarena-data", domain=full.FORMAL_DOMAINS[0], pins=pins)[0]
    job = Job("full", "full", tmp_path / "run", 100, 300)
    outer = CampaignBudget(max_calls=100, max_seconds=300)
    outer.reserve(job)
    ctx = JobContext(job, outer, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))
    cap = proof["capability"]
    with pytest.raises(ValueError, match="production memory qualification"):
        client = full.FullArenaTransport(
            profile,
            receipt=tmp_path / "http.jsonl",
            budget=ctx,
            permit=full.FrozenDatasetTask(sp, ap),
            task=task,
            capability=full.FrozenNativeToolCapability(
                Path(cap["receipt_path"]),
                cap["receipt_sha256"],
                Path(cap["wire_path"]),
                cap["wire_sha256"],
            ),
        )
        client.close()
    assert ctx.calls == 0


@pytest.mark.parametrize(
    "fault", ["transport", "model-context", "tokenizer-context", "response-model", "usage"]
)
def test_native_full_transport_propagates_genuine_shared_transport_fault(tmp_path, fault):
    import httpx

    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )
    from gluomem_bench.bench.rack import GenerationBudgets, RackProfile

    profile = RackProfile(
        base_url="https://example.invalid/v1",
        model="authored-offline",
        profile_version="role-budgets-v6",
        generation_budgets=GenerationBudgets(
            extraction=4096, consolidation=2048, reader=4096, judge=4096
        ),
    )
    job = Job("fault", "fault", tmp_path / "task", 8, 300)
    outer = CampaignBudget(max_calls=8, max_seconds=300)
    outer.reserve(job)
    ctx = JobContext(job, outer, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))

    def wire(req):
        if req.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": profile.model,
                            "max_model_len": 131073 if fault == "model-context" else 131072,
                        }
                    ]
                },
            )
        if req.url.path == "/tokenize":
            return httpx.Response(
                200,
                json={
                    "count": 100,
                    "max_model_len": 131073 if fault == "tokenizer-context" else 131072,
                },
            )
        if fault == "transport":
            return httpx.Response(503, json={"error": "authored shared service unavailable"})
        return httpx.Response(
            200,
            json={
                "model": "other" if fault == "response-model" else profile.model,
                "choices": [{"finish_reason": "stop", "message": {"content": "Authored"}}],
                "usage": {
                    "prompt_tokens": 999 if fault == "usage" else 100,
                    "completion_tokens": 10,
                },
            },
        )

    client = full.FullArenaTransport.authored(
        profile, receipt=tmp_path / "http.jsonl", budget=ctx, transport=httpx.MockTransport(wire)
    )
    try:
        with pytest.raises(native.NativeAbort):
            client.create(
                "actor", model=profile.model, messages=[{"role": "user", "content": "Authored"}]
            )
        assert ctx.calls == (0 if fault in {"model-context", "tokenizer-context"} else 1)
        assert outer.failed is True
    finally:
        client.close()


@pytest.mark.parametrize("shared", [False, True])
def test_full_matrix_middle_generation_timeout_is_local_but_explicit_shared_fault_cancels(
    tmp_path, monkeypatch, shared
):
    import httpx

    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.rack import GenerationBudgets, RackProfile, SharedRackError
    from gluomem_bench.bench.trajectory_suite import ArenaSession, ArenaTask

    profile = RackProfile(
        base_url="https://example.invalid/v1",
        model="authored-offline",
        profile_version="role-budgets-v6",
        generation_budgets=GenerationBudgets(
            extraction=4096, consolidation=2048, reader=4096, judge=4096
        ),
    )
    # Admission fixtures only; exercise the actual full transport/native loop/matrix.
    monkeypatch.setattr(
        full.FrozenDatasetTask,
        "validate",
        lambda *a: {"bounds": {"calls_per_task_method": 8, "seconds_per_task_method": 120}},
    )
    monkeypatch.setattr(full, "verify_memory_qualification", lambda *a: {})
    monkeypatch.setattr(full.FrozenNativeToolCapability, "validate", lambda *a: {})
    permit = full.FrozenDatasetTask(tmp_path / "spec", tmp_path / "approval")
    capability = full.FrozenNativeToolCapability(tmp_path / "receipt", "", tmp_path / "wire", "")
    source = native._sources(ROOT / "MemoryArena")
    tasks = [
        ArenaTask(i, full.FORMAL_DOMAINS[0], (ArenaSession("authored"),), ("5",)) for i in (1, 2, 3)
    ]

    def run_one(task, system, context, target):
        def wire(req):
            if req.url.path == "/v1/models":
                return httpx.Response(
                    200, json={"data": [{"id": profile.model, "max_model_len": 131072}]}
                )
            if req.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 100, "max_model_len": 131072})
            if task.id == 2:
                if shared:
                    raise SharedRackError("explicit shared health evidence")
                raise httpx.ReadTimeout("single generation expired", request=req)
            payload = json.loads(req.content)
            return httpx.Response(
                200,
                json={
                    "model": profile.model,
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "yes"
                                if "equivalence" in payload["messages"][0]["content"]
                                else "5"
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                },
            )

        client = full.FullArenaTransport(
            profile,
            receipt=target / "http.jsonl",
            budget=context,
            permit=permit,
            task=task,
            capability=capability,
            transport=httpx.MockTransport(wire),
        )
        try:
            return full.run_native_task(
                task,
                source=source,
                transport=client,
                memory=full.RawExperienceMemory("no_memory", budget=context),
                out=target,
                max_tool_iterations=3,
            )
        finally:
            client.close()

    out = tmp_path / "matrix"
    result = full.execute_matrix(
        tasks,
        systems=["no_memory"],
        out=out,
        call_budget=8,
        seconds_budget=120,
        campaign_seconds=600,
        run_one=run_one,
    )
    assert [r["status"] for r in result["tasks"]] == [
        "complete",
        "failed",
        "unrun" if shared else "complete",
    ]
    assert result["calls"] == (3 if shared else 5)
    failed = out / tasks[1].domain / "2/no_memory"
    assert json.loads((failed / "failure.json").read_bytes())["calls"] == 1
    assert not (failed / "result.json").exists()
    if not shared:
        assert full.verify_task(out / tasks[2].domain / "3/no_memory")["avg_progress_score"] == 1


def test_native_scoring_cannot_publish_after_whole_task_deadline(tmp_path, monkeypatch):
    import httpx

    from gluomem_bench.bench import arena_driver as full
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )
    from gluomem_bench.bench.rack import RackProfile
    from gluomem_bench.bench.trajectory_suite import ArenaSession, ArenaTask

    out = tmp_path / "late"
    profile = RackProfile(
        base_url="https://example.invalid/v1", model="authored-offline", max_tokens=128
    )
    job = Job("late", "late", out, 8, 120)
    outer = CampaignBudget(max_calls=8, max_seconds=120)
    outer.reserve(job)
    ctx = JobContext(job, outer, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))

    def wire(req):
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": profile.model}]})
        if req.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100})
        payload = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "model": profile.model,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "yes"
                            if "equivalence" in payload["messages"][0]["content"]
                            else "5"
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    score = full.official_metrics

    def late(*args, **kwargs):
        result = score(*args, **kwargs)
        ctx.deadline = 0
        return result

    monkeypatch.setattr(full, "official_metrics", late)
    client = native.ArenaTransport(
        profile,
        receipt=out / "http.jsonl",
        budget=ctx,
        transport=httpx.MockTransport(wire),
        tool_capability="offline-authored",
    )
    try:
        with pytest.raises(RuntimeError, match="full native task failed"):
            full.run_native_task(
                ArenaTask(1, "formal_reasoning_math", (ArenaSession("authored"),), ("5",)),
                source=native._sources(ROOT / "MemoryArena"),
                transport=client,
                memory=full.RawExperienceMemory("no_memory", budget=ctx),
                out=out,
                max_tool_iterations=1,
            )
    finally:
        client.close()
    assert json.loads((out / "failure.json").read_bytes())["calls"] == 2
    assert not (out / "result.json").exists()
    assert not (out / "result.pending.json").exists()
