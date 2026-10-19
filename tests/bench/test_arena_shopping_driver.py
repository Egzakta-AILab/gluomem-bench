"""Shopping admission and native runtime tests; no model access."""

import json
from pathlib import Path

import pytest

ROOT = Path(
    'tests/fixtures/native/operator-supplied'
)


def driver():
    from gluomem_bench.bench import arena_shopping_driver

    return arena_shopping_driver


def test_full_released_inventory_preserves_every_ordered_occurrence():
    d = driver()
    plan = d.plan_dataset(
        ROOT / "memoryarena-data", json.loads((ROOT / "memoryarena-pins.json").read_bytes())
    )
    assert plan["tasks_count"] == 150
    assert plan["sessions_count"] == 900
    assert len({r["id"] for r in plan["tasks"]}) == 150
    assert [r["session_index"] for r in plan["occurrences"][:6]] == list(range(6))
    assert len({(r["task_id"], r["session_index"]) for r in plan["occurrences"]}) == 900
    assert plan["quality_calls"] == 0
    assert plan["systems"] == ["no_memory", "bm25", "dense", "hybrid", "full_context", "gluomem"]
    assert plan["optional_adapters"] == {"mem0": "unqualified", "cognee": "unqualified"}


def test_actual_native_authored_environment_reward_and_close(tmp_path):
    d = driver()
    runtime = d.prepare_runtime(ROOT, tmp_path / "runtime")
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )

    job = Job("authored-shopping", "authored-shopping", tmp_path, 4, 600)
    outer = CampaignBudget(max_calls=4, max_seconds=600)
    outer.reserve(job)
    budget = JobContext(job, outer, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))
    with d.ShoppingRuntime(runtime, budget=budget, out=tmp_path / "evidence") as worker:
        opened = worker.request(
            "open",
            row={
                "id": -1,
                "category": "authored_item_0",
                "questions": ["Authored integration exercise. Product 1: Find an item."],
                "answers": [{"target_asin": "AUTHORED000", "attributes": []}],
            },
            index=0,
            max_rounds=2,
        )
        assert opened["observation"]["state"]
        assert "target_products" not in opened["public_observation"]
        result = worker.request("step", action="search[authored integration item]")
        assert isinstance(result["observation"]["state"], str)
        assert result["reward"] is None or type(result["reward"]) is bool
        with pytest.raises(ValueError, match="declared"):
            worker.request("exec", code="print(1)")
        closed = worker.request("close_session")
        assert closed["native_closed"] is True
        assert closed["raw_environment_active"] is False
    assert budget.calls == 0
    assert (
        json.loads((tmp_path / "evidence" / "runtime-closed.json").read_bytes())["process_exit"]
        == 0
    )


def profile():
    from gluomem_bench.bench.rack import RackProfile

    return RackProfile(
        base_url="https://example.invalid/v1",
        model="Qwen/declared-authored",
        max_tokens=128,
        max_generation_calls=8,
    )


def context(tmp_path, calls=8):
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )

    job = Job("authored", "authored", tmp_path, calls, 600)
    outer = CampaignBudget(max_calls=calls, max_seconds=600)
    outer.reserve(job)
    return JobContext(job, outer, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))


def test_native_actor_uses_exact_text_action_wire_without_tools(tmp_path):
    import httpx

    d = driver()
    runtime = d.prepare_runtime(ROOT, tmp_path / "runtime")
    sent = []

    def http(request):
        payload = json.loads(request.content) if request.content else None
        sent.append((request.url.path, payload))
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": profile().model, "max_model_len": profile().context_window}]},
            )
        if request.url.path == "/tokenize":
            return httpx.Response(
                200, json={"count": 100, "max_model_len": profile().context_window}
            )
        return httpx.Response(
            200,
            json={
                "model": profile().model,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "search[authored]"},
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5},
            },
        )

    budget = context(tmp_path)
    client = d.ShoppingTransport.offline(
        profile(),
        receipt=tmp_path / "http.jsonl",
        budget=budget,
        transport=httpx.MockTransport(http),
    )
    with d.native_actor(runtime, client) as actor:
        assert actor.act("Authored input") == "search[authored]"
        entry = json.loads(
            actor.build_memory_entry(
                "Authored input", "search[authored]", {"state": "public"}, reward=None, history=[]
            )
        )
        assert entry["final_observation"] == {"state": "public"}
    assert budget.calls == 1
    generation = sent[-1][1]
    assert generation["model"] == profile().model
    assert generation["stream"] is False and generation["n"] == 1
    assert "tools" not in generation and "tool_choice" not in generation
    assert sent[-2][1]["messages"] == generation["messages"]
    from gluomem_bench.bench.arena_campaign import NativeAbort

    with pytest.raises(NativeAbort, match="shopping.*tools"):
        client.create("actor", model=profile().model, messages=[], tools=[])
    assert budget.calls == 1
    client.close()


def test_native_task_full_ordered_sessions_and_dataset_free_evidence(tmp_path, monkeypatch):
    import httpx

    d = driver()
    runtime = d.prepare_runtime(ROOT, tmp_path / "runtime")
    row = {
        "id": -1,
        "category": "authored_item_0",
        "questions": [
            "Authored only. Product 1: Find an audio item.",
            "Authored only. Product 2: Find the preceding audio item.",
        ],
        "answers": [{"target_asin": "B09M975R23", "attributes": ["EVALUATOR_ONLY_SECRET"]}] * 2,
    }
    actions = iter(["search[MTX Audio RT8PT]", "click[B09M975R23]", "click[Buy Now]"] * 2)

    def http(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": profile().model}]})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100})
        return httpx.Response(
            200,
            json={
                "model": profile().model,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": next(actions)},
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5},
            },
        )

    budget = context(tmp_path)
    out = tmp_path / "task"
    client = d.ShoppingTransport.offline(
        profile(), receipt=out / "http.jsonl", budget=budget, transport=httpx.MockTransport(http)
    )
    result = d.run_task(
        row,
        runtime=runtime,
        client=client,
        memory=lambda b, o: d.common.RawExperienceMemory("full_context", budget=b),
        out=out,
        max_rounds=4,
    )
    client.close()
    assert result["status"] == "complete" and result["completed_sessions"] == 2
    assert result["calls"] == 6
    assert d.verify_task(out) == result
    rows = [json.loads(s) for s in (out / "steps.jsonl").read_bytes().splitlines()]
    assert [r["query_id"] for r in rows] == [0, 1]
    assert rows[0]["memory_context"] == "" and rows[1]["memory_context"]
    assert all(r["reward"] is True for r in rows)
    assert all("target_products" not in r["experience"]["content"] for r in rows)
    assert result["metrics"]["overall_success"] is True
    assert result["supplemental_metrics"]["rewards"] == [1.0, 1.0]
    assert result["supplemental_metrics"]["official_success_metric"] is False
    assert all(r["summary"]["reward"] is None for r in rows)
    assert "EVALUATOR_ONLY_SECRET" not in json.dumps([r["experience"] for r in rows])
    prompts = [t["input_messages"] for r in rows for t in r["turns"]]
    assert "EVALUATOR_ONLY_SECRET" not in json.dumps(prompts)
    assert "evaluator_reward" not in json.dumps(prompts)
    assert all(
        t["native_result"]["evaluator_reward"] is None for r in rows for t in r["turns"][:-1]
    )
    # Dataset-free replay must neither reopen assets/dataset nor create sockets.
    import builtins
    import socket

    real_open = builtins.open
    real_io_open = __import__("io").open

    def guarded_open(original):
        def checked(path, *a, **kw):
            if isinstance(path, (str, Path)):
                resolved = str(Path(path).resolve()).replace("\\", "/").lower()
                assert "/memoryarena-data/" not in resolved and "/native-assets/" not in resolved
            return original(path, *a, **kw)

        return checked

    with monkeypatch.context() as offline:
        offline.setattr(socket, "socket", lambda *a, **kw: pytest.fail("offline replay socket"))
        offline.setattr(builtins, "open", guarded_open(real_open))
        offline.setattr(__import__("io"), "open", guarded_open(real_io_open))
        assert d.verify_task(out) == result
    # Rehash every score-bearing artifact: actual native replay still rejects forgery.
    names = ["evaluator-rewards.jsonl", "steps.jsonl", "result.json"]
    original = {name: (out / name).read_bytes() for name in names}
    records = [json.loads(line) for line in original["evaluator-rewards.jsonl"].splitlines()]
    records[0]["evidence"]["result"]["reward"] = 0.123
    forged_steps = [json.loads(line) for line in original["steps.jsonl"].splitlines()]
    forged_steps[0]["turns"][-1]["native_result"]["evaluator_reward"] = records[0]["evidence"]
    for name, values in (("evaluator-rewards.jsonl", records), ("steps.jsonl", forged_steps)):
        (out / name).write_bytes(
            b"".join(json.dumps(v, ensure_ascii=False).encode("utf-8") + b"\n" for v in values)
        )
        result["artifact_sha256"][name] = d.sha(out / name)
    (out / "result.json").write_bytes(d.native._json(result))
    with pytest.raises(ValueError, match="reward"):
        d.verify_task(out)
    for name, material in original.items():
        (out / name).write_bytes(material)
    result = json.loads(original["result.json"])
    # Rehashing a forged wire file cannot promote invalid usage/status evidence.
    wire_path = out / "http.jsonl"
    original_wire, original_result = wire_path.read_bytes(), (out / "result.json").read_bytes()
    wire = [json.loads(s) for s in original_wire.splitlines()]
    next(w for w in wire if w["url"].endswith("/chat/completions"))["response"]["usage"][
        "prompt_tokens"
    ] = 99
    wire_path.write_bytes(b"".join(json.dumps(w).encode() + b"\n" for w in wire))
    result["artifact_sha256"]["http.jsonl"] = d.sha(wire_path)
    (out / "result.json").write_bytes(d.native._json(result))
    with pytest.raises(ValueError, match="usage|wire"):
        d.verify_task(out)
    wire_path.write_bytes(original_wire)
    (out / "result.json").write_bytes(original_result)
    closed_path = out / "runtime/runtime-closed.json"
    closed_bytes = closed_path.read_bytes()
    closed_path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        d.verify_task(out)
    closed_path.write_bytes(closed_bytes)
    assert d.verify_task(out)["status"] == "complete"


def test_full_freeze_requires_independent_approval_before_writes(tmp_path):
    d = driver()
    runtime = d.prepare_runtime(ROOT, tmp_path / "runtime")
    p = profile().model_copy(update={"max_generation_calls": 8, "max_run_seconds": 600})
    spec = d.build_spec(
        data_root=ROOT / "memoryarena-data",
        pins=json.loads((ROOT / "memoryarena-pins.json").read_bytes()),
        runtime=runtime,
        profile=p,
        output_root=tmp_path / "qa",
        embedding_url="https://example.invalid",
        calls_per_task=8,
        seconds_per_task=600,
        campaign_seconds=100000,
        max_rounds=25,
    )
    path = tmp_path / "spec.json"
    path.write_bytes(d.native._json(spec))
    approval = tmp_path / "approval.json"
    approval.write_bytes(b"{}")
    with pytest.raises(ValueError, match="independent"):
        d.execute_shopping(path, approval)
    assert not (tmp_path / "qa").exists()
    approval.write_bytes(
        d.native._json({"approved": True, "reviewer_kind": "parent", "spec_sha256": d.sha(path)})
    )
    assert d.verify_spec(path, approval)["sessions_count"] == 900
    spec["occurrences"].pop()
    path.write_bytes(d.native._json(spec))
    approval.write_bytes(
        d.native._json({"approved": True, "reviewer_kind": "parent", "spec_sha256": d.sha(path)})
    )
    with pytest.raises(ValueError, match="coverage"):
        d.verify_spec(path, approval)
    assert not (tmp_path / "qa").exists()


@pytest.mark.parametrize("fault", ["local", "shared", "verify", "publish"])
def test_matrix_failure_accounting_and_root_authority(tmp_path, fault):
    d = driver()
    from gluomem_bench.bench.rack import SharedRackError
    from gluomem_bench.bench.trajectory_suite import ArenaSession, ArenaTask

    tasks = [
        ArenaTask(i, d.DOMAIN, (ArenaSession("authored"),), ({"target_asin": "A"},))
        for i in range(3)
    ]
    seen = []

    def run_one(task, system, budget, target):
        seen.append(task.id)
        budget.begin_call()
        if task.id == 1 and fault in {"local", "shared"}:
            if fault == "shared":
                budget.budget.failed = True
                raise SharedRackError("identity differs")
            raise TimeoutError("one request timed out")
        target.mkdir(parents=True)
        (target / "result.json").write_bytes(
            d.native._json({"status": "complete", "completed_sessions": 1, "calls": 1})
        )
        return {"status": "complete", "completed_sessions": 1, "calls": 1}

    def verifier(target):
        if fault == "verify" and target.parent.name == "1":
            raise ValueError("external verifier failed")
        return json.loads((target / "result.json").read_bytes())

    def publish(path, value):
        if fault == "publish":
            raise OSError("root result write failed")
        path.write_bytes(d.native._json(value))

    out = tmp_path / "matrix"
    if fault == "publish":
        with pytest.raises(OSError):
            d.execute_matrix(
                tasks,
                systems=["no_memory"],
                out=out,
                call_budget=2,
                seconds_budget=10,
                campaign_seconds=100,
                run_one=run_one,
                verifier=verifier,
                publish=publish,
            )
        assert not (out / "campaign-result.json").exists()
        statuses = json.loads((out / "task-statuses.json").read_bytes())
        assert sum(r["calls"] for r in statuses) == 3
    else:
        result = d.execute_matrix(
            tasks,
            systems=["no_memory"],
            out=out,
            call_budget=2,
            seconds_budget=10,
            campaign_seconds=100,
            run_one=run_one,
            verifier=verifier,
            publish=publish,
        )
        assert result["tasks"][1]["status"] == "failed"
        assert result["tasks"][2]["status"] == ("unrun" if fault == "shared" else "complete")
        assert result["calls"] == (2 if fault == "shared" else 3)
        assert result["quality_complete"] is False
        assert json.loads((out / "campaign-result.json").read_bytes()) == result
        assert seen == ([0, 1] if fault == "shared" else [0, 1, 2])


def _authored_root_run(tmp_path, *, scored=False, **options):
    """Pure authored leaves, not native SDK or quality evidence."""
    d = driver()
    from gluomem_bench.bench.trajectory_suite import ArenaSession, ArenaTask

    task = ArenaTask(0, d.DOMAIN, (ArenaSession("authored"),), ({"target_asin": "A"},))

    def run_one(task, system, budget, target):
        budget.begin_call()
        target.mkdir(parents=True)
        value = {"status": "complete", "completed_sessions": 1, "calls": 1}
        if scored:
            source = (ROOT / "shopping-full-native-runtime-v1/source/run_shopping.py").read_bytes()
            (target / "native-source").mkdir()
            (target / "native-source/run_shopping.py").write_bytes(source)
            value["metrics"] = d.native_metrics(source, [{"match": True, "reward": None}])
            value["supplemental_metrics"] = {
                "metric": d.REWARD_METRIC,
                "official_success_metric": False,
                "sessions": 1,
                "rewards": [0.5],
                "mean_reward": 0.5,
            }
        (target / "result.json").write_bytes(d.native._json(value))
        return value

    return d.execute_matrix(
        [task],
        systems=["no_memory"],
        out=tmp_path,
        call_budget=2,
        seconds_budget=10,
        campaign_seconds=100,
        run_one=run_one,
        verifier=lambda p: json.loads((p / "result.json").read_bytes()),
        **options,
    )


@pytest.mark.parametrize("phase", ["after_leaf", "finalizer"])
def test_root_deadline_covers_leaf_release_and_finalizer(tmp_path, monkeypatch, phase):
    from gluomem_bench.bench.parallel_campaign import CampaignBudget, CampaignStopped

    budgets = []
    original_release = CampaignBudget.release

    def release(budget, context):
        original_release(budget, context)
        budgets.append(budget)
        if phase == "after_leaf":
            budget.clock = lambda: budget.deadline

    def publish(path, value):
        path.write_bytes(driver().native._json(value))
        budgets[0].clock = lambda: budgets[0].deadline

    monkeypatch.setattr(CampaignBudget, "release", release)
    root = tmp_path / "expired"
    with pytest.raises(CampaignStopped):
        _authored_root_run(root, publish=publish if phase == "finalizer" else None)
    assert not (root / "campaign-result.json").exists()
    assert json.loads((root / "task-statuses.json").read_bytes())[0]["calls"] == 1
    assert (root / "0/no_memory/result.json").exists()


def test_external_full_root_verifier_failure_blocks_promotion(tmp_path, monkeypatch):
    d = driver()
    seen = []

    def reject(*args, **kwargs):
        seen.append(True)
        raise ValueError("external full root verifier rejected")

    monkeypatch.setattr(d, "verify_root", reject, raising=False)
    root = tmp_path / "rejected"
    with pytest.raises(ValueError, match="external full root"):
        _authored_root_run(root)
    assert seen
    assert not (root / "campaign-result.json").exists()
    assert json.loads((root / "task-statuses.json").read_bytes())[0]["calls"] == 1


@pytest.mark.parametrize(
    "fault", [None, "method_metrics", "spec_sha256", "scope", "optional_coverage", "matrix"]
)
def test_root_rejects_rehashed_metadata_forgery(tmp_path, fault):
    d = driver()
    root = tmp_path / "root"

    def publish(path, value):
        frozen = json.loads((root / "frozen-matrix.json").read_bytes())
        spec = {
            "systems": ["no_memory"],
            "native_document_factories": {},
            "tasks": [{"id": 0, "sessions": 1, "input_sha256": frozen[0]["input_sha256"]}],
        }
        (root / "execution-spec.json").write_bytes(d.native._json(spec))
        value.update(
            spec_sha256=d.sha(root / "execution-spec.json"),
            scope=(
                "all released 150 tasks / 900 ordered session occurrences per admitted method; "
                "local simulated WebShop only"
            ),
            optional_coverage=[
                {
                    "task_id": 0,
                    "system": s,
                    "expected_sessions": 1,
                    "status": "unqualified",
                    "calls": 0,
                }
                for s in ("mem0", "cognee")
            ],
            method_metrics=d.method_metrics(
                value["tasks"], (root / "0/no_memory/native-source/run_shopping.py").read_bytes()
            ),
        )
        if fault == "matrix":
            spec["tasks"][0]["input_sha256"] = "forged"
            (root / "execution-spec.json").write_bytes(d.native._json(spec))
            value["spec_sha256"] = d.sha(root / "execution-spec.json")
        elif fault == "method_metrics":
            value["method_metrics"]["no_memory"]["official_metrics"] = {"forged": 0}
        elif fault is not None:
            value[fault] = "forged"
        path.write_bytes(d.native._json(value))

    if fault is None:
        result = _authored_root_run(root, scored=True, publish=publish)
        assert (
            d.verify_root(root, verifier=lambda p: json.loads((p / "result.json").read_bytes()))
            == result
        )
        return
    with pytest.raises(ValueError, match="root"):
        _authored_root_run(root, scored=True, publish=publish)
    assert not (root / "campaign-result.json").exists()


def test_root_publication_cannot_rewrite_actual_spend(tmp_path):
    d = driver()
    root = tmp_path / "root"

    def publish(path, value):
        value["tasks"][0]["calls"] = 9
        value["calls"] = 9
        leaf = root / "0/no_memory/result.json"
        saved = json.loads(leaf.read_bytes())
        saved["calls"] = 9
        leaf.write_bytes(d.native._json(saved))
        value["tasks"][0]["result_sha256"] = d.sha(leaf)
        (root / "task-statuses.json").write_bytes(d.native._json(value["tasks"]))
        path.write_bytes(d.native._json(value))

    with pytest.raises(ValueError, match="root"):
        _authored_root_run(root, publish=publish)
    assert not (root / "campaign-result.json").exists()


@pytest.mark.parametrize("phase", ["verify", "write", "staged_read", "replace", "final_read"])
@pytest.mark.parametrize("fault", ["deadline", "io"])
def test_root_finalization_faults_retain_leaves_and_allow_fresh_recovery(
    tmp_path, monkeypatch, phase, fault
):
    d = driver()
    from gluomem_bench.bench.parallel_campaign import CampaignBudget, CampaignStopped

    budgets = []
    release = CampaignBudget.release
    read = Path.read_bytes
    write = Path.write_bytes
    replace = Path.replace
    verify = d.verify_root

    def capture(budget, context):
        release(budget, context)
        budgets.append(budget)

    def trigger():
        if fault == "deadline":
            budgets[0].clock = lambda: budgets[0].deadline
        else:
            raise OSError("authored finalization fault")

    def checked_read(path):
        value = read(path)
        if (phase == "staged_read" and path.name == "campaign-result.pending.json") or (
            phase == "final_read" and path.name == "campaign-result.json"
        ):
            trigger()
        return value

    def checked_write(path, value):
        count = write(path, value)
        if phase == "write" and path.name == "campaign-result.pending.json":
            trigger()
        return count

    def checked_replace(path, target):
        value = replace(path, target)
        if phase == "replace" and path.name == "campaign-result.pending.json":
            trigger()
        return value

    def checked_verify(*args, **kwargs):
        value = verify(*args, **kwargs)
        if phase == "verify":
            trigger()
        return value

    root = tmp_path / "fault"
    with monkeypatch.context() as m:
        m.setattr(CampaignBudget, "release", capture)
        m.setattr(Path, "read_bytes", checked_read)
        m.setattr(Path, "write_bytes", checked_write)
        m.setattr(Path, "replace", checked_replace)
        m.setattr(d, "verify_root", checked_verify)
        with pytest.raises(CampaignStopped if fault == "deadline" else OSError):
            _authored_root_run(root)
    assert not (root / "campaign-result.json").exists()
    assert not (root / "campaign-result.pending.json").exists()
    assert (root / "0/no_memory/result.json").exists()
    assert json.loads((root / "task-statuses.json").read_bytes())[0]["calls"] == 1
    with pytest.raises(FileExistsError):
        _authored_root_run(root)
    recovered = _authored_root_run(tmp_path / "fresh")
    assert recovered["quality_complete"] is True and recovered["calls"] == 1


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "digest",
        "metrics",
        "supplemental",
        "calls",
        "total",
        "unrun",
        "failed",
    ],
)
def test_read_only_root_verifier_rejects_reconciled_leaf_and_boundary_forgeries(
    tmp_path, monkeypatch, fault
):
    import socket

    d = driver()
    root = tmp_path / "root"
    result = _authored_root_run(root, scored=True)
    row = result["tasks"][0]
    if fault == "missing":
        result["tasks"] = []
    elif fault == "duplicate":
        result["tasks"].append(dict(row))
    elif fault == "digest":
        row["result_sha256"] = "forged"
    elif fault == "metrics":
        row["metrics"]["overall_success"] = not row["metrics"]["overall_success"]
    elif fault == "supplemental":
        row["supplemental_metrics"]["mean_reward"] = 0.9
    elif fault == "calls":
        row["calls"] = 2
        result["calls"] = 2
    elif fault == "total":
        result["completed_sessions"] = 0
    else:
        row["status"] = fault
        result["quality_complete"] = False
    (root / "task-statuses.json").write_bytes(d.native._json(result["tasks"]))
    (root / "campaign-result.json").write_bytes(d.native._json(result))
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(socket, "socket", lambda *a, **kw: pytest.fail("root verification network"))
    with pytest.raises(ValueError, match="root"):
        d.verify_root(root, verifier=lambda p: json.loads((p / "result.json").read_bytes()))
    assert {p: p.read_bytes() for p in before} == before


def test_cli_prepare_freeze_and_unapproved_run(tmp_path, capsys):
    d = driver()
    p = tmp_path / "profile.json"
    p.write_bytes(
        d.native._json(
            profile().model_copy(update={"max_run_seconds": 600}).model_dump(mode="json")
        )
    )
    assert (
        d.main(
            ["prepare-runtime", "--external-root", str(ROOT), "--out", str(tmp_path / "runtime")]
        )
        == 0
    )
    spec_path = tmp_path / "spec.json"
    assert (
        d.main(
            [
                "freeze",
                "--data-root",
                str(ROOT / "memoryarena-data"),
                "--pins",
                str(ROOT / "memoryarena-pins.json"),
                "--runtime",
                str(tmp_path / "runtime/runtime.json"),
                "--profile",
                str(p),
                "--output-root",
                str(tmp_path / "qa"),
                "--embedding-url",
                "https://example.invalid",
                "--calls-per-task",
                "8",
                "--seconds-per-task",
                "600",
                "--campaign-seconds",
                "100000",
                "--max-rounds",
                "25",
                "--spec-out",
                str(spec_path),
            ]
        )
        == 0
    )
    approval = tmp_path / "approval.json"
    assert d.main(["approval-template", "--spec", str(spec_path), "--out", str(approval)]) == 0
    assert json.loads(approval.read_bytes())["approved"] is False
    with pytest.raises(ValueError, match="independent"):
        d.main(["run", "--spec", str(spec_path), "--approval", str(approval)])
    assert not (tmp_path / "qa").exists()


@pytest.mark.parametrize("fault,shared", [("timeout", False), ("usage", True), ("tool", False)])
def test_transport_failures_latch_without_native_retry(tmp_path, fault, shared):
    import httpx

    d = driver()
    runtime = d.prepare_runtime(ROOT, tmp_path / "runtime")
    attempts = []

    def http(request):
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100})
        attempts.append(request.url.path)
        if fault == "timeout":
            raise httpx.ReadTimeout("one request timeout")
        message = {"role": "assistant", "content": "search[authored]"}
        if fault == "tool":
            message["tool_calls"] = [
                {
                    "id": "x",
                    "type": "function",
                    "function": {"name": "python", "arguments": '{"code":"pass"}'},
                }
            ]
        return httpx.Response(
            200,
            json={
                "model": profile().model,
                "choices": [
                    {
                        "finish_reason": "tool_calls" if fault == "tool" else "stop",
                        "message": message,
                    }
                ],
                "usage": {"prompt_tokens": 99 if fault == "usage" else 100, "completion_tokens": 1},
            },
        )

    budget = context(tmp_path)
    client = d.ShoppingTransport.offline(
        profile(),
        receipt=tmp_path / "http.jsonl",
        budget=budget,
        transport=httpx.MockTransport(http),
    )
    with d.native_actor(runtime, client) as actor:
        with pytest.raises(d.native.NativeAbort):
            actor.act("Authored only")
    assert budget.calls == 1 and len(attempts) == 1
    assert budget.budget.failed is shared
    with pytest.raises(d.native.NativeAbort):
        client.create("actor", model=profile().model, messages=[])
    assert budget.calls == 1 and len(attempts) == 1
    client.close()


def test_memory_initialization_failure_retains_whole_task_spend(tmp_path):
    import httpx

    d = driver()
    runtime = d.prepare_runtime(ROOT, tmp_path / "runtime")
    budget = context(tmp_path)
    out = tmp_path / "task"
    client = d.ShoppingTransport.offline(
        profile(),
        receipt=out / "http.jsonl",
        budget=budget,
        transport=httpx.MockTransport(lambda r: pytest.fail("no HTTP admitted")),
    )

    def memory(budget, output):
        budget.begin_call()
        raise ValueError("authored memory ingestion failure")

    row = {
        "id": -1,
        "category": "authored_item_0",
        "questions": ["Authored Product 1: item"],
        "answers": [{"target_asin": "AUTHORED", "attributes": []}],
    }
    with pytest.raises(RuntimeError, match="task failed"):
        d.run_task(row, runtime=runtime, client=client, memory=memory, out=out, max_rounds=2)
    failure = json.loads((out / "failure.json").read_bytes())
    assert failure["calls"] == 1 and failure["completed_sessions"] == 0
    assert not (out / "result.json").exists()
    assert failure["sessions"][0]["status"] == "unrun"


def test_worker_has_explicit_utf8_for_released_unicode_inputs(tmp_path):
    d = driver()
    runtime = d.prepare_runtime(ROOT, tmp_path / "runtime")
    with d.ShoppingRuntime(runtime, budget=context(tmp_path), out=tmp_path / "worker") as worker:
        assert worker.process.args[1:3] == ["-X", "utf8"]


def test_closed_production_store_readback_is_independently_verified(tmp_path):
    import shutil
    import sqlite3

    d = driver()
    source = ROOT / "arena-full-authored-memory/task"
    for name in ("production.mind", "memory-events.jsonl"):
        shutil.copyfile(source / name, tmp_path / name)
    steps = [json.loads(s) for s in (source / "steps.jsonl").read_bytes().splitlines()]
    before = d.sha(tmp_path / "production.mind")
    d.verify_memory_store(tmp_path, steps)
    assert d.sha(tmp_path / "production.mind") == before
    assert not (tmp_path / "production.mind-wal").exists()
    db = sqlite3.connect(tmp_path / "production.mind")
    db.execute("DELETE FROM memory_provenance")
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="provenance"):
        d.verify_memory_store(tmp_path, steps)


@pytest.fixture
def privacy_store(tmp_path, monkeypatch):
    """Real commit/readback API, offline typed extraction, fresh synthetic DB."""
    from types import SimpleNamespace

    from gluomem.embedding.providers import MockEmbedder
    from gluomem.llm.modules import DspyExtractors
    from gluomem.llm.outputs import ExtractedClaimSpec

    from gluomem_bench.bench.arena_driver import GluoExperienceMemory
    from gluomem_bench.config import AppConfig

    cfg = AppConfig()
    cfg.storage.vector_dim = 384
    monkeypatch.setattr(
        DspyExtractors, "extract_event_claims",
        lambda *a, **kw: [ExtractedClaimSpec(
            text="The synthetic item is blue.", kind="fact",
            verbatim_span="synthetic item is blue",
        )],
    )
    memory = GluoExperienceMemory(
        cfg=cfg, extractors=DspyExtractors(cfg), embedder=MockEmbedder(dim=384),
        budget=SimpleNamespace(check=lambda: None), db_path=tmp_path / "production.mind",
        evidence_path=tmp_path / "memory-events.jsonl",
    )
    experience = {
        "authorship": "agent", "feedback_in_memory": False,
        "source_ref": "synthetic/shopping/0", "content": "The synthetic item is blue.",
    }
    try:
        receipt = memory.insert(experience)
    finally:
        memory.close()
    return tmp_path, [{"experience": experience, "memory_readback": receipt}]


@pytest.mark.parametrize("operation", ["correct", "forget"])
def test_closed_provenance_accepts_actual_privacy_transition(privacy_store, operation):
    from gluomem.embedding.providers import MockEmbedder
    from gluomem.frames.lifecycle import correct_memory, forget_memory
    from gluomem.frames.store import FrameStore
    from gluomem.storage.db import MindDB

    out, steps = privacy_store
    frame_id = steps[0]["memory_readback"]["provenance"][0]["frame_id"]
    db = MindDB(out / "production.mind", dim=384)
    try:
        store = FrameStore(db, embedder=MockEmbedder(dim=384))
        if operation == "correct":
            correct_memory(store, frame_id, "A corrected synthetic observation.")
        else:
            forget_memory(store, frame_id)
        assert db.fetchone("SELECT payload_json FROM source_events WHERE id=1")[0] == "{}"
        assert db.fetchone(
            "SELECT verbatim_span FROM memory_provenance WHERE frame_id=?", (frame_id,)
        )[0] is None
    finally:
        db.close()
    before = driver().sha(out / "production.mind")
    driver().verify_memory_store(out, steps)
    assert driver().sha(out / "production.mind") == before
    assert not (out / "production.mind-wal").exists()


@pytest.mark.parametrize("mutation", [
    "provenance_count", "boolean_provenance_count", "messages_float", "revision_bool",
    "row_id_bool", "content_hash", "unverified", "missing_frame", "extra_frame",
    "span", "null_active", "null_superseded_unredacted", "provenance_scope",
    "frame_scope", "source_hash", "source_id", "source_ref", "source_type",
    "source_revision", "source_scope", "missing_source", "extra_source", "missing_ack",
    "provenance_frame_bool", "provenance_event_bool", "orphan_provenance", "receipt_frame_missing",
])
def test_closed_provenance_rejects_forged_coverage(privacy_store, mutation):
    import sqlite3

    out, steps = privacy_store
    receipt = steps[0]["memory_readback"]
    db = sqlite3.connect(out / "production.mind")
    try:
        if mutation == "provenance_frame_bool":
            receipt["provenance"][0]["frame_id"] = True
        elif mutation == "provenance_event_bool":
            receipt["provenance"][0]["source_event_id"] = True
        elif mutation == "orphan_provenance":
            db.execute("INSERT INTO memory_provenance "
                       "(frame_id, source_type, source_ref) VALUES (999, 'document', 'orphan')")
        elif mutation == "receipt_frame_missing":
            receipt["provenance"] = []
            receipt["provenance_count"] = 0
            db.execute("DELETE FROM memory_provenance")
        elif mutation == "provenance_count":
            receipt["provenance_count"] += 1
        elif mutation == "boolean_provenance_count":
            receipt["provenance_count"] = True
        elif mutation == "messages_float":
            receipt["messages"] = 0.0
        elif mutation == "revision_bool":
            receipt["revision"] = True
        elif mutation == "row_id_bool":
            receipt["row_id"] = True
        elif mutation == "content_hash":
            receipt["content_sha256"] = "0" * 64
        elif mutation == "unverified":
            receipt["readback_verified"] = False
        elif mutation == "missing_frame":
            db.execute("DELETE FROM memory_provenance")
        elif mutation == "extra_frame":
            db.execute("INSERT INTO memory_provenance "
                       "(frame_id, source_event_id, source_type, source_ref, verbatim_span) "
                       "SELECT frame_id+99, source_event_id, source_type, source_ref, verbatim_span "
                       "FROM memory_provenance")
        elif mutation == "span":
            db.execute("UPDATE memory_provenance SET verbatim_span='changed'")
        elif mutation.startswith("null_"):
            db.execute("UPDATE memory_provenance SET verbatim_span=NULL")
            # A redacted shared envelope does not authorize erasing active or
            # still-plaintext frame spans.
            db.execute("UPDATE source_events SET payload_json='{}'")
            if mutation == "null_superseded_unredacted":
                db.execute("UPDATE memory_frames SET status='superseded'")
        elif mutation == "provenance_scope":
            db.execute("UPDATE memory_provenance SET namespace='forged'")
        elif mutation == "frame_scope":
            db.execute("UPDATE memory_frames SET namespace='forged'")
        elif mutation == "missing_source":
            db.execute("DELETE FROM source_events")
        elif mutation == "extra_source":
            db.execute("INSERT INTO source_events "
                       "(event_id, revision, source_type, source_ref, payload_json, payload_hash) "
                       "SELECT event_id || '/extra', revision, source_type, source_ref, "
                       "payload_json, payload_hash FROM source_events")
        elif mutation == "missing_ack":
            receipt.clear()
        else:
            key = {"source_hash": "payload_hash", "source_id": "event_id",
                   "source_ref": "source_ref", "source_type": "source_type",
                   "source_revision": "revision", "source_scope": "namespace"}[mutation]
            value = 2 if key == "revision" else "forged"
            db.execute(f"UPDATE source_events SET {key}=?", (value,))
            # A self-declared DB/receipt agreement must not establish trust.
            receipt[key] = value
        db.commit()
    finally:
        db.close()
    envelope = json.loads((out / "memory-events.jsonl").read_bytes())
    envelope["receipt"] = receipt
    (out / "memory-events.jsonl").write_text(json.dumps(envelope) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        driver().verify_memory_store(out, steps)


@pytest.mark.parametrize("task_id", range(7))
def test_closed_actual_shopping_sources_verify_only_on_readonly_copies(tmp_path, task_id):
    import shutil
    import sqlite3

    d = driver()
    source = ROOT / "arena-shopping-full-quality-v2" / str(task_id) / "gluomem"
    failure = json.loads((source / "failure.json").read_bytes())
    assert failure["native_closed"] is True and failure["status"] == "failed"
    names = ["production.mind", "memory-events.jsonl", "steps.jsonl", "failure.json"]
    before = {name: d.sha(source / name) for name in names}
    for name in names:
        shutil.copyfile(source / name, tmp_path / name)
    steps = [json.loads(s) for s in (tmp_path / "steps.jsonl").read_bytes().splitlines()]
    assert len(steps) == 6
    d.verify_memory_store(tmp_path, steps)
    db = sqlite3.connect((tmp_path / "production.mind").as_uri() + "?immutable=1", uri=True)
    try:
        assert db.execute("SELECT count(*) FROM source_events").fetchone()[0] == 6
        assert db.execute("SELECT count(*) FROM source_events WHERE payload_json='{}'").fetchone()[0] > 0
    finally:
        db.close()
    # Leaf diagnostic compatibility is NOT failed-bundle or root promotion.
    assert not (tmp_path / "result.json").exists()
    assert json.loads((tmp_path / "failure.json").read_bytes())["status"] == "failed"
    assert {name: d.sha(source / name) for name in names} == before
    assert {name: d.sha(tmp_path / name) for name in names} == before
    assert not (source / "production.mind-wal").exists()
    assert not (tmp_path / "production.mind-wal").exists()


def test_full_method_metrics_do_not_score_incomplete_jobs(tmp_path):
    d = driver()
    d.prepare_runtime(ROOT, tmp_path / "runtime")
    source = (tmp_path / "runtime/source/run_shopping.py").read_bytes()
    matrix = []
    for system in ("no_memory", "bm25"):
        for task_id in (0, 1):
            matrix.append(
                {
                    "system": system,
                    "task_id": task_id,
                    "status": "failed" if (system, task_id) == ("bm25", 1) else "complete",
                    "supplemental_metrics": {
                        "metric": d.REWARD_METRIC,
                        "official_success_metric": False,
                        "sessions": 6,
                        "rewards": [0.5] * 6,
                        "mean_reward": 0.5,
                    },
                    "metrics": {
                        "task_id": task_id,
                        "task_done": True,
                        "overall_success": task_id == 0,
                        "total_steps": 6,
                        "completed_steps": 6,
                        "matched_steps": 6 if task_id == 0 else 0,
                    },
                }
            )
    metrics = d.method_metrics(matrix, source)
    assert metrics["no_memory"]["official_metrics"]["total_steps"] == 12
    assert metrics["no_memory"]["official_metrics"]["num_task_success"] == 1
    assert metrics["no_memory"]["supplemental_metrics"]["mean_reward"] == 0.5
    assert metrics["no_memory"]["supplemental_metrics"]["sessions"] == 12
    assert "supplemental_metrics" not in metrics["bm25"]
    assert metrics["bm25"]["status"] == "incomplete-no-full-score"
    assert "official_metrics" not in metrics["bm25"]


def test_full_count_does_not_admit_modified_release_bytes(tmp_path):
    d = driver()
    data = ROOT / "memoryarena-data/bundled_shopping/data.jsonl"
    rows = [json.loads(line) for line in data.read_bytes().splitlines()]
    rows[0]["category"] = "modified-authored-category"
    target = tmp_path / "bundled_shopping/data.jsonl"
    target.parent.mkdir()
    target.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in rows))
    with pytest.raises(ValueError, match="released.*SHA256"):
        d.plan_dataset(tmp_path, {"bundled_shopping/data.jsonl": d.sha(target)})


def reward_dispatch_fixture(tmp_path, monkeypatch, *, done=True, missing=False, published=False):
    """Actual worker dispatch + pinned Plus method/scorer; authored env boundary."""
    import ast
    import importlib.util
    import sys
    from types import SimpleNamespace

    source = ROOT / "shopping-full-native-runtime-v1/source"
    helper_path = source / "env/env_systems/web_shopping_env/runtime/reward_helpers.py"
    spec = importlib.util.spec_from_file_location("_authored_native_reward", helper_path)
    helper = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, helper)
    spec.loader.exec_module(helper)
    tree = ast.parse(
        (source / "env/env_systems/web_shopping_env/webshop_plus_client.py").read_bytes()
    )
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WebshopPlusEnvClient"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_compute_reward_entry"
    )
    scope = {}
    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), "pinned-plus", "exec"),
        {"Optional": __import__("typing").Optional, "Dict": dict, "Any": object},
        scope,
    )
    gt = {
        "step": 1,
        "target_asin": "TARGET",
        "requirements": {
            "attributes": ["red", "missing"],
            "price_constraints": {"price_upper": 20},
        },
    }
    if published:
        gt = {
            "step": 1,
            "target_asin": "TARGET",
            "step_description": "Product 1",
            "requirements": {
                "product_type": "",
                "attributes": ["red", "missing"],
                "options": {},
                "price_constraints": {},
            },
        }
    catalog = helper.CatalogLookup(
        {"TARGET": "Home", "OTHER": "Home"}, {"TARGET": "red lamp", "OTHER": "red lamp alternative"}
    )

    class Client:
        enable_feedback = False
        _reward_helpers = {"compute_reward_for_step": helper.compute_reward_for_step}
        _compute_reward_entry = scope["_compute_reward_entry"]

        def _get_ordered_steps(self):
            return [gt]

        def _load_reward_helpers(self):
            pass

        def _get_catalog(self):
            return catalog

    client = Client()
    if missing:
        client._reward_helpers = {}
    observation = {"state": "public terminal"}
    info = {"purchased_asins": ["OTHER"], "purchased_prices": [10.0]}
    env = SimpleNamespace(
        _client=client,
        task_def={"target_products": ["TARGET"]},
        step=lambda *a, **kw: (observation, False if done else None, done, info),
    )
    worker_tree = ast.parse((ROOT / "shopping_native_worker.py").read_bytes())
    funcs = [
        n
        for n in worker_tree.body
        if isinstance(n, ast.FunctionDef) and n.name in {"dispatch", "public"}
    ]
    namespace = {
        "env": env,
        "step_def": gt,
        "re": __import__("re"),
        "source": source,
        "manifest": {"native_commit": driver().native.PIN},
        "Path": Path,
        "json": json,
        "hashlib": __import__("hashlib"),
        "summarize_step_from_final_state": lambda *a: {"reward": None},
    }
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "actual-worker", "exec"), namespace)
    return namespace["dispatch"]


def test_actual_worker_terminal_evaluator_reward_is_separate(tmp_path, monkeypatch):
    dispatch = reward_dispatch_fixture(tmp_path, monkeypatch)
    result = dispatch({"op": "step", "action": "click[Buy Now]"})
    assert result["reward"] is False
    assert result["summary"]["reward"] is None
    evidence = result["evaluator_reward"]
    assert evidence["result"]["reward"] == pytest.approx(2 / 3)
    assert evidence["enable_feedback"] is False
    assert evidence["metric"] == "supplemental_plus_attribute_price_reward"
    assert evidence["inputs"]["gt_step"]["target_asin"] == "TARGET"
    assert "TARGET" not in json.dumps(result["public_observation"])
    assert (
        reward_dispatch_fixture(tmp_path, monkeypatch, done=False)(
            {"op": "step", "action": "search[lamp]"}
        )["evaluator_reward"]
        is None
    )
    with pytest.raises(RuntimeError, match="reward"):
        reward_dispatch_fixture(tmp_path, monkeypatch, missing=True)(
            {"op": "step", "action": "click[Buy Now]"}
        )


def test_saved_reward_replay_executes_pinned_code_and_rejects_tampering(tmp_path, monkeypatch):
    import copy
    import shutil
    import socket

    d = driver()
    evidence = reward_dispatch_fixture(tmp_path, monkeypatch)(
        {"op": "step", "action": "click[Buy Now]"}
    )["evaluator_reward"]
    source = ROOT / "shopping-full-native-runtime-v1/source"
    for name in evidence["provenance"]["source_sha256"]:
        target = tmp_path / "native-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    monkeypatch.setattr(socket, "socket", lambda *a, **kw: pytest.fail("offline scorer network"))
    assert d.replay_evaluator_reward(tmp_path, evidence) == evidence["result"]
    for mutate in (
        lambda e: e["result"].update(reward=0.9),
        lambda e: e.update(enable_feedback=True),
        lambda e: e["inputs"]["catalog"]["name_by_asin"].update(OTHER="unrelated"),
        lambda e: e["provenance"].update(native_commit="forged"),
    ):
        forged = copy.deepcopy(evidence)
        mutate(forged)
        with pytest.raises(ValueError, match="reward"):
            d.replay_evaluator_reward(tmp_path, forged)
    with pytest.raises(ValueError, match="reward"):
        d.replay_evaluator_reward(tmp_path, None)


@pytest.mark.parametrize("fault", ["missing", "tampered", "worker_failed", "report"])
def test_driver_reward_failure_retains_spend_before_memory(tmp_path, monkeypatch, fault):
    import httpx

    d = driver()
    runtime = d.prepare_runtime(ROOT, tmp_path / "runtime")
    dispatch = reward_dispatch_fixture(tmp_path, monkeypatch, published=fault == "report")
    terminal = dispatch({"op": "step", "action": "click[Buy Now]"})
    if fault == "missing":
        terminal["evaluator_reward"] = None
    elif fault == "tampered":
        terminal["evaluator_reward"]["result"]["reward"] = 0.91

    class Worker:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def request(self, op, **kw):
            if op == "open":
                return {"conversation_start": [], "public_observation": {"state": "public"}}
            if op == "step":
                if fault == "worker_failed":
                    raise RuntimeError("native reward failed")
                return terminal
            return {"native_closed": True, "raw_environment_active": False}

    monkeypatch.setattr(d, "ShoppingRuntime", Worker)

    def http(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": profile().model}]})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100})
        return httpx.Response(
            200,
            json={
                "model": profile().model,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "click[Buy Now]"},
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5},
            },
        )

    budget = context(tmp_path)
    out = tmp_path / "task"
    client = d.ShoppingTransport.offline(
        profile(), receipt=out / "http.jsonl", budget=budget, transport=httpx.MockTransport(http)
    )
    memory = d.common.RawExperienceMemory("full_context", budget=budget)
    inserted = []
    monkeypatch.setattr(
        memory, "insert", lambda e: inserted.append(e) or {"readback_verified": True}
    )
    row = {
        "id": -1,
        "category": "authored_item_0",
        "questions": ["Authored Product 1: lamp"],
        "answers": [{"target_asin": "TARGET", "attributes": ["red", "missing"]}],
    }
    if fault == "report":
        # Reporting boundary only; full saved-bundle verifier runs in native integration.
        monkeypatch.setattr(
            d,
            "verify_task",
            lambda p, **kw: json.loads((p / kw.get("manifest_name", "result.json")).read_bytes()),
        )
        result = d.run_task(
            row, runtime=runtime, client=client, memory=memory, out=out, max_rounds=1
        )
        assert result["supplemental_metrics"]["metric"] == d.REWARD_METRIC
        assert result["supplemental_metrics"]["rewards"] == [0.5]
        assert result["metrics"]["steps"][0]["reward"] is None
        assert result["calls"] == 1
        return
    with pytest.raises(RuntimeError, match="task failed"):
        d.run_task(row, runtime=runtime, client=client, memory=memory, out=out, max_rounds=1)
    failure = json.loads((out / "failure.json").read_bytes())
    assert failure["calls"] == 1
    assert failure["completed_sessions"] == 0
    assert "reward" in failure["error"]
    assert inserted == []
    assert not (out / "result.json").exists()


def test_verifier_requires_evaluator_reward_artifact(tmp_path):
    d = driver()
    old_required = [
        "http.jsonl",
        "input-task.json",
        "effective-profile.json",
        "steps.jsonl",
        "turns.jsonl",
        "runtime/runtime.jsonl",
        "runtime/runtime-closed.json",
        "native-source/run_shopping.py",
        "native-source/agent/webshop.py",
    ]
    (tmp_path / "result.json").write_bytes(
        d.native._json(
            {
                "protocol": d.PROTOCOL,
                "status": "complete",
                "native_closed": True,
                "artifact_sha256": dict.fromkeys(old_required, "not-opened"),
            }
        )
    )
    with pytest.raises(ValueError, match="reward"):
        d.verify_task(tmp_path)


def test_reward_coverage_binds_native_inputs_and_reports_separate_metric(tmp_path, monkeypatch):
    import copy
    import shutil

    d = driver()
    source = ROOT / "shopping-full-native-runtime-v1/source"
    for name in [
        *d.REWARD_SOURCES,
        "env/env_systems/web_shopping_env/runtime/runner/task_files.py",
    ]:
        target = tmp_path / "native-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    final = reward_dispatch_fixture(tmp_path, monkeypatch, published=True)(
        {"op": "step", "action": "click[Buy Now]"}
    )
    row = {
        "id": -1,
        "category": "authored_item_0",
        "questions": ["Authored Product 1: lamp"],
        "answers": [{"target_asin": "TARGET", "attributes": ["red", "missing"]}],
    }
    step = {"query_id": 0, "turns": [{"native_result": final}]}
    records = [{"query_id": 0, "evidence": final["evaluator_reward"]}]
    report = d.verify_reward_coverage(tmp_path, row, [step], records)
    assert report == {
        "metric": d.REWARD_METRIC,
        "official_success_metric": False,
        "sessions": 1,
        "rewards": [0.5],
        "mean_reward": 0.5,
    }
    for fault in ("result", "input", "coverage", "nonterminal"):
        changed, altered = copy.deepcopy([step]), copy.deepcopy(records)
        if fault == "result":
            altered[0]["evidence"]["result"]["reward"] = 0.8
            changed[0]["turns"][-1]["native_result"]["evaluator_reward"] = altered[0]["evidence"]
        if fault == "input":
            altered[0]["evidence"]["inputs"]["gt_step"]["requirements"]["attributes"] = []
            changed[0]["turns"][-1]["native_result"]["evaluator_reward"] = altered[0]["evidence"]
        if fault == "coverage":
            altered = []
        if fault == "nonterminal":
            changed[0]["turns"].insert(
                0, {"native_result": {"done": False, "evaluator_reward": altered[0]["evidence"]}}
            )
        with pytest.raises(ValueError, match="reward"):
            d.verify_reward_coverage(tmp_path, row, changed, altered)


def test_native_memory_replay_rejects_evaluator_payload_injection(tmp_path):
    import copy
    import shutil
    from types import SimpleNamespace

    d = driver()
    runtime = d.prepare_runtime(ROOT, tmp_path / "runtime")
    path = "agent/webshop.py"
    target = tmp_path / "native-source" / path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tmp_path / "runtime/source" / path, target)
    row = {"questions": ["Authored public instruction"]}
    observation = {"state": "public"}
    turn = {
        "turn_idx": 1,
        "input_messages": [{"role": "user", "content": "public prompt"}],
        "action": "click[Buy Now]",
        "raw_output": "click[Buy Now]",
        "native_result": {"done": True, "public_observation": observation},
    }
    client = SimpleNamespace(profile=profile(), sdk=lambda role: None)
    with d.native_actor(runtime, client) as actor:
        actor._last_raw_output = turn["raw_output"]
        actor.record_turn(
            turn_idx=1,
            prompt="public prompt",
            action=turn["action"],
            observation=observation,
            reward=None,
            done=True,
            info={},
        )
        content = actor.build_memory_entry(
            row["questions"][0], turn["action"], observation, reward=None
        )
    step = {"query_id": 0, "turns": [turn], "experience": {"content": content}}
    d.verify_public_experience(tmp_path, row, step)
    for field in ("evaluator_reward", "gold", "feedback"):
        forged = copy.deepcopy(step)
        payload = json.loads(content)
        payload[field] = "PRIVATE EVALUATOR ONLY"
        forged["experience"]["content"] = json.dumps(payload)
        with pytest.raises(ValueError, match="experience"):
            d.verify_public_experience(tmp_path, row, forged)


def test_reward_source_closure_matches_official_pinned_git_objects():
    import hashlib
    import subprocess

    d = driver()
    expected = {
        **d.REWARD_SOURCES,
        "run_shopping.py": "",
        "env/env_systems/web_shopping_env/runtime/runner/task_files.py": (
            "1ed0b8e7d2ab825e69f7cc475f625b88a121e48ef3de3676c9c4dbec8fd40d8a"
        ),
        "agent/webshop.py": "b346bf2d7d4786770848c95c1159a87a00bea5cdb2724fd062c223311a7e1918",
    }
    source = ROOT / "shopping-full-native-runtime-v1/source"
    for name, digest in expected.items():
        official = subprocess.check_output(
            ["git", "-C", str(ROOT / "MemoryArena"), "show", d.native.PIN + ":" + name]
        )
        assert official == (source / name).read_bytes()
        if digest:
            assert hashlib.sha256(official).hexdigest() == digest
