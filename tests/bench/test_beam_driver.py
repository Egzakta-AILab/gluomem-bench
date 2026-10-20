"""Authored HTTP boundary only; no live quality calls in this module."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def test_full_matrix_all_four_tiers_and_missing_duplicate_rejection(tmp_path):
    driver = api()
    selections = []
    for split, count in (("100K", 20), ("500K", 35), ("1M", 35), ("10M", 10)):
        for i in range(1, count + 1):
            history = f"beam-{split}-{i}"
            selections.append(dict(split=split, history_id=history, questions=20,
                                   case_ids=[f"{history}-q{j}" for j in range(20)],
                                   typed_ingestion_admitted=True))
    systems = ["gluomem", "mem0", "no_memory", "bm25", "dense", "hybrid", "full_context"]
    jobs = driver.full_job_matrix(selections, systems, tmp_path, 16384, 21600)
    spec = dict(selections=selections, systems=systems, jobs=jobs, output_root=str(tmp_path))
    assert driver.validate_full_matrix(spec) == 14000
    for bad in (jobs[:-1], jobs + [jobs[0]]):
        with pytest.raises(ValueError, match="matrix"):
            driver.validate_full_matrix(dict(spec, jobs=bad))
    with pytest.raises(ValueError, match="histories"):
        driver.full_job_matrix(selections[:-1], systems, tmp_path, 16384, 21600)


def test_clockless_selection_preserves_original_anchors_and_explicit_nonadmission():
    adapter = authored_adapter()
    for batch in adapter._raw[0]["chat"]:
        for message in batch:
            message.pop("time_anchor", None)
    receipt = api().selection_receipt(adapter)
    assert receipt["typed_ingestion_admitted"] is False
    assert "no invented timestamp" in receipt["ingestion_blocker"]
    assert receipt["projection_sha256"] is None
    assert receipt["questions"] == 20
    assert all(m["time_anchor"] is None for m in adapter.chat_messages())


@pytest.mark.parametrize("shared_failure", [False, True])
@pytest.mark.parametrize("root_fault", [False, True])
def test_full_executor_failed_middle_job_continues_and_na_is_not_quality(tmp_path, upstream, monkeypatch, root_fault, shared_failure):
    from gluomem_bench.bench.rack import RackProfile
    profile = RackProfile(base_url="http://localhost:1/v1", model="authored/Qwen-boundary",
                          profile_version="role-budgets-v5", generation_budgets=dict(
                              extraction=4096, consolidation=512, reader=512, judge=1024))
    root = tmp_path / "full"
    spec = dict(protocol="beam-full-development@v1", output_root=str(root),
                profile=profile.model_dump(mode="json"), embedding_url=profile.base_url,
                systems=["no_memory"], config_overrides={}, upstream_dir=str(upstream),
                data_dir=str(upstream.parent), campaign_policy=dict(max_workers=1,
                    max_in_flight=1, max_in_flight_tokens=200000, max_calls=600,
                    max_seconds=900, stop_on_failure=False),
                selections=[dict(history_id=f"h{i}") for i in range(3)],
                jobs=[dict(job_id=f"j{i}", history_id=f"h{i}", split="100K", system="no_memory",
                    output=str(root / "jobs" / f"j{i}"), call_budget=100,
                    seconds_budget=120, case_ids=[f"q{i}-{j}" for j in range(20)]) for i in range(3)])
    spec_path, approval = tmp_path / "spec.json", tmp_path / "approval.json"
    spec_path.write_bytes(json.dumps(spec).encode())
    approval.write_bytes(b'{}')
    monkeypatch.setattr(api(), "verify_execution_spec", lambda *a: spec)
    monkeypatch.setattr(api(), "select_history", lambda *a, **k: authored_adapter())
    monkeypatch.setattr(api(), "selection_receipt", lambda a: dict(history_id="ignore"))
    # Selection equality is admitted by verify_execution_spec; fault seam is execution only.
    spec["selections"] = [dict(history_id=f"h{i}", admitted=True) for i in range(3)]
    seen = []
    def run(**kw):
        seen.append(kw["out"].name)
        kw["outer_budget"].begin_call()
        if kw["out"].name == "j1":
            if shared_failure:
                kw["outer_budget"].budget.failed = True
            raise ValueError("authored middle failure")
        out = kw["out"]
        out.mkdir(parents=True)
        value = dict(status="not_applicable_context_overflow", complete=False)
        api().write(out / "root-status.json", value)
        return value
    monkeypatch.setattr(api(), "run_history", run)
    # This three-row scheduler fixture is deliberately not a native full matrix.
    # The full verifier is exercised by the 700-row root tests and real authored
    # request isolation cases below, rather than accepting invalid pilot metadata.
    monkeypatch.setattr(api(), "verify_campaign", lambda out, _candidate_path=None:
        json.loads((Path(_candidate_path) if _candidate_path else Path(out) / "root-status.json").read_bytes()))
    expected_seen = ["j0", "j1"] if shared_failure else ["j0", "j1", "j2"]
    expected_calls = 2 if shared_failure else 3
    if root_fault:
        from gluomem_bench.bench.parallel_campaign import CampaignReceiptError
        original_write = api().write
        def fail_root(path, value):
            if path == root / "root-status.json.pending" and "accounting" in value:
                raise OSError("authored root finalization fault")
            return original_write(path, value)
        monkeypatch.setattr(api(), "write", fail_root)
        with pytest.raises(CampaignReceiptError) as raised:
            api().execute(spec_path, approval)
        assert raised.value.receipt["accounting"]["calls"] == expected_calls
        assert json.loads((root / "root-status.json").read_bytes())["complete"] is False
        assert seen == expected_seen
        return
    result = api().execute(spec_path, approval)
    assert seen == expected_seen
    assert [j["status"] for j in result["jobs"]] == ["complete", "failed", "cancelled" if shared_failure else "complete"]
    assert result["accounting"]["calls"] == expected_calls
    assert result["complete"] is False
    assert "report" not in result
    assert result["frozen_inputs"] == []  # Fault-injection spec has no source inventory.
    assert all("artifact_sha256" in row for row in result["jobs"])
    assert json.loads((root / "root-status.json").read_bytes()) == result


@pytest.mark.parametrize("fault", ["timeout", "production_timeout", "wrong_model", "tokenizer", "preflight", "accounting"])
def test_full_request_timeout_keeps_failed_leaf_and_runs_healthy_next_job(tmp_path, upstream, monkeypatch, fault):
    import httpx
    from gluomem_bench.bench.rack import RackProfile
    from gluomem_bench.bench.parallel_campaign import CampaignStopped

    driver = api()
    with server() as (url, calls):
        profile = RackProfile(base_url=url, model="authored/Qwen-boundary",
            profile_version="role-budgets-v6", max_generation_calls=200,
            max_input_chars=200000, generation_budgets=dict(extraction=16384,
                consolidation=4096, reader=4096, judge=1024))
        root = tmp_path / "full-timeout"
        adapters = {}
        for i in range(3):
            adapter = authored_adapter()
            adapter._raw[0]["conversation_id"] = f"authored-{i}"
            adapters[i] = adapter
        spec = dict(protocol=driver.FULL_PROTOCOL, output_root=str(root),
            profile=profile.model_dump(mode="json"), embedding_url=url,
            systems=["no_memory"], config_overrides={}, upstream_dir=str(upstream),
            data_dir=str(upstream.parent), campaign_policy=dict(max_workers=1,
                max_in_flight=1, max_in_flight_tokens=200000, max_calls=600,
                max_seconds=3600, stop_on_failure=False),
            jobs=[dict(job_id=f"j{i}", history_id=adapters[i].load()[0].conv_id, split="100K",
                system="no_memory", output=str(root / "jobs" / f"j{i}"),
                call_budget=200, seconds_budget=900, case_ids=[s.id for s in adapters[i].load()]) for i in range(3)])
        spec_path, approval = tmp_path / "spec.json", tmp_path / "approval.json"
        driver.write(spec_path, spec)
        driver.write(approval, dict(approved=True,
            spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest()))
        monkeypatch.setattr(driver, "verify_execution_spec", lambda *a: spec)
        by_history = {a.load()[0].conv_id: a for a in adapters.values()}
        monkeypatch.setattr(driver, "select_history", lambda split, data, history, **k: by_history[history])
        monkeypatch.setattr(driver, "validate_full_matrix", lambda s: 60)
        original_run, original_request = driver.run_history, httpx.Client.request
        active, contexts, seen = {}, {}, []
        def run(**kw):
            name = kw["out"].name
            active["job"] = name
            seen.append(name)
            contexts[name] = kw["outer_budget"]
            return original_run(**kw)
        if fault == "production_timeout":
            spec["jobs"][1]["system"] = "gluomem"
            driver.write(spec_path, spec)
            driver.write(approval, dict(approved=True,
                spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest()))
            import dspy
            import litellm
            original_lm = dspy.LM.__call__
            def lm(model, *a, **kw):
                if active.get("job") == "j1":
                    raise litellm.Timeout(model=profile.model, llm_provider="openai",
                                          message="authored production generation timeout")
                return original_lm(model, *a, **kw)
            monkeypatch.setattr(dspy.LM, "__call__", lm)
        readers = 0
        def request(client, method, url, **kw):
            nonlocal readers
            body = kw.get("json") or {}
            if active.get("job") == "j1":
                if fault == "preflight" and str(url).endswith("/models"):
                    raise httpx.ConnectError("authored shared preflight unavailable")
                if fault == "tokenizer" and str(url).endswith("/tokenize"):
                    return httpx.Response(200, json={"count": None},
                                          request=httpx.Request(method, url))
                if fault == "accounting" and str(url).endswith("/chat/completions"):
                    contexts["j1"].budget.failed = True
                    raise OSError("authored broken campaign call accounting")
            if (active.get("job") == "j1" and str(url).endswith("/chat/completions")
                    and body["messages"][0]["content"] == driver.READER_INSTRUCTION):
                readers += 1
                if readers == 2 and fault == "timeout":
                    raise httpx.ReadTimeout("authored single generation request timeout")
                if readers == 2 and fault == "wrong_model":
                    return httpx.Response(200, json={"model": "wrong-model"},
                                          request=httpx.Request(method, url))
            return original_request(client, method, url, **kw)
        monkeypatch.setattr(driver, "run_history", run)
        monkeypatch.setattr(httpx.Client, "request", request)
        result = driver.execute(spec_path, approval)
    shared = fault not in {"timeout", "production_timeout"}
    assert seen == (["j0", "j1"] if shared else ["j0", "j1", "j2"])
    assert [j["status"] for j in result["jobs"]] == ["complete", "failed", "cancelled" if shared else "complete"]
    assert not result["complete"] and result["status"] == "incomplete"
    assert result["matrix_executed"] is not shared and "report" not in result
    assert contexts["j1"].budget.failed is shared
    if not shared:
        assert contexts["j1"].failed
        with pytest.raises(CampaignStopped):
            contexts["j1"].begin_call()  # Never unlatch or retry this failed job.
    leaf = driver.verify_history(root / "jobs/j1")
    assert leaf["status"] == "failed"
    assert leaf["completed_questions"] == (1 if fault in {"timeout", "wrong_model"} else 0)
    assert leaf["generation_calls"] == contexts["j1"].calls
    cost = json.loads((root / "jobs/j1/cost.json").read_bytes())
    if not shared:
        assert cost["unmetered_generation_calls"] == 1
    assert cost["wall_seconds"] > 0
    assert result["accounting"]["calls"] == sum(c.calls for c in contexts.values())
    assert driver.verify_campaign(root) == result
    if not shared:
        assert driver.verify_history(root / "jobs/j2")["complete"]
        assert sum(path.endswith("/models") for path, _ in calls) >= 3
    if fault == "production_timeout":
        assert list((root / "jobs/j1/stores").glob("*.mind"))
        assert (root / "jobs/j1/production-raw.jsonl").is_file()


@pytest.mark.parametrize("field", ["samples_sha256", "rubrics_sha256", "chat_sha256", "profile"])
def test_root_binds_native_leaf_to_external_frozen_selection(tmp_path, upstream, monkeypatch, field):
    driver = api()
    with server() as (url, _):
        args, outer = setup_run(url, tmp_path, upstream, system="no_memory")
        leaf = driver.run_history(**args)
    assert leaf["complete"], leaf.get("error")
    selection = driver.selection_receipt(args["adapter"])
    root = tmp_path
    record = dict(job_id="authored", history_id=selection["history_id"], split="100K",
        system="no_memory", case_ids=selection["case_ids"], output=str(args["out"]))
    spec = dict(jobs=[record], selections=[selection], protocol=driver.FULL_PROTOCOL,
                profile=args["profile"].model_dump(mode="json"))
    saved = dict(quality_status=leaf["status"], result=leaf,
        root_status_sha256=hashlib.sha256((args["out"] / "root-status.json").read_bytes()).hexdigest())
    state = dict(complete=True, status="complete_verified", jobs=[dict(job_id="authored", status="complete",
        result=saved, calls=outer.calls, artifact_sha256={p.relative_to(root).as_posix():
        hashlib.sha256(p.read_bytes()).hexdigest() for p in args["out"].rglob("*") if p.is_file()})])
    def seal():
        driver.write(root / "execution-spec.json", spec)
        state["spec_sha256"] = hashlib.sha256((root / "execution-spec.json").read_bytes()).hexdigest()
        driver.write(root / "independent-approval.json", dict(approved=True, spec_sha256=state["spec_sha256"]))
        driver.write(root / "root-status.json", state)
    monkeypatch.setattr(driver, "validate_full_matrix", lambda s: 20)
    seal()
    driver.verify_campaign(root)
    if field == "profile":
        spec["profile"]["model"] = "forged/model"
    else:
        selection[field] = "0" * 64
    seal()
    with pytest.raises(ValueError, match="frozen.*selection|selection.*binding"):
        driver.verify_campaign(root)


@pytest.mark.parametrize("fault", ["verifier", "write", "replace", "final_readback", "deadline"])
def test_complete_campaign_publication_failure_revokes_even_a_verified_native_leaf(tmp_path, upstream, monkeypatch, fault):
    from gluomem_bench.bench.parallel_campaign import CampaignBudget, CampaignReceiptError
    driver = api()
    with server() as (url, _):
        args, outer = setup_run(url, tmp_path, upstream, system="no_memory")
        leaf = driver.run_history(**args)
    assert leaf["complete"]
    selection = driver.selection_receipt(args["adapter"])
    root = tmp_path
    record = dict(job_id="authored", history_id=selection["history_id"], split="100K",
                  system="no_memory", case_ids=selection["case_ids"], output=str(args["out"]))
    spec = dict(jobs=[record], selections=[selection], protocol=driver.FULL_PROTOCOL)
    driver.write(root / "execution-spec.json", spec)
    sha = hashlib.sha256((root / "execution-spec.json").read_bytes()).hexdigest()
    driver.write(root / "independent-approval.json", dict(approved=True, spec_sha256=sha))
    saved = dict(quality_status=leaf["status"], result=leaf,
        root_status_sha256=hashlib.sha256((args["out"] / "root-status.json").read_bytes()).hexdigest())
    state = dict(complete=True, status="complete_verified", spec_sha256=sha,
        jobs=[dict(job_id="authored", status="complete", result=saved, calls=outer.calls,
        artifact_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in args["out"].rglob("*") if p.is_file()})])
    monkeypatch.setattr(driver, "validate_full_matrix", lambda s: 20)
    clock = [0.0]
    budget = CampaignBudget(max_calls=200, max_seconds=120, clock=lambda: clock[0])
    for _ in range(outer.calls):
        budget.begin_call()
    original_write, original_read, original_verify = driver.write, Path.read_bytes, driver.verify_campaign
    if fault == "write":
        def write(path, value):
            if path == root / "root-status.json.pending":
                raise OSError("authored staged write failure")
            return original_write(path, value)
        monkeypatch.setattr(driver, "write", write)
    elif fault == "replace":
        monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("authored replacement failure")))
    elif fault == "final_readback":
        def read(path):
            if path == root / "root-status.json":
                raise OSError("authored promoted readback failure")
            return original_read(path)
        monkeypatch.setattr(Path, "read_bytes", read)
    else:
        def verify(*a, **kw):
            result = original_verify(*a, **kw)
            if fault == "verifier":
                raise ValueError("authored root verifier failure")
            clock[0] = 121
            return result
        monkeypatch.setattr(driver, "verify_campaign", verify)
    with pytest.raises(CampaignReceiptError) as raised:
        driver._publish_campaign(root, state, budget)
    persisted = json.loads(original_read(root / "root-status.json"))
    assert persisted["complete"] is False and persisted["status"] == "incomplete"
    assert persisted["accounting"]["calls"] == raised.value.receipt["accounting"]["calls"] == outer.calls
    assert not (root / "root-status.json.pending").exists()
    assert json.loads(original_read(args["out"] / "root-status.json"))["complete"] is True


def test_full_profile_requires_v6_reader4096_and_exact_native_judge_bound():
    from types import SimpleNamespace
    def profile(version="role-budgets-v6", reader=4096, judge=4096):
        return SimpleNamespace(profile_version=version,
                               output_budget=lambda task: reader if task == "answer" else judge)
    assert api().validate_profile(profile(), full=True) == 4096
    assert api().validate_profile(profile(judge=1024), full=True) == 1024
    for invalid in (profile(version="role-budgets-v5"), profile(reader=512), profile(judge=2048)):
        with pytest.raises(ValueError):
            api().validate_profile(invalid, full=True)


def test_full_prepare_native_all_100_histories_2000_questions_and_no_clock(upstream, tmp_path, monkeypatch):
    settings = dict(profile=dict(base_url="http://localhost:1/v1", model="authored/Qwen-boundary",
        profile_version="role-budgets-v6", max_generation_calls=16384, max_run_seconds=21600,
        max_input_chars=2000000, generation_budgets=dict(extraction=16384,
            consolidation=4096, reader=4096, judge=1024)), embedding_url="http://localhost:2/v1",
        systems=list(api().SUPPORTED_SYSTEMS), config_overrides={},
        campaign_policy=dict(max_workers=8, max_in_flight=16, max_in_flight_tokens=2000000,
                             max_calls=12000000, max_seconds=999999, stop_on_failure=False))
    monkeypatch.setattr("httpx.Client", lambda *a, **k: pytest.fail("prepare must be offline"))
    settings_path, spec_path = tmp_path / "settings.json", tmp_path / "spec.json"
    settings_path.write_bytes(json.dumps(settings).encode())
    assert api().main(["prepare-full", "--settings", str(settings_path),
        "--data-dir", str(upstream.parent), "--upstream-dir", str(upstream),
        "--output-root", str(tmp_path / "full"), "--out", str(spec_path)]) == 0
    spec = json.loads(spec_path.read_bytes())
    assert len(spec["selections"]) == 100
    assert len(spec["jobs"]) == 700
    assert api().validate_full_matrix(spec) == 14000
    missing_clock = [s for s in spec["selections"] if not s["typed_ingestion_admitted"]]
    assert [s["history_id"] for s in missing_clock] == ["beam-10M-7"]
    assert spec["scope"] == "all published development; no heldout claim"
    assert not (tmp_path / "full").exists()
    spec_path, approval_path = tmp_path / "spec.json", tmp_path / "approval.json"
    spec_path.write_bytes(json.dumps(spec).encode())
    approval_path.write_bytes(json.dumps(dict(approved=True, reviewer="independent-authored-test",
        reviewer_kind="AI-assisted", spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest())).encode())
    assert api().verify_execution_spec(spec_path, approval_path) == spec


@pytest.mark.parametrize("system", ["no_memory", "bm25", "dense", "hybrid", "full_context"])
def test_clockless_faithful_controls_never_create_a_date(tmp_path, upstream, system):
    with server() as (url, _):
        args, outer = setup_run(url, tmp_path, upstream, system=system)
        for batch in args["adapter"]._raw[0]["chat"]:
            for message in batch:
                message.pop("time_anchor", None)
        result = api().run_history(**args)
    assert result["complete"], result.get("error")
    assert outer.calls > 0
    assert json.loads((args["out"] / "projected-conversation.json").read_bytes()) is None
    source = json.loads((args["out"] / "source-chat.json").read_bytes())
    assert all(row["time_anchor"] is None for row in source)
    assert api().verify_history(args["out"]) == result


@pytest.mark.parametrize("system", ["gluomem", "mem0"])
def test_clockless_typed_input_not_admitted_before_any_client(tmp_path, upstream, system):
    with server() as (url, calls):
        args, outer = setup_run(url, tmp_path, upstream, system=system)
        args["adapter"]._raw[0]["chat"][0][0].pop("time_anchor")
        result = api().run_history(**args)
    assert not calls and outer.calls == 0
    assert result["status"] == "input_not_admitted_missing_source_clock"
    assert result["complete"] is False and "report" not in result
    assert api().verify_history(args["out"]) == result


def test_v6_native_judge4096_exact_wire_and_offline_replay(tmp_path, upstream):
    from gluomem_bench.bench.rack import RackProfile, controlled_config
    from gluomem_bench.bench.rack_trace import WorkloadBudget
    with server(judge_tokens=4096) as (url, calls):
        args, _ = setup_run(url, tmp_path, upstream, system="no_memory")
        material = args["profile"].model_dump(mode="json")
        material.update(profile_version="role-budgets-v6", generation_budgets=dict(
                        extraction=16384, consolidation=4096, reader=4096, judge=4096))
        args["profile"] = RackProfile.model_validate(material)
        args["cfg"] = controlled_config(args["profile"], data=args["adapter"].path,
                                         sha256=args["adapter"].sha, embedder=args["embedder"])
        args["outer_budget"] = WorkloadBudget(max_calls=200, max_seconds=120, policy="role-budgets-v6")
        result = api().run_history(**args)
    assert result["complete"], result.get("error")
    assert all(body["max_tokens"] == 4096 for path, body in calls if path.endswith("/chat/completions"))
    assert api().verify_history(args["out"]) == result


def test_full_root_offline_authority_rejects_missing_matrix_without_scoring_unrun(tmp_path):
    selections = [dict(split=split, history_id=f"beam-{split}-{i}", questions=20,
                       case_ids=[f"beam-{split}-{i}-q{j}" for j in range(20)])
        for split, count in (("100K", 20), ("500K", 35), ("1M", 35), ("10M", 10))
        for i in range(1, count + 1)]
    jobs = api().full_job_matrix(selections, list(api().SUPPORTED_SYSTEMS), tmp_path, 16384, 21600)
    spec = dict(protocol=api().FULL_PROTOCOL, selections=selections, jobs=jobs,
                systems=list(api().SUPPORTED_SYSTEMS), output_root=str(tmp_path))
    api().write(tmp_path / "execution-spec.json", spec)
    sha = hashlib.sha256((tmp_path / "execution-spec.json").read_bytes()).hexdigest()
    api().write(tmp_path / "independent-approval.json", dict(approved=True, spec_sha256=sha))
    state = dict(complete=False, status="incomplete", spec_sha256=sha,
        jobs=[dict(job_id=j["job_id"], status="unrun", artifact_sha256={}) for j in jobs])
    api().write(tmp_path / "root-status.json", state)
    assert api().verify_campaign(tmp_path) == state
    state["jobs"].pop()
    api().write(tmp_path / "root-status.json", state)
    with pytest.raises(ValueError, match="matrix"):
        api().verify_campaign(tmp_path)


def api():
    return importlib.import_module("gluomem_bench.bench.beam_driver")


@pytest.fixture
def upstream():
    path = os.environ.get("BEAM_UPSTREAM_DIR")
    if not path:
        pytest.skip("BEAM_UPSTREAM_DIR must name the actual byte-pinned archive")
    return Path(path)


def authored_adapter():
    from gluomem_bench.bench.beam_scoring import CATEGORIES
    from gluomem_bench.bench.beam_suite import BeamSuiteAdapter

    return BeamSuiteAdapter.from_rows(
        "100K",
        [
            {
                "conversation_id": "authored",
                "chat": [
                    [
                        {
                            "id": "gold-label-unsafe",
                            "role": "user",
                            "content": "Station locker code is cobalt.",
                            "time_anchor": "January-01-2025",
                        },
                        {"id": "assistant-native", "role": "assistant", "content": "Acknowledged."},
                    ],
                    [
                        {
                            "id": "update-native",
                            "role": "user",
                            "content": "Station locker code is willow.",
                            "time_anchor": "January-02-2025",
                        }
                    ],
                ],
                "probing_questions": {
                    category: [
                        {
                            "question": f"{category} probe {i}: what is the code?",
                            "rubric": [
                                "Station locker code is cobalt.",
                                "Station locker code is willow.",
                            ]
                            if category == "event_ordering"
                            else ["Use the code."],
                        }
                        for i in range(2)
                    ]
                    for category in CATEGORIES
                },
                "profile": "NEVER INDEX PROFILE",
                "plan": "NEVER INDEX PLAN",
            }
        ],
    )


@contextmanager
def server(*, finish="stop", count=None, bad_judge=False, judge_tokens=1024):
    """Real socket so production DSPy/LMRouter is exercised, not patched."""
    from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe

    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            calls.append((self.path, None))
            self.reply({"data": [{"id": "authored/Qwen-boundary", "max_model_len": 131072}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body))
            if self.path == "/tokenize":
                return self.reply({"count": count(body["messages"]) if count else 100})
            if self.path.endswith("/embeddings"):
                return self.reply(
                    {
                        "recipe": EmbeddingRecipe().as_dict(),
                        "role": body["input_type"],
                        "data": [
                            {"index": i, "embedding": [1.0] + [0.0] * 767}
                            for i, _ in enumerate(body["input"])
                        ],
                    }
                )
            messages = json.dumps(body["messages"])
            if "supporting verbatim quote" in messages:
                output = {
                    "claims": [
                        {"text": text, "verbatim_span": text}
                        for text in (
                            "Station locker code is cobalt.",
                            "Station locker code is willow.",
                        )
                        if text in body["messages"][-1]["content"]
                    ]
                }
            elif "target_frame_id" in messages:
                output = {"decision": {"action": "ADD"}}
            elif body.get("top_p") == 1.0:
                output = {"memory": [{"text": "Station locker code is cobalt."}]}
            elif (body["max_tokens"] == judge_tokens
                  and body["messages"][0]["content"] != api().READER_INSTRUCTION):
                output = (
                    "YES"
                    if len(body["messages"]) > 1
                    else (
                        "malformed" if bad_judge else '{"score": 1, "reason": "authored verdict"}'
                    )
                )
            else:
                output = "Station locker code is cobalt.\nStation locker code is willow."
            text = json.dumps(output) if isinstance(output, dict) else output
            tokens = count(body["messages"]) if count else 100
            self.reply(
                {
                    "id": "authored",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": finish,
                            "message": {"role": "assistant", "content": text},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": tokens,
                        "completion_tokens": 10,
                        "total_tokens": tokens + 10,
                    },
                }
            )

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1", calls
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def setup_run(url, tmp_path, upstream, system="gluomem", max_calls=200):
    from gluomem_bench.bench.embedding_runtime import RemoteEmbedder
    from gluomem_bench.bench.rack import RackProfile, controlled_config
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    profile = RackProfile(
        base_url=url,
        model="authored/Qwen-boundary",
        profile_version="role-budgets-v5",
        max_generation_calls=max_calls,
        max_input_chars=200000,
        generation_budgets={"extraction": 4096, "consolidation": 512, "reader": 512, "judge": 1024},
    )
    embedder = RemoteEmbedder(url)
    cfg = controlled_config(
        profile, data=tmp_path / "authored.json", sha256="0" * 64, embedder=embedder
    )
    outer = WorkloadBudget(max_calls=max_calls, max_seconds=120, policy=profile.profile_version)
    return dict(
        adapter=authored_adapter(),
        cfg=cfg,
        profile=profile,
        out=tmp_path / "history",
        system=system,
        upstream_dir=upstream,
        embedder=embedder,
        outer_budget=outer,
        authored=True,
    ), outer


@pytest.mark.parametrize("version", ["role-budgets-v5", "role-budgets-v6"])
def test_authored_vertical_production_sqlite_native_scoring_and_offline_replay(
    tmp_path, upstream, monkeypatch, version
):
    monkeypatch.setenv("GLUOMEM_RACK_API_KEY", "authored-placeholder")
    with server() as (url, calls):
        args, outer = setup_run(url, tmp_path, upstream)
        if version == "role-budgets-v6":
            from gluomem_bench.bench.rack import RackProfile, controlled_config
            from gluomem_bench.bench.rack_trace import WorkloadBudget
            material = args["profile"].model_dump(mode="json")
            material["profile_version"] = version
            material["generation_budgets"]["reader"] = 4096
            args["profile"] = RackProfile.model_validate(material)
            args["cfg"] = controlled_config(args["profile"], data=args["adapter"].path,
                sha256=args["adapter"].sha, embedder=args["embedder"])
            outer = args["outer_budget"] = WorkloadBudget(max_calls=200, max_seconds=120, policy=version)
        result = api().run_history(**args)
    assert result["status"] == "complete_verified"
    assert result["evidence_kind"] == "authored-offline-integration"
    assert result["expected_questions"] == result["completed_questions"] == 20
    assert api().verify_history(args["out"]) == result
    rows = json.loads((args["out"] / "predictions.json").read_bytes())
    assert len(rows) == 20
    assert rows[0]["sample"]["question_date"] is None
    production = [
        json.loads(line)
        for line in (args["out"] / "production-raw.jsonl").read_bytes().splitlines()
    ]
    assert {r["task"] for r in production} >= {"event_claims", "consolidate"}
    assert all("probe" not in json.dumps(r["messages"]) for r in production)
    assert all("gold-label-unsafe" not in json.dumps(r["messages"]) for r in production)
    coverage = json.loads((args["out"] / "coverage.json").read_bytes())
    assert coverage["messages"] == 3 and coverage["events"] == 2
    assert coverage["assistant_claim_extraction"].startswith("excluded_by_core_policy")
    assert result["stores_closed"] is True
    assert not list((args["out"] / "stores").glob("*-wal"))
    assert (
        outer.calls
        == result["generation_calls"]
        == len(production) + 20 + sum(len(r["scoring"]["judgments"]) for r in rows)
    )
    assert any(path.endswith("/embeddings") for path, _ in calls)
    report = result["report"]["rows"][0]["by_category"]
    assert report["event_ordering"]["metric"] == "tau_norm"
    assert report["event_ordering"]["questions"] == 2
    assert result["report"]["judge_fidelity"] == "adapted-model; not reference-model reproduction"


def test_whole_line_ceiling_and_rehashed_raw_input_tamper_is_rejected(tmp_path, upstream):
    def count(messages):
        if messages[0]["content"] == api().READER_INSTRUCTION:
            return 100 + 35 * len(json.loads(messages[-1]["content"])["evidence"])
        return 100

    with server(count=count) as (url, _):
        args, _ = setup_run(url, tmp_path, upstream, system="bm25")
        result = api().run_history(**args)
    assert result["complete"]
    inputs = json.loads((args["out"] / "rawReaderInputs.json").read_bytes())
    assert inputs[0]["original_tokens"] > 4096
    assert inputs[0]["kept_tokens"] <= 4096
    assert inputs[0]["original_text"].startswith(inputs[0]["kept_text"])
    assert inputs[0]["kept_text"].endswith("\n")
    assert api().verify_history(args["out"]) == result
    inputs[0]["kept_tokens"] += 1
    path = args["out"] / "rawReaderInputs.json"
    path.write_bytes(json.dumps(inputs).encode())
    result["artifact_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args["out"] / "root-status.json").write_bytes(json.dumps(result).encode())
    with pytest.raises(ValueError, match="reader|token|evidence"):
        api().verify_history(args["out"])


def test_full_context_fit_only_checks_every_probe_before_any_generation(tmp_path, upstream, monkeypatch):
    def count(messages):
        if messages[0]["content"] != api().READER_INSTRUCTION:
            return 100
        question = json.loads(messages[-1]["content"])["question"]
        return 5000 if question.startswith("temporal_reasoning probe 1") else 100

    with server(count=count) as (url, calls):
        args, outer = setup_run(url, tmp_path, upstream, system="full_context")
        result = api().run_history(**args)
    assert result["status"] == "not_applicable_context_overflow"
    assert result["complete"] is False and "report" not in result
    assert outer.calls == 0
    assert len([p for p, _ in calls if p == "/tokenize"]) == 20
    assert not any(p.endswith("/chat/completions") for p, _ in calls)
    assert api().verify_history(args["out"]) == result
    driver = api()
    selection = driver.selection_receipt(args["adapter"])
    spec = dict(protocol=driver.FULL_PROTOCOL, profile=args["profile"].model_dump(mode="json"),
        selections=[selection], jobs=[dict(job_id="authored", history_id=selection["history_id"],
        split="100K", system="full_context", output=str(args["out"]), case_ids=selection["case_ids"])])
    driver.write(tmp_path / "execution-spec.json", spec)
    sha = hashlib.sha256((tmp_path / "execution-spec.json").read_bytes()).hexdigest()
    driver.write(tmp_path / "independent-approval.json", dict(approved=True, spec_sha256=sha))
    state = dict(complete=False, status="incomplete", spec_sha256=sha, jobs=[dict(
        job_id="authored", status="complete", calls=0, result=dict(quality_status=result["status"],
        result=result, root_status_sha256=hashlib.sha256((args["out"] / "root-status.json").read_bytes()).hexdigest()),
        artifact_sha256={p.relative_to(tmp_path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in args["out"].rglob("*") if p.is_file()})])
    driver.write(tmp_path / "root-status.json", state)
    monkeypatch.setattr(driver, "validate_full_matrix", lambda s: 20)
    assert driver.verify_campaign(tmp_path) == state


def test_real_parquet_plan_freezes_sources_and_refuses_missing_approval_before_models(
    tmp_path, upstream, monkeypatch
):
    from gluomem_bench.bench.rack import RackProfile

    profile = RackProfile(
        base_url="http://127.0.0.1:1/v1",
        model="explicit/Qwen",
        profile_version="role-budgets-v5",
        generation_budgets={"extraction": 4096, "consolidation": 512, "reader": 512, "judge": 1024},
    )
    settings = dict(
        profile=profile.model_dump(mode="json"),
        embedding_url=profile.base_url,
        systems=["no_memory", "gluomem"],
        config_overrides={},
        max_calls=256,
        max_seconds=120,
    )
    spec = api().build_execution_spec(
        split="100K",
        history_id="beam-100K-1",
        data_dir=upstream.parent,
        upstream_dir=upstream,
        output_root=tmp_path / "native-run",
        settings=settings,
    )
    assert spec["selection"]["questions"] == 20
    assert len(spec["selection"]["case_ids"]) == 20
    assert spec["selection"]["messages"] > 100
    assert spec["selection"]["roles"] == ["assistant", "user"]
    assert spec["evidence_token_budget"] == 4096
    paths = {item["path"] for item in spec["files"]}
    from gluomem_bench.bench.beam_scoring import SOURCE_SHA256

    assert {str((upstream / name).resolve()) for name in SOURCE_SHA256} <= paths
    assert str(Path(api().__file__).resolve()) in paths
    assert str((Path(api().__file__).resolve().parents[2] / "tests/bench/test_beam_driver_continuation.py").resolve()) in paths
    from scipy.stats import _stats

    assert str(Path(_stats.__file__).resolve()) in paths
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(json.dumps(spec).encode())
    monkeypatch.setattr("httpx.Client", lambda *a, **k: pytest.fail("models before approval"))
    with pytest.raises(ValueError, match="approval"):
        api().execute(spec_path, tmp_path / "absent-approval.json")
    assert not (tmp_path / "native-run").exists()
    approval = tmp_path / "approval.json"
    approval.write_bytes(
        json.dumps(
            {
                "approved": True,
                "reviewer_kind": "human",
                "reviewer": "independent-test-reviewer",
                "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            }
        ).encode()
    )
    assert api().verify_execution_spec(spec_path, approval)["selection"] == spec["selection"]
    spec["files"] = [f for f in spec["files"] if not f["path"].endswith("compute_metrics.py")]
    spec_path.write_bytes(json.dumps(spec).encode())
    approval.write_bytes(
        json.dumps(
            {
                "approved": True,
                "reviewer_kind": "human",
                "reviewer": "independent-test-reviewer",
                "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            }
        ).encode()
    )
    with pytest.raises(ValueError, match="inventory"):
        api().execute(spec_path, approval)
    assert not (tmp_path / "native-run").exists()


def test_cli_prepare_is_offline_and_requires_frozen_native_all_question_selection(
    tmp_path, upstream, monkeypatch, capsys
):
    from gluomem_bench.bench.rack import RackProfile

    profile = RackProfile(
        base_url="http://127.0.0.1:1/v1",
        model="operator/Qwen",
        profile_version="role-budgets-v5",
        generation_budgets={"extraction": 4096, "consolidation": 512, "reader": 512, "judge": 1024},
    )
    settings = dict(
        profile=profile.model_dump(mode="json"),
        embedding_url=profile.base_url,
        systems=["no_memory"],
        config_overrides={},
        max_calls=100,
        max_seconds=120,
    )
    settings_path = tmp_path / "settings.json"
    settings_path.write_bytes(json.dumps(settings).encode())
    spec_path = tmp_path / "native-spec.json"
    monkeypatch.setattr(
        "httpx.Client", lambda *a, **k: pytest.fail("prepare cannot construct clients")
    )
    assert (
        api().main(
            [
                "prepare",
                "--settings",
                str(settings_path),
                "--split",
                "100K",
                "--history-id",
                "beam-100K-1",
                "--data-dir",
                str(upstream.parent),
                "--upstream-dir",
                str(upstream),
                "--output-root",
                str(tmp_path / "never-created"),
                "--out",
                str(spec_path),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["spec_sha256"] == hashlib.sha256(spec_path.read_bytes()).hexdigest()
    assert output["inference_admitted"] is False
    assert json.loads(spec_path.read_bytes())["selection"]["questions"] == 20
    assert not (tmp_path / "never-created").exists()


def test_shared_budget_failure_keeps_raw_reader_and_failed_judgments_without_quality_zero(
    tmp_path, upstream
):
    with server() as (url, _):
        args, outer = setup_run(url, tmp_path, upstream, system="no_memory", max_calls=1)
        result = api().run_history(**args)
    assert result["status"] == "failed" and result["complete"] is False
    assert outer.calls == result["generation_calls"] == 1
    assert result["completed_questions"] == 0 and "report" not in result
    predictions = json.loads((args["out"] / "predictions.json").read_bytes())
    assert len(predictions) == 1 and "scoring" not in predictions[0]
    assert json.loads((args["out"] / "rawReaderInputs.json").read_bytes())[0]["response"]["text"]
    failure = json.loads((args["out"] / "failure.json").read_bytes())
    assert failure["scoring_progress"]["judgments"][0]["status"] == "request_failed"
    cost = json.loads((args["out"] / "cost.json").read_bytes())
    assert cost["prompt_tokens_observed"] == 100
    assert cost["completion_tokens_observed"] == 10
    assert cost["unmetered_generation_calls"] == 0
    assert api().verify_history(args["out"]) == result


@pytest.mark.parametrize("system", ["bm25", "dense", "hybrid"])
def test_raw_controls_bind_chat_only_corpus_roles_dates_and_same_common_reader(
    tmp_path, upstream, system
):
    with server() as (url, calls):
        args, outer = setup_run(url, tmp_path, upstream, system=system)
        result = api().run_history(**args)
    assert result["complete"]
    chunks = json.loads((args["out"] / "raw-inputs.json").read_bytes())
    assert len(chunks) == 3
    assert {c["provenance"]["role"] for c in chunks} == {"assistant", "user"}
    assert all("2025-01-" in c["provenance"]["time"] for c in chunks)
    assert all("NEVER INDEX" not in c["text"] and "probe" not in c["text"] for c in chunks)
    assert all(c["provenance"]["source_turn_id"].startswith("s") for c in chunks)
    backend = json.loads((args["out"] / "raw-backend.json").read_bytes())
    assert backend["mode"] == system
    assert outer.calls == 20 + sum(
        len(row["scoring"]["judgments"])
        for row in json.loads((args["out"] / "predictions.json").read_bytes())
    )
    inputs = json.loads((args["out"] / "rawReaderInputs.json").read_bytes())
    assert all(row["kept_tokens"] <= 4096 for row in inputs)
    assert all("Current Date:" not in json.dumps(row["messages"]) for row in inputs)
    assert any("assistant: Acknowledged." in chunk["text"] for chunk in chunks)
    assert api().verify_history(args["out"]) == result


def test_execution_retains_attempted_failure_and_all_unrun_plans_as_root_authority(
    tmp_path, upstream, monkeypatch
):
    # Orchestrator fault injection only; native read-only byte gate is tested above.
    from gluomem_bench.bench.rack import RackProfile

    profile = RackProfile(
        base_url="http://127.0.0.1:1/v1",
        model="operator/Qwen",
        profile_version="role-budgets-v5",
        generation_budgets={"extraction": 4096, "consolidation": 512, "reader": 512, "judge": 1024},
    )
    spec = dict(
        profile=profile.model_dump(mode="json"),
        embedding_url=profile.base_url,
        systems=["gluomem", "no_memory", "full_context"],
        config_overrides={},
        max_calls=100,
        max_seconds=120,
        output_root=str(tmp_path / "campaign"),
        history_id="beam-100K-authored",
        upstream_dir=str(upstream),
        selection=api().selection_receipt(authored_adapter()),
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(json.dumps(spec).encode())
    approval = tmp_path / "approval.json"
    approval.write_bytes(b'{"test":"orchestrator fault injection, not native admission"}')
    monkeypatch.setattr(api(), "verify_execution_spec", lambda *a: spec)
    monkeypatch.setattr(api(), "dependency_preflight", lambda *a: authored_adapter())

    def fail(**kwargs):
        raise RuntimeError("authored pre-store fault")

    monkeypatch.setattr(api(), "run_history", fail)
    result = api().execute(spec_path, approval)
    assert (tmp_path / "campaign/execution-spec.json").read_bytes() == spec_path.read_bytes()
    assert result["complete"] is False and result["status"] == "failed"
    assert [j["status"] for j in result["jobs"]] == ["failed", "unrun", "unrun"]
    assert all(len(j["case_ids"]) == 20 for j in result["jobs"])
    assert result["generation_calls"] == 0
    assert "report" not in result
    assert json.loads((tmp_path / "campaign/root-status.json").read_bytes()) == result


def test_boolean_flag_cannot_bypass_native_independent_approval(tmp_path, upstream):
    with server() as (url, calls):
        args, _ = setup_run(url, tmp_path, upstream, system="no_memory")
        args.update(authored=False, _admitted=True)
        with pytest.raises(ValueError, match="approval"):
            api().run_history(**args)
    assert not calls and not args["out"].exists()


def test_native_mem0_chat_guarded_helper_executes_in_fresh_process(tmp_path, upstream):
    import subprocess
    import sys

    code = """
import json, os, runpy, sys
from pathlib import Path
os.environ.pop('OPENAI_API_KEY', None)
os.environ.pop('OPENROUTER_API_KEY', None)
os.environ['MEM0_TELEMETRY'] = 'False'
os.environ['MEM0_DIR'] = str(Path(sys.argv[2]) / 'native-home')
t = runpy.run_path(sys.argv[4])
args, outer = t['setup_run'](sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), system='mem0')
result = t['api']().run_history(**args)
print(json.dumps(result))
"""
    with server() as (url, calls):
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                url,
                str(tmp_path),
                str(upstream),
                str(Path(__file__).resolve()),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout.splitlines()[-1])
    assert result["complete"], result.get("error")
    assert result["stores_closed"]
    native = json.loads((tmp_path / "history/mem0-ingestion.json").read_bytes())
    assert native["complete"] and native["submitted_turns"] == 3
    inputs = json.loads((tmp_path / "history/mem0-source-inputs.json").read_bytes())
    assert inputs[0]["messages"][1]["role"] == "assistant"
    assert all("probe" not in json.dumps(row["messages"]) for row in inputs)
    coverage = json.loads((tmp_path / "history/coverage.json").read_bytes())
    assert coverage["messages"] == 3 and coverage["events"] == 2
    assert api().verify_history(tmp_path / "history") == result


def test_staged_native_source_cannot_claim_authored_loopback_exemption(tmp_path, upstream):
    with server() as (url, calls):
        args, _ = setup_run(url, tmp_path, upstream, system="no_memory")
        args["adapter"]._source_files = [
            (
                upstream / "LICENSE",
                hashlib.sha256((upstream / "LICENSE").read_bytes()).hexdigest(),
                (upstream / "LICENSE").stat().st_size,
            )
        ]
        with pytest.raises(ValueError, match="approval"):
            api().run_history(**args)
    assert not calls and not args["out"].exists()


def test_final_offline_verification_failure_never_leaves_complete_root_status(
    tmp_path, upstream, monkeypatch
):
    def reject(*args, **kwargs):
        raise ValueError("authored finalization verifier fault")

    monkeypatch.setattr(api(), "verify_history", reject)
    with server() as (url, _):
        args, _ = setup_run(url, tmp_path, upstream, system="no_memory")
        result = api().run_history(**args)
    assert result["status"] == "failed" and result["complete"] is False
    assert "report" not in result
    assert "finalization" in result["error"]
    assert json.loads((args["out"] / "root-status.json").read_bytes()) == result
    assert (
        json.loads((args["out"] / "failure.json").read_bytes())["phase"] == "offline-verification"
    )


def test_preflight_pins_download_parser_source_not_only_scorer_labels(tmp_path, upstream):
    from gluomem_bench.bench.beam_scoring import SOURCE_SHA256
    from gluomem_bench.bench.rack import RackProfile

    archive = tmp_path / "archive"
    for name in set(SOURCE_SHA256) | {"src/beam/download_dataset.py"}:
        path = archive / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((upstream / name).read_bytes())
    parser = archive / "src/beam/download_dataset.py"
    parser.write_bytes(parser.read_bytes() + b"\n# unapproved changed bytes\n")
    profile = RackProfile(
        base_url="http://127.0.0.1:1/v1",
        model="operator/Qwen",
        profile_version="role-budgets-v5",
        generation_budgets={"extraction": 4096, "consolidation": 512, "reader": 512, "judge": 1024},
    )
    settings = dict(
        profile=profile.model_dump(mode="json"),
        embedding_url=profile.base_url,
        systems=["no_memory"],
        config_overrides={},
        max_calls=100,
        max_seconds=120,
    )
    from gluomem_bench.bench.adapters.locomo import DatasetError

    with pytest.raises(DatasetError, match="source.*hash|hash mismatch"):
        api().build_execution_spec(
            split="100K",
            history_id="beam-100K-1",
            data_dir=upstream.parent,
            upstream_dir=archive,
            output_root=tmp_path / "never-created",
            settings=settings,
        )
    assert not (tmp_path / "never-created").exists()
