"""Offline authored Search integration: never contact a benchmark/model endpoint."""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(
    os.environ.get(
        "ARENA_SEARCH_NATIVE_ROOT",
        'tests/fixtures/native/operator-supplied',
    )
)


def test_native_tool_and_scorer_authored_boundary(monkeypatch):
    import socket

    def deny(*args, **kwargs):
        raise OSError("offline test denies all network")

    monkeypatch.setattr(socket.socket, "connect", deny)
    from gluomem_bench.bench import arena_search_driver as search

    native = search.load_native(ROOT)

    class Corpus:
        def search(self, query, k, allowed_docids=None):
            assert (query, k, allowed_docids) == ("authored lake", 10, None)
            return [{"docid": "original-01", "text": "An authored blue lake.", "score": -0.5}]

        def get_document(self, docid):
            return (
                {"docid": docid, "text": "An authored blue lake."}
                if docid == "original-01"
                else None
            )

        def search_description(self, k):
            return f"Search {k} documents"

        def get_document_description(self):
            return "Fetch original document"

    tools = native.SearchToolHandler(Corpus(), snippet_max_tokens=None, k=10)
    assert [t["function"]["name"] for t in tools.get_tool_definitions()] == [
        "search",
        "get_document",
    ]
    assert (
        json.loads(tools.execute_tool("search", {"query": "authored lake"}))[0]["docid"]
        == "original-01"
    )
    assert json.loads(tools.execute_tool("get_document", {"docid": "missing"}))["error"]
    parsed = native.parse_judge_response(
        "extracted_final_answer: blue\nreasoning: matches\ncorrect: yes\nconfidence: 92"
    )
    assert parsed["correct"] is True and parsed["confidence"] == 92
    assert native.parse_judge_response("invalid")["parse_error"] is True


def test_refuse_unapproved_encoder_before_any_output(tmp_path, monkeypatch):
    from gluomem_bench.bench import arena_search_driver as search

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"protocol": search.PROTOCOL, "output_root": str(tmp_path / "run")}))
    approval = tmp_path / "approval.json"
    approval.write_text("{}")
    with pytest.raises(ValueError, match="approval"):
        search.execute(spec, approval)
    assert not (tmp_path / "run").exists()


def test_projection_keeps_ids_and_every_character():
    from gluomem_bench.bench import arena_search_driver as search

    row = {
        "ordinal": 0,
        "docid": "0007",
        "text_sha256": search.hashlib.sha256(b"abcdef").hexdigest(),
        "spans": [[0, 2, 4], [2, 6, 8]],
    }
    assert search.project_document({"docid": "0007", "text": "abcdef"}, row) == ["ab", "cdef"]
    with pytest.raises(ValueError, match="coverage"):
        search.project_document(
            {"docid": "0007", "text": "abcdef"}, {**row, "spans": [[0, 2, 4], [3, 6, 8]]}
        )
    with pytest.raises(ValueError, match="identity"):
        search.project_document({"docid": 7, "text": "abcdef"}, row)


def test_native_actor_guarded_http_fixture(tmp_path):
    import httpx

    from gluomem_bench.bench import arena_search_driver as search
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )
    from gluomem_bench.bench.rack import RackProfile

    payloads = []

    def wire(request):
        assert "authorization" not in request.headers
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "authored-model"}]})
        body = json.loads(request.content)
        payloads.append(body)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 20})
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "id": "authored",
                "object": "chat.completion",
                "created": 0,
                "model": "authored-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "An authored blue lake."},
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
            },
        )

    budget = CampaignBudget(max_calls=8, max_seconds=120)
    job = Job("fixture", "fixture", tmp_path, 8, 120)
    assert budget.reserve(job)
    ctx = JobContext(job, budget, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))
    profile = RackProfile(base_url="http://fixture.invalid/v1", model="authored-model")
    native = search.load_native(ROOT)
    client = search.SearchTransport.authored(
        profile, receipt=tmp_path / "http.jsonl", budget=ctx, transport=httpx.MockTransport(wire)
    )

    class Handler:
        def get_tool_definitions(self):
            return []

    request = native.build_request(
        "An authored query", "authored-model", 512, Handler(), temperature=0
    )
    final, trace, usage, _ = native.run_conversation_with_tools(
        client.sdk("actor"), request, Handler(), max_iterations=1
    )
    assert final.choices[0].message.content == "An authored blue lake."
    assert usage["output_tokens"] == 8 and ctx.calls == 1
    assert payloads[-1]["max_completion_tokens"] == 512
    assert payloads[-1]["chat_template_kwargs"] == {"enable_thinking": False}
    client.close()


def test_adapted_build_without_decision_has_no_side_effects(tmp_path):
    from gluomem_bench.bench import arena_search_driver as search

    config = tmp_path / "build.json"
    config.write_text("{}")
    approval = tmp_path / "approval.json"
    approval.write_text("{}")
    with pytest.raises(ValueError, match="approval"):
        search.build_adapted(config, approval)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["approval.json", "build.json"]


def test_full_plan_revalidates_native_id_bridge():
    from gluomem_bench.bench import arena_search_driver as search

    root = ROOT.parent
    plan = search.dataset_plan(
        root / "memoryarena-data",
        root / "arena-search-groundtruth-coverage.json",
        root / "native-assets/Joanna690--websearch-embeddings/browsecomp_all_jsons.jsonl",
    )
    assert len(plan["tasks"]) == 221
    assert sum(t["sessions"] for t in plan["tasks"]) == 1641
    assert len({t["native_query_id"] for t in plan["tasks"]}) == 221


def test_encoder_default_is_not_identity_evidence(tmp_path):
    from gluomem_bench.bench import arena_search_driver as search

    with pytest.raises(ValueError, match="identity"):
        search.verify_index_admission(
            {"kind": "original", "dimension": 1536, "model": "text-embedding-ada-002"}
        )


def test_native_task_all_sessions_and_final_judge_only(tmp_path, monkeypatch):
    import socket

    import httpx

    from gluomem_bench.bench import arena_search_driver as search
    from gluomem_bench.bench.arena_driver import RawExperienceMemory
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )
    from gluomem_bench.bench.rack import RackProfile
    from gluomem_bench.bench.trajectory_suite import ArenaSession, ArenaTask

    monkeypatch.setattr(
        socket.socket, "connect", lambda *a, **kw: (_ for _ in ()).throw(OSError("offline"))
    )
    task = ArenaTask(
        -1,
        "progressive_search",
        (ArenaSession("Authored first lake?"), ArenaSession("Authored final lake?")),
        ("blue", "blue"),
    )
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps({"query_id": "fixture", "query": "Authored final lake?", "answer": "blue"})
        + "\n"
    )

    def wire(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "fixture"}]})
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 30})
        judge = (
            "correct_answer" in body["messages"][0]["content"]
            or "Correct Answer" in body["messages"][0]["content"]
            or body["max_completion_tokens"] == 8000
        )
        text = (
            "extracted_final_answer: blue\nreasoning: matches\ncorrect: yes\nconfidence: 90"
            if judge
            else "blue"
        )
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": text},
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
            },
        )

    outer = CampaignBudget(max_calls=8, max_seconds=120)
    job = Job("fixture", "fixture", tmp_path / "run", 8, 120)
    assert outer.reserve(job)
    ctx = JobContext(job, outer, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))
    client = search.SearchTransport.authored(
        RackProfile(base_url="http://fixture.invalid/v1", model="fixture"),
        receipt=tmp_path / "run" / "http.jsonl",
        budget=ctx,
        transport=httpx.MockTransport(wire),
    )
    native = search.load_native(ROOT)

    class Corpus:
        search_description = native.BaseSearcher.search_description
        get_document_description = native.BaseSearcher.get_document_description

    memory = RawExperienceMemory("full_context", budget=ctx)
    result = search.run_task(
        task,
        native=native,
        client=client,
        tools=native.SearchToolHandler(Corpus(), snippet_max_tokens=None, k=10),
        memory=memory,
        out=tmp_path / "run",
        native_query_id="fixture",
        gold_path=gold,
        qrels={"fixture": ["authored-doc"]},
        max_iterations=2,
    )
    assert result["completed_sessions"] == 2 and result["judgement"]["correct"] is True
    assert result["recall"] == 0.0 and ctx.calls == 3
    assert len(memory.entries) == 1 and "Predicted Answer: blue" in memory.entries[0]["content"]
    assert "correct_answer" not in memory.entries[0]["content"]
    assert search.verify_task(tmp_path / "run")["completed_sessions"] == 2
    tampered = search.read_json(tmp_path / "run/result.json")
    tampered["judgement"]["correct"] = False
    search.write_json(tmp_path / "run/result.json", tampered)
    with pytest.raises(ValueError, match="judge"):
        search.verify_task(tmp_path / "run")
    client.close()


def test_real_native_faiss_worker_authored_rows(tmp_path):
    import subprocess
    from types import SimpleNamespace

    import numpy as np

    from gluomem_bench.bench import arena_search_driver as search

    worker = ROOT.parent / "native_search_worker.py"
    python = ROOT.parent / "native-environment/Scripts/python.exe"
    ids = [f"original-00{i}" for i in range(8)]
    matrix = np.zeros((8, 768), dtype="float32")
    for i in range(8):
        matrix[i, i] = 1
    np.save(tmp_path / "matrix.npy", matrix)
    (tmp_path / "ids.json").write_text(json.dumps(ids))
    (tmp_path / "corpus.jsonl").write_text(
        "".join(
            json.dumps({"docid": docid, "text": f"Authored row {i}"}) + "\n"
            for i, docid in enumerate(ids)
        )
    )
    subprocess.run(
        [
            str(python),
            "-I",
            str(worker),
            "build",
            "--matrix",
            str(tmp_path / "matrix.npy"),
            "--ids",
            str(tmp_path / "ids.json"),
            "--out",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
    )

    class Encoder:
        def embed_query(self, q):
            return matrix[2]

    budget = SimpleNamespace(check=lambda: None, remaining=lambda: 60)
    config = {"python": str(python), "source": {"path": str(worker), "sha256": search.sha(worker)}}
    admission = {
        "dimension": 768,
        "index_path": str(tmp_path / "shard*.index"),
        "id_map_path": str(tmp_path / "shard*_id_map.json"),
        "corpus_path": str(tmp_path / "corpus.jsonl"),
    }
    native = search.load_native(ROOT)
    corpus = search.WorkerSearch(
        config,
        admission,
        native_root=ROOT,
        native=native,
        embedder=Encoder(),
        budget=budget,
        expected_rows=8,
    )
    try:
        hits = corpus.search("Authored query", 3)
        assert hits[0]["docid"] == "original-002" and hits[0]["score"] == 0
        assert corpus.get_document("original-002")["text"] == "Authored row 2"
        assert corpus.get_document("missing") is None
        assert (
            corpus.search("Authored query", 3, allowed_docids=["original-001"])[0]["docid"]
            == "original-001"
        )
    finally:
        corpus.close()


def test_freeze_full_matrix_still_refuses_missing_index(tmp_path):
    from gluomem_bench.bench import arena_search_driver as search
    from gluomem_bench.bench.rack import GenerationBudgets, RackProfile

    profile = RackProfile(
        base_url="http://fixture.invalid/v1",
        model="fixture",
        profile_version="role-budgets-v6",
        max_generation_calls=2048,
        max_run_seconds=1800,
        generation_budgets=GenerationBudgets(
            extraction=4096, consolidation=1024, reader=4096, judge=4096
        ),
    )
    config = {
        "native_root": str(ROOT),
        "data_root": str(ROOT.parent / "memoryarena-data"),
        "bridge_path": str(ROOT.parent / "arena-search-groundtruth-coverage.json"),
        "legacy_path": str(
            ROOT.parent / "native-assets/Joanna690--websearch-embeddings/browsecomp_all_jsons.jsonl"
        ),
        "profile": profile.model_dump(mode="json"),
        "output_root": str(tmp_path / "run"),
        "memory_embedding_url": "http://127.0.0.1:8766/v1",
        "bounds": {
            "calls_per_task_method": 2048,
            "seconds_per_task_method": 1800,
            "campaign_seconds": 100000,
            "max_iterations": 30,
            "actor_output_tokens": 15000,
            "judge_output_tokens": 8000,
            "top_k": 10,
            "snippet_tokens": 512,
            "feedback_in_memory": False,
        },
        "index_admission": {"kind": "original", "dimension": 1536},
    }
    spec = search.freeze_spec(config)
    assert spec["systems"] == list(search.SYSTEMS) and len(spec["tasks"]) == 221
    sp = tmp_path / "spec.json"
    search.write_json(sp, spec)
    ap = tmp_path / "approval.json"
    search.write_json(
        ap,
        {
            "decision": "approve",
            "purpose": "search-quality",
            "spec_sha256": search.sha(sp),
            "reviewer": "parent-test",
            "independent": True,
        },
    )
    with pytest.raises(ValueError, match="identity"):
        search.execute(sp, ap)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "fault,shared_failure",
    [
        ("timeout", False),
        ("length", False),
        ("identity", True),
        ("usage", True),
        ("bad_tool_json", False),
    ],
)
def test_native_fault_domain_retains_spend_no_retry(tmp_path, fault, shared_failure):
    import httpx

    from gluomem_bench.bench import arena_search_driver as search
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )
    from gluomem_bench.bench.rack import RackProfile

    outer = CampaignBudget(max_calls=8, max_seconds=120)
    job = Job("fixture", "fixture", tmp_path, 8, 120)
    assert outer.reserve(job)
    ctx = JobContext(job, outer, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))
    dispatches = []

    def wire(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "fixture"}]})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 20})
        dispatches.append(1)
        if fault == "timeout":
            raise httpx.ReadTimeout("authored isolated timeout", request=request)
        usage = (
            {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23}
            if fault != "usage"
            else {}
        )
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "wrong" if fault == "identity" else "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": (
                            "tool_calls"
                            if fault == "bad_tool_json"
                            else "length"
                            if fault == "length"
                            else "stop"
                        ),
                        "message": {
                            "role": "assistant",
                            "content": "authored",
                            "tool_calls": [
                                {
                                    "id": "authored-call",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": "not-json"},
                                }
                            ]
                            if fault == "bad_tool_json"
                            else None,
                        },
                    }
                ],
                "usage": usage,
            },
        )

    client = search.SearchTransport.authored(
        RackProfile(base_url="http://fixture.invalid/v1", model="fixture"),
        receipt=tmp_path / "wire.jsonl",
        budget=ctx,
        transport=httpx.MockTransport(wire),
    )
    for _ in range(2):
        with pytest.raises(search.shared.NativeAbort):
            client.sdk("actor").chat.completions.create(
                model="fixture",
                max_completion_tokens=512,
                messages=[{"role": "user", "content": "Authored query"}],
            )
    assert ctx.calls == 1 and len(dispatches) == 1 and ctx.failed is True
    assert outer.failed is shared_failure
    client.close()


def test_query_document_actual_wire_roles_and_shapes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import httpx

    from gluomem_bench.bench import arena_search_driver as search

    requests = []
    real = httpx.Client

    def wire(request):
        body = json.loads(request.content)
        requests.append(body)
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "recipe": search.EmbeddingRecipe().as_dict(),
                "role": body["input_type"],
                "model": body["model"],
                "token_counts": [10] * len(body["input"]),
                "usage": {"prompt_tokens": 10 * len(body["input"])},
                "data": [
                    {"index": i, "embedding": [1.0] + [0.0] * 767}
                    for i in range(len(body["input"]))
                ],
            },
        )

    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(wire), **kw)
    )
    budget = SimpleNamespace(check=lambda: None, remaining=lambda: 60)
    encoder = search.SearchEmbedding(
        "http://127.0.0.1:8766/v1", budget=budget, receipt=tmp_path / "embedding-wire.jsonl"
    )
    encoder.embed_batch(["Authored a\nb", "Authored c"])
    encoder.embed_query("Authored a\nb")
    assert [r["input_type"] for r in requests] == ["document", "query"]
    assert [len(r["input"]) for r in requests] == [2, 1] and requests[0]["input"][
        0
    ] == "Authored a\nb"
    receipts = [
        json.loads(line) for line in (tmp_path / "embedding-wire.jsonl").read_bytes().splitlines()
    ]
    assert [r["shape"] for r in receipts] == [[2, 768], [1, 768]] and all(
        r["complete"] for r in receipts
    )


def test_cost_accounting_preserves_failed_unknown_usage(tmp_path):
    from gluomem_bench.bench import arena_search_driver as search

    records = [
        {
            "ordinal": 1,
            "role": "actor",
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            "complete": True,
        },
        {"ordinal": 2, "role": "judge", "usage": None, "complete": False},
    ]
    (tmp_path / "generation-receipts.jsonl").write_bytes(
        b"".join(search.shared._json(r).replace(b"\n", b"") + b"\n" for r in records)
    )
    result = search.cost_accounting(tmp_path)
    assert result["attempts"] == 2 and result["unknown_usage_attempts"] == 1
    assert result["known_prompt_tokens"] == 12 and result["known_completion_tokens"] == 3
    assert result["currency_cost"] is None


def test_native_projection_uses_official_gold_not_legacy_generated_answers():
    from gluomem_bench.bench import arena_search_driver as search

    root = ROOT.parent
    result = search.verify_gold_projection(
        {
            "path": str(root / "search-evaluator-projection-v1.jsonl"),
            "sha256": search.sha(root / "search-evaluator-projection-v1.jsonl"),
        },
        {
            "path": str(root / "search-evaluator-projection-v1.json"),
            "sha256": search.sha(root / "search-evaluator-projection-v1.json"),
        },
    )
    assert result["rows"] == 830
    assert len(result["query_ids"]) == 830


def test_native_snippets_use_actual_frozen_qwen_tokenizer(monkeypatch):
    import socket
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    from gluomem_bench.bench import arena_search_driver as search

    monkeypatch.setattr(
        socket.socket, "connect", lambda *a, **kw: (_ for _ in ()).throw(OSError("offline"))
    )
    binding = search.read_json(ROOT.parent / "search-native-snippet-tokenizer-v1.json")
    tokenizer = AutoTokenizer.from_pretrained(binding["path"], local_files_only=True)
    native = search.load_native(ROOT)
    native.AutoTokenizer = SimpleNamespace(from_pretrained=lambda name: tokenizer)
    text = "Authored blue lake observation. " * 1000

    class Corpus:
        def search(self, *a, **kw):
            return [{"docid": "original-001", "score": 0.0, "text": text}]

        def get_document(self, docid):
            return {"docid": docid, "text": text}

    handler = native.SearchToolHandler(Corpus(), snippet_max_tokens=512, k=10)
    snippet = json.loads(handler.execute_tool("search", {"query": "Authored lake"}))[0]["snippet"]
    assert len(tokenizer.encode(snippet, add_special_tokens=False)) == 512
    assert (
        json.loads(handler.execute_tool("get_document", {"docid": "original-001"}))["text"] == text
    )
