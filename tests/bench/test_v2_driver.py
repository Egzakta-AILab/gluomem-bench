"""Authored contract tests; no benchmark QA generations."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest


def driver():
    return importlib.import_module("gluomem_bench.bench.v2_driver")


def test_cache_key_accepts_actual_remote_encoder_fingerprint(tmp_path):
    from types import SimpleNamespace as NS
    from gluomem_bench.bench.embedding_runtime import RemoteEmbedder
    v = driver()
    inner = RemoteEmbedder('http://unused.invalid')
    inner.embed_batch = lambda texts: [[1.0, 0.0] for _ in texts]
    cache = v.ImmutableVectorCache(inner, root=tmp_path, budget=NS(check=lambda: None))
    assert cache.embed_batch(['same\nbytes']) == [[1.0, 0.0]]
    json.dumps(cache.receipt)


def test_prepare_discovers_actual_native_factory_source_without_sdk_imports():
    import sys
    from gluomem_bench.bench import v2_memory_factories as m
    v = driver()
    # Discovery must not import native Mem0 (its import-time home is operator-owned).
    before = 'mem0' in sys.modules
    sources = v.native_factory_sources({'mem0': {'source_path': m.__file__}})
    assert 'mem0/memory/main.py' in sources
    assert Path(sources['mem0/memory/main.py']).is_file()
    assert ('mem0' in sys.modules) == before


def test_cached_raw_retrieval_uses_only_allowed_corpus_and_local_bm25(tmp_path):
    from types import SimpleNamespace as NS
    v = driver()
    class Encoder:
        fingerprint = 'authored'
        def embed_batch(self, texts): return [[1.0, float(len(t))] for t in texts]
        def embed_query(self, text): return [1.0, 1.0]
    budget = NS(check=lambda: None)
    cache = v.ImmutableVectorCache(Encoder(), root=tmp_path, budget=budget)
    a = [{'id': 'a', 'text': 'cobalt cobalt', 'offset': 0}, {'id': 'b', 'text': 'willow', 'offset': 0}]
    b = [{'id': 'b', 'text': 'willow', 'offset': 0}]
    broad = v.RawTextMemory('hybrid', a, {'complete': True, 'documents_sha256': v.digest(a)}, budget=budget, embedder=cache)
    narrow = v.RawTextMemory('hybrid', b, {'complete': True, 'documents_sha256': v.digest(b)}, budget=budget, embedder=cache)
    assert set(narrow.backend.chunks) == {'b'}
    assert narrow.backend._bm25('cobalt') == []
    assert narrow.query('cobalt') == [{'type': 'text', 'value': 'willow'}]
    assert cache.receipt['hits'] == 1
    broad.close(); narrow.close()


def test_campaign_budget_keeps_generous_group_bounds_distinct_from_outer():
    v = driver()
    from gluomem_bench.bench.rack_trace import WorkloadBudget
    outer = v.campaign_budget(max_calls=40000, max_seconds=50000)
    child = WorkloadBudget(max_calls=16384, max_seconds=21600,
                           policy='role-budgets-v6', outer_budget=outer)
    child.begin_call()
    assert child.calls == outer.calls == 1
    assert child.max_calls == 16384 and outer.max_calls == 40000
    with pytest.raises(ValueError):
        v.campaign_budget(max_calls=True, max_seconds=50000)


def test_concrete_factory_spec_binds_options_and_rejects_callable_substitution():
    from gluomem_bench.bench import v2_memory_factories as m
    v = driver()
    entry = {'source_path': m.__file__, 'sha256': v.file_hash(m.__file__),
             'module': m.__name__, 'callable': 'cognee_factory',
             'options': {'python_executable': 'isolated-python', 'chunk_size': 512}}
    selected = v._load_document_factories({'config': {'factory_specs': {'cognee': entry}}})
    assert selected['cognee'].keywords['python_executable'] == 'isolated-python'
    entry['callable'] = 'mem0_factory'
    with pytest.raises(ValueError, match='concrete native factory'):
        v._load_document_factories({'config': {'factory_specs': {'cognee': entry}}})


def test_generation_reconciliation_counts_native_wire_once_not_wrapper(tmp_path):
    v = driver()
    response = {'usage': {'prompt_tokens': 32, 'completion_tokens': 4}}
    (tmp_path/'calls.jsonl').write_text(json.dumps({'url': 'https://example.org/v1/chat/completions', 'response': response})+'\n')
    # Mem0's native trace duplicates the exact same direct request: not another call.
    (tmp_path/'mem0-native.jsonl').write_text(json.dumps({'task': 'event_claims', 'response': response})+'\n')
    native = tmp_path/'native-cognee'; native.mkdir()
    import base64
    body = base64.b64encode(json.dumps(response).encode()).decode()
    (native/'generation.jsonl').write_text('\n'.join(json.dumps(row) for row in [
        {'kind': 'dispatch-intent', 'calls': 1},
        {'kind': '/chat/completions', 'response_b64': body},
        {'kind': 'generation', 'response_b64': body}])+'\n')
    assert v.generation_usage(tmp_path, consumed_calls=2) == {
        'generation_calls': 2, 'prompt_tokens': 64, 'completion_tokens': 8}
    with pytest.raises(ValueError, match='accounting mismatch'):
        v.generation_usage(tmp_path, consumed_calls=3)


def test_immutable_vector_cache_reuses_exact_single_shape_and_rejects_corruption(tmp_path):
    from types import SimpleNamespace as NS
    v = driver()
    calls = []
    class Encoder:
        fingerprint = 'exact-qualified-encoder'
        def embed_batch(self, texts):
            calls.append(('document', list(texts)))
            return [[float(len(text)), 1.0] for text in texts]
        def embed_query(self, text):
            calls.append(('query', [text]))
            return [float(len(text)), 2.0]
    budget = NS(check=lambda: None)
    a = v.ImmutableVectorCache(Encoder(), root=tmp_path, budget=budget)
    assert a.embed_batch(['A\nB', 'C']) == [[3.0, 1.0], [1.0, 1.0]]
    assert a.embed_batch(['C', 'A\nB']) == [[1.0, 1.0], [3.0, 1.0]]
    assert a.embed_query('C') == [1.0, 2.0]
    assert calls == [('document', ['A\nB']), ('document', ['C']), ('query', ['C'])]
    assert a.receipt['hits'] == 2 and a.receipt['misses'] == 3
    assert a.receipt['batch_shape'] == 'singleton; exact text/role/encoder'
    entry = next(tmp_path.glob('*.json'))
    saved = json.loads(entry.read_bytes())
    saved['vector'] = [99.0, 99.0]
    entry.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='cache integrity'):
        a.embed_batch(['A\nB', 'C'])


def test_lossless_document_events_preserve_all_states_without_answer_ids_or_epoch(tmp_path):
    v = driver()
    pins, trajectory = dataset(tmp_path)
    data = v.SourceDataset(tmp_path, pins=pins)
    group = data.groups()[0]
    docs, coverage = v.source_documents(data, group, chunk_chars=127)
    assert len(docs) > 1
    rebuilt = json.loads("".join(d["text"] for d in docs))
    assert rebuilt["trajectory_ordinal"] == 0
    assert [s["action"] for s in rebuilt["states"]] == ["incoming0", "incoming1", "incoming2"]
    assert len(rebuilt["states"]) == len(trajectory["states"])
    assert "source-secret-id" not in json.dumps(docs)
    assert "GOLD_ONLY" not in json.dumps(docs)
    assert "screenshot" not in rebuilt["states"][0]
    assert coverage["history_screenshot_refs"] == [s["screenshot"] for s in trajectory["states"]]
    assert coverage["states"] == 3
    assert coverage["complete"] is True
    events = v.document_events(docs)
    assert "".join(e.content for e in events) == "".join(d["text"] for d in docs)
    assert all(
        e.source_type == "document" and e.kind is None and not e.messages and e.event_time is None
        for e in events
    )
    assert len({e.event_id for e in events}) == len(docs)


NATIVE_ROOT = Path(
    'tests/fixtures/native/operator-supplied'
)


def test_native_ast_exact_prompt_image_and_scoring_unknown_still_spends_judge(tmp_path):
    v = driver()
    native = v.NativeInterface(NATIVE_ROOT)
    image = tmp_path / "query.png"
    image.write_bytes(b"authored PNG")
    messages = native.messages("web", "Where?", str(image), [{"type": "text", "value": "Memory"}])
    assert messages[1]["content"][-1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert messages[1]["content"][0]["text"] == "### Memory context:\n"
    assert messages[1]["content"][-2]["text"] == "\n\n### Question to answer:\nWhere?"
    assert "GOLD_ONLY" not in json.dumps(messages)
    assert native.score(
        {"question": "Where?", "answer": "blue menu", "eval_function": "norm_phrase_set_match"},
        r"Answer \\boxed{blue menu}",
    )["correct"]
    calls = []

    class Client:
        profile = type(
            "Profile",
            (),
            {
                "model": "authored",
                "base_url": "https://example.org/v1",
                "timeout_seconds": 30,
                "output_budget": lambda self, task: 128,
            },
        )()

        def generate_messages(self, task, messages, **kwargs):
            calls.append((task, messages))
            return {
                "text": '{"label": 1, "reason": "authored"}',
                "prompt_tokens": 100,
                "completion_tokens": 10,
            }

        def _latch(self, exc):
            pass

    result = native.score(
        {"question": "Where?", "answer": "GOLD_ONLY", "eval_function": "llm_gotchas_checker"},
        r"\\boxed{UNKNOWN}",
        client=Client(),
    )
    assert len(calls) == 1 and calls[0][0] == "rubric"
    assert "GOLD_ONLY" in json.dumps(calls[0][1])
    assert result["correct"] is False and result["is_unknown"] is True
    assert result["judge_model_substitution"] == "authored (official gpt-5.2)"
    assert native.receipt["eager_upstream_client_imports"] is False


def test_native_whole_item_prefix_and_complete_reader_pack_includes_image_and_output(tmp_path):
    v = driver()

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            self.parts = messages[0]["content"]
            return json.dumps(self.parts)

        def __call__(self, *, text, images, **kwargs):
            assert len(images or []) == sum(p["type"] == "image" for p in self.parts)
            n = sum(len(p.get("text", "")) if p["type"] == "text" else 100 for p in self.parts)
            return {"input_ids": type("Tensor", (), {"shape": (1, n)})()}

    native = v.NativeInterface(NATIVE_ROOT, processor=Processor())

    class Client:
        profile = type(
            "Profile",
            (),
            {
                "schema_reserve_tokens": 20,
                "context_window": 5000,
                "output_budget": lambda self, task: 100,
            },
        )()

        def prompt_token_count(self, messages):
            return 150 + sum(
                len(p.get("text", "")) if p["type"] == "text" else 100
                for p in messages[1]["content"]
            )

        def admit_context(self, tokens, **kwargs):
            if tokens + kwargs["output_budget"] + 20 > 5000:
                raise ValueError("overflow")

    question = {"domain": "web", "question": "Where?", "image": None}
    items = [{"type": "text", "value": "A" * 2000}, {"type": "text", "value": "B" * 2000}]
    packed = v.reader_pack(native, Client(), question, items)
    assert packed["items"] == items[:1]
    assert packed["complete_prompt_tokens"] <= 4096
    assert packed["native_original_tokens"] == 4000
    assert packed["native_kept_tokens"] == 2000
    assert packed["output_reserve"] == 100
    assert packed["items"][0]["value"] == "A" * 2000
    assert packed["operating_point"] == "4096-complete-reader-prompt-tokens; whole-item-prefix"
    huge = [{"type": "text", "value": "C" * 6000}]
    with pytest.raises(v.ContextNotApplicable):
        v.reader_pack(native, Client(), question, huge, full_fit_only=True)


def test_authored_native_vision_qualification_uses_exact_transport_and_rejects_usage_mismatch(
    tmp_path,
):
    import httpx

    v = driver()
    from gluomem_bench.bench.rack import RackProfile

    profile = RackProfile(base_url="https://example.org/v1", model="authored")
    seen = []

    def handler(request):
        payload = json.loads(request.content) if request.content else {}
        seen.append((request.url.path, payload))
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "authored"}]})
        if request.url.path == "/tokenize":
            n = 200 if "image_url" in json.dumps(payload) else 100
            return httpx.Response(200, json={"count": n})
        return httpx.Response(
            200,
            json={
                "model": "authored",
                "choices": [{"finish_reason": "stop", "message": {"content": r"\\boxed{red}"}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 5},
            },
        )

    receipt = v.qualify_vision(
        profile, NATIVE_ROOT, tmp_path / "ok", transport=httpx.MockTransport(handler)
    )
    assert receipt["qualified"] is True and receipt["generation_calls"] <= 4
    assert receipt["benchmark_questions_used"] == 0
    assert receipt["tokenizer_usage_parity"] is True
    assert receipt["memory_context_actual_served_processor"] is True
    assert any(p.get("add_generation_prompt") is False for path, p in seen if path == "/tokenize")
    assert all("chat_template_kwargs" in p for path, p in seen if path != "/v1/models")

    def bad(request):
        response = handler(request)
        if request.url.path.endswith("completions"):
            value = response.json()
            value["usage"]["prompt_tokens"] = 1
            return httpx.Response(200, json=value)
        return response

    receipt = v.qualify_vision(
        profile, NATIVE_ROOT, tmp_path / "bad", transport=httpx.MockTransport(bad)
    )
    assert receipt["qualified"] is False
    assert "usage" in receipt["blocker"]
    assert receipt["generation_calls"] == 1
    assert (tmp_path / "bad" / "calls.jsonl").is_file()


def test_raw_text_factories_index_complete_source_and_retrieve_without_gold(tmp_path):
    v = driver()
    pins, _ = dataset(tmp_path)
    data = v.SourceDataset(tmp_path, pins=pins)
    group = data.groups()[0]
    docs, proof = v.source_documents(data, group, chunk_chars=127)

    class Budget:
        def check(self):
            pass

    class Encoder:
        def embed_batch(self, texts):
            return [[float("menu" in t), 1.0] for t in texts]

        def embed_query(self, text):
            return [1.0, 1.0]

    for method in ("no_memory", "bm25", "dense", "hybrid", "full_context"):
        memory = v.RawTextMemory(method, docs, proof, budget=Budget(), embedder=Encoder())
        assert memory.coverage["source_documents"] == len(docs)
        assert memory.coverage["complete"] is True
        result = memory.query("blue menu", query_image="kept for reader")
        assert all(item["type"] == "text" for item in result)
        if method == "no_memory":
            assert result == []
        elif method == "full_context":
            assert "".join(i["value"] for i in result) == "".join(d["text"] for d in docs)
        else:
            assert result
        assert "GOLD_ONLY" not in json.dumps(result)
        memory.close()


def test_production_gluomem_factory_requires_whole_ingest_budget_and_active_dspy(
    tmp_path, monkeypatch
):
    v = driver()
    from gluomem_bench.bench.rack import RackProfile, controlled_config
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    pins, _ = dataset(tmp_path)
    data = v.SourceDataset(tmp_path, pins=pins)
    group = data.groups()[0]
    docs, proof = v.source_documents(data, group, chunk_chars=127)
    profile = RackProfile(base_url="https://example.org/v1", model="authored")
    cfg = controlled_config(
        profile, data=tmp_path / "questions.jsonl", sha256=pins["questions.jsonl"]
    )
    client = type("Client", (), {"profile": profile})()
    with pytest.raises(v.ResourceNotAdmitted, match="whole-history"):
        v.GluoDocumentMemory(
            cfg=cfg,
            group=group,
            docs=docs,
            proof=proof,
            out=tmp_path / "out",
            client=client,
            budget=WorkloadBudget(max_calls=2),
            embedder=None,
            qa_reserve=1,
        )
    outer = WorkloadBudget(max_calls=2)
    child = WorkloadBudget(max_calls=128, outer_budget=outer)
    with pytest.raises(v.ResourceNotAdmitted, match='whole-history'):
        v.GluoDocumentMemory(cfg=cfg, group=group, docs=docs, proof=proof,
            out=tmp_path / 'outer-refused', client=client, budget=child,
            embedder=None, qa_reserve=1)
    captured = []
    from gluomem.llm.modules import DspyExtractors

    import gluomem_bench.bench.beam_ingest as bi

    def fail_ingest(cfg, adapter, stores, extractors, embedder, conv, completion_guard, **kwargs):
        assert isinstance(extractors, DspyExtractors)
        assert [e.content for e in kwargs["source_events"]] == [d["text"] for d in docs]
        assert kwargs["coverage_protocol"] == v.PROTOCOL
        assert completion_guard.__self__ is budget
        captured.append(True)
        raise RuntimeError("authored whole-ingest failure")

    monkeypatch.setattr(bi, "_ingest_conversation", fail_ingest)
    budget = WorkloadBudget(max_calls=128)
    with pytest.raises(RuntimeError, match="whole-ingest failure"):
        v.GluoDocumentMemory(
            cfg=cfg,
            group=group,
            docs=docs,
            proof=proof,
            out=tmp_path / "out",
            client=client,
            budget=budget,
            embedder=None,
            qa_reserve=1,
        )
    assert captured == [True]
    assert not (tmp_path / "out" / "coverage.json").exists()


def test_group_vertical_slice_native_no_memory_and_ingest_failure_never_becomes_qa_zero(tmp_path):
    import httpx

    v = driver()
    from gluomem_bench.bench.rack import RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    pins, _ = dataset(tmp_path / "data", count=2)
    data = v.SourceDataset(tmp_path / "data", pins=pins, expected_questions=2, expected_images=2)
    native = v.NativeInterface(NATIVE_ROOT)
    profile = RackProfile(base_url="https://example.org/v1", model="authored")
    budget = WorkloadBudget(max_calls=16)

    def handler(request):
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 300})
        return httpx.Response(
            200,
            json={
                "model": "authored",
                "choices": [
                    {"finish_reason": "stop", "message": {"content": r"\\boxed{GOLD_ONLY}"}}
                ],
                "usage": {"prompt_tokens": 300, "completion_tokens": 10},
            },
        )

    out = tmp_path / "worker"
    out.mkdir()
    client = RackClient(
        profile, receipt=out / "calls.jsonl", budget=budget, transport=httpx.MockTransport(handler)
    )
    group = data.groups()[0]
    result = v.run_group(
        data=data,
        group=group,
        method="no_memory",
        native=native,
        client=client,
        budget=budget,
        out=out,
        authored=True,
    )
    assert result["status"] == "complete"
    assert [r["question_id"] for r in result["rows"]] == list(group.question_ids)
    assert all(r["status"] == "complete" and r["score"]["correct"] is True for r in result["rows"])
    assert budget.calls == 2
    assert result["usage"] == {"generation_calls": 2, "prompt_tokens": 600, "completion_tokens": 20}
    assert all("image_url" in json.dumps(r["pack"]["messages"]) for r in result["rows"])
    assert all("GOLD_ONLY" not in json.dumps(r["pack"]["messages"]) for r in result["rows"])
    client.close()
    out2 = tmp_path / "failed"
    out2.mkdir()

    def factory(**kwargs):
        raise RuntimeError("whole-source ingest failed")

    result = v.run_group(
        data=data,
        group=group,
        method="mem0",
        native=native,
        client=client,
        budget=WorkloadBudget(max_calls=16),
        out=out2,
        authored=True,
        document_factories={"mem0": factory},
    )
    assert result["status"] == "failed"
    assert all(r["status"] == "unrun" and "score" not in r for r in result["rows"])
    assert "ingest failed" in result["error"]


def test_full_matrix_and_verifier_reject_missing_occurrences_and_fake_failure_scores(tmp_path):
    v = driver()
    pins, _ = dataset(tmp_path / "data")
    data = v.SourceDataset(tmp_path / "data", pins=pins)
    matrix = v.scope_matrix(data)
    assert len(matrix) == 902 * len(v.METHODS)
    report = v.verify_matrix(matrix, data)
    assert report["expected_questions_per_tier"] == {"small": 451, "medium": 451}
    assert set(report["methods"]) == set(v.METHODS)
    assert all(
        m["small"]["unrun"] == 451 and m["medium"]["unrun"] == 451
        for m in report["methods"].values()
    )
    with pytest.raises(ValueError, match="occurrence"):
        v.verify_matrix(matrix[:-1], data)
    matrix[0]["score"] = {"correct": False}
    with pytest.raises(ValueError, match="unrun"):
        v.verify_matrix(matrix, data)


def test_cli_example_is_executable_schema_no_private_defaults_and_run_needs_approval(tmp_path):
    v = driver()
    config = v.example_config()
    assert config["scope"] == {
        "tiers": ["small", "medium"],
        "questions_per_tier": 451,
        "query_images": 29,
    }
    assert config["methods"] == list(v.METHODS)
    assert config["reader_prompt_tokens"] == 4096
    assert config["document_chunk_chars"] == 65536
    assert config["raw_chunk_chars"] == 1000
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(config))
    approval = tmp_path / "approval.json"
    approval.write_text("{}")
    with pytest.raises(ValueError, match="approval"):
        v.execute(path, approval, tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


def test_no_memory_declares_complete_ignored_source_without_serializing_500_haystacks(
    tmp_path, monkeypatch
):
    import httpx

    v = driver()
    from gluomem_bench.bench.rack import RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    pins, _ = dataset(tmp_path / "data", count=2)
    data = v.SourceDataset(tmp_path / "data", pins=pins, expected_questions=2, expected_images=2)

    def forbid(*args, **kwargs):
        raise AssertionError("NoRetrieval must not project/index ignored source")

    monkeypatch.setattr(v, "source_documents", forbid)
    profile = RackProfile(base_url="https://example.org/v1", model="authored")
    budget = WorkloadBudget(max_calls=4)
    out = tmp_path / "run"
    out.mkdir()

    def handler(request):
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 200})
        return httpx.Response(
            200,
            json={
                "model": "authored",
                "choices": [{"finish_reason": "stop", "message": {"content": r"\\boxed{UNKNOWN}"}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 2},
            },
        )

    client = RackClient(
        profile, receipt=out / "calls.jsonl", budget=budget, transport=httpx.MockTransport(handler)
    )
    result = v.run_group(
        data=data,
        group=data.groups()[0],
        method="no_memory",
        native=v.NativeInterface(NATIVE_ROOT),
        client=client,
        budget=budget,
        out=out,
        authored=True,
    )
    client.close()
    assert result["status"] == "complete"
    assert result["coverage"]["trajectories"] == 1
    assert result["coverage"]["states"] == 3
    assert (
        result["coverage"]["processing"] == "native NoRetrieval ignores complete source; no index"
    )
    assert result["coverage"]["source_archive_sha256"] == pins["trajectories.jsonl"]


@pytest.mark.parametrize("bad_optional", [False, True])
def test_frozen_authored_full_cli_path_archives_source_and_verifies_all_902_no_memory_rows(
    tmp_path, monkeypatch, bad_optional
):
    import httpx

    import gluomem_bench.bench.rack as rack

    v = driver()
    pins, _ = dataset(tmp_path / "data")
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps(pins))
    profile = rack.RackProfile(
        base_url="https://example.org/v1",
        model="authored",
        profile_version="role-budgets-v6",
        max_generation_calls=16384,
        max_run_seconds=21600,
        generation_budgets={
            "extraction": 1024,
            "consolidation": 1024,
            "reader": 4096,
            "judge": 4096,
        },
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(profile.model_dump_json())

    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "authored", "max_model_len": 131072}]})
        if request.url.path == "/tokenize":
            p = json.loads(request.content)
            return httpx.Response(200, json={"count": 300 if "image_url" in json.dumps(p) else 100})
        return httpx.Response(
            200,
            json={
                "model": "authored",
                "choices": [
                    {"finish_reason": "stop", "message": {"content": r"\\boxed{GOLD_ONLY}"}}
                ],
                "usage": {
                    "prompt_tokens": 300 if "image_url" in request.content.decode() else 100,
                    "completion_tokens": 10,
                },
            },
        )

    # Synthetic gate receipt, never a deployment or quality assertion.
    q = tmp_path / "qualification.json"
    q.write_text(
        json.dumps(
            {
                "qualified": True,
                "model": "authored",
                "profile_sha256": v.digest(profile.model_dump(mode="json")),
            }
        )
    )
    config = v.example_config()
    config.update(
        dataset_root=str(tmp_path / "data"),
        pins_path=str(pins_path),
        native_root=str(NATIVE_ROOT),
        profile_path=str(profile_path),
        qualification_path=str(q),
        methods=["no_memory"],
    )
    if bad_optional:
        config["methods"] = ["mem0", "no_memory"]
        config["factory_specs"] = {
            "mem0": {
                "module": "missing_native_sdk_authored_test",
                "callable": "build",
                "source_path": v.__file__,
                "sha256": v.file_hash(v.__file__),
            }
        }
    prepared = v.prepare(config, tmp_path / "frozen", authored=True)
    assert Path(prepared["source_archive"]["path"]).is_file()
    import zipfile
    with zipfile.ZipFile(prepared['source_archive']['path']) as archive:
        assert 'core/gluomem/storage/schema.sql' in archive.namelist()
        assert 'core/gluomem/migrations/script.py.mako' in archive.namelist()
    assert prepared["counts"]["occurrences"] == 902
    original = rack.RackClient
    monkeypatch.setattr(
        rack,
        "RackClient",
        lambda *args, **kwargs: original(*args, transport=httpx.MockTransport(handler), **kwargs),
    )
    result = v._execute_approved(tmp_path / "frozen/spec.json", tmp_path / "run", authored=True)
    assert result["status"] == ("incomplete" if bad_optional else "complete")
    assert result["summary"]["methods"]["no_memory"]["small"]["complete"] == 451
    assert result["summary"]["methods"]["no_memory"]["medium"]["complete"] == 451
    assert result["usage"]["generation_calls"] == 902
    verified = v.verify_root(tmp_path / "run")
    assert verified["verified"] is True
    assert verified["expected_questions_per_tier"] == {"small": 451, "medium": 451}
    tampered = json.loads((tmp_path / "run/root-status.json").read_bytes())
    completed = next(r for r in tampered["matrix"] if r["status"] == "complete")
    completed["pack"]["messages"][1]["content"] = [
        p for p in completed["pack"]["messages"][1]["content"] if p["type"] != "image_url"
    ]
    (tmp_path / "run/root-status.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="image|leaf"):
        v.verify_root(tmp_path / "run")
    (tmp_path / "run/root-status.json").write_text(json.dumps(result))
    # A self-consistent rewritten leaf must still fail wire/input-image checks.
    rewritten = json.loads((tmp_path / 'run/root-status.json').read_bytes())
    image_row = next(r for r in rewritten['matrix'] if r['status'] == 'complete')
    image_row['pack']['messages'][1]['content'] = [
        p for p in image_row['pack']['messages'][1]['content']
                                                if p['type'] != 'image_url']
    job = next(j for j in rewritten['jobs']
               if j['method'] == image_row['method'] and j['group_id'] == image_row['group_id'])
    leaf_path = Path(job['path']) / 'root-status.json'
    original_leaf = leaf_path.read_bytes()
    leaf = json.loads(original_leaf)
    next(r for r in leaf['rows']
         if r['question_id'] == image_row['question_id'])['pack'] = image_row['pack']
    leaf_path.write_text(json.dumps(leaf))
    job['root_sha256'] = v.file_hash(leaf_path)
    (tmp_path / 'run/root-status.json').write_text(json.dumps(rewritten))
    with pytest.raises(ValueError, match='image|wire|hash'):
        v.verify_root(tmp_path / 'run')
    leaf_path.write_bytes(original_leaf)
    (tmp_path / 'run/root-status.json').write_text(json.dumps(result))
    root = json.loads((tmp_path / 'run/root-status.json').read_bytes())
    root['matrix'].pop()
    (tmp_path / "run/root-status.json").write_text(json.dumps(root))
    with pytest.raises(ValueError, match="occurrence|hash"):
        v.verify_root(tmp_path / "run")


def authored_execution(tmp_path, monkeypatch, fault=None, *, semantic=False):
    """Real frozen entry point and native scorer, authored HTTP only."""
    import httpx
    import socket
    from gluomem_bench.bench import rack

    v = driver()
    pins, _ = dataset(tmp_path / 'data')
    if semantic:
        questions = list(map(json.loads, (tmp_path / 'data/questions.jsonl').read_bytes().splitlines()))
        questions[1]['eval_function'] = 'llm_gotchas_checker'
        pins['questions.jsonl'] = put(tmp_path / 'data', 'questions.jsonl', questions, lines=True)
    put(tmp_path, 'pins.json', pins)
    profile = rack.RackProfile(base_url='https://example.org/v1', model='authored',
                              profile_version='role-budgets-v6', max_generation_calls=16384,
                              max_run_seconds=21600, generation_budgets={
                                  'extraction': 1024, 'consolidation': 1024,
                                  'reader': 4096, 'judge': 4096})
    put(tmp_path, 'profile.json', profile.model_dump(mode='json'))
    put(tmp_path, 'qualification.json', {'qualified': True, 'model': 'authored',
                                       'profile_sha256': v.digest(profile.model_dump(mode='json'))})
    config = v.example_config()
    config.update(dataset_root=str(tmp_path / 'data'), pins_path=str(tmp_path / 'pins.json'),
                  native_root=str(NATIVE_ROOT), profile_path=str(tmp_path / 'profile.json'),
                  qualification_path=str(tmp_path / 'qualification.json'), methods=['no_memory'],
                  embedding_url='https://example.org/v1')
    v.prepare(config, tmp_path / 'prepared', authored=True)
    clients, requests = [], []
    def deny(*args, **kwargs):
        raise AssertionError('real socket forbidden in authored execution')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)
    from gluomem_bench.bench.embedding_runtime import RemoteEmbedder
    monkeypatch.setattr(RemoteEmbedder, '_encode', deny)
    monkeypatch.setattr(v, 'source_documents', deny)
    def handler(request):
        requests.append(request)
        replacement = fault(request, requests) if fault else None
        if replacement is not None:
            return replacement
        if request.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [{'id': 'authored', 'max_model_len': 131072}]})
        if request.url.path.endswith('/tokenize'):
            return httpx.Response(200, json={'count': 300})
        payload = json.loads(request.content)
        content = (r'\boxed{GOLD_ONLY}' if isinstance(payload['messages'][1]['content'], list)
                   else '{"label": 1, "reason": "authored"}')
        return httpx.Response(200, json={'model': 'authored', 'choices': [
            {'finish_reason': 'stop', 'message': {'content': content}}],
            'usage': {'prompt_tokens': 300, 'completion_tokens': 4}})
    original = rack.RackClient
    def client(*args, **kwargs):
        result = original(*args, transport=httpx.MockTransport(handler), **kwargs)
        clients.append(result)
        return result
    monkeypatch.setattr(rack, 'RackClient', client)
    return tmp_path / 'prepared/spec.json', tmp_path / 'run', clients, requests


@pytest.mark.parametrize('failure', ['length', 'schema', 'invalid_judge'])
def test_v2_local_failure_latches_only_failed_group_then_dispatches_healthy_group(
    tmp_path, monkeypatch, failure
):
    import httpx
    from gluomem_bench.bench.rack import FatalRackError
    v = driver()
    generations = []
    def fault(request, requests):
        if not request.url.path.endswith('/chat/completions'):
            return
        generations.append(request)
        target = 3 if failure == 'invalid_judge' else 2
        if len(generations) != target:
            return
        response = {'model': 'authored', 'choices': [
            {'finish_reason': 'length' if failure == 'length' else 'stop',
             'message': {'content': 'invalid verdict'}}],
            'usage': {'prompt_tokens': 300, 'completion_tokens': 4}}
        if failure == 'schema':
            response['choices'] = []
        return httpx.Response(200, json=response)
    spec, out, clients, _ = authored_execution(
        tmp_path, monkeypatch, fault, semantic=failure == 'invalid_judge')
    result = v._execute_approved(spec, out, authored=True)
    assert len(clients) == 2, 'a local failure must not cancel unrelated groups'
    assert [j['status'] for j in result['jobs']] == ['failed', 'complete']
    assert clients[0].budget.failed is True
    assert clients[1].budget.failed is False
    assert clients[0].budget.outer_budget.failed is False
    before = len(generations)
    with pytest.raises(FatalRackError):
        clients[0].generate_messages('answer', [])
    assert len(generations) == before
    first = json.loads((Path(result['jobs'][0]['path']) / 'root-status.json').read_bytes())
    assert first['rows'][0]['status'] == 'complete'
    assert all(r['status'] == 'unrun' and 'score' not in r for r in first['rows'][1:])
    assert result['usage']['generation_calls'] == len(generations) == (3 + 451 + 1 if failure == 'invalid_judge' else 2 + 451)
    assert result['status'] == 'incomplete'
    assert len(result['matrix']) == 7216
    for method in v.METHODS:
        if method != 'no_memory':
            assert all(r['status'] == 'unrun' and 'score' not in r
                       for r in result['matrix'] if r['method'] == method)
    assert v.verify_root(out)['verified'] is True


@pytest.mark.parametrize('failure', ['discovery', 'model', 'tokenizer', 'parity', 'accounting',
                                    'judge_parity', 'judge_accounting', 'campaign'])
def test_v2_explicit_shared_failure_cancels_later_groups(tmp_path, monkeypatch, failure):
    import httpx
    v = driver()
    generations = []
    def fault(request, requests):
        if failure == 'discovery' and request.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [{'id': 'authored', 'max_model_len': 1}]})
        if failure == 'tokenizer' and request.url.path.endswith('/tokenize'):
            return httpx.Response(200, json={'count': -1})
        if not request.url.path.endswith('/chat/completions'):
            return
        generations.append(request)
        if failure == 'campaign':
            outer = clients[0].budget.outer_budget
            outer.started -= outer.max_seconds
        if failure in {'model', 'parity', 'accounting'}:
            response = {'model': 'wrong' if failure == 'model' else 'authored',
                        'choices': [{'finish_reason': 'stop',
                                     'message': {'content': r'\boxed{GOLD_ONLY}'}}],
                        'usage': {'prompt_tokens': 301 if failure == 'parity' else 300,
                                  'completion_tokens': 4}}
            return httpx.Response(200, json=response)
        if failure.startswith('judge_') and len(generations) == 3:
            return httpx.Response(200, json={'model': 'authored', 'choices': [
                {'finish_reason': 'stop', 'message': {'content': '{"label": 1, "reason": "authored"}'}}],
                'usage': {'prompt_tokens': 301 if failure == 'judge_parity' else 300,
                          'completion_tokens': None if failure == 'judge_accounting' else 4}})
    spec, out, clients, _ = authored_execution(
        tmp_path, monkeypatch, fault, semantic=failure.startswith('judge_'))
    if failure == 'accounting':
        def reject(*args, **kwargs):
            raise ValueError('authored unreconciled dispatch accounting')
        monkeypatch.setattr(v, 'generation_usage', reject)
    try:
        result = v._execute_approved(spec, out, authored=True)
    except ValueError:
        result = json.loads((out / 'root-status.json').read_bytes())
    assert len(clients) == 1
    assert clients[0].budget.outer_budget.failed is True
    assert result['status'] == 'failed'
    assert result['usage']['generation_calls'] == len(generations)
    assert all(r['status'] == 'unrun' and 'score' not in r
               for r in result['matrix'] if r['tier'] == 'medium')


def test_v2_local_scorer_oserror_is_not_shared_transport_evidence(tmp_path, monkeypatch):
    v = driver()
    spec, out, clients, requests = authored_execution(tmp_path, monkeypatch)
    original = v.NativeInterface.score
    calls = []
    def score(native, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError('authored local scorer artifact failure, no shared transport evidence')
        return original(native, *args, **kwargs)
    monkeypatch.setattr(v.NativeInterface, 'score', score)
    result = v._execute_approved(spec, out, authored=True)
    assert [j['status'] for j in result['jobs']] == ['failed', 'complete']
    assert clients[0].budget.failed is True
    assert clients[0].budget.outer_budget.failed is False
    assert result['usage']['generation_calls'] == 1 + 451


@pytest.mark.parametrize('fault', ['verifier', 'root_write_before', 'root_write_after',
                                 'checkpoint_before', 'checkpoint_after'])
def test_v2_root_finalization_failure_never_leaves_complete_authority(tmp_path, monkeypatch, fault):
    v = driver()
    spec, out, clients, requests = authored_execution(tmp_path, monkeypatch)
    original_verify, original_write = v.verify_root, v.write_json
    observed = []
    def verify(*args, **kwargs):
        if fault == 'verifier':
            observed.append(json.loads((out / 'root-status.json').read_bytes())['status'])
            original_verify(*args, **kwargs)
            raise ValueError('authored final verifier rejection')
        return original_verify(*args, **kwargs)
    def write(path, value):
        is_checkpoint = (Path(path) == out / 'root-status.json'
                         and value.get('status') == 'incomplete')
        if is_checkpoint and fault == 'checkpoint_before':
            raise OSError('authored final root checkpoint write failure')
        is_completion = Path(path).parent == out and value.get('status') == 'complete'
        if is_completion and fault == 'root_write_before':
            raise OSError('authored root write before bytes')
        original_write(path, value)
        if is_checkpoint and fault == 'checkpoint_after':
            raise OSError('authored final root checkpoint after bytes')
        if is_completion and fault == 'root_write_after':
            raise OSError('authored root write after bytes')
    monkeypatch.setattr(v, 'verify_root', verify)
    monkeypatch.setattr(v, 'write_json', write)
    with pytest.raises((ValueError, OSError), match='authored'):
        v._execute_approved(spec, out, authored=True)
    authority = json.loads((out / 'root-status.json').read_bytes())
    assert authority['status'] in {'running', 'incomplete', 'failed'}
    assert 'complete' not in observed
    assert authority['usage']['generation_calls'] == sum(c.budget.calls for c in clients) == 902
    assert len(authority['jobs']) == 2
    assert len(authority['matrix']) == 7216
    assert all(j['usage']['generation_calls'] == 451 for j in authority['jobs'])


@pytest.mark.parametrize('fault', ['replace', 'readback', 'post_verify_change'])
def test_v2_atomic_root_promotion_requires_verified_unchanged_readback(tmp_path, monkeypatch, fault):
    v = driver()
    spec, out, clients, requests = authored_execution(tmp_path, monkeypatch)
    original_verify, original_replace, original_write = v.verify_root, Path.replace, v.write_json
    def replace(path, target):
        if Path(target) == out / 'root-status.json':
            assert json.loads((out / 'root-status.json').read_bytes())['status'] != 'complete'
            raise OSError('authored atomic promotion failure')
        return original_replace(path, target)
    def verify(*args, **kwargs):
        result = original_verify(*args, **kwargs)
        if fault == 'post_verify_change':
            candidate = kwargs['_root_path']
            value = json.loads(candidate.read_bytes())
            value['usage']['generation_calls'] = 0
            original_write(candidate, value)
        return result
    def write(path, value):
        original_write(path, value)
        if fault == 'readback' and Path(path).name == '.root-status.pending.json':
            Path(path).write_bytes(b'{}')
    monkeypatch.setattr(v, 'verify_root', verify)
    monkeypatch.setattr(v, 'write_json', write)
    if fault == 'replace':
        monkeypatch.setattr(Path, 'replace', replace)
    with pytest.raises((ValueError, OSError), match='authored|readback|changed'):
        v._execute_approved(spec, out, authored=True)
    authority = json.loads((out / 'root-status.json').read_bytes())
    assert authority['status'] == 'failed'
    assert authority['usage']['generation_calls'] == 902


@pytest.mark.parametrize('tamper', ['promote_unrun', 'drop_job', 'duplicate_job', 'root_usage',
                                  'summary', 'group_id', 'rewritten_wire', 'complete_failed_job',
                                  'complete_unrun_rows', 'root_counts'])
def test_v2_rehashed_root_tampering_cannot_forge_completion(tmp_path, monkeypatch, tamper):
    v = driver()
    spec, out, clients, requests = authored_execution(tmp_path, monkeypatch)
    result = v._execute_approved(spec, out, authored=True)
    assert result['status'] == 'complete'
    assert result['counts'] == {'questions': 451, 'query_images': 29, 'small': 451,
                                'medium': 451, 'occurrences': 902}
    assert len(result['matrix']) == 902 * len(v.METHODS) == 7216
    for method in v.METHODS:
        for tier in ('small', 'medium'):
            assert result['summary']['methods'][method][tier][
                'complete' if method == 'no_memory' else 'unrun'] == 451
    root = json.loads((out / 'root-status.json').read_bytes())
    job = root['jobs'][0]
    leaf_path = Path(job['path']) / 'root-status.json'
    leaf = json.loads(leaf_path.read_bytes())
    if tamper == 'promote_unrun':
        root['jobs'] = []
        root['usage'] = {'generation_calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
        root['matrix'] = json.loads(Path(spec).read_bytes())['matrix']
    elif tamper == 'drop_job':
        removed = root['jobs'].pop()
        root['usage'] = {k: val - removed['usage'][k] for k, val in root['usage'].items()}
    elif tamper == 'duplicate_job':
        root['jobs'].append(dict(job))
        root['usage'] = {k: val + job['usage'][k] for k, val in root['usage'].items()}
    elif tamper == 'root_counts':
        root['counts']['occurrences'] = 0
    elif tamper == 'complete_unrun_rows':
        for row in root['matrix']:
            row['status'] = 'unrun'
            for key in ('pack', 'answer', 'score'):
                row.pop(key, None)
        root['summary'] = v._matrix_summary(root['matrix'], {'small': 451, 'medium': 451})
        for retained in root['jobs']:
            path = Path(retained['path']) / 'root-status.json'
            value = json.loads(path.read_bytes())
            value['rows'] = [r for r in root['matrix']
                             if r['method'] == retained['method'] and r['group_id'] == retained['group_id']]
            v.write_json(path, value)
            retained['root_sha256'] = v.file_hash(path)
    elif tamper == 'root_usage':
        root['usage']['prompt_tokens'] += 1
    elif tamper == 'summary':
        root['summary']['methods']['no_memory']['small']['complete'] = 0
    elif tamper == 'group_id':
        next(r for r in root['matrix'] if r['method'] == 'bm25')['group_id'] = 'invented'
    elif tamper == 'complete_failed_job':
        job['status'] = leaf['status'] = 'failed'
        v.write_json(leaf_path, leaf)
        job['root_sha256'] = v.file_hash(leaf_path)
    elif tamper == 'rewritten_wire':
        row = next(r for r in root['matrix'] if r['method'] == 'no_memory')
        messages = row['pack']['messages']
        messages[1]['content'] = [p for p in messages[1]['content'] if p['type'] != 'image_url']
        leaf['rows'][0]['pack']['messages'] = messages
        v.write_json(leaf_path, leaf)
        job['root_sha256'] = v.file_hash(leaf_path)
        calls = Path(job['path']) / 'calls.jsonl'
        wires = list(map(json.loads, calls.read_bytes().splitlines()))
        next(w for w in wires if w['url'].endswith('/chat/completions'))['request']['messages'] = messages
        calls.write_bytes(('\n'.join(map(json.dumps, wires)) + '\n').encode())
        root['files'][calls.relative_to(out).as_posix()] = v.file_hash(calls)
    v.write_json(out / 'root-status.json', root)
    with pytest.raises(ValueError):
        v.verify_root(out)


def test_real_cli_qualifier_dispatch_loads_classes_before_main(tmp_path):
    import subprocess
    import sys

    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"base_url": "https://example.org/v1", "model": "authored"}))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gluomem_bench.bench.v2_driver",
            "qualify-vision",
            "--profile",
            str(profile),
            "--native-root",
            str(tmp_path / "missing-native-source"),
            "--out",
            str(tmp_path / "not-created"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "NameError" not in result.stderr
    assert "FileNotFoundError" in result.stderr


def test_native_prefix_can_count_through_actual_served_image_processor_without_fake_estimates(
    tmp_path,
):
    import httpx

    v = driver()
    from gluomem_bench.bench.rack import RackClient, RackProfile

    native = v.NativeInterface(NATIVE_ROOT)
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200, json={"count": sum(len(p.get("text", "")) for p in body["messages"][0]["content"])}
        )

    client = RackClient(
        RackProfile(base_url="https://example.org/v1", model="authored"),
        receipt=tmp_path / "tokenize.jsonl",
        transport=httpx.MockTransport(handler),
    )
    native.use_served_context_counter(client)
    items = [{"type": "text", "value": "A" * 30}, {"type": "text", "value": "B" * 30}]
    assert native.truncate(items, max_tokens=40) == (items[:1], 60, 30)
    assert all(r["add_generation_prompt"] is False for r in seen)
    assert all(r["chat_template_kwargs"] == {"enable_thinking": False} for r in seen)
    assert (
        native.receipt["memory_processor"]
        == "actual served authored; substituted for native Qwen/Qwen3.5-9B"
    )
    client.close()


def test_embedding_dispatch_batches_complete_inputs_under_shared_deadline_and_never_truncates(
    tmp_path, monkeypatch
):
    v = driver()
    from gluomem_bench.bench.embedding_runtime import RemoteEmbedder
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    now = [0.0]
    budget = WorkloadBudget(max_seconds=10, clock=lambda: now[0])
    seen = []

    def encode(self, texts, role):
        seen.append((list(texts), role, self.timeout))
        now[0] += 1
        return [[1.0] * self.dim for _ in texts]

    monkeypatch.setattr(RemoteEmbedder, "_encode", encode)
    remote = RemoteEmbedder("https://example.org/v1", timeout=120)
    bounded = v.DeadlineEncoder(
        remote, budget=budget, receipt=tmp_path / "embeddings.jsonl", batch_size=2
    )
    texts = ["line\n" + str(i) for i in range(5)]
    assert len(bounded.embed_batch(texts)) == 5
    assert [t for batch, role, timeout in seen for t in batch] == texts
    assert [x[2] for x in seen] == [10, 9, 8]
    bounded.embed_query("exact query\n")
    assert seen[-1][1] == "query"
    now[0] = 11
    with pytest.raises(RuntimeError):
        bounded.embed("must not dispatch")
    assert len(seen) == 4
    assert bounded.fingerprint == remote.fingerprint


def test_all_native_specs_resolve_before_clients_and_unknown_specs_fail():
    v = driver()
    native = v.NativeInterface(NATIVE_ROOT)
    rows = [{'eval_function': 'norm_phrase_set_match|lower=true'},
            {'eval_function': 'llm_abstention_checker'}]
    assert native.validate_specs(rows) == {'objective': 1, 'llm': 1}
    with pytest.raises(ValueError, match='Unknown|unsupported'):
        native.validate_specs(rows + [{'eval_function': 'invented_metric'}])


def put(root, name, value, *, lines=False):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(
        ("\n".join(json.dumps(x) for x in value) if lines else json.dumps(value)).encode()
    )
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dataset(root, count=451):
    qs = [
        dict(
            id=f"q{i}",
            domain="web",
            environment="authored",
            question_type="procedure",
            question="Where is the blue menu?",
            image=None,
            answer="GOLD_ONLY",
            eval_function="norm_phrase_set_match",
        )
        for i in range(count)
    ]
    pins = {}
    for i in range(min(29, count)):
        name = f"question_screenshots/{i}.png"
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"authored image bytes")
        pins[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        qs[i]["image"] = name
    t = dict(
        id="source-secret-id",
        domain="web",
        environment="authored",
        goal="Find menu",
        outcome="success",
        start_url="https://example.org",
        states=[
            dict(
                state_index=i,
                step=i,
                url="https://example.org",
                action=f"incoming{i}",
                thought="Navigate",
                accessibility_tree="blue menu " + ("X" * 30),
                screenshot=f"absent-history/{i}.png",
            )
            for i in range(3)
        ],
    )
    pins["questions.jsonl"] = put(root, "questions.jsonl", qs, lines=True)
    pins["trajectories.jsonl"] = put(root, "trajectories.jsonl", [t], lines=True)
    for tier in ("small", "medium"):
        name = f"haystacks/lme_v2_{tier}.json"
        pins[name] = put(root, name, {q["id"]: [t["id"]] for q in qs})
    return pins, t


def test_full_scope_preserves_occurrences_histories_images_and_no_fake_clock(tmp_path):
    v = driver()
    pins, t = dataset(tmp_path)
    data = v.SourceDataset(tmp_path, pins=pins)
    assert data.counts() == {
        "questions": 451,
        "query_images": 29,
        "small": 451,
        "medium": 451,
        "occurrences": 902,
    }
    assert len(data.groups()) == 2
    for group in data.groups():
        assert len(group.question_ids) == 451
        assert list(data.histories(group)) == [t]
    assert data.reader_question("q0") == {
        "question": "Where is the blue menu?",
        "domain": "web",
        "image": str(tmp_path / "question_screenshots/0.png"),
        "question_time": None,
    }
    assert data.receipt["history_images_required"] is False
    (tmp_path / "questions.jsonl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        v.SourceDataset(tmp_path, pins=pins)
