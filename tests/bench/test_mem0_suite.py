"""Mem0's native JSON extraction must use the existing guarded transport."""
import json

import httpx
import pytest

from gluomem_bench.bench.rack import RackClient, RackProfile


def test_native_json_request_preserves_raw_response_and_guards(tmp_path):
    requests = []
    raw = {"id": "authored-response", "object": "chat.completion", "created": 1,
           "model": "example/served-model",
           "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"memory": []}'}}],
           "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}

    def serve(request):
        requests.append(request)
        return httpx.Response(200, json={"count": 10} if request.url.path == "/tokenize" else raw)

    client = RackClient(RackProfile(base_url="http://localhost:8000/v1",
                                   model="example/served-model"),
                        receipt=tmp_path / "calls.jsonl", transport=httpx.MockTransport(serve))
    try:
        result = client.generate_messages("event_claims", [{"role": "user", "content": "Extract."}],
                                          response_format={"type": "json_object"}, top_p=1.0)
        assert result["response"] == raw
        assert json.loads(requests[-1].content)["response_format"] == {"type": "json_object"}
        assert json.loads(requests[-1].content)["model"] == "example/served-model"
        assert json.loads(requests[-1].content)["temperature"] == 0
        assert json.loads(requests[-1].content)["top_p"] == 1.0
    finally:
        client.close()


def test_local_mem0_config_is_explicit_and_ignores_commercial_environment(tmp_path, monkeypatch):
    from gluomem_bench.bench.mem0 import build_mem0_config
    from gluomem_bench.bench.rack import controlled_config

    profile = RackProfile(base_url="http://localhost:8000/v1", model="example/served-model")
    cfg = controlled_config(profile, data=tmp_path / "source.json", sha256="a" * 64)
    cfg = cfg.with_overrides({"benchmark.mem0.model": "rack-qwen",
                              "benchmark.mem0.embedder_model": "minilm"})
    monkeypatch.setenv("OPENROUTER_API_KEY", "authored-not-a-real-key")
    config = build_mem0_config(cfg, conv_dir=tmp_path / "store", profile=profile)
    assert config["llm"]["config"]["openai_base_url"] == profile.base_url
    assert config["llm"]["config"]["model"] == profile.model
    assert "openrouter_base_url" not in config["llm"]["config"]
    assert config["llm"]["config"]["temperature"] == 0
    assert config["llm"]["config"]["max_tokens"] == profile.output_budget("event_claims")
    assert config["embedder"]["config"]["model_kwargs"]["revision"] == cfg.models["minilm"].revision


def test_mem0_sdk_bridge_returns_the_actual_response_without_provider_fallback(tmp_path):
    import pytest

    from gluomem_bench.bench.mem0 import GuardedMem0SDK
    from gluomem_bench.bench.rack import FatalRackError

    raw = {"id": "authored", "object": "chat.completion", "created": 1,
           "model": "example/served-model",
           "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"memory": []}'}}],
           "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}
    requests = []

    def serve(request):
        requests.append(request)
        return httpx.Response(200, json={"count": 10} if request.url.path == "/tokenize" else raw)

    client = RackClient(RackProfile(base_url="http://localhost:8000/v1",
                                   model="example/served-model"),
                        receipt=tmp_path / "calls.jsonl", transport=httpx.MockTransport(serve))
    sdk = GuardedMem0SDK(client, trace=tmp_path / "native.jsonl", expected_system="Extract facts.")
    try:
        result = sdk.chat.completions.create(
            model=client.profile.model, temperature=0, top_p=1.0, max_tokens=512,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": "Extract facts."},
                      {"role": "user", "content": "Authored input."}],
        )
        assert result.id == raw["id"]
        assert result.choices[0].message.content == raw["choices"][0]["message"]["content"]
        assert sdk.records[0]["response"] == raw
        before = len(requests)
        with pytest.raises(FatalRackError):
            sdk.chat.completions.create(model="other/model", messages=[], tools=[])
        assert len(requests) == before
        with pytest.raises(FatalRackError):
            client.generate_messages("answer", [{"role": "user", "content": "Do not run."}])
        assert len(requests) == before
    finally:
        client.close()


def test_native_empty_is_distinct_from_parse_and_schema_failures():
    from gluomem_bench.bench.mem0 import inspect_native_extraction

    assert inspect_native_extraction('{"memory": []}')["status"] == "valid_empty"
    value = inspect_native_extraction('{"memory": [{"text": "Authored fact."}]}')
    assert value["status"] == "valid"
    assert inspect_native_extraction("not JSON")["status"] == "malformed_json"
    assert inspect_native_extraction('{"facts": []}')["status"] == "schema_incompatible"
    assert inspect_native_extraction('{"memory": [1]}')["status"] == "schema_incompatible"


def test_native_warning_is_evidence_not_silent_feature_degradation(tmp_path):
    import logging

    from gluomem_bench.bench.mem0 import NativeMem0Diagnostics
    from gluomem_bench.bench.rack import FatalRackError

    path = tmp_path / "diagnostics.json"
    observer = NativeMem0Diagnostics(path)
    try:
        observer.check()
        logging.getLogger("mem0.memory.main").warning("Failed to embed authored memory text")
        with pytest.raises(FatalRackError, match="native Mem0 runtime"):
            observer.check()
        assert json.loads(path.read_bytes())[0]["level"] == "WARNING"
    finally:
        observer.close()


def test_prepared_mem0_retriever_cannot_construct_an_uncontrolled_client(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import gluomem_bench.bench.mem0 as mod
    from gluomem_bench.bench.rack import controlled_config
    from gluomem_bench.bench.types import Sample

    def forbidden():
        pytest.fail("prepared retrieval must never construct a fresh native provider")

    monkeypatch.setattr(mod, "_import_memory_cls", forbidden)
    cfg = controlled_config(RackProfile(base_url="http://localhost:8000/v1",
                                       model="example/served-model"),
                            data=tmp_path / "unused.json", sha256="a" * 64)
    memory = SimpleNamespace(search=lambda *args, **kwargs: {"results": [
        {"memory": "Authored red postcard fact.", "metadata": {"session_date": "2023-01-01"}}]})
    calls = []

    def answer(q, context):
        calls.append((q, context))
        return SimpleNamespace(text="red", prompt_tokens=1, completion_tokens=1)

    retriever = mod.Mem0Retriever(cfg, db_dir=tmp_path, answer_fn=answer,
                                  prepared_memories={"authored": memory})
    sample = Sample(id="q", conv_id="authored", question="Which color?", gold="red", category=1,
                    category_name="single-session-user", question_date="2023-01-02")
    assert retriever.answer(sample).answer == "red"
    assert calls[0][0] == sample.reader_question
    assert calls[0][1].startswith("[2023-01-01]")
    from dataclasses import replace

    with pytest.raises(FileNotFoundError):
        retriever.answer(replace(sample, conv_id="unprepared"))


@pytest.mark.slow
def test_real_native_mem0_add_uses_guarded_transport_and_committed_readback(tmp_path, monkeypatch):
    from datetime import datetime

    from gluomem_bench.bench.mem0 import close_mem0_memory, ingest_guarded_mem0
    from gluomem_bench.bench.rack import controlled_config
    from gluomem_bench.bench.rack_trace import WorkloadBudget
    from gluomem_bench.bench.suite_evidence import validate_mem0_evidence
    from gluomem_bench.ingest import Conversation, Session, Turn

    monkeypatch.setenv("MEM0_TELEMETRY", "False")
    monkeypatch.setenv("MEM0_DIR", str(tmp_path / "native-home"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pytest.importorskip("mem0")
    profile = RackProfile(base_url="http://localhost:8000/v1", model="example/served-model",
                          max_input_chars=1000000)
    cfg = controlled_config(profile, data=tmp_path / "unused.json", sha256="a" * 64)
    cfg = cfg.with_overrides({"benchmark.mem0.model": "rack-qwen",
                              "benchmark.mem0.embedder_model": "minilm"})
    conv = Conversation("authored", ("user", "assistant"), [
        Session(1, datetime(2023, 1, 1), [Turn("user", "I collect red postcards."),
                                       Turn("assistant", "You might organize them in a binder.")])])
    budget = WorkloadBudget(max_calls=6, max_seconds=120)
    text = '{"memory": [{"text": "User collects red postcards.", "event": "ADD"}]}'

    def serve(request):
        return httpx.Response(200, json={"count": 10} if request.url.path == "/tokenize" else {
            "id": "authored", "object": "chat.completion", "created": 1,
            "model": profile.model, "choices": [{"index": 0, "finish_reason": "stop",
                                               "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
        })

    out = tmp_path / "run"
    out.mkdir()
    client = RackClient(profile, receipt=out / "calls.jsonl", budget=budget,
                        transport=httpx.MockTransport(serve))
    memory = None
    try:
        memory = ingest_guarded_mem0(cfg, conv=conv, out=out, client=client, budget=budget)
        receipt = json.loads((out / "mem0-ingestion.json").read_bytes())
        assert receipt["complete"] is True
        assert receipt["submitted_turns"] == 2
        assert len(receipt["sessions"]) == 1
        assert receipt["sessions"][0]["extraction"]["status"] == "valid"
        assert receipt["memories"] == 1
        assert budget.calls == 1
        assert memory.get_all(filters={"user_id": conv.conv_id})["results"][0]["memory"] == (
            "User collects red postcards.")
        inventory = {p.relative_to(out).as_posix(): "checked-by-outer-verifier"
                     for p in out.rglob("*") if p.is_file()}
        direct = [json.loads(line) for line in (out / "calls.jsonl").read_bytes().splitlines()
                  if json.loads(line)["url"].endswith("/chat/completions")]
        protocol = {"mem0_source_inputs_sha256": receipt["source_inputs_sha256"]}
        assert validate_mem0_evidence(out, inventory, profile, protocol, direct) == 1
        native_file = out / "mem0-native.jsonl"
        native = json.loads(native_file.read_bytes())
        native["task"] = "answer"
        native_file.write_bytes((json.dumps(native) + "\n").encode())
        with pytest.raises(ValueError, match="task"):
            validate_mem0_evidence(out, inventory, profile, protocol, direct)
    finally:
        if memory is not None:
            close_mem0_memory(memory)
        client.close()


def test_campaign_routes_mem0_through_shared_budget_and_retains_ingest_failure(
    tmp_path, monkeypatch
):
    from datetime import datetime
    from types import SimpleNamespace

    from gluomem_bench.bench import mem0, suite_campaign
    from gluomem_bench.bench.rack import controlled_config
    from gluomem_bench.bench.types import Sample
    from gluomem_bench.ingest import Conversation, Session, Turn

    profile = RackProfile(base_url="http://localhost:8000/v1", model="example/served-model")
    cfg = controlled_config(profile, data=tmp_path / "unused", sha256="a" * 64)
    sample = Sample(id="q", conv_id="authored", question="Color?", gold="red", category=1,
                    category_name="single-session-user", question_date="2023-01-02")
    conv = Conversation("authored", ("user", "assistant"), [
        Session(1, datetime(2023, 1, 1), [Turn("user", "Red.")])])
    adapter = SimpleNamespace(name="longmemeval_oracle", sha="a" * 64, verify=lambda: None,
                              load=lambda: [sample], conversations=lambda: [conv])
    monkeypatch.setattr(suite_campaign, "load_longmemeval_prompt", lambda _: object())

    def stop(cfg, *, conv, out, client, budget):
        assert client.budget is budget
        raise RuntimeError("authored ingestion checkpoint")

    monkeypatch.setattr(mem0, "ingest_guarded_mem0", stop)

    def serve(request):
        assert request.method == "GET"
        return httpx.Response(200, json={"data": [{"id": profile.model, "max_model_len": 131072}]})

    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="ingestion checkpoint"):
        suite_campaign.run_history(cfg=cfg, adapter=adapter, samples=[sample], profile=profile,
                                   out=out, system="mem0", judge_source=tmp_path / "scorer.py",
                                   transport=httpx.MockTransport(serve))
    assert json.loads((out / "failure.json").read_bytes())["generation_calls"] == 0
    assert not (out / "manifest.json").exists()
