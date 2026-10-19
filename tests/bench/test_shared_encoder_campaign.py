"""Offline shared-encoder and versioned whole-history admission contracts."""

import json

import numpy as np
import pytest

from gluomem_bench.bench.rack import RackProfile, controlled_config
from gluomem_bench.bench.rack_trace import WorkloadBudget


@pytest.fixture
def conflict_adapter(tmp_path, bench_cfg):
    from tests.bench.test_suite_campaign import conflict_adapter as original_fixture

    return original_fixture.__wrapped__(tmp_path, bench_cfg)


def profile(**kwargs):
    return RackProfile(base_url="http://localhost:8000/v1", model="example/served-model", **kwargs)


@pytest.mark.parametrize("system", ["bm25", "dense", "hybrid", "cognee"])
def test_explicit_raw_retriever_factory_reuses_full_guarded_history(
    tmp_path,
    conflict_adapter,
    system,
):
    import httpx

    from gluomem_bench.bench.suite_campaign import run_history, verify_history
    from gluomem_bench.bench.types import Prediction

    cfg, adapter = conflict_adapter
    closed = []

    def factory(actual_cfg, conv, answer):
        assert actual_cfg is cfg
        assert conv.conv_id == adapter.load()[0].conv_id

        class Raw:
            def answer(self, sample):
                result = answer(sample.reader_question, "Authored raw ranked evidence.")
                return Prediction(sample_id=sample.id, answer=result.text)

            def close(self):
                closed.append(True)

        return Raw()

    out = tmp_path / system
    result = run_history(
        cfg=cfg,
        adapter=adapter,
        samples=adapter.load(),
        profile=profile(),
        out=out,
        system=system,
        retriever_factory=factory,
        evidence_token_budget=4096,
        transport=httpx.MockTransport(serve_reader),
    )
    assert verify_history(out) == result
    assert result["system"] == system
    assert closed == [True]
    assert len(json.loads((out / "reader-evidence.json").read_bytes())) == 2


def test_expanded_history_budget_requires_explicit_v5_policy():
    budgets = dict(extraction=8192, consolidation=1024, reader=512, judge=256)
    p = profile(
        profile_version="role-budgets-v5",
        generation_budgets=budgets,
        max_generation_calls=16384,
        max_run_seconds=21600,
    )
    b = WorkloadBudget(
        max_calls=p.max_generation_calls, max_seconds=p.max_run_seconds, policy=p.profile_version
    )
    assert (b.max_calls, b.max_seconds) == (16384, 21600)
    for version in ("legacy-v1", "role-budgets-v2", "role-budgets-v4"):
        for bounds in (dict(max_generation_calls=257), dict(max_run_seconds=901)):
            with pytest.raises(ValueError):
                profile(
                    profile_version=version,
                    generation_budgets=None if version == "legacy-v1" else budgets,
                    **bounds,
                )
    for bounds in (dict(max_calls=257), dict(max_seconds=901)):
        with pytest.raises(ValueError):
            WorkloadBudget(**bounds)
    for bounds in (dict(max_calls=16385), dict(max_seconds=21601)):
        with pytest.raises(ValueError):
            WorkloadBudget(policy="role-budgets-v5", **bounds)
    with pytest.raises(ValueError):
        WorkloadBudget(policy="unknown")
    with pytest.raises(ValueError):
        profile(
            profile_version="role-budgets-v5",
            generation_budgets=budgets,
            max_generation_calls=16385,
        )


class SyntheticEncoder:
    """Authored deterministic vectors, never a qualified model or quality evidence."""

    def __init__(self):
        from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe, RemoteEmbedder

        self.recipe = EmbeddingRecipe()
        self.dim = 768
        self.base_url = "http://127.0.0.1:8766/v1"
        self.fingerprint = RemoteEmbedder(self.base_url, recipe=self.recipe).fingerprint
        self.calls = []

    def _vector(self, text, role):
        import hashlib

        self.calls.append((role, text))
        value = np.zeros(self.dim, dtype=np.float32)
        value[0] = 1
        value[1 + hashlib.sha256((role + text).encode()).digest()[0]] = 0.01
        return value / np.linalg.norm(value)

    def embed(self, text):
        return self._vector(text, "document")

    def embed_query(self, text):
        return self._vector(text, "query")

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]


def test_explicit_shared_config_binds_recipe_and_fingerprint(tmp_path):
    from gluomem_bench.bench.rack import shared_encoder_evidence

    encoder = SyntheticEncoder()
    cfg = controlled_config(profile(), data=tmp_path / "source", sha256="a" * 64, embedder=encoder)
    entry = cfg.models[cfg.embedding.model]
    assert (entry.provider_model, entry.revision, entry.dim) == (
        encoder.recipe.model,
        encoder.recipe.revision,
        768,
    )
    assert cfg.benchmark.mem0.embedder_model == cfg.embedding.model
    evidence = shared_encoder_evidence(cfg, encoder)
    assert evidence["fingerprint"] == encoder.fingerprint.as_dict()
    assert evidence["recipe"] == encoder.recipe.as_dict()
    with pytest.raises(ValueError, match="configuration"):
        shared_encoder_evidence(
            cfg.with_overrides({"models.shared-encoder.revision": "wrong"}), encoder
        )
    from dataclasses import replace

    encoder.fingerprint = replace(encoder.fingerprint, preprocessing="wrong")
    with pytest.raises(ValueError, match="fingerprint"):
        shared_encoder_evidence(cfg, encoder)
    old = controlled_config(profile(), data=tmp_path / "source", sha256="a" * 64)
    assert old.embedding.model == "minilm"


def test_real_native_instance_shared_adapter_preserves_roles_newlines_and_vectors(tmp_path):
    import os

    os.environ.setdefault("MEM0_TELEMETRY", "False")
    os.environ.setdefault("MEM0_DIR", str(tmp_path / "native-home"))
    from mem0 import Memory
    from mem0.embeddings.openai import OpenAIEmbedding

    from gluomem_bench.bench.mem0 import (
        build_mem0_config,
        close_mem0_memory,
        install_shared_mem0_encoder,
    )

    original_embed = OpenAIEmbedding.embed
    encoder = SyntheticEncoder()
    remote = os.environ.get("GLUOMEM_TEST_SHARED_EMBEDDING_URL")
    if remote:
        from gluomem_bench.bench.embedding_runtime import RemoteEmbedder

        encoder = RemoteEmbedder(remote)
    cfg = controlled_config(profile(), data=tmp_path / "source", sha256="a" * 64, embedder=encoder)
    config = build_mem0_config(
        cfg, conv_dir=tmp_path / "native", profile=profile(), embedder=encoder
    )
    assert config["embedder"]["provider"] == "openai"
    assert config["embedder"]["config"]["openai_base_url"] == "http://127.0.0.1:8766/v1"
    memory = Memory.from_config(config)
    try:
        assert isinstance(memory.embedding_model, OpenAIEmbedding)
        install_shared_mem0_encoder(memory, cfg=cfg, embedder=encoder)
        adapter = memory.embedding_model
        text = "Alpha\nBeta"
        parity_checks = []
        for action, expected in (
            ("search", encoder.embed_query),
            ("add", encoder.embed),
            ("update", encoder.embed),
            (None, encoder.embed),
        ):
            actual, direct = adapter.embed(text, action), expected(text)
            np.testing.assert_array_equal(actual, direct)
            import hashlib

            parity_checks.append(
                {
                    "action": action,
                    "batch": False,
                    "vector_sha256": hashlib.sha256(
                        np.asarray(actual, dtype=np.float32).tobytes()
                    ).hexdigest(),
                }
            )
        for action in ("add", "update", "search"):
            expected_vectors = (
                [encoder.embed_query(text), encoder.embed_query("Other")]
                if action == "search"
                else encoder.embed_batch([text, "Other"])
            )
            actual = adapter.embed_batch([text, "Other"], action)
            np.testing.assert_array_equal(actual, expected_vectors)
            parity_checks.append(
                {
                    "action": action,
                    "batch": True,
                    "vector_sha256": hashlib.sha256(
                        np.asarray(actual, dtype=np.float32).tobytes()
                    ).hexdigest(),
                }
            )
        if not remote:
            assert all("Alpha Beta" != text for _, text in encoder.calls)
        assert adapter.fingerprint == encoder.fingerprint
        assert OpenAIEmbedding.embed is original_embed
        (tmp_path / "native-vector-parity.json").write_bytes(
            json.dumps(
                {
                    "synthetic_vectors": not bool(remote),
                    "actual_generation_calls": 0,
                    "fingerprint": encoder.fingerprint.as_dict(),
                    "checks": parity_checks,
                    "comparison": "exact equality, same role/text/batch shape; preserve newlines",
                },
                indent=2,
            ).encode()
        )
        with pytest.raises(ValueError):
            adapter.embed(text, "unknown")
        encoder.embed_batch = lambda texts: [np.zeros(384, dtype=np.float32) for _ in texts]
        with pytest.raises(ValueError, match="vector"):
            adapter.embed(text, "add")
    finally:
        close_mem0_memory(memory)


def serve_reader(request):
    import httpx

    if request.url.path.endswith("/models"):
        return httpx.Response(
            200, json={"data": [{"id": "example/served-model", "max_model_len": 131072}]}
        )
    return httpx.Response(
        200,
        json={"count": 10}
        if request.url.path == "/tokenize"
        else {
            "id": "authored",
            "object": "chat.completion",
            "created": 1,
            "model": "example/served-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "right"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        },
    )


def test_full_production_history_uses_injected_encoder_and_outer_budget(
    tmp_path,
    conflict_adapter,
    monkeypatch,
):
    import httpx

    from gluomem_bench.bench.suite_campaign import run_history, verify_history
    from tests.bench.test_beam_ingest import RecordingExtractors

    _, adapter = conflict_adapter
    encoder = SyntheticEncoder()
    p = profile(
        profile_version="role-budgets-v5",
        max_generation_calls=4096,
        max_run_seconds=3600,
        generation_budgets=dict(extraction=8192, consolidation=1024, reader=512, judge=256),
    )
    cfg = controlled_config(p, data=tmp_path / "unused", sha256="a" * 64, embedder=encoder)
    # Only extraction is authored; production ingest, readback and hybrid retrieval are real.
    recorder = RecordingExtractors()
    monkeypatch.setattr("gluomem.llm.modules.DspyExtractors", lambda *a, **kw: recorder)

    class Outer:
        failed = False
        calls = 0
        max_calls = 8192
        max_seconds = 3600

        def check(self):
            if self.failed:
                raise RuntimeError("outer exhausted")

        def remaining(self):
            self.check()
            return 100

        def begin_call(self):
            self.check()
            self.calls += 1

    outer = Outer()
    out = tmp_path / "shared-production"
    result = run_history(
        cfg=cfg,
        adapter=adapter,
        samples=adapter.load(),
        profile=p,
        out=out,
        system="gluomem",
        embedder=encoder,
        outer_budget=outer,
        transport=httpx.MockTransport(serve_reader),
    )
    assert result["questions"] == 2
    assert verify_history(out) == result
    assert recorder.extractions
    assert encoder.calls
    assert outer.calls == json.loads((out / "usage.json").read_bytes())["generation_calls"]
    parity = json.loads((out / "shared-encoder.json").read_bytes())
    assert parity["fingerprint"] == encoder.fingerprint.as_dict()
    assert json.loads((out / "profile.json").read_bytes())["max_generation_calls"] == 4096
    saved_protocol = json.loads((out / "protocol.json").read_bytes())
    assert saved_protocol["outer_budget"]["max_calls"] == 8192
    assert saved_protocol["outer_budget"]["max_seconds"] == 3600
    import hashlib

    parity["fingerprint"]["revision"] = "wrong"
    path = out / "shared-encoder.json"
    path.write_bytes(json.dumps(parity).encode())
    manifest = json.loads((out / "manifest.json").read_bytes())
    manifest["artifact_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "manifest.json").write_bytes(json.dumps(manifest).encode())
    with pytest.raises(ValueError, match="encoder"):
        verify_history(out)


def test_full_native_history_shared_encoder_guarded_llm_and_readback(tmp_path, monkeypatch):
    import os
    from datetime import datetime
    from types import SimpleNamespace

    import httpx
    from mem0.configs.base import mem0_dir

    from gluomem_bench.bench.suite_campaign import run_history, verify_history
    from gluomem_bench.bench.types import Sample
    from gluomem_bench.ingest import Conversation, Session, Turn

    monkeypatch.setenv("MEM0_TELEMETRY", "False")
    monkeypatch.setenv("MEM0_DIR", mem0_dir)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    encoder = SyntheticEncoder()
    remote = os.environ.get("GLUOMEM_TEST_SHARED_EMBEDDING_URL")
    if remote:
        from gluomem_bench.bench.embedding_runtime import RemoteEmbedder

        encoder = RemoteEmbedder(remote)
    p = profile(max_input_chars=1000000)
    cfg = controlled_config(p, data=tmp_path / "unused", sha256="a" * 64, embedder=encoder)
    sample = Sample(
        id="authored-q",
        conv_id="authored",
        question="Which postcards?",
        gold="red",
        category=1,
        category_name="single-session-user",
        question_date="2023-01-02",
    )
    conv = Conversation(
        "authored",
        ("user", "assistant"),
        [
            Session(
                1,
                datetime(2023, 1, 1),
                [Turn("user", "I collect red postcards."), Turn("assistant", "Try a binder.")],
            )
        ],
    )
    adapter = SimpleNamespace(
        name="longmemeval_oracle",
        sha="a" * 64,
        verify=lambda: None,
        load=lambda: [sample],
        conversations=lambda: [conv],
    )
    monkeypatch.setattr(
        "gluomem_bench.bench.suite_campaign.load_longmemeval_prompt",
        lambda _: lambda *a, **kw: "Authored synthetic judgment.",
    )

    def serve(request):
        if request.url.path.endswith("/chat/completions"):
            payload = json.loads(request.content)
            response = serve_reader(request).json()
            text = (
                '{"memory": [{"text": "User collects red postcards.", "event": "ADD"}]}'
                if "response_format" in payload
                else "red"
                if payload["messages"][0]["role"] == "system"
                else "yes"
            )
            response["choices"][0]["message"]["content"] = text
            return httpx.Response(200, json=response)
        return serve_reader(request)

    out = tmp_path / "native-run"
    result = run_history(
        cfg=cfg,
        adapter=adapter,
        samples=[sample],
        profile=p,
        out=out,
        system="mem0",
        embedder=encoder,
        evidence_token_budget=4096,
        judge_source=tmp_path / "authored-not-upstream",
        transport=httpx.MockTransport(serve),
    )
    assert verify_history(out) == result
    assert json.loads((out / "mem0-ingestion.json").read_bytes())["memories"] == 1
    assert json.loads((out / "usage.json").read_bytes())["generation_calls"] == 3
    assert json.loads((out / "mem0-embedding.json").read_bytes())["fingerprint"] == (
        encoder.fingerprint.as_dict()
    )
    if not remote:
        assert any(role == "query" for role, _ in encoder.calls)
        assert any(role == "document" for role, _ in encoder.calls)


@pytest.mark.parametrize("system", ["bm25", "no_memory", "full_context"])
def test_optional_evidence_ceiling_uses_served_tokenizer_and_saves_reader_truth(
    tmp_path,
    conflict_adapter,
    system,
):
    import hashlib

    import httpx

    from gluomem_bench.bench.suite_campaign import run_history, verify_history

    cfg, adapter = conflict_adapter

    def serve(request):
        if request.url.path == "/tokenize":
            payload = json.loads(request.content)
            evidence = json.loads(payload["messages"][1]["content"])["evidence"]
            return httpx.Response(200, json={"count": 10 + len(evidence)})
        return serve_reader(request)

    out = tmp_path / system
    result = run_history(
        cfg=cfg,
        adapter=adapter,
        samples=adapter.load(),
        profile=profile(),
        out=out,
        system=system,
        evidence_token_budget=10,
        transport=httpx.MockTransport(serve),
    )
    assert verify_history(out) == result
    receipts = json.loads((out / "reader-evidence.json").read_bytes())
    assert len(receipts) == 2
    assert receipts[0]["sample_id"] == adapter.load()[0].id
    for row in receipts:
        for prefix in ("original", "kept"):
            assert (
                row[prefix + "_sha256"]
                == hashlib.sha256(row[prefix + "_text"].encode()).hexdigest()
            )
        assert row["kept_tokens"] == 10 + len(row["kept_text"])
        if system == "full_context":
            assert row["kept_text"] == row["original_text"] != ""
            assert row["exempt"] is True
        else:
            assert row["kept_text"] == ""
    assert json.loads((out / "protocol.json").read_bytes())["evidence_token_budget"] == 10
    # Rehashing a dishonest receipt must not make it accepted as reader truth.
    receipts[0]["kept_tokens"] += 1
    path = out / "reader-evidence.json"
    path.write_bytes(json.dumps(receipts).encode())
    manifest = json.loads((out / "manifest.json").read_bytes())
    manifest["artifact_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "manifest.json").write_bytes(json.dumps(manifest).encode())
    with pytest.raises(ValueError, match="evidence"):
        verify_history(out)


def test_outer_call_refusal_latches_local_budget_and_preserves_spend():
    class Outer:
        def check(self):
            pass

        def remaining(self):
            return 17

        def begin_call(self):
            raise RuntimeError("outer calls exhausted")

    outer = Outer()
    budget = WorkloadBudget(outer_budget=outer)
    assert budget.remaining() <= 17
    with pytest.raises(RuntimeError, match="outer"):
        budget.begin_call()
    assert budget.failed
    assert budget.calls == 0


def test_evidence_receipts_cover_zero_reader_abstention_without_inventing_calls(
    tmp_path,
    conflict_adapter,
):
    from dataclasses import replace
    from types import SimpleNamespace

    import httpx

    from gluomem_bench.bench.suite_campaign import run_history, verify_history

    cfg, original = conflict_adapter
    samples = [replace(s, question="zzzzunmatchedzzzz") for s in original.load()]
    adapter = SimpleNamespace(
        name=original.name,
        sha=original.sha,
        verify=original.verify,
        load=lambda: samples,
        conversations=original.conversations,
    )
    out = tmp_path / "no-readers"
    result = run_history(
        cfg=cfg,
        adapter=adapter,
        samples=samples,
        profile=profile(),
        out=out,
        system="bm25",
        evidence_token_budget=4096,
        transport=httpx.MockTransport(serve_reader),
    )
    assert verify_history(out) == result
    assert json.loads((out / "reader-evidence.json").read_bytes()) == []
    assert json.loads((out / "usage.json").read_bytes())["generation_calls"] == 0
