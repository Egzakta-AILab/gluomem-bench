"""Guarded native Cognee qualification, not benchmark quality tests."""

import base64
import json
from contextlib import contextmanager

import httpx
import pytest

from gluomem_bench.bench.rack import RackProfile
from gluomem_bench.bench.rack_trace import WorkloadBudget


def profile(**changes):
    return RackProfile(base_url="http://qwen.example.invalid/v1", model="served/qwen", **changes)


def payload():
    return {
        "model": "served/qwen",
        "temperature": 0,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Native prompt\nuntouched"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "Graph",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            },
        },
    }


def upstream(seen, *, finish="stop", count=12, prompt_tokens=12):
    def handle(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.url.path, body))
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "served/qwen"}]})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": count})
        return httpx.Response(
            200,
            json={
                "model": "served/qwen",
                "choices": [
                    {
                        "finish_reason": finish,
                        "message": {"role": "assistant", "content": '{"ok":true}'},
                    }
                ],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 4},
            },
        )

    return httpx.MockTransport(handle)


@pytest.mark.parametrize(
    "change",
    [
        {"model": "other"},
        {"temperature": 0.1},
        {"temperature": False},
        {"n": True},
        {"stream": True},
        {"max_tokens": 0},
        {"max_tokens": 513},
        {"fallbacks": ["other"]},
        {"n": 2},
        {"chat_template_kwargs": {"enable_thinking": True}},
    ],
)
def test_invalid_native_request_latches_before_any_generation(tmp_path, change):
    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge

    seen = []
    budget = WorkloadBudget(max_calls=2)
    with CogneeGenerationBridge(
        profile(), budget=budget, receipt=tmp_path / "wire", transport=upstream(seen)
    ) as bridge:
        with pytest.raises(RuntimeError):
            bridge.complete(json.dumps({**payload(), **change}).encode())
        with pytest.raises(RuntimeError, match="latched"):
            bridge.complete(json.dumps(payload()).encode())
    assert budget.failed and budget.calls == 0
    assert not any(path == "/v1/chat/completions" for path, _ in seen)


@pytest.mark.parametrize("count", [True, -1, 131000])
def test_unknown_or_overflow_token_count_fatal_before_dispatch(tmp_path, count):
    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge

    seen = []
    budget = WorkloadBudget()
    with CogneeGenerationBridge(
        profile(), budget=budget, receipt=tmp_path / "wire", transport=upstream(seen, count=count)
    ) as bridge:
        with pytest.raises(RuntimeError):
            bridge.complete(json.dumps(payload()).encode())
    assert budget.failed and budget.calls == 0
    assert not any(path == "/v1/chat/completions" for path, _ in seen)


@pytest.mark.parametrize("finish,prompt_tokens", [("length", 12), ("stop", 13)])
def test_response_protocol_failure_retains_actual_spend(tmp_path, finish, prompt_tokens):
    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge

    budget = WorkloadBudget()
    receipt = tmp_path / "wire"
    with CogneeGenerationBridge(
        profile(),
        budget=budget,
        receipt=receipt,
        transport=upstream([], finish=finish, prompt_tokens=prompt_tokens),
    ) as bridge:
        with pytest.raises(RuntimeError):
            bridge.complete(json.dumps(payload()).encode())
        with pytest.raises(RuntimeError, match="latched"):
            bridge.complete(json.dumps(payload()).encode())
    assert budget.failed and budget.calls == 1
    row = json.loads(receipt.read_text().splitlines()[-1])
    assert row["status"] == "failed" and row["calls"] == 1
    assert "response_b64" in row


def test_loopback_proxy_binds_existing_settings_and_returns_native_bytes(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge
    from gluomem_bench.bench.comparison_backends import CogneeSettings

    seen = []
    budget = WorkloadBudget(max_calls=4)
    native = CogneeSettings(
        python_executable="python",
        root=tmp_path / "native",
        dataset="synthetic",
        llm_url="http://forbidden.example.invalid/v1",
        llm_model="served/qwen",
        llm_api_key="secret",
        embedding_url="http://127.0.0.1:8766/v1",
        embedding_model="google/embeddinggemma-2",
        embedding_dimensions=768,
        embedding_api_key="local",
        max_completion_tokens=64,
    )
    with CogneeGenerationBridge(
        profile(), budget=budget, receipt=tmp_path / "wire", transport=upstream(seen)
    ) as bridge:
        settings = bridge.bind(native)
        assert settings.root == native.root and settings.llm_url.startswith("http://127.0.0.1:")
        assert settings.llm_url != native.llm_url and settings.llm_api_key != "secret"

        def send(_):
            return httpx.post(
                settings.llm_url + "/chat/completions",
                json=payload(),
                headers={"Authorization": "Bearer " + settings.llm_api_key},
                trust_env=False,
            )

        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(send, range(3)))
        assert all(r.status_code == 200 for r in results)
        assert all(r.json()["model"] == "served/qwen" for r in results)
    assert budget.calls == 3
    rows = [json.loads(line) for line in (tmp_path / "wire").read_text().splitlines()]
    assert len([r for r in rows if r["kind"] == "generation"]) == 3


def test_native_schema_incompatible_response_is_fatal_not_zero(tmp_path):
    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge

    budget = WorkloadBudget()
    request = payload()
    request["response_format"]["json_schema"]["schema"]["properties"]["ok"] = {"type": "string"}
    with CogneeGenerationBridge(
        profile(), budget=budget, receipt=tmp_path / "wire", transport=upstream([])
    ) as bridge:
        with pytest.raises(RuntimeError):
            bridge.complete(json.dumps(request).encode())
    assert budget.failed and budget.calls == 1


def native_store(tmp_path, chunks):
    import sqlite3
    from dataclasses import asdict

    root = tmp_path / "native"
    (root / "system" / "databases").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "inputs.jsonl").write_text("".join(json.dumps(asdict(c)) + "\n" for c in chunks))
    db = root / "system" / "databases" / "cognee_db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE datasets (id TEXT, name TEXT)")
        conn.execute(
            "INSERT INTO datasets VALUES ('12345678123456781234567812345678', 'synthetic')"
        )
        conn.execute(
            "CREATE TABLE data (id TEXT, dataset_id TEXT, "
            "raw_data_location TEXT, pipeline_status TEXT)"
        )
        for i, text in enumerate(dict.fromkeys(c.text for c in chunks)):
            raw = root / "data" / f"{i}.txt"
            raw.write_bytes(text.encode())
            status = {
                stage: {"12345678-1234-5678-1234-567812345678": "DATA_ITEM_PROCESSING_COMPLETED"}
                for stage in ("add_pipeline", "cognify_pipeline")
            }
            conn.execute(
                "INSERT INTO data VALUES (?,?,?,?)",
                (
                    str(i),
                    "12345678123456781234567812345678",
                    raw.resolve().as_uri(),
                    json.dumps(status),
                ),
            )
    return root, db


def test_persisted_native_coverage_matches_trusted_input_and_completed_processing(tmp_path):
    from gluomem_bench.bench.cognee_campaign import CHUNKS_CONDITION, verify_native_coverage
    from gluomem_bench.bench.comparison_backends import Evidence

    chunks = [
        Evidence("a", "Synthetic exact text\nline"),
        Evidence("b", "Synthetic exact text\nline"),
    ]
    root, _ = native_store(tmp_path, chunks)
    coverage = verify_native_coverage(root, dataset="synthetic", expected=chunks, quiescent=True)
    assert coverage["input_count"] == 2 and coverage["native_data_count"] == 1
    assert coverage["processed_input_ids"] == ["a", "b"]
    assert coverage["condition"] == CHUNKS_CONDITION
    assert coverage["graph_retrieval_qualified"] is False


def test_actual_schema_token_count_expands_reserve_before_dispatch(tmp_path):
    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge

    seen, leased = [], []
    base = upstream(seen)

    def handler(request):
        if request.url.path == "/tokenize" and "prompt" in json.loads(request.content):
            return httpx.Response(200, json={"count": 4096})
        return base.handle_request(request)

    class Budget(WorkloadBudget):
        @contextmanager
        def lease(self, tokens):
            leased.append(tokens)
            yield

    with CogneeGenerationBridge(
        profile(),
        budget=Budget(),
        receipt=tmp_path / "wire",
        transport=httpx.MockTransport(handler),
    ) as bridge:
        bridge.complete(json.dumps(payload()).encode())
    assert leased == [12 + 64 + 4096]


def test_dispatch_receipt_failure_charges_but_prevents_generation(tmp_path):
    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge

    seen = []
    budget = WorkloadBudget()

    class BrokenReceipt(CogneeGenerationBridge):
        def _append(self, row):
            if row["kind"] == "dispatch-intent":
                raise OSError("disk full")
            return super()._append(row)

    with BrokenReceipt(
        profile(), budget=budget, receipt=tmp_path / "wire", transport=upstream(seen)
    ) as bridge:
        with pytest.raises(RuntimeError):
            bridge.complete(json.dumps(payload()).encode())
    assert budget.failed and budget.calls == 1
    assert not any(path == "/v1/chat/completions" for path, _ in seen)


def test_token_lease_is_held_through_response_validation(tmp_path):
    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge

    class Budget(WorkloadBudget):
        held = False

        @contextmanager
        def lease(self, tokens):
            self.held = True
            try:
                yield
            finally:
                self.held = False

        def check(self):
            super().check()
            if self.calls:
                assert self.held, "lease released before validation"

    budget = Budget()
    with CogneeGenerationBridge(
        profile(), budget=budget, receipt=tmp_path / "wire", transport=upstream([])
    ) as bridge:
        bridge.complete(json.dumps(payload()).encode())
    assert budget.calls == 1 and not budget.failed


def test_generation_receipt_verifier_binds_raw_requests_responses_and_actual_calls(tmp_path):
    from gluomem_bench.bench.cognee_campaign import (
        CogneeGenerationBridge,
        verify_generation_evidence,
    )

    budget = WorkloadBudget()
    receipt = tmp_path / "wire"
    with CogneeGenerationBridge(
        profile(), budget=budget, receipt=receipt, transport=upstream([])
    ) as bridge:
        bridge.complete(json.dumps(payload()).encode())
    verified = verify_generation_evidence(receipt, profile=profile(), expected_calls=1)
    assert verified["calls"] == 1 and verified["prompt_tokens"] == 12
    assert verified["completion_tokens"] == 4
    with pytest.raises(ValueError):
        verify_generation_evidence(receipt, profile=profile(), expected_calls=2)


@pytest.mark.parametrize("invalid_request", [False, True])
def test_failure_latches_wrapped_shared_job_context(tmp_path, invalid_request):
    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )

    outer = CampaignBudget(max_calls=4, max_seconds=60)
    job = Job("cognee", "h", tmp_path / "job", 4, 60)
    assert outer.reserve(job)
    limiter = RequestLimiter(max_in_flight=1, max_in_flight_tokens=10000)
    context = JobContext(job, outer, limiter)
    budget = WorkloadBudget(max_calls=4, max_seconds=60, outer_budget=context)
    with CogneeGenerationBridge(
        profile(), budget=budget, receipt=tmp_path / "wire", transport=upstream([], finish="length")
    ) as bridge:
        with pytest.raises(RuntimeError):
            bridge.complete(
                json.dumps(
                    {**payload(), **({"model": "other"} if invalid_request else {})}
                ).encode()
            )
    assert budget.calls == context.calls == outer.calls == (0 if invalid_request else 1)
    assert context.failed and budget.failed
    assert limiter.snapshot() == {"in_flight": 0, "in_flight_tokens": 0}


def test_combined_admission_reads_native_store_and_latches_coverage_failure(tmp_path):
    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge, verify_native_evidence
    from gluomem_bench.bench.comparison_backends import Evidence

    chunks = [Evidence("a", "Synthetic exact input")]
    root, _ = native_store(tmp_path, chunks)
    budget = WorkloadBudget()
    with CogneeGenerationBridge(
        profile(), budget=budget, receipt=tmp_path / "wire", transport=upstream([])
    ) as bridge:
        bridge.complete(json.dumps(payload()).encode())
        proof = verify_native_evidence(
            root, dataset="synthetic", expected=chunks, bridge=bridge, quiescent=True
        )
        assert proof["input_processing_qualified"] and proof["generation"]["calls"] == 1
        (root / "data" / "0.txt").write_bytes(b"corrupt")
        with pytest.raises(ValueError):
            verify_native_evidence(
                root, dataset="synthetic", expected=chunks, bridge=bridge, quiescent=True
            )
        with pytest.raises(RuntimeError, match="latched"):
            bridge.check()
    assert budget.failed and budget.calls == 1


def test_synthetic_driver_preserves_failure_receipt_without_calling_qwen(tmp_path):
    import asyncio

    from gluomem_bench.bench.cognee_campaign import qualify_native_synthetic
    from gluomem_bench.bench.comparison_backends import CogneeSettings

    out = tmp_path / "qualification"
    settings = CogneeSettings(
        python_executable=str(tmp_path / "missing-python.exe"),
        root=out / "native",
        dataset="synthetic",
        llm_url="http://unused.example.invalid/v1",
        llm_model="served/qwen",
        llm_api_key="local",
        embedding_url="http://127.0.0.1:8766/v1",
        embedding_model="google/embeddinggemma-2",
        embedding_dimensions=768,
        embedding_api_key="local",
        max_completion_tokens=64,
    )
    budget = WorkloadBudget(max_calls=16)
    with pytest.raises(RuntimeError):
        asyncio.run(qualify_native_synthetic(settings, profile=profile(), budget=budget, out=out))
    receipt = json.loads((out / "qualification.json").read_bytes())
    assert receipt["status"] == "failed" and receipt["calls"] == budget.calls == 0
    assert receipt["synthetic_only"] is True and budget.failed
    assert not receipt["complete_campaign_admission"]


def test_embedding_role_evidence_uses_exact_text_hash_and_single_item_batch():
    import hashlib

    from gluomem_bench.bench.cognee_campaign import verify_embedding_role_evidence

    document, query = "Synthetic\nintact text", "Which synthetic object?"
    recipe = {
        "model": "google/embeddinggemma-2",
        "revision": "qualified-by-operator",
        "dim": 768,
        "precision": "bfloat16",
        "max_tokens": 8192,
        "text_only": True,
    }

    def row(text, role):
        return {
            "input_sha256": hashlib.sha256(
                json.dumps([text], ensure_ascii=False).encode()
            ).hexdigest(),
            "role": role,
            "items": 1,
            "token_counts": [7],
            "recipe": recipe,
            "complete": True,
        }

    rows = [row(document, "document"), row(query, "query")]
    result = verify_embedding_role_evidence(rows, documents=[document], query=query, recipe=recipe)
    assert result["wire_roles_qualified"] and not result["vector_parity_qualified"]
    rows[1]["role"] = "document"
    with pytest.raises(ValueError):
        verify_embedding_role_evidence(rows, documents=[document], query=query, recipe=recipe)


@pytest.mark.parametrize(
    "corruption", ["unprocessed", "missing", "extra", "input-order", "wrong-dataset"]
)
def test_native_coverage_never_accepts_add_receipt_alone(tmp_path, corruption):
    import sqlite3

    from gluomem_bench.bench.cognee_campaign import verify_native_coverage
    from gluomem_bench.bench.comparison_backends import Evidence

    chunks = [Evidence("a", "first text"), Evidence("b", "second text")]
    root, db = native_store(tmp_path, chunks)
    with sqlite3.connect(db) as conn:
        if corruption == "unprocessed":
            conn.execute("UPDATE data SET pipeline_status = '{}' WHERE id = '0'")
        elif corruption == "missing":
            conn.execute("DELETE FROM data WHERE id = '0'")
        elif corruption == "extra":
            conn.execute(
                "INSERT INTO data SELECT 'extra', dataset_id, raw_data_location, "
                "pipeline_status FROM data LIMIT 1"
            )
        elif corruption == "wrong-dataset":
            conn.execute("UPDATE datasets SET name = 'other'")
    if corruption == "input-order":
        lines = (root / "inputs.jsonl").read_bytes().splitlines()
        (root / "inputs.jsonl").write_bytes(b"\n".join(reversed(lines)) + b"\n")
    with pytest.raises(ValueError):
        verify_native_coverage(root, dataset="synthetic", expected=chunks, quiescent=True)


@pytest.mark.parametrize("phase", ["calls", "lease", "deadline"])
def test_dispatch_rechecks_shared_admission_after_tokenization(tmp_path, phase):
    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )

    seen, clock = [], [0.0]
    outer = CampaignBudget(max_calls=2, max_seconds=60, clock=lambda: clock[0])
    job = Job("cognee", "h", tmp_path / "job", 1, 30)
    assert outer.reserve(job)
    limiter = RequestLimiter(max_in_flight=1, max_in_flight_tokens=1 if phase == "lease" else 10000)
    context = JobContext(job, outer, limiter)
    if phase == "calls":
        context.begin_call()
    original = upstream(seen)

    def handle(request):
        response = original.handle_request(request)
        if phase == "deadline" and request.url.path == "/tokenize":
            clock[0] = 31.0
        return response

    with CogneeGenerationBridge(
        profile(), budget=context, receipt=tmp_path / "wire", transport=httpx.MockTransport(handle)
    ) as bridge:
        with pytest.raises(RuntimeError):
            bridge.complete(json.dumps(payload()).encode())
    assert context.failed
    assert context.calls == (1 if phase == "calls" else 0)
    assert not any(path == "/v1/chat/completions" for path, _ in seen)
    assert limiter.snapshot() == {"in_flight": 0, "in_flight_tokens": 0}


def test_dispatch_intent_is_durable_before_actual_generation(tmp_path, monkeypatch):
    import os

    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge

    synced = []
    real_fsync = os.fsync

    def fsync(fd):
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    original = upstream([])

    def handle(request):
        if request.url.path == "/v1/chat/completions":
            assert len(synced) >= 4, "model/tokenizer/schema/intent receipts not durable"
        return original.handle_request(request)

    with CogneeGenerationBridge(
        profile(),
        budget=WorkloadBudget(),
        receipt=tmp_path / "wire",
        transport=httpx.MockTransport(handle),
    ) as bridge:
        bridge.complete(json.dumps(payload()).encode())


@pytest.mark.slow
def test_real_native_guarded_synthetic_qualification_is_explicit_opt_in():
    """Operator must wait for the separate pilot to exit before setting this file.

    JSON needs explicit ``profile`` (RackProfile fields), ``settings``
    (CogneeSettings fields) and fresh external ``out``. No deployment defaults.
    """
    import asyncio
    import os
    from pathlib import Path

    from gluomem_bench.bench.cognee_campaign import qualify_native_synthetic
    from gluomem_bench.bench.comparison_backends import CogneeSettings

    location = os.environ.get("COGNEE_BRIDGE_LIVE_CONFIG")
    if not location:
        pytest.skip("live native/Qwen qualification deferred; explicit operator config required")
    config = json.loads(Path(location).read_bytes())
    try:
        selected = RackProfile.model_validate(config["profile"])
        settings = dict(config["settings"])
        settings["root"] = Path(settings["root"])
        native = CogneeSettings(**settings)
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("invalid private qualification config; details withheld") from None
    result = asyncio.run(qualify_native_synthetic(
        native, profile=selected, budget=WorkloadBudget(max_calls=16), out=Path(config["out"])))
    assert result["status"] == "synthetic-native-lifecycle-qualified"
    assert 1 <= result["calls"] <= 16
    assert not result["graph_retrieval_qualified"]
    assert not result["complete_campaign_admission"]


def test_native_schema_and_prompt_preserved_with_lease_before_dispatch(tmp_path):
    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge

    seen, events = [], []

    class Budget(WorkloadBudget):
        @contextmanager
        def lease(self, tokens):
            events.append(("lease", tokens))
            yield

        def begin_call(self):
            events.append(("call", self.calls))
            super().begin_call()

    budget = Budget(max_calls=2)
    receipt = tmp_path / "wire.jsonl"
    with CogneeGenerationBridge(
        profile(), budget=budget, receipt=receipt, transport=upstream(seen)
    ) as bridge:
        request = payload()
        result = bridge.complete(json.dumps(request).encode())
        assert json.loads(result)["choices"][0]["finish_reason"] == "stop"
        assert seen[-1][1]["messages"] == request["messages"]
        assert seen[-1][1]["response_format"] == request["response_format"]
        assert seen[-1][1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert events == [("lease", 12 + 64 + 2048), ("call", 0)]
    assert budget.calls == 1
    rows = [json.loads(x) for x in receipt.read_text().splitlines()]
    call = rows[-1]
    assert json.loads(base64.b64decode(call["native_request_b64"])) == payload()
    assert call["calls"] == 1 and call["status"] == "complete"
    assert "qwen.example.invalid" not in receipt.read_text()
