"""Offline contracts and real pinned-package integration (no live Qwen)."""

import asyncio
import importlib
import json
from pathlib import Path

import pytest

from gluomem_bench.bench.comparison_backends import CogneeSettings, Evidence


def graph():
    return importlib.import_module("gluomem_bench.bench.cognee_graph")


@pytest.fixture
def native_python():
    import os

    path = Path(
        os.environ.get(
            "COGNEE_TEST_PYTHON",
            str(
                Path(__file__).resolve().parents[2]
                / 'tests/fixtures/native/operator-supplied'
                / "cognee/venv/Scripts/python.exe"
            ),
        )
    )
    if not path.is_file():
        pytest.skip("requires isolated cognee==1.6.5 with lancedb, ladybug, litellm, tiktoken")
    return path


def test_real_native_graph_lifecycle_network_only_fixture(tmp_path, native_python, source_text=None):
    """Author graph through real add/cognify, not direct graph writes or fake retrievers."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import httpx

    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge, verify_native_evidence
    from gluomem_bench.bench.rack import RackProfile
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    seen, embeddings = [], []
    fixture = {
        "nodes": [
            {
                "id": "rhea",
                "name": "Rhea",
                "type": "Person",
                "description": "Rhea owns a cobalt telescope.",
            },
            {
                "id": "telescope",
                "name": "cobalt telescope",
                "type": "Object",
                "description": "Rhea's cobalt telescope.",
            },
        ],
        "edges": [
            {
                "source_node_id": "rhea",
                "target_node_id": "telescope",
                "relationship_name": "owns",
                "description": "Rhea owns a cobalt telescope.",
            }
        ],
    }

    def upstream(request):
        body = json.loads(request.content) if request.content else None
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "served/qwen"}]})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 32})
        seen.append(body)
        schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
        if body.get("response_format") == {"type": "json_object"}:
            system = body["messages"][0]["content"]
            schema = json.loads(system.split("```json\n", 1)[1].split("```", 1)[0])
        props = schema.get("properties", {})
        if (
            not body.get("response_format")
            and 'following string: "test"' in body["messages"][0]["content"]
        ):
            result = "test"
        elif "nodes" in props and "edges" in props:
            result = fixture
        elif "summary" in props:
            result = {
                "summary": "Rhea owns a cobalt telescope.",
                "description": "Synthetic source.",
            }
        else:
            raise AssertionError("unexpected native generation schema: " + str(list(props)))
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 1,
                "model": "served/qwen",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(result)},
                    }
                ],
                "usage": {"prompt_tokens": 32, "completion_tokens": 16, "total_tokens": 48},
            },
        )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            embeddings.append(body)
            texts = body["input"]
            texts = [texts] if isinstance(texts, str) else texts
            data = {
                "object": "list",
                "model": "encoder",
                "data": [
                    {"object": "embedding", "index": i, "embedding": [0.8, 0.6, 0.0]}
                    for i, _ in enumerate(texts)
                ],
                "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
            }
            raw = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    budget = WorkloadBudget(max_calls=12, max_seconds=300)
    settings = CogneeSettings(
        python_executable=str(native_python),
        root=tmp_path / "native",
        dataset="history",
        llm_url="http://unused.invalid/v1",
        llm_model="served/qwen",
        llm_api_key="local",
        embedding_url=f"http://127.0.0.1:{server.server_port}/v1",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=3,
        embedding_api_key="local",
        timeout_seconds=180,
        max_completion_tokens=512,
    )
    chunks = [Evidence("source-1", source_text if source_text is not None else "Rhea owns a cobalt telescope.")]
    try:
        with CogneeGenerationBridge(
            RackProfile(base_url="http://offline.invalid/v1", model="served/qwen"),
            budget=budget,
            receipt=tmp_path / "generation.jsonl",
            transport=httpx.MockTransport(upstream),
        ) as bridge:
            backend = graph().CogneeGraphEvidenceBackend(bridge.bind(settings))

            async def exercise():
                try:
                    await backend.add(chunks)
                    verified = verify_native_evidence(
                        settings.root, dataset="history", expected=chunks, bridge=bridge
                    )
                    assert verified["input_processing_qualified"]
                    count = len(seen)
                    hits = await backend.search("What does Rhea own?", top_k=30)
                    assert hits and "Connections:" in hits[0].text
                    assert "cobalt telescope" in hits[0].text and "owns" in hits[0].text
                    assert any(
                        e["kind"] == "graph_edge" for e in hits[0].provenance["native_evidence"]
                    )
                    assert len(seen) == count, (
                        "baseline graph retrieval should not generate a final answer"
                    )
                    assert embeddings[-1]["input_type"] == "query"
                    if source_text is not None:
                        (tmp_path / "authored-embedding-wire.json").write_bytes((json.dumps(embeddings, ensure_ascii=False) + "\n").encode())
                    # Native DataPoint.get_embeddable_data strips field boundaries
                    # (1.6.5 DataPoint.py:250-278), AFTER exact raw-file readback above.
                    # Do not normalize the common source to emulate native internals.
                    expected_embed = chunks[0].text if source_text is None else chunks[0].text.strip()
                    assert any(
                        row.get("input_type") == "document"
                        and expected_embed
                        in (row["input"] if isinstance(row["input"], list) else [row["input"]])
                        for row in embeddings
                    ), [(r.get("input_type"), r["input"]) for r in embeddings]
                    assert all(row.get("max_tokens") == 512 for row in seen)
                    assert (await backend.source_pins())["version"] == "1.6.5"
                finally:
                    await backend.close()

            asyncio.run(exercise())
            final = verify_native_evidence(
                settings.root, dataset="history", expected=chunks, bridge=bridge, quiescent=True
            )
            assert final["coverage"]["native_data_count"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_backend_uses_graph_rpc_and_keeps_native_triplet_limit(tmp_path):
    module = graph()
    settings = CogneeSettings(
        python_executable="unused",
        root=tmp_path / "native",
        dataset="history",
        llm_url="http://example.invalid/v1",
        llm_model="served/qwen",
        llm_api_key="local",
        embedding_url="http://example.invalid/v1",
        embedding_model="encoder",
        embedding_dimensions=3,
        embedding_api_key="local",
    )
    backend = module.CogneeGraphEvidenceBackend(settings)
    calls = []

    async def rpc(operation, **payload):
        calls.append((operation, payload))
        return {
            "dataset_id": "scope",
            "dataset_name": "history",
            "context": "native",
            "evidence": [{"kind": "graph_edge", "artifact_id": "edge"}],
        }

    backend._rpc = rpc  # Transport seam only; native integration is tested separately.

    async def exercise():
        assert await backend.search(" ", top_k=2) == []
        assert await backend.search("query", top_k=0) == []
        hits = await backend.search("query", limit=9)
        assert hits[0].text == "native"
        assert calls == [("search", {"query": "query", "top_k": 9})]
        await backend.close()
        with pytest.raises(RuntimeError, match="closed"):
            await backend.search("query", top_k=1)

    asyncio.run(exercise())


def test_graph_admission_reuses_committed_input_and_generation_verifier(tmp_path):
    from test_cognee_campaign import native_store, payload, profile, upstream

    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    expected = [Evidence("source", "Synthetic text only.")]
    root, _ = native_store(tmp_path, expected)
    with CogneeGenerationBridge(
        profile(),
        budget=WorkloadBudget(max_calls=2),
        receipt=tmp_path / "wire",
        transport=upstream([]),
    ) as bridge:
        bridge.complete(json.dumps(payload()).encode())
        report = graph().verify_graph_evidence(
            root, dataset="synthetic", expected=expected, bridge=bridge
        )
        assert report["condition"] == graph().GRAPH_CONDITION
        assert report["coverage"]["processed_input_ids"] == [item.id for item in expected]
        assert report["generation"]["calls"] == 1
        assert report["complete_campaign_admission"] is False
        assert report["graph_retrieval_qualified"] is False


def test_nonempty_context_requires_native_graph_lineage():
    module = graph()
    with pytest.raises(ValueError, match="lineage"):
        module.graph_context_evidence(
            {
                "dataset_id": "scope",
                "dataset_name": "history",
                "context": "Not attributable",
                "evidence": [],
            },
            dataset="history",
        )


def test_native_context_is_evidence_not_completion_or_templated_question():
    module = graph()
    payload = {
        "dataset_id": "dataset-1",
        "dataset_name": "history",
        "context": "Rhea --owns--> cobalt telescope\nNative formatting retained.",
        "evidence": [{"kind": "graph_edge", "id": "edge-1"}],
    }
    hits = module.graph_context_evidence(payload, dataset="history")
    assert len(hits) == 1
    assert hits[0].text == payload["context"]
    assert hits[0].provenance["native_evidence"] == payload["evidence"]
    assert hits[0].provenance["dataset_id"] == "dataset-1"
    assert hits[0].score == 0
    assert module.graph_context_evidence(payload, dataset="history") == hits
    assert module.graph_context_evidence({**payload, "context": ""}, dataset="history") == []
    with pytest.raises(ValueError, match="scope"):
        module.graph_context_evidence(payload, dataset="other")
    with pytest.raises(ValueError, match="completion"):
        module.graph_context_evidence({**payload, "completion": "answer"}, dataset="history")
    structured = {**payload, "context": ["native", {"b": 2, "a": 1}]}
    assert module.graph_context_evidence(structured, dataset="history")[0].text == json.dumps(
        structured["context"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
