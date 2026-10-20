"""Matched evidence contracts; synthetic inputs only, no model calls by default."""

import asyncio

import pytest

from gluomem_bench.bench import comparison_backends as cb


def test_bm25_retains_exact_chunks_and_source_provenance():
    async def exercise():
        backend = cb.RawEvidenceBackend("bm25")
        chunks = [
            cb.Evidence("a", "Rhea keeps a cobalt telescope.", {"session": "s1"}),
            cb.Evidence("b", "Ivo grows basil.", {"session": "s2"}),
        ]
        await backend.add(chunks)
        hits = await backend.search("cobalt telescope", top_k=1)
        assert len(hits) == 1
        assert hits[0].id == "a"
        assert hits[0].text == chunks[0].text
        assert hits[0].provenance == {"session": "s1"}
        assert hits[0].score > 0
        await backend.close()
        with pytest.raises(RuntimeError, match="closed"):
            await backend.search("cobalt")

    asyncio.run(exercise())


def test_dense_and_hybrid_share_chunks_and_encoder_cosine_geometry():
    calls = []

    def encoder(texts):
        calls.append(list(texts))
        vectors = {"scarlet": [1.0, 0.0], "red": [3.0, 0.0], "blue": [0.0, 1.0]}
        return [vectors[text] for text in texts]

    async def exercise():
        chunks = [cb.Evidence("r", "red", {"source": 1}), cb.Evidence("b", "blue")]
        for mode in ("dense", "hybrid"):
            backend = cb.RawEvidenceBackend(mode, encoder=encoder)
            await backend.add(chunks)
            hits = await backend.search("scarlet", top_k=2)
            assert [hit.id for hit in hits] == ["r", "b"]
            assert hits[0].text == "red"
            assert hits[0].provenance == {"source": 1}
            assert list(backend.chunks.values()) == chunks
            await backend.close()
        assert calls == [["red", "blue"], ["scarlet"], ["red", "blue"], ["scarlet"]]

    asyncio.run(exercise())


def test_packing_counts_rendered_tokens_and_never_truncates_or_overflows():
    items = [
        cb.Evidence("a", "one two three four"),
        cb.Evidence("b", "five"),
        cb.Evidence("c", "six"),
    ]
    packed = cb.pack_evidence(items, token_budget=5, count_tokens=lambda text: len(text.split()))
    assert [hit.id for hit in packed.items] == ["a"]
    assert packed.text == "[a]\none two three four"
    assert packed.tokens == 5
    assert (
        cb.pack_evidence(items, token_budget=0, count_tokens=lambda text: len(text.split())).text
        == ""
    )
    with pytest.raises(ValueError):
        cb.pack_evidence(items, token_budget=-1, count_tokens=len)


def test_rejects_duplicate_ids_and_malformed_embeddings_atomically():
    async def exercise():
        backend = cb.RawEvidenceBackend(
            "dense", encoder=lambda texts: [[float("nan")]] * len(texts)
        )
        with pytest.raises(ValueError, match="finite"):
            await backend.add([cb.Evidence("a", "text")])
        assert backend.chunks == {}
        raw = cb.RawEvidenceBackend("bm25")
        with pytest.raises(ValueError, match="duplicate"):
            await raw.add([cb.Evidence("a", "text"), cb.Evidence("a", "other")])
        assert raw.chunks == {}
        await raw.add([cb.Evidence("a", "text")])
        with pytest.raises(ValueError, match="duplicate"):
            await raw.add([cb.Evidence("a", "changed")])
        dense = cb.RawEvidenceBackend("dense", encoder=lambda texts: [[1.0, 0.0]] * len(texts))
        await dense.add([cb.Evidence("a", "text")])
        dense.encoder = lambda texts: [[1.0]] * len(texts)
        with pytest.raises(ValueError, match="dimension"):
            await dense.search("query")

    asyncio.run(exercise())


def test_cognee_settings_require_explicit_endpoints_and_hide_credentials(tmp_path):
    from dataclasses import replace

    settings = cb.CogneeSettings(
        python_executable="python",
        root=tmp_path / "native",
        dataset="synthetic",
        llm_url="http://llm.example.invalid/v1",
        llm_model="Qwen/Qwen3.8-27B-FP8",
        llm_api_key="secret-value",
        embedding_url="http://embedding.example.invalid/v1",
        embedding_model="operator-encoder",
        embedding_dimensions=768,
        embedding_api_key="another-secret",
    )
    assert "secret" not in repr(settings)
    assert "example.invalid" not in repr(settings)
    with pytest.raises(ValueError, match="endpoint"):
        replace(settings, llm_url="")
    with pytest.raises(ValueError, match="dimensions"):
        replace(settings, embedding_dimensions=0)
    with pytest.raises(ValueError, match="dataset"):
        replace(settings, dataset="")


def test_native_chunk_results_preserve_native_ids_payload_and_distance():
    results = [
        {
            "dataset_id": "ds",
            "dataset_name": "synthetic",
            "text_result": [
                {
                    "id": "native-1",
                    "text": "Rhea has a telescope.",
                    "score": 0.15,
                    "is_part_of": "document-1",
                }
            ],
        }
    ]
    hits = cb.cognee_chunk_evidence(results)
    assert hits[0].id == "native-1"
    assert hits[0].text == "Rhea has a telescope."
    assert hits[0].score == -0.15
    assert hits[0].provenance == {
        "backend": "cognee",
        "dataset_id": "ds",
        "dataset_name": "synthetic",
        "native_payload": results[0]["text_result"][0],
        "score_kind": "negative_distance",
    }
    with pytest.raises(RuntimeError, match="native search"):
        cb.cognee_chunk_evidence([{"error": "missing collection"}])
    with pytest.raises(ValueError, match="payload"):
        cb.cognee_chunk_evidence([{"text_result": [{"text": "lost provenance"}]}])


def _settings(tmp_path, **overrides):
    fields = dict(
        python_executable="python",
        root=tmp_path / "native",
        dataset="synthetic",
        llm_url="http://llm.example.invalid/v1",
        llm_model="Qwen/Qwen3.8-27B-FP8",
        llm_api_key="local",
        embedding_url="http://embedding.example.invalid/v1",
        embedding_model="operator-encoder",
        embedding_dimensions=768,
        embedding_api_key="local",
    )
    fields.update(overrides)
    return cb.CogneeSettings(**fields)


def test_native_facade_uses_add_cognify_search_not_raw_fallback(tmp_path):
    calls = []
    backend = cb.CogneeEvidenceBackend(_settings(tmp_path))

    async def rpc(operation, **payload):
        calls.append((operation, payload))
        if operation == "search":
            return [
                {
                    "dataset_id": "ds",
                    "dataset_name": "synthetic",
                    "text_result": [{"id": "native-a", "text": "native chunk", "score": 0.1}],
                }
            ]
        return {}

    backend._rpc = rpc

    async def exercise():
        await backend.add([cb.Evidence("input-a", "synthetic input", {"session": "s1"})])
        hits = await backend.search("synthetic", top_k=3)
        assert hits[0].id == "native-a"
        assert calls[0][0] == "add"
        assert calls[0][1]["chunks"][0]["text"] == "synthetic input"
        assert calls[1] == ("search", {"query": "synthetic", "top_k": 3})
        await backend.close()
        with pytest.raises(RuntimeError, match="closed"):
            await backend.search("query")

    asyncio.run(exercise())


def test_native_refuses_nonfresh_store_without_modifying_it(tmp_path):
    root = tmp_path / "native"
    root.mkdir()
    marker = root / "existing"
    marker.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        cb.CogneeEvidenceBackend(_settings(tmp_path))
    assert marker.read_bytes() == b"preserve"


@pytest.mark.slow
def test_real_cognee_synthetic_qualification():
    """Run only with an operator-owned external settings JSON and bounded proxy."""
    import json
    import os
    from pathlib import Path

    path = os.environ.get("COGNEE_QUALIFICATION_SETTINGS")
    if not path:
        pytest.skip("external Cognee qualification settings not supplied")
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    settings["root"] = Path(settings["root"])

    receipt_path = os.environ.get("COGNEE_QUALIFICATION_EMBEDDING_RECEIPT")
    receipt = Path(receipt_path) if receipt_path else None
    receipt_start = len(receipt.read_text().splitlines()) if receipt else 0

    async def exercise():
        backend = cb.CogneeEvidenceBackend(cb.CogneeSettings(**settings))
        try:
            await backend.add(
                [
                    cb.Evidence(
                        "synthetic-rhea",
                        "Rhea owns a cobalt telescope.",
                        {"fixture": "synthetic-only"},
                    )
                ]
            )
            hits = await backend.search("What does Rhea own?", top_k=3)
            assert hits
            assert any("cobalt telescope" in hit.text for hit in hits)
            assert all(hit.provenance["dataset_name"] == "synthetic" for hit in hits)
            if receipt:
                import hashlib

                def digest(text):
                    return hashlib.sha256(
                        json.dumps([text], ensure_ascii=False).encode()
                    ).hexdigest()

                rows = [
                    json.loads(line) for line in receipt.read_text().splitlines()[receipt_start:]
                ]
                query_rows = [
                    row for row in rows if row.get("input_sha256") == digest("What does Rhea own?")
                ]
                document_rows = [
                    row
                    for row in rows
                    if row.get("input_sha256") == digest("Rhea owns a cobalt telescope.")
                ]
                assert query_rows and all(row["role"] == "query" for row in query_rows)
                assert document_rows and all(row["role"] == "document" for row in document_rows)
        finally:
            await backend.close()

    asyncio.run(exercise())


def test_separate_query_encoder_and_limit_alias_preserve_prefix_roles():
    calls = []

    def documents(texts):
        calls.append(("document", list(texts)))
        return [[1.0, 0.0] for _ in texts]

    def query(text):
        calls.append(("query", text))
        return [1.0, 0.0]

    async def exercise():
        backend = cb.RawEvidenceBackend("dense", encoder=documents, query_encoder=query)
        await backend.add([cb.Evidence("a", "original text")])
        assert len(await backend.search("question", limit=1)) == 1
        assert calls == [("document", ["original text"]), ("query", "question")]
        with pytest.raises(ValueError, match="both"):
            await backend.search("question", limit=1, top_k=2)

    asyncio.run(exercise())


def test_native_environment_and_role_config_are_explicit(tmp_path, monkeypatch):
    from dataclasses import asdict

    settings = _settings(tmp_path)
    settings.root.mkdir()
    import os

    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setenv("LLM_MODEL", "hostile-commercial-model")
    cb._native_environment(asdict(settings))
    import os

    assert os.environ["EMBEDDING_PROVIDER"] == "openai_compatible"
    assert os.environ["LLM_QUERY_MODEL"] == settings.llm_model
    assert os.environ["LLM_EXTRACTION_ENDPOINT"] == settings.llm_url
    assert os.environ["COGNEE_TRACING_ENABLED"] == "false"
    import json

    assert json.loads(os.environ["LLM_ARGS"])["custom_llm_provider"] == "openai"
    document = cb._embedding_kwargs(asdict(settings), "document")
    query = cb._embedding_kwargs(asdict(settings), "query")
    assert document["embedding_input_type"] == "document"
    assert query["embedding_input_type"] == "query"
    assert query["embedding_endpoint"] == settings.embedding_url
    assert query["embedding_model"] == settings.embedding_model
    assert query["embedding_dimensions"] == settings.embedding_dimensions


def test_worker_dependency_gate_executes_without_stdlib_shadowing(tmp_path):
    import importlib.util
    import sys

    if importlib.util.find_spec("cognee") is not None:
        pytest.skip("dependency-negative test needs interpreter without Cognee")

    async def exercise():
        backend = cb.CogneeEvidenceBackend(_settings(tmp_path, python_executable=sys.executable))
        with pytest.raises(RuntimeError, match="PackageNotFoundError"):
            await backend.search("synthetic")
        assert backend.closed
        await backend.close()

    asyncio.run(exercise())


def test_native_role_switch_invalidates_cached_vector_adapters():
    class Factory:
        def __init__(self):
            self.clears = 0

        def cache_clear(self):
            self.clears += 1

    factory = Factory()
    cb._refresh_embedding_role(factory)
    assert factory.clears == 1


def test_native_rechecks_freshness_at_first_operation(tmp_path):
    backend = cb.CogneeEvidenceBackend(_settings(tmp_path))
    backend.settings.root.mkdir()
    marker = backend.settings.root / "late-existing"
    marker.write_bytes(b"preserve")

    async def exercise():
        with pytest.raises(FileExistsError):
            await backend.search("synthetic")
        await backend.close()

    asyncio.run(exercise())
    assert [path.name for path in backend.settings.root.iterdir()] == ["late-existing"]
    assert marker.read_bytes() == b"preserve"


def test_rrf_deduplicates_per_lane_and_breaks_ties_by_id():
    a, b = cb.Evidence("a", "A"), cb.Evidence("b", "B")
    hits = cb.reciprocal_rank_fusion([[b, b, a], [a, b]], k=60)
    assert [hit.id for hit in hits] == ["a", "b"]
    assert hits[0].score == pytest.approx(1 / 61 + 1 / 62)
    with pytest.raises(ValueError):
        cb.reciprocal_rank_fusion([[a]], k=-1)
