"""Bounded authored three-backend admission tests; no real endpoints."""

import importlib.util
import pytest


def test_exact_adapter_is_opt_in_leaving_legacy_projection_historical():
    import gluomem_bench.bench.lme_projection as projection
    from gluomem_bench.bench.adapters.longmemeval import LongMemEvalAdapter

    cls = getattr(projection, "ExactLongMemEvalAdapter", None)
    assert cls is not None, "focused runner needs an explicit exact adapter seam"
    raw = {"question_id": "authored", "haystack_dates": ["2024/02/01 (Thu) 12:00"],
           "haystack_session_ids": ["answer_annotation"],
           "haystack_sessions": [[{"role": "user", "content": "  exact\n"},
                                  {"role": "assistant", "content": ""}]]}
    legacy = object.__new__(LongMemEvalAdapter)
    exact = object.__new__(cls)
    legacy._raw_questions = exact._raw_questions = lambda: [raw]
    assert [t.text for t in legacy.conversations()[0].sessions[0].turns] == ["exact"]
    assert [t.text for t in exact.conversations()[0].sessions[0].turns] == ["  exact\n", ""]
    assert exact.projection_policy == projection.PROJECTION_POLICY
    assert cls.load is LongMemEvalAdapter.load  # Same native question_date, gold/scorer boundary.


def test_two_real_native_graph_workers_use_disjoint_stores(tmp_path, monkeypatch):
    """Real SDK add/cognify/retrieve/readback with two simultaneously resident workers.

    Calls the retained authored graph lifecycle fixture twice, not real Qwen or
    encoder endpoints. Same fixture content tests store/handle isolation, NOT
    adversarial cross-user retrieval (which remains a separate admission gate).
    """
    from pathlib import Path
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from tests.bench.test_cognee_graph import test_real_native_graph_lifecycle_network_only_fixture
    from gluomem_bench.bench.cognee_graph import CogneeGraphEvidenceBackend

    native = Path(__file__).resolve().parents[2] / 'tests/fixtures/native/operator-supplied'
    if not native.is_file():
        pytest.skip("requires installed isolated Cognee 1.6.5")
    barrier = threading.Barrier(2, timeout=120)
    pids = set()
    lock = threading.Lock()
    original = CogneeGraphEvidenceBackend._send

    def send(self, request):
        result = original(self, request)
        if request["operation"] == "add":
            with lock:
                pids.add(self._process.pid)
            barrier.wait()
        return result

    monkeypatch.setattr(CogneeGraphEvidenceBackend, "_send", send)
    roots = [tmp_path / "a", tmp_path / "b"]
    for root in roots:
        root.mkdir()
    from gluomem_bench.bench.cognee_retriever import session_documents
    from gluomem_bench.bench.lme_projection import project_history
    raw = {"question_id": "authored", "haystack_dates": ["2024/02/01 (Thu) 12:00"],
           "haystack_session_ids": ["answer_annotation"],
           "haystack_sessions": [[{"role": "assistant", "content": "  Rhea owns a cobalt telescope.\n"},
                                  {"role": "user", "content": ""}]]}
    exact_text = session_documents(project_history(raw))[0].text
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(test_real_native_graph_lifecycle_network_only_fixture, root, native, exact_text) for root in roots]
        for future in futures:
            future.result(timeout=240)
    assert len(pids) == 2
    assert all((root / "native/system/databases/cognee_db").is_file() for root in roots)
    assert all((root / "generation.jsonl").is_file() for root in roots)


@pytest.mark.parametrize("status", [400, 503])
def test_cognee_error_wire_receipt_never_persists_echoed_body(tmp_path, status):
    import base64
    import hashlib
    import json
    import httpx
    from gluomem_bench.bench.cognee_campaign import CogneeGenerationBridge
    from gluomem_bench.bench.rack import RackProfile
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    private = b"PRIVATE_BODY_AUTH_HEADER_REASON_ECHO" * 400
    bridge = CogneeGenerationBridge(
        RackProfile(base_url="http://authored.invalid/v1", model="served/qwen"),
        budget=WorkloadBudget(max_calls=4, max_seconds=30), receipt=tmp_path / "wire.jsonl",
        transport=httpx.MockTransport(lambda request: httpx.Response(status, content=private)),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            bridge._request("GET", "/models")
        material = (tmp_path / "wire.jsonl").read_bytes()
        assert private not in material
        assert base64.b64encode(private) not in material, "HTTP error bodies are private, not normal generation evidence"
        row = json.loads(material)
        assert "response_b64" not in row
        assert row["error_response"] == {
            "body_prefix_sha256": hashlib.sha256(private[:4096]).hexdigest(),
            "captured_bytes": 4096, "truncated": True,
        }
    finally:
        bridge.__exit__()


@pytest.mark.parametrize("mutation", ["role", "content", "parallel_sessions"])
def test_projection_rejects_malformed_source_without_coercion(mutation):
    from gluomem_bench.bench.lme_projection import project_history

    raw = {"question_id": "authored", "haystack_dates": ["2024/02/01 (Thu) 12:00"],
           "haystack_session_ids": ["original"],
           "haystack_sessions": [[{"role": "user", "content": "exact"}]]}
    if mutation == "parallel_sessions":
        raw["haystack_session_ids"] = []
    else:
        raw["haystack_sessions"][0][0][mutation] = None
    with pytest.raises(ValueError, match="source"):
        project_history(raw)


def test_exact_common_projection_preserves_empty_and_boundary_text():
    name = "gluomem_bench.bench.lme_projection"
    assert importlib.util.find_spec(name) is not None, "shared exact LME projection is missing"
    from gluomem_bench.bench.lme_projection import PROJECTION_POLICY, project_history
    from gluomem_bench.bench.beam_ingest import _events
    from gluomem_bench.bench.mem0 import _session_messages
    from gluomem_bench.bench.cognee_retriever import session_documents

    raw = {
        "question_id": "authored", "question": "NEVER_PROJECT_QUESTION",
        "answer": "NEVER_PROJECT_GOLD", "answer_session_ids": ["answer_late"],
        "question_date": "2024/02/03 (Sat) 12:00",
        "haystack_dates": ["2024/02/02 (Fri) 12:00", "2024/02/01 (Thu) 12:00"],
        "haystack_session_ids": ["answer_late", "irrelevant"],
        "haystack_sessions": [
            [{"role": "assistant", "content": "  cobalt telescope\n", "has_answer": True},
             {"role": "user", "content": ""}],
            [{"role": "user", "content": "old value"}],
        ],
    }
    conv = project_history(raw)
    assert PROJECTION_POLICY == "lme-exact-text-empty-retained-chronological-opaque@v1"
    assert [s.index for s in conv.sessions] == [2, 1]
    assert [t.text for s in conv.sessions for t in s.turns] == ["old value", "  cobalt telescope\n", ""]
    assert [t.dia_id for s in conv.sessions for t in s.turns] == ["s1t0", "s0t0", "s0t1"]
    events = _events(conv)
    assert [m.content for e in events for m in e.messages] == ["old value", "  cobalt telescope\n", ""]
    mem0 = [_session_messages(s) for s in conv.sessions]
    assert mem0[1][0]["content"].endswith("assistant:   cobalt telescope\n")
    assert mem0[1][1]["content"].endswith("user: ")
    docs = session_documents(conv)
    assert docs[1].text.endswith("assistant:   cobalt telescope\n\nuser: ")
    visible = repr(events) + repr(mem0) + repr(docs)
    for forbidden in ("NEVER_PROJECT_QUESTION", "NEVER_PROJECT_GOLD", "answer_late", "has_answer"):
        assert forbidden not in visible
    assert raw["haystack_sessions"][0][0]["content"] == "  cobalt telescope\n"
