"""The measured retriever must render production usable evidence to the reader."""

import json
from datetime import UTC, datetime

from gluomem.client.sdk import EmbeddedMemoryClient
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.llm import ConsolidationDecision, ExtractedClaimSpec, FakeExtractors
from gluomem.storage.db import MindDB

from gluomem_bench.bench.retriever import GluoMemRetriever
from gluomem_bench.bench.suite_campaign import reader_messages
from gluomem_bench.bench.types import Sample
from gluomem_bench.llm.outputs import AnswerResult


def test_final_rendered_reader_messages_keep_complete_typed_source_time_lineage(tmp_path, cfg):
    cfg = cfg.with_overrides(
        {
            "write.source_capture_policy": "attributed-interaction",
            "search.abstain.enabled": False,
            "search.rerank.enabled": False,
            "benchmark.answer.context_k": 1,
        }
    )
    db = MindDB(tmp_path / "authored.mind", dim=384)
    store = FrameStore(db, embedder=MockEmbedder())
    client = EmbeddedMemoryClient(db, store, cfg=cfg)
    old = client.ingest_event(
        event_id="old", source_ref="chat:old", kind="fact", content="The launch owner is Morgan."
    ).frames[0]
    text = "The launch owner is Avery."
    client.extractors = FakeExtractors(
        event_claims=[[ExtractedClaimSpec(text=text, message_id="m-current", verbatim_span=text)]],
        consolidation_decisions=[ConsolidationDecision(action="SUPERSEDE", target_frame_id=old.id)],
        banner=False,
    )
    current = client.ingest_event(
        event_id="current",
        source_ref="chat:current",
        created_at="2026-10-02T09:00:00Z",
        messages=[
            {
                "message_id": "m-current",
                "role": "user",
                "content": text,
                "timestamp": "2026-10-02T08:59:00Z",
            }
        ],
    ).frames[0]
    detail = "Their shortlist excludes weekends.\nOnly weekdays."
    client.extractors = FakeExtractors(event_claims=[[ExtractedClaimSpec(
        text=detail, message_id="m-detail", verbatim_span=detail,
        event_date=datetime(2026, 10, 3, tzinfo=UTC),
    )]], consolidation_decisions=[
        ConsolidationDecision(action="REFINE", target_frame_id=current.id)],
        banner=False)
    delta = client.ingest_event(
        event_id="detail", source_ref="chat:detail", created_at="2026-10-04T09:00:00Z",
        messages=[{"message_id": "m-detail", "role": "user", "content": detail,
                   "timestamp": "2026-10-04T08:59:00Z"}],
    ).frames[0]
    assert delta.frame_type == "P" and delta.base_frame_id == current.id
    db.close()
    rendered = []

    def reader(question, context):
        rendered.extend(reader_messages(question, context))
        return AnswerResult(text="authored reader result", prompt_tokens=0, completion_tokens=0)

    retriever = GluoMemRetriever(
        cfg, db_dir=tmp_path, answer_fn=reader, content_to_dias={}, embedder=MockEmbedder()
    )
    prediction = retriever.answer(
        Sample(
            id="authored:qa:0",
            conv_id="authored",
            question="shortlist",
            gold="not used",
            category=0,
            category_name="authored",
            question_date="2026-11-01",
        )
    )
    assert not prediction.abstained
    assert {doc.frame_id for doc in prediction.retrieved} == {current.id, delta.id}
    wire = json.loads(rendered[-1]["content"])
    assert wire["question"].startswith("Current Date: 2026-11-01")
    assert len(wire["evidence"].splitlines()) == 1, "embedded newlines must not split a group"
    group = json.loads(wire["evidence"].removeprefix("[evidence] "))
    assert group["frame_ids"] == [current.id, delta.id]
    frames = {frame["id"]: frame for frame in group["frames"]}
    assert frames[delta.id]["frame_type"] == "P"
    assert frames[delta.id]["base_frame_id"] == current.id
    assert frames[delta.id]["event_time"] == "2026-10-03T00:00:00"
    assert frames[current.id]["supersedes"] == old.id
    assert frames[current.id]["status"] == "active"
    provenance = frames[current.id]["provenance"][0]
    assert provenance["source_ref"] == "chat:current"
    assert provenance["message_id"] == "m-current"
    assert provenance["message_role"] == "user"
    assert provenance["message_time"] == "2026-10-02T08:59:00Z"
    assert provenance["observed_at"] == "2026-10-02T08:59:00Z"
    assert "The launch owner is Morgan." not in wire["evidence"]
    assert retriever.manifest_extra()["evidence_contract"]["surface"] == "HybridSearch.recall"
    for stack in retriever._stacks.values():
        stack._db.close()


def test_strict_assistant_attribution_reaches_reader_after_envelope_redaction(tmp_path, cfg):
    cfg = cfg.with_overrides(
        {
            "write.source_capture_policy": "attributed-interaction",
            "search.abstain.enabled": False,
            "search.rerank.enabled": False,
            "benchmark.answer.context_k": 1,
        }
    )
    quote = "I recommended the blue trail."
    claim = "The assistant recommended the blue trail."
    db = MindDB(tmp_path / "assistant.mind", dim=384)
    store = FrameStore(db, embedder=MockEmbedder())
    client = EmbeddedMemoryClient(
        db,
        store,
        cfg=cfg,
        extractors=FakeExtractors(
            event_claims=[[ExtractedClaimSpec(text=claim, message_id="a1", verbatim_span=quote)]],
            banner=False,
        ),
    )
    created = client.ingest_event(
        event_id="assistant",
        source_ref="chat:assistant",
        created_at="2026-10-01T09:00:00Z",
        messages=[
            {
                "message_id": "a1",
                "role": "assistant",
                "content": quote,
                "timestamp": "2026-10-01T08:59:00Z",
            }
        ],
    ).frames[0]
    assert created.source == "agent_inferred"
    db.execute("UPDATE source_events SET payload_json = '{}' ")
    db.commit()
    db.close()
    rendered = []

    def reader(question, context):
        rendered.extend(reader_messages(question, context))
        return AnswerResult(text="authored reader fixture")

    retriever = GluoMemRetriever(
        cfg, db_dir=tmp_path, answer_fn=reader, content_to_dias={}, embedder=MockEmbedder()
    )
    prediction = retriever.answer(
        Sample(
            id="assistant:q",
            conv_id="assistant",
            question="blue",
            gold="not used",
            category=0,
            category_name="authored",
        )
    )
    assert [doc.frame_id for doc in prediction.retrieved] == [created.id]
    context = json.loads(rendered[-1]["content"])["evidence"]
    group = json.loads(context.removeprefix("[evidence] "))
    frame = group["frames"][0]
    assert frame["source"] == "agent_inferred"
    assert frame["content"] == claim
    provenance = frame["provenance"][0]
    assert provenance["message_role"] == "assistant"
    assert provenance["observed_at"] == "2026-10-01T08:59:00Z"
    assert provenance["verbatim_span"] is None
    assert not provenance["span_verified"]
    for stack in retriever._stacks.values():
        stack._db.close()


def test_temporal_selection_metadata_reaches_final_reader_under_host_policy(tmp_path, cfg):
    from gluomem.frames.temporal import TemporalObservation, TemporalPolicy, TemporalStore

    cfg = cfg.with_overrides({"search.abstain.enabled": False, "search.rerank.enabled": False,
                              "benchmark.answer.context_k": 1})
    db = MindDB(tmp_path / "timeline.mind", dim=384)
    store = FrameStore(db, embedder=MockEmbedder())
    new = store.create_i_frame("new", "Riley lives in Lyon.")
    old = store.create_i_frame("old", "Riley lives in Oslo.")
    temporal = TemporalStore(store)
    temporal.record_observation(old.id, TemporalObservation("2025-01-02", "2025-01-01"))
    temporal.record_transition(temporal.revision(old.id), new.id,
                               TemporalObservation("2025-05-01", "2025-06-01"))
    db.close()
    rendered = []
    def reader(question, context):
        rendered.extend(reader_messages(question, context))
        return AnswerResult(text="authored reader fixture")
    retriever = GluoMemRetriever(cfg, db_dir=tmp_path, answer_fn=reader, content_to_dias={},
        embedder=MockEmbedder(), temporal_policy=TemporalPolicy(allow_historical=True),
        as_of="2025-03-01", known_at="2025-02-01")
    sample = Sample(id="timeline:q", conv_id="timeline", question="Oslo", gold="not used",
                    category=0, category_name="authored")
    prediction = retriever.answer(sample)
    assert [doc.frame_id for doc in prediction.retrieved] == [old.id]
    context = json.loads(rendered[-1]["content"])["evidence"]
    group = json.loads(context.removeprefix("[evidence] "))
    assert group["view"] == "historical" and group["as_of"] == "2025-03-01"
    assert group["known_at"] == "2025-02-01"
    assert group["temporal"][0]["selected_frame_id"] == old.id
    assert group["temporal"][0]["effective_at"] == "2025-01-01T00:00:00.000000Z"
    assert "Lyon" not in context
    for stack in retriever._stacks.values():
        stack._db.close()
    rendered.clear()
    current = GluoMemRetriever(cfg, db_dir=tmp_path, answer_fn=reader, content_to_dias={},
                               embedder=MockEmbedder())
    prediction = current.answer(sample)
    assert [doc.frame_id for doc in prediction.retrieved] == [new.id]
    context = json.loads(rendered[-1]["content"])["evidence"]
    assert "Riley lives in Oslo." not in context
    assert '"effective_at":"2025-06-01T00:00:00.000000Z"' in context
    for stack in current._stacks.values():
        stack._db.close()
