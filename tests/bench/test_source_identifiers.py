"""Opaque model identifiers preserve native source IDs outside model input."""

from datetime import UTC, datetime

from gluomem.write.conversation import Conversation, Session, Turn


def test_native_ids_cannot_influence_formatted_extraction_input():
    from gluomem.write.events import _format_event_source

    from gluomem_bench.bench.beam_ingest import _events
    from gluomem_bench.bench.source_identifiers import ordinal_turn_ids

    clock = datetime(2026, 1, 1, tzinfo=UTC)
    first = Conversation("authored", ("user", "assistant"), [Session(8, clock, [
        Turn("user", "I use Neovim.\nKeep this newline.", "answer_target:0"),
        Turn("assistant", "Recorded.", "answer_target:1"),
    ])])
    second = Conversation("authored", first.speakers, [Session(8, clock, [
        Turn(t.speaker, t.text, f"unrelated-{i}") for i, t in enumerate(first.sessions[0].turns)
    ])])
    adapted, bindings = ordinal_turn_ids(first)
    alternative, _ = ordinal_turn_ids(second)
    assert _format_event_source(_events(adapted)[0]) == _format_event_source(
        _events(alternative)[0])
    assert [t.dia_id for t in adapted.sessions[0].turns] == ["s0t0", "s0t1"]
    assert [t.dia_id for t in first.sessions[0].turns] == ["answer_target:0", "answer_target:1"]
    assert [b.original_id for b in bindings] == ["answer_target:0", "answer_target:1"]
    assert [b.model_id for b in bindings] == ["s0t0", "s0t1"]
    assert adapted.conv_id == first.conv_id and adapted.speakers == first.speakers
    assert adapted.sessions[0].index == 8 and adapted.sessions[0].date_time == clock
    assert [(t.speaker, t.text) for t in adapted.sessions[0].turns] == [
        (t.speaker, t.text) for t in first.sessions[0].turns]
    assert adapted is not first and adapted.sessions[0] is not first.sessions[0]


def test_coordinate_mapping_preserves_duplicate_missing_ids_and_literal_text():
    from gluomem_bench.bench.source_identifiers import ordinal_turn_ids

    clock = datetime(2026, 1, 1, tzinfo=UTC)
    original = Conversation("authored", ("user", "assistant"), [
        Session(4, clock, [Turn("user", "Keep answer_counter verbatim.", "duplicate")]),
        Session(9, clock, [Turn("tool", "Observed result.", "duplicate"),
                           Turn("assistant", "A suggestion.", None)]),
    ])
    adapted, bindings = ordinal_turn_ids(original)
    assert [b.model_id for b in bindings] == ["s0t0", "s1t0", "s1t1"]
    assert [b.original_id for b in bindings] == ["duplicate", "duplicate", None]
    assert [(b.session_ordinal, b.turn_ordinal) for b in bindings] == [(0, 0), (1, 0), (1, 1)]
    assert [s.index for s in adapted.sessions] == [4, 9]
    assert [(t.speaker, t.text) for s in adapted.sessions for t in s.turns] == [
        (t.speaker, t.text) for s in original.sessions for t in s.turns]


def test_driver_applies_bridge_only_to_explicit_gluomem_policy():
    from types import SimpleNamespace

    from gluomem_bench.bench.comprehensive_driver import _model_input_history

    clock = datetime(2026, 1, 1, tzinfo=UTC)
    conv = Conversation("authored", ("user", "assistant"), [
        Session(0, clock, [Turn("user", "An observation.", "answer_native:0")])])
    adapter = SimpleNamespace(name="authored", sha="digest", path="authored-path",
                              verify=lambda: None, load=lambda: [], conversations=lambda: [conv])
    policy = "opaque-ordinal-turns@v1"
    for system in ("no_memory", "bm25", "dense", "hybrid", "full_context", "mem0"):
        actual, original, bindings = _model_input_history(adapter, conv, system, policy)
        assert actual is adapter and original is conv and bindings == ()
    actual, adapted, bindings = _model_input_history(adapter, conv, "gluomem", policy)
    assert actual.conversations() == [adapted]
    assert actual.load is adapter.load and actual.verify is adapter.verify
    assert actual.name == adapter.name and actual.sha == adapter.sha and actual.path == adapter.path
    assert adapted.sessions[0].turns[0].dia_id == "s0t0"
    assert bindings[0].original_id == "answer_native:0"
