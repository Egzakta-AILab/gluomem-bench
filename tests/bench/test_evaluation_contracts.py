"""Authored bounded-evaluator regression cases; no retained conversations."""

import json
from copy import deepcopy
from hashlib import sha256

import pytest

import gluomem_bench.evaluation as q
from gluomem_bench.evaluation import bind_origin_checks

SCOPE = {"tenant_id": "fixture", "principal_id": "beacon", "namespace": "contracts"}
OLD = "The current beacon code is C1."
NEW = "The current beacon code is C2."
QUOTE = "My earlier statement that it was C1 was mistaken."
MESSAGE = NEW + " " + QUOTE


def digest(raw):
    return sha256(raw).hexdigest()


def native_hash(text):
    return digest(" ".join(text.split()).encode("utf8"))


def sources():
    return [
        {
            "event_id": eid,
            "revision": 1,
            "source_ref": "fixture/beacon/" + eid,
            "scope": SCOPE,
            "source_type": "chat_turn",
            "created_at": "2026-01-03T00:00:00Z",
            "messages": [
                {
                    "message_id": "m-" + eid,
                    "role": role,
                    "content": text,
                    "timestamp": "2026-01-03T00:00:00Z",
                }
            ],
        }
        for eid, role, text in [
            ("original", "user", OLD),
            ("correction", "user", MESSAGE),
            ("other", "assistant", "The assistant recommends the violet ferry."),
        ]
    ]


def source_bytes():
    # Independent authored source input, never reconstructed from snapshot data.
    return (json.dumps(sources(), indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf8"
    )


def declaration():
    start = len((NEW + " ").encode("utf8"))
    return {
        "event_id": "correction",
        "revision": 1,
        "source_ref": "fixture/beacon/correction",
        "message_id": "m-correction",
        "message_role": "user",
        "message_utf8_sha256": digest(MESSAGE.encode("utf8")),
        "span_start_utf8": start,
        "span_end_utf8": len(MESSAGE.encode("utf8")),
        "span_utf8_sha256": digest(QUOTE.encode("utf8")),
    }


def frame(fid, text, source, *, status="active", supersedes=None, superseded_by=None):
    return {
        "id": fid,
        **SCOPE,
        "scope": dict(SCOPE),
        "content": text,
        "content_hash": native_hash(text),
        "frame_type": "I",
        "status": status,
        "trust": "normal",
        "source": source,
        "supersedes": supersedes,
        "superseded_by": superseded_by,
    }


def provenance(pid, fid, eid, quote):
    return {
        "id": pid,
        "frame_id": fid,
        **SCOPE,
        "source_event_id": 1 if eid == "original" else 2,
        "source_ref": "fixture/beacon/" + eid,
        "message_id": "m-" + eid,
        "message_role": "user",
        "source_type": "chat_turn",
        "span": json.dumps({"verbatim_span": quote}),
        "verbatim_span": quote,
        "span_verified": True,
    }


def fixture(with_report=True):
    check = {
        "id": "authored-correction",
        "query": "What is the current beacon code?",
        "contains": ["C2"],
        "accepted_values": ["C2"],
        "answer_contract": "canonical-value-only-v1",
        "forbidden_current_plaintext": ["C1"],
        "forbidden_current_plaintext_scope": "eligible-frame-content",
        "state_contract": "source-bound-state-v1",
        "state_predicates": {
            "correction_retirement": {
                "scope": dict(SCOPE),
                "target": {
                    "source_ref": "fixture/beacon/original",
                    "message_id": "m-original",
                    "frame_type": "I",
                },
                "replacement": {
                    "source_ref": "fixture/beacon/correction",
                    "message_id": "m-correction",
                    "frame_type": "I",
                    "contains": ["C2"],
                },
            }
        },
        "state_origin_contract": "source-materialization-binding-v1",
        "state_origin_bindings": {"correction_retirement": {"target": {"contains": ["C1"]}}},
        "correction_context_policy": {
            "mode": "source-bound-verbatim-report-v1",
            "reports": [declaration()],
        },
    }
    original = frame(41, OLD, "user")
    pold = provenance(1, 41, "original", OLD)
    receipt = {
        "source_event_row_id": 1,
        "deduped_event": False,
        "frames": [{**original, "provenance": [pold]}],
    }
    checks, bindings = bind_origin_checks([check], [receipt])
    old = frame(41, "[forgotten]", "user", status="superseded", superseded_by=7)
    old["content_hash"] = native_hash(OLD)
    new = frame(7, NEW, "user", supersedes=41)
    report = frame(19, QUOTE, "user")
    pnew = provenance(2, 7, "correction", NEW)
    prep = provenance(3, 19, "correction", QUOTE)
    rows = [old, new] + ([report] if with_report else [])
    prov = [pold, pnew] + ([prep] if with_report else [])
    snapshot = {
        "memory_frames": rows,
        "memory_provenance": prov,
        "memory_frame_lineage": [
            {"source_frame_id": 41, "derived_frame_id": 7, "operation": "supersede"}
        ],
        "source_events": [
            {
                "id": i + 1,
                **SCOPE,
                "event_id": ev["event_id"],
                "revision": 1,
                "source_ref": ev["source_ref"],
                "source_type": "chat_turn",
                "payload_json": json.dumps(ev),
                "payload_hash": digest(json.dumps(ev).encode()),
            }
            for i, ev in enumerate(sources())
        ],
        "source_retractions": [],
    }
    hits = [
        {
            "frame": {
                **f,
                "scope": dict(SCOPE),
                "provenance": [p for p in prov if p["frame_id"] == f["id"]],
            }
        }
        for f in rows
        if f["status"] == "active"
    ]
    result = {
        "id": check["id"],
        "answer": "C2",
        "evidence": {"context": "\n".join(h["frame"]["content"] for h in hits), "hits": hits},
    }
    return checks[0], result, snapshot, bindings


def grade(check, result, snapshot):
    raw = source_bytes()
    authority = q.bind_report_sources(
        [check], source_bytes=raw, source_sha256=digest(raw), scope=SCOPE
    )
    return q.evaluate([result], [check], snapshot, correction_sources=authority)[0]


@pytest.mark.parametrize("with_report", [True, False])
def test_source_bound_report_is_bounded_not_semantic_absence(with_report):
    c, r, s, _ = fixture(with_report)
    row = grade(c, r, s)
    assert row["passed"]
    assert row["semantic_stale_positive_absence"] == "not_measured"
    assert row["eligible_payload_old_literal"] is with_report
    assert row["predicate_diagnostics"]["report_surface_policy"]["replacement_source_evidence"]


def _payload(result, snapshot, text):
    snapshot["memory_frames"][-1]["content"] = text
    result["evidence"]["hits"][-1]["frame"]["content"] = text
    result["evidence"]["context"] = NEW + "\n" + text


@pytest.mark.parametrize(
    "text",
    [
        "The current beacon code remains C1.",
        "The earlier C1 code report was an error.",
        QUOTE + " The setting remains C1.",
    ],
)
def test_same_source_stale_assertion_paraphrase_and_added_clause_rejected(text):
    c, r, s, _ = fixture()
    _payload(r, s, text)
    assert not grade(c, r, s)["passed"]


def test_no_literal_paraphrase_is_explicitly_not_measured():
    c, r, s, _ = fixture()
    _payload(r, s, "The originally reported setting remains in force.")
    row = grade(c, r, s)
    assert row["passed"] and row["semantic_stale_positive_absence"] == "not_measured"


@pytest.mark.parametrize("act", ["direct_assertion", "unknown", "report"])
def test_model_act_labels_do_not_grant_authority(act):
    c, r, s, _ = fixture()
    s["memory_frames"][-1]["assertion_act"] = act
    r["evidence"]["hits"][-1]["frame"]["assertion_act"] = act
    assert grade(c, r, s)["passed"]


def test_intended_origin_storewide_retirement_and_typed_lineage():
    for mutation in ("hash", "active", "clone", "lineage", "reciprocal"):
        c, r, s, _ = fixture()
        if mutation == "hash":
            s["memory_frames"][0]["content_hash"] = "f" * 64
        elif mutation == "active":
            s["memory_frames"][0]["status"] = "active"
        elif mutation == "clone":
            s["memory_frames"].append(frame(99, OLD, "user"))
        elif mutation == "lineage":
            s["memory_frame_lineage"] = []
        else:
            s["memory_frames"][0]["superseded_by"] = 99
        row = grade(c, r, s)
        assert not row["passed"], mutation
        assert not row["predicate_diagnostics"]["correction_retirement"]["passed"], mutation


@pytest.mark.parametrize(
    "where,field,value",
    [
        ("source", "revision", True),
        ("source", "revision", 1.0),
        ("snapshot", "id", 19.0),
        ("snapshot", "id", True),
        ("hit", "id", 19.0),
        ("hit", "id", None),
        ("provenance", "id", None),
        ("provenance", "source_event_id", 2.0),
        ("provenance", "frame_id", 19.0),
        ("provenance", "id", 3.0),
        ("returned", "id", 3.0),
        ("returned", "source_event_id", None),
        ("edge", "source_frame_id", 41.0),
        ("replacement", "supersedes", 41.0),
    ],
)
def test_observation_identity_joins_reject_bool_float_and_missing(where, field, value):
    c, r, s, _ = deepcopy(fixture())
    targets = {
        "source": s["source_events"][1],
        "snapshot": s["memory_frames"][-1],
        "hit": r["evidence"]["hits"][-1]["frame"],
        "provenance": s["memory_provenance"][-1],
        "returned": r["evidence"]["hits"][-1]["frame"]["provenance"][0],
        "edge": s["memory_frame_lineage"][0],
        "replacement": s["memory_frames"][1],
    }
    targets[where][field] = value
    row = grade(c, r, s)
    assert not row["passed"]
    assert "identity_failure" in row["predicate_diagnostics"]["report_surface_policy"]


@pytest.mark.parametrize(
    "mutation",
    ["role", "scope", "quote", "verified", "retraction", "payload", "short-whole-witness"],
)
def test_report_requires_committed_whole_verbatim_source_witness(mutation):
    c, r, s, _ = deepcopy(fixture())
    hit = r["evidence"]["hits"][-1]["frame"]
    if mutation == "role":
        hit["provenance"][0]["message_role"] = "assistant"
    elif mutation == "scope":
        s["memory_frames"][-1]["namespace"] = "other"
    elif mutation == "quote":
        hit["provenance"][0]["verbatim_span"] = "C1"
    elif mutation == "verified":
        hit["provenance"][0]["span_verified"] = "true"
    elif mutation == "retraction":
        s["source_retractions"].append({**SCOPE, "source_ref": "fixture/beacon/correction"})
    elif mutation == "payload":
        s["source_events"][1]["payload_json"] = "{}"
    else:
        # A genuine short quote cannot witness the entire allowed report payload.
        s["memory_provenance"][-1]["verbatim_span"] = "C1"
        hit["provenance"][0]["verbatim_span"] = "C1"
    assert not grade(c, r, s)["passed"]


@pytest.mark.parametrize("value", [True, 1.0, 0, None, "1"])
def test_source_and_replay_revisions_are_typed(value):
    c, _, _, _ = fixture()
    events = sources()
    events[1]["revision"] = value
    bad = json.dumps(events).encode()
    with pytest.raises(ValueError, match="invalid-correction-source-authority"):
        q.bind_report_sources([c], source_bytes=bad, source_sha256=digest(bad), scope=SCOPE)
    raw = source_bytes()
    authority = q.bind_report_sources([c], source_bytes=raw, source_sha256=digest(raw), scope=SCOPE)
    with pytest.raises(ValueError, match="invalid-correction-source-projection"):
        q.validate_report_projection([c], authority, events)


def test_source_omission_default_and_created_at_only_projection():
    c, _, _, _ = fixture()
    events = sources()
    for event in events:
        event.pop("revision")
    raw = json.dumps(events).encode()
    effective = deepcopy(events)
    for event in effective:
        event["created_at"] = "2026-11-01T00:00:00Z"
    authority = q.bind_report_sources(
        [c], source_bytes=raw, source_sha256=digest(raw), scope=SCOPE, effective_events=effective
    )
    for event in effective:
        event["revision"] = 1
    q.validate_report_projection([c], authority, effective)


@pytest.mark.parametrize("created_at", ["absent", None, "2026-01-03T00:00:00Z"])
@pytest.mark.parametrize("override", [False, True])
def test_effective_source_preserves_optional_clock_and_rejects_mutation(created_at, override):
    c, _, _, _ = fixture()
    events = sources()
    for event in events:
        if created_at == "absent":
            event.pop("created_at")
        else:
            event["created_at"] = created_at
    raw = json.dumps(events).encode()
    effective = deepcopy(events)
    if override:
        for event in effective:
            event["created_at"] = "2026-11-01T00:00:00Z"
    before = deepcopy(effective)

    def bind():
        return q.bind_report_sources(
            [c],
            source_bytes=raw,
            source_sha256=digest(raw),
            scope=SCOPE,
            effective_events=effective,
        )

    assert bind() is not None
    assert effective == before
    effective[1]["messages"][0]["content"] += " altered"
    with pytest.raises(ValueError, match="invalid-correction-source-authority"):
        bind()


def test_source_pin_policy_and_forged_authority_refuse_before_result_lookup():
    c, _, s, _ = fixture()
    with pytest.raises(q.Refusal, match="invalid-correction-source-authority"):
        q.evaluate([], [c], s, correction_sources={"entries": {}})
    with pytest.raises(ValueError, match="invalid-correction-source-authority"):
        q.bind_report_sources([c], source_bytes=source_bytes(), source_sha256="f" * 64, scope=SCOPE)
    c["correction_context_policy"]["mode"] = "anything-goes"
    with pytest.raises(q.Refusal, match="invalid-correction-context-policy"):
        q.evaluate([], [c], s)


def test_literal_hard_forget_does_not_acquire_report_exception():
    c, r, s, _ = fixture()
    c["forbidden_current_plaintext_scope"] = "full-reader-context"
    with pytest.raises(q.Refusal, match="invalid-correction-context-policy"):
        q.evaluate([r], [c], s)
    for key in ("correction_context_policy", "state_contract", "state_predicates"):
        c.pop(key)
    assert not q.evaluate([r], [c], s)[0]["passed"]


@pytest.mark.parametrize("answer", ["not C2", "C2 (not C1)", "C2 with explanation"])
def test_canonical_answer_has_no_guessed_normalizer(answer):
    c, r, s, _ = fixture()
    r["answer"] = answer
    assert not grade(c, r, s)["passed"]


def test_independently_pinned_alias_supplement_is_evaluator_only():
    c, r, s, _ = fixture()
    originals = {"sources.json": source_bytes(), "checks.json": json.dumps([c]).encode()}
    supplement = {
        "protocol": "q30-prospective-evaluator-only@v1",
        "original_inputs": {key: digest(raw) for key, raw in originals.items()},
        "histories": {"authored": {c["id"]: {"accepted_values": ["C2", "C2 code"]}}},
        "label_catalog": {},
        "authorship": "source-authored before responses",
        "semantic_stale_positive_absence": "not_measured",
    }
    raw = json.dumps(supplement).encode()

    def compose(data=raw, inputs=originals):
        return q.apply_supplement(
            [c],
            supplement_bytes=data,
            supplement_sha256=digest(data),
            history_id="authored",
            original_inputs=inputs,
        )

    composed = compose()
    assert c["accepted_values"] == ["C2"]
    r["answer"] = "C2 code"
    assert grade(composed[0], r, s)["passed"]
    assert not grade(c, r, s)["passed"]
    with pytest.raises(q.Refusal, match="evaluator-original-input-pin-drift"):
        compose(inputs={**originals, "sources.json": b"[]"})
    supplement["histories"]["authored"][c["id"]]["query"] = "leaked gold"
    with pytest.raises(q.Refusal, match="invalid-evaluator-supplement-change"):
        compose(data=json.dumps(supplement).encode())


def test_public_module_entrypoint_executes_independent_files(tmp_path, capsys):
    from gluomem_bench.evaluation.__main__ import main

    c, r, s, _ = fixture()
    (tmp_path / "sources.json").write_bytes(source_bytes())
    path = tmp_path / "case.json"
    path.write_text(
        json.dumps(
            {
                "checks": [c],
                "results": [r],
                "snapshot": s,
                "source": {
                    "path": "sources.json",
                    "sha256": digest(source_bytes()),
                    "scope": SCOPE,
                },
            }
        )
    )
    assert main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out)[0]["passed"]


@pytest.mark.parametrize("fault", [None, "unrelated-edge", "dates", "query", "selected", "source"])
def test_temporal_selection_requires_intended_path_dates_and_selected_evidence(fault):
    c, r, s, _ = fixture(False)
    scope = dict(SCOPE)
    previous = frame(41, OLD, "user")
    following = frame(7, NEW, "user")
    old_time = {"effective_at": "2025-01-01T00:00:00Z", "observed_at": "2025-01-02T00:00:00Z"}
    new_time = {"effective_at": "2025-06-01T00:00:00Z", "observed_at": "2025-06-02T00:00:00Z"}
    check = {
        "id": "authored-time",
        "contains": ["C1"],
        "accepted_values": ["C1"],
        "answer_contract": "canonical-value-only-v1",
        "as_of": "2025-03-01T00:00:00Z",
        "known_at": "2025-03-01T00:00:00Z",
        "state_contract": "source-bound-state-v1",
        "state_predicates": {
            "temporal_selection": {
                "scope": scope,
                "previous": c["state_predicates"]["correction_retirement"]["target"],
                "next": c["state_predicates"]["correction_retirement"]["replacement"],
                "previous_time": old_time,
                "next_time": new_time,
                "selected": "previous",
            }
        },
    }
    s["memory_frames"] = [previous, following]
    s["memory_frame_temporal"] = [
        {
            **scope,
            "frame_id": 41,
            "operation": "observation",
            "previous_frame_id": None,
            **old_time,
        },
        {**scope, "frame_id": 7, "operation": "transition", "previous_frame_id": 41, **new_time},
    ]
    p = provenance(1, 41, "original", OLD)
    evidence = {
        "context": OLD,
        "hits": [{"frame": {**previous, "provenance": [p]}}],
        "as_of": check["as_of"],
        "known_at": check["known_at"],
        "groups": [
            {
                "root_frame_id": 41,
                "frame_ids": [41],
                "view": "historical",
                "as_of": check["as_of"],
                "known_at": check["known_at"],
                "temporal": [
                    {
                        "path": [41, 7],
                        "requested_frame_id": 7,
                        "selected_frame_id": 41,
                        "operation": "observation",
                        "previous_frame_id": None,
                        **old_time,
                    }
                ],
            }
        ],
    }
    if fault == "unrelated-edge":
        s["memory_frame_temporal"][-1]["previous_frame_id"] = 99
    elif fault == "dates":
        s["memory_frame_temporal"][0]["observed_at"] = "2025-02-01T00:00:00Z"
    elif fault == "query":
        evidence["as_of"] = "2025-07-01T00:00:00Z"
    elif fault == "selected":
        evidence["groups"][0]["temporal"][0]["selected_frame_id"] = 7
    elif fault == "source":
        evidence["hits"][0]["frame"]["provenance"][0]["span_verified"] = False
    row = q.evaluate([{"id": check["id"], "answer": "C1", "evidence": evidence}], [check], s)[0]
    assert row["passed"] is (fault is None)
    assert row["predicate_diagnostics"]["temporal_selection"]["passed"] is (fault is None)


@pytest.mark.parametrize("fault", ["hash", "range", "utf8", "duplicate", "bool-revision"])
def test_report_declarations_cannot_change_hash_ranges_or_utf8_boundaries(fault):
    c, _, _, _ = fixture()
    raw = source_bytes()
    declaration = c["correction_context_policy"]["reports"][0]
    if fault == "hash":
        declaration["message_utf8_sha256"] = "a" * 64
    elif fault == "range":
        declaration["span_start_utf8"] = 1
    elif fault == "duplicate":
        c["correction_context_policy"]["reports"].append(deepcopy(declaration))
    elif fault == "bool-revision":
        declaration["revision"] = True
    else:
        events = sources()
        events[1]["messages"][0]["content"] = "É " + MESSAGE
        text = events[1]["messages"][0]["content"].encode()
        declaration.update(
            message_utf8_sha256=digest(text),
            span_start_utf8=1,
            span_end_utf8=len(text),
            span_utf8_sha256=digest(text[1:]),
        )
        raw = json.dumps(events, ensure_ascii=False).encode()
    with pytest.raises(ValueError, match="invalid-correction"):
        q.bind_report_sources([c], source_bytes=raw, source_sha256=digest(raw), scope=SCOPE)
