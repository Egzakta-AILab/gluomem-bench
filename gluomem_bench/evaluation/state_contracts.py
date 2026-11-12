"""Opt-in evaluator-only predicates. No producer imports, DB access, or model calls."""

import re
from datetime import UTC, datetime


def validate_selector(selector):
    if (
        not isinstance(selector, dict)
        or not {"source_ref", "message_id"} <= set(selector)
        or set(selector) - {"source_ref", "message_id", "frame_type", "contains", "content_hash"}
        or any(
            type(selector.get(k)) is not str or not selector[k].strip()
            for k in ("source_ref", "message_id")
        )
        or (
            "frame_type" in selector
            and (
                type(selector["frame_type"]) is not str
                or selector["frame_type"] not in {"I", "P", "B"}
            )
        )
        or (
            "content_hash" in selector
            and (
                type(selector["content_hash"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", selector["content_hash"]) is None
            )
        )
    ):
        raise ValueError("invalid-state-contract")
    if "contains" in selector and (
        not isinstance(selector["contains"], list)
        or not selector["contains"]
        or any(type(v) is not str or not v.strip() for v in selector["contains"])
    ):
        raise ValueError("invalid-state-contract")


def validate_state_contract(check):
    predicates = check.get("state_predicates")
    if (
        check.get("state_contract") != "source-bound-state-v1"
        or not isinstance(predicates, dict)
        or not predicates
        or set(predicates) - {"correction_retirement", "temporal_selection"}
    ):
        raise ValueError("invalid-state-contract")
    for name, spec in predicates.items():
        selectors = (
            ("target", "replacement") if name == "correction_retirement" else ("previous", "next")
        )
        required = {"scope", *selectors}
        if name == "temporal_selection":
            required |= {"previous_time", "next_time", "selected"}
        if not isinstance(spec, dict) or set(spec) != required:
            raise ValueError("invalid-state-contract")
        scope = spec["scope"]
        if (
            not isinstance(scope, dict)
            or set(scope) != {"tenant_id", "principal_id", "namespace"}
            or any(type(v) is not str or not v.strip() for v in scope.values())
        ):
            raise ValueError("invalid-state-contract")
        for key in selectors:
            validate_selector(spec[key])
        if name == "temporal_selection":
            if type(spec["selected"]) is not str or spec["selected"] not in {"previous", "next"}:
                raise ValueError("invalid-state-contract")
            for key in ("previous_time", "next_time"):
                times = spec[key]
                if (
                    not isinstance(times, dict)
                    or set(times) != {"effective_at", "observed_at"}
                    or any(instant(value) is None for value in times.values())
                ):
                    raise ValueError("invalid-state-contract")
            if any(instant(check.get(key)) is None for key in ("as_of", "known_at")) or instant(
                spec["next_time"]["effective_at"]
            ) <= instant(spec["previous_time"]["effective_at"]):
                raise ValueError("invalid-state-contract")


def instant(value):
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.astimezone(UTC) if parsed.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def times_match(row, expected):
    return all(
        instant(expected.get(key)) is not None and instant(row.get(key)) == instant(expected[key])
        for key in ("effective_at", "observed_at")
    )


def scoped(row, scope):
    return all(row.get(key) == value for key, value in scope.items())


def select_frames(snapshot, selector, scope):
    """Resolve identities from authored source/message/scope, never expected IDs."""
    ids = {
        p["frame_id"]
        for p in snapshot.get("memory_provenance", [])
        if scoped(p, scope)
        and p.get("source_ref") == selector["source_ref"]
        and p.get("message_id") == selector["message_id"]
    }
    return [
        f
        for f in snapshot.get("memory_frames", [])
        if f["id"] in ids
        and scoped(f, scope)
        and ("frame_type" not in selector or f.get("frame_type") == selector["frame_type"])
        and ("content_hash" not in selector or f.get("content_hash") == selector["content_hash"])
        and all(v.casefold() in f["content"].casefold() for v in selector.get("contains", []))
    ]


def selected_evidence(frame, selector, scope, evidence):
    return (
        frame.get("status") == "active" and frame.get("trust") in {"normal", "trusted", "low"}
    ) and any(
        hit["frame"].get("id") == frame["id"]
        and hit["frame"].get("scope") == scope
        and hit["frame"].get("content") == frame["content"]
        and any(
            p.get("source_ref") == selector["source_ref"]
            and p.get("message_id") == selector["message_id"]
            and p.get("span_verified") is True
            for p in hit["frame"].get("provenance", [])
        )
        for hit in evidence.get("hits", [])
    )


def correction_retirement(spec, evidence, snapshot):
    targets = select_frames(snapshot, spec["target"], spec["scope"])
    replacements = select_frames(snapshot, spec["replacement"], spec["scope"])
    lineage = retired = supported = False
    if len(targets) == len(replacements) == 1:
        old, new = targets[0], replacements[0]
        lineage = (
            old["id"] != new["id"]
            and old.get("superseded_by") == new["id"]
            and new.get("supersedes") == old["id"]
            and any(
                e.get("source_frame_id") == old["id"]
                and e.get("derived_frame_id") == new["id"]
                and e.get("operation") in {"supersede", "correct"}
                for e in snapshot.get("memory_frame_lineage", [])
            )
        )
        # Inspect the intended original in the whole store, not only recall hits.
        retired = (
            old.get("status") == "superseded"
            and bool(old.get("content_hash"))
            and not any(
                f.get("status") == "active"
                and scoped(f, spec["scope"])
                and f.get("content_hash") == old["content_hash"]
                for f in snapshot.get("memory_frames", [])
            )
        )
        supported = selected_evidence(new, spec["replacement"], spec["scope"], evidence)
    return {
        "passed": bool(lineage and retired and supported),
        "target_resolved": len(targets) == 1,
        "replacement_resolved": len(replacements) == 1,
        "intended_lineage": bool(lineage),
        "storewide_retired": bool(retired),
        "replacement_evidence": bool(supported),
    }


def temporal_selection(spec, evidence, snapshot, check):
    previous = select_frames(snapshot, spec["previous"], spec["scope"])
    following = select_frames(snapshot, spec["next"], spec["scope"])
    lineage = dates = choice = proof = False
    if len(previous) == len(following) == 1:
        old, new = previous[0], following[0]
        ids = [old["id"], new["id"]]
        temporal = [
            r for r in snapshot.get("memory_frame_temporal", []) if scoped(r, spec["scope"])
        ]
        old_rows = [r for r in temporal if r.get("frame_id") == old["id"]]
        new_rows = [r for r in temporal if r.get("frame_id") == new["id"]]
        if len(old_rows) == len(new_rows) == 1:
            old_row, new_row = old_rows[0], new_rows[0]
            lineage = (
                old["id"] != new["id"]
                and all(
                    f.get("frame_type") == "I"
                    and f.get("status") == "active"
                    and f.get("supersedes") is None
                    and f.get("superseded_by") is None
                    for f in (old, new)
                )
                and old_row.get("operation") == "observation"
                and old_row.get("previous_frame_id") is None
                and new_row.get("operation") == "transition"
                and new_row.get("previous_frame_id") == old["id"]
                and all(r is new_row for r in temporal if r.get("previous_frame_id") in ids)
            )
            dates = times_match(old_row, spec["previous_time"]) and times_match(
                new_row, spec["next_time"]
            )
            as_of, known_at = instant(check.get("as_of")), instant(check.get("known_at"))
            available = [
                name
                for name in ("previous", "next")
                if as_of is not None
                and known_at is not None
                and instant(spec[name + "_time"]["effective_at"]) <= as_of
                and instant(spec[name + "_time"]["observed_at"]) <= known_at
            ]
            choice = bool(available) and available[-1] == spec["selected"]
            selected = old if spec["selected"] == "previous" else new
            selected_row = old_row if spec["selected"] == "previous" else new_row
            groups = [
                g
                for g in evidence.get("groups", [])
                if any(fid in ids for fid in g.get("frame_ids", []))
            ]
            selections = [t for g in groups for t in g.get("temporal", [])]
            proof = (
                selected_evidence(selected, spec[spec["selected"]], spec["scope"], evidence)
                and all(
                    instant(evidence.get(key)) == instant(check.get(key))
                    for key in ("as_of", "known_at")
                )
                and bool(groups)
                and all(
                    g.get("root_frame_id") == selected["id"]
                    and g.get("view") == "historical"
                    and all(
                        instant(g.get(key)) == instant(check.get(key))
                        for key in ("as_of", "known_at")
                    )
                    and all(fid == selected["id"] for fid in g.get("frame_ids", []) if fid in ids)
                    for g in groups
                )
                and bool(selections)
                and all(
                    t.get("path") == ids
                    and t.get("requested_frame_id") in ids
                    and t.get("selected_frame_id") == selected["id"]
                    and t.get("operation") == selected_row["operation"]
                    and t.get("previous_frame_id") == selected_row["previous_frame_id"]
                    and times_match(t, spec[spec["selected"] + "_time"])
                    for t in selections
                )
                and all(
                    h["frame"].get("id") == selected["id"]
                    for h in evidence.get("hits", [])
                    if h["frame"].get("id") in ids
                )
            )
    return {
        "passed": bool(lineage and dates and choice and proof),
        "previous_resolved": len(previous) == 1,
        "next_resolved": len(following) == 1,
        "intended_lineage": bool(lineage),
        "expected_dates": bool(dates),
        "query_selection": bool(choice),
        "selected_evidence": bool(proof),
    }


def state_diagnostics(check, evidence, snapshot):
    results = {}
    for name, spec in check["state_predicates"].items():
        if name == "correction_retirement":
            results[name] = correction_retirement(spec, evidence, snapshot)
        elif name == "temporal_selection":
            results[name] = temporal_selection(spec, evidence, snapshot, check)
    return results


REPORT_MODE = "source-bound-verbatim-report-v1"
_REPORT_KEYS = {
    "event_id",
    "revision",
    "source_ref",
    "message_id",
    "message_role",
    "message_utf8_sha256",
    "span_start_utf8",
    "span_end_utf8",
    "span_utf8_sha256",
}


def validate_correction_policy(check):
    """Opt-in schema only; never infer report authority from a model act label."""
    if "correction_context_policy" not in check:
        return False
    policy = check["correction_context_policy"]
    from .origin import validate_origin_check

    try:
        validate_state_contract(check)
        origin = validate_origin_check(check)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("invalid-correction-context-policy") from exc
    if (
        not origin
        or set(check["state_predicates"]) != {"correction_retirement"}
        or check.get("forbidden_current_plaintext_scope") != "eligible-frame-content"
        or check.get("answer_contract") != "canonical-value-only-v1"
        or not isinstance(policy, dict)
        or set(policy) != {"mode", "reports"}
        or type(policy.get("mode")) is not str
        or policy["mode"] != REPORT_MODE
        or not isinstance(policy.get("reports"), list)
        or not isinstance(check.get("forbidden_current_plaintext"), list)
        or not check["forbidden_current_plaintext"]
        or any(type(v) is not str or not v.strip() for v in check["forbidden_current_plaintext"])
    ):
        raise ValueError("invalid-correction-context-policy")
    seen = set()
    ranges = {}
    for row in policy["reports"]:
        if (
            not isinstance(row, dict)
            or set(row) != _REPORT_KEYS
            or any(
                type(row.get(k)) is not str or not row[k].strip()
                for k in ("event_id", "source_ref", "message_id", "message_role")
            )
            or row["message_role"] not in {"user", "assistant", "system", "tool"}
            or any(
                type(row.get(k)) is not str or re.fullmatch(r"[0-9a-f]{64}", row[k]) is None
                for k in ("message_utf8_sha256", "span_utf8_sha256")
            )
            or type(row.get("revision")) is not int
            or row["revision"] < 1
            or any(type(row.get(k)) is not int for k in ("span_start_utf8", "span_end_utf8"))
            or not 0 <= row["span_start_utf8"] < row["span_end_utf8"]
        ):
            raise ValueError("invalid-correction-context-policy")
        identity = (row["event_id"], row["revision"], row["source_ref"], row["message_id"])
        token = (*identity, row["span_start_utf8"], row["span_end_utf8"])
        if token in seen or any(
            row["span_start_utf8"] < end and start < row["span_end_utf8"]
            for start, end in ranges.get(identity, [])
        ):
            raise ValueError("invalid-correction-context-policy")
        seen.add(token)
        ranges.setdefault(identity, []).append((row["span_start_utf8"], row["span_end_utf8"]))
    return True


class _ReportSources:
    """Evaluator-private, nonserialized authority issued from independent bytes."""

    __slots__ = ("entries", "messages", "scope", "source_sha256")

    def __init__(self, entries, messages, scope, source_sha256):
        self.entries, self.messages = entries, messages
        self.scope, self.source_sha256 = dict(scope), source_sha256


def _report_key(check):
    import json

    return (
        check["id"],
        json.dumps(
            check["correction_context_policy"],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def _positive_identity(value):
    return type(value) is int and value > 0


def _nonempty_string(value):
    return type(value) is str and bool(value.strip())


def _valid_scope(value):
    return (
        type(value) is dict
        and set(value) == {"tenant_id", "principal_id", "namespace"}
        and all(_nonempty_string(v) for v in value.values())
    )


def _typed_equal(left, right):
    """Preserve JSON primitive types, including nested non-identity input fields."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (
            all(type(k) is str for k in (*left, *right))
            and left.keys() == right.keys()
            and all(_typed_equal(left[k], right[k]) for k in left)
        )
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(
            _typed_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(left) is float:
        from math import isfinite

        return isfinite(left) and left == right
    return type(left) in (str, int, bool, type(None)) and left == right


def _message_rows(payload, fail, stage):
    rows = payload.get("messages")
    if type(rows) is not list or not rows:
        fail(stage + ".messages")
    seen = set()
    for message in rows:
        if type(message) is not dict:
            fail(stage + ".messages")
        if not _nonempty_string(message.get("message_id")) or message["message_id"] in seen:
            fail(stage + ".message_id")
        seen.add(message["message_id"])
        if not _nonempty_string(message.get("role")) or message["role"] not in {
            "user",
            "assistant",
            "system",
            "tool",
        }:
            fail(stage + ".role")
        if type(message.get("content")) is not str:
            fail(stage + ".content")
    return rows


def _source_messages(events, scope, stage):
    def fail(field):
        raise ValueError("invalid-correction-source-identity:" + field)

    if type(events) is not list or not events:
        fail(stage + ".events")
    messages, seen = [], set()
    for event in events:
        if type(event) is not dict:
            fail(stage + ".event")
        for key in ("event_id", "source_ref"):
            if not _nonempty_string(event.get(key)):
                fail(stage + "." + key)
        revision = event.get("revision", 1)  # Only omission has the declared integer default.
        if not _positive_identity(revision):
            fail(stage + ".revision")
        if not _valid_scope(event.get("scope", scope)) or not _typed_equal(
            event.get("scope", scope), scope
        ):
            fail(stage + ".scope")
        identity = (event["event_id"], revision)
        if identity in seen:
            fail(stage + ".event_id")
        seen.add(identity)
        for message in _message_rows(event, fail, stage):
            messages.append(
                {
                    "event_id": event["event_id"],
                    "revision": revision,
                    "source_ref": event["source_ref"],
                    "message_id": message["message_id"],
                    "message_role": message["role"],
                    "content": message["content"],
                }
            )
    return messages


def _validate_report_observations(snapshot, evidence):
    """Validate actual observation identities BEFORE any new opt-in equality join."""
    import json

    def fail(field):
        raise ValueError("invalid-correction-observation:" + field)

    def row(value, stage):
        if type(value) is not dict:
            fail(stage)

    def ids(value, fields, stage):
        row(value, stage)
        for field in fields:
            if not _positive_identity(value.get(field)):
                fail(stage + "." + field)

    def strings(value, fields, stage):
        for field in fields:
            if not _nonempty_string(value.get(field)):
                fail(stage + "." + field)

    def relations(value, stage):
        for field in ("supersedes", "superseded_by", "base_frame_id"):
            if value.get(field) is not None and not _positive_identity(value[field]):
                fail(stage + "." + field)

    def rows(value, name):
        result = value.get(name, [])
        if type(result) is not list:
            fail(name)
        return result

    row(snapshot, "snapshot")
    row(evidence, "evidence")
    for frame in rows(snapshot, "memory_frames"):
        ids(frame, ("id",), "memory_frames")
        relations(frame, "memory_frames")
        if not _valid_scope({k: frame.get(k) for k in ("tenant_id", "principal_id", "namespace")}):
            fail("memory_frames.scope")
        if type(frame.get("content")) is not str:
            fail("memory_frames.content")
    for source in rows(snapshot, "source_events"):
        ids(source, ("id", "revision"), "source_events")
        strings(
            source,
            ("event_id", "source_ref", "tenant_id", "principal_id", "namespace"),
            "source_events",
        )
        try:
            if type(source.get("payload_json")) is not str:
                fail("source_payload")
            payload = json.loads(source["payload_json"])
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid-correction-observation:source_payload") from exc
        row(payload, "source_payload")
        # Native erasure legitimately leaves {}. It cannot supply a witness.
        for field in ("event_id", "source_ref"):
            if field in payload and not _nonempty_string(payload[field]):
                fail("source_payload." + field)
        if "revision" in payload and not _positive_identity(payload["revision"]):
            fail("source_payload.revision")
        if "scope" in payload and not _valid_scope(payload["scope"]):
            fail("source_payload.scope")
        if "messages" in payload:
            _message_rows(payload, fail, "source_payload")
    for provenance in rows(snapshot, "memory_provenance"):
        ids(provenance, ("id", "frame_id", "source_event_id"), "memory_provenance")
        strings(
            provenance,
            ("source_ref", "message_id", "message_role", "tenant_id", "principal_id", "namespace"),
            "memory_provenance",
        )
        if provenance["message_role"] not in {"user", "assistant", "system", "tool"}:
            fail("memory_provenance.message_role")
    for retraction in rows(snapshot, "source_retractions"):
        row(retraction, "source_retractions")
        strings(
            retraction,
            ("source_ref", "tenant_id", "principal_id", "namespace"),
            "source_retractions",
        )
    for edge in rows(snapshot, "memory_frame_lineage"):
        ids(edge, ("source_frame_id", "derived_frame_id"), "memory_frame_lineage")
        if "id" in edge:
            ids(edge, ("id",), "memory_frame_lineage")
    for hit in rows(evidence, "hits"):
        row(hit, "returned_hit")
        frame = hit.get("frame")
        ids(frame, ("id",), "returned_frame")
        relations(frame, "returned_frame")
        if not _valid_scope(frame.get("scope")):
            fail("returned_frame.scope")
        if type(frame.get("content")) is not str:
            fail("returned_frame.content")
        for provenance in rows(frame, "provenance"):
            ids(provenance, ("id", "source_event_id"), "returned_provenance")
            if "frame_id" in provenance:
                ids(provenance, ("frame_id",), "returned_provenance")
            strings(provenance, ("source_ref", "message_id", "message_role"), "returned_provenance")
            if provenance["message_role"] not in {"user", "assistant", "system", "tool"}:
                fail("returned_provenance.message_role")


def bind_report_sources(checks, *, source_bytes, source_sha256, scope, effective_events=None):
    """Resolve prospective declarations BEFORE ingestion/answers against pinned input.

    The digest is supplied independently by the caller, never read from results.
    No authority is restored from a saved snapshot or a model-provided dictionary.
    """
    import json
    from hashlib import sha256

    for check in checks:
        validate_correction_policy(check)
    if not any("correction_context_policy" in c for c in checks):
        return None
    if (
        type(source_bytes) is not bytes
        or type(source_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
        or sha256(source_bytes).hexdigest() != source_sha256
        or not isinstance(scope, dict)
        or set(scope) != {"tenant_id", "principal_id", "namespace"}
        or any(type(v) is not str or not v.strip() for v in scope.values())
    ):
        raise ValueError("invalid-correction-source-authority")
    try:
        events = json.loads(source_bytes)
        messages = _source_messages(events, scope, "original")
        if effective_events is not None:
            # The existing created-at-only projection may not change source text,
            # role, event/revision or any other original input field.
            _source_messages(effective_events, scope, "effective")
            projected = [dict(event) for event in effective_events]
            if len(projected) != len(events):
                raise ValueError
            for original, effective in zip(events, projected, strict=True):
                if "created_at" in original:
                    effective["created_at"] = original["created_at"]
                else:
                    effective.pop("created_at", None)
            if not _typed_equal(projected, events):
                raise ValueError
        entries = {}
        for check in checks:
            if "correction_context_policy" not in check:
                continue
            if check["state_predicates"]["correction_retirement"]["scope"] != scope:
                raise ValueError
            reports = []
            for declaration in check["correction_context_policy"]["reports"]:
                found = [
                    m
                    for m in messages
                    if all(
                        m[k] == declaration[k]
                        for k in (
                            "event_id",
                            "revision",
                            "source_ref",
                            "message_id",
                            "message_role",
                        )
                    )
                ]
                if len(found) != 1:
                    raise ValueError
                text = found[0]["content"].encode("utf8")
                start, end = declaration["span_start_utf8"], declaration["span_end_utf8"]
                if (
                    end > len(text)
                    or sha256(text).hexdigest() != declaration["message_utf8_sha256"]
                ):
                    raise ValueError
                # Decode prefix and suffix too: offsets must both be UTF8 boundaries.
                text[:start].decode("utf8")
                text[end:].decode("utf8")
                span = text[start:end].decode("utf8")
                if sha256(text[start:end]).hexdigest() != declaration["span_utf8_sha256"]:
                    raise ValueError
                reports.append((dict(declaration), span))
            key = _report_key(check)
            if key in entries:
                raise ValueError
            entries[key] = tuple(reports)
    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
        raise ValueError("invalid-correction-source-authority") from exc
    return _ReportSources(entries, tuple(messages), scope, source_sha256)


def require_report_sources(check, authority, *, bound_origin=True):
    if (
        type(authority) is not _ReportSources
        or _report_key(check) not in authority.entries
        or check["state_predicates"]["correction_retirement"]["scope"] != authority.scope
    ):
        raise ValueError("invalid-correction-source-authority")
    selector = check["state_predicates"]["correction_retirement"]["target"]
    if bound_origin and (
        type(selector.get("content_hash")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", selector["content_hash"]) is None
    ):
        raise ValueError("unbound-correction-original")


def validate_report_projection(checks, authority, events):
    """Replay admission uses independent authority before saved answers/state."""
    selected = [c for c in checks if validate_correction_policy(c)]
    if not selected:
        return
    for check in selected:
        require_report_sources(check, authority, bound_origin=False)
    try:
        messages = _source_messages(events, authority.scope, "replay")
        if not _typed_equal(tuple(messages), authority.messages):
            raise ValueError
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("invalid-correction-source-projection") from exc


def _committed_witness(frame, hit, message, snapshot, scope, whole_quote=None):
    """Join actual scoped committed source -> provenance -> returned verified hit."""
    import json

    if (
        not scoped(frame, scope)
        or hit.get("scope") != scope
        or frame.get("status") != "active"
        or hit.get("status") != "active"
        or frame.get("trust") not in {"normal", "trusted", "low"}
        or hit.get("trust") != frame.get("trust")
        or hit.get("content") != frame.get("content")
        or any(
            scoped(r, scope) and r.get("source_ref") == message["source_ref"]
            for r in snapshot.get("source_retractions", [])
        )
    ):
        return False
    sources = [
        r
        for r in snapshot.get("source_events", [])
        if scoped(r, scope)
        and all(r.get(k) == message[k] for k in ("event_id", "revision", "source_ref"))
    ]
    if len(sources) != 1 or type(sources[0].get("id")) is not int:
        return False
    source = sources[0]
    try:
        payload = json.loads(source["payload_json"])
        actual = [
            m for m in payload.get("messages", []) if m.get("message_id") == message["message_id"]
        ]
    except (ValueError, KeyError, TypeError):
        return False
    if (
        len(actual) != 1
        or actual[0].get("role") != message["message_role"]
        or actual[0].get("content") != message["content"]
    ):
        return False
    provenances = [
        p
        for p in snapshot.get("memory_provenance", [])
        if scoped(p, scope)
        and p.get("frame_id") == frame["id"]
        and p.get("source_event_id") == source["id"]
        and p.get("source_ref") == message["source_ref"]
        and p.get("message_id") == message["message_id"]
        and p.get("message_role") == message["message_role"]
    ]
    for p in provenances:
        quote = p.get("verbatim_span")
        if (
            type(quote) is not str
            or not quote
            or quote not in message["content"]
            or whole_quote is not None
            and quote != whole_quote
        ):
            continue
        if any(
            w.get("id") == p.get("id")
            and w.get("source_event_id") == source["id"]
            and w.get("source_ref") == message["source_ref"]
            and w.get("message_id") == message["message_id"]
            and w.get("message_role") == message["message_role"]
            and w.get("verbatim_span") == quote
            and w.get("span_verified") is True
            for w in hit.get("provenance", [])
        ):
            return True
    return False


def report_surface_diagnostics(check, evidence, snapshot, authority):
    require_report_sources(check, authority)
    try:
        _validate_report_observations(snapshot, evidence)
    except ValueError as exc:
        return {
            "passed": False,
            "mode": REPORT_MODE,
            "allowed_report_ids": [],
            "unsupported_payload_ids": [
                h["frame"].get("id")
                for h in evidence.get("hits", [])
                if isinstance(h, dict)
                and isinstance(h.get("frame"), dict)
                and type(h["frame"].get("content")) is str
                and any(
                    v.casefold() in h["frame"]["content"].casefold()
                    for v in check["forbidden_current_plaintext"]
                )
            ],
            "replacement_source_evidence": False,
            "identity_failure": str(exc),
            "semantic_stale_positive_absence": "not_measured",
        }
    spec = check["state_predicates"]["correction_retirement"]
    selected = {
        f["id"]
        for key in ("target", "replacement")
        for f in select_frames(snapshot, spec[key], spec["scope"])
    }
    rejected, allowed = [], []
    for hit in evidence.get("hits", []):
        payload = hit["frame"]
        if not any(
            v.casefold() in payload["content"].casefold()
            for v in check["forbidden_current_plaintext"]
        ):
            continue
        rows = [f for f in snapshot.get("memory_frames", []) if f.get("id") == payload.get("id")]
        matches = 0
        if len(rows) == 1 and payload.get("id") not in selected:
            for declaration, span in authority.entries[_report_key(check)]:
                if payload["content"].encode("utf8") != span.encode("utf8"):
                    continue
                messages = [
                    m
                    for m in authority.messages
                    if all(
                        m[k] == declaration[k]
                        for k in (
                            "event_id",
                            "revision",
                            "source_ref",
                            "message_id",
                            "message_role",
                        )
                    )
                ]
                if len(messages) == 1 and _committed_witness(
                    rows[0], payload, messages[0], snapshot, spec["scope"], span
                ):
                    matches += 1
        (allowed if matches == 1 else rejected).append(payload.get("id"))
    replacements = select_frames(snapshot, spec["replacement"], spec["scope"])
    replacement_bound = False
    if len(replacements) == 1:
        replacement = replacements[0]
        messages = [
            m
            for m in authority.messages
            if all(m[k] == spec["replacement"][k] for k in ("source_ref", "message_id"))
        ]
        replacement_bound = len(messages) == 1 and any(
            h["frame"].get("id") == replacement["id"]
            and _committed_witness(replacement, h["frame"], messages[0], snapshot, spec["scope"])
            for h in evidence.get("hits", [])
        )
    return {
        "passed": not rejected and replacement_bound,
        "mode": REPORT_MODE,
        "allowed_report_ids": allowed,
        "unsupported_payload_ids": rejected,
        "replacement_source_evidence": bool(replacement_bound),
        "semantic_stale_positive_absence": "not_measured",
    }
