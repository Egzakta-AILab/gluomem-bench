"""Opt-in evaluator origin binding from pre-correction source materializations.

This binds an authored literal selector to a retained content hash before the
later writer's chosen target is inspected. It never restores erased payloads.
"""

import json
from copy import deepcopy
from hashlib import sha256

CONTRACT = "source-materialization-binding-v1"


def validate_origin_check(check):
    present = {key for key in ("state_origin_contract", "state_origin_bindings") if key in check}
    if not present:
        return False
    bindings = check.get("state_origin_bindings")
    if (
        present != {"state_origin_contract", "state_origin_bindings"}
        or check.get("state_origin_contract") != CONTRACT
        or check.get("state_contract") != "source-bound-state-v1"
        or not isinstance(bindings, dict)
        or set(bindings) != {"correction_retirement"}
        or not isinstance(bindings["correction_retirement"], dict)
        or set(bindings["correction_retirement"]) != {"target"}
    ):
        raise ValueError("invalid-source-materialization-binding")
    value = bindings["correction_retirement"]["target"]
    predicate = check.get("state_predicates", {}).get("correction_retirement")
    if (
        not isinstance(value, dict)
        or set(value) != {"contains"}
        or not isinstance(value["contains"], list)
        or not value["contains"]
        or any(type(v) is not str or not v.strip() for v in value["contains"])
        or not isinstance(predicate, dict)
        or not isinstance(predicate.get("target"), dict)
        or not isinstance(predicate.get("scope"), dict)
    ):
        raise ValueError("invalid-source-materialization-binding")
    return True


def bind_origin_checks(checks, materializations):
    bound = deepcopy(checks)
    bindings = []
    for check in bound:
        if not validate_origin_check(check):
            continue
        predicate = check["state_predicates"]["correction_retirement"]
        selector, scope = predicate["target"], predicate["scope"]
        literals = check["state_origin_bindings"]["correction_retirement"]["target"]["contains"]
        matches = []
        for index, receipt in enumerate(materializations):
            row_id = receipt.get("source_event_row_id")
            if type(row_id) is not int or row_id <= 0 or receipt.get("deduped_event") is not False:
                continue
            # Bind only provenance newly materialized by this source event,
            # not an older witness copied into a later reused frame result.
            for frame in receipt.get("frames", []):
                if (
                    frame.get("scope") != scope
                    or frame.get("status") != "active"
                    or type(frame.get("id")) is not int
                    or frame["id"] <= 0
                    or type(frame.get("content")) is not str
                    or "frame_type" in selector
                    and frame.get("frame_type") != selector["frame_type"]
                    or not all(
                        value.casefold() in frame["content"].casefold() for value in literals
                    )
                ):
                    continue
                witnesses = [
                    p
                    for p in frame.get("provenance", [])
                    if type(p.get("source_event_id")) is int
                    and p["source_event_id"] == row_id
                    and p.get("source_ref") == selector["source_ref"]
                    and p.get("message_id") == selector["message_id"]
                ]
                if witnesses:
                    matches.append((index, receipt, frame))
        if len(matches) != 1:
            raise ValueError("source-materialization-origin-unresolved")
        index, receipt, frame = matches[0]
        from gluomem.frames.content_hash import content_hash

        digest = content_hash(frame["content"])
        if "content_hash" in selector and selector["content_hash"] != digest:
            raise ValueError("source-materialization-declared-hash-conflict")
        selector["content_hash"] = digest
        raw = json.dumps(
            receipt, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
        bindings.append(
            {
                "check_id": check["id"],
                "predicate": "correction_retirement",
                "selector": "target",
                "materialization_index": index,
                "materialization_sha256": sha256(raw).hexdigest(),
                "origin_frame_id": frame["id"],
                "content_hash": digest,
                "source_ref_sha256": sha256(selector["source_ref"].encode()).hexdigest(),
                "message_id_sha256": sha256(selector["message_id"].encode()).hexdigest(),
            }
        )
    return bound, {"protocol": "source-materialization-bindings@v1", "bindings": bindings}
