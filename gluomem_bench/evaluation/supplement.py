"""Compose independently pinned, source-authored aliases; never inspect answers.

This is the portable input seam of the prospective evaluator supplement. The
wire protocol is retained for compatibility; fixed campaign paths/history IDs,
execution authorities and lifecycle operators are deliberately not imported.
"""

import json
import re
from copy import deepcopy
from hashlib import sha256

from .grader import Refusal
from .state_contracts import validate_correction_policy, validate_state_contract

PROTOCOL = "q30-prospective-evaluator-only@v1"


def apply_supplement(checks, *, supplement_bytes, supplement_sha256, history_id, original_inputs):
    """Return new checks after verifying *every* declared original input pin.

    ``original_inputs`` maps the supplement's authored names to independent raw
    input bytes, not values reconstructed from results. Call before ingestion or
    answering; aliases are evaluator-only and are not response normalizations.
    No supplement is selected implicitly by this API or by the benchmark runner.
    """
    if (
        type(supplement_bytes) is not bytes
        or type(supplement_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", supplement_sha256) is None
        or sha256(supplement_bytes).hexdigest() != supplement_sha256
    ):
        raise Refusal("invalid-evaluator-supplement-pin")
    try:
        supplement = json.loads(supplement_bytes)
    except (ValueError, UnicodeError) as exc:
        raise Refusal("invalid-evaluator-supplement") from exc
    if (
        not isinstance(supplement, dict)
        or set(supplement)
        != {
            "protocol",
            "original_inputs",
            "histories",
            "label_catalog",
            "authorship",
            "semantic_stale_positive_absence",
        }
        or supplement.get("protocol") != PROTOCOL
        or supplement.get("semantic_stale_positive_absence") != "not_measured"
        or not isinstance(supplement.get("histories"), dict)
        or history_id not in supplement["histories"]
        or not isinstance(supplement.get("original_inputs"), dict)
        or not supplement["original_inputs"]
        or not isinstance(original_inputs, dict)
        or set(original_inputs) != set(supplement["original_inputs"])
    ):
        raise Refusal("invalid-evaluator-supplement")
    for name, digest in supplement["original_inputs"].items():
        if (
            type(name) is not str
            or not name
            or type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(original_inputs[name]) is not bytes
            or sha256(original_inputs[name]).hexdigest() != digest
        ):
            raise Refusal("evaluator-original-input-pin-drift")
    updates = supplement["histories"][history_id]
    check_ids = {check["id"] for check in checks}
    if len(check_ids) != len(checks) or not isinstance(updates, dict) or set(updates) != check_ids:
        raise Refusal("invalid-evaluator-supplement-checks")
    composed = deepcopy(checks)
    for check in composed:
        change = updates[check["id"]]
        if (
            not isinstance(change, dict)
            or "accepted_values" not in change
            or set(change) - {"accepted_values", "correction_context_policy"}
            or not isinstance(change["accepted_values"], list)
            or not change["accepted_values"]
            or any(type(v) is not str or not v or v != v.strip() for v in change["accepted_values"])
            or len(set(change["accepted_values"])) != len(change["accepted_values"])
        ):
            raise Refusal("invalid-evaluator-supplement-change")
        check.update(deepcopy(change))
        try:
            if "state_contract" in check or "state_predicates" in check:
                validate_state_contract(check)
            validate_correction_policy(check)
        except ValueError as exc:
            raise Refusal(str(exc)) from exc
    return composed
