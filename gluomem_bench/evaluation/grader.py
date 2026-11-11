"""Literal authored contracts, not an LLM/semantic judge or benchmark score.

Evaluate actual reader evidence and an independent store snapshot. Checks and
bound source authorities are evaluator-private and must never enter model input.
"""


class Refusal(ValueError):
    """Invalid evaluator input/authority, not a negative model-quality grade."""


def _validate_observation_ids(results, checks):
    """Require unique exact built-in strings containing non-whitespace text.

    Leading, internal and trailing whitespace is allowed and remains part of the
    exact identity. strip() only detects whitespace-only IDs; no ID is normalized,
    stringified or repaired. Validate both collections before grading or lookup.
    """
    for label, items in (("check", checks), ("result", results)):
        seen = set()
        for item in items:
            identifier = item.get("id")
            if type(identifier) is not str or not identifier.strip():
                raise Refusal("invalid-" + label + "-id")
            if identifier in seen:
                raise Refusal("duplicate-" + label + "-id")
            seen.add(identifier)


def evaluate(results, checks, snapshot, *, correction_sources=None):
    """Literal authored contracts; unique exact string IDs, no semantic score."""
    _validate_observation_ids(results, checks)
    for check in checks:
        if "correction_context_policy" in check:
            from .state_contracts import (
                require_report_sources,
                validate_correction_policy,
            )

            try:
                validate_correction_policy(check)
                require_report_sources(check, correction_sources)
            except ValueError as exc:
                raise Refusal(str(exc)) from exc
    rows = []
    by_id = {r["id"]: r for r in results}
    for check in checks:
        if "answer_contract" in check or "accepted_values" in check:
            values = check.get("accepted_values")
            if (
                check.get("answer_contract") != "canonical-value-only-v1"
                or not isinstance(values, list)
                or not values
                or any(type(value) is not str or not value.strip() for value in values)
            ):
                raise Refusal("invalid-answer-contract")
        if "state_contract" in check or "state_predicates" in check:
            from .state_contracts import state_diagnostics, validate_state_contract

            try:
                validate_state_contract(check)
            except ValueError as exc:
                raise Refusal("invalid-state-contract") from exc
        result = by_id.get(check["id"])
        if result is None:
            rows.append({"id": check["id"], "passed": False, "reason": "unrun"})
            continue
        evidence, answer = result["evidence"], result["answer"].casefold()
        expected = check.get("contains", [])
        actual_context = evidence["context"].casefold()
        passed = all(v.casefold() in answer for v in expected)
        if check.get("answer_contract") == "canonical-value-only-v1":
            passed = result["answer"].strip() in check["accepted_values"]
        passed = passed and all(v.casefold() in actual_context for v in expected)
        passed = passed and not any(
            v.casefold() in answer for v in check.get("forbidden_current_plaintext", [])
        )
        scope = check.get("forbidden_current_plaintext_scope", "full-reader-context")
        if scope == "eligible-frame-content":
            payloads = [hit["frame"]["content"].casefold() for hit in evidence["hits"]]
            # A newly supplied, attributed quote can correctly say the old value
            # was mistaken. It is not an eligible assertion of that old value.
            # Require the replacement payload, not just a mention in a quote.
            passed = (
                passed
                and bool(payloads)
                and all(any(v.casefold() in payload for payload in payloads) for v in expected)
            )
            forbidden_surface = "\n".join(payloads)
        elif scope == "full-reader-context":
            forbidden_surface = actual_context
        else:
            raise Refusal("invalid-plaintext-guard-scope")
        report_policy = None
        if "correction_context_policy" in check:
            from .state_contracts import report_surface_diagnostics

            report_policy = report_surface_diagnostics(
                check, evidence, snapshot, correction_sources
            )
            passed = passed and report_policy["passed"]
        else:
            passed = passed and not any(
                v.casefold() in forbidden_surface
                for v in check.get("forbidden_current_plaintext", [])
            )
        if check.get("required_trusted_source_role"):
            relevant = [
                hit["frame"]
                for hit in evidence["hits"]
                if any(v.casefold() in hit["frame"]["content"].casefold() for v in expected)
            ]
            attribution_contract = check.get("attribution_contract", "literal-all-v1")
            if attribution_contract == "bound-source-witness-v2":
                selector = check.get("required_evidence_source")
                if (
                    not isinstance(selector, dict)
                    or set(selector) != {"source_ref", "message_id"}
                    or any(
                        type(value) is not str or not value.strip() for value in selector.values()
                    )
                ):
                    raise Refusal("invalid-attribution-witness-source")

                def matches_source(provenance, selector=selector):
                    return all(provenance.get(key) == value for key, value in selector.items())

                relevant = [
                    frame
                    for frame in relevant
                    if any(matches_source(p) for p in frame["provenance"])
                ]
                passed = (
                    passed
                    and bool(relevant)
                    and all(
                        any(
                            p.get("message_role") == check["required_trusted_source_role"]
                            and p.get("span_verified") is True
                            and matches_source(p)
                            for p in frame["provenance"]
                        )
                        and frame["source"] != check.get("forbidden_frame_source")
                        for frame in relevant
                    )
                    and all(
                        any(v.casefold() in frame["content"].casefold() for frame in relevant)
                        for v in expected
                    )
                )
            elif attribution_contract != "literal-all-v1":
                raise Refusal("invalid-attribution-contract")
            else:
                passed = (
                    passed
                    and bool(relevant)
                    and all(
                        any(
                            p["message_role"] == check["required_trusted_source_role"]
                            and p["span_verified"]
                            for p in frame["provenance"]
                        )
                        and frame["source"] != check.get("forbidden_frame_source")
                        for frame in relevant
                    )
                )
        rows.append(
            {
                "id": check["id"],
                "passed": bool(passed),
                "plaintext_guard_scope": scope,
                "nonpayload_old_literal": scope == "eligible-frame-content"
                and any(
                    v.casefold() in actual_context and v.casefold() not in forbidden_surface
                    for v in check.get("forbidden_current_plaintext", [])
                ),
                "grading": (
                    "literal answer+eligible evidence and independent typed-state checks; "
                    "not benchmark/semantic score"
                ),
            }
        )
        if check.get("attribution_contract") == "bound-source-witness-v2":
            rows[-1]["attribution_contract"] = "bound-source-witness-v2"
        if check.get("answer_contract") == "canonical-value-only-v1":
            rows[-1]["answer_contract"] = "canonical-value-only-v1"
            rows[-1]["predicate_diagnostics"] = {
                "answer_format": {
                    "passed": result["answer"].strip() in check["accepted_values"],
                    "format_limited": True,
                }
            }
        if "state_contract" in check:
            from .state_contracts import state_diagnostics

            diagnostics = state_diagnostics(check, evidence, snapshot)
            diagnostics["literal_retention"] = {
                "passed": not any(
                    v.casefold() in answer or v.casefold() in forbidden_surface
                    for v in check.get("forbidden_current_plaintext", [])
                ),
                "scope": scope,
                "policy": "unchanged-literal-exclusion",
            }
            if report_policy is not None:
                diagnostics["report_surface_policy"] = report_policy
                diagnostics["literal_retention"] = {
                    "passed": report_policy["passed"]
                    and not any(
                        v.casefold() in answer for v in check["forbidden_current_plaintext"]
                    ),
                    "scope": scope,
                    "policy": "bounded-verbatim-report-exception",
                }
                rows[-1].update(
                    eligible_payload_old_literal=any(
                        v.casefold() in forbidden_surface
                        for v in check["forbidden_current_plaintext"]
                    ),
                    reader_context_old_literal=any(
                        v.casefold() in actual_context for v in check["forbidden_current_plaintext"]
                    ),
                    semantic_stale_positive_absence="not_measured",
                    grading=(
                        "exact authored answer+intended retirement+source-bound evidence+bounded "
                        "verbatim report policy; semantic absence not measured"
                    ),
                )
            rows[-1]["state_contract"] = check["state_contract"]
            rows[-1].setdefault("predicate_diagnostics", {}).update(diagnostics)
            rows[-1]["passed"] = bool(passed and all(d["passed"] for d in diagnostics.values()))
    return rows
