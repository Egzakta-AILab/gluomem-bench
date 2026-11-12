"""Socket-free pool evidence replay; no runtime health, client or budget objects.

The enclosing root must authenticate source/config/artifact inventories and pass
actual physical receipts. Native source schemas/messages are rebuilt by the
existing strict-extraction/projected-consolidation verifier. Plain reader/judge
messages require an independently supplied expected_requests binding.
"""

from __future__ import annotations

import math

from gluomem_bench.bench.rack_pool import digest, pool_key, retry_after_seconds, retryable_wire
from gluomem_bench.bench.rack_provider import (
    POOL_VERSION,
    estimated_prompt_tokens,
    pool_attempt_options,
    pool_attempt_profile,
    response_evidence,
)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_pool_call(row, profile, *, expected_request=None, require_success=True,
                       endpoint_metadata=None, admitted_endpoint_tags=None):
    from gluomem_bench.bench.rack_pricing_replay import validate_pricing_row
    validate_pricing_row(row, profile, endpoint_metadata=endpoint_metadata,
                         admitted_endpoint_tags=admitted_endpoint_tags)
    if (
        profile.profile_version not in {POOL_VERSION, "observed-provider-v13"}
        or row.get("declared_pool") != profile.openrouter_pool.model_dump(mode="json")
        or row.get("pool_key") != pool_key(profile)
    ):
        raise ValueError("pool declaration differs from independently bound profile")
    pool = profile.openrouter_pool
    logical = row.get("logical_ordinal")
    started, deadline = row.get("logical_started"), row.get("logical_deadline")
    if (
        type(logical) is not int
        or logical < 1
        or not _number(started)
        or not _number(deadline)
        or not 0 < deadline - started <= pool.logical_deadline_seconds
    ):
        raise ValueError("pool original logical deadline/ordinal differs")
    attempts = row.get("pool_attempts")
    if (
        not isinstance(attempts, list)
        or not 1 <= len(attempts) <= pool.max_attempts_per_logical_call
    ):
        raise ValueError("pool physical attempt coverage differs")
    previous, semantic = None, None
    unknown = infra = 0
    for index, attempt in enumerate(attempts, 1):
        endpoint = attempt.get("endpoint", {})
        if admitted_endpoint_tags is not None and endpoint.get("tag") not in admitted_endpoint_tags:
            raise ValueError("pool physical endpoint outside admitted scope")
        single = pool_attempt_profile(profile, endpoint.get("tag"))
        member = next(e for e in pool.endpoints if e.tag == endpoint["tag"])
        ordinal = attempt.get("ordinal")
        if (
            type(ordinal) is not int
            or ordinal < 1
            or previous is not None
            and ordinal != previous + 1
            or endpoint != member.model_dump(mode="json")
            or attempt.get("attempt") != index
            or attempt.get("logical_ordinal") != logical
            or attempt.get("pool_key") != pool_key(profile)
            or attempt.get("logical_deadline") != deadline
            or attempt.get("intent") != digest([pool_key(profile), logical, index])
        ):
            raise ValueError("pool ordinal/intent/endpoint price binding differs")
        previous = ordinal
        wire = attempt.get("wire", {})
        request = wire.get("request", {})
        if profile.pricing_policy is not None and (
                estimated_prompt_tokens(request['messages'])
                > profile.pricing_policy.prompt_token_cap
                or type(request.get('max_tokens')) is not int
                or not 1 <= request['max_tokens'] <= profile.pricing_policy.output_token_cap):
            raise ValueError('pricing physical prompt/output allowance exceeded')
        if (
            wire.get("url") != profile.base_url + "/chat/completions"
            or request.get("model") != profile.model
            or any(
                request.get(key) != value
                for key, value in pool_attempt_options(profile, endpoint["tag"]).items()
            )
            or request.get("n", 1) != 1
            or request.get("stream", False) is not False
        ):
            raise ValueError("pool singleton route/model/fallback controls differ")
        body = {k: v for k, v in request.items() if k != "provider"}
        if semantic is None:
            semantic = body
        elif body != semantic:
            raise ValueError("pool retry changed messages/schema/output semantics")
        if expected_request is not None:
            expected = {k: v for k, v in expected_request.items() if k != "provider"}
        else:
            if "messages" not in row or "generation_kwargs" not in row:
                raise ValueError("unbound pool direct reader/judge request")
            expected = {
                "model": profile.model,
                "messages": row["messages"],
                "temperature": 0,
                "max_tokens": profile.output_budget(row["task"]),
                "reasoning": {"enabled": False},
            }
            if "response_format" in row["generation_kwargs"]:
                expected["response_format"] = row["generation_kwargs"]["response_format"]
        if body != expected:
            raise ValueError(
                "pool request schema/messages differ from independently bound invocation"
            )
        before, after = attempt.get("started_monotonic"), attempt.get("completed_monotonic")
        seconds = attempt.get("timeout_seconds")
        if (
            not _number(before)
            or not _number(after)
            or not _number(seconds)
            or not started <= before <= after
            or not 0 < seconds <= min(profile.timeout_seconds, deadline - before)
        ):
            raise ValueError("pool attempt clock/timeout differs from original deadline")
        outcome = attempt.get("outcome")
        if index < len(attempts):
            if outcome != "infrastructure-retry" or not retryable_wire(wire, single):
                raise ValueError("pool retry lacks allowlisted infrastructure HTTP evidence")
            infra += 1
        elif outcome != "success":
            if require_success:
                raise ValueError("failed pool call cannot be admitted as complete")
        else:
            if not 200 <= wire.get("status", 0) < 300 or after >= deadline:
                raise ValueError("pool success lacks timely HTTP acceptance")
            evidence = response_evidence(
                single,
                wire.get("response", {}),
                estimated_prompt_tokens(request["messages"]),
                output_budget=request["max_tokens"],
            )
            if (
                "provider_evidence" in row
                and row["provider_evidence"] != evidence["provider_evidence"]
            ):
                raise ValueError("pool final serving provider evidence differs")
            if "native_response" in attempt:
                from gluomem_bench.bench.rack_schema import validate_native_projection

                validate_native_projection(attempt["response"], attempt["native_response"])
                if attempt["native_response"]["canonical"]["provider_data"] != wire["response"]:
                    raise ValueError("pool native SDK response differs from physical wire")
            if "native_response" in row and row["native_response"] != attempt.get(
                "native_response"
            ):
                raise ValueError("pool final native result differs from serving attempt")
        usage = (
            wire.get("response", {}).get("usage", {})
            if isinstance(wire.get("response"), dict)
            else {}
        )
        unknown += not all(
            type(usage.get(k)) is int and usage[k] >= 0
            for k in ("prompt_tokens", "completion_tokens")
        )
    if row.get("ordinal", previous) != previous:
        raise ValueError("pool final ordinal differs from physical attempts")
    return {
        "physical_attempts": len(attempts),
        "infrastructure_failures": infra,
        "unknown_usage_attempts": unknown,
    }


def validate_pool_history(
    native_rows,
    attempt_rows,
    profile,
    cfg,
    protocol,
    expected,
    *,
    physical_rows,
    expected_requests=None,
    endpoint_metadata=None,
    admitted_endpoint_tags=None,
):
    """Bind every native/direct call to exact physical receipts and original routes.

    Currency reservations remain the enclosing root allowance's responsibility;
    this verifier never equates a missing cost/usage field with zero actual spend.
    """
    expected_requests = expected_requests or {}
    if (profile.profile_version != POOL_VERSION
            or getattr(profile, "semantic_producer_contract", None) is not None):
        raise ValueError("only the separately qualified legacy pool history is admitted")
    if profile.pricing_policy is not None and endpoint_metadata is None:
        raise ValueError("independently bound pricing endpoint metadata required")
    by_logical, grouped = {}, {}
    for row in native_rows:
        logical = row.get("logical_ordinal")
        if logical in by_logical:
            raise ValueError("pool repeated native logical ordinal")
        by_logical[logical] = row
    for record in attempt_rows:
        from gluomem_bench.bench.rack_pricing_replay import validate_pricing_row
        validate_pricing_row(record, profile, endpoint_metadata=endpoint_metadata,
                             admitted_endpoint_tags=admitted_endpoint_tags)
        if "pool_attempt" not in record:
            raise ValueError("pool physical attempt record missing")
        grouped.setdefault(record["logical_ordinal"], []).append(record)
    if set(by_logical) - grouped.keys() or sorted(grouped) != list(range(1, len(grouped) + 1)):
        raise ValueError("pool logical history coverage differs")
    if len(physical_rows) != len(attempt_rows):
        raise ValueError("pool physical receipt coverage differs")
    # The caller supplies physical order, independent of trace hashes/labels.
    for record, physical in zip(attempt_rows, physical_rows, strict=True):
        if any(record.get(k) != physical.get(k) for k in ("request", "response", "status")):
            raise ValueError("pool recorded attempt differs from physical root receipt")
    totals = {"physical_attempts": 0, "infrastructure_failures": 0, "unknown_usage_attempts": 0}
    cooldowns, failures, seen_ordinals = {}, {}, set()
    for logical in sorted(grouped):
        records = grouped[logical]
        attempts = [r["pool_attempt"] for r in records]
        row = by_logical.get(logical)
        if row is None:
            row = {**records[0], "pool_attempts": attempts, "ordinal": attempts[-1]["ordinal"]}
        elif row.get("pool_attempts") != attempts:
            raise ValueError("pool native attempted route history differs from physical records")
        if row.get("task") in {"event_claims", "consolidate", "identity_alignment"}:
            from gluomem_bench.bench.rack_schema import validate_production_row

            validate_production_row(row, profile, cfg, protocol, expected)
        elif logical not in expected_requests:
            raise ValueError("unbound pool plain reader/judge invocation")
        result = validate_pool_call(row, profile, expected_request=expected_requests.get(logical),
            endpoint_metadata=endpoint_metadata if admitted_endpoint_tags is not None else None,
            admitted_endpoint_tags=admitted_endpoint_tags)
        for key in totals:
            totals[key] += result[key]
        for attempt in attempts:
            ordinal = attempt["ordinal"]
            if ordinal in seen_ordinals:
                raise ValueError("pool repeated physical ordinal")
            seen_ordinals.add(ordinal)
            before, after = attempt["started_monotonic"], attempt["completed_monotonic"]
            eligible = [
                e for e in profile.openrouter_pool.endpoints if cooldowns.get(e.tag, 0) <= before
            ]
            if not eligible or eligible[0].tag != attempt["endpoint"]["tag"]:
                raise ValueError("pool shared health/order/cooldown differs")
            tag = eligible[0].tag
            if attempt["outcome"] == "infrastructure-retry":
                failures[tag] = failures.get(tag, 0) + 1
                pool = profile.openrouter_pool
                backoff = min(
                    pool.max_backoff_seconds,
                    pool.cooldown_seconds * 2 ** min(failures[tag] - 1, 20),
                )
                jitter = attempt.get("jitter_seconds")
                if not _number(jitter) or not 0 <= jitter <= min(
                    backoff * pool.jitter_fraction, pool.max_backoff_seconds - backoff
                ):
                    raise ValueError("pool jitter exceeds frozen backoff bound")
                retry_after = retry_after_seconds(
                    attempt["wire"].get("retry_after"),
                    wall=lambda received=attempt["wire"]["received_wall"]: received,
                )
                until = after + max(retry_after, backoff + jitter)
                if (
                    attempt.get("retry_after_seconds") != retry_after
                    or attempt.get("cooldown_until") != until
                ):
                    raise ValueError("pool Retry-After/cooldown evidence differs")
                cooldowns[tag] = until
    return totals
