"""Independent socket-free native diagnostic replay, never run authorization."""

import base64
import hashlib
import json
import math

from gluomem_bench.bench.rack_diagnostic import (
    VERSION,
    digest,
    model_material,
    validate_endpoint_metadata,
    validate_local_schema,
    validate_request,
)


def validate_model_metadata(profile, model):
    """Recheck the frozen effort/canonical declaration; never alias a response."""
    policy = profile.native_diagnostic
    if (
        profile.pricing_policy is None
        and digest(model) != policy.model_source_sha256
        or digest(model_material(model, pricing_policy=profile.pricing_policy))
        != policy.model_contract_sha256
        or model["id"] != profile.model
        or model["canonical_slug"] != policy.canonical_slug
        or model["context_length"] != profile.context_window
        or (
            policy.reasoning.model_dump() == {"effort": "none"}
            and "none" not in model["reasoning"]["supported_efforts"]
        )
    ):
        raise ValueError("diagnostic frozen model metadata/effort drift")
    if profile.pricing_policy is not None:
        from gluomem_bench.bench.rack_pricing import validate_prices

        validate_prices(profile.pricing_policy, model["pricing"])


def validate_physical_row(row, profile):
    """Replay original raw-wire bytes and SDK data, not reconstructed pseudo-wire."""
    from gluomem_bench.bench.rack_pricing_replay import validate_pricing_row
    from gluomem_bench.bench.rack_provider import estimated_prompt_tokens, response_evidence
    from gluomem_bench.bench.rack_schema import validate_native_projection

    validate_pricing_row(row, profile)

    try:
        attempt = row["physical_attempt"]
        if (
            attempt["protocol"] != "native-diagnostic-singleton-attempt@v1"
            or attempt["sent"] is not True
            or type(attempt["ordinal"]) is not int
            or attempt["ordinal"] != row["ordinal"]
            or attempt["ordinal"] < 1
        ):
            raise ValueError("diagnostic physical ordinal/attempt differs")
        if profile.profile_version == "observed-provider-v13":
            from gluomem_bench.bench.rack_observed_provider import singleton_intent

            if attempt.get("intent") != singleton_intent(profile, row) or attempt.get(
                "root_ordinal"
            ) != row.get("root_ordinal"):
                raise ValueError("observed singleton intent/root ordinal differs")
        start, end, deadline, seconds = [
            attempt[k]
            for k in (
                "started_monotonic",
                "completed_monotonic",
                "deadline_monotonic",
                "timeout_seconds",
            )
        ]
        if (
            any(
                type(n) not in (int, float) or not math.isfinite(n)
                for n in (start, end, deadline, seconds)
            )
            or not 0 < seconds <= profile.timeout_seconds
            or not start <= end <= deadline
            or abs(deadline - start - seconds) > 1e-7
        ):
            raise ValueError("diagnostic physical original deadline differs")
        wire = attempt["wire"]
        for kind in ("request", "response"):
            raw = wire[kind + "_raw_utf8"].encode("utf-8")
            if base64.b64decode(wire[kind + "_b64"], validate=True) != raw:
                raise ValueError("diagnostic physical byte representations differ")
            if (
                hashlib.sha256(raw).hexdigest() != wire[kind + "_sha256"]
                or json.loads(raw) != wire[kind]
            ):
                raise ValueError("diagnostic physical raw-byte digest/projection differs")
        if wire["status"] != 200 or wire["url"] != profile.base_url + "/chat/completions":
            raise ValueError("diagnostic physical status/URL differs")
        validate_request(
            profile,
            wire["request"],
            messages=row["messages"],
            output=profile.output_budget(row["task"]),
            response_format=row["generation_kwargs"].get("response_format"),
        )
        validate_native_projection(row["response"], row["native_response"])
        if wire["response"] != row["native_response"]["canonical"]["provider_data"]:
            raise ValueError("diagnostic physical response differs from original SDK provider data")
        if (
            profile.native_diagnostic is not None
            and profile.native_diagnostic.response_identity_policy is not None
            and row["response"].get("model") != wire["response"].get("model")
        ):
            raise ValueError("diagnostic raw/SDK returned model values differ")
        tokens = estimated_prompt_tokens(row["messages"])
        if tokens != row["prompt_token_count"] or tokens > (
            profile.native_diagnostic.prompt_token_cap
            if profile.native_diagnostic is not None
            else profile.pricing_policy.prompt_token_cap
        ):
            raise ValueError("diagnostic physical conservative prompt allowance differs")
        evidence = response_evidence(
            profile, wire["response"], tokens, output_budget=profile.output_budget(row["task"])
        )
        if any(row.get(k) != v for k, v in evidence.items()):
            raise ValueError("diagnostic physical provider/usage evidence differs")
        validate_local_schema(row)
    except (KeyError, TypeError):
        raise ValueError("complete diagnostic physical/raw evidence required") from None


def _semantic_projection_coverage(claims, seen):
    if any(
        not claim.meaning.independently_versioned and index not in seen
        for index, claim in enumerate(claims)
    ):
        raise ValueError("semantic dependent projection coverage differs")


def _semantic_source_projection(row, event, claims, seen):
    """Bind a projected invocation to its actual extraction, not its own witness."""
    from gluomem.llm.semantic import SemanticClaimContext, source_witness
    from gluomem.timeutil import to_storage

    invocation = row["consolidation_invocation"]
    index = invocation["projection"]["claim_index"]
    if type(index) is not int or not 0 <= index < len(claims) or seen and index <= max(seen):
        raise ValueError("semantic source projection claim order differs")
    claim = claims[index]
    witness = source_witness(event, claim)
    wanted = SemanticClaimContext(
        kind=event.kind or claim.kind,
        source_role=witness.source_role,
        observed_at=witness.observed_at,
        effective_at=to_storage(claim.event_date) if claim.event_date is not None else None,
        meaning=claim.meaning,
        witness=witness,
    )
    if (
        invocation["new_claim"] != claim.text.strip()
        or SemanticClaimContext.model_validate(invocation["new_claim_context"]) != wanted
    ):
        raise ValueError("semantic projection differs from extracted source claim/witness")
    seen.add(index)


def validate_diagnostic_history(
    rows, profile, cfg, protocol, expected, *, source_events, endpoint_metadata, model_metadata
):
    """Full source/typed-parser/physical replay; not benchmark or model quality.

    The parent separately authenticates the root source/config/output inventory,
    reservation journal and committed store. This helper cannot authorize a run
    or renew its original call/currency/deadline allowance.
    """
    from gluomem.write.events import SourceEvent

    from gluomem_bench.bench.rack_schema import validate_production_row

    if (
        profile.profile_version not in {VERSION, "provider-roles-v10"}
        or profile.writer_contract not in {"action-target-writer-v3", "semantic-state-writer-v4"}
        or profile.native_schema_strict is not False
        or cfg["dspy"]["schema_mode"] != "required"
        or cfg["dspy"]["schema_strict"] is not False
        or cfg["dspy"]["retry"]["max_attempts"] != 1
        or cfg["dspy"]["validation_retries"]
        or any((t.get("refine") or {}).get("enabled") for t in cfg["dspy"]["tasks"].values())
    ):
        raise ValueError("explicit v3 nonstrict native no-retry diagnostic history required")
    if profile.native_diagnostic is not None:
        validate_endpoint_metadata(profile, endpoint_metadata)
        validate_model_metadata(profile, model_metadata)
    sources = [SourceEvent.model_validate(e).model_dump(mode="json") for e in source_events]
    if len({e["event_id"] for e in sources}) != len(sources):
        raise ValueError("diagnostic source event identities must be unique")
    observed, ordinals = [], []
    semantic = profile.writer_contract == "semantic-state-writer-v4"
    source_event, source_claims, seen = None, [], set()
    for row in rows:
        from gluomem_bench.bench.rack_pricing_replay import validate_pricing_row

        validate_pricing_row(row, profile, endpoint_metadata=endpoint_metadata)
        if row.get("error") or row.get("finish_reasons") != ["stop"]:
            raise ValueError("failed diagnostic generation cannot be admitted as complete")
        validate_production_row(row, profile, cfg, protocol, expected)
        if row["task"] == "event_claims":
            observed.append(row["extraction_invocation"]["source_event"])
            if semantic:
                _semantic_projection_coverage(source_claims, seen)
                from gluomem.llm.structured_adapter import make_adapter

                from gluomem_bench.bench.rack_extraction import reconstructed_signature

                source_event = SourceEvent.model_validate(observed[-1])
                adapter = make_adapter(
                    cfg["dspy"]["adapter"],
                    canonicalize=cfg["dspy"].get("canonicalize", False),
                    schema_mode="required",
                    schema_strict=False,
                    validation_retries=0,
                )
                source_claims = adapter.parse(
                    reconstructed_signature(row, cfg),
                    row["response"]["choices"][0]["message"]["content"],
                )["claims"]
                seen = set()
        elif row["task"] != "consolidate" or not observed:
            raise ValueError("unadmitted diagnostic task/order")
        elif semantic:
            _semantic_source_projection(row, source_event, source_claims, seen)
        ordinals.extend(
            [a["ordinal"] for a in row["pool_attempts"]]
            if "pool_attempts" in row
            else [row["ordinal"]]
        )
    if semantic:
        _semantic_projection_coverage(source_claims, seen)
    if observed != sources or ordinals != list(range(1, len(ordinals) + 1)):
        raise ValueError("diagnostic complete source coverage/physical accounting differs")
    return {
        "protocol": "native-diagnostic-history-replay@v1",
        "logical_calls": len(rows),
        "generation_calls": len(ordinals),
        "source_events": len(sources),
        "source_events_sha256": digest(sources),
        "model_quality_admitted": False,
        "native_schema_enforcement": "not-guaranteed",
    }
