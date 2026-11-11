"""Socket-free quote replay. The caller authenticates source bytes and snapshot freshness."""

from __future__ import annotations

from gluomem_bench.bench.rack_provider import endpoint_discovery


def validate_pricing_row(row, profile, *, endpoint_metadata=None, admitted_endpoint_tags=None):
    """Rebuild projections from raw metadata, not from a reported price receipt.

    Without endpoint_metadata this is self-consistency only. Full root/history
    consumers bind the independent snapshot explicitly; no quote becomes a bill.
    """
    from gluomem_bench.bench.rack_pool import digest
    from gluomem_bench.bench.rack_provider import admitted_routes

    tags = admitted_routes(profile, admitted_endpoint_tags)
    attempts = row.get("pool_attempts", [row.get("pool_attempt", {})])
    if profile.pricing_policy is None:
        if (
            "pricing_evidence" in row
            or "pricing_observation" in row
            or any("pricing_observation" in a for a in attempts)
        ):
            raise ValueError("legacy pricing opt-out cannot admit bounded price evidence")
        return
    try:
        evidence = row["pricing_evidence"]
        metadata = evidence["provider_metadata"]
        if tags is not None and endpoint_metadata is None:
            raise ValueError("independently bound admitted-route pricing metadata required")
        if endpoint_metadata is not None and (
                digest(metadata) != digest(endpoint_metadata) if tags is not None
                else metadata != endpoint_metadata):
            raise ValueError("pricing metadata differs from independently bound snapshot")
        projection = endpoint_discovery(profile, metadata, admitted_endpoint_tags=tags)
        wanted = {
            "protocol": "bounded-provider-price-evidence@v1",
            "pricing_policy": profile.pricing_policy.model_dump(mode="json"),
            "provider_metadata": metadata,
            "pricing_observations": projection["pricing_observations"],
        }
        if (digest(evidence) != digest(wanted) if tags is not None else evidence != wanted):
            raise ValueError("pricing observation differs from raw public metadata/policy")
        observations = {o["endpoint"]["tag"]: o for o in projection["pricing_observations"]}
        if profile.openrouter_pool is None:
            actual = row.get("pricing_observation")
            expected = observations[profile.openrouter_endpoint.tag]
            if (digest(actual) != digest(expected) if tags is not None else actual != expected):
                raise ValueError("pricing singleton receipt differs from admitted metadata")
        else:
            for attempt in attempts:
                if tags is not None and attempt["endpoint"]["tag"] not in tags:
                    raise ValueError("pricing pool attempt outside admitted scope")
                if tags is not None and (
                        attempt["endpoint"]["tag"] not in observations
                        or type(attempt.get("pricing_observation")) is not dict
                        or digest(attempt["pricing_observation"])
                        != digest(observations[attempt["endpoint"]["tag"]])):
                    raise ValueError("pricing pool attempt receipt differs from admitted metadata")
                if attempt.get("pricing_observation") != observations.get(
                    attempt["endpoint"]["tag"]
                ):
                    raise ValueError("pricing pool attempt receipt differs from admitted metadata")
    except (KeyError, TypeError):
        raise ValueError("complete pricing metadata/observation receipt required") from None


def validate_direct_pricing_receipt(
    row, profile, *, endpoint_metadata, expected_request, physical_row
):
    """Bind one actual direct singleton return to external metadata/request/HTTP receipts."""
    from gluomem_bench.bench.rack_diagnostic import validate_request
    from gluomem_bench.bench.rack_provider import estimated_prompt_tokens, response_evidence

    if profile.pricing_policy is None or profile.openrouter_pool is not None:
        raise ValueError("explicit bounded pricing singleton receipt required")
    validate_pricing_row(row, profile, endpoint_metadata=endpoint_metadata)
    if (
        row.get("method") != "POST"
        or row.get("url") != profile.base_url + "/chat/completions"
        or row.get("status") != 200
        or row.get("request") != expected_request
        or any(row.get(k) != physical_row.get(k) for k in ("request", "response", "status"))
    ):
        raise ValueError("direct pricing physical receipt/request differs")
    validate_request(
        profile,
        row["request"],
        messages=expected_request["messages"],
        output=expected_request["max_tokens"],
        response_format=expected_request.get("response_format"),
    )
    tokens = estimated_prompt_tokens(expected_request["messages"])
    policy = profile.pricing_policy
    if tokens > policy.prompt_token_cap or expected_request["max_tokens"] > policy.output_token_cap:
        raise ValueError("direct pricing physical prompt/output allowance exceeded")
    evidence = response_evidence(
        profile, row["response"], tokens, output_budget=expected_request["max_tokens"]
    )
    if any(row.get(k) != value for k, value in evidence.items()):
        raise ValueError("direct pricing provider/usage receipt differs")
    return evidence
