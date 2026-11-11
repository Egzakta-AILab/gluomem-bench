"""Closed v13 replay: independent physical stream plus authenticated private inputs.

The caller authenticates source/config/compiled artifacts, metadata, preparations,
reader requests and PrivateAuthority. This module grants no live authority.
"""

from gluomem_bench.bench.rack_observed import PROVIDER_POLICIES, PrivateAuthority
from gluomem_bench.bench.rack_observed_provider import VERSION


def reconcile_observed_provider_attempts(rows, physical_rows, *, ordinal_bindings=None):
    """Reconcile ONLY the supplied native subset, preserving shared/root ordinal gaps.

    The caller authenticates the partition and binding. Direct RackClient reader
    receipts are not native mirrors and MUST be joined separately at the root.
    """
    from copy import deepcopy

    from gluomem_bench.bench.rack_observed_replay import reconcile_physical_attempts

    mirrors = [r for r in rows if r.get("ordinal") is not None]
    if ordinal_bindings is None:
        ordinal_bindings = [
            {"native_ordinal": i, "root_ordinal": None} for i in range(1, len(physical_rows) + 1)
        ]
    if len(ordinal_bindings) != len(physical_rows) or len(mirrors) != len(physical_rows):
        raise ValueError("native subset/independent ordinal binding coverage differs")
    previous, roots, normalized = 0, set(), []
    for local, (row, binding) in enumerate(zip(mirrors, ordinal_bindings, strict=True), 1):
        if type(binding) is not dict or set(binding) != {"native_ordinal", "root_ordinal"}:
            raise ValueError("explicit immutable native/root ordinal binding required")
        native, root = binding["native_ordinal"], binding["root_ordinal"]
        if type(row["ordinal"]) is not int or row["ordinal"] < 1:
            raise ValueError("recorded native ordinal must be a positive integer")
        if (
            type(native) is not int
            or native <= previous
            or row["ordinal"] != native
            or root is not None
            and (type(root) is not int or root < 1 or root in roots)
        ):
            raise ValueError("native/root ordinal binding differs from actual dispatch")
        attempt = row.get("pool_attempts", [row.get("physical_attempt", {})])[-1]
        actual_root = attempt.get("root_ordinal")
        if actual_root is not None and (type(actual_root) is not int or actual_root < 1):
            raise ValueError("recorded root ordinal must be None or a positive integer")
        if actual_root != root:
            raise ValueError("native root ordinal differs from independent physical binding")
        previous = native
        if root is not None:
            roots.add(root)
        view = deepcopy(row)
        view["ordinal"] = local  # Accounting index only; originals are never rewritten.
        normalized.append(view)
    normalized.extend(deepcopy(r) for r in rows if r.get("ordinal") is None)
    totals = reconcile_physical_attempts(normalized, physical_rows)
    return {
        **totals,
        "ordinal_bindings": deepcopy(ordinal_bindings),
        "coverage": "supplied-native-physical-subset-only",
    }


def validate_observed_provider_history(
    rows,
    profile,
    cfg,
    protocol,
    expected,
    *,
    authority,
    prepared_events,
    source_events,
    physical_rows,
    attempt_rows,
    endpoint_metadata,
    expected_requests=None,
    ordinal_bindings=None,
    expected_endpoints=None,
    admitted_endpoint_tags=None,
):
    from gluomem_bench.bench.rack_diagnostic import validate_request
    from gluomem_bench.bench.rack_observed import validate_observed_response
    from gluomem_bench.bench.rack_observed_replay import (
        _parsed,
        replay_source_history,
    )
    from gluomem_bench.bench.rack_pool_replay import validate_pool_call
    from gluomem_bench.bench.rack_pricing_replay import validate_pricing_row
    from gluomem_bench.bench.rack_provider import (
        OpenRouterEndpoint,
        endpoint_discovery,
        estimated_prompt_tokens,
        pool_attempt_profile,
    )
    from gluomem_bench.bench.rack_schema import (
        transport_binding,
        validate_production_row,
        validate_writer_config,
    )

    if (
        profile.profile_version != VERSION
        or profile.semantic_producer_contract not in PROVIDER_POLICIES
        or type(authority) is not PrivateAuthority
        or endpoint_metadata is None
    ):
        raise ValueError("separately admitted v13 provider/private/pool history required")
    validate_writer_config(profile, cfg)
    endpoint_discovery(profile, endpoint_metadata, admitted_endpoint_tags=admitted_endpoint_tags)
    totals = reconcile_observed_provider_attempts(
        rows, physical_rows, ordinal_bindings=ordinal_bindings
    )
    if any("error" in row for row in rows):
        raise ValueError("recorded native operation error; physical accounting is not success")
    if (
        totals["pre_dispatch_refusals"]
        or totals["failed_http_attempts"]
        or totals["unknown_usage_attempts"]
        or len(attempt_rows) != (len(physical_rows) if profile.openrouter_pool else 0)
    ):
        raise ValueError("incomplete provider history; retained debit is not success")
    binding = transport_binding(
        profile, cfg["dspy"], capture_policy=cfg["write"]["source_capture_policy"]
    )
    if (
        expected is None
        or expected.get("rack_schema_transport") != binding
        or protocol.get("rack_schema_transport") != binding
        or any(row.get("model") != binding["lm_model"] for row in rows)
    ):
        raise ValueError("unbound v13 native primary transport/model")
    expected_requests = expected_requests or {}
    expected_endpoints = expected_endpoints if expected_endpoints is not None else {}
    if (
        type(expected_endpoints) is not dict
        or any(type(k) is not int or k < 1 for k in expected_endpoints)
        or set(expected_endpoints) - {r["ordinal"] for r in rows}
    ):
        raise ValueError("independent native endpoint binding coverage differs")
    if admitted_endpoint_tags is not None and set(expected_endpoints) != {
        r["ordinal"] for r in rows
    }:
        raise ValueError("explicit admitted route requires independent endpoint bindings")
    parsed = []
    records = attempt_rows if profile.openrouter_pool is not None else [None] * len(rows)
    for row, physical, record in zip(rows, physical_rows, records, strict=True):
        validate_pricing_row(row, profile, endpoint_metadata=endpoint_metadata,
                             admitted_endpoint_tags=admitted_endpoint_tags)
        serving_profile = profile
        if profile.openrouter_pool is not None:
            validate_pricing_row(record, profile, endpoint_metadata=endpoint_metadata,
                                 admitted_endpoint_tags=admitted_endpoint_tags)
            if (
                record.get("logical_ordinal") != row.get("logical_ordinal")
                or len(row.get("pool_attempts", [])) != 1
                or record.get("pool_attempt") != row["pool_attempts"][0]
                or record.get("task") != row.get("task")
            ):
                raise ValueError("physical/native ordinal/intent/attempt coverage differs")
            attempt = row["pool_attempts"][0]
            # Alternate routes need separate authenticated root/pre-send authority;
            # the stored attempt, response or provider declaration cannot grant it.
            supplied = expected_endpoints.get(row["ordinal"])
            bound = (
                OpenRouterEndpoint.model_validate(supplied)
                if row["ordinal"] in expected_endpoints
                else (
                    OpenRouterEndpoint.model_validate(
                        profile.openrouter_pool.endpoints[0].model_dump(
                            mode="json", exclude={"pricing"}
                        )
                    )
                )
            )
            members = [
                e
                for e in profile.openrouter_pool.endpoints
                if e.model_dump(mode="json", exclude={"pricing"}) == bound.model_dump(mode="json")
            ]
            if (
                len(members) != 1
                or admitted_endpoint_tags is not None
                and bound.tag not in admitted_endpoint_tags
            ):
                raise ValueError("independently bound endpoint is outside declared pool")
            serving_profile = pool_attempt_profile(profile, bound.tag)
            wire = attempt["wire"]
            if (
                attempt["ordinal"] != row["ordinal"]
                or attempt["endpoint"] != members[0].model_dump(mode="json")
                or any(
                    wire.get(k) != physical.get(k) or record.get(k) != physical.get(k)
                    for k in (
                        "url",
                        "request",
                        "request_raw_utf8",
                        "request_sha256",
                        "response",
                        "response_raw_utf8",
                        "response_sha256",
                        "status",
                    )
                )
            ):
                raise ValueError("independent physical bytes differ from pool/native mirrors")
            validate_pool_call(row, profile, expected_request=expected_requests.get(row["ordinal"]),
                endpoint_metadata=endpoint_metadata, admitted_endpoint_tags=admitted_endpoint_tags)
        else:
            from gluomem_bench.bench.rack_diagnostic_replay import validate_physical_row

            if (
                row["ordinal"] in expected_endpoints
                and OpenRouterEndpoint.model_validate(expected_endpoints[row["ordinal"]])
                != profile.openrouter_endpoint
            ):
                raise ValueError("independently bound singleton endpoint differs")
            validate_physical_row(row, profile)
            wire = row["physical_attempt"]["wire"]
            if any(
                wire.get(k) != physical.get(k)
                for k in (
                    "url",
                    "request",
                    "request_raw_utf8",
                    "request_sha256",
                    "response",
                    "response_raw_utf8",
                    "response_sha256",
                    "status",
                )
            ):
                raise ValueError("independent physical bytes differ from singleton/native mirror")
        request = physical["request"]
        validate_request(
            serving_profile,
            request,
            messages=row["messages"],
            output=profile.output_budget(row["task"]),
            response_format=row["generation_kwargs"].get("response_format"),
        )
        if row.get("prompt_token_count") != estimated_prompt_tokens(request["messages"]):
            raise ValueError("physical conservative UTF8 token estimate differs")
        validate_observed_response(row, profile)
        if row["task"] in {"answer", "rubric"}:
            if expected_requests.get(row["ordinal"]) != request:
                raise ValueError("unbound plain reader/judge request")
            parsed.append(None)
        else:
            validate_production_row(row, profile, cfg, protocol, expected, authority=authority)
            parsed.append(_parsed(row, cfg, authority))
    return {
        **totals,
        **replay_source_history(rows, parsed, authority, prepared_events, source_events,
                                producer_contract=profile.semantic_producer_contract),
        "protocol": "observed-provider-history-replay@v13",
        "model_quality_admitted": False,
        "execution_authorized": False,
    }
