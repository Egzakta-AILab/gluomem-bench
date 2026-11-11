"""Explicit DeepSeek observed native profile. No transport, allowance or fallback here."""

import re

VERSION = "observed-provider-v13"
POLICY = "observed-assertion-date-v1"


def validate_profile(profile):
    from gluomem_bench.bench.rack_provider import DEEPSEEK

    if (
        profile.provider != "openrouter"
        or profile.model != DEEPSEEK
        or profile.requested_model != DEEPSEEK
        or not profile.api_key_env
        or profile.token_count_policy != "utf8-conservative-v1"
        or profile.context_window != 1048576
        or profile.semantic_producer_contract not in {
            POLICY,
            "observed-property-identity-v3",
            "observed-assertion-scope-v4",
        }
        or profile.writer_contract != "semantic-state-writer-v4"
        or profile.identity_output_tokens is None
        or profile.generation_budgets is None
        or profile.native_schema_strict is not False
        or profile.production_json_transport is not None
        or profile.native_diagnostic is not None
        or profile.pricing_policy is None
        or profile.pricing_policy.version != "bounded-provider-prices-v2"
        or (profile.openrouter_pool is None) == (profile.openrouter_endpoint is None)
    ):
        raise ValueError("v13 requires exact DeepSeek observed FP8 native nonstrict bounded policy")
    if profile.openrouter_pool is not None:
        if profile.openrouter_pool.max_attempts_per_logical_call != 1:
            raise ValueError("v13 forbids retries/provider fallback; one original attempt only")
    else:
        endpoint = profile.openrouter_endpoint
        if endpoint.quantization != "fp8" or not re.fullmatch(
            r"[a-z0-9]+(?:-[a-z0-9]+)*/fp8", endpoint.tag
        ):
            raise ValueError("v13 requires an explicit standard FP8 endpoint")
    caps = [profile.identity_output_tokens, *profile.generation_budgets.model_dump().values()]
    if (
        max(caps) > profile.pricing_policy.output_token_cap
        or max(caps) + profile.schema_reserve_tokens > profile.context_window
    ):
        raise ValueError("v13 role/identity output exceeds bounded native allowance")


def singleton_intent(profile, row):
    from gluomem_bench.bench.rack_pool import digest

    return digest(
        [
            VERSION,
            profile.base_url,
            profile.model,
            row["task"],
            row["ordinal"],
            row.get("root_ordinal"),
        ]
    )


def provider_view(profile):
    """Pure existing metadata/route policy view, never another model or native engine."""
    from gluomem_bench.bench.rack_provider import ENDPOINT_VERSION, POOL_VERSION

    if profile.profile_version != VERSION:
        raise ValueError("explicit observed provider profile required")
    value = profile.model_dump(mode="json")
    for field in ("semantic_producer_contract", "identity_output_tokens"):
        value.pop(field)
    value.update(
        profile_version=POOL_VERSION if profile.openrouter_pool is not None else ENDPOINT_VERSION,
        judge_profile=None,
    )
    if profile.openrouter_pool is None:
        value.pop("native_schema_strict")
    return type(profile).model_validate(value)
