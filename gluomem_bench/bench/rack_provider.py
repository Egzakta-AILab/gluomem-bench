"""Explicit opt-in provider contracts; no discovery, fallback or private defaults."""

from __future__ import annotations

import json
import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VERSION = "provider-roles-v8"
ENDPOINT_VERSION = "provider-roles-v9"
POOL_VERSION = "provider-roles-v10"
DIAGNOSTIC_VERSION = "provider-roles-v11"
# Only the RackClient/native GluoMem production path is qualified for v9.
PRODUCTION_PROVIDER_VERSIONS = frozenset(
    {VERSION, ENDPOINT_VERSION, POOL_VERSION, DIAGNOSTIC_VERSION}
)
DEEPSEEK = "deepseek/deepseek-v4.1-flash-1017"
QWEN_ALIAS = "qwen-3.8-27b"


class OpenRouterEndpoint(BaseModel):
    """Explicit exact metadata identifiers; no provider-specific enum or defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    tag: str = Field(min_length=1, pattern=r"^\S+$", strict=True)
    provider_name: str = Field(min_length=1, strict=True)
    quantization: str = Field(min_length=1, pattern=r"^\S+$", strict=True)

    @field_validator("provider_name")
    @classmethod
    def exact_name(cls, value):
        if value.strip() != value or any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError("explicit exact provider name required")
        return value


class OpenRouterPoolPricing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    prompt: str = Field(strict=True, pattern=r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
    completion: str = Field(strict=True, pattern=r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
    request: Literal["0"]


class OpenRouterPoolEndpoint(OpenRouterEndpoint):
    pricing: OpenRouterPoolPricing


class OpenRouterPool(BaseModel):
    """Frozen standard-tier FP8 scope, not a claim of live qualification."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: Literal["client-owned-fp8-pool-v1"]
    endpoints: tuple[OpenRouterPoolEndpoint, ...] = Field(min_length=6, max_length=6)
    max_attempts_per_logical_call: int = Field(default=6, ge=1, le=6, strict=True)
    logical_deadline_seconds: float = Field(default=300, gt=0, le=300, allow_inf_nan=False)
    cooldown_seconds: float = Field(default=1, gt=0, le=30, allow_inf_nan=False)
    max_backoff_seconds: float = Field(default=30, gt=0, le=300, allow_inf_nan=False)
    jitter_fraction: float = Field(default=0.1, ge=0, le=0.25, allow_inf_nan=False)

    @model_validator(mode="after")
    def standard_fp8_scope(self):
        if (
            len({e.tag for e in self.endpoints}) != len(self.endpoints)
            or len({e.provider_name for e in self.endpoints}) != len(self.endpoints)
            or any(
                e.quantization != "fp8" or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*/fp8", e.tag)
                for e in self.endpoints
            )
        ):
            raise ValueError("v10 pool requires six distinct explicit standard FP8 endpoints")
        if self.cooldown_seconds > self.max_backoff_seconds:
            raise ValueError("pool cooldown must fit bounded backoff")
        return self


def pool_attempt_profile(profile, endpoint_tag):
    """Derive one validated singleton pin; never choose an undeclared endpoint."""
    if (
        profile.profile_version not in {POOL_VERSION, "observed-provider-v13"}
        or profile.openrouter_pool is None
    ):
        raise ValueError("explicit v10 pool profile required")
    selected = [e for e in profile.openrouter_pool.endpoints if e.tag == endpoint_tag]
    if len(selected) != 1:
        raise ValueError("endpoint is not in the frozen pool")
    fields = profile.model_dump(mode="json")
    fields.pop("openrouter_pool")
    # Strictness is bound to the enclosing invocation/schema, not a v9 route pin.
    # The physical LM receives that exact response_format from its v10 parent.
    if profile.profile_version != "observed-provider-v13":
        fields.pop("native_schema_strict", None)
    endpoint = selected[0].model_dump(mode="json", exclude={"pricing"})
    fields.update(
        profile_version=(
            "observed-provider-v13"
            if profile.profile_version == "observed-provider-v13"
            else ENDPOINT_VERSION
        ),
        openrouter_endpoint=endpoint,
        judge_profile=None,
    )
    return type(profile).model_validate(fields)


def pool_attempt_options(profile, endpoint_tag):
    return generation_options(pool_attempt_profile(profile, endpoint_tag))


def is_openrouter(profile):
    return profile.provider in {"openrouter-baidu-fp8", "openrouter"}


def refuse_unqualified_endpoint_comparator(profile):
    if getattr(profile, "semantic_producer_contract", None) is not None:
        raise ValueError("observed producer comparator native boundary unqualified")
    if getattr(profile, "profile_version", None) == DIAGNOSTIC_VERSION:
        raise ValueError("v11 diagnostic comparator native boundary unqualified")
    if getattr(profile, "profile_version", None) == ENDPOINT_VERSION:
        raise ValueError("v9 endpoint comparator native boundary unqualified")
    if getattr(profile, "profile_version", None) == POOL_VERSION:
        raise ValueError("v10 pool comparator native boundary unqualified")


def selected_endpoint(profile):
    """Keep the immutable v8 pin; v9 must supply every endpoint identifier."""
    if (
        profile.profile_version in {POOL_VERSION, "observed-provider-v13"}
        and profile.openrouter_pool is not None
        and profile.provider == "openrouter"
    ):
        # Initial capability/config pin only. Physical dispatch selects explicitly
        # through pool_attempt_profile and the shared pool attempt context.
        return profile.openrouter_pool.endpoints[0]
    if (
        profile.profile_version in {ENDPOINT_VERSION, DIAGNOSTIC_VERSION, "observed-provider-v13"}
        and profile.provider == "openrouter"
    ):
        if profile.openrouter_endpoint is None:
            raise ValueError("v9 requires an explicit OpenRouter endpoint")
        return profile.openrouter_endpoint
    if profile.profile_version == VERSION and profile.provider == "openrouter-baidu-fp8":
        return OpenRouterEndpoint(tag="baidu/fp8", provider_name="Baidu", quantization="fp8")
    raise ValueError("no admitted OpenRouter endpoint for this profile")


def generation_options(profile):
    if is_openrouter(profile):
        endpoint = selected_endpoint(profile)
        from gluomem_bench.bench.rack_pricing import wire_price_limits

        price_bound = (
            {"max_price": wire_price_limits(profile.pricing_policy)}
            if profile.pricing_policy is not None
            else {}
        )
        if profile.profile_version == DIAGNOSTIC_VERSION:
            return {
                "provider": {
                    "only": [endpoint.tag],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                    **price_bound,
                    **(
                        {"quantizations": [endpoint.quantization]}
                        if endpoint.quantization != "unknown"
                        else {}
                    ),
                },
                "reasoning": profile.native_diagnostic.reasoning.model_dump(mode="json"),
            }
        return {
            "provider": {
                "only": [endpoint.tag],
                "allow_fallbacks": False,
                "require_parameters": True,
                "quantizations": [endpoint.quantization],
                **price_bound,
            },
            "reasoning": {"enabled": False},
        }
    return {"chat_template_kwargs": {"enable_thinking": False}}


def validate_provider_profile(profile):
    """The new version must declare a complete, exact condition, never infer one."""
    if not profile.provider or not profile.api_key_env or not profile.requested_model:
        raise ValueError("provider profile requires provider, requested_model and api_key_env")
    if profile.profile_version == "observed-provider-v13":
        from gluomem_bench.bench.rack_observed_provider import validate_profile

        validate_profile(profile)
        return
    if profile.profile_version == DIAGNOSTIC_VERSION:
        from gluomem_bench.bench.rack_diagnostic import validate_profile

        validate_profile(profile)
        if urlsplit(profile.base_url).path != "/api/v1":
            raise ValueError("v11 requires the exact OpenRouter /api/v1 path")
        return
    if profile.profile_version == POOL_VERSION:
        if (
            profile.provider != "openrouter"
            or profile.openrouter_pool is None
            or profile.openrouter_endpoint is not None
            or profile.production_json_transport is not None
        ):
            raise ValueError("v10 requires only an explicit native-schema OpenRouter pool")
    elif profile.openrouter_pool is not None:
        raise ValueError("legacy profile cannot opt into v10 pool fields")
    elif profile.profile_version == ENDPOINT_VERSION:
        if profile.provider != "openrouter" or profile.openrouter_endpoint is None:
            raise ValueError("v9 requires generic openrouter and an explicit endpoint")
        if profile.production_json_transport is not None:
            raise ValueError("v9 production requires native structured schema transport")
    elif profile.provider == "openrouter" or profile.openrouter_endpoint is not None:
        raise ValueError("legacy profile cannot opt into v9 OpenRouter endpoint fields")
    path = urlsplit(profile.base_url).path
    if is_openrouter(profile):
        if (
            profile.model != DEEPSEEK
            or profile.requested_model != DEEPSEEK
            or path != "/api/v1"
            or profile.context_window != 1048576
            or profile.token_count_policy != "utf8-conservative-v1"
        ):
            raise ValueError("exact DeepSeek Baidu /api/v1 FP8 condition required")
    elif (
        profile.requested_model != QWEN_ALIAS
        or path != "/v1"
        or profile.context_window != 131072
        or profile.token_count_policy != "served-tokenizer"
    ):
        raise ValueError(
            "explicit Qwen requested alias and served-tokenizer /v1 condition required"
        )


def estimated_prompt_tokens(messages):
    """Conservative text estimate, NOT an identified tokenizer or template parity.

    UTF-8 bytes plus explicit envelope headroom. Refuse nontext modalities rather
    than assign invented image costs. Actual provider usage is checked on return.
    """
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(m, dict)
            or set(m) != {"role", "content"}
            or m["role"] not in {"system", "user", "assistant"}
            or not isinstance(m["content"], str)
            for m in messages
        )
    ):
        raise ValueError("provider token estimate supports plain text messages only")
    return (
        len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 256 * len(messages) + 1024
    )


def response_model_matches(profile, returned_model):
    """Only the explicit source-bound v11 policy admits the second catalogue literal."""
    if returned_model == profile.model:
        return True
    from gluomem_bench.bench.rack_diagnostic import RESPONSE_IDENTITY_POLICY

    policy = profile.native_diagnostic
    return (
        profile.profile_version == DIAGNOSTIC_VERSION
        and policy is not None
        and policy.response_identity_policy == RESPONSE_IDENTITY_POLICY
        and type(returned_model) is str
        and returned_model == policy.canonical_slug
    )


def response_evidence(profile, response, admitted_tokens, *, output_budget):
    """Require returned provider plus actual usage; precision remains declarative."""
    if not is_openrouter(profile):
        return {}
    from gluomem_bench.bench.rack import FatalRackError, SharedRackError

    if profile.profile_version == "observed-provider-v13" and response.get("service_tier") not in (
        None,
        "standard",
    ):
        raise SharedRackError("observed provider returned service tier is not standard")
    endpoint = selected_endpoint(profile)
    if not response_model_matches(profile, response.get("model")):
        raise SharedRackError("response model differs from exact provider profile")
    if response.get("provider") != endpoint.provider_name:
        raise SharedRackError(
            "missing or different returned provider evidence; "
            + endpoint.provider_name
            + " required"
        )
    choices = response.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or choices[0].get("finish_reason") != "stop"
    ):
        raise FatalRackError("incomplete raw provider response; one explicit stop required")
    usage = response.get("usage") or {}
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if any(type(n) is not int or n < 0 for n in (prompt, completion)):
        raise SharedRackError("actual provider token usage required")
    if profile.profile_version == "observed-provider-v13":
        total = usage.get("total_tokens")
        message = choices[0].get("message", {})
        if type(total) is not int or total != prompt + completion:
            raise SharedRackError("observed exact integer total usage required")
        if (
            message.get("role") != "assistant"
            or type(message.get("content")) is not str
            or message.get("refusal") is not None
            or message.get("tool_calls")
        ):
            raise FatalRackError("observed raw assistant/refusal output cannot be successful")
    if completion > output_budget:
        raise SharedRackError("provider output token usage exceeds admitted role cap")
    if profile.native_diagnostic is not None:
        # A returned literal only: never send a tier override or rewrite raw data.
        tier = response.get("service_tier")
        alias = profile.native_diagnostic.returned_service_tier_policy
        if tier not in (None, "standard") and not (
            alias == "openai-default-is-standard-v1" and type(tier) is str and tier == "default"
        ):
            raise SharedRackError("diagnostic returned service tier is not standard")
        total = usage.get("total_tokens")
        if type(total) is not int or total != prompt + completion:
            raise SharedRackError("diagnostic exact integer total usage required")
        if prompt > profile.native_diagnostic.prompt_token_cap:
            raise SharedRackError("diagnostic actual prompt exceeds sub-tier allowance")
    if (
        prompt > admitted_tokens
        or prompt + completion + profile.schema_reserve_tokens > profile.context_window
    ):
        raise SharedRackError("provider usage exceeds admitted conservative context estimate")
    evidence = {
        "provider_evidence": {
            "requested_endpoint": endpoint.tag,
            "returned_provider": endpoint.provider_name,
            "requested_quantizations": [endpoint.quantization],
            "precision_attestation": "provider-declared; not independently verified",
        },
        "token_accounting": {
            "method": profile.token_count_policy,
            "admitted_prompt_tokens": admitted_tokens,
            "actual_prompt_tokens": prompt,
            "actual_completion_tokens": completion,
            "tokenizer_parity": False,
        },
    }
    if profile.native_diagnostic is not None:
        evidence["provider_evidence"].update(
            requested_quantizations=(
                [] if endpoint.quantization == "unknown" else [endpoint.quantization]
            ),
            declared_quantization=endpoint.quantization,
            requested_model=profile.requested_model,
            returned_model=response["model"],
            returned_service_tier=response.get("service_tier"),
            service_tier_evidence="returned" if "service_tier" in response else "absent",
            native_schema_enforcement="not-guaranteed",
        )
    return evidence


def validate_lm_policy(profile, lm):
    """Check the effective native dispatch policy, not merely its factory helper."""
    if profile.provider is None:
        return
    body = generation_options(profile)
    name = "reasoning" if is_openrouter(profile) else "chat_template_kwargs"
    allowed = {name, "api_base", "api_key", "timeout", "max_tokens", "temperature", "n"}
    if profile.native_diagnostic is not None:
        from gluomem_bench.bench.rack_diagnostic import sampling_options

        allowed.add("top_p")
        actual = {k: lm.kwargs[k] for k in ("temperature", "top_p") if k in lm.kwargs}
        if actual != sampling_options(profile):
            raise ValueError("effective diagnostic sampling policy differs from frozen profile")
    if (
        set(lm.kwargs) - allowed
        or lm.kwargs.get("n", 1) != 1
        or json.dumps(lm.kwargs.get(name), sort_keys=True) != json.dumps(body[name], sort_keys=True)
        or lm.num_retries != 0
        or lm.cache
        or lm._engine_spec != "lm15"
    ):
        raise ValueError("effective provider policy differs from frozen profile")


def _endpoint_discovery(profile, response, *, admitted_endpoint_tags=None):
    """Project the selected public endpoint into the existing rack discovery seam.

    The original metadata remains attached and in the actual HTTP receipt. This
    projection is not a synthetic server /models response or weight attestation.
    """
    from gluomem_bench.bench.rack import SharedRackError

    if profile.profile_version == "observed-provider-v13":
        from gluomem_bench.bench.rack_observed_provider import provider_view

        projection = _endpoint_discovery(
            provider_view(profile), response, admitted_endpoint_tags=admitted_endpoint_tags
        )
        if any(
            type(e.get("max_completion_tokens")) is not int
            or e["max_completion_tokens"] < profile.identity_output_tokens
            for e in projection["data"]
        ):
            raise SharedRackError("observed identity output exceeds endpoint capacity")
        return projection
    if profile.profile_version == DIAGNOSTIC_VERSION:
        from gluomem_bench.bench.rack_diagnostic import validate_endpoint_metadata

        selected = validate_endpoint_metadata(profile, response)
        return {
            "data": [
                {
                    **selected,
                    "id": selected["model_id"],
                    "max_model_len": selected["context_length"],
                }
            ],
            "discovery_projection": "openrouter-diagnostic-to-rack@v1",
            "provider_metadata": response,
        }
    if profile.profile_version == POOL_VERSION:
        projected = []
        try:
            for member in profile.openrouter_pool.endpoints:
                if admitted_endpoint_tags is not None and member.tag not in admitted_endpoint_tags:
                    continue
                single = pool_attempt_profile(profile, member.tag)
                selected = endpoint_discovery(
                    single,
                    response,
                    admitted_endpoint_tags=None
                    if admitted_endpoint_tags is None
                    else (member.tag,),
                )["data"][0]
                if (
                    selected.get("status") != 0
                    or type(selected.get("status")) is not int
                    or selected.get("service_tier") not in (None, "standard")
                    or type(selected.get("max_completion_tokens")) is not int
                    or selected["max_completion_tokens"]
                    < max(profile.generation_budgets.model_dump().values())
                ):
                    raise ValueError("pool endpoint capacity/status/tier mismatch")
                raw_prices = selected["pricing"]
                frozen = member.pricing.model_dump(mode="json")
                if profile.pricing_policy is not None:
                    from gluomem_bench.bench.rack_pricing import validate_prices

                    validate_prices(profile.pricing_policy, raw_prices)
                elif any(
                    raw_prices.get(key, "0" if key == "request" else None) != value
                    for key, value in frozen.items()
                ):
                    raise ValueError("pool endpoint price drift")
                projected.append(selected)
        except (KeyError, ValueError, TypeError):
            raise SharedRackError(
                "pool public endpoint metadata/price differs from frozen contract"
            ) from None
        return {
            "data": projected,
            "discovery_projection": "openrouter-pool-to-rack@v1",
            "provider_metadata": response,
        }
    endpoint = selected_endpoint(profile)
    data = response.get("data", {})
    endpoints = data.get("endpoints", []) if isinstance(data, dict) else []
    matches = [e for e in endpoints if isinstance(e, dict) and e.get("tag") == endpoint.tag]
    required = {
        "temperature",
        "max_tokens",
        "response_format",
        "structured_outputs",
        "reasoning",
        "top_p",
    }
    if (
        data.get("id") != profile.model
        or len(matches) != 1
        or matches[0].get("model_id") != profile.model
        or matches[0].get("provider_name") != endpoint.provider_name
        or matches[0].get("quantization") != endpoint.quantization
        or matches[0].get("context_length") != profile.context_window
        or not required.issubset(matches[0].get("supported_parameters", []))
    ):
        raise SharedRackError(
            "public exact selected OpenRouter endpoint metadata differs from frozen condition"
        )
    selected = matches[0]
    return {
        "data": [
            {**selected, "id": selected["model_id"], "max_model_len": selected["context_length"]}
        ],
        "discovery_projection": "openrouter-endpoint-to-rack@v1",
        "provider_metadata": response,
    }


def admitted_routes(profile, admitted_endpoint_tags):
    """Validate an explicit scope, never infer routing authority from metadata."""
    from gluomem_bench.bench.rack import SharedRackError

    if admitted_endpoint_tags is None:
        return None
    known = (
        {e.tag for e in profile.openrouter_pool.endpoints}
        if profile.openrouter_pool is not None
        else {selected_endpoint(profile).tag}
    )
    if (
        type(admitted_endpoint_tags) not in (tuple, list)
        or not admitted_endpoint_tags
        or any(type(tag) is not str for tag in admitted_endpoint_tags)
        or len(set(admitted_endpoint_tags)) != len(admitted_endpoint_tags)
        or not set(admitted_endpoint_tags) <= known
    ):
        raise SharedRackError("explicit admitted endpoint scope invalid")
    return tuple(admitted_endpoint_tags)


def endpoint_discovery(profile, response, *, admitted_endpoint_tags=None):
    """Preserve default projection; explicit opt-in adds current quote evidence."""
    admitted_endpoint_tags = admitted_routes(profile, admitted_endpoint_tags)
    result = _endpoint_discovery(profile, response, admitted_endpoint_tags=admitted_endpoint_tags)
    if admitted_endpoint_tags is not None:
        from gluomem_bench.bench.rack import SharedRackError

        for endpoint in result["data"]:
            if (
                type(endpoint.get("status")) is not int
                or endpoint["status"] != 0
                or endpoint.get("service_tier") not in (None, "standard")
                or type(endpoint.get("max_completion_tokens")) is not int
                or endpoint["max_completion_tokens"]
                < max(profile.generation_budgets.model_dump().values())
                or endpoint.get("max_prompt_tokens") is not None
                and (
                    type(endpoint["max_prompt_tokens"]) is not int
                    or endpoint["max_prompt_tokens"] <= 0
                )
                or any(
                    type(endpoint.get(key)) is not str
                    for key in ("tag", "model_id", "provider_name", "quantization")
                )
                or type(endpoint.get("context_length")) is not int
                or type(endpoint.get("supported_parameters")) is not list
                or any(type(p) is not str for p in endpoint["supported_parameters"])
            ):
                raise SharedRackError("admitted endpoint strict metadata types required")
    if profile.pricing_policy is not None:
        from gluomem_bench.bench.rack import SharedRackError
        from gluomem_bench.bench.rack_pricing import price_observations

        try:
            result["pricing_policy"] = profile.pricing_policy.model_dump(mode="json")
            result["pricing_observations"] = price_observations(profile, response, result["data"])
        except (KeyError, ValueError, TypeError):
            raise SharedRackError("bounded provider pricing metadata/capacity refused") from None
    return result
