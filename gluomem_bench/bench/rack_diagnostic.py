"""Explicit native non-strict diagnostic policy; not benchmark admission.

Snapshots bind public declarations, not provider enforcement or observed quality.
No discovery, network, credentials, schema rewrites or output repair live here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer

VERSION = "provider-roles-v11"
TERRA = "openai/gpt-6.1-terra"
RESPONSE_IDENTITY_POLICY = "catalog-alias-canonical-v1"

TRANSPORT = "declared-native-json-schema-nonstrict-v1"
_DIGEST = r"^[0-9a-f]{64}$"
_ACTIVE = ContextVar("rack_native_diagnostic_attempt", default=None)
_CAPTURE_LOCK = threading.RLock()


class DisabledReasoning(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: Literal[False]


class NoEffortReasoning(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    effort: Literal["none"]


class DiagnosticPricing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    prompt: str = Field(strict=True, pattern=r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
    completion: str = Field(strict=True, pattern=r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
    request: Literal["0"]


class NativeDiagnosticPolicy(BaseModel):
    """Required declared controls: null means OMIT, never an SDK default value."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: Literal["targeted-memory-native-v1"]
    schema_strict: Literal[False]
    temperature: Literal[0] | None
    top_p: Literal[1] | None
    reasoning: DisabledReasoning | NoEffortReasoning
    canonical_slug: str = Field(strict=True, min_length=1, pattern=r"^\S+$")
    prompt_token_cap: int = Field(strict=True, ge=1, le=65536)
    pricing: DiagnosticPricing
    request_price_source: Literal["absent-sku-documented-zero-v1", "explicit-zero-sku-v1"]
    endpoint_source_sha256: str = Field(pattern=_DIGEST)
    model_source_sha256: str = Field(pattern=_DIGEST)
    endpoint_contract_sha256: str = Field(pattern=_DIGEST)
    model_contract_sha256: str = Field(pattern=_DIGEST)
    response_identity_policy: Literal["catalog-alias-canonical-v1"] | None = None
    returned_service_tier_policy: Literal["openai-default-is-standard-v1"] | None = None
    pricing_identity_policy: (
        Literal["bounded-provider-prices-v1", "bounded-provider-prices-v2"] | None
    ) = None

    @model_serializer(mode="wrap")
    def _serialize_optional_identity(self, handler):
        value = handler(self)
        if self.response_identity_policy is None:
            value.pop("response_identity_policy", None)
        if self.returned_service_tier_policy is None:
            value.pop("returned_service_tier_policy", None)
        if self.pricing_identity_policy is None:
            value.pop("pricing_identity_policy", None)
        return value


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def endpoint_material(endpoint, *, response_identity_policy=None, pricing_policy=None):
    keys = (
        "model_id",
        "tag",
        "provider_name",
        "quantization",
        "context_length",
        "service_tier",
        "status",
        "max_completion_tokens",
        "max_prompt_tokens",
        "pricing",
    )
    result = {k: endpoint[k] for k in keys}
    if pricing_policy is not None:
        result.pop("pricing")
    if response_identity_policy == RESPONSE_IDENTITY_POLICY:
        result["name"] = endpoint["name"]
    result["supported_parameters"] = sorted(endpoint["supported_parameters"])
    return result


def model_material(model, *, pricing_policy=None):
    result = {
        k: model[k]
        for k in (
            "id",
            "canonical_slug",
            "context_length",
            "pricing",
            "supported_parameters",
            "reasoning",
        )
    }
    if pricing_policy is not None:
        result.pop("pricing")
    return result


def diagnostic_policy_from_metadata(
    endpoints,
    model,
    *,
    endpoint_tag,
    prompt_token_cap,
    temperature,
    top_p,
    reasoning,
    response_identity_policy=None,
    pricing_policy=None,
    returned_service_tier_policy=None,
):
    """Construct an explicit freeze from supplied public JSON, with no IO/default controls."""
    selected = [e for e in endpoints["data"]["endpoints"] if e["tag"] == endpoint_tag]
    if len(selected) != 1 or endpoints["data"]["id"] != model["id"]:
        raise ValueError("exact unique diagnostic metadata required")
    endpoint = selected[0]
    if pricing_policy is not None:
        from gluomem_bench.bench.rack_pricing import ProviderPricingPolicy, validate_prices
        pricing_policy = ProviderPricingPolicy.model_validate(pricing_policy)
        if prompt_token_cap != pricing_policy.prompt_token_cap:
            raise ValueError("diagnostic and bounded pricing prompt caps must match")
        validate_prices(pricing_policy, endpoint["pricing"])
        validate_prices(pricing_policy, model["pricing"])
    if response_identity_policy is not None:
        if response_identity_policy != RESPONSE_IDENTITY_POLICY:
            raise ValueError("unsupported diagnostic response identity policy")
        if (
            endpoint.get("model_id") != model["id"]
            or type(model.get("canonical_slug")) is not str
            or endpoint.get("name")
            != endpoint.get("provider_name", "") + " | " + model["canonical_slug"]
        ):
            raise ValueError("source-bound endpoint/canonical identity required")
    request = endpoint["pricing"].get("request")
    if pricing_policy is None and "request" in endpoint["pricing"] and request != "0":
        raise ValueError("diagnostic request SKU is not absent or explicit decimal zero")
    if reasoning == {"effort": "none"} and "none" not in model.get("reasoning", {}).get(
        "supported_efforts", []
    ):
        raise ValueError(
            "diagnostic reasoning effort none is not advertised by frozen model metadata"
        )
    return NativeDiagnosticPolicy(
        protocol="targeted-memory-native-v1",
        schema_strict=False,
        temperature=temperature,
        top_p=top_p,
        reasoning=reasoning,
        canonical_slug=model["canonical_slug"],
        prompt_token_cap=prompt_token_cap,
        pricing={
            "prompt": endpoint["pricing"]["prompt"],
            "completion": endpoint["pricing"]["completion"],
            "request": "0",
        },
        request_price_source=(
            "explicit-zero-sku-v1"
            if "request" in endpoint["pricing"]
            else "absent-sku-documented-zero-v1"
        ),
        endpoint_source_sha256=digest(endpoints),
        model_source_sha256=digest(model),
        endpoint_contract_sha256=digest(
            endpoint_material(endpoint, response_identity_policy=response_identity_policy,
                              pricing_policy=pricing_policy)
        ),
        model_contract_sha256=digest(model_material(model, pricing_policy=pricing_policy)),
        response_identity_policy=response_identity_policy,
        returned_service_tier_policy=returned_service_tier_policy,
        pricing_identity_policy=pricing_policy.version if pricing_policy is not None else None,
    )


def validate_profile(profile):
    """Only the exact Terra singleton; DeepSeek keeps its existing v10 pool path."""
    policy = profile.native_diagnostic
    endpoint = profile.openrouter_endpoint
    bounded = profile.pricing_policy
    if policy is not None and policy.pricing_identity_policy != (
            bounded.version if bounded is not None else None):
        raise ValueError("pricing identity projection differs; rebuild from source metadata")
    if bounded is not None and policy is not None:
        from gluomem_bench.bench.rack_pricing import validate_prices
        validate_prices(bounded, policy.pricing.model_dump(mode="json"))
        if policy.prompt_token_cap != bounded.prompt_token_cap:
            raise ValueError("pricing identity prompt cap differs; rebuild from metadata")
    declared = {
        TERRA: (
            1050000,
            "openai/gpt-6.1-terra-20261026",
            "openai",
            "OpenAI",
            "unknown",
            None,
            None,
            {"effort": "none"},
        ),
    }.get(profile.model)
    if (
        declared is None
        or policy is None
        or endpoint is None
        or profile.provider != "openrouter"
        or profile.requested_model != profile.model
        or profile.openrouter_pool is not None
        or profile.production_json_transport is not None
        or profile.token_count_policy != "utf8-conservative-v1"
        or not profile.api_key_env
    ):
        raise ValueError("v11 requires an exact declared native singleton diagnostic")
    actual = (
        profile.context_window,
        policy.canonical_slug,
        endpoint.tag,
        endpoint.provider_name,
        endpoint.quantization,
        policy.temperature,
        policy.top_p,
        policy.reasoning.model_dump(mode="json"),
    )
    if actual != declared:
        raise ValueError("v11 diagnostic identity/declared controls differ from admitted condition")
    if bounded is None and profile.model == TERRA and policy.pricing.model_dump() != {
        "prompt": "0.000001",
        "completion": "0.000005",
        "request": "0",
    }:
        raise ValueError("v11 Terra standard sub-tier prices required")


def sampling_options(profile):
    """Return physical controls, not logical legacy task placeholders."""
    policy = getattr(profile, "native_diagnostic", None)
    if policy is None:
        return {"temperature": profile.temperature}
    return {
        k: v
        for k, v in (("temperature", policy.temperature), ("top_p", policy.top_p))
        if v is not None
    }


def validate_sampling_overrides(profile, kwargs):
    declared = sampling_options(profile)
    if any(
        k in kwargs and (k not in declared or kwargs[k] != declared[k])
        for k in ("temperature", "top_p")
    ):
        raise ValueError("unsupported diagnostic sampling override")


def validate_endpoint_metadata(profile, response):
    """Match stable source-bound metadata, retaining original raw snapshots upstream."""
    from gluomem_bench.bench.rack import SharedRackError

    try:
        data = response["data"]
        pin, policy = profile.openrouter_endpoint, profile.native_diagnostic
        selected = [e for e in data["endpoints"] if e["tag"] == pin.tag]
        if len(selected) != 1 or data["id"] != profile.model:
            raise ValueError("missing exact endpoint")
        endpoint = selected[0]
        if (
            policy.response_identity_policy == RESPONSE_IDENTITY_POLICY
            and endpoint.get("name") != pin.provider_name + " | " + policy.canonical_slug
        ):
            raise ValueError("diagnostic endpoint/canonical identity drift")
        required = {
            "max_tokens",
            "response_format",
            "structured_outputs",
            "reasoning",
            *sampling_options(profile),
        }
        if (
            endpoint["model_id"] != profile.model
            or endpoint["provider_name"] != pin.provider_name
            or endpoint["quantization"] != pin.quantization
            or type(endpoint["status"]) is not int
            or endpoint["status"] != 0
            or endpoint["service_tier"] is not None
            or endpoint["context_length"] != profile.context_window
            or not required <= set(endpoint["supported_parameters"])
            or type(endpoint["max_completion_tokens"]) is not int
            or max(profile.generation_budgets.model_dump().values())
            > endpoint["max_completion_tokens"]
            or endpoint["max_prompt_tokens"] is not None
            and (
                type(endpoint["max_prompt_tokens"]) is not int
                or policy.prompt_token_cap > endpoint["max_prompt_tokens"]
            )
            or digest(
                endpoint_material(
                    endpoint, response_identity_policy=policy.response_identity_policy,
                    pricing_policy=profile.pricing_policy
                )
            )
            != policy.endpoint_contract_sha256
        ):
            raise ValueError("diagnostic metadata drift")
        tiers = endpoint["pricing"].get("overrides", [])
        if profile.pricing_policy is not None:
            from gluomem_bench.bench.rack_pricing import validate_prices
            validate_prices(profile.pricing_policy, endpoint["pricing"])
        elif any(
            type(t.get("min_prompt_tokens")) is not int
            or t["min_prompt_tokens"] <= policy.prompt_token_cap
            for t in tiers
        ):
            raise ValueError("diagnostic prompt allowance crosses an unqualified price tier")
        return endpoint
    except (ValueError, KeyError, TypeError):
        raise SharedRackError("v11 diagnostic endpoint metadata/price/tier/control drift") from None


def validate_request(profile, request, *, messages, output, response_format=None):
    """Exact effective physical controls; omitted unsupported controls stay absent."""
    from gluomem_bench.bench.rack_provider import estimated_prompt_tokens, generation_options

    if profile.pricing_policy is not None and (
            estimated_prompt_tokens(messages) > profile.pricing_policy.prompt_token_cap
            or type(output) is not int
            or not 1 <= output <= profile.pricing_policy.output_token_cap):
        raise ValueError("pricing physical prompt/output allowance exceeded")
    wanted = {
        "model": profile.model,
        "messages": messages,
        "max_tokens": output,
        **generation_options(profile),
        **sampling_options(profile),
    }
    if response_format is not None:
        wanted["response_format"] = response_format
    actual = dict(request)
    for key, default in (("stream", False), ("n", 1)):
        if key in actual:
            if type(actual[key]) is not type(default) or actual.pop(key) != default:
                raise ValueError("diagnostic requires one nonstreaming physical completion")
    if actual != wanted:
        raise ValueError("diagnostic physical request differs from declared controls/schema")


def native_request(lm, row, *, profile=None):
    """Use the public canonical SDK plus declared verbatim reasoning extension.

    This pinned SDK's OpenRouter legacy importer rejects effort='none'. Do not
    translate it to enabled=False or edit the SDK: Config.extensions is its
    explicit wire-preserving door. The full native response schema is unchanged.
    """
    from dspy.lm15 import request_from_openai_chat

    from gluomem_bench.bench.rack_provider import generation_options

    profile = profile if profile is not None else lm.rack_client.profile
    body = {
        "model": lm.model,
        "messages": row["messages"],
        "max_tokens": row["output_budget"],
        **sampling_options(profile),
    }
    if "response_format" in row["generation_kwargs"]:
        body["response_format"] = row["generation_kwargs"]["response_format"]
    request = request_from_openai_chat(body)
    config = replace(
        request.config,
        extensions={"reasoning": generation_options(profile)["reasoning"]},
    )
    return replace(request, config=config)


def current_diagnostic_attempt():
    """Expose the current attempt read-only by convention for outer deadline clipping."""
    return _ACTIVE.get()


@contextmanager
def capture_native_attempt(profile, row, seconds):
    """One fresh owned native connection; compose outer physical accounting hooks.

    No pool context, fallback or retry is installed. Preserve exact raw UTF-8
    request/response bytes privately, without authentication headers.
    """
    from dspy._vendor.lm15.transports import StdlibTransport, TransportResponse

    from gluomem_bench.bench.rack_pool import wire_record

    started = time.monotonic()
    attempt = {
        "protocol": "native-diagnostic-singleton-attempt@v1",
        "ordinal": row["ordinal"],
        "started_monotonic": started,
        "deadline_monotonic": started + seconds,
        "timeout_seconds": seconds,
    }
    if profile.profile_version == "observed-provider-v13":
        from gluomem_bench.bench.rack_observed_provider import singleton_intent
        attempt["intent"] = singleton_intent(profile, row)
        if "root_ordinal" in row:
            attempt["root_ordinal"] = row["root_ordinal"]
    row["physical_attempt"] = attempt
    token = _ACTIVE.set(attempt)
    with _CAPTURE_LOCK:
        original = StdlibTransport.stream

        def stream(transport, request):
            if _ACTIVE.get() is not attempt:
                return original(transport, request)
            if attempt.get("sent"):
                raise ValueError("diagnostic native hidden resend refused")
            body = json.loads(request.body)
            validate_request(
                profile,
                body,
                messages=row["messages"],
                output=row["output_budget"],
                response_format=row.get("generation_kwargs", {}).get("response_format"),
            )
            if request.url != profile.base_url + "/chat/completions":
                raise ValueError("diagnostic native endpoint override refused")
            remaining = attempt["deadline_monotonic"] - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("diagnostic original deadline exhausted")
            owned = StdlibTransport(
                connect_timeout=remaining,
                read_timeout=remaining,
                write_timeout=remaining,
                pool_timeout=remaining,
                max_connections=1,
                trust_env=False,
            )
            clipped = replace(
                request, connect_timeout=remaining, read_timeout=remaining, write_timeout=remaining
            )
            timer = threading.Timer(remaining, owned.close)
            timer.daemon = True
            timer.start()
            attempt["sent"] = True
            chunks, status, headers = [], 0, []
            try:
                with original(owned, clipped) as response:
                    status, headers = response.status, response.headers
                    size = 0
                    for chunk in response:
                        chunks.append(chunk)
                        size += len(chunk)
                        if size > 8_000_000:
                            raise ValueError("diagnostic response byte bound")
                    raw = b"".join(chunks)
                    return TransportResponse(
                        status=status,
                        reason=response.reason,
                        headers=headers,
                        http_version=response.http_version,
                        chunks=iter([raw]),
                        release=lambda consumed: None,
                    )
            finally:
                raw = b"".join(chunks)
                try:
                    wire = {
                        **wire_record(request.url, request.body, status, raw, headers),
                        "request_raw_utf8": request.body.decode("utf-8"),
                        "request_b64": base64.b64encode(request.body).decode("ascii"),
                        "response_b64": base64.b64encode(raw).decode("ascii"),
                    }
                    try:
                        wire["response_raw_utf8"] = raw.decode("utf-8", errors="strict")
                    except UnicodeDecodeError:
                        wire["response_utf8"] = "invalid; original octets retained in response_b64"
                    attempt["wire"] = wire
                    attempt["completed_monotonic"] = time.monotonic()
                finally:
                    timer.cancel()
                    owned.close()

        StdlibTransport.stream = stream
        try:
            yield
        finally:
            StdlibTransport.stream = original
            _ACTIVE.reset(token)


def validate_local_schema(row):
    """Validate untouched output against the FULL sent schema, without a codec."""
    from gluomem.llm.structured_adapter import _object
    from jsonschema import Draft202012Validator
    from referencing import Registry

    response_format = row.get("generation_kwargs", {}).get("response_format")
    if response_format is None:
        return
    try:
        schema = response_format["json_schema"]["schema"]
        if (
            response_format["type"] != "json_schema"
            or response_format["json_schema"]["strict"] is not False
        ):
            raise ValueError("nonstrict schema required")
        value = json.loads(
            row["response"]["choices"][0]["message"]["content"], object_pairs_hook=_object
        )
        Draft202012Validator(schema, registry=Registry()).validate(value)
    except Exception:
        raise ValueError(
            "diagnostic local full-schema validation failed; no semantic retry"
        ) from None


def validate_diagnostic_history(*args, **kwargs):
    """Public lazy replay seam; imports no model SDK until explicitly invoked."""
    from gluomem_bench.bench.rack_diagnostic_replay import validate_diagnostic_history as validate

    return validate(*args, **kwargs)
