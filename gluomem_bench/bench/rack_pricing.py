"""Explicit bounded catalogue prices. Quotes are not billed usage or budget grants."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VERSION = "bounded-provider-prices-v1"
CACHE_WRITE_VERSION = "bounded-provider-prices-v2"
_DECIMAL = r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$"


class ProviderPricingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["bounded-provider-prices-v1", "bounded-provider-prices-v2"]
    max_prompt_usd_per_token: str = Field(strict=True, pattern=_DECIMAL)
    max_completion_usd_per_token: str = Field(strict=True, pattern=_DECIMAL)
    max_request_usd: Literal["0"]
    prompt_token_cap: int = Field(strict=True, ge=1, le=65536)
    output_token_cap: int = Field(strict=True, ge=1, le=16384)


def decimal_price(value):
    if type(value) is not str or not re.fullmatch(
        r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", value
    ):
        raise ValueError("explicit decimal price required")
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise ValueError("malformed decimal price") from None
    if not number.is_finite() or number < 0:
        raise ValueError("finite nonnegative price required")
    return number


def validate_prices(policy, pricing):
    """Validate all tier shapes; bound every tier reachable at physical dispatch.

    Automatic cache reads are prompt SKUs. Explicit v2 also includes automatic
    cache writes: absence of cache directives does not make them unreachable.
    Legacy v1 retains its original classification. Web search remains inapplicable
    to these no-tool transports. Never subtract discounts or change service tier.
    """
    allowed = {
        "prompt",
        "completion",
        "request",
        "input_cache_read",
        "input_cache_write",
        "web_search",
        "discount",
    }
    maxima = {"prompt": Decimal(0), "completion": Decimal(0), "request": Decimal(0)}

    def tier(value, *, applicable, override=False):
        if not isinstance(value, dict) or set(value) - allowed - (
            {"min_prompt_tokens"} if override else {"overrides"}
        ):
            raise ValueError("unclassified provider pricing SKU/tier")
        for key in ("prompt", "completion"):
            if key not in value:
                raise ValueError("missing provider tier price")
        for key in allowed & value.keys():
            raw = value[key]
            if key == "discount":
                if type(raw) not in (int, float, str):
                    raise ValueError("invalid price discount")
                number = decimal_price(str(raw))
                if number > 1:
                    raise ValueError("invalid price discount")
                continue
            number = decimal_price(raw)
            if key == "request" and number != 0:
                raise ValueError("nonzero provider request fee refused")
            prompt_sku = key == "input_cache_read" or (
                key == "input_cache_write" and policy.version == CACHE_WRITE_VERSION
            )
            role = "prompt" if prompt_sku else key
            if applicable and role in maxima:
                cap = getattr(
                    policy, "max_" + role + ("_usd" if role == "request" else "_usd_per_token")
                )
                if number > Decimal(cap):
                    raise ValueError("public provider price exceeds approved ceiling")
                maxima[role] = max(maxima[role], number)

    tier(pricing, applicable=True)
    overrides = pricing.get("overrides", [])
    if not isinstance(overrides, list):
        raise ValueError("explicit provider price tiers required")
    seen = set()
    for value in overrides:
        if not isinstance(value, dict):
            raise ValueError("malformed provider price tier")
        threshold = value.get("min_prompt_tokens")
        if type(threshold) is not int or threshold < 1 or threshold in seen:
            raise ValueError("unclassified/duplicate provider price tier threshold")
        seen.add(threshold)
        tier(value, applicable=threshold <= policy.prompt_token_cap, override=True)
    return {key: format(value, "f") for key, value in maxima.items()}


def price_observations(profile, metadata, selected):
    """Public catalogue observations; retain source quote bytes upstream, never bill them."""
    from copy import deepcopy

    from gluomem_bench.bench.rack_pool import digest

    policy = profile.pricing_policy
    observations = []
    for endpoint in selected:
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
                or endpoint["max_prompt_tokens"] < policy.prompt_token_cap
            )
        ):
            raise ValueError("bounded pricing endpoint capacity/status/tier differs")
        prices = endpoint["pricing"]
        maxima = validate_prices(policy, prices)
        observations.append(
            {
                "protocol": "provider-price-observation@v1",
                "metadata_sha256": digest(metadata),
                "endpoint": {
                    k: endpoint[k] for k in ("tag", "provider_name", "quantization", "model_id")
                },
                "raw_pricing": deepcopy(prices),
                "effective_max_prices": maxima,
                "request_price_source": "explicit-zero-sku-v1"
                if "request" in prices
                else "absent-sku-documented-zero-v1",
                "quote_status": "public-metadata-not-billed",
            }
        )
    return observations


def wire_price_limits(policy):
    """OpenRouter routing caps are JSON numbers in USD/M tokens, USD/request."""
    import math

    def numeric(value, exponent):
        sign, digits, power = Decimal(value).as_tuple()
        scaled = Decimal((sign, digits, power + exponent))
        if scaled == scaled.to_integral_value():
            return int(scaled)
        number = float(scaled)
        if not math.isfinite(number):
            raise ValueError("pricing cap cannot be represented as a finite wire number")
        # Serialization must not round an approved cap upward. This conversion is
        # wire-only; all metadata acceptance and ceiling comparisons use Decimal.
        if Decimal(str(number)) > scaled:
            number = math.nextafter(number, 0.0)
        return number

    return {
        "prompt": numeric(policy.max_prompt_usd_per_token, 6),
        "completion": numeric(policy.max_completion_usd_per_token, 6),
        "request": numeric(policy.max_request_usd, 0),
    }
