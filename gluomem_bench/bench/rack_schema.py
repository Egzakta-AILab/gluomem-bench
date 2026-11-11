"""Explicit operator capability declaration, not proof of server qualification."""

from __future__ import annotations

import hashlib
import json

TRANSPORT = "declared-openai-schema-v1"
TEXT_TRANSPORT = "declared-openai-typed-text-json-v1"


def _native_projection(canonical):
    message, usage = canonical["message"], canonical["usage"]
    parts = message["parts"]
    if (message["role"] != "assistant" or len(parts) != 1 or parts[0].get("type") != "text"
            or type(parts[0].get("text")) is not str or canonical["adaptations"]):
        raise ValueError("unadapted single-text native response required")
    view = {
        "model": canonical["model"],
        "choices": [{"message": {"role": message["role"], "content": parts[0]["text"]},
                     "finish_reason": canonical["finish_reason"]}],
        "usage": {"prompt_tokens": usage["input_tokens"],
                  "completion_tokens": usage["output_tokens"],
                  "total_tokens": usage["total_tokens"]},
    }
    raw = canonical.get("provider_data")
    if isinstance(raw, dict) and "provider" in raw:
        view["provider"] = raw["provider"]
    return view


def project_native_response(response):
    """Project actual SDK fields; this is NOT a captured OpenAI wire response."""
    from dataclasses import asdict

    from dspy.lm15 import Response, TextPart

    if (type(response) is not Response
            or any(type(part) is not TextPart for part in response.message.parts)):
        raise ValueError("plain native SDK response required")
    canonical = asdict(response)
    evidence = {"protocol": "lm15-native-response-projection@v1", "canonical": canonical}
    return _native_projection(canonical), evidence


def validate_native_projection(view, evidence):
    if (evidence.get("protocol") != "lm15-native-response-projection@v1"
            or view != _native_projection(evidence["canonical"])):
        raise ValueError("native response projection differs from SDK evidence")


def schema_binding(profile, dspy_config, *, capture_policy=None):
    """Deterministic identity for independently frozen config/protocol expectations."""
    if hasattr(dspy_config, "model_dump"):
        dspy_config = dspy_config.model_dump(mode="json")
    if dspy_config.get("schema_mode") != "required":
        raise ValueError("declared schema transport requires required schema mode")
    if profile.production_json_transport is not None:
        raise ValueError("typed-text profile cannot declare native schema transport")
    diagnostic = profile.native_diagnostic
    if (
        profile.native_schema_strict is not None
        and dspy_config.get("schema_strict") is not profile.native_schema_strict
    ):
        raise ValueError("explicit native schema strictness profile/config binding differs")
    if diagnostic is not None and dspy_config.get("schema_strict") is not False:
        raise ValueError("v11 diagnostic requires explicitly non-strict native schema")
    material = {
        "transport": TRANSPORT,
        "base_url": profile.base_url,
        "model": profile.model,
        "adapter": dspy_config["adapter"],
        "schema_mode": "required",
        "schema_strict": dspy_config["schema_strict"],
    }
    if diagnostic is not None or profile.native_schema_strict is False:
        from gluomem_bench.bench.rack_diagnostic import TRANSPORT as DIAGNOSTIC_TRANSPORT
        material.update(transport=DIAGNOSTIC_TRANSPORT, native_schema_enforcement="not-guaranteed",
            local_validation="full-native-schema-and-typed-source-target-guards-v1")
    if profile.provider is not None:
        from gluomem_bench.bench.rack_provider import generation_options
        material["provider_contract"] = {
            "provider": profile.provider,
            "requested_model": profile.requested_model,
            "api_key_env": profile.api_key_env,
            "token_count_policy": profile.token_count_policy,
            "context_window": profile.context_window,
            "generation_options": generation_options(profile),
        }
        from gluomem_bench.bench.rack_provider import ENDPOINT_VERSION
        if profile.profile_version == ENDPOINT_VERSION:
            material["provider_contract"].update(profile_version=ENDPOINT_VERSION,
                openrouter_endpoint=profile.openrouter_endpoint.model_dump(mode="json"))
        from gluomem_bench.bench.rack_provider import POOL_VERSION
        if profile.profile_version == POOL_VERSION:
            material["provider_contract"].update(profile_version=POOL_VERSION,
                openrouter_pool=profile.openrouter_pool.model_dump(mode="json"))
        if profile.profile_version == "observed-provider-v13":
            material["provider_contract"].update(
                profile_version=profile.profile_version,
                openrouter_pool=(profile.openrouter_pool.model_dump(mode="json")
                                 if profile.openrouter_pool else None),
                openrouter_endpoint=(profile.openrouter_endpoint.model_dump(mode="json")
                                     if profile.openrouter_endpoint else None),
                native_response_policy="exact-raw-integer-usage-output-equality-v1",
                physical_attempt_policy="one-original-attempt-no-fallback-v1",
            )
        if diagnostic is not None:
            material["provider_contract"].update(profile_version=profile.profile_version,
                openrouter_endpoint=profile.openrouter_endpoint.model_dump(mode="json"),
                native_diagnostic=diagnostic.model_dump(mode="json"))
        if profile.native_schema_strict is not None:
            material["provider_contract"]["native_schema_strict"] = profile.native_schema_strict
        if profile.pricing_policy is not None:
            material["provider_contract"]["pricing_policy"] = profile.pricing_policy.model_dump(
                mode="json")
    limits = dspy_config.get("output_field_limits", {})
    if limits:
        from gluomem.config.app_config import DspyConfig

        limits = DspyConfig.model_validate(dspy_config).output_field_limits
        material["output_field_limits"] = limits
    from gluomem_bench.bench.rack_projection import contract_binding
    material.update(contract_binding(profile))
    from gluomem_bench.bench.rack_extraction import contract_binding as extraction_binding
    material.update(extraction_binding(profile, capture_policy))
    digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    return {**material, "lm_model": "rack-schema-" + digest + "/" + profile.model}


def typed_text_binding(profile, dspy_config, *, capture_policy=None):
    """A distinct, explicit local-validation contract; never native enforcement."""
    if hasattr(dspy_config, "model_dump"):
        dspy_config = dspy_config.model_dump(mode="json")
    if (dspy_config.get("schema_mode") != "typed-text"
        or profile.production_json_transport != "typed-text-json-v1"
        or dspy_config.get("schema_strict")):
        raise ValueError("explicit typed-text profile and matching local parser policy required")
    from gluomem_bench.bench.rack_provider import generation_options
    material = {
        "transport": TEXT_TRANSPORT,
        "base_url": profile.base_url,
        "model": profile.model,
        "adapter": dspy_config["adapter"],
        "schema_mode": "typed-text",
        "schema_strict": False,
        "native_response_format": "none",
        "local_validation": "schema-derived-required-literals-and-values-v1",
        "provider_contract": {
            "provider": profile.provider,
            "requested_model": profile.requested_model,
            "api_key_env": profile.api_key_env,
            "token_count_policy": profile.token_count_policy,
            "context_window": profile.context_window,
            "generation_options": generation_options(profile),
        },
    }
    limits = dspy_config.get("output_field_limits", {})
    if limits:
        from gluomem.config.app_config import DspyConfig
        material["output_field_limits"] = DspyConfig.model_validate(dspy_config).output_field_limits
    from gluomem_bench.bench.rack_projection import contract_binding
    material.update(contract_binding(profile))
    from gluomem_bench.bench.rack_extraction import contract_binding as extraction_binding
    material.update(extraction_binding(profile, capture_policy))
    digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    return {**material, "lm_model": "rack-typed-text-" + digest + "/" + profile.model}


def transport_binding(profile, dspy_config, *, capture_policy=None):
    mode = (
        dspy_config.schema_mode
        if hasattr(dspy_config, "schema_mode")
        else dspy_config.get("schema_mode")
    )
    bind = typed_text_binding if mode == "typed-text" else schema_binding
    return bind(profile, dspy_config, capture_policy=capture_policy)


def production_expectation(system, profile, cfg):
    return ({"rack_schema_transport": transport_binding(
                profile, cfg.dspy, capture_policy=cfg.write.source_capture_policy)}
            if system == "gluomem" and cfg.dspy.schema_mode in {"required", "typed-text"} else {})


def make_router(cfg, *, client, trace, budget):
    from gluomem_bench.bench.rack_trace import RackRouter

    options = ({"schema_transport": TEXT_TRANSPORT} if cfg.dspy.schema_mode == "typed-text"
        else {"schema_transport": TRANSPORT} if cfg.dspy.schema_mode == "required" else {})
    return RackRouter(cfg, client=client, trace=trace, budget=budget, **options)


def validate_observed_history(*args, **kwargs):
    """Separate offline-only replay; old study/operator profiles are not widened."""
    from gluomem_bench.bench.rack_observed_replay import validate_observed_history as validate
    return validate(*args, **kwargs)


def validate_observed_provider_history(*args, **kwargs):
    """Separately versioned DeepSeek provider/private replay; no legacy admission."""
    from gluomem_bench.bench.rack_observed_provider_replay import (
        validate_observed_provider_history as validate,
    )
    return validate(*args, **kwargs)


def register_schema(profile, cfg):
    validate_writer_config(profile, cfg)
    return _register_native(
        profile, schema_binding(profile, cfg.dspy, capture_policy=cfg.write.source_capture_policy),
        response_schema=True,
    )


def register_typed_text(profile, cfg):
    validate_writer_config(profile, cfg)
    return _register_native(
        profile, typed_text_binding(
            profile, cfg.dspy, capture_policy=cfg.write.source_capture_policy
        ),
        response_schema=False,
    )


def _register_native(profile, binding, *, response_schema):
    from dspy.lm15 import (
        AccessPolicy,
        ModelSupport,
        OpenAIChatCompat,
        ProviderDefinition,
        register_provider,
    )

    from gluomem_bench.bench.rack_provider import generation_options, is_openrouter
    deepseek = is_openrouter(profile)
    body = generation_options(profile)
    register_provider(
        ProviderDefinition.chat(
            AccessPolicy(
                provider=binding["lm_model"].split("/", 1)[0],
                env_keys=(profile.api_key_env or "GLUOMEM_RACK_API_KEY",),
                base_url=profile.base_url,
            ),
            compat=OpenAIChatCompat(
                thinking_format="openrouter" if deepseek else "qwen_chat_template",
                instruction_role="system",
                max_tokens_field="max_tokens",
                routing=body["provider"] if deepseek else None,
                extensions={"reasoning": body["reasoning"]} if deepseek else None,
            ),
        ),
        models={
            profile.model: ModelSupport(response_schema=response_schema, function_calling=False)
        },
    )
    return binding


def validate_writer_config(profile, cfg):
    """The profile opt-in and actual writer/task policy must agree exactly."""
    if hasattr(cfg, "model_dump"):
        cfg = cfg.model_dump(mode="json")
    if cfg.get("write", {}).get("writer_contract") != profile.writer_contract:
        raise ValueError("writer contract profile/config binding differs")
    if cfg.get("write", {}).get("semantic_producer_contract") != profile.semantic_producer_contract:
        raise ValueError("semantic producer profile/config binding differs")
    if profile.semantic_producer_contract is not None and (
            cfg.get("write", {}).get("extraction_cache", {}).get("mode") != "off"
            or cfg.get("dspy", {}).get("cache", {}).get("disk") is not False):
        raise ValueError("observed bridge cache collectors are unqualified; caches must be off")
    if profile.semantic_producer_contract is not None:
        for task in ("event_claims", "identity_alignment", "consolidate", "answer"):
            if (cfg.get("dspy", {}).get("tasks", {}).get(task, {}).get("max_tokens")
                    != profile.output_budget(task)):
                raise ValueError("observed producer role output allowance differs from profile")


def validate_schema_row(row, profile, cfg, protocol, expected, *, authority=None):
    """Accept the custom identity only against independently supplied expectations."""
    validate_writer_config(profile, cfg)
    if profile.semantic_producer_contract:
        from gluomem_bench.bench.rack_observed import validate_observed_response
        validate_observed_response(row, profile)
    if (profile.semantic_producer_contract
            and row.get("task") in {"identity_alignment", "consolidate"}):
        from gluomem_bench.bench.rack_observed import validate_production_row as validate_observed
        return validate_observed(row, profile, cfg, protocol, expected, authority=authority)
    if "identity_invocation" in row or row.get("task") == "identity_alignment":
        raise ValueError("legacy production contract cannot admit identity invocation")
    if profile.native_diagnostic is not None:
        from gluomem_bench.bench.rack_diagnostic_replay import validate_physical_row
        validate_physical_row(row, profile)
    from gluomem_bench.bench.rack_provider import POOL_VERSION
    if profile.profile_version == POOL_VERSION:
        from gluomem_bench.bench.rack_pool_replay import validate_pool_call
        validate_pool_call(row, profile)
    if row.get("task") == "event_claims" and profile.extraction_contract is not None:
        schema_binding(profile, cfg["dspy"], capture_policy=cfg["write"]["source_capture_policy"])
        from gluomem_bench.bench.rack_extraction import validate_strict_production_row
        return validate_strict_production_row(row, profile, cfg, protocol, expected)
    if "extraction_invocation" in row:
        raise ValueError("legacy production contract cannot admit strict extraction invocation")
    if profile.consolidation_contract is not None:
        schema_binding(
            profile, cfg["dspy"], capture_policy=cfg.get("write", {}).get("source_capture_policy")
        )
        from gluomem_bench.bench.rack_projection import validate_projected_production_row
        return validate_projected_production_row(row, profile, cfg, protocol, expected)
    if "consolidation_invocation" in row:
        raise ValueError("legacy production contract cannot admit projected invocation")
    from gluomem.llm.signatures import ClaimConsolidation, EventClaimExtraction
    from gluomem.llm.structured_adapter import configured_signature, output_schema

    binding = schema_binding(
        profile, cfg["dspy"], capture_policy=cfg.get("write", {}).get("source_capture_policy")
    )
    if (
        expected is None
        or expected.get("rack_schema_transport") != binding
        or protocol.get("rack_schema_transport") != binding
        or row.get("model") != binding["lm_model"]
    ):
        raise ValueError("unbound declared production schema transport")
    signatures = {"event_claims": EventClaimExtraction, "consolidate": ClaimConsolidation}
    if row.get("task") not in signatures:
        raise ValueError("unadmitted schema production task")
    wanted = {
        "type": "json_schema",
        "json_schema": {
            "name": "GluoMemOutput",
            "strict": binding["schema_strict"],
            "schema": output_schema(configured_signature(
                signatures[row["task"]],
                cfg["dspy"].get("output_field_limits", {}).get(row["task"], {}),
            )),
        },
    }
    actual = row.get("generation_kwargs", {}).get("response_format")
    if json.dumps(actual, sort_keys=True) != json.dumps(wanted, sort_keys=True):
        raise ValueError("production schema differs from native signature/strictness")
    return binding["lm_model"]


def validate_production_row(row, profile, cfg, protocol, expected, *, authority=None):
    """Dispatch only between independently bound, separately versioned contracts."""
    validate_writer_config(profile, cfg)
    if profile.semantic_producer_contract:
        from gluomem_bench.bench.rack_observed import validate_observed_response
        validate_observed_response(row, profile)
    if (profile.semantic_producer_contract
            and row.get("task") in {"identity_alignment", "consolidate"}):
        from gluomem_bench.bench.rack_observed import validate_production_row as validate_observed
        return validate_observed(row, profile, cfg, protocol, expected, authority=authority)
    if "identity_invocation" in row or row.get("task") == "identity_alignment":
        raise ValueError("legacy production contract cannot admit identity invocation")
    if profile.native_diagnostic is not None:
        from gluomem_bench.bench.rack_diagnostic_replay import validate_physical_row
        validate_physical_row(row, profile)
    from gluomem_bench.bench.rack_provider import POOL_VERSION
    if profile.profile_version == POOL_VERSION:
        from gluomem_bench.bench.rack_pool_replay import validate_pool_call
        validate_pool_call(row, profile)
    if row.get("task") == "event_claims" and profile.extraction_contract is not None:
        from gluomem_bench.bench.rack_extraction import validate_strict_production_row
        return validate_strict_production_row(row, profile, cfg, protocol, expected)
    if "extraction_invocation" in row:
        raise ValueError("legacy production contract cannot admit strict extraction invocation")
    if profile.consolidation_contract is not None:
        from gluomem_bench.bench.rack_projection import validate_projected_production_row
        return validate_projected_production_row(row, profile, cfg, protocol, expected)
    if "consolidation_invocation" in row:
        raise ValueError("legacy production contract cannot admit projected invocation")
    if cfg.get("dspy", {}).get("schema_mode") != "typed-text":
        return validate_schema_row(row, profile, cfg, protocol, expected)
    from gluomem.llm.signatures import ClaimConsolidation, EventClaimExtraction
    from gluomem.llm.structured_adapter import configured_signature, make_adapter
    binding = typed_text_binding(
        profile, cfg["dspy"], capture_policy=cfg.get("write", {}).get("source_capture_policy")
    )
    if (
        expected is None
        or expected.get("rack_schema_transport") != binding
        or protocol.get("rack_schema_transport") != binding
        or row.get("model") != binding["lm_model"]
    ):
        raise ValueError("unbound declared typed-text production transport")
    if "response_format" in row.get("generation_kwargs", {}):
        raise ValueError("typed-text forbids native response_format")
    signatures = {"event_claims": EventClaimExtraction, "consolidate": ClaimConsolidation}
    if row.get("task") not in signatures:
        raise ValueError("unadmitted typed-text production task")
    validate_native_projection(row.get("response"), row.get("native_response", {}))
    signature = configured_signature(signatures[row["task"]],
        cfg["dspy"].get("output_field_limits", {}).get(row["task"], {}))
    adapter = make_adapter(
        cfg["dspy"]["adapter"],
        canonicalize=cfg["dspy"].get("canonicalize", False),
        schema_mode="typed-text",
        schema_strict=False,
        validation_retries=0,
    )
    try:
        adapter.parse(signature, row["response"]["choices"][0]["message"]["content"])
    except Exception as exc:
        raise ValueError("typed-text output differs from frozen local contract") from exc
    return binding["lm_model"]
