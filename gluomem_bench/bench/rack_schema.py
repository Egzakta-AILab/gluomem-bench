"""Explicit operator capability declaration, not proof of server qualification."""

from __future__ import annotations

import hashlib
import json

TRANSPORT = "declared-openai-schema-v1"


def _native_projection(canonical):
    message, usage = canonical["message"], canonical["usage"]
    parts = message["parts"]
    if (message["role"] != "assistant" or len(parts) != 1 or parts[0].get("type") != "text"
            or type(parts[0].get("text")) is not str or canonical["adaptations"]):
        raise ValueError("unadapted single-text native response required")
    return {
        "model": canonical["model"],
        "choices": [{"message": {"role": message["role"], "content": parts[0]["text"]},
                     "finish_reason": canonical["finish_reason"]}],
        "usage": {"prompt_tokens": usage["input_tokens"],
                  "completion_tokens": usage["output_tokens"],
                  "total_tokens": usage["total_tokens"]},
    }


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


def schema_binding(profile, dspy_config):
    """Deterministic identity for independently frozen config/protocol expectations."""
    if hasattr(dspy_config, "model_dump"):
        dspy_config = dspy_config.model_dump(mode="json")
    if dspy_config.get("schema_mode") != "required":
        raise ValueError("declared schema transport requires required schema mode")
    material = {
        "transport": TRANSPORT,
        "base_url": profile.base_url,
        "model": profile.model,
        "adapter": dspy_config["adapter"],
        "schema_mode": "required",
        "schema_strict": dspy_config["schema_strict"],
    }
    limits = dspy_config.get("output_field_limits", {})
    if limits:
        from gluomem.config.app_config import DspyConfig

        limits = DspyConfig.model_validate(dspy_config).output_field_limits
        material["output_field_limits"] = limits
    digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    return {**material, "lm_model": "rack-schema-" + digest + "/" + profile.model}


def production_expectation(system, profile, cfg):
    return ({"rack_schema_transport": schema_binding(profile, cfg.dspy)}
            if system == "gluomem" and cfg.dspy.schema_mode == "required" else {})


def make_router(cfg, *, client, trace, budget):
    from gluomem_bench.bench.rack_trace import RackRouter

    options = {"schema_transport": TRANSPORT} if cfg.dspy.schema_mode == "required" else {}
    return RackRouter(cfg, client=client, trace=trace, budget=budget, **options)


def register_schema(profile, cfg):
    from dspy.lm15 import (
        AccessPolicy,
        ModelSupport,
        OpenAIChatCompat,
        ProviderDefinition,
        register_provider,
    )

    binding = schema_binding(profile, cfg.dspy)
    register_provider(
        ProviderDefinition.chat(
            AccessPolicy(
                provider=binding["lm_model"].split("/", 1)[0],
                env_keys=("GLUOMEM_RACK_API_KEY",),
                base_url=profile.base_url,
            ),
            compat=OpenAIChatCompat(
                thinking_format="qwen_chat_template",
                instruction_role="system",
                max_tokens_field="max_tokens",
            ),
        ),
        models={profile.model: ModelSupport(response_schema=True, function_calling=False)},
    )
    return binding


def validate_schema_row(row, profile, cfg, protocol, expected):
    """Accept the custom identity only against independently supplied expectations."""
    from gluomem.llm.signatures import ClaimConsolidation, EventClaimExtraction
    from gluomem.llm.structured_adapter import configured_signature, output_schema

    binding = schema_binding(profile, cfg["dspy"])
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
