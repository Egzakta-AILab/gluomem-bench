"""Opt-in observed projected invocation; never infer a menu from prompt text/schema."""

from contextvars import ContextVar
from copy import deepcopy
from types import SimpleNamespace

INVOCATION = "projected-consolidation-invocation@v1"
SEMANTIC_INVOCATION = "semantic-projected-consolidation-invocation@v1"
SEMANTIC_WRITER = "semantic-state-writer-v4"
PRODUCER = "DspyExtractors.consolidate_claim_with_projection"
ACTIVE_INVOCATION = ContextVar("rack_projected_consolidation", default=None)


def contract_binding(profile):
    if profile.consolidation_contract is None:
        return {}
    if profile.semantic_producer_contract is not None:
        from gluomem_bench.bench.rack_observed import CONSOLIDATION_INVOCATION

        return {
            "consolidation_contract": {
                "protocol": profile.consolidation_contract,
                "producer": PRODUCER,
                "observation": CONSOLIDATION_INVOCATION,
            }
        }
    return {
        "consolidation_contract": {
            "protocol": profile.consolidation_contract,
            "producer": PRODUCER,
            "observation": (
                SEMANTIC_INVOCATION if profile.writer_contract == SEMANTIC_WRITER else INVOCATION
            ),
        }
    }


class ObservedConsolidationAdapter:
    """Invocation-local proxy: observe typed inputs, leave the real adapter unchanged."""

    def __init__(self, adapter, profile):
        self.adapter, self.profile = adapter, profile

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        if self.profile.semantic_producer_contract is not None:
            from gluomem_bench.bench.rack_observed import ObservedAdapter

            return ObservedAdapter(self.adapter, self.profile, "consolidate")(
                lm, lm_kwargs, signature, demos, inputs
            )
        from gluomem.llm.outputs import ConsolidationProjection

        projection = inputs.get("projection")
        if projection is None and self.profile.consolidation_contract is None:
            return self.adapter(lm, lm_kwargs, signature, demos, inputs)
        semantic = self.profile.writer_contract == SEMANTIC_WRITER
        if semantic:
            from gluomem.llm.semantic import SemanticClaimContext, SemanticProjection

            expected_projection = SemanticProjection
            context = inputs.get("new_claim_context")
            if type(context) is not SemanticClaimContext or context.meaning != getattr(
                projection, "meaning", None
            ):
                raise ValueError("semantic projection requires matching bound claim meanings")
        else:
            expected_projection = ConsolidationProjection
        if (
            type(projection) is not expected_projection
            or projection.protocol != self.profile.consolidation_contract
        ):
            raise ValueError("projected consolidation requires an explicit matching profile/method")
        projection.validate_candidates(inputs["candidates"])
        invocation = {
            "protocol": SEMANTIC_INVOCATION if semantic else INVOCATION,
            "producer": PRODUCER,
            **{
                name: (
                    [value.model_dump(mode="json") for value in value]
                    if name == "candidates"
                    else value.model_dump(mode="json")
                    if hasattr(value, "model_dump")
                    else value
                )
                for name, value in inputs.items()
            },
        }
        if self.profile.writer_contract is not None:
            if signature.fields["decision"].annotation.model_fields["rationale"].is_required():
                raise ValueError("writer contract requires its declared optional diagnostic parser")
            invocation["writer_contract"] = self.profile.writer_contract
        state = {"invocation": invocation, "sends": 0}
        token = ACTIVE_INVOCATION.set(state)
        try:
            return self.adapter(lm, lm_kwargs, signature, demos, inputs)
        finally:
            ACTIVE_INVOCATION.reset(token)


def observed_invocation():
    state = ACTIVE_INVOCATION.get()
    if state is None:
        return None
    if state["sends"]:
        raise ValueError("projected consolidation permits only one observed native send")
    state["sends"] += 1
    return deepcopy(state["invocation"])


def _validate_semantic_receipts(candidates, *, producer_contract=None):
    """Retained receipt context must reproduce its literal, role-bound witness."""
    import json

    from gluomem.llm.semantic import SemanticClaimSpec, source_witness
    from gluomem.write.events import SourceEvent

    spec_type = SemanticClaimSpec
    if producer_contract == "observed-assertion-scope-v4":
        from gluomem.llm.assertion_scope import AssertionScopedClaimSpec

        spec_type = AssertionScopedClaimSpec

    for candidate in candidates:
        if candidate.semantic is None:
            continue
        for receipt in candidate.semantic.receipts:
            event = SourceEvent.model_validate(json.loads(receipt.source_context))
            spec = spec_type(
                text=candidate.content,
                meaning=candidate.semantic.meaning,
                message_id=receipt.witness.message_id,
                verbatim_span=receipt.witness.verbatim_span,
            )
            if source_witness(event, spec) != receipt.witness:
                raise ValueError("semantic candidate receipt differs from source witness")


def reconstructed_signature(row, cfg, *, authority=None):
    """Rebuild from frozen task/artifact policy and typed observed inputs, not saved schema."""
    if cfg.get("write", {}).get("semantic_producer_contract") is not None:
        from gluomem_bench.bench.rack_observed import reconstructed_signature as reconstruct

        return reconstruct(row, cfg, authority=authority)
    import dspy
    from gluomem.config.app_config import DspyConfig
    from gluomem.llm.bootstrap import load_or_build
    from gluomem.llm.outputs import (
        ClaimConsolidationContext,
        ConsolidationProjection,
        StoredCandidate,
    )
    from gluomem.llm.signatures import projected_consolidation_module
    from gluomem.llm.structured_adapter import make_adapter

    observed = row.get("consolidation_invocation")
    fields = {"protocol", "producer", "new_claim", "candidates", "new_claim_context", "projection"}
    writer_contract = cfg.get("write", {}).get("writer_contract")
    semantic = writer_contract == SEMANTIC_WRITER
    if writer_contract is not None:
        fields.add("writer_contract")
    if (
        not isinstance(observed, dict)
        or observed.keys() != fields
        or observed["protocol"] != (SEMANTIC_INVOCATION if semantic else INVOCATION)
        or observed["producer"] != PRODUCER
        or observed.get("writer_contract") != writer_contract
        or type(observed["new_claim"]) is not str
    ):
        raise ValueError("bound projected consolidation invocation required")
    if semantic:
        from gluomem.llm.semantic import (
            SemanticClaimContext,
            SemanticProjection,
            SemanticStoredCandidate,
        )

        projection = SemanticProjection.model_validate(observed["projection"])
        context = SemanticClaimContext.model_validate(observed["new_claim_context"])
        candidates = [SemanticStoredCandidate.model_validate(c) for c in observed["candidates"]]
        _validate_semantic_receipts(candidates)
        if context.meaning != projection.meaning:
            raise ValueError("semantic projection differs from bound claim meaning")
    else:
        projection = ConsolidationProjection.model_validate(observed["projection"])
        context = ClaimConsolidationContext.model_validate(observed["new_claim_context"])
        candidates = [StoredCandidate.model_validate(c) for c in observed["candidates"]]
    projection.validate_candidates(candidates)
    policy = DspyConfig.model_validate(cfg["dspy"])
    task = policy.tasks["consolidate"]
    samples = row.get("generation_kwargs", {}).get("n", 1)
    if (
        policy.retry.max_attempts != 1
        or policy.validation_retries
        or (task.refine or {}).get("enabled")
        or (not projection.targets and (not semantic or context.meaning.independently_versioned))
        or type(samples) is not int
        or samples != 1
    ):
        raise ValueError("projected consolidation requires one-sample/no-retries policy and a menu")
    base = load_or_build("consolidate", SimpleNamespace(dspy=policy))
    if (
        not isinstance(base, (dspy.Predict, dspy.ChainOfThought))
        or len(base.predictors()) != 1
        or isinstance(base, dspy.Refine)
        or any(base.named_sub_modules(type_=dspy.Refine, skip_compiled=False))
    ):
        raise ValueError("unadmitted projected consolidation program")
    predictor = projected_consolidation_module(
        base, projection, writer_contract=writer_contract
    ).predictors()[0]
    rebuilt = predictor.signature
    inputs = {
        "new_claim": observed["new_claim"],
        "candidates": candidates,
        "new_claim_context": context,
        "projection": projection,
    }
    adapter = make_adapter(
        policy.adapter,
        canonicalize=policy.canonicalize,
        schema_mode=policy.schema_mode,
        schema_strict=policy.schema_strict,
        validation_retries=0,
    )
    if adapter.format(rebuilt, predictor.demos, inputs) != row.get("messages"):
        raise ValueError("projected invocation differs from independently rendered native messages")
    return rebuilt


def validate_projected_production_row(row, profile, cfg, protocol, expected):
    import json

    from gluomem.llm.signatures import EventClaimExtraction
    from gluomem.llm.structured_adapter import configured_signature, make_adapter, output_schema

    from gluomem_bench.bench.rack_schema import transport_binding, validate_native_projection

    binding = transport_binding(
        profile, cfg["dspy"], capture_policy=cfg.get("write", {}).get("source_capture_policy")
    )
    if (
        expected is None
        or expected.get("rack_schema_transport") != binding
        or protocol.get("rack_schema_transport") != binding
        or row.get("model") != binding["lm_model"]
    ):
        raise ValueError("unbound declared projected production transport")
    if row.get("task") == "consolidate":
        signature = reconstructed_signature(row, cfg)
    elif row.get("task") == "event_claims" and "consolidation_invocation" not in row:
        signature = configured_signature(
            EventClaimExtraction, cfg["dspy"].get("output_field_limits", {}).get("event_claims", {})
        )
    else:
        raise ValueError("unadmitted projected production task/invocation")
    kwargs = row.get("generation_kwargs", {})
    if binding["schema_mode"] == "typed-text":
        if "response_format" in kwargs:
            raise ValueError("typed-text forbids native response_format")
    else:
        wanted = {
            "type": "json_schema",
            "json_schema": {
                "name": "GluoMemOutput",
                "strict": binding["schema_strict"],
                "schema": output_schema(signature),
            },
        }
        if json.dumps(kwargs.get("response_format"), sort_keys=True) != json.dumps(
            wanted, sort_keys=True
        ):
            raise ValueError(
                "production schema differs from independently reconstructed projected contract"
            )
    validate_native_projection(row.get("response"), row.get("native_response", {}))
    adapter = make_adapter(
        cfg["dspy"]["adapter"],
        canonicalize=cfg["dspy"].get("canonicalize", False),
        schema_mode=binding["schema_mode"],
        schema_strict=binding["schema_strict"],
        validation_retries=0,
    )
    try:
        adapter.parse(signature, row["response"]["choices"][0]["message"]["content"])
    except Exception as exc:
        raise ValueError(
            "output differs from independently reconstructed projected contract"
        ) from exc
    return binding["lm_model"]
