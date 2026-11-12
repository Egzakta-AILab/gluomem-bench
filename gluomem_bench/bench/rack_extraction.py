"""Explicit strict-source contract and typed invocation replay, never saved-schema authority."""

from contextvars import ContextVar
from copy import deepcopy
from types import SimpleNamespace

CONTRACT = "strict-source-quotes-v1"
INVOCATION = "source-quote-extraction-invocation@v1"
SEMANTIC_INVOCATION = "semantic-source-extraction-invocation@v1"
OBSERVED_INVOCATION = "observed-dated-source-extraction-invocation@v1"
SEMANTIC_WRITER = "semantic-state-writer-v4"
PRODUCER = "DspyExtractors.extract_event_claims"
ACTIVE_INVOCATION = ContextVar("rack_source_quote_extraction", default=None)


def contract_binding(profile, capture_policy):
    if profile.extraction_contract is None:
        return {}
    if profile.extraction_contract != CONTRACT or capture_policy not in {
        "trusted-only",
        "attributed-interaction",
    }:
        raise ValueError("strict extraction contract requires an explicit source capture policy")
    result = {
        "extraction_contract": {
            "protocol": CONTRACT,
            "capture_policy": capture_policy,
            "producer": PRODUCER,
            "observation": (
                SEMANTIC_INVOCATION if profile.writer_contract == SEMANTIC_WRITER else INVOCATION
            ),
        }
    }
    if profile.writer_contract is not None:
        if profile.writer_contract not in {
            "bounded-writer-v2",
            "action-target-writer-v3",
            SEMANTIC_WRITER,
        }:
            raise ValueError("unknown writer contract")
        result["writer_contract"] = {
            "protocol": profile.writer_contract,
            "rationale": "optional-bounded-string-v1",
            "quote_alignment": "unicode-lexical-boundaries-v1",
            "source_observation": "trusted-source-envelope-v1",
        }
        if profile.writer_contract in {"action-target-writer-v3", SEMANTIC_WRITER}:
            result["writer_contract"]["target_policy"] = "action-dependent-nullable-defaults-v1"
    if profile.semantic_producer_contract is not None:
        result["semantic_producer_contract"] = {
            "protocol": profile.semantic_producer_contract,
            "profile_version": profile.profile_version,
            "identity_task": "identity_alignment",
            "identity_output_tokens": profile.identity_output_tokens,
            "capability": "semantic_producer_capability",
            "producer": "DspyExtractors.align_claim_identity",
            "observation": "observed-identity-invocation@v1",
            "effective_date": "required-nullable-world-date-v1",
        }
        result["extraction_contract"]["observation"] = OBSERVED_INVOCATION
    return result


def trusted_source_event(observed, capture_policy):
    """Use the writer-observed envelope, never parse message/role data from prose."""
    from gluomem.timeutil import parse_utc
    from gluomem.write.events import SourceEvent, _format_event_source

    event = SourceEvent.model_validate(observed["source_event"])
    if observed.get("writer_contract") == SEMANTIC_WRITER:
        import json

        if (
            type(observed.get("source_context")) is not dict
            or len(
                {
                    json.dumps(value, sort_keys=True, allow_nan=False)
                    for value in (
                        observed["source_context"],
                        observed["source_event"],
                        event.model_dump(mode="json"),
                    )
                }
            )
            != 1
        ):
            raise ValueError("semantic source_context differs from trusted source envelope")
    if event.capture_policy != capture_policy:
        raise ValueError("writer source envelope capture policy differs")
    source, _, default_mid = _format_event_source(event)
    reference = event.event_time or event.created_at
    try:
        reference = parse_utc(reference).isoformat() if reference else ""
    except ValueError:
        reference = ""
    if source != observed["source_text"] or reference != observed["reference_date"]:
        raise ValueError("writer source envelope differs from typed extraction inputs")
    return event, default_mid


class ObservedSourceQuoteAdapter:
    """Record typed inputs at the real adapter, with the mode declared before dispatch."""

    def __init__(self, adapter, profile, capture_policy):
        self.adapter, self.profile, self.capture_policy = adapter, profile, capture_policy

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        annotation = signature.fields["claims"].annotation
        from typing import get_args

        quote = get_args(annotation)[0].model_fields["verbatim_span"]
        if self.profile.extraction_contract is None:
            if quote.is_required():
                raise ValueError("strict source quote wire requires a declared extraction contract")
            return self.adapter(lm, lm_kwargs, signature, demos, inputs)
        contract_binding(self.profile, self.capture_policy)
        semantic = self.profile.writer_contract == SEMANTIC_WRITER
        producer = self.profile.semantic_producer_contract
        if producer and not get_args(annotation)[0].model_fields["event_date"].is_required():
            raise ValueError("observed producer requires an explicit nullable event_date")
        fields = {"source_text", "reference_date"} | ({"source_context"} if semantic else set())
        if (
            not quote.is_required()
            or inputs.keys() != fields
            or any(type(inputs.get(name)) is not str for name in ("source_text", "reference_date"))
            or (
                semantic
                and (
                    type(inputs.get("source_context")) is not dict
                    or not get_args(annotation)[0].model_fields["meaning"].is_required()
                )
            )
        ):
            raise ValueError(
                "strict source extraction requires native typed source inputs and quote"
            )
        invocation = {
            "protocol": SEMANTIC_INVOCATION if semantic else INVOCATION,
            "producer": PRODUCER,
            **deepcopy(inputs),
        }
        if producer:
            invocation.update(protocol=OBSERVED_INVOCATION, semantic_producer_contract=producer)
        if self.profile.writer_contract is not None:
            from gluomem.write.source_binding import ACTIVE_SOURCE_ENVELOPE

            source = ACTIVE_SOURCE_ENVELOPE.get()
            if source is None or source.get("writer_contract") != self.profile.writer_contract:
                raise ValueError("writer extraction requires a bound trusted source envelope")
            invocation.update(deepcopy(source))
            trusted_source_event(invocation, self.capture_policy)
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
        raise ValueError("strict extraction permits only one observed native send")
    state["sends"] += 1
    return deepcopy(state["invocation"])


def reconstructed_signature(row, cfg):
    """Load pinned program/demos, finalize cfg source types/limits, re-render exact inputs."""
    from gluomem.config.app_config import DspyConfig
    from gluomem.llm.bootstrap import load_or_build
    from gluomem.llm.structured_adapter import make_adapter

    observed = row.get("extraction_invocation")
    writer_contract = cfg["write"].get("writer_contract")
    semantic = writer_contract == SEMANTIC_WRITER
    producer = cfg["write"].get("semantic_producer_contract")
    fields = {"protocol", "producer", "source_text", "reference_date"}
    if semantic:
        fields.add("source_context")
    if writer_contract is not None:
        fields.update(("writer_contract", "source_event"))
    if producer:
        fields.add("semantic_producer_contract")
    if (
        not isinstance(observed, dict)
        or observed.keys() != fields
        or observed["protocol"]
        != (OBSERVED_INVOCATION if producer else SEMANTIC_INVOCATION if semantic else INVOCATION)
        or observed["producer"] != PRODUCER
        or observed.get("writer_contract") != writer_contract
        or observed.get("semantic_producer_contract") != producer
        or type(observed["source_text"]) is not str
        or type(observed["reference_date"]) is not str
    ):
        raise ValueError("bound strict extraction invocation required")
    if writer_contract is not None:
        trusted_source_event(observed, cfg["write"]["source_capture_policy"])
    policy = DspyConfig.model_validate(cfg["dspy"])
    if semantic:
        samples = row.get("generation_kwargs", {}).get("n", 1)
        if (
            policy.retry.max_attempts != 1
            or policy.validation_retries
            or (policy.tasks["event_claims"].refine or {}).get("enabled")
            or type(samples) is not int
            or samples != 1
        ):
            raise ValueError("semantic extraction requires a single native send policy")
    module = load_or_build(
        "event_claims",
        SimpleNamespace(
            dspy=policy,
            write=SimpleNamespace(
                source_capture_policy=cfg["write"]["source_capture_policy"],
                writer_contract=writer_contract,
                semantic_producer_contract=producer,
            ),
        ),
    )
    predictors = module.predictors()
    if semantic:
        import dspy

        if (
            not isinstance(module, (dspy.Predict, dspy.ChainOfThought))
            or isinstance(module, dspy.Refine)
            or any(module.named_sub_modules(type_=dspy.Refine, skip_compiled=False))
        ):
            raise ValueError("semantic extraction requires a single native send program")
    if len(predictors) != 1:
        raise ValueError("strict extraction requires one loaded native predictor")
    predictor = predictors[0]
    adapter = make_adapter(
        policy.adapter,
        canonicalize=policy.canonicalize,
        schema_mode=policy.schema_mode,
        schema_strict=policy.schema_strict,
        validation_retries=0,
    )
    inputs = {
        name: observed[name]
        for name in (
            ("source_text", "reference_date", "source_context")
            if semantic
            else ("source_text", "reference_date")
        )
    }
    if adapter.format(predictor.signature, predictor.demos, inputs) != row.get("messages"):
        raise ValueError("strict extraction differs from independently rendered native messages")
    return predictor.signature


def validate_strict_production_row(row, profile, cfg, protocol, expected):
    import json

    from gluomem.llm.structured_adapter import make_adapter, output_schema

    from gluomem_bench.bench.rack_schema import transport_binding, validate_native_projection

    binding = transport_binding(
        profile, cfg["dspy"], capture_policy=cfg["write"]["source_capture_policy"]
    )
    if (
        expected is None
        or expected.get("rack_schema_transport") != binding
        or protocol.get("rack_schema_transport") != binding
        or row.get("model") != binding["lm_model"]
    ):
        raise ValueError("unbound declared strict extraction transport")
    if row.get("task") != "event_claims" or "consolidation_invocation" in row:
        raise ValueError("unadmitted strict extraction task/invocation")
    signature = reconstructed_signature(row, cfg)
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
                "production schema differs from independently reconstructed "
                "strict extraction contract"
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
        parsed = adapter.parse(signature, row["response"]["choices"][0]["message"]["content"])
    except Exception as exc:
        raise ValueError(
            "output differs from independently reconstructed strict extraction contract"
        ) from exc
    if profile.writer_contract is not None:
        from gluomem.write.source_binding import bind_source_claim

        event, default_mid = trusted_source_event(
            row["extraction_invocation"], cfg["write"]["source_capture_policy"]
        )
        for claim in parsed["claims"]:
            if profile.writer_contract == SEMANTIC_WRITER:
                from gluomem.llm.semantic import source_witness

                source_witness(event, claim)
                continue
            bind_source_claim(
                event,
                message_id=claim.message_id if claim.message_id is not None else default_mid,
                quote=claim.verbatim_span,
                capture_policy=event.capture_policy,
                writer_contract=profile.writer_contract,
            )
    return binding["lm_model"]
