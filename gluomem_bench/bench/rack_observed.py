"""Versioned Q14 native observations with separately retained PRIVATE authority.

No model, target selection, source rewrite, persistence or accounting is implemented
here. The facade delegates to the released core. Full receipts never enter native
rows: callers must keep PrivateAuthority private and authenticate it independently
before replay; it is not a tracking payload or an execution allowance.
"""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

POLICY = "observed-assertion-date-v1"
PROPERTY_POLICY = "observed-property-identity-v3"
ASSERTION_POLICY = "observed-assertion-scope-v4"
PROVIDER_POLICIES = (POLICY, PROPERTY_POLICY, ASSERTION_POLICY)
VERSION = "observed-producer-v12"
IDENTITY_INVOCATION = "observed-identity-invocation@v1"
CONSOLIDATION_INVOCATION = "observed-projected-consolidation-invocation@v1"
_ACTIVE_AUTHORITY = ContextVar("rack_observed_private_authority", default=None)
_ACTIVE_INVOCATION = ContextVar("rack_observed_native_invocation", default=None)


def _json(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    return value


class PrivateAuthority:
    """Invocation-local private evidence; export ONLY into authenticated private storage."""

    def __init__(self, records=None):
        self._records = deepcopy(records) if records is not None else {}

    def snapshot(self):
        return deepcopy(self._records)

    def lookup(self, reference):
        if type(reference) is not str or reference not in self._records:
            raise ValueError("independently retained private invocation authority required")
        return deepcopy(self._records[reference])

    @contextmanager
    def observe(
        self, task, new_claim, candidates, context, projection, *, producer_contract=POLICY
    ):
        reference = uuid4().hex  # Opaque, not a reusable content-derived public identifier.
        record = {
            "task": task,
            "semantic_producer_contract": producer_contract,
            "new_claim": new_claim,
            "candidates": _json(candidates),
            "new_claim_context": _json(context),
            "projection": _json(projection),
            "native_sends": 0,
        }
        self._records[reference] = record
        token = _ACTIVE_AUTHORITY.set((self, reference))
        try:
            yield reference
        except Exception:
            record["outcome"] = "refused"
            raise
        finally:
            _ACTIVE_AUTHORITY.reset(token)

    def finish(self, reference, result):
        self._records[reference]["result"] = _json(result)
        self._records[reference]["outcome"] = "returned"


def observe_extractors(extractors, authority):
    """Preserve the actual released producer and observe its public typed boundaries."""
    if (
        getattr(extractors, "semantic_producer_capability", None) not in PROVIDER_POLICIES
        or type(authority) is not PrivateAuthority
    ):
        raise ValueError("observed facade requires the released producer and private authority")
    return _ObservedExtractors(extractors, authority)


class _ObservedExtractors:
    def __init__(self, inner, authority):
        self._inner, self._authority = inner, authority

    @property
    def semantic_producer_capability(self):
        return self._inner.semantic_producer_capability

    @property
    def consolidation_projection_protocol(self):
        return self._inner.consolidation_projection_protocol

    @property
    def writer_capability(self):
        return self._inner.writer_capability

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def align_claim_identity(self, new_claim, candidates, *, context, projection):
        with self._authority.observe(
            "identity_alignment",
            new_claim,
            candidates,
            context,
            projection,
            producer_contract=self.semantic_producer_capability,
        ) as ref:
            result = self._inner.align_claim_identity(
                new_claim, candidates, context=context, projection=projection
            )
            self._authority.finish(ref, result)
            return result

    def consolidate_claim_with_projection(self, new_claim, candidates, *, context, projection):
        with self._authority.observe(
            "consolidate",
            new_claim,
            candidates,
            context,
            projection,
            producer_contract=self.semantic_producer_capability,
        ) as ref:
            result = self._inner.consolidate_claim_with_projection(
                new_claim, candidates, context=context, projection=projection
            )
            self._authority.finish(ref, result)
            return result


def producer_types(producer_contract):
    """Select exact known leaves only from independently bound producer policy."""
    if producer_contract == ASSERTION_POLICY:
        from gluomem.llm.assertion_scope import (
            AssertionScopedCandidate,
            AssertionScopedContext,
            AssertionScopedIdentityContext,
            AssertionScopedProjection,
        )

        return (
            AssertionScopedContext,
            AssertionScopedCandidate,
            AssertionScopedIdentityContext,
            AssertionScopedProjection,
        )
    if producer_contract not in (POLICY, PROPERTY_POLICY):
        raise ValueError("unadmitted observed producer types")
    from gluomem.llm.observed_identity import IdentityContext, ObservedSemanticProjection
    from gluomem.llm.semantic import SemanticClaimContext, SemanticStoredCandidate

    return (
        SemanticClaimContext,
        SemanticStoredCandidate,
        IdentityContext,
        ObservedSemanticProjection,
    )


def _typed(record, task, *, producer_contract=POLICY):
    from gluomem.llm.outputs import StoredCandidate

    context_type, candidate_type, identity_type, projection_type = producer_types(producer_contract)

    from gluomem_bench.bench.rack_projection import _validate_semantic_receipts

    if (
        record.get("task") != task
        or producer_contract not in PROVIDER_POLICIES
        or record.get("semantic_producer_contract") != producer_contract
        or type(record.get("new_claim")) is not str
    ):
        raise ValueError("private producer invocation task/policy differs")
    context = context_type.model_validate(record["new_claim_context"])
    candidates = [candidate_type.model_validate(c) for c in record["candidates"]]
    _validate_semantic_receipts(candidates, producer_contract=producer_contract)
    if task == "identity_alignment":
        projection = identity_type.model_validate(record["projection"])
        if not projection.targets:
            raise ValueError("empty identity context cannot represent a native invocation")
        projection.validate_candidates(candidates, context)
        wire_projection = projection
    elif task == "consolidate":
        projection = projection_type.model_validate(record["projection"])
        if projection.context != context or projection.meaning != context.meaning:
            raise ValueError("private consolidation source/meaning differs")
        projection.validate_candidates(candidates)
        projection.identity_context.validate_candidates(candidates, context)
        for binding in projection.target_meanings:
            if binding.semantic is not None and binding.meaning != binding.semantic.meaning:
                raise ValueError("private target meaning differs from observed semantic receipt")
        wire_projection = projection.wire_view()
    else:
        raise ValueError("unadmitted observed producer task")
    inputs = {"new_claim": record["new_claim"], "new_claim_context": context}
    if task == "identity_alignment":
        inputs["identity_context"] = wire_projection
        if producer_contract in (PROPERTY_POLICY, ASSERTION_POLICY):
            from gluomem.llm.property_identity import property_slot_context

            inputs["property_slot_context"] = property_slot_context(projection, context, candidates)
    else:
        inputs.update(
            candidates=[StoredCandidate.model_validate(c.model_dump()) for c in candidates],
            projection=wire_projection,
        )
    return context, projection, inputs


class ObservedAdapter:
    """Bind wire-only inputs to the private typed authority at the actual adapter."""

    def __init__(self, adapter, profile, task):
        self.adapter, self.profile, self.task = adapter, profile, task

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        active = _ACTIVE_AUTHORITY.get()
        if self.profile.semantic_producer_contract not in PROVIDER_POLICIES or active is None:
            raise ValueError("observed native task requires explicit private producer observation")
        authority, reference = active
        _, _, expected = _typed(
            authority.lookup(reference),
            self.task,
            producer_contract=self.profile.semantic_producer_contract,
        )
        if _json(inputs) != _json(expected):
            raise ValueError("observed native inputs differ from private typed authority")
        key = "identity_context" if self.task == "identity_alignment" else "projection"
        identity_type = producer_types(self.profile.semantic_producer_contract)[2]

        if (self.task == "identity_alignment" and type(inputs[key]) is not identity_type) or (
            self.task == "consolidate" and type(inputs[key]) is not dict
        ):
            raise ValueError("observed native input types differ")
        invocation = {
            "protocol": IDENTITY_INVOCATION
            if self.task == "identity_alignment"
            else CONSOLIDATION_INVOCATION,
            "producer": "DspyExtractors."
            + (
                "align_claim_identity"
                if self.task == "identity_alignment"
                else "consolidate_claim_with_projection"
            ),
            "semantic_producer_contract": self.profile.semantic_producer_contract,
            "authority_ref": reference,
            **_json(inputs),
        }
        state = {"task": self.task, "invocation": invocation, "sends": 0, "authority": authority}
        token = _ACTIVE_INVOCATION.set(state)
        try:
            return self.adapter(lm, lm_kwargs, signature, demos, inputs)
        finally:
            _ACTIVE_INVOCATION.reset(token)


def observed_invocation(task):
    state = _ACTIVE_INVOCATION.get()
    if state is None or state["task"] != task or state["sends"]:
        raise ValueError("observed producer requires exactly one invocation-owned native send")
    state["sends"] += 1
    reference = state["invocation"]["authority_ref"]
    state["authority"]._records[reference]["native_sends"] += 1
    return deepcopy(state["invocation"])


def reconstructed_signature(row, cfg, *, authority):
    """Load the effective artifacts/demos/limits, then rebuild from PRIVATE typed inputs."""
    import dspy
    from gluomem.config.app_config import DspyConfig
    from gluomem.llm.bootstrap import load_or_build
    from gluomem.llm.observed_signatures import identity_alignment_module
    from gluomem.llm.signatures import projected_consolidation_module
    from gluomem.llm.structured_adapter import make_adapter

    task = row.get("task")
    if (
        cfg.get("write", {}).get("semantic_producer_contract") not in PROVIDER_POLICIES
        or task not in {"identity_alignment", "consolidate"}
        or type(authority) is not PrivateAuthority
    ):
        raise ValueError("explicit observed producer private authority required")
    name = "identity_invocation" if task == "identity_alignment" else "consolidation_invocation"
    observed = row.get(name)
    if not isinstance(observed, dict):
        raise ValueError("bound observed producer invocation required")
    reference = observed.get("authority_ref")
    record = authority.lookup(reference)
    producer_contract = cfg["write"]["semantic_producer_contract"]
    context, projection, inputs = _typed(record, task, producer_contract=producer_contract)
    expected = {
        "protocol": IDENTITY_INVOCATION
        if task == "identity_alignment"
        else CONSOLIDATION_INVOCATION,
        "producer": "DspyExtractors."
        + (
            "align_claim_identity"
            if task == "identity_alignment"
            else "consolidate_claim_with_projection"
        ),
        "semantic_producer_contract": producer_contract,
        "authority_ref": reference,
        **_json(inputs),
    }
    if observed != expected:
        raise ValueError(
            "observed invocation differs from independently retained private authority"
        )
    policy = DspyConfig.model_validate(cfg["dspy"])
    task_policy = policy.tasks[task]
    if (
        policy.retry.max_attempts != 1
        or policy.validation_retries
        or (task_policy.refine or {}).get("enabled")
        or type(row.get("generation_kwargs", {}).get("n", 1)) is not int
        or row.get("generation_kwargs", {}).get("n", 1) != 1
    ):
        raise ValueError("observed producer requires single native attempt policy")
    native_cfg = SimpleNamespace(dspy=policy, write=SimpleNamespace(**cfg["write"]))
    base = load_or_build(task, native_cfg)
    if (
        type(base) not in {dspy.Predict, dspy.ChainOfThought}
        or len(base.predictors()) != 1
        or any(base.named_sub_modules(type_=dspy.Refine, skip_compiled=False))
    ):
        raise ValueError("unadmitted observed native program")
    module = (
        identity_alignment_module(base, projection, context, producer_contract=producer_contract)
        if task == "identity_alignment"
        else projected_consolidation_module(
            base,
            projection,
            writer_contract=cfg["write"]["writer_contract"],
            semantic_producer_contract=producer_contract,
        )
    )
    predictor = module.predictors()[0]
    adapter = make_adapter(
        policy.adapter,
        canonicalize=policy.canonicalize,
        schema_mode=policy.schema_mode,
        schema_strict=policy.schema_strict,
        validation_retries=0,
    )
    if adapter.format(predictor.signature, predictor.demos, inputs) != row.get("messages"):
        raise ValueError("observed invocation differs from independently rendered native messages")
    return predictor.signature


def validate_production_row(row, profile, cfg, protocol, expected, *, authority):
    from gluomem.llm.structured_adapter import _object, make_adapter, output_schema
    from jsonschema import Draft202012Validator

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
        raise ValueError("unbound declared observed producer transport")
    signature = reconstructed_signature(row, cfg, authority=authority)
    task = row["task"]
    kwargs = row.get("generation_kwargs", {})
    if (
        row.get("output_budget") != profile.output_budget(task)
        or kwargs.get("max_tokens") != profile.output_budget(task)
        or cfg["dspy"]["tasks"][task]["max_tokens"] != profile.output_budget(task)
    ):
        raise ValueError("observed producer output allowance differs")
    schema = output_schema(signature)
    wanted = {
        "type": "json_schema",
        "json_schema": {
            "name": "GluoMemOutput",
            "strict": binding["schema_strict"],
            "schema": schema,
        },
    }
    if json.dumps(kwargs.get("response_format"), sort_keys=True) != json.dumps(
        wanted, sort_keys=True
    ):
        raise ValueError("observed production schema differs from independently rebuilt contract")
    validate_native_projection(row.get("response"), row.get("native_response", {}))
    text = row["response"]["choices"][0]["message"]["content"]
    try:
        Draft202012Validator(schema).validate(json.loads(text, object_pairs_hook=_object))
        adapter = make_adapter(
            cfg["dspy"]["adapter"],
            canonicalize=cfg["dspy"].get("canonicalize", False),
            schema_mode=binding["schema_mode"],
            schema_strict=binding["schema_strict"],
            validation_retries=0,
        )
        adapter.parse(signature, text)
    except Exception as exc:
        raise ValueError("output differs from independently rebuilt observed contract") from exc
    return binding["lm_model"]


def validate_observed_response(row, profile):
    """New opt-in raw/schema guard before returning from the history-owned LM."""
    from gluomem.llm.structured_adapter import _object
    from jsonschema import Draft202012Validator
    from referencing import Registry

    raw = row.get("native_response", {}).get("canonical", {}).get("provider_data")
    if type(raw) is not dict:
        raise ValueError("complete original SDK raw response required")
    usage = raw.get("usage", {})
    choices = raw.get("choices")
    if (
        raw.get("model") != profile.model
        or type(usage) is not dict
        or type(choices) is not list
        or len(choices) != 1
        or type(choices[0]) is not dict
        or choices[0].get("finish_reason") != "stop"
        or choices[0].get("message", {}).get("role") != "assistant"
        or any(
            type(usage.get(k)) is not int or usage[k] < 0
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
        or usage["completion_tokens"] > row["output_budget"]
    ):
        raise ValueError("observed raw model/finish/integer usage contract differs")
    if profile.profile_version == "observed-provider-v13" and (
        choices[0]["message"].get("refusal") is not None or choices[0]["message"].get("tool_calls")
    ):
        raise ValueError("observed native refusal/tool output cannot be successful")
    from gluomem_bench.bench.rack_schema import validate_native_projection

    evidence = row["native_response"]
    normalized = row.get("response")
    validate_native_projection(normalized, evidence)
    text = choices[0]["message"].get("content")
    canonical_usage = evidence["canonical"]["usage"]
    if (
        type(text) is not str
        or normalized["model"] != raw["model"]
        or normalized["choices"][0]["message"]["role"] != choices[0]["message"]["role"]
        or normalized["choices"][0]["message"]["content"] != text
        or normalized["choices"][0]["finish_reason"] != choices[0]["finish_reason"]
        or normalized["usage"]
        != (
            {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
            if profile.profile_version == "observed-provider-v13"
            else usage
        )
        or any(type(value) is not int for value in normalized["usage"].values())
        or any(
            type(canonical_usage.get(k)) is not int
            for k in ("input_tokens", "output_tokens", "total_tokens")
        )
        or row.get("outputs") != [text]
        or row.get("finish_reasons") != ["stop"]
    ):
        raise ValueError("observed raw/normalized response differs from canonical/native outputs")
    response_format = row["generation_kwargs"].get("response_format")
    if response_format is None:
        if row["task"] not in {"answer", "rubric"}:
            raise ValueError("observed producer requires native output schema")
        return
    try:
        if response_format["type"] != "json_schema":
            raise ValueError("native schema required")
        payload = json.loads(raw["choices"][0]["message"]["content"], object_pairs_hook=_object)
        Draft202012Validator(
            response_format["json_schema"]["schema"], registry=Registry()
        ).validate(payload)
    except Exception as exc:
        raise ValueError("observed output violates the exact sent schema; no regeneration") from exc
