"""Closed Q14 history replay from private typed authority and physical receipts.

The enclosing owner authenticates source, effective config/artifacts, original
source events, private preparations and independent physical stream. This verifier
neither opens a store nor authorizes execution, and never repairs captured output.
"""

import hashlib
import json

from gluomem_bench.bench.rack_observed import POLICY, VERSION, PrivateAuthority

TASKS = ("event_claims", "identity_alignment", "consolidate", "answer", "rubric")


def _wire(wire):
    for name in ("request", "response"):
        if name not in wire:
            continue
        raw = wire[name + "_raw_utf8"].encode("utf-8")
        if (
            hashlib.sha256(raw).hexdigest() != wire[name + "_sha256"]
            or json.loads(raw) != wire[name]
        ):
            raise ValueError("physical raw bytes/digest/projection differ")


def reconcile_physical_attempts(rows, physical_rows):
    """Count authoritative HTTP attempts ONCE; native rows are mirrors, never additive."""
    mirrors = {}
    pre_dispatch = 0
    for row in rows:
        ordinal = row.get("ordinal")
        if ordinal is None:
            pre_dispatch += 1
            continue
        if type(ordinal) is not int or ordinal < 1 or ordinal in mirrors:
            raise ValueError("physical/native duplicate or invalid ordinal")
        mirrors[ordinal] = row
    if sorted(mirrors) != list(range(1, len(physical_rows) + 1)):
        raise ValueError("physical/native attempted coverage differs")
    result = {
        "physical_attempts": len(physical_rows),
        "native_mirrors": len(mirrors),
        "pre_dispatch_refusals": pre_dispatch,
        "unknown_usage_attempts": 0,
        "failed_http_attempts": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tasks": dict.fromkeys(TASKS, 0),
        "currency_cost": None,
    }
    for ordinal, wire in enumerate(physical_rows, 1):
        _wire(wire)
        row = mirrors[ordinal]
        task = row.get("task")
        if task not in TASKS:
            raise ValueError("unadmitted physical task")
        result["tasks"][task] += 1
        result["failed_http_attempts"] += (
            type(wire.get("status")) is not int or not 200 <= wire["status"] < 300
        )
        if "native_response" in row:
            from gluomem_bench.bench.rack_schema import validate_native_projection

            validate_native_projection(row.get("response"), row["native_response"])
            if row["native_response"]["canonical"]["provider_data"] != wire.get("response"):
                raise ValueError("physical response differs from native SDK provider data")
        response = wire.get("response")
        usage = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict) or any(
            type(usage.get(k)) is not int or usage[k] < 0
            for k in ("prompt_tokens", "completion_tokens")
        ):
            result["unknown_usage_attempts"] += 1
        else:
            result["prompt_tokens"] += usage["prompt_tokens"]
            result["completion_tokens"] += usage["completion_tokens"]
    result["known_prompt_tokens"] = result["prompt_tokens"]
    result["known_completion_tokens"] = result["completion_tokens"]
    if result["unknown_usage_attempts"]:
        result["prompt_tokens"] = None
        result["completion_tokens"] = None
    return result


def _physical(row, wire, tokenizer, profile):
    request = wire["request"]
    wanted = {
        "model": profile.model,
        "messages": row["messages"],
        "temperature": 0,
        "max_tokens": profile.output_budget(row["task"]),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if "response_format" in row["generation_kwargs"]:
        wanted["response_format"] = row["generation_kwargs"]["response_format"]
    if request != wanted:
        raise ValueError("physical native messages/schema/role budget/controls differ")
    expected_tokenizer = {
        "model": profile.model,
        "messages": row["messages"],
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if (
        tokenizer.get("request") != expected_tokenizer
        or type(tokenizer.get("response", {}).get("count")) is not int
        or tokenizer["response"]["count"] != row.get("prompt_token_count")
        or tokenizer["response"]["count"] < 0
    ):
        raise ValueError("physical served-tokenizer receipt differs")
    raw = wire.get("response", {})
    choices = raw.get("choices")
    usage = raw.get("usage", {})
    if (
        raw.get("model") != profile.model
        or type(choices) is not list
        or len(choices) != 1
        or choices[0].get("finish_reason") != "stop"
        or choices[0].get("message", {}).get("role") != "assistant"
        or any(
            type(usage.get(k)) is not int or usage[k] < 0
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
        or usage["completion_tokens"] > wanted["max_tokens"]
    ):
        raise ValueError("physical raw model/finish/integer usage differs")
    from gluomem_bench.bench.rack_observed import validate_observed_response

    validate_observed_response(row, profile)


def _parsed(row, cfg, authority):
    from gluomem.llm.structured_adapter import make_adapter

    if row["task"] == "event_claims":
        from gluomem_bench.bench.rack_extraction import reconstructed_signature

        signature = reconstructed_signature(row, cfg)
    else:
        from gluomem_bench.bench.rack_observed import reconstructed_signature

        signature = reconstructed_signature(row, cfg, authority=authority)
    adapter = make_adapter(
        cfg["dspy"]["adapter"],
        canonicalize=cfg["dspy"].get("canonicalize", False),
        schema_mode=cfg["dspy"]["schema_mode"],
        schema_strict=cfg["dspy"]["schema_strict"],
        validation_retries=0,
    )
    return adapter.parse(signature, row["response"]["choices"][0]["message"]["content"])


def validate_observed_history(
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
    tokenizer_rows,
    expected_requests=None,
):
    """Complete source/typed/physical replay; no sockets, target repair or store writes."""

    from gluomem_bench.bench.rack_schema import validate_production_row, validate_writer_config

    if (
        profile.profile_version != VERSION
        or profile.semantic_producer_contract != POLICY
        or type(authority) is not PrivateAuthority
    ):
        raise ValueError("separately admitted observed producer history required")
    validate_writer_config(profile, cfg)
    totals = reconcile_physical_attempts(rows, physical_rows)
    if any("error" in row for row in rows):
        raise ValueError("recorded native operation error; physical accounting is not success")
    if (
        totals["pre_dispatch_refusals"]
        or totals["failed_http_attempts"]
        or totals["unknown_usage_attempts"]
    ):
        raise ValueError("incomplete observed history; retained physical accounting is not success")
    if len(tokenizer_rows) != len(physical_rows):
        raise ValueError("served-tokenizer coverage differs")
    expected_requests = expected_requests or {}
    parsed = []
    for row, wire, tokenizer in zip(rows, physical_rows, tokenizer_rows, strict=True):
        _physical(row, wire, tokenizer, profile)
        if row["task"] in {"answer", "rubric"}:
            if expected_requests.get(row["ordinal"]) != wire["request"]:
                raise ValueError("unbound plain reader/judge request")
            parsed.append(None)
        else:
            validate_production_row(row, profile, cfg, protocol, expected, authority=authority)
            parsed.append(_parsed(row, cfg, authority))
    return {
        **totals,
        **replay_source_history(rows, parsed, authority, prepared_events, source_events),
        "protocol": "observed-producer-history-replay@v1",
        "model_quality_admitted": False,
        "execution_authorized": False,
    }


def replay_source_history(
    rows, parsed, authority, prepared_events, source_events, *, producer_contract=POLICY
):
    """Shared unchanged source-only semantic replay, independent of physical dialect."""
    from gluomem.llm.observed_identity import IdentityProposal, PropertyIdentityProposal
    from gluomem.llm.outputs import ProjectedConsolidationDecision
    from gluomem.llm.semantic import source_witness
    from gluomem.timeutil import to_storage
    from gluomem.write.events import SourceEvent

    events = [SourceEvent.model_validate(e) for e in source_events]
    from gluomem_bench.bench.rack_observed import producer_types

    context_type, _, _, projection_type = producer_types(producer_contract)
    property_identity = producer_contract in {
        "observed-property-identity-v3",
        "observed-assertion-scope-v4",
    }
    proposal_type = PropertyIdentityProposal if property_identity else IdentityProposal
    if len(events) != len(prepared_events):
        raise ValueError("private prepared/source coverage differs")
    records = authority.snapshot()
    call_iter = iter(records.items())
    cursor = 0
    no_identity = 0
    for event, prepared in zip(events, prepared_events, strict=True):
        if prepared.get("event") != event.model_dump(mode="json"):
            raise ValueError("private prepared source event differs")
        if cursor >= len(rows) or rows[cursor]["task"] != "event_claims":
            raise ValueError("observed source extraction order/coverage differs")
        if rows[cursor]["extraction_invocation"]["source_event"] != event.model_dump(mode="json"):
            raise ValueError("native extraction differs from independently bound source event")
        claims = parsed[cursor]["claims"]
        cursor += 1
        binding = json.loads(prepared["producer_inputs"])
        if (
            binding.get("policy") != producer_contract
            or len(binding.get("claims", [])) != len(claims)
            or len(binding.get("projections", [])) != len(claims)
            or len(binding.get("decisions", [])) != len(claims)
        ):
            raise ValueError("private prepared producer/claim coverage differs")
        for index, spec in enumerate(claims):
            witness = source_witness(event, spec)
            context = context_type(
                kind=event.kind or spec.kind,
                source_role=witness.source_role,
                observed_at=witness.observed_at,
                effective_at=to_storage(spec.event_date) if spec.event_date is not None else None,
                meaning=spec.meaning,
                witness=witness,
            )
            projection = projection_type.model_validate(binding["projections"][index])
            claimed = binding["claims"][index]
            if (
                projection.claim_index != index
                or projection.context != context
                or claimed["content"] != spec.text.strip()
                or claimed["meaning"] != spec.meaning.model_dump(mode="json")
                or claimed["semantic_witness"] != witness.model_dump(mode="json")
                or any(
                    claimed[k] != getattr(context, k)
                    for k in ("kind", "source_role", "observed_at", "effective_at")
                )
            ):
                raise ValueError(
                    "private preparation differs from extracted source/role/date/witness"
                )
            for item in projection.identity_context.targets:
                previous = item.target.claim_index
                if previous is not None and (
                    previous >= index
                    or item.meaning != claims[previous].meaning
                    or item.target.content != claims[previous].text.strip()
                    or binding["decisions"][previous]["action"]
                    not in {"ADD", "SUPERSEDE", "TRANSITION"}
                ):
                    raise ValueError(
                        "identity prior-publication authority differs from source/actions"
                    )
            if projection.identity_context.targets:
                reference, record = next(call_iter, (None, None))
                if (
                    record is None
                    or record["task"] != "identity_alignment"
                    or record["new_claim_context"] != context.model_dump(mode="json")
                    or record["projection"] != projection.identity_context.model_dump(mode="json")
                    or record["new_claim"] != spec.text.strip()
                ):
                    raise ValueError(
                        "private identity observation differs from extracted source/projection"
                    )
                if (
                    cursor >= len(rows)
                    or rows[cursor]["task"] != "identity_alignment"
                    or rows[cursor]["identity_invocation"]["authority_ref"] != reference
                    or record["native_sends"] != 1
                    or record.get("result") != parsed[cursor]["proposal"].model_dump(mode="json")
                    or proposal_type.model_validate(record["result"]) != projection.proposal
                ):
                    raise ValueError("identity physical/proposal coverage differs")
                cursor += 1
            else:
                no_identity += 1
                wanted = proposal_type(
                    relation="independent"
                    if context.meaning.independently_versioned
                    else "unresolved",
                    **(
                        {
                            "assertion_relation": "unrelated"
                            if context.meaning.independently_versioned
                            else "unresolved",
                            "subject_alignment": "not_aligned",
                            "property_alignment": "not_aligned",
                        }
                        if property_identity
                        else {}
                    ),
                )
                if projection.identity_call != "empty-context" or projection.proposal != wanted:
                    raise ValueError("deterministic identity no-call proposal differs")
            reference, record = next(call_iter, (None, None))
            if (
                record is None
                or record["task"] != "consolidate"
                or record["new_claim"] != spec.text.strip()
                or record["new_claim_context"] != context.model_dump(mode="json")
                or record["projection"] != projection.model_dump(mode="json")
            ):
                raise ValueError("private consolidation observation differs from prepared source")
            selected = ProjectedConsolidationDecision.model_validate(binding["decisions"][index])
            projection.validate_decision(selected)
            if record.get("result") != selected.model_dump(mode="json"):
                raise ValueError("prepared decision differs from actual core return")
            native = bool(projection.targets) or not (
                "ADD" in projection.untargeted_actions
                if producer_contract == "observed-assertion-scope-v4"
                else context.meaning.independently_versioned
            )
            if native:
                if (
                    cursor >= len(rows)
                    or rows[cursor]["task"] != "consolidate"
                    or rows[cursor]["consolidation_invocation"]["authority_ref"] != reference
                    or record["native_sends"] != 1
                    or parsed[cursor]["decision"].model_dump(mode="json")
                    != selected.model_dump(mode="json")
                ):
                    raise ValueError("consolidation physical/decision coverage differs")
                cursor += 1
            elif selected.action != "ADD" or record["native_sends"] != 0:
                raise ValueError("deterministic consolidation no-call path differs")
    if next(call_iter, None) is not None:
        raise ValueError("unconsumed private producer observations")
    while cursor < len(rows) and rows[cursor]["task"] in {"answer", "rubric"}:
        cursor += 1
    if cursor != len(rows):
        raise ValueError("unconsumed native producer attempts")
    return {"source_events": len(events), "deterministic_identity_no_calls": no_identity}
