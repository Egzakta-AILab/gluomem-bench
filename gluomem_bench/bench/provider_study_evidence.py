"""Explicitly versioned saved-wire replay; legacy scope admission stays closed."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from gluomem_bench.bench.artifacts import verify_artifact_hashes
from gluomem_bench.bench.rack import RackProfile
from gluomem_bench.bench.rack_provider import (
    ENDPOINT_VERSION,
    POOL_VERSION,
    PRODUCTION_PROVIDER_VERSIONS,
    VERSION,
    endpoint_discovery,
    estimated_prompt_tokens,
    generation_options,
    is_openrouter,
    response_evidence,
)
from gluomem_bench.bench.types import Sample
from gluomem_bench.bench.upstream_scoring import load_longmemeval_prompt

PROTOCOL = "adapted:longmemeval-provider-study@v8"
ENDPOINT_PROTOCOL = "adapted:longmemeval-provider-study@v9"
POOL_PROTOCOL = "adapted:longmemeval-provider-study@v10"
HISTORY_PROTOCOLS = {
    VERSION: PROTOCOL,
    ENDPOINT_VERSION: ENDPOINT_PROTOCOL,
    POOL_VERSION: POOL_PROTOCOL,
}


def history_protocol(profile):
    return HISTORY_PROTOCOLS[profile.profile_version]


def validate_history_scope(profile, system):
    if profile.semantic_producer_contract is not None:
        raise ValueError(
            "observed producer bridge is not study/operator admission; all cells unqualified"
        )
    if (
        profile.pricing_policy is not None
        or profile.judge_profile is not None
        and profile.judge_profile.pricing_policy is not None
    ):
        raise ValueError("bounded pricing is not benchmark admission; all cells unqualified")
    if profile.profile_version == "provider-roles-v11":
        raise ValueError(
            "v11 targeted diagnostic is not benchmark admission; all cells unqualified"
        )
    if profile.profile_version == POOL_VERSION:
        from gluomem_bench.bench.provider_study import POOL_SUPPORTED

        if system not in POOL_SUPPORTED:
            raise ValueError("v10 native comparator/control unqualified")
    if profile.profile_version == ENDPOINT_VERSION and system != "gluomem":
        raise ValueError(
            "v9 endpoint contract supports only native gluomem; comparator/control unqualified"
        )


def read_json(path):
    return json.loads(Path(path).read_bytes())


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line]


def key(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def validate_config(cfg, profile):
    """Validate actual role bindings, never accept plain LMRouter commercial defaults."""
    if (
        profile.semantic_producer_contract is not None
        or cfg.get("write", {}).get("semantic_producer_contract") is not None
    ):
        raise ValueError("observed producer study/operator collector unqualified")
    judge = profile.judge_profile or profile
    if profile.profile_version == POOL_VERSION:
        if (
            profile.judge_profile is None
            or judge.profile_version != POOL_VERSION
            or judge.judge_profile is not None
            or judge.production_json_transport is not None
            or judge.consolidation_contract is not None
            or judge.extraction_contract is not None
            or judge.openrouter_pool != profile.openrouter_pool
            or profile.production_json_transport is not None
            or profile.extraction_contract != "strict-source-quotes-v1"
            or profile.consolidation_contract != "evolving-per-claim-v1"
            or cfg["dspy"]["schema_mode"] != "required"
            or cfg["dspy"]["schema_strict"] is not True
            or cfg["dspy"]["validation_retries"] != 0
            or cfg["dspy"]["retry"]["max_attempts"] != 1
            or any((t.get("refine") or {}).get("enabled") for t in cfg["dspy"]["tasks"].values())
        ):
            raise ValueError(
                "v10 strict native primary and independent flat common judge required; "
                "no semantic retries"
            )
    elif profile.profile_version == ENDPOINT_VERSION:
        if (
            judge.profile_version not in PRODUCTION_PROVIDER_VERSIONS
            or not is_openrouter(judge)
            or judge.consolidation_contract is not None
            or judge.production_json_transport is not None
            or cfg["dspy"]["schema_mode"] != "required"
            or cfg["dspy"]["schema_strict"] is not True
        ):
            raise ValueError(
                "v9 requires an independent native common judge and required strict "
                "production schema"
            )
    elif judge.provider != "openrouter-baidu-fp8":
        raise ValueError("V8 LME study requires the common exact DeepSeek judge")
    if (
        cfg["components"]["allow_fake"]
        or cfg["dspy"]["model"] != "rack-primary"
        or cfg["dspy"]["schema_mode"]
        != ("typed-text" if profile.production_json_transport else "required")
        or cfg["write"]["source_capture_policy"] != "attributed-interaction"
    ):
        raise ValueError("nonisolated provider study configuration")
    for role, active in (
        ("event_claims", profile),
        ("consolidate", profile),
        ("answer", profile),
        ("rubric", judge),
    ):
        task = cfg["dspy"]["tasks"][role]
        name = "rack-judge" if role == "rubric" and profile.judge_profile else "rack-primary"
        model = cfg["models"][name]
        if (
            task["model"] != name
            or task["temperature"] != 0
            or task["max_tokens"] != active.output_budget(role)
            or model["provider_model"] != "openai/" + active.model
            or model["api_base"] != active.base_url
            or model["api_key_env"] != active.api_key_env
            or model["extra"]["num_retries"] != 0
            or model["extra"]["extra_body"] != generation_options(active)
        ):
            raise ValueError("provider study effective role binding differs")


def validate_discovery(rows, profile):
    path = "/models/" + profile.model + "/endpoints" if is_openrouter(profile) else "/models"
    found = [
        r for r in rows if r.get("method") == "GET" and r.get("url") == profile.base_url + path
    ]
    if len(found) != 1 or found[0].get("status") != 200:
        raise ValueError("exact provider metadata evidence missing")
    result = found[0]["response"]
    if is_openrouter(profile):
        endpoint_discovery(profile, result)
    else:
        selected = [r for r in result["data"] if r["id"] == profile.model]
        if len(selected) != 1 or selected[0].get("max_model_len") != profile.context_window:
            raise ValueError("served Qwen model/context differs")


def token_counts(rows, profile):
    counts = {}
    for row in rows:
        if not row.get("url", "").endswith("/tokenize"):
            continue
        if profile.token_count_policy != "served-tokenizer":
            raise ValueError("DeepSeek must not invent a served tokenizer")
        request, value = row["request"], row.get("response", {}).get("count")
        if (
            row.get("status") != 200
            or type(value) is not int
            or value < 0
            or request.get("model") != profile.model
            or request.get("chat_template_kwargs") != {"enable_thinking": False}
            or request.get("add_generation_prompt") is not True
        ):
            raise ValueError("invalid exact served-tokenizer evidence")
        k = key(request["messages"])
        if k in counts and counts[k] != value:
            raise ValueError("inconsistent served-tokenizer evidence")
        counts[k] = value
    return counts


def validate_response(profile, request, response, *, output, counts):
    if request.get("model") != profile.model or request.get("temperature") != 0:
        raise ValueError("provider generation role/model differs")
    options = generation_options(profile)
    if any(request.get(k) != v for k, v in options.items()):
        raise ValueError("provider generation controls differ")
    if profile.profile_version == ENDPOINT_VERSION and "service_tier" in request:
        raise ValueError("v9 standard provider controls forbid service-tier overrides")
    if is_openrouter(profile) and "chat_template_kwargs" in request:
        raise ValueError("Qwen controls forwarded to DeepSeek")
    if request.get("max_tokens", request.get("max_completion_tokens")) != output:
        raise ValueError("provider generation output allowance differs")
    usage, choices = response.get("usage", {}), response.get("choices", [])
    if (
        response.get("model") != profile.model
        or len(choices) != 1
        or choices[0].get("finish_reason") != "stop"
        or not isinstance(choices[0].get("message", {}).get("content"), str)
        or any(
            type(usage.get(k)) is not int or usage[k] < 0
            for k in ("prompt_tokens", "completion_tokens")
        )
        or usage["completion_tokens"] > output
    ):
        raise ValueError("incomplete or invalid actual provider response")
    count = (
        estimated_prompt_tokens(request["messages"])
        if profile.token_count_policy == "utf8-conservative-v1"
        else counts.get(key(request["messages"]))
    )
    if count is None:
        raise ValueError("exact Qwen served-tokenizer admission missing")
    if profile.token_count_policy == "served-tokenizer" and usage["prompt_tokens"] != count:
        raise ValueError("Qwen served-tokenizer usage mismatch")
    if count + output + profile.schema_reserve_tokens > profile.context_window:
        raise ValueError("provider admitted context overflow")
    return response_evidence(profile, response, count, output_budget=output)


def validate_provider_bundle(
    out, manifest, *, expected=None, require_real=False, judge_source=None
):
    if manifest.get("protocol") == POOL_PROTOCOL:
        return validate_pool_bundle(out, manifest, expected=expected, judge_source=judge_source)
    from gluomem_bench.bench.suite_evidence import REQUIRED

    out = Path(out)
    inventory = manifest.get("artifact_sha256", {})
    if not REQUIRED <= inventory.keys():
        raise ValueError("required evidence missing from artifact hash inventory")
    verify_artifact_hashes(out, inventory, canonical_names=True)
    if (out / "failure.json").exists():
        raise ValueError("failed provider workload cannot be published")
    protocol = read_json(out / "protocol.json")
    if (
        manifest.get("protocol") not in HISTORY_PROTOCOLS.values()
        or manifest.get("dataset") != "longmemeval_s"
    ):
        raise ValueError("V8 provider scope admits only LongMemEval-S")
    for k in ("protocol", "dataset", "system", "sample_ids"):
        if protocol.get(k) != manifest.get(k) or (
            expected is not None and k != "protocol" and expected.get(k) != manifest.get(k)
        ):
            raise ValueError("provider workload differs from frozen plan")
    rows = read_json(out / "predictions.json")
    if (
        not rows
        or [r["sample"]["id"] for r in rows] != manifest["sample_ids"]
        or len(set(manifest["sample_ids"])) != len(rows)
        or any(r["prediction"].get("failure") for r in rows)
    ):
        raise ValueError("incomplete provider prediction coverage")
    if (
        hashlib.sha256(json.dumps([r["sample"] for r in rows], sort_keys=True).encode()).hexdigest()
        != protocol["samples_sha256"]
    ):
        raise ValueError("provider sample binding differs")
    profile = RackProfile.model_validate(read_json(out / "profile.json"))
    if (
        profile.profile_version not in PRODUCTION_PROVIDER_VERSIONS
        or manifest["protocol"] != history_protocol(profile)
        or hashlib.sha256(profile.model_dump_json().encode()).hexdigest()
        != protocol["profile_sha256"]
    ):
        raise ValueError("provider profile binding differs")
    validate_history_scope(profile, manifest["system"])
    cfg = read_json(out / "effective-config.json")
    validate_config(cfg, profile)
    primary = read_lines(out / "calls.jsonl")
    judge_profile = profile.judge_profile or profile
    judge_rows = read_lines(out / "calls-judge.jsonl") if profile.judge_profile else primary
    if profile.judge_profile and "calls-judge.jsonl" not in inventory:
        raise ValueError("common judge stream missing from inventory")
    validate_discovery(primary, profile)
    if profile.judge_profile:
        validate_discovery(judge_rows, judge_profile)
    counts = token_counts(primary, profile)
    judge_counts = token_counts(judge_rows, judge_profile)
    direct = [r for r in primary if r.get("url") == profile.base_url + "/chat/completions"]
    judge_calls = (
        [r for r in judge_rows if r.get("url") == judge_profile.base_url + "/chat/completions"]
        if profile.judge_profile
        else []
    )
    readers = []
    mem0_calls = 0
    mem0_direct = []
    if profile.judge_profile is None:
        judge_calls = [r for r in direct if r["request"]["messages"][0]["role"] != "system"]
        direct = [r for r in direct if r not in judge_calls]
    if manifest["system"] == "mem0":
        from gluomem_bench.bench.suite_evidence import validate_mem0_evidence

        mem0_calls = validate_mem0_evidence(
            out, inventory, profile, protocol, direct, expected=expected
        )
        mem0_direct, direct = direct[:mem0_calls], direct[mem0_calls:]
        for row in mem0_direct:
            evidence = validate_response(
                profile,
                row["request"],
                row["response"],
                output=profile.output_budget("event_claims"),
                counts=counts,
            )
            if any(row.get(k) != v for k, v in evidence.items()):
                raise ValueError("native Mem0 provider evidence differs")
        clock = protocol.get("mem0_native_clock", {})
        if (
            clock.get("mode") != "native"
            or clock.get("source_dates_override_native_clock") is not False
        ):
            raise ValueError(
                "Mem0 source-clock override unsupported; native clock disclosure required"
            )
    for row in direct:
        if row.get("status") != 200:
            raise ValueError("failed generation in complete provider workload")
        request = row["request"]
        if request["messages"][0] != {"role": "system", "content": protocol["reader_instruction"]}:
            raise ValueError("unaccounted native provider generation")
        evidence = validate_response(
            profile, request, row["response"], output=profile.output_budget("answer"), counts=counts
        )
        if any(row.get(k) != v for k, v in evidence.items()):
            raise ValueError("returned provider receipt differs")
        readers.append(row)
    expected_readers = [r for r in rows if not r["prediction"].get("abstained")]
    if len(readers) != len(expected_readers) or len(judge_calls) != len(rows):
        raise ValueError("provider reader/common judge coverage differs")
    if "evidence_token_budget" in protocol:
        validate_reader_packing(
            out, inventory, protocol, manifest["system"], profile, expected_readers, readers
        )
    for receipt, row in zip(readers, expected_readers, strict=True):
        user = json.loads(receipt["request"]["messages"][1]["content"])
        if (
            user["question"] != Sample(**row["sample"]).reader_question
            or row["prediction"]["sample_id"] != row["sample"]["id"]
            or receipt["response"]["choices"][0]["message"]["content"]
            != row["prediction"]["answer"]
            or (manifest["system"] == "no_memory" and user["evidence"] != "")
        ):
            raise ValueError("provider reader request/prediction differs")
    prompt = load_longmemeval_prompt(Path(judge_source)) if judge_source else None
    for receipt, row in zip(judge_calls, rows, strict=True):
        if receipt.get("status") != 200:
            raise ValueError("incomplete common judge generation")
        evidence = validate_response(
            judge_profile,
            receipt["request"],
            receipt["response"],
            output=judge_profile.output_budget("rubric"),
            counts=judge_counts,
        )
        if any(receipt.get(k) != v for k, v in evidence.items()):
            raise ValueError("common judge returned provider receipt differs")
        verdict = receipt["response"]["choices"][0]["message"]["content"].strip().lower()
        if verdict not in ("yes", "no") or verdict != row["judge_verdict"]:
            raise ValueError("common judge verdict differs")
        if prompt:
            sample = row["sample"]
            task = (
                "temporal-reasoning"
                if sample["category_name"] == "abstention"
                else sample["category_name"]
            )
            rendered = prompt(
                task,
                sample["question"],
                sample["gold"],
                row["prediction"]["answer"],
                abstention=sample["id"].endswith("_abs"),
            )
            if receipt["request"]["messages"] != [{"role": "user", "content": rendered}]:
                raise ValueError("common judge prompt differs from exact pinned native rubric")
    generated = []
    if manifest["system"] == "gluomem":
        from gluomem.write.events import SourceEvent

        from gluomem_bench.bench.beam_ingest import _expected_receipt
        from gluomem_bench.bench.rack_schema import (
            validate_native_projection,
            validate_production_row,
        )

        if not {"production-raw.jsonl", "source-events.json"} <= inventory.keys():
            raise ValueError("native production source/trace inventory missing")
        generated = read_lines(out / "production-raw.jsonl")
        if [r.get("ordinal") for r in generated] != list(range(1, len(generated) + 1)):
            raise ValueError("native production generation ordinals differ")
        for row in generated:
            validate_production_row(row, profile, cfg, protocol, expected)
            validate_native_projection(row["response"], row["native_response"])
            if row.get("error") or row.get("finish_reasons") != ["stop"]:
                raise ValueError("failed native production generation")
            raw_response = row["native_response"]["canonical"]["provider_data"]
            request = {
                "model": profile.model,
                "messages": row["messages"],
                "temperature": 0,
                "max_tokens": profile.output_budget(row["task"]),
                **generation_options(profile),
            }
            evidence = validate_response(
                profile,
                request,
                raw_response,
                output=profile.output_budget(row["task"]),
                counts=counts,
            )
            if any(row.get(k) != v for k, v in evidence.items()):
                raise ValueError("native production returned provider evidence differs")
        sentinels = [
            name
            for name in inventory
            if name.startswith("stores/") and name.endswith(".ingest.json")
        ]
        stores = [
            name for name in inventory if name.startswith("stores/") and name.endswith(".mind")
        ]
        if len(sentinels) != 1 or len(stores) != 1:
            raise ValueError("native committed input store/receipt inventory differs")
        sentinel = read_json(out / sentinels[0])
        source = [SourceEvent.model_validate(e) for e in read_json(out / "source-events.json")]
        if (
            sentinel.get("complete") is not True
            or sentinel.get("events") != len(source)
            or sentinel.get("dataset_sha256") != protocol["dataset_sha256"]
            or sentinel.get("config_sha256")
            != hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
            or any(
                e.capture_policy != "attributed-interaction" or e.created_at != e.event_time
                for e in source
            )
            or [
                {k: v for k, v in r.items() if k != "row_id"}
                for r in sentinel["committed_receipts"]
            ]
            != [_expected_receipt(e) for e in source]
        ):
            raise ValueError("native committed whole-source/clock coverage differs")
    elif "production-raw.jsonl" in inventory:
        raise ValueError("unexpected production trace in control")
    if profile.token_count_policy == "served-tokenizer":
        admitted = Counter(
            key(r["request"]["messages"]) for r in primary if r.get("url", "").endswith("/tokenize")
        )
        consumed = Counter(key(r["request"]["messages"]) for r in direct)
        consumed.update(key(r["request"]["messages"]) for r in mem0_direct)
        consumed.update(key(r["messages"]) for r in generated)
        if consumed != admitted:
            raise ValueError("all native/direct served-tokenizer coverage differs")
    native_calls = 0
    if manifest["system"] == "cognee":
        native_calls = validate_cognee_history(out, inventory, profile, rows, readers)
    calls = len(direct) + len(judge_calls) + len(generated) + mem0_calls + native_calls
    if read_json(out / "usage.json")["generation_calls"] != calls:
        raise ValueError("all-attempt provider accounting differs")
    return {
        "questions": len(rows),
        "generation_calls": calls,
        "frozen_plan_bound": expected is not None,
        "quality_admitted": True,
    }


def source_projection(conv):
    """Complete exact shared-source projection; no gold or evidence annotations."""
    return {
        "conv_id": conv.conv_id,
        "sessions": [
            {
                "index": s.index,
                "date": s.date_time.isoformat(),
                "turns": [{"id": t.dia_id, "role": t.speaker, "text": t.text} for t in s.turns],
            }
            for s in conv.sessions
        ],
    }


def validate_pool_source(out, system, conv):
    """Pure source reconstruction for controls; native committed rows use readonly SQL."""
    if system == "full_context":
        from gluomem_bench.bench.baselines import transcript

        if any(
            r["original_text"] != transcript(conv)
            or r["kept_text"] != transcript(conv)
            or r["exempt"] is not True
            for r in read_json(out / "reader-evidence.json")
        ):
            raise ValueError("root full-context complete untruncated source differs")
    elif system in {"bm25", "dense", "hybrid"}:
        expected = []
        for session in conv.sessions:
            for ti, turn in enumerate(session.turns):
                for offset in range(0, len(turn.text), 4096):
                    expected.append(
                        {
                            "id": f"{conv.conv_id}/s{session.index}/t{ti}/c{offset // 4096}",
                            "text": f"[{session.date_time.isoformat()}] {turn.speaker}: "
                            + turn.text[offset : offset + 4096],
                            "provenance": {
                                "source_turn_id": turn.dia_id
                                or f"{conv.conv_id}/session/{session.index}/{ti}",
                                "session_index": session.index,
                                "role": turn.speaker,
                                "time": session.date_time.isoformat(),
                                "offset": offset,
                            },
                            "score": 0.0,
                        }
                    )
        if read_json(out / "raw-inputs.json") != expected:
            raise ValueError("root complete raw control source differs")
        for row in read_json(out / "predictions.json"):
            hits = row["prediction"]["retrieved"]
            if any(
                not any(
                    hit["text"] == e["text"] and hit["dia_id"] == e["provenance"]["source_turn_id"]
                    for e in expected
                )
                for hit in hits
            ):
                raise ValueError("root retrieved control evidence lacks exact source binding")
        for packed, row in zip(
            read_json(out / "reader-evidence.json"),
            read_json(out / "predictions.json"),
            strict=True,
        ):
            if packed["original_text"] != "\n".join(
                hit["text"] for hit in row["prediction"]["retrieved"]
            ):
                raise ValueError("root control reader differs from retrieved raw evidence")


def pool_bundle_calls(out, profile, predictions, protocol, *, judge_source):
    """Reconstruct direct invocation messages from admitted prediction/packing/rubric."""
    from gluomem_bench.bench.suite_campaign import reader_messages

    primary, judges = read_lines(out / "calls.jsonl"), read_lines(out / "calls-judge.jsonl")
    attempts = [r for r in primary + judges if "pool_attempt" in r]
    attempts.sort(key=lambda r: r["pool_attempt"]["ordinal"])
    native = (
        read_lines(out / "production-raw.jsonl") if (out / "production-raw.jsonl").exists() else []
    )
    grouped = {}
    for row in attempts:
        grouped.setdefault(row["logical_ordinal"], []).append(row)
    readers = [r[-1] for r in grouped.values() if r[0]["task"] == "answer"]
    rubric = [r[-1] for r in grouped.values() if r[0]["task"] == "rubric"]
    expected_readers = [r for r in predictions if not r["prediction"].get("abstained")]
    packing = (
        read_json(out / "reader-evidence.json") if "evidence_token_budget" in protocol else None
    )
    if len(readers) != len(expected_readers) or len(rubric) != len(predictions):
        raise ValueError("pool reader/common judge invocation coverage differs")
    if packing is not None:
        validate_reader_packing(
            out,
            read_json(
                out / "manifest.pending.json"
                if (out / "manifest.pending.json").exists()
                else out / "manifest.json"
            )["artifact_sha256"],
            protocol,
            protocol["system"],
            profile,
            expected_readers,
            readers,
        )
    expected_requests = {}
    for index, (receipt, row) in enumerate(zip(readers, expected_readers, strict=True)):
        evidence = (
            packing[index]["kept_text"]
            if packing is not None
            else (
                ""
                if protocol["system"] == "no_memory"
                else "\n".join(r["text"] for r in row["prediction"]["retrieved"])
            )
        )
        expected_requests[receipt["logical_ordinal"]] = {
            "model": profile.model,
            "temperature": 0,
            "max_tokens": profile.output_budget("answer"),
            "reasoning": {"enabled": False},
            "messages": reader_messages(Sample(**row["sample"]).reader_question, evidence),
        }
        if (
            receipt["response"]["choices"][0]["message"]["content"] != row["prediction"]["answer"]
            or row["prediction"]["sample_id"] != row["sample"]["id"]
            or protocol["system"] == "no_memory"
            and evidence != ""
        ):
            raise ValueError("pool reader request/prediction differs")
    if judge_source is None:
        raise ValueError("pool replay requires the independently pinned native scorer")
    prompt = load_longmemeval_prompt(Path(judge_source))
    for receipt, row in zip(rubric, predictions, strict=True):
        sample = row["sample"]
        task = (
            "temporal-reasoning"
            if sample["category_name"] == "abstention"
            else sample["category_name"]
        )
        rendered = prompt(
            task,
            sample["question"],
            sample["gold"],
            row["prediction"]["answer"],
            abstention=sample["id"].endswith("_abs"),
        )
        expected_requests[receipt["logical_ordinal"]] = {
            "model": profile.judge_profile.model,
            "temperature": 0,
            "max_tokens": profile.judge_profile.output_budget("rubric"),
            "reasoning": {"enabled": False},
            "messages": [{"role": "user", "content": rendered}],
        }
        verdict = receipt["response"]["choices"][0]["message"]["content"].strip().lower()
        if verdict not in {"yes", "no"} or verdict != row["judge_verdict"]:
            raise ValueError("pool common judge verdict differs")
    return native, attempts, grouped, expected_requests


def validate_pool_bundle(out, manifest, *, expected, judge_source):
    from gluomem_bench.bench.rack_pool_replay import validate_pool_call
    from gluomem_bench.bench.rack_schema import validate_production_row
    from gluomem_bench.bench.suite_evidence import REQUIRED

    out = Path(out)
    inventory = manifest["artifact_sha256"]
    if (
        not REQUIRED.union({"calls-judge.jsonl", "source-projection.json"}) <= inventory.keys()
        or (out / "failure.json").exists()
    ):
        raise ValueError("pool workload incomplete or required inventory missing")
    verify_artifact_hashes(out, inventory, canonical_names=True)
    protocol, profile = (
        read_json(out / "protocol.json"),
        RackProfile.model_validate(read_json(out / "profile.json")),
    )
    if (
        profile.profile_version != POOL_VERSION
        or profile.judge_profile is None
        or manifest["dataset"] != "longmemeval_s"
        or any(
            protocol.get(k) != manifest[k] for k in ("protocol", "dataset", "system", "sample_ids")
        )
        or (
            expected is not None
            and any(expected.get(k) != manifest[k] for k in ("dataset", "system", "sample_ids"))
        )
    ):
        raise ValueError("pool workload/profile differs from frozen plan")
    validate_history_scope(profile, manifest["system"])
    cfg = read_json(out / "effective-config.json")
    validate_config(cfg, profile)
    rows = read_json(out / "predictions.json")
    if (
        not rows
        or [r["sample"]["id"] for r in rows] != manifest["sample_ids"]
        or len(set(manifest["sample_ids"])) != len(rows)
        or any(r["prediction"].get("failure") for r in rows)
        or hashlib.sha256(
            json.dumps([r["sample"] for r in rows], sort_keys=True).encode()
        ).hexdigest()
        != protocol["samples_sha256"]
        or hashlib.sha256(profile.model_dump_json().encode()).hexdigest()
        != protocol["profile_sha256"]
    ):
        raise ValueError("pool prediction/sample/profile binding differs")
    validate_discovery(read_lines(out / "calls.jsonl"), profile)
    validate_discovery(read_lines(out / "calls-judge.jsonl"), profile.judge_profile)
    native, attempts, grouped, requests = pool_bundle_calls(
        out, profile, rows, protocol, judge_source=judge_source
    )
    by_logical = {r["logical_ordinal"]: r for r in native}
    if any(row.get("error") or row.get("finish_reasons") != ["stop"] for row in native):
        raise ValueError("failed native pool generation cannot be published as complete")
    for logical, records in grouped.items():
        row = by_logical.get(
            logical,
            {
                **records[0],
                "pool_attempts": [r["pool_attempt"] for r in records],
                "ordinal": records[-1]["pool_attempt"]["ordinal"],
            },
        )
        if logical in by_logical:
            if row["pool_attempts"] != [r["pool_attempt"] for r in records]:
                raise ValueError("pool native physical attempts differ")
            validate_production_row(row, profile, cfg, protocol, expected)
        elif logical not in requests:
            raise ValueError("unaccounted pool native generation")
        validate_pool_call(row, profile, expected_request=requests.get(logical))
    if set(by_logical) - grouped.keys() or read_json(out / "usage.json")["generation_calls"] != len(
        attempts
    ):
        raise ValueError("pool all-attempt accounting differs")
    if manifest["system"] == "gluomem":
        from gluomem.write.events import SourceEvent

        from gluomem_bench.bench.beam_ingest import _read_rows, _validate_rows

        stores = [n for n in inventory if n.startswith("stores/") and n.endswith(".mind")]
        sentinels = [n for n in inventory if n.startswith("stores/") and n.endswith(".ingest.json")]
        if (
            len(stores) != 1
            or len(sentinels) != 1
            or not {"production-raw.jsonl", "source-events.json"} <= inventory.keys()
        ):
            raise ValueError("pool native source/store inventory missing")
        source = [SourceEvent.model_validate(e) for e in read_json(out / "source-events.json")]
        committed, _ = _validate_rows(_read_rows(out / stores[0]), source)
        sentinel = read_json(out / sentinels[0])
        if (
            sentinel.get("complete") is not True
            or sentinel.get("events") != len(source)
            or sentinel.get("committed_receipts") != committed
            or sentinel.get("dataset_sha256") != protocol["dataset_sha256"]
            or sentinel.get("config_sha256")
            != hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
            or any(
                e.capture_policy != "attributed-interaction" or e.created_at != e.event_time
                for e in source
            )
        ):
            raise ValueError("pool native committed source/clock coverage differs")
    elif native:
        raise ValueError("unexpected pool native trace in control")
    return {
        "questions": len(rows),
        "generation_calls": len(attempts),
        "frozen_plan_bound": expected is not None,
        "quality_admitted": True,
    }


def validate_reader_packing(out, inventory, protocol, system, profile, predictions, readers):
    from gluomem_bench.bench.suite_campaign import reader_messages

    if not {"reader-evidence.json", "evidence-tokenizer.jsonl"} <= inventory.keys():
        raise ValueError("reader evidence packing inventory missing")
    receipts = read_lines(out / "evidence-tokenizer.jsonl")
    counts = token_counts(receipts, profile)
    if profile.token_count_policy == "utf8-conservative-v1" and receipts:
        raise ValueError("conservative DeepSeek count must not invent tokenizer receipts")

    def count(question, text):
        messages = reader_messages(question, text)
        value = (
            estimated_prompt_tokens(messages)
            if profile.token_count_policy == "utf8-conservative-v1"
            else counts.get(key(messages))
        )
        if type(value) is not int or value < 0:
            raise ValueError("reader evidence token admission missing")
        return value

    packing = read_json(out / "reader-evidence.json")
    if len(packing) != len(predictions):
        raise ValueError("reader evidence packing coverage differs")
    for item, prediction, wire in zip(packing, predictions, readers, strict=True):
        actual = json.loads(wire["request"]["messages"][1]["content"])
        if (
            item["sample_id"] != prediction["sample"]["id"]
            or item["question"] != actual["question"]
            or item["kept_text"] != actual["evidence"]
            or item["budget"] != protocol["evidence_token_budget"]
            or item["exempt"] != (system == "full_context")
        ):
            raise ValueError("reader evidence differs from dispatched messages")
        for prefix in ("original", "kept"):
            text = item[prefix + "_text"]
            if (
                hashlib.sha256(text.encode()).hexdigest() != item[prefix + "_sha256"]
                or len(text.splitlines()) != item[prefix + "_lines"]
                or count(item["question"], text) != item[prefix + "_tokens"]
            ):
                raise ValueError("reader evidence text/hash/token count differs")
        kept = item["original_text"]
        if system not in {"no_memory", "full_context"} and item["original_tokens"] > item["budget"]:
            kept = ""
            if count(item["question"], kept) > item["budget"]:
                raise ValueError("reader evidence ceiling below prompt overhead")
            for line in item["original_text"].splitlines(keepends=True):
                candidate = kept + line
                if count(item["question"], candidate) > item["budget"]:
                    break
                kept = candidate
        if item["kept_text"] != kept:
            raise ValueError("reader evidence is not the exact admitted whole-line ranked prefix")


def validate_cognee_history(out, inventory, profile, predictions, readers):
    """Read-only native SQLite/input+wire+graph context replay, without SDK workers."""
    from gluomem_bench.bench.cognee_campaign import verify_native_coverage
    from gluomem_bench.bench.cognee_graph import GRAPH_CONDITION, NATIVE_SOURCE_PINS
    from gluomem_bench.bench.comparison_backends import COGNEE_WHEEL_SHA256, Evidence

    native = out / "graph"
    required = {"graph/native-ledger.json", "graph/generation.jsonl", "graph/graph-contexts.json"}
    if not required <= inventory.keys():
        raise ValueError("native Cognee graph ledger/context inventory missing")
    ledger = read_json(native / "native-ledger.json")
    pins = {
        "version": "1.6.5",
        "wheel_sha256": COGNEE_WHEEL_SHA256,
        "source_sha256": NATIVE_SOURCE_PINS,
    }
    if (
        ledger.get("status") != "closed_verified"
        or ledger.get("condition") != GRAPH_CONDITION
        or ledger.get("native_worker_closed") is not True
        or ledger.get("proxy_closed") is not True
        or ledger.get("source_pins") != pins
    ):
        raise ValueError("closed native Cognee/source-pin evidence differs")
    files = {
        p.relative_to(native).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in native.rglob("*")
        if p.is_file() and p.name != "native-ledger.json"
    }
    if ledger["artifact_sha256"] != files:
        raise ValueError("closed native Cognee artifact inventory differs")
    documents = [Evidence(**r) for r in ledger["source_inputs"]]
    coverage = verify_native_coverage(
        native / "native", dataset=ledger["dataset"], expected=documents, quiescent=True
    )
    coverage = {
        k: v
        for k, v in coverage.items()
        if k not in {"condition", "graph_retrieval_qualified", "graph_retrieval_gap"}
    }
    if coverage != ledger["verification"]["coverage"]:
        raise ValueError("closed native processed source readback differs")
    generation = verify_cognee_generations(
        native / "generation.jsonl", profile=profile, expected_calls=ledger["generation_calls"]
    )
    if generation != ledger["generation"]:
        raise ValueError("Cognee all-attempt raw generation ledger differs")
    contexts = read_json(native / "graph-contexts.json")
    if len(contexts) != len(predictions) or ledger["graph_context_receipts"] != len(contexts):
        raise ValueError("Cognee graph context coverage differs")
    packing = (
        read_json(out / "reader-evidence.json") if (out / "reader-evidence.json").exists() else None
    )
    for index, (context, row, reader) in enumerate(
        zip(contexts, predictions, readers, strict=True)
    ):
        rendered = "\n".join(h["text"] for h in context["hits"])
        actual = json.loads(reader["request"]["messages"][1]["content"])
        if (
            context.get("sample_id") != row["sample"]["id"]
            or context.get("query") != row["sample"]["question"]
            or context.get("native_final_completion_used") is not False
            or [r["text"] for r in row["prediction"]["retrieved"]]
            != [h["text"] for h in context["hits"]]
            or (
                packing[index]["original_text"] != rendered
                if packing
                else actual["evidence"] != rendered
            )
        ):
            raise ValueError("Cognee graph context differs from actual common reader evidence")
        for hit in context["hits"]:
            p = hit["provenance"]
            if (
                p.get("condition") != GRAPH_CONDITION
                or p.get("dataset_name") != ledger["dataset"]
                or str(p.get("dataset_id")).replace("-", "")
                != str(coverage["dataset_id"]).replace("-", "")
                or not any(ref.get("kind") == "graph_edge" for ref in p["native_evidence"])
                or p.get("completion") is not None
            ):
                raise ValueError("native graph context lacks scoped graph-edge lineage")
    return generation["calls"]


def verify_cognee_generations(receipt, *, profile, expected_calls, starting_calls=0):
    """V8 bridge replay binds original native body, provider controls and all attempts."""
    import base64

    from gluomem_bench.bench.rack_provider import refuse_unqualified_endpoint_comparator

    refuse_unqualified_endpoint_comparator(profile)

    from jsonschema import Draft202012Validator
    from referencing import Registry

    raw = Path(receipt).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line]
    if type(expected_calls) is not int or expected_calls < 1:
        raise ValueError("positive actual native generation count required")

    def decoded(row, name="response_b64"):
        return json.loads(base64.b64decode(row[name], validate=True))

    path = (
        "/models/" + profile.model + "/endpoints"
        if profile.provider == "openrouter-baidu-fp8"
        else "/models"
    )
    metadata = [r for r in rows if r.get("kind") == path]
    if len(metadata) != 1 or metadata[0].get("http_status") != 200:
        raise ValueError("exact native provider metadata missing")
    validate_discovery(
        [
            {
                "method": "GET",
                "url": profile.base_url + path,
                "status": 200,
                "response": decoded(metadata[0]),
            }
        ],
        profile,
    )
    calls = [r for r in rows if r.get("kind") == "generation"]
    if len(calls) != expected_calls or sorted(r.get("calls", -1) for r in calls) != list(
        range(starting_calls + 1, starting_calls + expected_calls + 1)
    ):
        raise ValueError("native all-attempt generation coverage differs")
    if len({r["request_id"] for r in calls}) != len(calls):
        raise ValueError("duplicate native request identity")
    prompt_total = completion_total = 0
    consumed = {id(metadata[0])}
    for call in calls:
        stream = [
            r for r in rows if r.get("request_id") == call["request_id"] and r is not metadata[0]
        ]
        native = decoded(call, "native_request_b64")
        forwarded = {**native, **generation_options(profile)}
        if call.get("status") != "complete" or call.get("forwarded_request") != forwarded:
            raise ValueError("native prompt/schema/provider controls altered")
        if profile.provider == "openrouter-baidu-fp8" and "chat_template_kwargs" in native:
            raise ValueError("Qwen native controls forwarded to DeepSeek")
        output = native.get("max_tokens", native.get("max_completion_tokens"))
        if (
            type(output) is not int
            or not 0 < output <= profile.output_budget("cognify")
            or native.get("stream", False) is not False
            or native.get("n", 1) != 1
            or native.get("top_p", 1) != 1
        ):
            raise ValueError("native generation output/settings differ")
        tokens = [r for r in stream if r.get("kind") == "/tokenize"]
        conservative = profile.token_count_policy == "utf8-conservative-v1"
        fmt = native.get("response_format")
        reserve = profile.schema_reserve_tokens
        if conservative:
            if tokens:
                raise ValueError("DeepSeek native path invented a tokenizer")
            count = estimated_prompt_tokens(native["messages"])
            if fmt:
                reserve = max(
                    reserve, len(json.dumps(fmt, ensure_ascii=False).encode("utf-8")) + 512
                )
        else:
            if len(tokens) != (2 if fmt is not None else 1):
                raise ValueError("native served-tokenizer/schema coverage differs")
            expected_prompt = {
                "model": profile.model,
                "messages": native["messages"],
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if tokens[0].get("http_status") != 200 or tokens[0]["request"] != expected_prompt:
                raise ValueError("native served-tokenizer prompt differs")
            count = decoded(tokens[0])["count"]
            if fmt:
                if tokens[1].get("http_status") != 200 or tokens[1]["request"] != {
                    "model": profile.model,
                    "prompt": json.dumps(fmt, ensure_ascii=False),
                    "add_special_tokens": False,
                }:
                    raise ValueError("native schema tokenizer differs")
                schema_count = decoded(tokens[1])["count"]
                if type(schema_count) is not int or schema_count < 0:
                    raise ValueError("native schema count invalid")
                reserve = max(reserve, schema_count)
        if (
            type(count) is not int
            or count < 0
            or call.get("prompt_tokens") != count
            or call.get("reserved_tokens") != count + output + reserve
            or call["reserved_tokens"] > profile.context_window
            or call.get("token_count_policy") != profile.token_count_policy
        ):
            raise ValueError("native context/count policy admission differs")
        intents = [r for r in stream if r.get("kind") == "dispatch-intent"]
        wires = [r for r in stream if r.get("kind") == "/chat/completions"]
        if len(intents) != 1 or len(wires) != 1:
            raise ValueError("native physical dispatch coverage differs")
        intent, wire = intents[0], wires[0]
        if (
            intent.get("calls") != call["calls"]
            or intent.get("native_request_b64") != call["native_request_b64"]
            or intent.get("forwarded_request") != forwarded
            or intent.get("reserved_tokens") != call["reserved_tokens"]
            or wire.get("http_status") != 200
            or wire.get("request") != forwarded
            or wire.get("response_b64") != call.get("response_b64")
        ):
            raise ValueError("native dispatch-intent/raw response binding differs")
        counts = {key(native["messages"]): count}
        response = decoded(call)
        evidence = validate_response(profile, forwarded, response, output=output, counts=counts)
        if any(call.get(k) != v for k, v in evidence.items()):
            raise ValueError("native returned provider evidence differs")
        if fmt:
            schema = (
                fmt["json_schema"]["schema"] if fmt["type"] == "json_schema" else {"type": "object"}
            )
            Draft202012Validator(schema, registry=Registry()).validate(
                json.loads(response["choices"][0]["message"]["content"])
            )
        if [r["kind"] for r in stream] != ["/tokenize"] * len(tokens) + [
            "dispatch-intent",
            "/chat/completions",
            "generation",
        ]:
            raise ValueError("unexpected or missing native generation evidence")
        consumed.update(id(r) for r in stream)
        prompt_total += response["usage"]["prompt_tokens"]
        completion_total += response["usage"]["completion_tokens"]
    if len(consumed) != len(rows):
        raise ValueError("unaccounted native wire rows")
    return {
        "calls": expected_calls,
        "prompt_tokens": prompt_total,
        "completion_tokens": completion_total,
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
    }
