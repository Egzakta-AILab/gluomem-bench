"""Offline admission of complete pilot bundles, separately from score aggregation.

This verifier never opens datasets or SQLite, constructs clients, or calls models.
Pass the externally frozen job record to bind identity beyond run-owned metadata.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from gluomem_bench.bench.rack import RackProfile
from gluomem_bench.bench.types import Sample
from gluomem_bench.bench.upstream_scoring import load_longmemeval_prompt

REQUIRED = frozenset({"protocol.json", "effective-config.json", "profile.json",
                      "predictions.json", "usage.json", "calls.jsonl", "scores.json"})


def _json(path):
    return json.loads(path.read_bytes())


def _lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def _key(messages):
    return json.dumps(messages, ensure_ascii=False, sort_keys=True)


def validate_mem0_evidence(out, inventory, profile, protocol, direct, *, expected=None):
    """Bind native extraction to exact HTTP receipts, inputs and stored ADD records.

    Called after outer inventory byte verification; imports no Mem0 or dataset.
    The first N direct generations are native ingestion, not answering readers.
    """
    required = {"mem0-config.json", "mem0-source.json", "mem0-source-inputs.json",
                "mem0-native.jsonl", "mem0-input-receipts.json", "mem0-ingestion.json",
                "mem0-memories.json", "mem0-history.json", "mem0-diagnostics.json"}
    if not required <= inventory.keys():
        raise ValueError("required Mem0 evidence missing")
    if _json(out / "mem0-diagnostics.json"):
        raise ValueError("native Mem0 runtime feature warning/error in complete workload")
    inputs = _json(out / "mem0-source-inputs.json")
    def digest(value):
        return hashlib.sha256(_key(value).encode()).hexdigest()
    if not inputs or digest(inputs) != protocol.get("mem0_source_inputs_sha256"):
        raise ValueError("Mem0 source inputs differ from admitted history")
    source = _json(out / "mem0-source.json")
    if source["versions"]["mem0ai"] != "2.2.1":
        raise ValueError("unadmitted Mem0 version")
    if expected is not None:
        bindings = {"mem0_source_inputs_sha256": digest(inputs),
                    "mem0_extractor_system_sha256": hashlib.sha256(
                        source["extractor_system"].encode()).hexdigest(),
                    "profile_sha256": protocol.get("profile_sha256"),
                    "dataset_sha256": protocol.get("dataset_sha256")}
        if any(expected.get(k) != v for k, v in bindings.items()):
            raise ValueError("Mem0 source/profile binding differs from frozen plan")
    config = _json(out / "mem0-config.json")
    llm = config["llm"]["config"]
    if (config["llm"]["provider"] != "openai" or llm["model"] != profile.model
            or llm.get("openai_base_url") != profile.base_url or llm["api_key"] != "<redacted>"
            or llm["temperature"] != 0 or llm["top_p"] != 1.0
            or llm["max_tokens"] != profile.output_budget("event_claims")):
        raise ValueError("Mem0 native configuration differs from frozen local model")
    native = _lines(out / "mem0-native.jsonl")
    receipts = _json(out / "mem0-input-receipts.json")
    ingestion = _json(out / "mem0-ingestion.json")
    memories = _json(out / "mem0-memories.json")
    if (len(native) != len(inputs) or len(receipts) != len(inputs)
            or len(direct) < len(native) or ingestion.get("complete") is not True
            or ingestion["sessions"] != receipts or ingestion["memories"] != len(memories)
            or ingestion["source_inputs_sha256"] != digest(inputs)):
        raise ValueError("Mem0 native ingestion coverage differs")
    allowed = Counter()
    committed = []
    for i, (event, trace, receipt, actual) in enumerate(
            zip(inputs, native, receipts, direct[:len(native)], strict=True), 1):
        if trace.get("task") != "event_claims" or trace.get("ordinal") != i or trace.get("error"):
            raise ValueError("invalid Mem0 extraction task/ordinal")
        request = trace["request"]
        wanted = dict(request, chat_template_kwargs={"enable_thinking": False})
        if (actual.get("status") != 200 or actual["request"] != wanted
                or actual["response"] != trace["response"]
                or request["model"] != profile.model or actual["response"]["model"] != profile.model
                or request["temperature"] != 0 or request["top_p"] != 1.0
                or request["response_format"] != {"type": "json_object"}
                or request["max_tokens"] != profile.output_budget("event_claims")
                or request["messages"][0] != {"role": "system",
                                              "content": source["extractor_system"]}
                or len(actual["response"]["choices"]) != 1
                or actual["response"]["choices"][0]["finish_reason"] != "stop"):
            raise ValueError("Mem0 native request differs from actual guarded generation")
        messages = event["messages"]
        if any(m["role"] not in ("user", "assistant") or not isinstance(m["content"], str)
               for m in messages):
            raise ValueError("Mem0 source roles differ from admitted chat protocol")
        parsed = "".join(f"{m['role']}: {m['content']}\n" for m in messages)
        segment = "## New Messages\n" + parsed + "\n\n## Observation Date\n"
        if segment not in request["messages"][1]["content"]:
            raise ValueError("Mem0 extraction lacks complete submitted source")
        raw = actual["response"]["choices"][0]["message"]["content"]
        from gluomem_bench.bench.mem0 import inspect_native_extraction

        extraction = inspect_native_extraction(raw)
        if (extraction["status"] not in ("valid", "valid_empty")
                or receipt["extraction"] != extraction
                or receipt["ordinal"] != i or receipt["session_index"] != event["session_index"]
                or receipt["session_date"] != event["session_date"]
                or receipt["submitted_turns"] != len(messages)
                or receipt["messages_sha256"] != digest(messages)
                or receipt["native_request_sha256"] != digest(request)
                or receipt["native_response_sha256"] != digest(actual["response"])):
            raise ValueError("Mem0 extraction/receipt binding differs")
        allowed.update((m["role"], m["content"]) for m in messages)
        if (len(receipt["rolling_buffer"]) > 10
                or Counter(tuple(m) for m in receipt["rolling_buffer"]) - allowed):
            raise ValueError("Mem0 rolling buffer contains unsubmitted input")
        committed.extend(receipt["added"])
    if ingestion["submitted_turns"] != sum(len(e["messages"]) for e in inputs):
        raise ValueError("Mem0 submitted turn total differs")
    def key(row):
        return row["id"], row["memory"], row["user_id"]
    if (Counter(map(key, committed)) != Counter(map(key, memories))
            or len({m["id"] for m in memories}) != len(memories)):
        raise ValueError("Mem0 committed ADD inventory differs from final memories")
    history = _json(out / "mem0-history.json")
    expected_history = Counter((m["id"], m["memory"]) for m in memories)
    if any(r[2] != "ADD" for r in history) or Counter(
            (r[0], r[1]) for r in history) != expected_history:
        raise ValueError("Mem0 history differs from final inventory")
    return len(native)


def validate_bundle(out: Path, manifest: dict, *, expected=None, require_real=False,
                    judge_source: Path | None = None) -> dict:
    """Reject missing evidence, failed predictions, identity drift and lost calls."""
    out = Path(out).resolve()
    inventory = manifest.get("artifact_sha256", {})
    if not isinstance(inventory, dict) or not REQUIRED <= inventory.keys():
        raise ValueError("required evidence missing from artifact hash inventory")
    if expected is not None and any(manifest.get(k) != expected.get(k)
                                    for k in ("dataset", "system", "sample_ids")):
        raise ValueError("workload identity differs from frozen plan")
    if (out / "failure.json").exists():
        raise ValueError("complete workload contains failure receipt")
    for name, digest in inventory.items():
        target = (out / name).resolve()
        if (not target.is_relative_to(out) or target.relative_to(out).as_posix() != name
                or not target.is_file()
                or hashlib.sha256(target.read_bytes()).hexdigest() != digest):
            raise ValueError("artifact byte hash mismatch or noncanonical path")
    protocol = _json(out / "protocol.json")
    for key in ("protocol", "dataset", "system", "sample_ids"):
        if protocol.get(key) != manifest.get(key):
            raise ValueError("protocol and manifest identity differ")
    ids = manifest.get("sample_ids", [])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("distinct expected sample IDs required")
    rows = _json(out / "predictions.json")
    if [row["sample"]["id"] for row in rows] != ids:
        raise ValueError("prediction coverage differs from frozen identity")
    if any(row["prediction"].get("failure") is not None for row in rows):
        raise ValueError("failed prediction cannot be finalized")
    if any(row["prediction"]["sample_id"] != row["sample"]["id"] for row in rows):
        raise ValueError("prediction/sample identity differs")
    sample_material = json.dumps([r["sample"] for r in rows], sort_keys=True).encode()
    sample_hash = hashlib.sha256(sample_material).hexdigest()
    if sample_hash != protocol.get("samples_sha256"):
        raise ValueError("saved sample binding differs from admitted samples")
    profile = RackProfile.model_validate(_json(out / "profile.json"))
    profile_hash = hashlib.sha256(profile.model_dump_json().encode()).hexdigest()
    if profile_hash != protocol.get("profile_sha256"):
        raise ValueError("profile differs from admitted configuration")
    cfg = _json(out / "effective-config.json")
    if require_real:
        if (cfg["components"]["allow_fake"] or cfg["dspy"]["default_model"] != "rack-qwen"
                or any(t["model"] != "rack-qwen" for t in cfg["dspy"]["tasks"].values())
                or cfg["models"]["rack-qwen"]["provider_model"] != "openai/" + profile.model
                or cfg["models"]["rack-qwen"]["api_base"] != profile.base_url):
            raise ValueError("nonisolated or fake model configuration")
    calls = _lines(out / "calls.jsonl")
    direct = [r for r in calls if r.get("method") == "POST"
              and r.get("url", "").endswith("/chat/completions")]
    tokenizers = [r for r in calls if r.get("method") == "POST"
                  and r.get("url", "").endswith("/tokenize")]
    token_inputs = Counter(_key(r["request"]["messages"]) for r in tokenizers)
    consumed = Counter()
    readers = []
    judges = []
    mem0_calls = 0
    if manifest["system"] == "mem0":
        mem0_calls = validate_mem0_evidence(out, inventory, profile, protocol, direct,
                                             expected=expected)
        for row in direct[:mem0_calls]:
            consumed[_key(row["request"]["messages"])] += 1
        direct = direct[mem0_calls:]
    for row in direct:
        request, response = row["request"], row.get("response", {})
        if (row.get("status") != 200 or request.get("model") != profile.model
                or response.get("model") != profile.model or request.get("temperature") != 0
                or request.get("chat_template_kwargs", {}).get("enable_thinking") is not False
                or len(response.get("choices", [])) != 1
                or response["choices"][0].get("finish_reason") != "stop"):
            raise ValueError("invalid completed generation receipt")
        messages = request["messages"]
        consumed[_key(messages)] += 1
        if messages[0]["role"] == "system":
            if messages[0]["content"] != protocol["reader_instruction"]:
                raise ValueError("reader instruction differs from frozen protocol")
            if request["max_tokens"] != profile.output_budget("answer"):
                raise ValueError("reader allowance differs")
            readers.append(row)
        else:
            if request["max_tokens"] != profile.output_budget("rubric"):
                raise ValueError("judge allowance differs")
            judges.append(row)
    expected_readers = [r for r in rows if not r["prediction"].get("abstained", False)]
    if len(readers) != len(expected_readers):
        raise ValueError("reader receipt coverage differs from predictions")
    for receipt, row in zip(readers, expected_readers, strict=True):
        user = json.loads(receipt["request"]["messages"][1]["content"])
        actual_answer = receipt["response"]["choices"][0]["message"]["content"]
        if (user["question"] != Sample(**row["sample"]).reader_question
                or actual_answer != row["prediction"]["answer"]):
            raise ValueError("reader prediction differs from actual request/response")
        if manifest["system"] == "no_memory" and user["evidence"] != "":
            raise ValueError("no-memory condition contains evidence")
    is_mab = manifest["dataset"] == "memoryagentbench_conflict"
    quality_admitted = True
    if is_mab:
        from gluomem_bench.bench.adapters.memoryagentbench import CONFLICT_INSTRUCTIONS

        quality_admitted = all(r["sample"].get("task_instructions") == CONFLICT_INSTRUCTIONS
                               for r in rows)
        if manifest["protocol"].endswith("@v2") and not quality_admitted:
            raise ValueError("MAB conflict-rule instructions missing from admitted tasks")
    if len(judges) != (0 if is_mab else len(rows)):
        raise ValueError("judge receipt coverage differs from predictions")
    prompt = load_longmemeval_prompt(judge_source) if judge_source is not None else None
    for receipt, row in zip(judges, rows, strict=not is_mab):
        verdict = receipt["response"]["choices"][0]["message"]["content"].strip().lower()
        if verdict not in ("yes", "no") or verdict != row["judge_verdict"]:
            raise ValueError("judge verdict differs from actual response")
        if prompt is not None:
            sample = row["sample"]
            task = ("temporal-reasoning" if sample["category_name"] == "abstention"
                    else sample["category_name"])
            rendered = prompt(task, sample["question"], sample["gold"], row["prediction"]["answer"],
                              abstention=sample["id"].endswith("_abs"))
            if receipt["request"]["messages"] != [{"role": "user", "content": rendered}]:
                raise ValueError("judge prompt differs from pinned native prompt")
    production_path = out / "production-raw.jsonl"
    production = _lines(production_path) if production_path.exists() else []
    generated = [r for r in production if "ordinal" in r]
    if [r["ordinal"] for r in generated] != list(range(1, len(generated) + 1)):
        raise ValueError("production ordinal coverage differs")
    for row in generated:
        production_model = "openai/" + profile.model
        if cfg.get("dspy", {}).get("schema_mode") == "required":
            from gluomem_bench.bench.rack_schema import validate_native_projection, validate_schema_row
            production_model = validate_schema_row(row, profile, cfg, protocol, expected)
            validate_native_projection(row.get("response"), row.get("native_response", {}))
        if (row.get("error") or row.get("finish_reasons") != ["stop"]
                or row.get("model") != production_model
                or row.get("response", {}).get("model") != profile.model
                or row["output_budget"] != profile.output_budget(row["task"])):
            raise ValueError("invalid completed production generation")
        consumed[_key(row["messages"])] += 1
    if consumed != token_inputs:
        raise ValueError("served-tokenizer receipt coverage differs from generations")
    for row in tokenizers:
        count = row.get("response", {}).get("count")
        if row.get("status") != 200 or type(count) is not int or count < 0:
            raise ValueError("invalid served-tokenizer admission")
    usage = _json(out / "usage.json")
    if usage.get("generation_calls") != mem0_calls + len(direct) + len(generated):
        raise ValueError("consumed generation calls differ from actual receipts")
    if manifest["system"] == "gluomem" and require_real:
        dbs = list((out / "stores").glob("*.mind"))
        if len(dbs) != 1 or "production-raw.jsonl" not in inventory:
            raise ValueError("required production store/trace evidence missing")
        sentinel = dbs[0].with_suffix(".ingest.json")
        if (dbs[0].relative_to(out).as_posix() not in inventory
                or sentinel.relative_to(out).as_posix() not in inventory
                or _json(sentinel).get("complete") is not True):
            raise ValueError("required production ingest receipt missing")
    return {"questions": len(rows), "generation_calls": usage["generation_calls"],
            "frozen_plan_bound": expected is not None, "quality_admitted": quality_admitted}
