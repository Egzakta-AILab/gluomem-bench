"""Read only known driver receipt names and publish generated numeric views."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
from pathlib import Path

from gluomem_bench.tracking_projection import project_result

RECEIPTS = (
    "usage.json",
    "cost.json",
    "scores.json",
    "metrics.json",
    "summary.json",
    "denominators.json",
    "coverage.json",
    "root-status.json",
    "failure.json",
    "result.json",
    "campaign-result.json",
    "progress.json",
    "manifest.json",
    "task-statuses.json",
    "tuning_manifest.json",
    "ablation_report.json",
)
COVERAGE = (
    "expected_questions",
    "completed_questions",
    "expected_sessions",
    "completed_sessions",
    "expected_tasks",
    "completed_tasks",
    "planned_question_occurrences",
    "planned_history_occurrences",
    "planned_histories",
    "completed_histories",
)
ROLES = {
    "answer",
    "rubric",
    "fact_extraction",
    "event_claims",
    "consolidation",
    "consolidate",
    "reader",
    "judge",
    "extraction",
    "native_actor",
    "actor",
    "summarize",
    "reasoning",
}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _file(root, name):
    path = root / name
    if not path.is_file():
        return None
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("tracking receipt symlink escapes output root")
    return path


def _root(bound, result):
    for name in ("out", "out_dir", "output", "target"):
        if isinstance(bound.get(name), (str, Path)):
            return Path(bound[name])
    job = bound.get("job")
    if isinstance(getattr(job, "output", None), (str, Path)):
        return Path(job.output)
    root = getattr(result, "root", None)
    if isinstance(root, (str, Path)):
        return Path(root)
    if isinstance(result, Path) and result.is_dir():
        return result
    return None


def _result(result):
    if isinstance(result, (dict, list)):
        return result
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return dataclasses.asdict(result)
    return {}


def _pool_identity(attempt):
    key = tuple(attempt.get(k) for k in ("pool_key", "logical_ordinal", "ordinal", "intent"))
    if (
        not isinstance(key[0], str)
        or not key[0]
        or not isinstance(key[3], str)
        or not key[3]
        or any(type(n) is not int or n < 1 for n in key[1:3])
    ):
        raise ValueError("tracking pool attempt identity is incomplete")
    return key


def _wire_digest(row):
    physical = {k: row.get(k) for k in ("url", "request", "status", "response")}
    return hashlib.sha256(json.dumps(physical, sort_keys=True, allow_nan=False).encode()).digest()


def _wires(root):
    # Known disjoint physical sinks only: the Cognee driver stores its shared
    # reader in reader/ and native bridge in graph/. Never glob arbitrary files.
    for prefix in ("", "reader/"):
        # V10 mirrors native attempts into the primary physical stream. Its
        # separately configured common judge has a distinct physical sink.
        seen_pool = {}
        direct = [
            next(
                (prefix + name for name in aliases if _file(root, prefix + name) is not None), None
            )
            for aliases in (
                ("calls.jsonl", "http.jsonl"),
                ("calls-judge.jsonl", "http-judge.jsonl"),
            )
        ]
        for name in filter(None, (*direct, prefix + "production-raw.jsonl")):
            path = _file(root, name)
            if path is None:
                continue
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if name.endswith("production-raw.jsonl"):
                        if "ordinal" not in row:
                            continue  # wrapper/cache records are not physical attempts
                        if "pool_attempts" in row:
                            for attempt in row["pool_attempts"]:
                                wire = attempt.get("wire", {})
                                if not wire.get("url", "").endswith("/chat/completions"):
                                    continue  # pre-send failures remain authoritative unknown usage
                                key, digest = _pool_identity(attempt), _wire_digest(wire)
                                if key in seen_pool:
                                    if seen_pool[key] != digest:
                                        raise ValueError(
                                            "conflicting mirrored pool physical receipt"
                                        )
                                    continue
                                seen_pool[key] = digest
                                yield row.get("task") if row.get("task") in ROLES else "other", wire
                            continue
                    elif not row.get("url", "").endswith("/chat/completions"):
                        continue
                    elif "pool_attempt" in row:
                        key = _pool_identity(row["pool_attempt"])
                        if key in seen_pool:
                            raise ValueError("duplicate authoritative pool physical receipt")
                        seen_pool[key] = _wire_digest(row)
                    yield row.get("task") if row.get("task") in ROLES else "other", row
    for name in ("generation.jsonl", "native-cognee/generation.jsonl", "graph/generation.jsonl"):
        path = _file(root, name)
        if path is None:
            continue
        intents, responses = [], []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("kind") == "dispatch-intent":
                    intents.append(None)
                elif row.get("kind") == "/chat/completions":
                    try:
                        responses.append(json.loads(base64.b64decode(row["response_b64"])))
                    except (KeyError, ValueError, TypeError):
                        responses.append({})
        for index in range(len(intents)):
            yield "other", {"response": responses[index] if index < len(responses) else {}}


def collect_record(bound, result, *, exception=None, wall_seconds: float = 0.0):
    """Combine final/failure receipts without double-counting repeated summaries."""
    raw = _result(result)
    documents = {}
    root = _root(bound, result)
    if root is not None:
        for name in RECEIPTS:
            path = _file(root, name)
            if path is not None:
                if path.stat().st_size > 128 * 1024 * 1024:
                    raise ValueError("tracking receipt exceeds explicit 128 MiB parsing limit")
                documents[name] = json.loads(path.read_text(encoding="utf-8"))
    record = {
        "result": project_result(raw),
        "receipts": {
            name: project_result(
                value,
                _numeric=name
                in {
                    "scores.json",
                    "metrics.json",
                    "summary.json",
                    "denominators.json",
                    "coverage.json",
                    "ablation_report.json",
                    "tuning_manifest.json",
                },
            )
            for name, value in documents.items()
        },
        "timing": {"boundary_wall_seconds": wall_seconds},
        "usage": {},
        "coverage": {},
    }
    sources = [raw] + [
        documents[name]
        for name in (
            "usage.json",
            "cost.json",
            "failure.json",
            "root-status.json",
            "result.json",
            "campaign-result.json",
            "manifest.json",
        )
        if name in documents
    ]
    sources = [source for source in sources if isinstance(source, dict)]
    # Explicit shared coordinator/driver counters, not summed copies of them.
    calls = getattr(exception, "generation_calls", None)
    context = bound.get("context", bound.get("ctx"))
    if context is not None and _number(getattr(context, "calls", None)):
        calls = context.calls
    for source in sources:
        for view in (source, source.get("usage", {}), source.get("accounting", {})):
            if not isinstance(view, dict):
                continue
            candidate = view.get("generation_calls", view.get("calls"))
            if calls is None and _number(candidate):
                calls = candidate
        resources = source.get("resource_accounting", {})
        if isinstance(resources, dict) and resources.get("prior_debit_kind") == "reserved":
            actual = resources.get("actual_additional_spend", {}).get("calls")
            calls = actual if _number(actual) else source.get("accounting", {}).get("new_calls")
            allowance = resources.get("conservative_prior_allowance", {}).get("calls")
            record["usage"].update(
                reserved_prior_calls=allowance if _number(allowance) else None,
                call_count_kind="actual_additional_only",
            )
        for key in COVERAGE:
            if _number(source.get(key)):
                record["coverage"].setdefault(key, source[key])
    wire_count = known = prompt = completion = 0
    observed_cost, known_cost = 0.0, 0
    roles, latencies = {}, []
    if root is not None:
        for role, row in _wires(root):
            wire_count += 1
            stats = roles.setdefault(
                role,
                {
                    "calls": 0,
                    "known_usage_calls": 0,
                    "prompt_tokens_observed": 0,
                    "completion_tokens_observed": 0,
                },
            )
            stats["calls"] += 1
            response = row.get("response")
            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            if all(_number(usage.get(key)) for key in ("prompt_tokens", "completion_tokens")):
                known += 1
                prompt += usage["prompt_tokens"]
                completion += usage["completion_tokens"]
                stats["known_usage_calls"] += 1
                stats["prompt_tokens_observed"] += usage["prompt_tokens"]
                stats["completion_tokens_observed"] += usage["completion_tokens"]
            if _number(usage.get("cost")):
                known_cost += 1
                observed_cost += usage["cost"]
            if _number(row.get("elapsed_seconds")):
                latencies.append(row["elapsed_seconds"])
    if calls is not None:
        record["usage"]["generation_calls"] = calls
    if wire_count:
        if calls is not None and wire_count > calls:
            raise ValueError("physical trace attempts exceed authoritative call accounting")
        total = calls if calls is not None else wire_count
        record["usage"].update(
            traced_attempts=wire_count,
            known_usage_calls=known,
            unknown_usage_calls=total - known,
            prompt_tokens_observed=prompt,
            completion_tokens_observed=completion,
            prompt_tokens=prompt if known == total else None,
            completion_tokens=completion if known == total else None,
            cost_usd_observed=observed_cost if known_cost else None,
            cost_usd=observed_cost if known_cost == total else None,
            roles=roles,
        )
    else:
        for source in sources:
            for view in (source, source.get("usage", {})):
                if isinstance(view, dict):
                    # Preserve supplied quantities/estimates in a separate view;
                    # absence of wire accounting is not proof of all-attempt usage.
                    record["usage"].setdefault("reported", {}).update(project_result(view))
        if calls is not None:
            record["usage"]["unknown_usage_calls"] = calls
    if latencies:
        ordered = sorted(latencies)
        record["timing"].update(
            physical_attempt_seconds=sum(latencies),
            physical_attempt_count=len(latencies),
            physical_attempt_p50_seconds=ordered[len(ordered) // 2],
            physical_attempt_p95_seconds=ordered[
                min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
            ],
        )
    return record


def aggregate_children(record, children):
    """Roll up disjoint physical condition attempts, not copied result summaries."""
    if not children:
        return
    usages = [child.get("usage", {}) for child in children]
    observed = sum(u.get("known_usage_calls", 0) for u in usages)
    traced = sum(u.get("traced_attempts", 0) for u in usages)
    calls = record["usage"].get("generation_calls")
    if calls is None and all("generation_calls" in u for u in usages):
        calls = sum(u["generation_calls"] for u in usages)
        record["usage"]["generation_calls"] = calls
    if calls is not None and observed > calls:
        raise ValueError("child usage exceeds authoritative campaign call count")
    if traced:
        record["usage"].update(
            traced_attempts=traced,
            known_usage_calls=observed,
            unknown_usage_calls=calls - observed if calls is not None else None,
            prompt_tokens_observed=sum(u.get("prompt_tokens_observed", 0) for u in usages),
            completion_tokens_observed=sum(u.get("completion_tokens_observed", 0) for u in usages),
        )
        for key in ("prompt_tokens", "completion_tokens"):
            record["usage"][key] = record["usage"][key + "_observed"] if observed == calls else None
        costs = [u.get("cost_usd_observed") for u in usages]
        record["usage"]["cost_usd_observed"] = (
            sum(c for c in costs if c is not None) if any(c is not None for c in costs) else None
        )
        child_calls = sum(u.get("generation_calls", 0) for u in usages)
        record["usage"]["cost_usd"] = (
            sum(u["cost_usd"] for u in usages)
            if (all(u.get("cost_usd") is not None for u in usages) and child_calls == calls)
            else None
        )
    record["coverage"]["attempted_conditions"] = len(children)
    record["timing"]["child_boundary_seconds_sum"] = sum(
        c.get("timing", {}).get("boundary_wall_seconds", 0) for c in children
    )
    # Summed service time is explicitly different from scheduling-aware wall time.
    record["timing"]["physical_attempt_seconds_sum"] = sum(
        c.get("timing", {}).get("physical_attempt_seconds", 0) for c in children
    )


def metric_values(record, family):
    """Native metrics stay family/objective-specific; missing values stay absent."""
    result = {}

    def walk(value, path):
        if isinstance(value, dict):
            if isinstance(value.get("tag"), dict) and "metric_family" in value["tag"]:
                tag = value["tag"]
                path = path + [
                    tag["metric_family"],
                    tag.get("judge_family", "none"),
                    tag.get("backbone_sha256", "none")[:12],
                    value.get("scope", value.get("scope_sha256", "unknown"))[:16],
                ]
            for key, item in value.items():
                walk(item, path + [key.replace(" ", "_").replace("/", "_")])
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, path + [str(index)])
        elif type(value) in (int, float) and math.isfinite(value):
            result[".".join(path)[:240]] = float(value)

    for key in ("usage", "timing", "coverage"):
        walk(record.get(key, {}), [key])
    raw = record.get("result", {})
    if isinstance(raw, dict):
        for key in ("metrics", "native_metrics", "scores", "score", "summary", "report"):
            if key in raw:
                walk(raw[key], [family, "native", key])
    for name, view in record.get("receipts", {}).items():
        if name in {
            "scores.json",
            "metrics.json",
            "summary.json",
            "denominators.json",
            "coverage.json",
        }:
            walk(view, [family, "native" if name != "denominators.json" else "coverage", name[:-5]])
    return result
