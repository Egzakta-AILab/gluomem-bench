"""Content-free projections for experiment metadata and metric artifacts.

Only this generated view is eligible for MLflow upload. Never upload an input,
profile, trace, manifest, compiled prompt, response, or arbitrary directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

# Fixed fields are admitted outside metric/accounting subtrees. Inside those
# subtrees only finite numbers, nulls, booleans and containers survive.
CONTAINERS = frozenset(
    {
        "usage",
        "accounting",
        "cost",
        "timing",
        "coverage",
        "denominators",
        "metrics",
        "native_metrics",
        "report",
        "summary",
        "scores",
        "score",
        "result",
        "jobs",
        "tasks",
        "rows",
        "questions",
        "counts",
        "roles",
        "per_objective",
        "per_category",
        "matrix",
        "conditions",
        "unsupported_cells",
        "failures",
        "native",
        "totals",
        "resources",
        "all_attempt",
        "observed_usage",
        "tag",
        "by_category",
        "by_objective",
        "official_metrics",
        "supplemental_metrics",
        "method_metrics",
    }
)
COUNTERS = frozenset(
    {
        "generation_calls",
        "native_generation_calls",
        "reader_judge_generation_calls",
        "calls",
        "known_usage_calls",
        "unknown_usage_calls",
        "unmetered_generation_calls",
        "requests",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_observed",
        "completion_tokens_observed",
        "cost_usd",
        "currency_cost",
        "billed_cost_usd",
        "wall_seconds",
        "elapsed_seconds",
        "elapsed_ms",
        "build_seconds",
        "query_seconds",
        "predict_seconds",
        "judge_seconds",
        "expected_questions",
        "completed_questions",
        "expected_sessions",
        "completed_sessions",
        "planned_histories",
        "attempted_histories",
        "completed_histories",
        "expected_tasks",
        "completed_tasks",
        "planned_question_occurrences",
        "planned_history_occurrences",
        "matrix_complete",
        "quality_complete",
        "complete",
        "authoritative",
        "n",
        "k",
        "value",
        "total",
        "graded",
        "correct",
        "accuracy",
        "n_trials",
        "completed_trials",
        "pruned_trials",
        "incumbent_score",
        "best_score",
        "improvement",
        "best_trial_number",
        "incumbent_dev_score",
        "best_dev_score",
        "promoted",
        "model_calls",
        "cpu_seconds",
        "rss_bytes",
        "index_bytes",
        "cache_hit_tokens",
    }
)
IDENTITIES = frozenset(
    {
        "job_id",
        "history_id",
        "task_id",
        "case_id",
        "sample_id",
        "question_id",
        "group_id",
        "output",
        "root",
        "source_ref",
        "sample_ids",
        "case_ids",
        "trainset_path",
    }
)
STATUSES = frozenset(
    {
        "complete",
        "completed",
        "complete_verified",
        "failed",
        "incomplete",
        "unrun",
        "queued",
        "running",
        "cancelled",
        "unsupported",
        "not_supported",
        "not_admitted",
        "not_applicable",
        "not_applicable_context_overflow",
        "context_refused",
        "weak",
        "partial",
        "blocked",
        "not_attempted",
        "unknown",
        "absent",
        "verified",
        "success",
        "ok",
        "COMPLETE",
        "verified_pending_post_exit",
        "incomplete-no-full-score",
        "FAIL",
    }
)
LABELS = frozenset(
    {
        "gluomem",
        "mem0",
        "cognee",
        "bm25",
        "dense",
        "hybrid",
        "full_context",
        "no_memory",
        "vector_rag",
        "mem0/cognee",
        "mem0-source-clock",
        "cognee-final-answer",
        "100K",
        "500K",
        "1M",
        "10M",
        "Small",
        "Medium",
        "qa_accuracy",
        "ndcg",
        "recall",
        "hit",
        "event_ordering",
        "longmemeval_s",
        "longmemeval_oracle",
        "beam",
        "memoryagentbench",
        "locomo",
        "memoryarena",
        "longmemeval_v2",
    }
)
BLOCKED = re.compile(
    r"(?i)(password|passwd|secret|api.?key|authorization|credential|messages?|response|"
    r"request|prompt(?!_tokens)|content|question(?!s?$|_id$)|answer(?!_.*tokens|_.*cost)|"
    r"reason$|error$|trace|headers?|environment|config)"
)
KEY = re.compile(r"^[A-Za-z0-9_./@ -]{1,100}$")
PUBLIC_METRIC_KEYS = (
    COUNTERS
    | CONTAINERS
    | frozenset(
        {
            "by_category",
            "knowledge-update",
            "single-session-user",
            "single-session-assistant",
            "single-session-preference",
            "multi-session",
            "temporal-reasoning",
            "event_ordering",
            "conflict_resolution",
            "accurate_retrieval",
            "test_time_learning",
            "long_range_understanding",
            "preference",
            "preference_learning",
            "temporal_reasoning",
            "fact_recall",
            "contradiction",
            "exact_match",
            "accuracy",
            "mean",
            "std",
            "f1",
            "precision",
            "recall",
            "nugget_recall",
            "tau_norm",
            "final_score",
            "success_rate",
            "ndcg",
            "hit",
            "mrr",
            "overall",
            "total_steps",
            "math",
            "phys",
            "travel",
            "shopping",
            "search",
            "no_memory",
            "full_context",
            "bm25",
            "dense",
            "hybrid",
            "gluomem",
            "mem0",
            "cognee",
            "reader",
            "judge",
            "answer",
            "rubric",
            "extraction",
            "consolidation",
            "native_actor",
            "other",
            "prompt_tokens",
            "completion_tokens",
            "prior_calls",
            "new_calls",
            "elapsed_seconds",
            "count",
            "sessions",
            "source_events",
            "processed_events",
            "committed_memories",
            "total_tasks",
            "successful_tasks",
            "failed_tasks",
        }
    )
)


def digest(value):
    """Hash identity without publishing the source value."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True, default=lambda _: "<opaque>").encode()
    ).hexdigest()


def project_result(value, *, _numeric=False):
    """Allowlist counters and native objectives; keep unknown values null."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not KEY.fullmatch(key):
                continue
            if key == "failures" and isinstance(item, dict):
                result[key] = {
                    name
                    if name in {"answer", "judge", "extraction", "ingestion"}
                    else "other_" + digest(name)[:16]: len(value)
                    if isinstance(value, (list, tuple))
                    else int(bool(value))
                    for name, value in item.items()
                }
            elif key == "extraction_errors" and isinstance(item, (list, tuple)):
                result["failures"] = {**result.get("failures", {}), "extraction": len(item)}
            elif key in IDENTITIES:
                result[key + "_sha256"] = digest(item)
            elif key.endswith(("_sha256", "_digest", "_sha")):
                if isinstance(item, str) and re.fullmatch(r"[0-9a-f]{40,64}", item):
                    result[key] = item
            elif key in {"status", "quality_status"} and isinstance(item, str) and item in STATUSES:
                result[key] = item
            elif (
                key == "metric_family"
                and isinstance(item, str)
                and re.fullmatch(r"(?:qa_accuracy|recall|ndcg|hit|mrr|precision|f1)(?:@\d+)?", item)
            ):
                result[key] = item
            elif key in {"scope", "judge_family"} and item in (
                "overall",
                "single",
                "trio_strict",
                "none",
            ):
                result[key] = item
            elif key in {"scope", "backbone", "protocol"} and isinstance(item, str):
                result[key + "_sha256"] = digest(item)
            elif key in {"system", "method", "retriever", "dataset", "split", "metric", "family"}:
                if isinstance(item, str) and item in LABELS:
                    result[key] = item
            elif not BLOCKED.search(key) and (key in CONTAINERS or key in COUNTERS or _numeric):
                numeric = _numeric or key in CONTAINERS - {
                    "jobs",
                    "tasks",
                    "rows",
                    "questions",
                    "result",
                }
                projected = project_result(item, _numeric=numeric)
                if projected is not None or item is None:
                    name = key if key in PUBLIC_METRIC_KEYS else "label_" + digest(key)[:16]
                    result[name] = projected
        return result
    if isinstance(value, (list, tuple)):
        return [
            project_result(item, _numeric=_numeric)
            for item in value
            if isinstance(item, (dict, list, tuple, int, float)) or item is None
        ]
    if value is None or isinstance(value, bool):
        return value
    if type(value) in (int, float) and math.isfinite(value):
        return value
    return None
