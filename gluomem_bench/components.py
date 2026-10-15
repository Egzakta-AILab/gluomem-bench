"""Benchmark component readiness and fake gating."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from gluomem.components import (
    ComponentNotReadyError,
    ComponentStatus,
    FakeComponentError,
    ProviderUnavailableError,
    _classifier_status,
    _embedder_status,
    _llm_status,
    _reranker_status,
    require_fake_allowed,
)

if TYPE_CHECKING:
    from gluomem.storage.db import MindDB

    from gluomem_bench.config import AppConfig

__all__ = [
    "ComponentNotReadyError",
    "ComponentStatus",
    "FakeComponentError",
    "ProviderUnavailableError",
    "assert_ready",
    "component_report",
    "component_statuses",
    "require_fake_allowed",
]


def component_statuses(
    cfg: AppConfig, *, env: Mapping[str, str] | None = None
) -> list[ComponentStatus]:
    """One status per active component role, including benchmark answerer and judges."""
    env = os.environ if env is None else env
    statuses = [_embedder_status(cfg), _reranker_status(cfg), _classifier_status(cfg)]
    answer = cfg.benchmark.answer
    for task, task_cfg in sorted(cfg.dspy.tasks.items()):
        if task == answer.task:
            continue
        statuses.append(
            _llm_status(
                cfg, f"extraction:{task}", cfg.write.extraction_backend, task_cfg.model, env
            )
        )
    statuses.append(
        _llm_status(cfg, "answer", answer.backend, cfg.dspy.tasks[answer.task].model, env)
    )
    for backbone in cfg.benchmark.judge.backbones:
        statuses.append(
            _llm_status(cfg, f"judge:{backbone}", cfg.benchmark.judge.backend, backbone, env)
        )
    return statuses


def component_report(
    cfg: AppConfig,
    *,
    db: MindDB | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """JSON-ready description of the effective components."""
    statuses = component_statuses(cfg, env=env)
    report: dict[str, Any] = {
        "allow_fake": cfg.components.allow_fake,
        "ready": all(s.ready for s in statuses),
        "components": [s.as_dict() for s in statuses],
    }
    if db is not None:
        from gluomem.embedding.compat import has_vectors, stored_fingerprint

        stored = stored_fingerprint(db)
        configured = statuses[0].fingerprint
        report["store"] = {
            "path": str(db.path),
            "dim": db.dim,
            "has_vectors": has_vectors(db),
            "fingerprint": None if stored is None else stored.as_dict(),
            "embedder_matches": None
            if stored is None or configured is None
            else stored.as_dict() == configured,
        }
    return report


def assert_ready(cfg: AppConfig, *, env: Mapping[str, str] | None = None) -> None:
    """Raise :class:`ComponentNotReadyError` if any active component is blocked."""
    blocked = [s for s in component_statuses(cfg, env=env) if not s.ready]
    if blocked:
        raise ComponentNotReadyError(blocked)
