"""Optional-dependency availability and install hints for gluomem-bench."""

from __future__ import annotations

import importlib.util

from gluomem.extras import EXTRAS as CORE_EXTRAS
from gluomem.extras import MissingExtraError

EXTRAS: dict[str, tuple[tuple[str, ...], str]] = {
    **CORE_EXTRAS,
    "tune": (
        ("optuna",),
        "Offline retrieval parameter tuning with Optuna",
    ),
    "mem0": (
        ("mem0",),
        "Mem0 baseline retriever comparison",
    ),
}


def _missing(extra: str) -> list[str]:
    if extra not in EXTRAS:
        raise KeyError(f"unknown extra '{extra}'; known: {sorted(EXTRAS)}")
    modules, _ = EXTRAS[extra]
    return [m for m in modules if importlib.util.find_spec(m) is None]


def is_available(extra: str) -> bool:
    """Return True when every distribution for ``extra`` is importable."""
    return not _missing(extra)


def require(extra: str, feature: str | None = None) -> None:
    """Raise :class:`MissingExtraError` unless ``extra`` is installed."""
    missing = _missing(extra)
    if missing:
        _, default_feature = EXTRAS[extra]
        raise MissingExtraError(extra, feature or default_feature, missing)


def status() -> dict[str, bool]:
    """Installation status of every optional extra (no imports, no downloads)."""
    return {name: is_available(name) for name in EXTRAS}


__all__ = [
    "EXTRAS",
    "MissingExtraError",
    "_missing",
    "is_available",
    "require",
    "status",
]
