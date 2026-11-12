"""Allowlisted experiment identities; no credentials or ambient model discovery."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import re
import subprocess
from pathlib import Path

from gluomem_bench.tracking_projection import digest

PROFILE_FIELDS = frozenset(
    {
        "model",
        "provider",
        "provider_model",
        "requested_model",
        "served_model",
        "profile_version",
        "temperature",
        "max_tokens",
        "timeout_seconds",
        "max_input_chars",
        "context_window",
        "schema_reserve_tokens",
        "max_generation_calls",
        "max_run_seconds",
        "generation_budgets",
        "concurrency",
        "reader",
        "judge",
        "extraction",
        "consolidation",
        "answer",
        "rubric",
        "provider_routing",
        "routing",
        "only",
        "allow_fallbacks",
        "require_parameters",
        "quantizations",
        "roles",
        "role_profiles",
        "memory",
        "actor",
        "reasoning",
        "max_calls",
        "max_seconds",
        "seed",
        "revision",
        "dim",
        "embedding",
        "tokenizer",
        "thinking",
        "reasoning_effort",
    }
)
PARAM_FIELDS = frozenset(
    {
        "system",
        "method",
        "retriever",
        "dataset",
        "task",
        "seed",
        "k",
        "max_steps",
        "max_rounds",
        "max_tool_iterations",
        "evidence_token_budget",
        "max_workers",
        "max_calls",
        "max_seconds",
    }
)
TOKEN = re.compile(r"^[A-Za-z0-9_./:@+ -]{1,200}$")
SENSITIVE = re.compile(r"(?i)(sk[-_]|hf_|bearer|password|secret|credential|api.?key)")


def _mapping(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {}


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items() if key in PROFILE_FIELDS}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if value is None or type(value) in (int, float, bool):
        return value
    if isinstance(value, str) and TOKEN.fullmatch(value) and not SENSITIVE.search(value):
        return value
    return None


def _tree_identity(root):
    h = hashlib.sha256()
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".sql", ".mako"}]
    paths += [p for p in (root.parent / "pyproject.toml", root.parent / "uv.lock") if p.is_file()]
    for path in sorted(paths):
        h.update(path.relative_to(root.parent).as_posix().encode())
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def code_identities():
    harness = Path(__file__).resolve().parent
    result = {"code.harness_sha256": _tree_identity(harness)}
    core = importlib.util.find_spec("gluomem")
    if core is not None and core.origin:
        result["code.product_sha256"] = _tree_identity(Path(core.origin).parent)
    for label, path in (
        ("harness", harness.parent),
        ("product", Path(core.origin).parent.parent if core and core.origin else None),
    ):
        if path is None:
            continue
        try:
            sha = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            ).strip()
            if re.fullmatch(r"[0-9a-f]{40,64}", sha):
                result[f"code.{label}_commit"] = sha
            dirty = subprocess.check_output(
                ["git", "-C", str(path), "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            )
            result[f"code.{label}_dirty"] = bool(dirty.strip())
        except (OSError, subprocess.SubprocessError):
            pass  # source byte identity is authoritative outside a Git checkout
    return result


def experiment_params(function, bound, *, code=None):
    """Capture effective declared bindings only; never use env model fallbacks."""
    result = dict(code if code is not None else code_identities())
    result["entrypoint"] = function.__module__ + "." + function.__name__
    source = inspect.getsourcefile(function)
    if source and Path(source).is_file():
        result["code.entrypoint_sha256"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    for key in PARAM_FIELDS:
        value = _clean(bound.get(key))
        if value is not None and not isinstance(value, (dict, list)):
            result[key] = value
    spec = _mapping(bound.get("spec"))
    for name in ("spec_path", "candidate_path"):
        path = bound.get(name)
        if isinstance(path, (str, Path)):
            path = Path(path)
            if (
                path.is_file()
                and not SENSITIVE.search(path.name)
                and path.name not in {".env", "secrets.json"}
            ):
                raw = path.read_bytes()
                result["data." + name.removesuffix("_path") + "_sha256"] = hashlib.sha256(
                    raw
                ).hexdigest()
                try:
                    value = json.loads(raw)
                    if isinstance(value, dict):
                        spec = value
                except ValueError:
                    pass
    profile = _clean(_mapping(bound.get("profile", spec.get("profile"))))
    if profile:
        result["profile.sha256"] = digest(profile)

        def flatten(view, prefix):
            for key, value in view.items():
                if isinstance(value, dict):
                    flatten(value, prefix + key + ".")
                elif value is not None:
                    result[prefix + key] = (
                        json.dumps(value, separators=(",", ":"))
                        if isinstance(value, list)
                        else value
                    )

        flatten(profile, "profile.")
    adapter = bound.get("adapter")
    data = spec.get("dataset", {})
    if not isinstance(data, dict):
        data = {}
    sha = getattr(adapter, "sha", None) or data.get("sha256")
    if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha):
        result["data.sha256"] = sha
    selection = next(
        (
            bound[key]
            for key in ("case_ids", "sample_ids", "samples", "records", "jobs")
            if key in bound
        ),
        spec.get("selection"),
    )
    if selection is not None:
        if isinstance(selection, (list, tuple)):
            selection = [getattr(item, "id", getattr(item, "job_id", item)) for item in selection]
        result["data.selection_sha256"] = digest(selection)
    cfg = _mapping(bound.get("cfg"))
    if cfg:
        # Bind task policy/model aliases and selected model revisions, not raw
        # config dumps (which may contain credentials or private annotations).
        dspy = cfg.get("dspy", {})
        tasks = dspy.get("tasks", {}) if isinstance(dspy, dict) else {}
        models = cfg.get("models", {})
        effective = {
            "model": _clean(dspy.get("model", dspy.get("default_model"))),
            "tasks": {name: _clean(_mapping(task)) for name, task in tasks.items()},
            "models": {name: _clean(_mapping(model)) for name, model in models.items()},
        }
        for task in effective["tasks"].values():
            if task.get("model") is None:
                task["model"] = effective["model"]
        result["config.sha256"] = digest(effective)
        if "profile.sha256" not in result:
            result["profile.sha256"] = digest(
                {"model": effective["model"], "tasks": effective["tasks"]}
            )
            if effective["model"] is not None:
                result["profile.model"] = effective["model"]
        if sha is None and isinstance(cfg.get("benchmark"), dict):
            dataset = bound.get("dataset")
            entry = cfg["benchmark"].get("datasets", {}).get(dataset, {})
            if isinstance(entry, dict) and re.fullmatch(
                r"[0-9a-f]{64}", str(entry.get("sha256", ""))
            ):
                result["data.sha256"] = entry["sha256"]
    cache = bound.get("cache")
    manifest = getattr(cache, "manifest", None)
    if callable(getattr(manifest, "digest", None)):
        result["data.cache_sha256"] = manifest.digest()
    return result
