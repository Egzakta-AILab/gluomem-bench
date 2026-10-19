"""Dependency-light byte and inventory mechanics; admission policy stays local."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path):
    """Hash file bytes without materializing the entire file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_hashes(root, inventory, *, canonical_names,
                           error="artifact byte hash mismatch or noncanonical path"):
    """Check containment/bytes; callers choose canonical spelling and diagnostics."""
    root, resolved_root = Path(root), Path(root).resolve()
    for name, digest in inventory.items():
        target = (root / name).resolve()
        if (not target.is_relative_to(resolved_root)
                or (canonical_names and (target.relative_to(resolved_root).as_posix() != name
                                         or not target.is_file()))
                or sha256_file(target if canonical_names else root / name) != digest):
            raise ValueError(error)


def runtime_source_paths(project, core):
    """Shared producer/core code and resources; benchmark-specific inputs stay local."""
    paths = set((project / "gluomem_bench").rglob("*.py"))
    for pattern in ("*.py", "*.sql", "*.mako"):
        paths.update(core.rglob(pattern))
    paths.update((core.parent / "artifacts/dspy").glob("*.json"))
    paths.update(project / name for name in ("pyproject.toml", "uv.lock")
                 if (project / name).is_file())
    return paths


def dependency_paths(names, *, include_libraries=False, missing_error="missing native dependency"):
    """Enumerate runtime files without importing dependency packages."""
    import importlib.util

    paths = set()
    for name in names:
        location = importlib.util.find_spec(name)
        if location is None or not location.origin:
            raise ValueError(f"{missing_error}: {name}")
        package = Path(location.origin).parent
        paths.update(p for p in package.rglob("*")
                     if p.is_file() and p.suffix.lower() in {".py", ".pyd", ".dll", ".so", ".json"})
        libraries = package.parent / (package.name + ".libs")
        if include_libraries and libraries.is_dir():
            paths.update(p for p in libraries.rglob("*") if p.is_file())
    return paths


def file_inventory(paths):
    """Retain supplied path multiplicity and historical resolved-path ordering."""
    return [dict(path=str(p.resolve()), sha256=sha256_file(p))
            for p in sorted(paths, key=lambda p: str(p.resolve()))]
