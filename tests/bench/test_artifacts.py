"""Authored byte/path contracts; no native SDK, retained runs or datasets."""

import hashlib
import json
from pathlib import Path

import pytest


def test_file_hash_streams_exact_bytes(tmp_path, monkeypatch):
    from gluomem_bench.bench import artifacts

    paths = [tmp_path / "empty", tmp_path / "binary"]
    materials = [b"", bytes(range(256)) * 8193 + "é\r\n".encode()]
    for path, material in zip(paths, materials, strict=True):
        path.write_bytes(material)
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("whole-file allocation"))
    assert [artifacts.sha256_file(p) for p in paths] == [
        hashlib.sha256(b).hexdigest() for b in materials
    ]
    with pytest.raises(FileNotFoundError):
        artifacts.sha256_file(tmp_path / "missing")


def test_artifact_verification_preserves_path_policies(tmp_path, monkeypatch):
    from gluomem_bench.bench import artifacts

    root = tmp_path / "bundle"
    root.mkdir()
    (root / "a").write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    verify = artifacts.verify_artifact_hashes
    (root / "sub").mkdir()
    opened, original_open = [], Path.open

    def observe_open(path, *args, **kwargs):
        opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe_open)
    verify(root, {"sub/../a": digest}, canonical_names=False)
    assert opened == [root / "sub/../a"]
    verify(root, {"a": digest}, canonical_names=True)
    for name in ("./a", str(root / "a")):
        verify(root, {name: digest}, canonical_names=False)
        with pytest.raises(ValueError, match="caller diagnostic"):
            verify(root, {name: digest}, canonical_names=True, error="caller diagnostic")
    for canonical in (False, True):
        for name, value in (("../outside", digest), ("a", "wrong")):
            with pytest.raises(ValueError):
                verify(root, {name: value}, canonical_names=canonical)
    with pytest.raises(ValueError):
        verify(root, {"missing": digest}, canonical_names=True)
    with pytest.raises(FileNotFoundError):
        verify(root, {"missing": digest}, canonical_names=False)
    with pytest.raises(ValueError):
        verify(root, {".": digest}, canonical_names=True)


@pytest.mark.parametrize("name", ["beam_driver", "comprehensive_driver"])
def test_frozen_inventory_golden_bytes_and_coverage(tmp_path, monkeypatch, name):
    import importlib
    import importlib.util
    import sys
    from types import SimpleNamespace

    import gluomem

    from gluomem_bench.bench import beam_scoring, beam_suite

    driver = importlib.import_module("gluomem_bench.bench." + name)
    project, core = tmp_path / "project", tmp_path / "core/gluomem"
    common = [
        "project/gluomem_bench/bench/driver.py",
        "project/gluomem_bench/bench/artifacts.py",
        "core/gluomem/__init__.py",
        "core/gluomem/storage/schema.sql",
        "core/gluomem/template.mako",
        "core/artifacts/dspy/prompt.json",
        "project/tests/bench/test_full_runtime_resources.py",
        "python.exe",
        "project/pyproject.toml",
        "project/uv.lock",
    ]
    extra = (
        [
            "project/tests/bench/test_beam_driver.py",
            "project/tests/bench/test_beam_driver_continuation.py",
            "upstream/scorer.py",
            "upstream/adapter.py",
            "data/one",
            "data/two",
        ]
        if name == "beam_driver"
        else ["project/tests/bench/test_comprehensive_driver.py", "dataset.json", "judge.py"]
    )
    deps = ["httpx", "dspy", "litellm", "pydantic", "numpy", "mem0"]
    if name == "beam_driver":
        deps += ["scipy", "pyarrow"]
    packages = [f"site/{d}/file{s}" for d in deps for s in (".py", ".so", ".JSON")]
    libs = [f"site/{d}.libs/opaque.bin" for d in deps]
    included = common + extra + packages + (libs if name == "beam_driver" else [])
    for relative in set(included + libs + ["site/numpy/ignored.txt", "core/ignored.py"]):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"authored\r\n")
    monkeypatch.setattr(driver, "__file__", str(project / "gluomem_bench/bench/driver.py"))
    monkeypatch.setattr(gluomem, "__file__", str(core / "__init__.py"))
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    requested = []

    def find_spec(dependency):
        requested.append(dependency)
        return SimpleNamespace(origin=str(tmp_path / "site" / dependency / "file.py"))

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(beam_scoring, "SOURCE_SHA256", {"scorer.py": "unused"})
    monkeypatch.setattr(beam_suite, "SOURCE_SHA256", {"adapter.py": "unused"})
    monkeypatch.setattr(beam_suite, "SPLITS", ["one", "two"])
    monkeypatch.setattr(beam_suite, "dataset_spec", lambda tier: {"files": [{"file": tier}]})

    def inventory():
        if name == "beam_driver":
            return driver.frozen_inventory(None, tmp_path / "data", tmp_path / "upstream", ["mem0"])
        return driver.frozen_inventory(tmp_path / "dataset.json", tmp_path / "judge.py")

    actual = inventory()
    expected = [
        dict(path=str((tmp_path / p).resolve()), sha256=hashlib.sha256(b"authored\r\n").hexdigest())
        for p in sorted(included, key=lambda p: str((tmp_path / p).resolve()))
    ]
    assert json.dumps(actual).encode() == json.dumps(expected).encode()
    assert set(requested) == set(deps)
    normalized = json.dumps(actual).replace(str(tmp_path).replace("\\", "\\\\"), "<fixture>")
    print(name, "golden", hashlib.sha256(normalized.encode()).hexdigest())
    from gluomem_bench.bench import artifacts

    shared = artifacts.runtime_source_paths(project, core)
    shared.update(artifacts.dependency_paths(deps, include_libraries=name == "beam_driver"))
    caller_only = {
        str((tmp_path / p).resolve())
        for p in extra + ["python.exe", "project/tests/bench/test_full_runtime_resources.py"]
    }
    assert artifacts.file_inventory(shared) == [r for r in expected if r["path"] not in caller_only]
    assert len(artifacts.file_inventory([core / "__init__.py"] * 2)) == 2
    with pytest.raises(FileNotFoundError):
        artifacts.file_inventory([tmp_path / "missing"])
    if name == "beam_driver":
        subset = driver.frozen_inventory("one", tmp_path / "data", tmp_path / "upstream", [])
        assert subset == [
            r
            for r in expected
            if "mem0" not in r["path"] and r["path"] != str((tmp_path / "data/two").resolve())
        ]
    (core / "storage/schema.sql").write_bytes(b"mutated")
    assert inventory() != actual
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="missing native .*dependency: "):
        inventory()


def test_suite_bytes_and_beam_noncanonical_admission_remain_distinct(tmp_path):
    from gluomem_bench.bench import beam_driver as beam
    from gluomem_bench.bench import suite_campaign as suite
    from gluomem_bench.bench import suite_evidence

    value = {"z": "é", "a": [1]}
    expected = b'{\n  "a": [\n    1\n  ],\n  "z": "\xc3\xa9"\n}\n'
    for write in (suite._write, beam.write):
        write(tmp_path / "sample.json", value)
        assert (tmp_path / "sample.json").read_bytes() == expected
        assert suite._hash(tmp_path / "sample.json") == hashlib.sha256(expected).hexdigest()
    (tmp_path / "plan.json").write_bytes(b"[]")
    candidate = dict(
        artifact_sha256={"./plan.json": hashlib.sha256(b"[]").hexdigest()},
        plan_sha256=beam.digest([]),
        case_ids=[],
        complete=False,
    )
    assert beam.verify_history(tmp_path, _candidate=candidate) == candidate
    inventory = {n: hashlib.sha256(b"{}").hexdigest() for n in suite_evidence.REQUIRED}
    for n in inventory:
        (tmp_path / n).write_bytes(b"{}")
    inventory["./sample.json"] = hashlib.sha256(expected).hexdigest()
    with pytest.raises(ValueError, match="artifact byte hash mismatch or noncanonical path"):
        suite_evidence.validate_bundle(tmp_path, {"artifact_sha256": inventory})
