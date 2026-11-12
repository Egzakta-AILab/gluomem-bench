"""Shared bench test fixtures: fixture adapter, per-conversation ingested DBs, and fake backends."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
import threading
from pathlib import Path

import pytest

from gluomem_bench.bench.adapters.locomo import LoCoMoAdapter
from gluomem_bench.config import AppConfig, load_config
from gluomem_bench.llm.outputs import AnswerResult

FIXTURE = Path(__file__).parent.parent / "fixtures" / "locomo_tiny.json"
FIXTURE_SHA = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()

BENCH_OVERRIDES: dict[str, object] = {
    "dspy.model": "mock",
    "benchmark.datasets_dir": str(FIXTURE.parent),
    "benchmark.datasets.locomo.file": FIXTURE.name,
    "benchmark.datasets.locomo.sha256": FIXTURE_SHA,
    "embedding.model": "mock",
    "components.allow_fake": True,
    "search.rerank.enabled": False,
}

FAKE_BACKENDS: dict[str, object] = {
    "benchmark.answer.backend": "fake",
    "benchmark.judge.backend": "fake",
    "benchmark.sample.per_category": 1,
    "benchmark.sample.manifest": "does-not-exist.json",
}


@pytest.fixture
def rack_profile_path(tmp_path: Path) -> Path:
    from gluomem_bench.bench.rack import RackProfile

    path = tmp_path / "operator-profile.json"
    path.write_text(
        RackProfile(
            base_url="http://localhost:8000/v1", model="example/served-model"
        ).model_dump_json(),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def bench_cfg() -> AppConfig:
    return load_config(overrides=dict(BENCH_OVERRIDES))


@pytest.fixture
def fixture_adapter(bench_cfg: AppConfig) -> LoCoMoAdapter:
    return LoCoMoAdapter(bench_cfg, path=FIXTURE)


class RecordingAnswerFn:
    """Deterministic AnswerFn double: echoes a scripted answer and records calls."""

    def __init__(self, answer: str = "scripted answer") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def __call__(self, question: str, context: str) -> AnswerResult:
        with self._lock:
            self.calls.append((question, context))
        return AnswerResult(text=self.answer, prompt_tokens=10, completion_tokens=5)


@pytest.fixture
def answer_fn() -> RecordingAnswerFn:
    return RecordingAnswerFn()


@pytest.fixture
def recording_answer_fn_cls() -> type[RecordingAnswerFn]:
    return RecordingAnswerFn


@pytest.fixture
def ingested_db_dir(bench_cfg: AppConfig, tmp_path: Path) -> Path:
    from gluomem_bench.ingest import ingest_path
    from gluomem_bench.llm import NullExtractors

    cfg = bench_cfg.with_overrides({"write.extraction_backend": "none"})
    db_dir = tmp_path / "dbs"
    db_dir.mkdir()
    ingest_path(FIXTURE, cfg, facade=NullExtractors(), db_path=db_dir / "conv-tiny-1.mind")
    return db_dir


@pytest.fixture
def authored_private_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Copy public source into a TEST ONLY runtime; never copy captured authority.

    Chain admission intentionally rejects ambient checkouts. Import both actual
    packages from their isolated source copies, then restore the caller's module
    identities. No __file__, inventory, admission or source guard is mocked.
    """
    import gluomem

    from gluomem_bench.bench.artifacts import runtime_source_paths

    project = Path(__file__).resolve().parents[2]
    assert gluomem.__file__ is not None
    core = Path(gluomem.__file__).resolve().parent
    runtime = tmp_path / "authored-launch" / "runtime"
    bench_copy, core_copy = runtime / "gluomem-bench", runtime / "GluoMem"
    paths = runtime_source_paths(project, core)
    paths.update(project / "tests/bench" / name for name in (
        "test_comprehensive_driver.py", "test_full_runtime_resources.py",
    ))
    inventory = {}
    for source in sorted(paths):
        target = (bench_copy / source.relative_to(project) if source.is_relative_to(project)
                  else core_copy / source.relative_to(core.parent))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
        inventory[str(target)] = {"source": str(source), "sha256": expected}
    (runtime.parent / "TEST-ONLY-authored-runtime.json").write_text(
        json.dumps({"provenance": "synthetic source-copy fixture; no captured evidence",
                    "operational_execution_authorized": False, "files": inventory},
                   sort_keys=True), encoding="utf-8",
    )

    def product(name):
        return name.split(".")[0] in {"gluomem", "gluomem_bench"}

    saved = {name: module for name, module in sys.modules.copy().items() if product(name)}
    with monkeypatch.context() as imports:
        imports.syspath_prepend(str(core_copy))
        imports.syspath_prepend(str(bench_copy))
        try:
            for name in saved:
                del sys.modules[name]
            importlib.invalidate_caches()
            driver = importlib.import_module("gluomem_bench.bench.comprehensive_driver")
            assert driver.__file__ is not None
            assert Path(driver.__file__).resolve() == (
                bench_copy / "gluomem_bench/bench/comprehensive_driver.py"
            ).resolve()
            yield driver
        finally:
            observed = {}
            for name, module in sys.modules.copy().items():
                source = getattr(module, "__file__", None)
                if product(name) and source is not None:
                    path = Path(source).resolve()
                    observed[name] = {"path": str(path),
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            (runtime.parent / "TEST-ONLY-imported-identities.json").write_text(
                json.dumps(observed, sort_keys=True), encoding="utf-8",
            )
            for name in list(sys.modules):
                if product(name):
                    del sys.modules[name]
            sys.modules.update(saved)
            importlib.invalidate_caches()
