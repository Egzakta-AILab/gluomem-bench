"""Harness surfaces honor the caller's required explicit DSPy model."""

from pathlib import Path

import pytest

from gluomem_bench.config import load_config


def _config(tmp_path: Path):
    return load_config(
        env_file=tmp_path / "absent.env",
        overrides={
            "dspy.model": "deepseek-flash",
            "dspy.cache.dir": str(tmp_path / "cache"),
            "embedding.model": "mock",
            "components.allow_fake": True,
            "search.rerank.enabled": False,
        },
    )


def test_answer_preflight_uses_explicit_inherited_model(tmp_path, monkeypatch):
    from gluomem_bench.bench.runner import _make_answer_fn

    cfg = _config(tmp_path)
    assert cfg.dspy.tasks[cfg.benchmark.answer.task].model is None
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        _make_answer_fn(cfg)


def test_component_readiness_uses_explicit_inherited_model(tmp_path, monkeypatch):
    import gluomem.extras

    from gluomem_bench.components import component_statuses

    cfg = _config(tmp_path)
    monkeypatch.setattr(gluomem.extras, "is_available", lambda _: True)
    statuses = component_statuses(cfg, env={})
    by_role = {status.role: status for status in statuses}
    for role in ("answer", "extraction:fact_extraction"):
        status = by_role[role]
        assert status.model == cfg.models[cfg.dspy.model].provider_model
        assert status.api_key_env == "OPENROUTER_API_KEY"
        assert not status.ready


@pytest.mark.parametrize("override", [None, "haiku"])
def test_answerer_reports_effective_selected_model(tmp_path, override):
    from dspy.utils.dummies import DummyLM

    from gluomem_bench.llm.answer import Answerer

    cfg = _config(tmp_path)
    if override is not None:
        cfg = cfg.with_overrides({"dspy.tasks.answer.model": override})
    answerer = Answerer(cfg, lm=DummyLM([]))
    assert answerer.model == (override or cfg.dspy.model)


def test_completed_offline_run_records_effective_selected_model(tmp_path):
    import hashlib
    import json

    from gluomem_bench.bench.runner import run_benchmark

    fixture = Path(__file__).parent / "fixtures" / "locomo_tiny.json"
    cfg = _config(tmp_path).with_overrides({
        "benchmark.datasets_dir": str(fixture.parent),
        "benchmark.datasets.locomo.file": fixture.name,
        "benchmark.datasets.locomo.sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "benchmark.sample.manifest": "does-not-exist.json",
        "benchmark.sample.per_category": 1,
        "benchmark.answer.backend": "fake",
        "benchmark.judge.backend": "fake",
    })
    result = run_benchmark(cfg, dataset="locomo", retriever="bm25", out_root=tmp_path / "runs")
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["answer"]["model"] == cfg.dspy.model
    assert manifest["answer"]["model_slug"] == cfg.models[cfg.dspy.model].provider_model


def test_optimizer_preflight_uses_explicit_inherited_model(tmp_path, monkeypatch):
    from gluomem_bench.llm.optimize import fixture_examples, optimize_task

    cfg = _config(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        optimize_task("fact_extraction", cfg, fixture_examples("fact_extraction"))


def test_noop_optimizer_artifact_records_explicit_inherited_model(tmp_path, monkeypatch):
    import json

    from gluomem_bench.llm.optimize import fixture_examples, optimize_task

    cfg = _config(tmp_path).with_overrides({"dspy.artifacts.dir": str(tmp_path / "artifacts")})
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-only-no-model-call")

    class NoopTeleprompter:
        def __init__(self, metric):
            self.metric = metric

        def compile(self, student, *, trainset):
            assert trainset
            return student

    result = optimize_task(
        "fact_extraction", cfg, fixture_examples("fact_extraction", splits=("train",)),
        teleprompter_factory=NoopTeleprompter,
    )
    manifest = json.loads((result / "manifest.gluomem.json").read_text(encoding="utf-8"))
    assert manifest["model_ref"] == cfg.dspy.model
