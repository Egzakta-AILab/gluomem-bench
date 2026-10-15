"""Benchmark config, component readiness, optional extras, and CLI ingest tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from gluomem.components import FakeComponentError
from gluomem.embedding.providers import MockEmbedder, make_embedder
from gluomem.llm import make_extractors
from pydantic import ValidationError

from gluomem_bench import extras
from gluomem_bench.bench.runner import _make_answer_fn, _make_judge_fns
from gluomem_bench.cli import main
from gluomem_bench.components import assert_ready, component_report
from gluomem_bench.config import AppConfig, load_config

FIXTURE = str(Path(__file__).parent / "fixtures" / "locomo_tiny.json")
_FAKES = {
    "embedding.model": "mock",
    "write.extraction_backend": "fake",
    "benchmark.answer.backend": "fake",
    "benchmark.judge.backend": "fake",
}


def test_benchmark_default_config_loads() -> None:
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert "answer" in cfg.dspy.tasks
    assert cfg.benchmark.judge.family == "trio_strict"
    assert cfg.benchmark.judge.backbones == ["haiku", "gpt5-mini", "qwen3"]
    assert cfg.benchmark.mem0.model == "deepseek-flash"
    assert cfg.benchmark.mem0.embedder_model == "minilm"
    assert cfg.benchmark.mem0.top_k is None
    assert cfg.benchmark.mem0.infer is True


def test_status_lists_every_extra() -> None:
    assert set(extras.status()) >= {"llm", "embeddings", "postgres", "tune", "service"}


def test_fake_backends_require_allow_fake() -> None:
    cfg = load_config().with_overrides(_FAKES)
    for build in (make_embedder, make_extractors, _make_answer_fn, _make_judge_fns):
        with pytest.raises(FakeComponentError):
            build(cfg)
    allowed = cfg.with_overrides({"components.allow_fake": True})
    assert isinstance(make_embedder(allowed), MockEmbedder)
    for build in (make_extractors, _make_answer_fn, _make_judge_fns):
        assert build(allowed) is not None


def test_benchmark_component_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extras, "is_available", lambda _: True)
    import gluomem.extras as core_extras

    monkeypatch.setattr(core_extras, "is_available", lambda _: True)
    secret = "sk-or-very-secret-value"
    report = component_report(load_config(), env={"OPENROUTER_API_KEY": secret})
    by_role = {c["role"]: c for c in report["components"]}
    assert "answer" in by_role
    assert "judge:haiku" in by_role
    assert report["ready"] is True
    assert_ready(load_config(), env={"OPENROUTER_API_KEY": secret})


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"benchmark.judge.backbones": ["haiku", "gpt5-mini", "unknown-judge"]},
            "unknown-judge",
        ),
        ({"benchmark.judge.backbones": ["haiku", "gpt5-mini"]}, "trio_strict"),
        ({"benchmark.judge.backbones": ["haiku", "haiku", "qwen3"]}, "unique"),
        ({"benchmark.judge.family": "quintet"}, "family"),
        ({"benchmark.answer.backend": "real"}, "backend"),
        ({"benchmark.judge.backend": "llm"}, "backend"),
        ({"benchmark.answer.task": "no_such_task"}, "no_such_task"),
        ({"benchmark.sample.categories": [1, 1, 2]}, "categories"),
        ({"benchmark.sample.categories": [0, 6]}, "categories"),
        ({"benchmark.sample.categories": []}, "categories"),
        ({"benchmark.ci_gate.metric": "vibes"}, "vibes"),
        ({"benchmark.ci_gate.metric": "evidence_hit@7"}, "k=7"),
        ({"benchmark.ci_gate.min": 1.5}, "min"),
        ({"benchmark.full_context.keep": "middle"}, "keep"),
        ({"benchmark.mem0.model": "gpt-99"}, "gpt-99"),
        ({"benchmark.mem0.embedder_model": "reranker"}, "dim"),
        ({"benchmark.mem0.top_k": 0}, "top_k"),
        ({"benchmark.answer.style": "chatty"}, "preference_aware"),
    ],
)
def test_invalid_benchmark_config_overrides_fail_loudly(
    overrides: dict[str, object], match: str
) -> None:
    cfg = load_config()
    with pytest.raises(ValidationError, match=match):
        cfg.with_overrides(overrides)


def test_cli_ingest_backend_none_e2e(tmp_path: Path, capsys) -> None:
    rc = main(
        [
            "ingest",
            FIXTURE,
            "--db",
            str(tmp_path / "cli.mind"),
            "--override",
            "write.extraction_backend=none",
            "--override",
            "embedding.model=mock",
            "--override",
            "components.allow_fake=true",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "ingested 1 conversation(s), 3 session(s), 9 turn(s)" in out
    assert "quarantined=1" in out

    conn = sqlite3.connect(tmp_path / "cli.mind")
    count = conn.execute(
        "SELECT COUNT(*) FROM memory_frames WHERE gop_id LIKE '%:session:%'"
    ).fetchone()[0]
    conn.close()
    assert count == 9


def test_cli_ingest_fake_backend_extracts(tmp_path: Path, capsys) -> None:
    rc = main(
        [
            "ingest",
            FIXTURE,
            "--db",
            str(tmp_path / "cli.mind"),
            "--override",
            "write.extraction_backend=fake",
            "--override",
            "embedding.model=mock",
            "--override",
            "components.allow_fake=true",
        ]
    )
    assert rc == 0
    assert "entities=" in capsys.readouterr().out


def test_cli_ingest_missing_file_fails(tmp_path: Path, capsys) -> None:
    rc = main(["ingest", str(tmp_path / "nope.json"), "--db", str(tmp_path / "x.mind")])
    assert rc == 1
    assert "gluomem ingest:" in capsys.readouterr().err


def test_cli_bench_requires_subcommand(capsys) -> None:
    assert main(["bench"]) == 1
    assert "pick a subcommand" in capsys.readouterr().err
