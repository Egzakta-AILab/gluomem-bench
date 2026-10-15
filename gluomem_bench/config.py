"""Typed ``AppConfig`` for ``gluomem_bench`` extending Core ``gluomem.config.AppConfig``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import dotenv
from gluomem.config.app_config import (
    ABSTAIN_SCORES,
    AbstainConfig,
    ChunkingConfig,
    CompactionConfig,
    ComponentsConfig,
    ContradictionConfig,
    DspyArtifactsConfig,
    DspyCacheConfig,
    DspyConfig,
    DspyOptimizerConfig,
    DspyTaskConfig,
    EmbeddingConfig,
    KgConfig,
    LaneConfig,
    LanesConfig,
    LifecycleConfig,
    ModelEntry,
    ProfileWeights,
    RawTurnsConfig,
    RerankConfig,
    SafetyConfig,
    ScannerConfig,
    ScannerPattern,
    ScoringConfig,
    ScoringProfile,
    SearchConfig,
    StorageConfig,
    SubjectConfig,
    WriteCognifyConfig,
    WriteConfig,
    WriteDedupConfig,
    WriteExtractionCacheConfig,
    _default_dspy_tasks,
    _default_models,
    _PydanticDefaultsSource,
    dotted_to_nested,
    reranker_calibration_id,
)
from gluomem.config.app_config import (
    AppConfig as CoreAppConfig,
)
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

_BENCH_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_bench_dspy_tasks() -> dict[str, DspyTaskConfig]:
    tasks = _default_dspy_tasks()
    tasks.setdefault(
        "answer",
        DspyTaskConfig(
            model="deepseek-flash",
            module="predict",
            temperature=0.0,
            max_tokens=1024,
        ),
    )
    return tasks


def _default_bench_dspy_config() -> DspyConfig:
    return DspyConfig(tasks=_default_bench_dspy_tasks())


class BenchmarkJudgeConfig(BaseModel):
    family: str = "trio_strict"
    backbones: list[str] = Field(default_factory=lambda: ["haiku", "gpt5-mini", "qwen3"])
    also_record_single: bool = True
    backend: str = "dspy"


class BenchmarkAnswerConfig(BaseModel):
    task: str = "answer"
    backend: str = "dspy"
    context_k: int = Field(default=50, ge=1)
    abstain_text: str = "I don't know."
    style: str = "preference_aware"

    @model_validator(mode="after")
    def _check_style(self) -> BenchmarkAnswerConfig:
        if self.style not in ("base", "preference_aware"):
            raise ValueError(
                f"benchmark.answer.style must be 'base' or 'preference_aware', got {self.style!r}"
            )
        return self


class BenchmarkRetrievalConfig(BaseModel):
    ks: list[int] = Field(default_factory=lambda: [5, 10])


class BenchmarkSampleConfig(BaseModel):
    per_category: int = Field(default=80, ge=1)
    categories: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])
    seed: int = 42
    manifest: str = "benchmarks_data/locomo_sample_seed42.json"


class BenchmarkDatasetEntry(BaseModel):
    url: str
    sha256: str
    file: str


def _default_benchmark_datasets() -> dict[str, BenchmarkDatasetEntry]:
    return {
        "locomo": BenchmarkDatasetEntry(
            url="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
            sha256="79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
            file="locomo10.json",
        ),
        "longmemeval_oracle": BenchmarkDatasetEntry(
            url="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
            sha256="821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c",
            file="longmemeval_oracle.json",
        ),
        "longmemeval_s": BenchmarkDatasetEntry(
            url="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
            sha256="d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
            file="longmemeval_s_cleaned.json",
        ),
    }


class BenchmarkFullContextConfig(BaseModel):
    max_chars: int = Field(default=120_000, ge=1)
    keep: str = "tail"


class BenchmarkMem0Config(BaseModel):
    model: str = "deepseek-flash"
    embedder_model: str = "minilm"
    top_k: int | None = Field(default=None, ge=1)
    infer: bool = True


class BenchmarkCiGateConfig(BaseModel):
    metric: str = "evidence_hit@5"
    min: float = 0.7


class BenchmarkConfig(BaseModel):
    judge: BenchmarkJudgeConfig = Field(default_factory=BenchmarkJudgeConfig)
    answer: BenchmarkAnswerConfig = Field(default_factory=BenchmarkAnswerConfig)
    retrieval: BenchmarkRetrievalConfig = Field(default_factory=BenchmarkRetrievalConfig)
    sample: BenchmarkSampleConfig = Field(default_factory=BenchmarkSampleConfig)
    datasets: dict[str, BenchmarkDatasetEntry] = Field(default_factory=_default_benchmark_datasets)
    datasets_dir: str = "benchmarks_data"
    full_context: BenchmarkFullContextConfig = Field(default_factory=BenchmarkFullContextConfig)
    run_dir: str = "experiments"
    parallelism: int = Field(default=8, ge=1)
    mem0: BenchmarkMem0Config = Field(default_factory=BenchmarkMem0Config)
    ci_gate: BenchmarkCiGateConfig = Field(default_factory=BenchmarkCiGateConfig)


_BENCH_PYDANTIC_DEFAULTS: dict[str, Any] = {
    "models": {k: v.model_dump() for k, v in _default_models().items()},
    "storage": StorageConfig().model_dump(),
    "embedding": EmbeddingConfig().model_dump(),
    "components": ComponentsConfig().model_dump(),
    "chunking": ChunkingConfig().model_dump(),
    "lifecycle": LifecycleConfig().model_dump(),
    "search": SearchConfig().model_dump(),
    "scoring": ScoringConfig().model_dump(),
    "kg": KgConfig().model_dump(),
    "safety": SafetyConfig().model_dump(),
    "write": WriteConfig().model_dump(),
    "dspy": _default_bench_dspy_config().model_dump(),
    "benchmark": BenchmarkConfig().model_dump(),
}


class AppConfig(CoreAppConfig):
    """Benchmark-aware ``AppConfig`` extending Core ``gluomem.config.AppConfig``."""

    dspy: DspyConfig = Field(default_factory=_default_bench_dspy_config)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            _PydanticDefaultsSource(settings_cls, _BENCH_PYDANTIC_DEFAULTS),
            dotenv_settings,
            file_secret_settings,
        )

    @model_validator(mode="after")
    def _validate_benchmark_refs(self) -> AppConfig:
        judge = self.benchmark.judge
        for name in judge.backbones:
            if name not in self.models:
                known = ", ".join(sorted(self.models)) or "(none)"
                raise ValueError(
                    f"benchmark.judge.backbones entry '{name}' not in model registry: {known}"
                )
        if len(set(judge.backbones)) != len(judge.backbones) or not judge.backbones:
            raise ValueError(
                f"benchmark.judge.backbones must be non-empty and unique, got: {judge.backbones}"
            )
        families = {"trio_strict", "single"}
        if judge.family not in families:
            raise ValueError(
                f"benchmark.judge.family '{judge.family}' invalid; must be one "
                f"of: {sorted(families)}"
            )
        if judge.family == "trio_strict" and len(judge.backbones) != 3:
            raise ValueError(
                f"benchmark.judge.family 'trio_strict' requires exactly 3 "
                f"backbones, got {len(judge.backbones)}"
            )
        backends = {"dspy", "fake"}
        for where, backend in (
            ("benchmark.judge.backend", judge.backend),
            ("benchmark.answer.backend", self.benchmark.answer.backend),
        ):
            if backend not in backends:
                raise ValueError(f"{where} '{backend}' invalid; must be one of: {sorted(backends)}")
        if self.benchmark.answer.task not in self.dspy.tasks:
            raise ValueError(
                f"benchmark.answer.task '{self.benchmark.answer.task}' not in "
                f"dspy.tasks: {sorted(self.dspy.tasks)}"
            )
        keeps = {"head", "tail"}
        if self.benchmark.full_context.keep not in keeps:
            raise ValueError(
                f"benchmark.full_context.keep "
                f"'{self.benchmark.full_context.keep}' invalid; must be one "
                f"of: {sorted(keeps)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_benchmark_mem0(self) -> AppConfig:
        mem0 = self.benchmark.mem0
        for where, name in (
            ("benchmark.mem0.model", mem0.model),
            ("benchmark.mem0.embedder_model", mem0.embedder_model),
        ):
            if name not in self.models:
                known = ", ".join(sorted(self.models)) or "(none)"
                raise ValueError(f"{where} '{name}' not in model registry: {known}")
        embedder = self.models[mem0.embedder_model]
        if embedder.dim is None:
            raise ValueError(
                f"benchmark.mem0.embedder_model '{mem0.embedder_model}' has "
                f"no dim in the registry; faiss needs embedding_model_dims"
            )
        return self

    @model_validator(mode="after")
    def _validate_benchmark_sample(self) -> AppConfig:
        cats = self.benchmark.sample.categories
        if not cats or len(set(cats)) != len(cats):
            raise ValueError(
                f"benchmark.sample.categories must be non-empty and unique, got: {cats}"
            )
        if not all(1 <= c <= 7 for c in cats):
            raise ValueError(f"benchmark.sample.categories must be within 1..7, got: {cats}")
        return self

    @model_validator(mode="after")
    def _validate_ci_gate(self) -> AppConfig:
        gate = self.benchmark.ci_gate
        m = re.fullmatch(
            r"qa_accuracy|(?:precision|evidence_recall|evidence_hit)@(\d+)",
            gate.metric,
        )
        if m is None:
            raise ValueError(
                f"benchmark.ci_gate.metric '{gate.metric}' invalid; must be "
                f"'qa_accuracy' or one of precision|evidence_recall|"
                f"evidence_hit '@k'"
            )
        if m.group(1) is not None:
            k = int(m.group(1))
            if k not in self.benchmark.retrieval.ks:
                raise ValueError(
                    f"benchmark.ci_gate.metric '{gate.metric}' uses k={k} "
                    f"which is not computed (benchmark.retrieval.ks="
                    f"{self.benchmark.retrieval.ks})"
                )
        if not 0.0 <= gate.min <= 1.0:
            raise ValueError(f"benchmark.ci_gate.min {gate.min} must be within [0, 1]")
        return self

    def with_overrides(self, overrides: dict[str, Any]) -> AppConfig:
        """Return a new validated ``AppConfig`` with dotted-key overrides on top."""
        merged = {**self._load_overrides, **overrides}
        return load_config(overrides=merged)


BenchAppConfig = AppConfig


def load_config(
    overrides: dict[str, Any] | None = None,
    env_file: str | Path | None = None,
) -> AppConfig:
    """Load the typed benchmark ``AppConfig`` from Pydantic defaults + env + overrides."""
    dotenv_path = Path(env_file) if env_file is not None else _BENCH_PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        dotenv.load_dotenv(dotenv_path, override=False)

    init_data = dotted_to_nested(overrides or {})
    cfg = AppConfig(**init_data)
    cfg._load_overrides = dict(overrides or {})
    return cfg


__all__ = [
    "ABSTAIN_SCORES",
    "AbstainConfig",
    "AppConfig",
    "BenchAppConfig",
    "BenchmarkAnswerConfig",
    "BenchmarkCiGateConfig",
    "BenchmarkConfig",
    "BenchmarkDatasetEntry",
    "BenchmarkFullContextConfig",
    "BenchmarkJudgeConfig",
    "BenchmarkMem0Config",
    "BenchmarkRetrievalConfig",
    "BenchmarkSampleConfig",
    "ChunkingConfig",
    "CompactionConfig",
    "ComponentsConfig",
    "ContradictionConfig",
    "CoreAppConfig",
    "DspyArtifactsConfig",
    "DspyCacheConfig",
    "DspyConfig",
    "DspyOptimizerConfig",
    "DspyTaskConfig",
    "EmbeddingConfig",
    "KgConfig",
    "LaneConfig",
    "LanesConfig",
    "LifecycleConfig",
    "ModelEntry",
    "ProfileWeights",
    "RawTurnsConfig",
    "RerankConfig",
    "SafetyConfig",
    "ScannerConfig",
    "ScannerPattern",
    "ScoringConfig",
    "ScoringProfile",
    "SearchConfig",
    "StorageConfig",
    "SubjectConfig",
    "WriteCognifyConfig",
    "WriteConfig",
    "WriteDedupConfig",
    "WriteExtractionCacheConfig",
    "dotted_to_nested",
    "load_config",
    "reranker_calibration_id",
]
