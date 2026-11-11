"""Fresh-core config compatibility never supplies an ambient commercial model."""

import os

import pytest
from pydantic import ValidationError


def test_benchmark_import_defaults_leave_the_model_required(tmp_path, monkeypatch):
    from gluomem_bench.config import _BENCH_PYDANTIC_DEFAULTS, AppConfig, load_config

    for key in tuple(os.environ):
        if key.startswith("GLUOMEM_"):
            monkeypatch.delenv(key)
    assert "model" not in _BENCH_PYDANTIC_DEFAULTS["dspy"]
    with pytest.raises(ValidationError, match="model|dspy"):
        AppConfig()
    with pytest.raises(ValidationError, match="model|dspy"):
        load_config(env_file=tmp_path / "absent.env")
    cfg = load_config(overrides={"dspy.model": "mock"}, env_file=tmp_path / "absent.env")
    assert cfg.dspy.model == "mock"
    assert cfg.dspy.tasks["answer"].model is None


def test_only_new_provider_conditions_explicitly_capture_attributed_interaction(tmp_path):
    from gluomem_bench.bench.rack import RackProfile, controlled_config

    common = dict(data=tmp_path / "unused.json", sha256="0" * 64)
    legacy = RackProfile(base_url="https://rack.example.com/v1", model="legacy")
    assert controlled_config(legacy, **common).write.source_capture_policy is None
    profile = RackProfile(
        profile_version="provider-roles-v8", provider="qwen-vllm",
        requested_model="qwen-3.8-27b", model="Qwen/Qwen3.8-27B-FP8",
        base_url="https://rack.example.com/v1", context_window=131072,
        api_key_env="GLUOMEM_RACK_API_KEY", token_count_policy="served-tokenizer",
        generation_budgets=dict(extraction=4096, consolidation=2048, reader=1024, judge=1024),
    )
    cfg = controlled_config(profile, **common)
    assert cfg.write.source_capture_policy == "attributed-interaction"
    assert (
        cfg.with_overrides({"benchmark.answer.context_k": 5}).write.source_capture_policy
        == "attributed-interaction"
    )
