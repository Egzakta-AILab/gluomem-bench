import pytest

from gluomem_bench.bench.rack import RackClient, RackProfile
from gluomem_bench.bench.rack_schema import schema_binding
from gluomem_bench.bench.rack_trace import RackLM, WorkloadBudget


def test_constructor_refuses_forged_declared_alias(tmp_path):
    profile = RackProfile(base_url="http://fixture.invalid/v1", model="fixture/slash/model")
    client = RackClient(profile, receipt=tmp_path / "unused")
    declaration = schema_binding(
        profile, {"adapter": "json", "schema_mode": "required", "schema_strict": False}
    )
    declaration["lm_model"] = "arbitrary/" + profile.model
    try:
        with pytest.raises(ValueError, match="binding"):
            RackLM(
                declaration["lm_model"],
                client=client,
                task="event_claims",
                trace=tmp_path / "raw",
                budget=WorkloadBudget(),
                schema_binding=declaration,
                engine="lm15",
                api_base=profile.base_url,
            )
    finally:
        client.close()


@pytest.mark.parametrize(
    "override",
    [
        {"models.rack-qwen.provider_model": "openai/wrong/model"},
        {"models.rack-qwen.api_base": "http://other.invalid/v1"},
        {"dspy.tasks.event_claims.temperature": 1},
        {"dspy.tasks.event_claims.max_tokens": 128},
        {"models.rack-qwen.extra.extra_body.chat_template_kwargs.enable_thinking": True},
    ],
)
def test_router_refuses_source_configuration_forgeries(tmp_path, override):
    from gluomem_bench.bench.rack import controlled_config
    from gluomem_bench.bench.rack_trace import RackRouter

    profile = RackProfile(base_url="http://fixture.invalid/v1", model="fixture/slash/model")
    cfg = controlled_config(profile, data=tmp_path / "unused.json", sha256="a" * 64)
    cfg = cfg.with_overrides({"dspy.schema_mode": "required", **override})
    client = RackClient(profile, receipt=tmp_path / "unused")
    try:
        router = RackRouter(
            cfg,
            client=client,
            trace=tmp_path / "raw",
            budget=WorkloadBudget(),
            schema_transport="declared-openai-schema-v1",
        )
        with pytest.raises(ValueError, match="mismatch"):
            router.lm_for("event_claims")
    finally:
        client.close()


def test_unknown_metadata_remains_unsupported_without_explicit_opt_in(tmp_path):
    from gluomem.llm.signatures import EventClaimExtraction
    from gluomem_bench.bench.rack import controlled_config
    from gluomem_bench.bench.rack_trace import RackRouter

    profile = RackProfile(base_url="http://fixture.invalid/v1", model="fixture/slash/model")
    cfg = controlled_config(profile, data=tmp_path / "unused.json", sha256="a" * 64)
    cfg = cfg.with_overrides({"dspy.schema_mode": "required"})
    client = RackClient(profile, receipt=tmp_path / "unused")
    try:
        router = RackRouter(cfg, client=client, trace=tmp_path / "raw", budget=WorkloadBudget())
        lm = router.lm_for("event_claims")
        with pytest.raises(ValueError, match="capability required"):
            router._adapter._prepare_response_format(lm, {}, EventClaimExtraction)
        assert lm.model == "openai/" + profile.model
    finally:
        client.close()
