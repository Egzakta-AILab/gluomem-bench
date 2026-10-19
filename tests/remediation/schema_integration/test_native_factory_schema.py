"""Exercise the actual suite router factory through the native SDK wire."""
import json

import dspy
from gluomem_bench.bench import rack_schema
from test_schema_wire import Entity, Extract, wire  # noqa: F401


def test_suite_factory_selects_and_binds_required_transport(wire, tmp_path):  # noqa: F811 - imported pytest fixture
    profile, cfg, client, budget, requests = wire
    assert callable(getattr(rack_schema, "make_router", None)), "suite schema factory missing"
    router = rack_schema.make_router(cfg, client=client, trace=tmp_path / "native.jsonl", budget=budget)
    expected = rack_schema.production_expectation("gluomem", profile, cfg)
    assert expected == {"rack_schema_transport": rack_schema.schema_binding(profile, cfg.dspy)}
    with dspy.context(lm=router.lm_for("event_claims"), adapter=router._adapter):
        result = dspy.Predict(Extract)(text="Ada")
    assert result.entities == [Entity(kind="person", properties={"Name": "Ada"})]
    row = json.loads((tmp_path / "native.jsonl").read_bytes())
    rack_schema.validate_native_projection(row["response"], row["native_response"])
    assert requests[-1][1]["response_format"]["type"] == "json_schema"
    assert budget.calls == 1
    assert rack_schema.production_expectation("mem0", profile, cfg) == {}
    legacy = cfg.with_overrides({"dspy.schema_mode": "auto"})
    assert rack_schema.production_expectation("gluomem", profile, legacy) == {}
    assert rack_schema.make_router(legacy, client=client, trace=tmp_path / "legacy.jsonl", budget=budget).schema_binding is None
