import copy

import pytest

from gluomem.llm.signatures import ClaimConsolidation
from gluomem.llm.structured_adapter import output_schema
from gluomem_bench.bench import rack_schema
from gluomem_bench.bench.rack import RackProfile


def case():
    profile = RackProfile(base_url="http://fixture.invalid/v1", model="fixture/slash/model")
    cfg = {"dspy": {"adapter": "json", "schema_mode": "required", "schema_strict": False}}
    binding = rack_schema.schema_binding(profile, cfg["dspy"])
    row = {
        "task": "consolidate",
        "model": binding["lm_model"],
        "generation_kwargs": {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "GluoMemOutput",
                    "strict": False,
                    "schema": output_schema(ClaimConsolidation),
                },
            }
        },
    }
    return row, profile, cfg, {"rack_schema_transport": binding}, {"rack_schema_transport": binding}


def test_bound_required_production_identity_and_exact_schema():
    args = case()
    assert rack_schema.validate_schema_row(*args) == args[0]["model"]


@pytest.mark.parametrize(
    "forgery", ["unbound", "prefix", "case", "capability", "strict", "schema", "config"]
)
def test_schema_evidence_refuses_self_declarations_and_schema_forgeries(forgery):
    row, profile, cfg, protocol, expected = copy.deepcopy(case())
    if forgery == "unbound":
        expected = None
    elif forgery == "prefix":
        row["model"] = "arbitrary/" + profile.model
    elif forgery == "case":
        row["model"] = row["model"].upper()
    elif forgery == "capability":
        protocol["rack_schema_transport"]["transport"] = "invented"
    elif forgery == "strict":
        row["generation_kwargs"]["response_format"]["json_schema"]["strict"] = True
    elif forgery == "schema":
        row["generation_kwargs"]["response_format"]["json_schema"]["schema"]["$defs"].clear()
    else:
        cfg["dspy"]["schema_strict"] = True
    with pytest.raises(ValueError):
        rack_schema.validate_schema_row(row, profile, cfg, protocol, expected)
