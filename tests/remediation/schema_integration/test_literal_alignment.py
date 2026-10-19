from datetime import datetime
from typing import Literal

import dspy
from pydantic import BaseModel

from gluomem.llm.structured_adapter import make_adapter


class Decision(BaseModel):
    action: Literal["ADD", "UPDATE"] = "ADD"
    single: Literal["fixed"] = "fixed"
    properties: dict[str, str] = {}
    when: datetime | None = None
    nullable: Literal["one", "two"] | None = None


class Output(dspy.Signature):
    text: str = dspy.InputField()
    decision: Decision = dspy.OutputField()


class Capable:
    supported_params = ["response_format"]
    supports_response_schema = True


def test_default_literals_required_recursively_without_closing_maps():
    adapter = make_adapter("json", canonicalize=True, schema_mode="required")
    options = {}
    adapter._prepare_response_format(Capable(), options, Output)
    schema = options["response_format"]["json_schema"]["schema"]
    decision = schema["$defs"]["Decision"]
    assert decision["required"] == ["action", "single"]
    assert decision["properties"]["properties"]["additionalProperties"] == {"type": "string"}
    assert decision["properties"]["when"]["default"] is None
    assert decision["properties"]["nullable"]["default"] is None
    assert Decision.model_json_schema().get("required", []) == []


def test_schema_data_defaults_are_not_reinterpreted_as_schemas():
    class WithDefault(BaseModel):
        properties: dict = {"properties": {"action": {"enum": ["ADD"]}}}

    class Signature(dspy.Signature):
        result: WithDefault = dspy.OutputField()

    from gluomem.llm.structured_adapter import output_schema

    schema = output_schema(Signature)
    actual = schema["$defs"]["WithDefault"]["properties"]["properties"]["default"]
    assert actual == WithDefault.model_fields["properties"].default
