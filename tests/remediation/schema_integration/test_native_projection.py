"""Authored native-response projections, never synthetic API return values."""
from copy import deepcopy

import pytest
from dspy.lm15 import Message, RefusalPart, Response, Usage
from gluomem_bench.bench import rack_schema


def response(**kwargs):
    return Response(id="authored", model="fixture/slash/model", message=Message.assistant('{"ok":true}'),
                    finish_reason="stop", usage=Usage(input_tokens=17, output_tokens=9, total_tokens=26), **kwargs)


def test_projection_retains_original_native_evidence():
    assert callable(getattr(rack_schema, "project_native_response", None)), "native evidence support missing"
    view, evidence = rack_schema.project_native_response(response())
    assert evidence["protocol"] == "lm15-native-response-projection@v1"
    assert evidence["canonical"]["model"] == view["model"] == "fixture/slash/model"
    assert view["choices"] == [{"message": {"role": "assistant", "content": '{"ok":true}'}, "finish_reason": "stop"}]
    assert view["usage"] == {"prompt_tokens": 17, "completion_tokens": 9, "total_tokens": 26}
    rack_schema.validate_native_projection(view, evidence)
    forged = deepcopy(view)
    forged["usage"]["prompt_tokens"] = 0
    with pytest.raises(ValueError):
        rack_schema.validate_native_projection(forged, evidence)


def test_missing_usage_remains_unknown_and_missing_text_refuses():
    assert callable(getattr(rack_schema, "project_native_response", None)), "native evidence support missing"
    raw = Response(id=None, model="fixture/slash/model", message=Message.assistant("x"),
                   finish_reason="stop", usage=Usage())
    view, evidence = rack_schema.project_native_response(raw)
    assert all(value is None for value in view["usage"].values())
    rack_schema.validate_native_projection(view, evidence)
    empty = Response(id=None, model="fixture/slash/model", message=Message(role="assistant", parts=(RefusalPart("declined"),)),
                     finish_reason="stop", usage=Usage())
    with pytest.raises(ValueError):
        rack_schema.project_native_response(empty)
