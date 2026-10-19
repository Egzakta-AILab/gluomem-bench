import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Literal

import dspy
import pytest
from pydantic import BaseModel, ConfigDict, create_model

from gluomem_bench.bench.rack import RackClient, RackProfile, controlled_config
from gluomem_bench.bench.rack_trace import RackRouter, WorkloadBudget


class Entity(BaseModel):
    kind: Literal["person", "place"]
    properties: dict[str, str]


class Extract(dspy.Signature):
    text: str = dspy.InputField()
    entities: list[Entity] = dspy.OutputField()


@pytest.fixture
def wire(tmp_path):
    import __main__

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            if self.path == "/tokenize":
                response = {"count": 20}
            else:
                assert self.path == "/v1/chat/completions"
                response = {
                    "id": "owned-fixture",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "fixture/slash/model",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "entities": [
                                            {"kind": "person", "properties": {"Name": "Ada"}}
                                        ]
                                    }
                                ),
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32},
                }
            encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    __main__.allowed.add(server.server_address)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    profile = RackProfile(
        base_url=f"http://127.0.0.1:{server.server_port}/v1", model="fixture/slash/model"
    )
    cfg = controlled_config(profile, data=tmp_path / "unused.json", sha256="a" * 64)
    cfg = cfg.with_overrides({"dspy.schema_mode": "required", "dspy.schema_strict": False})
    client = RackClient(profile, receipt=tmp_path / "http.jsonl")
    budget = WorkloadBudget()
    try:
        yield profile, cfg, client, budget, requests
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join()
        __main__.allowed.remove(server.server_address)


@pytest.mark.parametrize("strict", [False, True])
def test_required_router_serializes_exact_nested_schema(wire, tmp_path, strict):
    profile, cfg, client, budget, requests = wire
    cfg = cfg.with_overrides({"dspy.schema_strict": strict})
    router = RackRouter(
        cfg,
        client=client,
        trace=tmp_path / "raw.jsonl",
        budget=budget,
        schema_transport="declared-openai-schema-v1",
    )
    lm = router.lm_for("event_claims")
    with dspy.context(lm=lm, adapter=router._adapter):
        result = dspy.Predict(Extract)(text="Ada")
    assert result.entities == [Entity(kind="person", properties={"Name": "Ada"})]
    assert lm.engine == "lm15" and lm.num_retries == 0
    assert lm.model != "openai/" + profile.model
    assert budget.calls == 1
    assert [path for path, body in requests] == ["/tokenize", "/v1/chat/completions"]
    body = requests[-1][1]
    expected_model = create_model(
        "GluoMemOutput", __config__=ConfigDict(extra="forbid"), entities=(list[Entity], ...)
    )
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "GluoMemOutput",
            "strict": strict,
            "schema": expected_model.model_json_schema(),
        },
    }
    assert body["model"] == profile.model
    assert body["temperature"] == 0 and body["max_tokens"] == profile.output_budget("event_claims")
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    row = json.loads((tmp_path / "raw.jsonl").read_text())
    assert row["model"] == lm.model
    assert row["generation_kwargs"]["response_format"] == body["response_format"]
