"""Exact-message reader/judge transport must retain fail-closed rack guards."""

import json

import httpx
import pytest

from gluomem_bench.bench.rack import FatalRackError, RackClient, RackProfile


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_exact_message_transport_keeps_prompts_and_latches_incomplete(tmp_path, finish_reason):
    requests = []
    messages = [{"role": "user", "content": "Return yes or no only."}]

    def serve(request):
        requests.append(request)
        return httpx.Response(200, json={"count": 10} if request.url.path == "/tokenize" else {
            "model": "example/served-model",
            "choices": [{"finish_reason": finish_reason, "message": {"content": "yes"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        })

    client = RackClient(
        RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
        receipt=tmp_path / "calls.jsonl", transport=httpx.MockTransport(serve),
    )
    try:
        assert hasattr(client, "generate_messages"), "exact upstream prompts need guarded transport"
        if finish_reason == "stop":
            result = client.generate_messages("rubric", messages)
            assert result["text"] == "yes"
            assert result["prompt_tokens"] == 10
            payload = json.loads(requests[-1].content)
            assert payload["messages"] == messages
            assert "response_format" not in payload
        else:
            with pytest.raises(FatalRackError, match="incomplete"):
                client.generate_messages("rubric", messages)
            n = len(requests)
            with pytest.raises(FatalRackError):
                client.generate_messages("rubric", messages)
            assert len(requests) == n
    finally:
        client.close()
