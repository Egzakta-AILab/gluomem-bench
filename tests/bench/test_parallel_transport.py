"""Real transport seams must hold the shared request/token lease at dispatch."""

from contextlib import contextmanager

import httpx
import pytest


@pytest.mark.parametrize("path", ["plain", "structured", "production"])
def test_all_generation_paths_hold_outer_lease_before_count_and_dispatch(
    tmp_path, monkeypatch, path
):
    import dspy

    from gluomem_bench.bench.rack import RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import RackLM, WorkloadBudget

    class Outer:
        def __init__(self):
            self.active = False
            self.calls = 0
            self.leases = []

        def check(self):
            pass

        def remaining(self):
            return 60

        def begin_call(self):
            assert self.active, "generation charged/dispatched outside the shared lease"
            self.calls += 1

        @contextmanager
        def lease(self, tokens):
            self.leases.append(tokens)
            self.active = True
            try:
                yield
            finally:
                self.active = False

    outer = Outer()
    budget = WorkloadBudget(outer_budget=outer)
    profile = RackProfile(base_url="http://localhost:8000/v1", model="example/served-model")

    def serve(request):
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 20})
        assert outer.active
        return httpx.Response(200, json={"model": profile.model, "choices": [
            {"finish_reason": "stop", "message": {"content": '{"answer":"ready"}'}}
        ], "usage": {"prompt_tokens": 20, "completion_tokens": 5}})

    client = RackClient(profile, receipt=tmp_path / "calls.jsonl", budget=budget,
                        transport=httpx.MockTransport(serve))
    try:
        if path == "plain":
            client.generate_messages("answer", [{"role": "user", "content": "authored"}])
        elif path == "structured":
            client.complete("answer", {"question": "authored"},
                            {"answer": {"type": "string"}})
        else:
            def provider(lm, *args, **kwargs):
                assert outer.active
                lm.history.append({"response": {"model": profile.model, "choices": [
                    {"finish_reason": "stop", "message": {"content": "ready"}}]}})
                return ["ready"]

            monkeypatch.setattr(dspy.LM, "__call__", provider)
            lm = RackLM("openai/" + profile.model, client=client, task="event_claims",
                        trace=tmp_path / "production.jsonl", budget=budget,
                        api_base=profile.base_url, max_tokens=512, temperature=0,
                        num_retries=0, cache=False)
            lm(messages=[{"role": "user", "content": "authored"}])
        assert outer.calls == budget.calls == 1
        assert outer.leases == [20 + 512 + 2048]
        assert not outer.active
    finally:
        client.close()
