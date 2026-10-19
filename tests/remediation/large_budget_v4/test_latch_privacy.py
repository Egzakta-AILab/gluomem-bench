"""Current categorical privacy contract; original legacy-text test is retained."""
import pytest
import dspy
import httpx
from gluomem_bench.bench.rack import FatalRackError, RackClient, RackProfile
from gluomem_bench.bench.rack_trace import RackLM, WorkloadBudget


def test_latched_client_refuses_before_tokenizer_trace_and_provider_without_private_text(tmp_path, monkeypatch):
    requests, calls = [], []
    monkeypatch.setattr(dspy.LM, '__call__', lambda *a, **kw: calls.append(True))
    profile = RackProfile(base_url='http://authored.invalid/v1', model='authored/model')
    client = RackClient(profile, receipt=tmp_path/'http.jsonl', transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200)))
    trace = tmp_path/'raw.jsonl'
    lm = RackLM('openai/'+profile.model, client=client, task='event_claims', trace=trace,
        budget=WorkloadBudget(), api_base=profile.base_url, max_tokens=512,
        temperature=0, num_retries=0, cache=False)
    client._latch(FatalRackError('AUTHORED_PRIVATE_FAILURE_SENTINEL'))
    try:
        with pytest.raises(FatalRackError) as caught:
            lm(messages=[{'role':'user','content':'AUTHORED_SOURCE'}])
        assert 'AUTHORED_PRIVATE_FAILURE_SENTINEL' not in str(caught.value)
        assert requests == calls == []
        assert not trace.exists()
    finally:
        client.close()
