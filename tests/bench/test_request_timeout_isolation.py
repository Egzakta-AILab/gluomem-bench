"""Offline scheduler/transport regression: request failure is not an outage."""
import json
from threading import Event

import httpx
import pytest

from gluomem_bench.bench.parallel_campaign import CampaignPolicy, Job, run_jobs
from gluomem_bench.bench.rack import RackClient, RackProfile
from gluomem_bench.bench.rack_trace import WorkloadBudget, RackLM


@pytest.mark.parametrize("direct_context", [False, True])
@pytest.mark.parametrize("fault", ["discovery", "tokenizer", "identity", "connection"])
def test_shared_admission_failure_cancels_queued_jobs(tmp_path, direct_context, fault):
    cancelled = Event()
    selected = profile()
    dispatched = []
    jobs = [Job(f"j{i}", f"h{i}", tmp_path / f"j{i}", 4, 20) for i in range(3)]

    def run(job, context):
        budget = context if direct_context else WorkloadBudget(
            policy="role-budgets-v6", outer_budget=context)
        def serve(request):
            if request.url.path == "/v1/models":
                if fault == "discovery":
                    raise httpx.ReadTimeout("discovery not available", request=request)
                return httpx.Response(200, json={"data": [{"id": selected.model}]})
            if request.url.path == "/tokenize":
                if fault == "tokenizer":
                    raise httpx.ReadTimeout("tokenizer not available", request=request)
                return httpx.Response(200, json={"count": 20})
            dispatched.append(job.job_id)
            if fault == "connection":
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200, json={"model": "wrong-model", "choices": [
                {"finish_reason": "stop", "message": {"content": "ok"}}]})
        client = RackClient(selected, receipt=job.output / "calls.jsonl", budget=budget,
                            transport=httpx.MockTransport(serve))
        try:
            client.preflight()
            client.generate_messages("answer", [{"role": "user", "content": "authored"}])
        except Exception:
            if context.budget.failed:
                cancelled.set()
            raise
        finally:
            client.close()

    result = run_jobs(jobs, policy=CampaignPolicy(1, 1, 131072, 12, 120, False),
        out=tmp_path / "coordination", run_job=run, cancel=cancelled)
    assert [r["status"] for r in result["jobs"]] == ["failed", "cancelled", "cancelled"]
    expected = 0 if fault in {"discovery", "tokenizer"} else 1
    assert result["accounting"]["calls"] == expected
    assert len(dispatched) == expected


@pytest.mark.parametrize("phase", ["lease", "outer_deadline"])
def test_deadline_origin_is_not_a_dispatched_generation_timeout(tmp_path, phase):
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget, CampaignStopped, JobContext, RequestLimiter)
    now = [0.0]
    campaign = CampaignBudget(max_calls=4, max_seconds=100, clock=lambda: now[0])
    limiter = RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072)
    context = JobContext(Job("j", "h", tmp_path / "j", 4, 10), campaign, limiter)
    budget = WorkloadBudget(policy="role-budgets-v6", outer_budget=context,
                            clock=lambda: now[0])
    client = RackClient(profile(), receipt=tmp_path / "calls.jsonl", budget=budget,
        transport=httpx.MockTransport(lambda request: pytest.fail("no HTTP dispatch admitted")))
    now[0] = 11 if phase == "lease" else 101
    try:
        with pytest.raises(CampaignStopped):
            if phase == "outer_deadline":
                client.check()  # normal generation entry checks the outer guard
            with client.generation_lease(100):
                budget.begin_call()
        if phase == "lease":
            assert context.failed
        else:
            assert budget.failed
        assert campaign.failed is (phase == "outer_deadline")
        assert campaign.calls == budget.calls == 0
        assert not client.receipt.exists()
    finally:
        client.close()


def profile():
    return RackProfile(base_url="http://localhost:8000/v1", model="offline",
        profile_version="role-budgets-v6", generation_budgets=dict(
            extraction=4096, consolidation=4096, reader=4096, judge=4096))


@pytest.mark.parametrize("path", ["plain", "structured", "production"])
def test_timed_out_middle_job_retains_spend_and_next_job_completes(tmp_path, monkeypatch, path):
    import dspy
    from dspy.utils.exceptions import LMTimeoutError

    selected = profile()
    attempts = []
    contexts = {}
    cancelled = Event()
    jobs = [Job(f"j{i}", f"h{i}", tmp_path / f"j{i}", 4, 20) for i in range(3)]

    def run(job, context):
        contexts[job.job_id] = context
        budget = WorkloadBudget(policy="role-budgets-v6", outer_budget=context)
        job.output.mkdir()
        (job.output / "partial.json").write_text('{"retained":true}')

        def serve(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": selected.model}]})
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 20})
            attempts.append(job.job_id)
            if job.job_id == "j1":
                raise httpx.ReadTimeout("authored request deadline", request=request)
            return httpx.Response(200, json={"model": selected.model, "choices": [
                {"finish_reason": "stop", "message": {"content": '{"answer":"ok"}'}}
            ], "usage": {"prompt_tokens": 20, "completion_tokens": 5}})

        client = RackClient(selected, receipt=job.output / "calls.jsonl", budget=budget,
                            transport=httpx.MockTransport(serve))
        trace = job.output / "production.jsonl"
        try:
            client.preflight()
            if path == "production":
                def provider(lm, *args, **kwargs):
                    attempts.append(job.job_id)
                    if job.job_id == "j1":
                        raise LMTimeoutError("authored SDK request deadline")
                    lm.history.append({"response": {"model": selected.model, "choices": [
                        {"finish_reason": "stop", "message": {"content": "ok"}}],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 5}}})
                    return ["ok"]
                monkeypatch.setattr(dspy.LM, "__call__", provider)
                lm = RackLM("openai/" + selected.model, client=client, task="event_claims",
                    trace=trace, budget=budget, api_base=selected.base_url,
                    max_tokens=4096, temperature=0, num_retries=0, cache=False)
                invoke = lambda: lm(messages=[{"role": "user", "content": "authored"}])
            elif path == "structured":
                invoke = lambda: client.complete("answer", {}, {"answer": {"type": "string"}})
            else:
                invoke = lambda: client.generate_messages("answer", [{"role": "user", "content": "authored"}])
            invoke()
            context.check()
            (job.output / "finalized.json").write_text('{"complete":true}')
            return {"quality_status": "complete_verified"}
        except Exception:
            before = {f.name: f.read_bytes() for f in job.output.iterdir()}
            with pytest.raises(Exception):
                invoke()  # no retry, provider dispatch, or trace append after failure
            assert before == {f.name: f.read_bytes() for f in job.output.iterdir()}
            if context.budget.failed:
                cancelled.set()
            raise
        finally:
            client.close()

    result = run_jobs(jobs, policy=CampaignPolicy(1, 1, 131072, 12, 120, False),
                      out=tmp_path / "coordination", run_job=run, cancel=cancelled)
    assert [r["status"] for r in result["jobs"]] == ["complete", "failed", "complete"]
    assert attempts == ["j0", "j1", "j2"]
    assert result["accounting"]["calls"] == 3
    assert [r["calls"] for r in result["jobs"]] == [1, 1, 1]
    assert contexts["j1"].failed
    assert not contexts["j1"].budget.failed
    assert (jobs[1].output / "partial.json").exists()
    assert not (jobs[1].output / "finalized.json").exists()
    saved = json.loads((tmp_path / "coordination/progress.json").read_bytes())
    assert saved == result
    receipt = jobs[1].output / ("production.jsonl" if path == "production" else "calls.jsonl")
    failure = json.loads(receipt.read_text().splitlines()[-1])
    assert "error" in failure
    if path == "production":
        assert failure["prompt_token_count"] == 20
    else:
        assert failure["request"]["max_tokens"] == 4096
