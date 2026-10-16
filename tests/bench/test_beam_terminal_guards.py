"""Offline HTTP-to-artifact regressions; no provider/model execution."""

import hashlib
import json

import httpx
import pytest

from gluomem_bench.bench.beam_eval import run_beam
from gluomem_bench.bench.rack import RackClient, RackProfile
from gluomem_bench.bench.rack_trace import WorkloadBudget
from tests.bench.test_beam import beam_data


@pytest.mark.parametrize("role", ["answer", "rubric"])
@pytest.mark.parametrize("fault", ["length", "context", "tokenizer", "protocol", "http"])
@pytest.mark.parametrize("shared_budget", [False, True])
def test_fatal_http_reader_or_late_judge_never_finalizes(tmp_path, role, fault, shared_budget):
    cfg, data, rows = beam_data(tmp_path)
    probes = json.loads(rows[0]["probing_questions"])
    probes["preference_following"] *= 2
    rows[0]["probing_questions"] = json.dumps(probes)
    data.write_text(json.dumps(rows), encoding="utf-8")
    cfg.benchmark.datasets["beam_128k"].sha256 = hashlib.sha256(data.read_bytes()).hexdigest()
    requests = []
    generations = []
    tokenizations = []

    def serve(request):
        requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": RackProfile(
                                base_url="http://localhost:8000/v1", model="example/served-model"
                            ).model
                        }
                    ]
                },
            )
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            task = "rubric" if "Task: rubric." in body["messages"][0]["content"] else "answer"
            tokenizations.append(task)
            failing = task == role and (role == "answer" or tokenizations.count("rubric") == 3)
            if failing and fault == "context":
                return httpx.Response(200, json={"count": 131000})
            if failing and fault == "tokenizer":
                return httpx.Response(404, text="tokenizer unavailable")
            return httpx.Response(200, json={"count": 20})
        task = body["response_format"]["json_schema"]["name"]
        generations.append(task)
        failing = task == role and (role == "answer" or generations.count("rubric") == 3)
        if failing and fault == "http":
            return httpx.Response(500, text="backend unavailable")
        result = {"answer": "blue"} if task == "answer" else {"score": 1, "reason": "yes"}
        response = {
            "model": "wrong-model"
            if failing and fault == "protocol"
            else RackProfile(
                base_url="http://localhost:8000/v1", model="example/served-model"
            ).model,
            "choices": [
                {
                    "finish_reason": "length" if failing and fault == "length" else "stop",
                    "message": {"content": json.dumps(result)},
                }
            ],
        }
        return httpx.Response(200, json=response)

    out = tmp_path / "run"
    budget = WorkloadBudget() if shared_budget else None
    client = RackClient(
        RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
        receipt=out / "calls.jsonl",
        budget=budget,
        transport=httpx.MockTransport(serve),
    )
    try:
        with pytest.raises((ValueError, RuntimeError, httpx.HTTPError)):
            run_beam(cfg, client=client, out=out, retriever="full_context")
        failure = json.loads((out / "failure.json").read_text())
        assert failure["complete"] is False
        assert failure["predictions"] == (0 if role == "answer" else 2)
        assert failure["judgments"] == (0 if role == "answer" else 2)
        assert not json.loads((out / "ingest.json").read_text())["complete"]
        for name in ("manifest.json", "scores.json", "report.md"):
            assert not (out / name).exists()
        if budget is not None:
            assert budget.failed
        before = len(requests)
        trace = client.receipt.read_bytes()
        for retry in (
            lambda: client.complete("answer", {}, {"answer": {"type": "string"}}),
            client.preflight,
        ):
            with pytest.raises((ValueError, RuntimeError, httpx.HTTPError)):
                retry()
        assert len(requests) == before
        assert client.receipt.read_bytes() == trace
        raw = [json.loads(line) for line in trace.decode().splitlines()]
        if fault == "length":
            assert raw[-1]["response"]["choices"][0]["finish_reason"] == "length"
        if fault in ("tokenizer", "http"):
            assert raw[-1]["status"] == (404 if fault == "tokenizer" else 500)
            assert raw[-1]["response_text"]
    finally:
        client.close()


@pytest.mark.parametrize("content", ["not JSON", '{"answer": 42}', '{"other": "blue"}'])
def test_complete_reader_format_noncompliance_keeps_fixed_denominator(tmp_path, content):
    cfg, _, _ = beam_data(tmp_path)
    budget = WorkloadBudget()
    requests = []

    def serve(request):
        requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": RackProfile(
                                base_url="http://localhost:8000/v1", model="example/served-model"
                            ).model
                        }
                    ]
                },
            )
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 20})
        return httpx.Response(
            200,
            json={
                "model": "example/served-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": content},
                    }
                ],
            },
        )

    out = tmp_path / "run"
    client = RackClient(
        RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
        receipt=out / "calls.jsonl",
        budget=budget,
        transport=httpx.MockTransport(serve),
    )
    try:
        scores = run_beam(cfg, client=client, out=out, retriever="full_context")
        assert scores["questions"] == 1 and scores["nuggets"] == 2
        assert scores["answer_failures"] == 1
        assert scores["rubric_question_mean"] == 0
        assert not budget.failed and budget.calls == 1
        assert (out / "manifest.json").exists()
        prediction = json.loads((out / "predictions.jsonl").read_text())
        assert prediction["failure"].startswith("AnswerFormatError:")
        judgments = [json.loads(x) for x in (out / "judgments.jsonl").read_text().splitlines()]
        assert all(j["failure"] == prediction["failure"] and j["score"] == 0 for j in judgments)
    finally:
        client.close()


def test_invalid_judge_output_is_not_a_zero_quality_measurement(tmp_path):
    cfg, _, _ = beam_data(tmp_path)
    budget = WorkloadBudget()

    def serve(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": RackProfile(
                                base_url="http://localhost:8000/v1", model="example/served-model"
                            ).model
                        }
                    ]
                },
            )
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 20})
        task = json.loads(request.content)["response_format"]["json_schema"]["name"]
        result = {"answer": "blue"} if task == "answer" else {"score": 9, "reason": "invalid"}
        return httpx.Response(
            200,
            json={
                "model": "example/served-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(result)},
                    }
                ],
            },
        )

    out = tmp_path / "run"
    client = RackClient(
        RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
        receipt=out / "calls.jsonl",
        budget=budget,
        transport=httpx.MockTransport(serve),
    )
    try:
        with pytest.raises(ValueError, match="invalid judge response"):
            run_beam(cfg, client=client, out=out, retriever="full_context")
        assert budget.failed
        failure = json.loads((out / "failure.json").read_text())
        assert failure["predictions"] == 1 and failure["judgments"] == 0
        assert not (out / "manifest.json").exists()
        assert not (out / "scores.json").exists()
    finally:
        client.close()
