"""Authored offline graph slice contracts; never benchmark QA inference."""

import importlib
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def test_document_factory_preserves_explicit_common_evidence(tmp_path, bench_cfg):
    from gluomem_bench.bench.cognee_retriever import CogneeCampaignRetriever, CogneeRetrieverFactory
    from gluomem_bench.bench.comparison_backends import CogneeSettings, Evidence
    from gluomem_bench.bench.rack import RackProfile

    settings = CogneeSettings(
        python_executable="unused",
        root=tmp_path / "graph/native",
        dataset="docs",
        llm_url="http://unused.invalid/v1",
        llm_model="served/qwen",
        llm_api_key="local",
        embedding_url="http://unused.invalid/v1",
        embedding_model="encoder",
        embedding_dimensions=768,
        embedding_api_key="local",
    )
    docs = (Evidence("document-000001", "  Complete authored document.\n", {"ordinal": 1}),)
    factory = CogneeRetrieverFactory(
        settings=settings,
        profile=RackProfile(base_url="http://unused.invalid/v1", model="served/qwen"),
        outer_budget=NS(check=lambda: None),
        out=tmp_path / "graph",
        documents=docs,
    )
    retriever = CogneeCampaignRetriever(factory, bench_cfg, None, lambda *a: None)
    assert retriever.documents == docs
    retriever.runner.close()
    assert not (tmp_path / "graph").exists()


@pytest.mark.parametrize("execution", ["factory", "driver", "budget", "finalization"])
def test_real_graph_factory_common_reader_network_boundary(
    tmp_path, bench_cfg, monkeypatch, execution
):
    """Real isolated Cognee engine; only authored network responses are replaced."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import httpx

    from gluomem_bench.bench.comparison_backends import CogneeSettings
    from gluomem_bench.bench.rack import RackProfile
    from gluomem_bench.bench.rack_trace import WorkloadBudget
    from gluomem_bench.bench.types import Sample

    native = (
        Path(__file__).resolve().parents[2]
        / 'tests/fixtures/native/operator-supplied'
    )
    if not native.is_file():
        pytest.skip("isolated Cognee 1.6.5 unavailable")
    calls, embedded, readers = [], [], []
    graph = {
        "nodes": [
            {
                "id": "rhea",
                "name": "Rhea",
                "type": "Person",
                "description": "Rhea owns a cobalt telescope.",
            },
            {
                "id": "scope",
                "name": "cobalt telescope",
                "type": "Object",
                "description": "Rhea's telescope.",
            },
        ],
        "edges": [
            {
                "source_node_id": "rhea",
                "target_node_id": "scope",
                "relationship_name": "owns",
                "description": "Rhea owns a cobalt telescope.",
            }
        ],
    }

    def upstream(request):
        body = json.loads(request.content) if request.content else {}
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"data": [{"id": "served/qwen", "max_model_len": 131072}]}
            )
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 32})
        from gluomem_bench.bench.suite_campaign import READER_INSTRUCTION

        if body["messages"][0] == {"role": "system", "content": READER_INSTRUCTION}:
            readers.append(json.loads(body["messages"][1]["content"]))
            result = "cobalt telescope"
        elif body["messages"][0]["role"] == "user":
            result = "yes"
        else:
            result = None
        if result is not None:
            return httpx.Response(
                200,
                json={
                    "model": "served/qwen",
                    "choices": [{"finish_reason": "stop", "message": {"content": result}}],
                    "usage": {"prompt_tokens": 32, "completion_tokens": 3},
                },
            )
        calls.append(body)
        schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
        if body.get("response_format") == {"type": "json_object"}:
            schema = json.loads(
                body["messages"][0]["content"].split("```json\n", 1)[1].split("```", 1)[0]
            )
        props = schema.get("properties", {})
        if "nodes" in props:
            result = graph
        elif "summary" in props:
            result = {"summary": "Rhea owns a cobalt telescope.", "description": "Authored source."}
        elif (
            not body.get("response_format")
            and 'following string: "test"' in body["messages"][0]["content"]
        ):
            result = "test"
        else:
            raise AssertionError("Unexpected native helper schema")
        return httpx.Response(
            200,
            json={
                "id": "authored",
                "object": "chat.completion",
                "created": 1,
                "model": "served/qwen",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(result)},
                    }
                ],
                "usage": {"prompt_tokens": 32, "completion_tokens": 16, "total_tokens": 48},
            },
        )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            embedded.append(body)
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            raw = json.dumps(
                {
                    "object": "list",
                    "model": "encoder",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [0.8, 0.6, 0.0]}
                        for i in range(len(texts))
                    ],
                    "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    outer = WorkloadBudget(max_calls=12, max_seconds=300)
    settings = CogneeSettings(
        str(native),
        tmp_path / "evidence/native",
        "authored-history",
        "http://unused.invalid/v1",
        "served/qwen",
        "local",
        f"http://127.0.0.1:{server.server_port}/v1",
        "text-embedding-3-small",
        3,
        "local",
        max_completion_tokens=512,
        timeout_seconds=180,
    )
    factory = module().CogneeRetrieverFactory(
        settings=settings,
        profile=RackProfile(base_url="http://authored.invalid/v1", model="served/qwen"),
        outer_budget=outer,
        out=tmp_path / "evidence",
        transport=httpx.MockTransport(upstream),
    )

    def answer(question, context):
        assert (tmp_path / "evidence/before-reader-000001.json").exists()
        readers.append((question, context))
        outer.begin_call()  # Stand-in reader ledger; not native bridge-local spend.
        return NS(text="cobalt telescope", prompt_tokens=40, completion_tokens=3)

    try:
        if execution != "factory":
            from dataclasses import replace

            from gluomem_bench.bench.parallel_campaign import (
                CampaignBudget,
                Job,
                JobContext,
                RequestLimiter,
            )
            from gluomem_bench.ingest import Conversation, Session, Turn

            driver = importlib.import_module("gluomem_bench.bench.cognee_driver")
            source = (
                Path(__file__).resolve().parents[2]
                / 'tests/fixtures/native/operator-supplied'
            )
            if not source.is_file():
                pytest.skip("pinned upstream scorer path unavailable")
            judge = Path(json.loads(source.read_bytes())["judge_source"])
            conv = Conversation(
                "authored",
                ("user", "assistant"),
                [
                    Session(
                        1,
                        datetime(2024, 1, 2),
                        [Turn("user", "Rhea owns a cobalt telescope.", "answer_label")],
                    )
                ],
            )
            sample = Sample(
                "authored",
                "authored",
                "What does Rhea own?",
                "cobalt telescope",
                1,
                "single-session-user",
                question_date="2024-01-03",
            )
            adapter = NS(
                name="longmemeval_s",
                sha="authored-only",
                verify=lambda: None,
                load=lambda: [sample],
                conversations=lambda: [conv],
            )
            job = Job(
                "authored-graph",
                "authored",
                tmp_path / "job",
                4 if execution == "budget" else 12,
                300,
            )
            campaign = CampaignBudget(max_calls=12, max_seconds=300)
            assert campaign.reserve(job)
            ctx = JobContext(
                job, campaign, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072)
            )
            kwargs = dict(
                job=job,
                context=ctx,
                cfg=bench_cfg,
                adapter=adapter,
                sample=sample,
                profile=factory.profile,
                judge_source=judge,
                settings=replace(settings, root=job.output / "graph/native"),
                transport=httpx.MockTransport(upstream),
                require_real=False,
            )
            if execution in {"budget", "finalization"}:
                if execution == "finalization":

                    def fail_verification(**kwargs):
                        raise RuntimeError("authored post-leaf verifier fault")

                    monkeypatch.setattr(driver, "reconcile_job", fail_verification)
                with pytest.raises(RuntimeError) as caught:
                    driver.run_graph_job(**kwargs)
                failure = json.loads((job.output / "pipeline-failure.json").read_bytes())
                assert failure["generation_calls"] == ctx.calls == caught.value.generation_calls
                assert ctx.calls == (4 if execution == "budget" else len(calls) + 2)
                assert failure["native_generation_calls"] == len(calls) > 0
                assert failure["reader_judge_generation_calls"] == ctx.calls - len(calls)
                assert ctx.failed
                assert not (job.output / "pipeline.json").exists()
                assert (job.output / "reader/manifest.json").exists() == (
                    execution == "finalization"
                )
                assert not (tmp_path / "campaign-result.json").exists()
                return
            from gluomem_bench.bench.parallel_campaign import CampaignPolicy, run_jobs

            contexts = []

            def actual_job(job, context):
                contexts.append(context)
                return driver.run_graph_job(**{**kwargs, "context": context})

            policy = CampaignPolicy(
                max_workers=1,
                max_in_flight=1,
                max_in_flight_tokens=131072,
                max_calls=12,
                max_seconds=350,
                stop_on_failure=True,
            )
            coordinated = run_jobs(
                [job], policy=policy, out=tmp_path / "coordination", run_job=actual_job
            )
            assert coordinated["jobs"][0]["status"] == "complete", coordinated
            ctx = contexts[0]
            value = coordinated["jobs"][0]["result"]
            assert value["generation_calls"] == ctx.calls == len(calls) + 2
            assert value["native_generation_calls"] == len(calls)
            assert value["reader_judge_generation_calls"] == 2
            assert value["quality_status"] == "verified_pending_post_exit"
            assert readers[0]["question"].startswith("Current Date: 2024-01-03")
            assert "Connections:" in readers[0]["evidence"]
            from dataclasses import asdict

            # Synthetic coordinator evidence, with process-liveness tested separately.
            root_result = driver.finalize_root(tmp_path, coordinated, spec_sha256="authored-only")
            assert not root_result["complete"]
            monkeypatch.setattr(driver, "assert_producer_exited", lambda record: None)
            spec = {
                "campaign_policy": asdict(policy),
                "judge_source": str(judge),
                "jobs": [
                    {
                        "job_id": job.job_id,
                        "history_id": job.history_id,
                        "output": str(job.output),
                        "call_budget": job.call_budget,
                        "seconds_budget": job.seconds_budget,
                        "sample_id": sample.id,
                    }
                ],
            }
            verified = driver.verify_root_evidence(
                root=tmp_path,
                spec=spec,
                cfg=bench_cfg,
                adapter=adapter,
                profile=factory.profile,
                spec_sha256="authored-only",
                require_real=False,
            )
            assert verified["complete"] and verified["accounting"]["calls"] == len(calls) + 2
            # Saved self-declarations cannot conceal a raw native receipt mutation.
            with (job.output / "graph/generation.jsonl").open("ab") as stream:
                stream.write(b"\n")
            with pytest.raises(ValueError, match="hash"):
                driver.verify_root_evidence(
                    root=tmp_path,
                    spec=spec,
                    cfg=bench_cfg,
                    adapter=adapter,
                    profile=factory.profile,
                    require_real=False,
                )
            assert json.loads((tmp_path / "campaign-result.json").read_bytes())["complete"] is False
            return
        outer.max_seconds = 60
        retriever = factory(NS(benchmark=NS(answer=NS(context_k=30))), conversation(), answer)
        sample = Sample(
            "authored",
            "answer_secret",
            "What does Rhea own?",
            "unused",
            1,
            "single-session-user",
            question_date="2024-01-03",
        )
        prediction = retriever.answer(sample)
        assert retriever.backend.settings.timeout_seconds <= outer.remaining() + 2
        assert prediction.answer == "cobalt telescope"
        assert "Connections:" in readers[0][1]
        assert "Current Date: 2024-01-03" in readers[0][0]
        assert prediction.retrieved[0].text == readers[0][1]
        assert embedded[-1]["input_type"] == "query"
        retriever.close()
        receipt = json.loads((tmp_path / "evidence/native-ledger.json").read_bytes())
        assert receipt["status"] == "closed_verified"
        assert receipt["generation_calls"] == len(calls) == outer.calls - 1
        assert receipt["generation"]["calls"] == len(calls)
        assert receipt["native_worker_closed"] and receipt["proxy_closed"]
        assert receipt["graph_context_receipts"] == 1
        assert receipt["source_pins"]["version"] == "1.6.5"
        assert receipt["artifact_sha256"]
    finally:
        factory.close()
        server.shutdown()
        server.server_close()
        thread.join()


def module():
    return importlib.import_module("gluomem_bench.bench.cognee_retriever")


def conversation():
    return NS(
        conv_id="answer_secret",
        sessions=[
            NS(
                index=99,
                date_time=datetime(2024, 1, 2),
                turns=[
                    NS(dia_id="answer_gold", speaker="user", text="Rhea owns a cobalt telescope."),
                    NS(
                        dia_id="native_label",
                        speaker="assistant",
                        text="Keep it dry.\nOriginal newline.",
                    ),
                ],
            )
        ],
    )


def test_native_subledger_retains_dispatch_charged_before_coordination_write_failure(tmp_path):
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )

    job = Job("authored", "authored", tmp_path / "job", 6, 120)
    campaign = CampaignBudget(max_calls=6, max_seconds=120)
    assert campaign.reserve(job)

    def fail(context):
        raise OSError("authored coordination persistence failure after debit")

    outer = JobContext(
        job, campaign, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072), on_spend=fail
    )
    budget = module().NativeWorkloadBudget(max_calls=6, max_seconds=120, outer_budget=outer)
    with pytest.raises(OSError):
        budget.begin_call()
    assert budget.calls == outer.calls == campaign.calls == 1
    assert budget.failed


def test_session_documents_preserve_body_role_date_without_native_identifiers():
    documents = module().session_documents(conversation())
    assert len(documents) == 1
    assert documents[0].id == "session-000001"
    assert documents[0].text == (
        "[2024-01-02T00:00:00]\n"
        "user: Rhea owns a cobalt telescope.\nassistant: Keep it dry.\nOriginal newline."
    )
    assert "answer_" not in repr(documents)
    assert documents[0].provenance == {"session_ordinal": 1, "date": "2024-01-02T00:00:00"}
