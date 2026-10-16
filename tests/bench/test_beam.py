"""BEAM schema contract: tiny synthetic rows, never quality evidence."""

import hashlib
import json

import pytest

from gluomem_bench.bench.adapters import make_adapter
from gluomem_bench.config import AppConfig


def beam_data(tmp_path):
    rows = [
        {
            "conversation_id": "1",
            "chat": [
                [
                    {
                        "id": 0,
                        "role": "user",
                        "time_anchor": "March-15-2024",
                        "content": "I prefer blue.",
                    },
                    {"id": 1, "role": "assistant", "content": "Blue noted."},
                ]
            ],
            "probing_questions": json.dumps(
                {
                    "preference_following": [
                        {
                            "question": "Which color?",
                            "ideal_response": "blue",
                            "rubric": ["Mentions blue", "No red"],
                        }
                    ]
                }
            ),
        }
    ]
    path = tmp_path / "beam.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    cfg = AppConfig(
        benchmark={
            "datasets_dir": str(tmp_path),
            "datasets": {"beam_128k": {"url": "", "sha256": sha, "file": path.name}},
        }
    )
    return cfg, path, rows


def test_mem0_preserves_native_assistant_role_without_recasting_named_locomo_speakers():
    from datetime import datetime

    from gluomem_bench.bench.mem0 import _session_messages
    from gluomem_bench.ingest import Session, Turn

    session = Session(
        index=1,
        date_time=datetime(2024, 3, 15),
        turns=[
            Turn(speaker="user", text="Blue"),
            Turn(speaker="assistant", text="Noted"),
            Turn(speaker="Alice", text="Hello"),
        ],
    )
    messages = _session_messages(session)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert "Alice: Hello" in messages[2]["content"]


def test_beam_registered_as_adapted_pinned_rubric_not_native_qa():
    from gluomem_bench.bench.registry import get_entry

    entry = get_entry("beam_128k")
    assert entry.mode == "adapted"
    assert entry.readiness == "ready"
    assert entry.scorers == ("rubric_compliance",)
    assert "3205395e897e7318c7b094ef4e6047b9b82dbb03" in entry.revision
    assert entry.split_unit == "history"
    assert not entry.official


def test_beam_cannot_accidentally_use_generic_qa_accuracy_runner(tmp_path):
    from gluomem_bench.bench.runner import run_benchmark

    cfg, _, _ = beam_data(tmp_path)
    with pytest.raises(ValueError, match="beam.*run"):
        run_benchmark(cfg, dataset="beam_128k", retriever="full_context")


def test_beam_cli_production_ingest_is_explicit_and_rack_only(
    tmp_path, monkeypatch, capsys, rack_profile_path
):
    from gluomem.llm.modules import DspyExtractors

    from gluomem_bench.bench import beam_ingest
    from gluomem_bench.bench.rack import RackClient
    from gluomem_bench.bench.rack_trace import RackRouter
    from gluomem_bench.cli import main

    cfg, data, _ = beam_data(tmp_path)
    seen = {}

    def fake_ingest(config, **kwargs):
        seen["cfg"] = config
        seen.update(kwargs)
        return [{"complete": True}]

    monkeypatch.setattr(beam_ingest, "ingest_beam", fake_ingest)
    monkeypatch.setattr(RackClient, "preflight", lambda self: None)
    assert (
        main(
            [
                "beam",
                "ingest",
                "--profile",
                str(rack_profile_path),
                "--data",
                str(data),
                "--sha256",
                cfg.benchmark.datasets["beam_128k"].sha256,
                "--db-dir",
                str(tmp_path / "stores"),
                "--conv-id",
                "beam-1",
            ]
        )
        == 0
    )
    assert seen["cfg"].dspy.default_model == "rack-qwen"
    assert seen["conv_ids"] == {"beam-1"}
    assert isinstance(seen["extractors"], DspyExtractors)
    assert isinstance(seen["extractors"]._router, RackRouter)
    assert seen["completion_guard"].__self__ is seen["extractors"]._router.budget
    assert json.loads(capsys.readouterr().out)[0]["complete"]


def test_beam_cli_provider_failure_is_visible_without_creating_completion(
    tmp_path, monkeypatch, capsys, rack_profile_path
):
    import httpx

    from gluomem_bench.bench.rack import RackClient
    from gluomem_bench.cli import main

    cfg, data, _ = beam_data(tmp_path)

    def unavailable(self):
        request = httpx.Request("GET", self.profile.base_url + "/models")
        raise httpx.HTTPStatusError(
            "rack backend unavailable",
            request=request,
            response=httpx.Response(500, request=request),
        )

    monkeypatch.setattr(RackClient, "preflight", unavailable)
    stores = tmp_path / "stores"
    assert (
        main(
            [
                "beam",
                "ingest",
                "--profile",
                str(rack_profile_path),
                "--data",
                str(data),
                "--sha256",
                cfg.benchmark.datasets["beam_128k"].sha256,
                "--db-dir",
                str(stores),
                "--conv-id",
                "beam-1",
            ]
        )
        == 1
    )
    assert "rack backend unavailable" in capsys.readouterr().err
    assert not (stores / "beam-1.ingest.json").exists()


def test_beam_cli_exposes_profile_without_loading_commercial_defaults(capsys):
    from gluomem_bench.cli import main

    assert main(["beam", "profile"]) == 0
    profile = json.loads(capsys.readouterr().out)
    assert profile["model"] == "example/served-model"
    assert profile["concurrency"] == 1


def test_beam_prepare_refuses_unpinned_source_before_parquet_import(tmp_path):
    from gluomem_bench.bench.beam_eval import prepare_beam

    path = tmp_path / "wrong.parquet"
    path.write_bytes(b"not upstream parquet")
    with pytest.raises(ValueError, match="sha256"):
        prepare_beam(path, tmp_path / "export.json")


def test_beam_adapter_preserves_roles_time_and_keeps_probes_out_of_ingest(tmp_path):
    cfg, _, _ = beam_data(tmp_path)
    adapter = make_adapter("beam_128k", cfg)
    from gluomem_bench.bench.adapters.beam import BeamAdapter

    assert isinstance(adapter, BeamAdapter)
    adapter.verify()
    sample = adapter.load()[0]
    assert sample.id == "beam-1:qa:0"
    assert sample.gold == "blue"
    assert sample.category_name == "preference_following"
    assert adapter.rubrics()[sample.id] == ("Mentions blue", "No red")
    conv = adapter.conversations()[0]
    assert [t.speaker for t in conv.sessions[0].turns] == ["user", "assistant"]
    assert conv.sessions[0].date_time.isoformat() == "2024-03-15T00:00:00"
    assert "Which color?" not in repr(conv)
    assert "Mentions blue" not in repr(conv)
    assert adapter.dia_content()["beam-1:0"] == "I prefer blue."


def test_beam_accepts_upstream_python_literal_probes(tmp_path):
    cfg, path, rows = beam_data(tmp_path)
    rows[0]["probing_questions"] = repr(json.loads(rows[0]["probing_questions"]))
    path.write_text(json.dumps(rows), encoding="utf-8")
    cfg.benchmark.datasets["beam_128k"].sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    adapter = make_adapter("beam_128k", cfg)
    adapter.verify()
    assert len(adapter.load()) == 1


def test_beam_gluomem_run_refuses_unverified_ingest_before_model_calls(tmp_path):
    from gluomem_bench.bench.beam_eval import run_beam
    from gluomem_bench.bench.rack import RackClient, RackProfile

    cfg, _, _ = beam_data(tmp_path)
    client = RackClient(
        RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
        receipt=tmp_path / "run" / "calls.jsonl",
    )
    try:
        with pytest.raises(ValueError, match="incomplete"):
            run_beam(
                cfg,
                client=client,
                out=tmp_path / "run",
                retriever="gluomem",
                db_dir=tmp_path / "missing",
            )
        assert not client.receipt.exists()
    finally:
        client.close()


@pytest.mark.parametrize("retriever", ["full_context", "bm25"])
def test_beam_run_writes_nuggets_and_rescores_offline_without_dataset(tmp_path, retriever):
    import httpx

    from gluomem_bench.bench.beam_eval import rescore_beam, run_beam
    from gluomem_bench.bench.rack import RackClient, RackProfile

    cfg, data, _ = beam_data(tmp_path)

    def serve(request):
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 20})
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "example/served-model"}]})
        body = json.loads(request.content)
        task = body["response_format"]["json_schema"]["name"]
        result = {"answer": "blue"} if task == "answer" else {"score": 0.5, "reason": "partial"}
        return httpx.Response(
            200,
            json={
                "model": "example/served-model",
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    out = tmp_path / "run"
    client = RackClient(
        RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
        receipt=out / "calls.jsonl",
        transport=httpx.MockTransport(serve),
    )
    scores = run_beam(cfg, client=client, out=out, retriever=retriever, limit=1)
    assert scores["rubric_question_mean"] == 0.5
    assert scores["questions"] == 1
    assert scores["nuggets"] == 2
    for name in [
        "manifest.json",
        "predictions.jsonl",
        "judgments.jsonl",
        "evidence.jsonl",
        "ingest.json",
        "usage.json",
        "timing.json",
        "scores.json",
        "report.md",
    ]:
        assert (out / name).is_file()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["official_result"] is False
    assert manifest["model"]["concurrency"] == 1
    data.unlink()
    client.close()
    assert rescore_beam(out, check=True) == scores
    rows = (out / "judgments.jsonl").read_text().splitlines()
    (out / "judgments.jsonl").write_text(rows[0] + "\n")
    with pytest.raises(ValueError, match="hash|coverage"):
        rescore_beam(out, check=True)


@pytest.mark.parametrize("corruption", ["role", "rubric", "duplicate", "clock"])
def test_beam_rejects_malformed_or_ambiguous_rows(tmp_path, corruption):
    cfg, path, rows = beam_data(tmp_path)
    if corruption == "role":
        rows[0]["chat"][0][0]["role"] = "system"
    elif corruption == "rubric":
        probes = json.loads(rows[0]["probing_questions"])
        probes["preference_following"][0]["rubric"] = []
        rows[0]["probing_questions"] = json.dumps(probes)
    elif corruption == "duplicate":
        rows.append(rows[0])
    else:
        del rows[0]["chat"][0][0]["time_anchor"]
    path.write_text(json.dumps(rows), encoding="utf-8")
    cfg.benchmark.datasets["beam_128k"].sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    from gluomem_bench.bench.adapters.locomo import DatasetError

    with pytest.raises(DatasetError):
        make_adapter("beam_128k", cfg).verify()
