"""Admission/finalization tests use authored files; never live QA clients."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest


def module():
    return importlib.import_module("gluomem_bench.bench.cognee_driver")


def approved_spec(tmp_path):
    """Construct byte-bound AUTHORED admission metadata, never execute its jobs."""
    m = module()
    base = {
        "protocol": "local-qwen-development-pilot@v1",
        "sample_ids": list(m.SAMPLE_IDS),
        "profile": {},
        "dataset": {},
        "config_overrides": {},
        "embedding_recipe": {},
        "embedding_url": "http://encoder.invalid/v1",
        "judge_source": str(tmp_path / "judge"),
        "files": [],
    }
    reference = tmp_path / "reference.json"
    reference.write_bytes(json.dumps(base).encode())
    spec = {
        "protocol": m.PROTOCOL,
        "producer_id": "producer",
        "reference_spec": {
            "path": str(reference),
            "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        },
        **{
            key: base[key]
            for key in (
                "sample_ids",
                "profile",
                "dataset",
                "config_overrides",
                "embedding_recipe",
                "embedding_url",
                "judge_source",
            )
        },
        "output_root": str(tmp_path / "fresh"),
        "evidence_token_budget": 4096,
        "files": [],
        "jobs": [
            {
                "sample_id": sid,
                "system": "cognee",
                "job_id": f"graph-{i}",
                "history_id": f"longmemeval_s:{sid}",
                "output": str(tmp_path / "fresh" / f"graph-{i}"),
                "call_budget": 1024,
                "seconds_budget": 1200,
            }
            for i, sid in enumerate(m.SAMPLE_IDS)
        ],
        "campaign_policy": {
            "max_workers": 1,
            "max_in_flight": 1,
            "max_in_flight_tokens": 131072,
            "max_calls": 7168,
            "max_seconds": 8400,
            "stop_on_failure": True,
        },
    }
    return spec


def persist_approval(tmp_path, spec):
    path, approval = tmp_path / "spec.json", tmp_path / "approval.json"
    path.write_bytes(json.dumps(spec).encode())
    approval.write_bytes(
        json.dumps(
            {
                "approved": True,
                "reviewer_kind": "AI-assisted",
                "reviewer_id": "reviewer",
                "spec_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ).encode()
    )
    return path, approval


def test_approval_cannot_omit_actual_producer_files(tmp_path):
    spec = approved_spec(tmp_path)
    paths = persist_approval(tmp_path, spec)
    with pytest.raises(ValueError, match="inventory"):
        module().verify_execution_spec(*paths)
    assert not Path(spec["output_root"]).exists()


@pytest.mark.parametrize(
    "mutation", ["selection", "budget", "ceiling", "profile", "matrix", "freshness"]
)
def test_scope_refuses_protocol_drift_before_output(tmp_path, mutation):
    spec = approved_spec(tmp_path)
    base = json.loads(Path(spec["reference_spec"]["path"]).read_bytes())
    if mutation == "selection":
        spec["sample_ids"] = spec["sample_ids"][:-1]
    if mutation == "budget":
        spec["jobs"][0]["call_budget"] = 1025
    if mutation == "ceiling":
        spec["evidence_token_budget"] = 4097
    if mutation == "profile":
        spec["profile"] = {"model": "not-the-reference"}
    if mutation == "matrix":
        spec["jobs"][1]["sample_id"] = spec["jobs"][0]["sample_id"]
    if mutation == "freshness":
        Path(spec["output_root"]).mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        module().validate_scope(spec, base)


def test_qualification_gate_rejects_missing_real_model_and_vector_evidence(tmp_path):
    spec = approved_spec(tmp_path)
    with pytest.raises(ValueError, match="qualification"):
        module().verify_qualifications(spec, {})


def test_root_publication_is_not_completion_and_write_failure_keeps_spend(tmp_path, monkeypatch):
    m = module()
    result = {
        "jobs": [{"job_id": "authored", "status": "failed", "calls": 5}],
        "accounting": {"calls": 5, "elapsed_seconds": 1},
    }
    record = m.finalize_root(tmp_path, result, spec_sha256="authored")
    assert record["complete"] is False
    assert record["status"] == "incomplete"
    assert record["completion_authority"] == "root-post-exit-only"
    assert record["full_suite"]["status"] == "unrun"
    saved = (tmp_path / "campaign-result.json").read_bytes()

    def fail(path, value):
        raise OSError("authored disk failure")

    monkeypatch.setattr(m, "_save", fail)
    with pytest.raises(RuntimeError) as caught:
        m.finalize_root(tmp_path, result, spec_sha256="authored")
    assert caught.value.generation_calls == 5
    assert caught.value.receipt["accounting"]["calls"] == 5
    assert (tmp_path / "campaign-result.json").read_bytes() == saved


def test_post_exit_verification_refuses_live_producer(tmp_path):
    import os

    result = {"producer_pid": os.getpid(), "complete": False}
    (tmp_path / "campaign-result.json").write_bytes(json.dumps(result).encode())
    with pytest.raises(RuntimeError, match="producer.*exited"):
        module().assert_producer_exited(result)


def test_real_scheduler_preserves_failed_native_spend_and_unrun_matrix(
    tmp_path, bench_cfg, monkeypatch
):
    from types import SimpleNamespace as NS

    from gluomem_bench.bench.rack import RackProfile

    m = module()
    spec = approved_spec(tmp_path)
    spec["native"] = {"python_executable": "authored", "chunk_size": 512, "timeout_seconds": 120}
    spec["embedding_recipe"] = {"model": "authored", "dim": 768}
    adapter = NS(load=lambda: [NS(id=sid) for sid in m.SAMPLE_IDS])

    def fail_after_dispatch(**kwargs):
        kwargs["context"].begin_call()
        raise RuntimeError("authored failed native dispatch")

    monkeypatch.setattr(m, "run_graph_job", fail_after_dispatch)
    result = m.run_admitted(
        spec,
        cfg=bench_cfg,
        adapter=adapter,
        profile=RackProfile(base_url="http://authored.invalid/v1", model="served/qwen"),
        encoder=None,
        spec_sha256="authored",
    )
    assert result["status"] == "incomplete"
    assert result["accounting"]["calls"] == 1
    assert [r["status"] for r in result["jobs"]] == ["failed"] + ["unrun"] * 6
    assert result["jobs"][0]["calls"] == 1
    assert not result["complete"]


def test_input_preflight_rejects_unpinned_dataset_before_clients(tmp_path):
    spec = approved_spec(tmp_path)
    source = tmp_path / "authored-not-released.json"
    source.write_bytes(b"[]")
    spec["dataset"] = {"name": "longmemeval_s", "path": str(source), "sha256": "0" * 64}
    with pytest.raises(ValueError, match="dataset"):
        module().prepare_inputs(spec)


def test_cli_help_and_missing_root_authority(tmp_path, capsys):
    with pytest.raises(SystemExit) as caught:
        module().main(["--help"])
    assert caught.value.code == 0
    assert "--verify-post-exit" in capsys.readouterr().out
    (tmp_path / "reader").mkdir()
    (tmp_path / "reader/manifest.json").write_bytes(b'{"complete": true}')
    with pytest.raises(FileNotFoundError):
        module().verify_root_evidence(root=tmp_path, spec={}, cfg=None, adapter=None, profile=None)


def test_hashing_failure_carries_actual_root_spend(tmp_path, monkeypatch):
    m = module()
    result = {"jobs": [{"status": "failed", "calls": 7}], "accounting": {"calls": 7}}

    def fail(root):
        raise OSError("authored hash read failure")

    monkeypatch.setattr(m, "_artifact_hashes", fail)
    with pytest.raises(RuntimeError) as caught:
        m.finalize_root(tmp_path, result, spec_sha256="authored")
    assert caught.value.generation_calls == 7
    assert not (tmp_path / "campaign-result.json").exists()


def test_reference_is_pinned_not_an_approved_lookalike(tmp_path):
    spec = approved_spec(tmp_path)
    m = module()
    package = Path(m.__file__).parent
    paths = [Path(spec["reference_spec"]["path"])] + [
        package / n
        for n in (
            "cognee_driver.py",
            "cognee_retriever.py",
            "cognee_graph.py",
            "cognee_campaign.py",
        )
    ]
    paths += [
        package.parents[1] / "tests/bench" / n
        for n in ("test_cognee_driver.py", "test_cognee_retriever.py")
    ]
    spec["files"] = [
        {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths
    ]
    with pytest.raises(ValueError, match="reference"):
        m.verify_execution_spec(*persist_approval(tmp_path, spec))


def test_cli_refusal_is_sanitized_without_traceback_or_endpoint(tmp_path, capsys):
    spec, approval = tmp_path / "spec.json", tmp_path / "approval.json"
    spec.write_bytes(b'{"protocol":"wrong","private_url":"http://secret.invalid/v1"}')
    approval.write_bytes(b"{}")
    with pytest.raises(SystemExit) as caught:
        module().main(["--spec", str(spec), "--approval", str(approval)])
    assert caught.value.code == 1
    output = capsys.readouterr()
    assert "ValueError" in output.err
    assert "secret.invalid" not in output.err + output.out
    assert "Traceback" not in output.err


def test_common_history_matches_adapter_strip_and_empty_filter(tmp_path, bench_cfg):
    from gluomem_bench.bench.adapters.longmemeval import LongMemEvalAdapter

    raw = {
        "question_id": "authored",
        "haystack_dates": ["2024/01/02 (Tue) 00:00"],
        "haystack_session_ids": ["answer_label"],
        "haystack_sessions": [
            [{"role": "user", "content": "  cobalt\n "}, {"role": "assistant", "content": "   "}]
        ],
    }
    adapter = LongMemEvalAdapter(bench_cfg, "longmemeval_s")
    adapter._raw = [raw]
    expected = adapter.conversations()[0]
    actual = module().original_history(raw)
    assert [(t.speaker, t.text) for s in actual.sessions for t in s.turns] == [
        (t.speaker, t.text) for s in expected.sessions for t in s.turns
    ]
    assert "answer_label" not in repr(actual)


def test_original_history_preserves_common_bodies_and_drops_label_metadata():
    raw = {
        "question_id": "authored",
        "haystack_dates": ["2024/01/02 (Tue) 00:00"],
        "haystack_session_ids": ["answer_label"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "  Rhea owns a cobalt telescope.\n",
                    "has_answer": True,
                },
                {"role": "assistant", "content": "\n Keep dry.  "},
            ]
        ],
        "question": "NOT A SOURCE",
        "answer": "NOT A SOURCE",
    }
    conv = module().original_history(raw)
    assert conv.sessions[0].turns[0].text == raw["haystack_sessions"][0][0]["content"].strip()
    from gluomem_bench.bench.cognee_retriever import session_documents

    docs = session_documents(conv)
    assert "user: Rhea owns a cobalt telescope.\n" in docs[0].text
    assert "assistant: Keep dry." in docs[0].text
    assert "answer_label" not in repr(docs) and "NOT A SOURCE" not in repr(docs)


def test_bad_approval_refuses_before_clients_stores_or_other_reads(tmp_path):
    spec = tmp_path / "spec.json"
    approval = tmp_path / "approval.json"
    output = tmp_path / "fresh"
    spec.write_bytes(
        json.dumps({"protocol": "cognee-graph-lme-seven@v1", "output_root": str(output)}).encode()
    )
    approval.write_bytes(json.dumps({"approved": True, "spec_sha256": "0" * 64}).encode())
    with pytest.raises(ValueError, match="independent.*approval"):
        module().execute_cognee(spec, approval)
    assert not output.exists()


def full_scope(tmp_path):
    spec = approved_spec(tmp_path)
    ids = [f"authored-{i:03d}" for i in range(500)]
    base = json.loads(Path(spec["reference_spec"]["path"]).read_bytes())
    base["sample_ids"] = ids
    spec["protocol"] = "cognee-graph-lme-complete@v2"
    spec["scope"] = "complete-longmemeval-s"
    spec["sample_ids"] = ids
    template = spec["jobs"][0]
    spec["jobs"] = [
        dict(
            template,
            sample_id=sid,
            job_id=f"graph-{i}",
            history_id="longmemeval_s:" + sid,
            output=str(tmp_path / "fresh" / f"graph-{i}"),
        )
        for i, sid in enumerate(ids)
    ]
    spec["campaign_policy"].update(max_calls=512000, max_seconds=600000)
    return spec, base


def test_complete_scope_accepts_all_500_ordered_reference_ids(tmp_path):
    spec, base = full_scope(tmp_path)
    module().validate_scope(spec, base)
    assert len(spec["jobs"]) == 500


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered", "partial_reference"])
def test_complete_scope_rejects_noncomplete_or_unpaired_selection(tmp_path, mutation):
    spec, base = full_scope(tmp_path)
    if mutation == "missing":
        spec["sample_ids"].pop()
    if mutation == "duplicate":
        spec["sample_ids"][1] = spec["sample_ids"][0]
    if mutation == "reordered":
        spec["sample_ids"] = list(reversed(spec["sample_ids"]))
    if mutation == "partial_reference":
        base["sample_ids"] = base["sample_ids"][:-1]
    with pytest.raises(ValueError):
        module().validate_scope(spec, base)


def test_select_inputs_uses_explicit_order_and_common_projection_and_actual_counts(bench_cfg):
    from types import SimpleNamespace as NS

    from gluomem_bench.bench.adapters.longmemeval import LongMemEvalAdapter

    raw = [
        {
            "question_id": sid,
            "haystack_dates": ["2024/01/02 (Tue) 00:00"],
            "haystack_session_ids": ["answer_leak"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "  source " + sid + "  "},
                    {"role": "assistant", "content": "   "},
                ]
            ],
        }
        for sid in ["b", "a"]
    ]
    adapter = LongMemEvalAdapter(bench_cfg, "longmemeval_s")
    adapter._raw = raw
    adapter.load = lambda: [
        NS(id=r["question_id"], conv_id=r["question_id"], question_date="date") for r in raw
    ]
    samples, conversations, counts = module().select_inputs(adapter, ["a", "b"])
    assert [s.id for s in samples] == ["a", "b"]
    assert [c.conv_id for c in conversations] == ["a", "b"]
    assert counts["questions"] == 2 and counts["sessions"] == 2 and counts["turns"] == 2
    assert counts["roles"] == {"user": 2}
    assert counts["documents"] == 2
    assert counts["text_utf8_bytes"] == len(b"source a") + len(b"source b")
    assert "answer_leak" not in repr(conversations)
    assert conversations[0].sessions[0].turns[0].text == "source a"
    with pytest.raises(ValueError, match="complete"):
        module().select_inputs(adapter, ["a"], complete=True)


def test_complete_scope_requires_matching_actual_counts(bench_cfg):
    from types import SimpleNamespace as NS

    adapter = NS(
        load=lambda: [NS(id="a", conv_id="a", question_date="date")], conversations=lambda: []
    )
    with pytest.raises(ValueError):
        module().select_inputs(adapter, ["a"])


def test_full_root_tracks_actual_protocol_and_scope(tmp_path):
    m = module()
    result = {"jobs": [{"status": "failed", "calls": 3}], "accounting": {"calls": 3}}
    record = m.finalize_root(
        tmp_path, result, spec_sha256="authored", protocol="cognee-graph-lme-complete@v2"
    )
    assert record["protocol"] == "cognee-graph-lme-complete@v2"
    assert record["full_suite"]["scope"] == "complete LongMemEval-S Cognee graph condition"
    assert record["full_suite"]["status"] == "unrun"


def test_complete_reference_needs_byte_bound_independent_approval(tmp_path):
    spec, base = full_scope(tmp_path)
    path = Path(spec["reference_spec"]["path"])
    path.write_bytes(json.dumps(base).encode())
    spec["reference_spec"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    approval = tmp_path / "reference-approval.json"
    approval.write_bytes(
        json.dumps(
            {
                "approved": True,
                "reviewer_kind": "AI-assisted",
                "reviewer_id": "reference-reviewer",
                "spec_sha256": spec["reference_spec"]["sha256"],
            }
        ).encode()
    )
    spec["reference_approval"] = {
        "path": str(approval),
        "sha256": hashlib.sha256(approval.read_bytes()).hexdigest(),
    }
    assert module().verify_reference(spec) == base
    approval.write_bytes(b"{}")
    with pytest.raises(ValueError, match="reference"):
        module().verify_reference(spec)
