"""All-tier BEAM admission contracts; fixtures are not quality evidence."""

import importlib

import pytest


def suite():
    return importlib.import_module("gluomem_bench.bench.beam_suite")


def row(cid="1"):
    return {
        "conversation_id": cid,
        "chat": [
            [
                {"id": 0, "role": "user", "content": "chat only", "time_anchor": "March-15-2024"},
                {"id": 1, "role": "assistant", "content": "assistant text", "time_anchor": None},
            ]
        ],
        "probing_questions": {
            "event_ordering": [
                {
                    "question": "probe secret",
                    "answer": "gold secret",
                    "rubric": ["event A", "event B"],
                }
            ]
        },
        "plans": [{"secret": "plan secret"}],
        "user_profile": {"user_info": "profile secret"},
    }


def test_all_tier_adapter_qualifies_ids_reuses_chat_roles_and_no_gold_leakage():
    mod = suite()
    adapter = mod.BeamSuiteAdapter.from_rows("500K", [row()])
    adapter.verify()
    samples = adapter.load()
    assert samples[0].id == "beam-500K-1:qa:0"
    assert samples[0].gold == "gold secret"
    assert adapter.rubrics()[samples[0].id] == ("event A", "event B")
    conv = adapter.conversations()[0]
    assert conv.conv_id == "beam-500K-1"
    assert [(t.speaker, t.text) for t in conv.sessions[0].turns] == [
        ("user", "chat only"),
        ("assistant", "assistant text"),
    ]
    assert conv.sessions[0].date_time.isoformat() == "2024-03-15T00:00:00"
    assert adapter.dia_content() == {
        "beam-500K-1:0": "chat only",
        "beam-500K-1:1": "assistant text",
    }
    assert "secret" not in str(adapter.conversations())


def test_10m_nullable_multi_plan_chat_preserves_order_not_lexicographic_keys():
    mod = suite()
    r = row()
    first, second = r["chat"][0]
    second = dict(second, id=100, time_anchor="April-01-2024")
    r["chat"] = [
        {"plan-1": [{"batch_number": 1, "time_anchor": None, "turns": [[first]]}], "plan-10": None},
        {
            "plan-1": None,
            "plan-10": [{"batch_number": 1, "time_anchor": None, "turns": [[second]]}],
        },
    ]
    adapter = mod.BeamSuiteAdapter.from_rows("10M", [r])
    adapter.verify()
    conv = adapter.conversations()[0]
    assert [s.date_time.month for s in conv.sessions] == [3, 4]
    assert [t.dia_id for s in conv.sessions for t in s.turns] == ["beam-10M-1:0", "beam-10M-1:100"]
    assert adapter.chat_anchors() == {
        "beam-10M-1:0": "March-15-2024",
        "beam-10M-1:100": "April-01-2024",
    }


def test_anchor_changes_inside_batch_are_not_retroactively_applied():
    r = row()
    r["chat"][0][1]["time_anchor"] = "April-01-2024"
    adapter = suite().BeamSuiteAdapter.from_rows("100K", [r])
    adapter.verify()
    assert [s.date_time.month for s in adapter.conversations()[0].sessions] == [3, 4]


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_history",
        "duplicate_message",
        "missing_anchor",
        "role",
        "no_chat",
        "no_rubric",
        "multi_active_plan",
    ],
)
def test_malformed_inputs_fail_closed(defect):
    r = row()
    rows = [r]
    if defect == "duplicate_history":
        rows.append(row())
    if defect == "duplicate_message":
        r["chat"][0][1]["id"] = 0
    if defect == "missing_anchor":
        r["chat"][0][0]["time_anchor"] = None
    if defect == "role":
        r["chat"][0][0]["role"] = "system"
    if defect == "no_chat":
        r["chat"] = []
    if defect == "no_rubric":
        r["probing_questions"]["event_ordering"][0]["rubric"] = []
    if defect == "multi_active_plan":
        r["chat"] = [{"plan-1": [], "plan-2": []}]
    with pytest.raises(suite().DatasetError):
        suite().BeamSuiteAdapter.from_rows(
            "10M" if defect == "multi_active_plan" else "100K", rows
        ).verify()


def test_upstream_event_ordering_semantic_seam_is_greedy_one_to_one_and_separate():
    calls = []

    def equivalent(reference, system):
        calls.append((reference, system))
        return (reference, system) in {("A", "alpha"), ("B", "beta")}

    result = suite().score_event_ordering(["A", "B"], "beta\nalpha", equivalent=equivalent)
    assert result["tau_norm"] == 0
    assert result["f1"] == 1
    assert result["final_score"] == 0
    assert result["system_canonical"] == ["B", "A"]
    assert calls == [("A", "beta"), ("B", "beta"), ("A", "alpha")]
    assert result["protocol"].startswith("adapted:")
    assert "rubric_compliance" not in result
    with pytest.raises(RuntimeError, match="semantic"):
        suite().score_event_ordering(["A", "B"], "A\nB")


@pytest.mark.parametrize(
    "reference,system",
    [
        (["A", "B", "C"], ["A", "C"]),
        (["A", "B"], ["A", "A", "X"]),
        (["A"], ["A"]),
        (["A", "B"], ["X", "Y"]),
        (["A", "B"], ["A", "B", ""]),
    ],
)
def test_event_tau_b_matches_pinned_upstream_function_without_loading_models(reference, system):
    # Transcribed objective tail: upstream missing events use a shared union tie rank;
    # duplicate literal items use their LAST position, NOT a permutation shortcut.
    import math

    from scipy.stats import kendalltau

    union = list(dict.fromkeys(reference + system))

    def ranks(seq):
        return [{s: i + 1 for i, s in enumerate(seq)}.get(u, len(union) + 1) for u in union]

    if len(union) < 2:
        tau = float("nan")
    else:
        tau = kendalltau(ranks(reference), ranks(system), variant="b").statistic
    result = suite().score_event_ordering(
        reference, "\n".join(system), equivalent=lambda a, b: a == b
    )
    if math.isnan(tau):
        assert result["tau_norm"] is None  # JSON-safe undefined, never fabricated zero.
        assert result["defined"] is False
    else:
        assert result["tau_norm"] == pytest.approx((tau + 1) / 2)
        assert result["defined"] is True
    assert result["report_metric"] == "tau_norm"  # NOT F1-weighted final_score.


def test_rubric_scores_and_protocol_gates_never_claim_native_judge_support():
    mod = suite()
    result = mod.score_rubric([1, 0.5, 0])
    assert result == {"protocol": "adapted:beam_rubric@v1", "rubric_compliance": 0.5, "nuggets": 3}
    for bad in ([], [True], [0.7], [float("nan")]):
        with pytest.raises(ValueError):
            mod.score_rubric(bad)
    gates = mod.scoring_gates()
    assert gates["native_headline_admitted"] is False
    assert gates["upstream_semantic_judge_implemented"] is False
    assert gates["event_ordering_report_metric"] == "tau_norm"


def test_pinned_specs_cover_actual_shards_and_10m_is_separate_dataset():
    mod = suite()
    assert [mod.dataset_spec(s)["histories"] for s in mod.SPLITS] == [20, 35, 35, 10]
    ten = mod.dataset_spec("10M")
    assert ten["repository"] == "Mohammadta/BEAM-10M"
    assert ten["license"] == "CC-BY-SA-4.0"
    assert ten["expected_conversation_ids"] == [str(i) for i in range(1, 11)]
    assert ten["revision"] == "9b2096193fe74e2837e4713e483351e19817773c"
    assert len(ten["files"]) == 2
    assert sum(f["size_bytes"] for f in ten["files"]) == 343825539
    with pytest.raises(mod.DatasetError):
        mod.dataset_spec("128K")


def local_parquet(tmp_path, monkeypatch, split="100K"):
    import hashlib

    import pyarrow as pa
    import pyarrow.parquet as pq

    mod = suite()
    path = tmp_path / "fixture.parquet"
    r = row()
    import json

    r["probing_questions"] = json.dumps(r["probing_questions"])
    pq.write_table(pa.Table.from_pylist([r]), path)
    spec = {
        "repository": "fixture/not-upstream",
        "revision": "0" * 40,
        "histories": 1,
        "files": [
            {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
    }
    monkeypatch.setattr(mod, "dataset_spec", lambda _: spec)
    return mod, path


def test_actual_parquet_transport_and_receipt_coverage(tmp_path, monkeypatch):
    mod, path = local_parquet(tmp_path, monkeypatch)
    adapters = list(mod.iter_parquet_adapters("100K", tmp_path))
    assert len(adapters) == 1
    assert adapters[0].load()[0].id == "beam-100K-1:qa:0"
    receipt = mod.admission_receipt("100K", tmp_path)
    assert receipt["complete"] is True
    assert receipt["histories"] == 1
    assert receipt["questions"] == 1
    assert receipt["messages"] == 2
    assert receipt["roles"] == {"user": 1, "assistant": 1}
    assert receipt["sample_ids"] == ["beam-100K-1:qa:0"]
    assert receipt["by_category"] == {"event_ordering": 1}
    assert receipt["sources"][0]["sha256"] == mod.dataset_spec("100K")["files"][0]["sha256"]
    assert receipt["scoring_gates"]["native_headline_admitted"] is False
    assert receipt["inference_admitted"] is False
    assert receipt["history_receipts"][0]["chat_sha256"]
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(mod.DatasetError, match="size|hash"):
        list(mod.iter_parquet_adapters("100K", tmp_path))


def test_parquet_conversation_ids_must_match_pinned_order(tmp_path, monkeypatch):
    mod, _ = local_parquet(tmp_path, monkeypatch)
    spec = dict(mod.dataset_spec("100K"), expected_conversation_ids=["not-1"])
    monkeypatch.setattr(mod, "dataset_spec", lambda _: spec)
    with pytest.raises(mod.DatasetError, match="conversation_id"):
        list(mod.iter_parquet_adapters("100K", tmp_path))


def test_same_size_corruption_is_hash_rejected(tmp_path, monkeypatch):
    mod, path = local_parquet(tmp_path, monkeypatch)
    b = path.read_bytes()
    path.write_bytes(b[:100] + bytes([b[100] ^ 1]) + b[101:])
    with pytest.raises(mod.DatasetError, match="hash"):
        list(mod.iter_parquet_adapters("100K", tmp_path))


def test_adapter_fetch_is_local_only_and_cannot_inherit_unconfigured_download(
    tmp_path, monkeypatch
):
    mod, path = local_parquet(tmp_path, monkeypatch)
    adapter = next(mod.iter_parquet_adapters("100K", tmp_path))
    assert adapter.fetch() == path
    assert adapter.eligible_counts() == {3: 1}
    with pytest.raises(mod.DatasetError, match="offline"):
        adapter.fetch(force=True)
    fixture = mod.BeamSuiteAdapter.from_rows("100K", [row()])
    with pytest.raises(mod.DatasetError, match="offline"):
        fixture.fetch()
    original = path.read_bytes()
    path.write_bytes(original[:100] + bytes([original[100] ^ 1]) + original[101:])
    with pytest.raises(mod.DatasetError, match="hash"):
        adapter.verify()


def test_missing_shard_fails_before_yield_and_no_partial_admission(tmp_path):
    mod = suite()
    with pytest.raises(mod.DatasetError, match="missing"):
        next(mod.iter_parquet_adapters("10M", tmp_path))


def test_admission_cli_writes_fresh_utf8_receipt(tmp_path, monkeypatch, capsys):
    import json

    mod, _ = local_parquet(tmp_path, monkeypatch)
    out = tmp_path / "admission.json"
    mod.main(["admit", "--split", "100K", "--data-dir", str(tmp_path), "--out", str(out)])
    assert json.loads(out.read_bytes())["histories"] == 1
    assert b"\r\n" not in out.read_bytes()
    with pytest.raises(FileExistsError):
        mod.main(["admit", "--split", "100K", "--data-dir", str(tmp_path), "--out", str(out)])


def test_upstream_reset_message_ids_across_batches_preserve_both_occurrences():
    r = row()
    r["chat"].append(
        [dict(r["chat"][0][0], content="different later chat", time_anchor="April-01-2024")]
    )
    adapter = suite().BeamSuiteAdapter.from_rows("1M", [r])
    adapter.verify()
    bridge = adapter.dia_content()
    assert len(bridge) == 3
    assert set(bridge.values()) == {"chat only", "assistant text", "different later chat"}
    ids = adapter.chat_message_ids()
    assert list(ids.values()) == [0, 1, 0]
    assert len(adapter.chat_anchors()) == 3


def test_module_cli_is_executable():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "gluomem_bench.bench.beam_suite", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--split" in result.stdout


def test_missing_timestamps_allow_parse_receipt_but_never_ingestion(tmp_path, monkeypatch):
    mod, path = local_parquet(tmp_path, monkeypatch)
    # Direct adapter seam still refuses typed ingestion without real timestamps.
    r = row()
    r["chat"][0][0]["time_anchor"] = None
    adapter = mod.BeamSuiteAdapter.from_rows("100K", [r])
    adapter.verify(require_timestamps=False)
    with pytest.raises(mod.DatasetError, match="time_anchor"):
        adapter.conversations()
    monkeypatch.setattr(mod, "iter_parquet_adapters", lambda *args, **kw: iter([adapter]))
    receipt = mod.admission_receipt("100K", tmp_path)
    assert receipt["complete"] is True
    assert receipt["messages"] == 2
    assert receipt["typed_ingestion_admitted"] is False
    assert receipt["history_receipts"][0]["typed_ingestion_admitted"] is False
    assert "time_anchor" in receipt["history_receipts"][0]["ingestion_blocker"]
    assert adapter.chat_messages()[0] == {
        "dia_id": "beam-100K-1:0",
        "source_id": 0,
        "role": "user",
        "content": "chat only",
        "time_anchor": None,
    }


def test_objective_tail_matches_actual_pinned_upstream_code_without_model_imports():
    import ast
    import math
    import os
    import random
    from pathlib import Path

    from scipy.stats import kendalltau

    root = os.environ.get("BEAM_UPSTREAM_DIR")
    if root is None:
        pytest.skip(
            "set BEAM_UPSTREAM_DIR to archived pinned upstream for offline differential test"
        )
    mod = suite()
    mod.upstream_source_receipt(Path(root))
    source = (Path(root) / "src/evaluation/compute_metrics.py").read_text(encoding="utf-8")
    nodes = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name in {"align_with_llm", "event_ordering_score"}
    ]
    assert len(nodes) == 2

    def equivalent(a, b):
        return a == b.removeprefix("alias-")

    namespace = {
        "List": list,
        "Tuple": tuple,
        "kendalltau": kendalltau,
        "llm_equivalence": lambda first_paragraph, second_paragraph, llm: equivalent(
            first_paragraph, second_paragraph
        ),
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "pinned-beam-objective-tail", "exec"),
        namespace,
    )
    rng = random.Random(42)
    for _ in range(300):
        reference = list("ABC")
        system = rng.choices(["A", "B", "C", "alias-A", "alias-B", "X", ""], k=rng.randint(1, 8))
        expected = namespace["event_ordering_score"](reference, system, "llm", None)
        actual = mod.score_event_ordering(reference, "\n".join(system), equivalent=equivalent)
        for key in ("precision", "recall", "f1", "tau_norm", "final_score"):
            if math.isnan(expected[key]):
                assert actual[key] is None
            else:
                assert actual[key] == pytest.approx(expected[key], abs=1e-14)


def test_upstream_source_hashes_are_required_not_merely_claimed(tmp_path):
    mod = suite()
    with pytest.raises(mod.DatasetError, match="source"):
        mod.upstream_source_receipt(tmp_path)
