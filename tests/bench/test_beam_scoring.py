"""Offline native-source conformance; never calls a live judge.

Set BEAM_UPSTREAM_DIR to the archived b2da22e source tree. Authored outputs
qualify prompt/metric/transport fidelity, NOT Qwen judgment quality.
"""

import ast
import os
from pathlib import Path

import pytest


@pytest.fixture
def source_root():
    root = os.environ.get("BEAM_UPSTREAM_DIR")
    if root is None:
        pytest.skip("requires byte-pinned BEAM archive: BEAM_UPSTREAM_DIR")
    return Path(root)


def test_native_rubric_prompt_preserves_replacement_order_and_omits_question(source_root):
    from gluomem_bench.bench.beam_scoring import NativeBeam

    native = NativeBeam.from_directory(source_root)
    tree = ast.parse((source_root / "src/prompts.py").read_bytes())
    template = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and getattr(n.targets[0], "id", "") == "unified_llm_judge_base_prompt"
    )
    calls = []

    def invoke(kind, messages):
        calls.append((kind, messages))
        return '{"score": 0.5, "reason": "authored partial verdict"}'

    result = native.evaluate(
        "preference_following",
        ["use <llm_response>"],
        "  Résumé\n<rubric_item>  ",
        "MUST NOT BE INSERTED",
        invoke,
    )
    assert calls == [
        (
            "rubric",
            [
                {
                    "role": "user",
                    "content": template.replace("<rubric_item>", "use <llm_response>").replace(
                        "<llm_response>", "  Résumé\n<rubric_item>  "
                    ),
                }
            ],
        )
    ]
    assert result == {
        "llm_judge_score": 0.5,
        "llm_judge_responses": [{"score": 0.5, "reason": "authored partial verdict"}],
    }
    assert native.source_receipt["revision"] == "b2da22eac88bb0874c64665f13457eb99835774a"


@pytest.mark.parametrize(
    "raw",
    [
        '{"score": true, "reason":"bad"}',
        '{"score":"1", "reason":"bad"}',
        '{"score":0.25, "reason":"bad"}',
        '{"score":NaN, "reason":"bad"}',
        '{"score":1}',
        '{"score":0,"reason":17}',
        '{"score":1,"reason":""}',
        "unparseable",
        '{"score":1,}',
        '{"score":0,"score":1,"reason":"duplicate"}',
    ],
)
def test_malformed_rubric_judge_is_fatal_not_zero_or_repaired(source_root, raw):
    from gluomem_bench.bench.beam_scoring import NativeBeam

    native = NativeBeam.from_directory(source_root)
    calls = []

    def invoke(*args):
        calls.append(args)
        return raw

    with pytest.raises(ValueError, match="invalid rubric judgment"):
        native.evaluate("abstention", ["first", "never called"], "answer", "question", invoke)
    assert len(calls) == 1


def upstream_namespace(source_root):
    """Independent direct extraction of the pinned upstream oracle, no imports."""
    import json
    import re

    from scipy.stats import kendalltau

    env = dict(json=json, re=re, List=list, Tuple=tuple, kendalltau=kendalltau)
    prompts = ast.parse((source_root / "src/prompts.py").read_bytes())
    for node in prompts.body:
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "unified_llm_judge_base_prompt"
        ):
            env[node.targets[0].id] = ast.literal_eval(node.value)
    tree = ast.parse((source_root / "src/evaluation/compute_metrics.py").read_bytes())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and (
            n.name.startswith("evaluate_")
            or n.name
            in ("llm_equivalence", "align_with_llm", "event_ordering_score", "parse_json_response")
        )
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<upstream oracle>", "exec"), env)
    return env


def test_native_event_ordering_exact_greedy_calls_blank_lines_then_rubrics(source_root):
    from types import SimpleNamespace

    from gluomem_bench.bench.beam_scoring import NativeBeam

    raw = ["NO", "YES", "YES", '{"score":1,"reason":"first"}', '{"score":0.5,"reason":"second"}']
    oracle_calls, actual_calls = [], []

    class Oracle:
        def invoke(self, prompt):
            oracle_calls.append(prompt)
            return SimpleNamespace(content=raw[len(oracle_calls) - 1])

    expected = upstream_namespace(source_root)["evaluate_event_ordering"](
        ["A", "B"], " b \n a \n", "unused", Oracle()
    )

    def invoke(kind, messages):
        actual_calls.append((kind, messages))
        return raw[len(actual_calls) - 1]

    actual = NativeBeam.from_directory(source_root).evaluate(
        "event_ordering", ["A", "B"], " b \n a \n", "unused", invoke
    )
    assert actual == expected
    assert [kind for kind, _ in actual_calls] == ["equivalence"] * 3 + ["rubric"] * 2
    assert [messages for _, messages in actual_calls] == [
        p if isinstance(p, list) else [{"role": "user", "content": p}] for p in oracle_calls
    ]
    assert "exaplanation" in actual_calls[0][1][0]["content"]  # native typo preserved
    assert actual["tau_norm"] != actual["final_score"]


@pytest.mark.parametrize(
    "raw", ["yes but no", "not yes", "NO YES", "", "probably", "**YES**", None]
)
def test_equivalence_requires_unambiguous_native_binary_output(source_root, raw):
    from gluomem_bench.bench.beam_scoring import NativeBeam

    calls = []

    def invoke(kind, messages):
        calls.append(kind)
        return raw if kind == "equivalence" else '{"score":1,"reason":"authored"}'

    with pytest.raises(ValueError, match="invalid equivalence judgment"):
        NativeBeam.from_directory(source_root).evaluate(
            "event_ordering", ["A", "B"], "answer", "question", invoke
        )
    assert calls == ["equivalence"]


def case(category="information_extraction"):
    return {
        "case_id": "100K-1:information_extraction:0",
        "system": "authored-control",
        "conversation_id": "1",
        "storage_split": "100K",
        "category": category,
        "question": "Question?  ",
        "rubric": ["one", "two", "three"],
        "llm_response": "  answer\n",
        "original_question_id": "untouched-source-token",
        "dataset": {
            "repository": "Mohammadta/BEAM",
            "license": "CC-BY-SA-4.0",
            "revision": "3205395e897e7318c7b094ef4e6047b9b82dbb03",
        },
    }


def rack(tmp_path, outputs, *, fault=None):
    import json

    import httpx

    from gluomem_bench.bench.rack import GenerationBudgets, RackClient, RackProfile
    from gluomem_bench.bench.rack_trace import WorkloadBudget

    wire = []
    replies = iter(outputs)

    def handle(request):
        body = json.loads(request.content) if request.content else None
        wire.append((request.url.path, body))
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"data": [] if fault == "preflight" else [{"id": "operator/Qwen-test"}]}
            )
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 131072 if fault == "context" else 123})
        assert request.url.path == "/v1/chat/completions"
        if fault == "http":
            return httpx.Response(503, text="authored upstream unavailable")
        return httpx.Response(
            200,
            json={
                "id": f"authored-{len(wire)}",
                "model": "wrong-model" if fault == "model" else "operator/Qwen-test",
                "choices": [
                    {
                        "finish_reason": "length" if fault == "length" else "stop",
                        "message": {"role": "assistant", "content": next(replies)},
                    }
                ],
                "usage": {"prompt_tokens": 123, "completion_tokens": 17},
            },
        )

    profile = RackProfile(
        base_url="http://rack.example.invalid/v1",
        model="operator/Qwen-test",
        profile_version="role-budgets-v2",
        generation_budgets=GenerationBudgets(
            extraction=128, consolidation=128, reader=128, judge=768
        ),
    )
    client = RackClient(
        profile,
        receipt=tmp_path / "wire.jsonl",
        budget=WorkloadBudget(max_calls=32, max_seconds=60),
        transport=httpx.MockTransport(handle),
    )
    return client, wire


def test_guarded_transport_native_prompt_no_adapted_system_or_schema(source_root, tmp_path):
    from gluomem_bench.bench.beam_scoring import NativeBeam, score_case

    client, wire = rack(
        tmp_path,
        [
            '{"score":0,"reason":"none"}',
            '{"score":0.5,"reason":"partial"}',
            '```json\n{"score":1,"reason":"all"}\n```',
        ],
    )
    try:
        bundle = score_case(
            NativeBeam.from_directory(source_root),
            case(),
            client,
            judge_model="operator/Qwen-test",
            judge_output_tokens=768,
        )
        assert bundle["case"] == case()
        assert bundle["scores"]["llm_judge_score"] == 0.5
        assert bundle["report_metric"] == "llm_judge_score"
        assert bundle["report_score"] == 0.5
        assert bundle["judge_fidelity"] == "adapted-model; not reference-model reproduction"
        assert bundle["reference_judge_model"] == "gpt-4.1-mini"
        generations = [body for path, body in wire if path == "/v1/chat/completions"]
        tokenizations = [body for path, body in wire if path == "/tokenize"]
        assert len(generations) == len(bundle["judgments"]) == client.budget.calls == 3
        assert wire[0][0] == "/v1/models"
        for index, payload in enumerate(generations):
            assert payload["messages"] == tokenizations[index]["messages"]
            assert payload["messages"] == bundle["judgments"][index]["request"]["messages"]
            assert [m["role"] for m in payload["messages"]] == ["user"]
            assert payload["max_tokens"] == 768
            assert payload["temperature"] == 0
            assert "response_format" not in payload
            assert "Task:" not in payload["messages"][0]["content"]
            assert bundle["judgments"][index]["response"]["admitted_prompt_tokens"] == 123
    finally:
        client.close()


@pytest.mark.parametrize(
    "fault",
    [
        "model",
        "output",
        "bool_output",
        "no_budget",
        "rubric",
        "category",
        "question",
        "answer",
        "case_id",
        "split",
        "dataset_license",
    ],
)
def test_admission_refuses_invalid_case_or_operator_configuration_before_network(
    source_root, tmp_path, fault
):
    from gluomem_bench.bench.beam_scoring import NativeBeam, score_case

    client, wire = rack(tmp_path, [])
    sample, model, output = case(), "operator/Qwen-test", 768
    if fault == "model":
        model = "other/Qwen"
    elif fault == "output":
        output = 512
    elif fault == "bool_output":
        output = True
    elif fault == "no_budget":
        client.budget = None
    elif fault == "rubric":
        sample["rubric"] = []
    elif fault == "category":
        sample["category"] = "adapted_event_order"
    elif fault == "question":
        sample["question"] = 17
    elif fault == "answer":
        sample["llm_response"] = None
    elif fault == "case_id":
        sample["case_id"] = ""
    elif fault == "split":
        sample["storage_split"] = "128K"
    else:
        sample["dataset"]["license"] = "unknown"
    try:
        with pytest.raises(ValueError, match="admission"):
            score_case(
                NativeBeam.from_directory(source_root),
                sample,
                client,
                judge_model=model,
                judge_output_tokens=output,
            )
        assert wire == []
    finally:
        client.close()


@pytest.mark.parametrize(
    "category,raw",
    [("information_extraction", '{"score":true,"reason":"bad"}'), ("event_ordering", "not yes")],
)
def test_invalid_judge_latches_shared_guard_and_retains_failed_progress(
    source_root, tmp_path, category, raw
):
    from gluomem_bench.bench.beam_scoring import NativeBeam, score_case

    client, wire = rack(tmp_path, [raw])
    try:
        with pytest.raises(ValueError, match="invalid .* judgment") as error:
            score_case(
                NativeBeam.from_directory(source_root),
                case(category),
                client,
                judge_model="operator/Qwen-test",
                judge_output_tokens=768,
            )
        assert client.budget.failed
        progress = error.value.beam_progress
        assert progress["complete"] is False
        assert progress["generation_calls"] == 1
        assert progress["judgments"][0]["response"]["text"] == raw
        before = len(wire)
        with pytest.raises(ValueError):
            client.generate_messages("rubric", [{"role": "user", "content": "must not run"}])
        assert len(wire) == before
    finally:
        client.close()


def test_dataset_free_rescore_uses_bound_raw_outputs_not_saved_scores(
    source_root, tmp_path, monkeypatch
):
    import json
    import socket

    from gluomem_bench.bench.beam_scoring import NativeBeam, rescore, score_case

    sample = case("event_ordering")
    sample.update(rubric=["A", "B"], llm_response="b\na\n")
    client, _ = rack(
        tmp_path, ["NO", "YES", "yes", '{"score":0,"reason":"none"}', '{"score":1,"reason":"all"}']
    )
    try:
        bundle = score_case(
            NativeBeam.from_directory(source_root),
            sample,
            client,
            judge_model="operator/Qwen-test",
            judge_output_tokens=768,
        )
    finally:
        client.close()
    saved = json.loads(json.dumps(bundle, ensure_ascii=False, allow_nan=False))
    saved["scores"] = {"llm_judge_score": 100, "tau_norm": 100}
    saved["report_score"] = 100

    def forbidden(*args, **kwargs):
        raise AssertionError("dataset/source/network access is forbidden in offline rescore")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    replay = rescore(saved)
    assert replay["scores"] == bundle["scores"]
    assert replay["report_metric"] == "tau_norm"
    assert replay["report_score"] == bundle["scores"]["tau_norm"]
    assert replay["report_score"] != bundle["scores"]["final_score"]
    assert replay["scores"]["llm_judge_score"] == 0.5


@pytest.fixture
def saved_bundle(source_root, tmp_path):
    from gluomem_bench.bench.beam_scoring import NativeBeam, score_case

    client, _ = rack(tmp_path, ['{"score":0,"reason":"none"}'] * 3)
    try:
        return score_case(
            NativeBeam.from_directory(source_root),
            case(),
            client,
            judge_model="operator/Qwen-test",
            judge_output_tokens=768,
        )
    finally:
        client.close()


def digest(value):
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize("fault", ["text", "model", "length", "choices", "tokens", "completion"])
def test_replay_revalidates_actual_response_envelope_even_with_consistent_hashes(
    saved_bundle, fault
):
    from gluomem_bench.bench.beam_scoring import rescore

    row = saved_bundle["judgments"][0]
    response = row["response"]
    if fault == "text":
        response["text"] = '{"score":1,"reason":"not the actual response"}'
    elif fault == "model":
        response["response"]["model"] = "other-model"
    elif fault == "length":
        response["response"]["choices"][0]["finish_reason"] = "length"
    elif fault == "choices":
        response["response"]["choices"] = []
    elif fault == "tokens":
        response["admitted_prompt_tokens"] = -1
    else:
        response["completion_tokens"] = 769
    row["response_sha256"] = digest(response)
    row["binding_sha256"] = digest([row["request_sha256"], row["response_sha256"]])
    with pytest.raises(ValueError, match="response envelope"):
        rescore(saved_bundle)


@pytest.mark.filterwarnings("ignore:One or more sample arguments is too small")
def test_native_undefined_tau_is_json_null_not_fabricated_zero(source_root, tmp_path):
    import json
    import math
    from types import SimpleNamespace

    from gluomem_bench.bench.beam_scoring import NativeBeam, rescore, score_case

    sample = case("event_ordering")
    sample.update(rubric=["A"], llm_response="A")
    oracle = upstream_namespace(source_root)["event_ordering_score"](
        ["A"], ["A"], "llm", SimpleNamespace(invoke=lambda _: SimpleNamespace(content="YES"))
    )
    assert math.isnan(oracle["tau_norm"])
    client, _ = rack(tmp_path, ["YES", '{"score":1,"reason":"all"}'])
    try:
        bundle = score_case(
            NativeBeam.from_directory(source_root),
            sample,
            client,
            judge_model="operator/Qwen-test",
            judge_output_tokens=768,
        )
    finally:
        client.close()
    json.dumps(bundle, allow_nan=False)
    assert bundle["scores"]["tau_norm"] is bundle["scores"]["final_score"] is None
    assert bundle["report_score"] is None
    assert bundle["undefined_metrics"] == ["tau_norm", "final_score"]
    assert rescore(bundle) == {
        "scores": bundle["scores"],
        "report_metric": "tau_norm",
        "report_score": None,
        "undefined_metrics": ["tau_norm", "final_score"],
    }


def test_native_report_history_weighting_all_categories_and_tau_column(source_root, tmp_path):
    import json
    from types import SimpleNamespace

    from gluomem_bench.bench.beam_scoring import CATEGORIES, NativeBeam, report, score_case

    native, bundles, upstream_files = NativeBeam.from_directory(source_root), [], {}
    for history in ("1", "2"):
        data = {}
        for category in CATEGORIES:
            data[category] = []
            for occurrence in range(
                2 if history == "1" and category == "information_extraction" else 1
            ):
                sample = case(category)
                sample.update(
                    conversation_id=history, case_id=f"100K-{history}:{category}:{occurrence}"
                )
                if category == "event_ordering":
                    sample.update(rubric=["A", "B"], llm_response="b\na\n")
                    outputs = ["NO", "YES", "YES"] + ['{"score":1,"reason":"authored"}'] * 2
                else:
                    outputs = [json.dumps({"score": int(history == "2"), "reason": "authored"})] * 3
                client, _ = rack(tmp_path, outputs)
                try:
                    bundle = score_case(
                        native,
                        sample,
                        client,
                        judge_model="operator/Qwen-test",
                        judge_output_tokens=768,
                    )
                finally:
                    client.close()
                bundles.append(bundle)
                data[category].append(bundle["scores"])
        directory = tmp_path / history
        directory.mkdir()
        (directory / "evaluation-results-authored.json").write_bytes(
            json.dumps(data).encode("utf-8")
        )

    # Execute the actual pinned reporter, substituting only its spreadsheet sink.
    class Frame:
        def __init__(self, values, columns):
            upstream_files["values"], upstream_files["columns"] = values, columns

        def insert(self, *args):
            pass

        def to_excel(self, *args, **kwargs):
            pass

    import os as actual_os

    env = dict(os=actual_os, json=json, pd=SimpleNamespace(DataFrame=Frame))
    tree = ast.parse((source_root / "src/evaluation/report_results.py").read_bytes())
    exec(
        compile(
            ast.Module(
                body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]
            ),
            "<upstream report oracle>",
            "exec",
        ),
        env,
    )
    env["report_llms_accuracies"](
        str(tmp_path), ["evaluation-results-authored.json"], "unused.xlsx"
    )
    result = report(bundles, expected_case_sha256=[b["case_sha256"] for b in bundles])
    row = result["rows"][0]
    assert row["by_category"]["information_extraction"]["score"] == 0.5  # NOT micro 1/3
    assert [row["by_category"][c]["score"] for c in CATEGORIES] == upstream_files["values"][0]
    assert row["by_category"]["event_ordering"]["metric"] == "tau_norm"
    assert result["cases"] == len(bundles)


@pytest.mark.parametrize(
    "fault", ["status", "request", "response", "pair", "missing", "extra", "reorder", "case"]
)
def test_saved_judgment_exact_binding_and_coverage_reject_tampering(saved_bundle, fault):
    from copy import deepcopy

    from gluomem_bench.bench.beam_scoring import rescore

    rows = saved_bundle["judgments"]
    if fault == "status":
        rows[0]["status"] = "request_failed"
    elif fault == "request":
        rows[0]["request"]["messages"][0]["content"] += " "
    elif fault == "response":
        rows[0]["response"]["text"] += " "
    elif fault == "pair":
        rows[0]["binding_sha256"] = "0" * 64
    elif fault == "missing":
        rows.pop()
    elif fault == "extra":
        rows.append(deepcopy(rows[-1]))
    elif fault == "reorder":
        rows.reverse()
    else:
        saved_bundle["case"]["question"] += " "
    with pytest.raises(ValueError):
        rescore(saved_bundle)


@pytest.mark.parametrize("fault,calls", [("context", 0), ("length", 1), ("model", 1), ("http", 1)])
def test_fatal_transport_keeps_attempted_request_progress_and_stops(
    source_root, tmp_path, fault, calls
):
    from gluomem_bench.bench.beam_scoring import NativeBeam, score_case

    client, wire = rack(tmp_path, ['{"score":1,"reason":"authored"}'], fault=fault)
    try:
        with pytest.raises(Exception) as error:
            score_case(
                NativeBeam.from_directory(source_root),
                case(),
                client,
                judge_model="operator/Qwen-test",
                judge_output_tokens=768,
            )
        assert client.budget.failed
        assert error.value.beam_progress["generation_calls"] == calls
        attempts = error.value.beam_progress["judgments"]
        assert len(attempts) == 1
        assert attempts[0]["request"]["index"] == 0
        assert attempts[0]["status"] == "request_failed"
        before = len(wire)
        with pytest.raises(ValueError):
            client.check()
        assert len(wire) == before
    finally:
        client.close()


@pytest.mark.parametrize(
    "category",
    [
        "abstention",
        "contradiction_resolution",
        "information_extraction",
        "instruction_following",
        "knowledge_update",
        "multi_session_reasoning",
        "preference_following",
        "summarization",
        "temporal_reasoning",
    ],
)
def test_offline_all_rubric_categories_match_pinned_functions_for_zero_half_one(
    source_root, category
):
    from types import SimpleNamespace

    from gluomem_bench.bench.beam_scoring import NativeBeam

    raw = [
        '{"score":0,"reason":"none"}',
        '{"score":0.5,"reason":"partial"}',
        '{"score":1,"reason":"complete"}',
    ]
    oracle_calls, actual_calls = [], []

    class Oracle:
        def invoke(self, prompt):
            oracle_calls.append(prompt)
            return SimpleNamespace(content=raw[len(oracle_calls) - 1])

    rubric, answer, question = (
        ["one", "<llm_response>", "three"],
        " Résumé\n",
        "ignored native question",
    )
    expected = upstream_namespace(source_root)[f"evaluate_{category}"](
        rubric, answer, question, Oracle()
    )

    def invoke(kind, messages):
        actual_calls.append(messages)
        assert kind == "rubric"
        return raw[len(actual_calls) - 1]

    assert (
        NativeBeam.from_directory(source_root).evaluate(category, rubric, answer, question, invoke)
        == expected
    )
    assert expected["llm_judge_score"] == 0.5
    assert actual_calls == [[{"role": "user", "content": p}] for p in oracle_calls]


@pytest.mark.parametrize(
    "name",
    [
        "src/prompts.py",
        "src/evaluation/compute_metrics.py",
        "src/evaluation/report_results.py",
        "src/evaluation/run_evaluation.py",
        "src/llm.py",
        "src/llms_config.json",
        "LICENSE",
    ],
)
def test_every_upstream_source_is_byte_pinned_before_compilation(source_root, name):
    from gluomem_bench.bench.beam_scoring import SOURCE_SHA256, NativeBeam

    sources = {n: (source_root / n).read_bytes() for n in SOURCE_SHA256}
    sources[name] += b"\n"
    with pytest.raises(ValueError, match="source hash mismatch"):
        NativeBeam(sources)


@pytest.mark.filterwarnings("ignore:One or more sample arguments is too small")
@pytest.mark.parametrize(
    "reference,answer",
    [
        (["A", "B"], "A\nB"),
        (["A", "B"], "B\nA"),
        (["A", "B", "C"], "B"),
        (["A", "B"], "unknown\nunknown"),
        (["A", "A", "B"], "A\nA\nB"),
        (["A", "B"], "A\nB\n"),
        (["A", "B"], ""),
        (["A"], "A"),
        (["A", "B"], "B\r\nA"),
    ],
)
def test_event_metric_tail_and_call_order_differential_edge_conformance(
    source_root, reference, answer
):
    import math
    from types import SimpleNamespace

    from gluomem_bench.bench.beam_scoring import NativeBeam

    oracle_calls, raws, actual_calls = [], [], []

    class Oracle:
        def invoke(self, prompt):
            oracle_calls.append(prompt)
            raw = (
                ("YES" if len(oracle_calls) % 3 == 0 else "NO")
                if isinstance(prompt, list)
                else ('{"score":0.5,"reason":"authored"}')
            )
            raws.append(raw)
            return SimpleNamespace(content=raw)

    expected = upstream_namespace(source_root)["evaluate_event_ordering"](
        reference, answer, "question", Oracle()
    )

    def invoke(kind, messages):
        actual_calls.append(messages)
        return raws[len(actual_calls) - 1]

    actual = NativeBeam.from_directory(source_root).evaluate(
        "event_ordering", reference, answer, "question", invoke
    )
    for key, value in expected.items():
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(actual[key])
        else:
            assert actual[key] == value
    assert actual_calls == [
        p if isinstance(p, list) else [{"role": "user", "content": p}] for p in oracle_calls
    ]


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate", "empty"])
def test_report_requires_exact_external_plan_coverage(saved_bundle, fault):
    from gluomem_bench.bench.beam_scoring import report

    expected = [saved_bundle["case_sha256"]]
    bundles = [saved_bundle]
    if fault == "missing":
        expected.append("0" * 64)
    elif fault == "extra":
        bundles *= 2
    elif fault == "duplicate":
        expected *= 2
        bundles *= 2
    else:
        expected, bundles = [], []
    with pytest.raises(ValueError, match="exact expected coverage"):
        report(bundles, expected_case_sha256=expected)


def test_reference_model_disclosure_matches_pinned_runner_and_config(source_root, saved_bundle):
    import json

    llm_tree = ast.parse((source_root / "src/llm.py").read_bytes())
    constructor = next(
        n.value
        for n in llm_tree.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "gpt_llm_obj"
    )
    keywords = {k.arg: k.value for k in constructor.keywords}
    assert ast.literal_eval(keywords["model_name"]) == saved_bundle["reference_judge_model"]
    assert ast.literal_eval(keywords["temperature"]) == 0
    runner = ast.parse((source_root / "src/evaluation/run_evaluation.py").read_bytes())
    calls = [
        n
        for n in ast.walk(runner)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "batch_run_evaluation"
    ]
    assert any(
        getattr(k.value, "id", "") == "gpt_llm"
        for c in calls
        for k in c.keywords
        if k.arg == "model"
    )
    config = json.loads((source_root / "src/llms_config.json").read_bytes())
    assert config["qwen"] == {"model_url": "", "model_name": "", "api_key": ""}


def test_preflight_failure_retains_zero_spend_and_no_judgments(source_root, tmp_path):
    from gluomem_bench.bench.beam_scoring import NativeBeam, score_case

    client, wire = rack(tmp_path, [], fault="preflight")
    try:
        with pytest.raises(ValueError, match="exact rack model") as error:
            score_case(
                NativeBeam.from_directory(source_root),
                case(),
                client,
                judge_model="operator/Qwen-test",
                judge_output_tokens=768,
            )
        assert error.value.beam_progress["generation_calls"] == 0
        assert error.value.beam_progress["judgments"] == []
        assert client.budget.failed
        assert len(wire) == 1
    finally:
        client.close()


def test_guarded_execution_uses_shared_token_leases(source_root, tmp_path, monkeypatch):
    from contextlib import contextmanager

    from gluomem_bench.bench.beam_scoring import NativeBeam, score_case

    client, _ = rack(tmp_path, ['{"score":1,"reason":"authored"}'] * 3)
    leases = []

    @contextmanager
    def lease(tokens):
        leases.append(("enter", tokens))
        yield
        leases.append(("exit", tokens))

    monkeypatch.setattr(client.budget, "lease", lease)
    try:
        score_case(
            NativeBeam.from_directory(source_root),
            case(),
            client,
            judge_model="operator/Qwen-test",
            judge_output_tokens=768,
        )
        reserve = 123 + 768 + client.profile.schema_reserve_tokens
        assert leases == [("enter", reserve), ("exit", reserve)] * 3
    finally:
        client.close()
