"""Contract fixtures are authored, not benchmark quality examples."""

import hashlib
import json

import pytest

from gluomem_bench.bench import trajectory_suite as ts


def pinned(root, name, value, *, jsonl=False):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ("\n".join(json.dumps(x) for x in value) if jsonl else json.dumps(value)).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def v2_files(root):
    question = {
        "id": "private-id",
        "domain": "web",
        "question_type": "procedure",
        "question": "Where is the menu?",
        "image": None,
        "answer": "SECRET",
        "eval_function": "secret-eval",
    }
    trajectory = {
        "id": "t1",
        "domain": "web",
        "goal": "Open menu",
        "outcome": "success",
        "states": [
            {
                "state_index": 0,
                "action": "click(1)",
                "thought": "Look here",
                "accessibility_tree": "Menu",
                "screenshot": "screenshots/t1/0.png",
            }
        ],
    }
    pins = {
        "questions.jsonl": pinned(root, "questions.jsonl", [question], jsonl=True),
        "trajectories.jsonl": pinned(root, "trajectories.jsonl", [trajectory], jsonl=True),
        "haystacks/lme_v2_small.json": pinned(
            root, "haystacks/lme_v2_small.json", {"private-id": ["t1"]}
        ),
    }
    return pins, trajectory


def test_v2_load_preserves_order_text_screenshot_reference_and_gold_boundary(tmp_path):
    assert ts is not None, "trajectory dataset seam is missing"
    pins, trajectory = v2_files(tmp_path)
    cases = ts.load_v2(tmp_path, pins=pins, tier="small")
    assert len(cases) == 1
    assert cases[0].trajectories == (trajectory,)
    assert cases[0].gold == "SECRET"
    assert cases[0].question == "Where is the menu?"
    assert cases[0].root == tmp_path.resolve()
    (tmp_path / "questions.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="SHA256"):
        ts.load_v2(tmp_path, pins=pins, tier="small")


class RecordingMemory:
    def __init__(self):
        self.inserted = []
        self.queries = []

    def insert(self, item):
        self.inserted.append(item)

    def query(self, *args, **kwargs):
        self.queries.append((args, kwargs))
        return ["retrieved"]


def test_v2_text_adapter_is_explicit_and_query_has_no_private_metadata(tmp_path):
    pins, _ = v2_files(tmp_path)
    case = ts.load_v2(tmp_path, pins=pins)[0]
    memory = RecordingMemory()
    result = ts.run_v2_memory(case, memory, mode="text-only")
    assert result.protocol == "adapted:longmemeval-v2:text-only@v1"
    assert result.context == ["retrieved"]
    assert memory.queries == [(("Where is the menu?",), {})]
    assert memory.inserted[0]["states"][0]["accessibility_tree"] == "Menu"
    assert "screenshot" not in memory.inserted[0]["states"][0]
    assert case.trajectories[0]["states"][0]["screenshot"] == "screenshots/t1/0.png"
    with pytest.raises(ValueError, match="mode"):
        ts.run_v2_memory(case, memory, mode="pretend-native")


def test_v2_multimodal_requires_both_capabilities_and_pinned_images_before_insertion(tmp_path):
    from dataclasses import replace

    pins, _ = v2_files(tmp_path)
    case = replace(ts.load_v2(tmp_path, pins=pins)[0], image="question.png")
    memory = RecordingMemory()
    with pytest.raises(ValueError, match="multimodal"):
        ts.run_v2_memory(case, memory, mode="multimodal")
    assert memory.inserted == []
    with pytest.raises(ValueError, match="SHA256"):
        ts.run_v2_memory(
            case, memory, mode="multimodal", reader_multimodal=True, memory_multimodal=True
        )
    assert memory.inserted == []
    image_pins = {}
    for name in ["question.png", "screenshots/t1/0.png"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"authored image asset, decoding belongs to runtime")
        image_pins[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = ts.run_v2_memory(
        case,
        memory,
        mode="multimodal",
        reader_multimodal=True,
        memory_multimodal=True,
        image_pins=image_pins,
    )
    assert result.protocol == "native-input:longmemeval-v2:multimodal@v1"
    assert memory.queries == [
        (("Where is the menu?",), {"query_image": str((tmp_path / "question.png").resolve())})
    ]
    assert memory.inserted[0]["states"][0]["screenshot"] == str(
        (tmp_path / "screenshots/t1/0.png").resolve()
    )
    assert case.trajectories[0]["states"][0]["screenshot"] == "screenshots/t1/0.png"


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate-question",
        "duplicate-trajectory",
        "duplicate-haystack",
        "cross-domain",
        "missing-trajectory",
        "bad-state",
        "empty-questions",
    ],
)
def test_v2_rejects_ambiguous_or_invalid_pinned_inputs(tmp_path, corruption):
    pins, _ = v2_files(tmp_path)
    qp = tmp_path / "questions.jsonl"
    tp = tmp_path / "trajectories.jsonl"
    hp = tmp_path / "haystacks/lme_v2_small.json"
    if corruption == "duplicate-question":
        qp.write_bytes(qp.read_bytes() + b"\n" + qp.read_bytes())
    elif corruption == "duplicate-trajectory":
        tp.write_bytes(tp.read_bytes() + b"\n" + tp.read_bytes())
    elif corruption == "empty-questions":
        qp.write_bytes(b"")
    elif corruption in {"duplicate-haystack", "missing-trajectory"}:
        hp.write_text(
            json.dumps(
                {"private-id": ["t1", "t1"] if corruption == "duplicate-haystack" else ["missing"]}
            )
        )
    else:
        row = json.loads(tp.read_text())
        if corruption == "cross-domain":
            row["domain"] = "enterprise"
        else:
            row["states"][0]["accessibility_tree"] = 42
        tp.write_text(json.dumps(row))
    pins = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in pins}
    with pytest.raises(ValueError):
        ts.load_v2(tmp_path, pins=pins)


@pytest.mark.parametrize(
    "domain",
    [
        "bundled_shopping",
        "progressive_search",
        "group_travel_planner",
        "formal_reasoning_math",
        "formal_reasoning_phys",
    ],
)
def test_arena_loads_real_jsonl_schema_without_flattening_sessions(tmp_path, domain):
    row = {"id": 9, "questions": ["first", "second"], "answers": ["GOLD1", {"secret": "GOLD2"}]}
    if domain.startswith("formal_reasoning"):
        row["backgrounds"] = ["definitions one", "definitions two"]
        row["paper_name"] = "private-paper-id"
    if domain == "group_travel_planner":
        row["base_person"] = {"name": "Alex", "query": "base trip", "daily_plans": [{"days": 1}]}
    name = domain + "/data.jsonl"
    pins = {name: pinned(tmp_path, name, [row], jsonl=True)}
    tasks = ts.load_arena(tmp_path, domain=domain, pins=pins)
    assert len(tasks) == 1
    assert tasks[0].id == 9
    assert [s.question for s in tasks[0].sessions] == ["first", "second"]
    assert tasks[0].gold == ("GOLD1", {"secret": "GOLD2"})
    assert not hasattr(tasks[0].sessions[0], "gold")
    if domain.startswith("formal_reasoning"):
        assert [s.background for s in tasks[0].sessions] == ["definitions one", "definitions two"]
    elif domain == "group_travel_planner":
        assert json.loads(tasks[0].sessions[0].background) == row["base_person"]
    else:
        assert tasks[0].sessions[0].background == ""


class ActionEnvironment:
    """Authored contract double, never represented as a native environment."""

    def __init__(self, four_tuple=False):
        self.calls = []
        self.closed = False
        self.four_tuple = four_tuple

    def reset(self, seed=None):
        self.calls.append(("reset", seed))
        return {"answers": ["LEAK"], "state": "menu", "judge_result": "LEAK"}

    def step(self, action, **kwargs):
        self.calls.append(("step", action, kwargs))
        obs = {
            "state": "search results" if action == "search[tea]" else "purchase",
            "target_products": ["LEAK"],
            "reward": 1,
            "judge_result": "LEAK",
        }
        if self.four_tuple:
            return obs, 1, action == "click[buy]", {"answers": ["LEAK"]}
        return obs, 1, {"done": action == "click[buy]", "answers": ["LEAK"]}

    def close(self):
        self.closed = True


@pytest.mark.parametrize("four_tuple", [False, True])
def test_arena_bridge_executes_actions_and_only_remembers_allowed_experience(four_tuple):
    task = ts.ArenaTask(7, "bundled_shopping", (ts.ArenaSession("Buy tea"),), ("LEAK",))
    env = ActionEnvironment(four_tuple)
    memory = RecordingMemory()
    seen = []

    def policy(session, observation, context):
        seen.append((session, observation, context))
        return ts.ArenaAction("search[tea]" if len(seen) == 1 else "click[buy]")

    run = ts.run_arena(task, env, policy, memory, max_steps=3)
    assert run.protocol == "adapted:memoryarena:environment-bridge-unscored@v1"
    assert run.completed_sessions == 1
    assert len(run.steps) == 2
    assert env.calls == [
        ("reset", None),
        ("step", "search[tea]", {"need_judge": False}),
        ("step", "click[buy]", {"need_judge": False}),
    ]
    assert memory.queries == [(("Buy tea",), {})]
    assert "LEAK" not in repr(seen) + repr(memory.inserted)
    assert len(memory.inserted) == 1
    assert "search results" in repr(memory.inserted)
    assert env.closed


def test_arena_failure_preserves_partial_trace_and_policy_cannot_mutate_receipts():
    task = ts.ArenaTask(0, "bundled_shopping", (ts.ArenaSession("buy"),), ("secret",))
    env = ActionEnvironment()
    memory = RecordingMemory()
    seen = []

    def policy(session, observation, context):
        seen.append(observation)
        if len(seen) == 2:
            observation["state"] = "tampered"
            raise OSError("agent transport failed")
        return ts.ArenaAction("search[tea]")

    with pytest.raises(ts.ArenaIncomplete, match="agent transport failed") as error:
        ts.run_arena(task, env, policy, memory, max_steps=3)
    assert len(error.value.steps) == 1
    assert error.value.steps[0]["observation"]["state"] == "search results"
    assert isinstance(error.value.__cause__, OSError)
    assert env.closed


def test_arena_rejects_failed_memory_write_and_cleanup_as_incomplete():
    task = ts.ArenaTask(0, "bundled_shopping", (ts.ArenaSession("buy"),), ("secret",))

    class FailedWrite(RecordingMemory):
        def insert(self, item):
            raise OSError("memory commit failed")

    env = ActionEnvironment()
    with pytest.raises(ts.ArenaIncomplete, match="commit failed") as error:
        ts.run_arena(
            task, env, lambda *args: ts.ArenaAction("click[buy]"), FailedWrite(), max_steps=1
        )
    assert error.value.completed_sessions == 0
    assert len(error.value.steps) == 1

    class FailedClose(ActionEnvironment):
        def close(self):
            raise OSError("close failed")

    with pytest.raises(ts.ArenaIncomplete, match="close failed") as error:
        ts.run_arena(
            task,
            FailedClose(),
            lambda *args: ts.ArenaAction("click[buy]"),
            RecordingMemory(),
            max_steps=1,
        )
    assert len(error.value.steps) == 1


def test_arena_keeps_ordered_sessions_but_blocks_early_global_done():
    task = ts.ArenaTask(
        1,
        "formal_reasoning_math",
        (ts.ArenaSession("first", "definition"), ts.ArenaSession("second")),
        ("secret", "secret"),
    )
    env = ActionEnvironment()
    memory = RecordingMemory()

    def policy(session, observation, context):
        return ts.ArenaAction({"type": "final", "answer": session.question}, session_complete=True)

    run = ts.run_arena(task, env, policy, memory, max_steps=1)
    assert run.completed_sessions == 2
    assert memory.queries == [(("definition\n\nfirst",), {}), (("second",), {})]
    assert [x["question"] for x in memory.inserted] == ["first", "second"]
    early = ActionEnvironment(four_tuple=True)
    with pytest.raises(ts.ArenaIncomplete, match="terminated") as error:
        ts.run_arena(
            task, early, lambda *args: ts.ArenaAction("click[buy]"), RecordingMemory(), max_steps=1
        )
    assert error.value.completed_sessions == 1
    assert len(error.value.steps) == 1
    assert early.closed


def test_arena_exhaustion_keeps_partial_actions_and_never_inserts_unfinished_session():
    task = ts.ArenaTask(1, "bundled_shopping", (ts.ArenaSession("find"),), ("secret",))
    env = ActionEnvironment()
    memory = RecordingMemory()
    with pytest.raises(ts.ArenaIncomplete, match="budget") as error:
        ts.run_arena(task, env, lambda *args: ts.ArenaAction("search[tea]"), memory, max_steps=2)
    assert error.value.completed_sessions == 0
    assert len(error.value.steps) == 2
    assert memory.inserted == []
    assert env.closed


@pytest.mark.parametrize(
    "corruption",
    ["duplicate", "empty", "answer-count", "question-type", "background-count", "id-type"],
)
def test_arena_rejects_invalid_tasks_without_silent_truncation(tmp_path, corruption):
    row = {"id": 1, "questions": ["q"], "answers": ["a"], "backgrounds": ["context"]}
    rows = [row]
    if corruption == "duplicate":
        rows.append(row)
    elif corruption == "empty":
        rows = []
    elif corruption == "answer-count":
        row["answers"] = []
    elif corruption == "question-type":
        row["questions"] = [1]
    elif corruption == "background-count":
        row["backgrounds"] = "not-an-aligned-list"
    elif corruption == "id-type":
        row["id"] = True
    name = "formal_reasoning_math/data.jsonl"
    pins = {name: pinned(tmp_path, name, rows, jsonl=True)}
    with pytest.raises(ValueError):
        ts.load_arena(tmp_path, domain="formal_reasoning_math", pins=pins)


def test_pinned_native_math_environment_offline(monkeypatch):
    """Opt-in upstream integration, not a synthetic/native quality score."""
    import os
    import socket
    import subprocess
    import sys
    from pathlib import Path

    source = os.environ.get("GLUOMEM_MEMORYARENA_SOURCE")
    if not source:
        pytest.skip("set GLUOMEM_MEMORYARENA_SOURCE to the external pinned checkout")
    source = Path(source)
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == "6cd9de14b71915e39ac742a20dc33785e14b6aab"
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")

    def no_network(*args, **kwargs):
        raise AssertionError("native offline integration must not connect")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    from env.env_systems.math_env import MathEnvironment

    env = MathEnvironment(
        {
            "backend": "openai",
            "model_name": "offline-no-generation",
            "base_url": "https://example.invalid/v1",
        }
    )
    monkeypatch.setattr(env.llm_backend, "chat", no_network)
    # Native MathEnvironment consumes agent-executed tool results; it does not
    # execute coding actions. Execute a fixed authored tool, not dataset code.
    output = subprocess.check_output([sys.executable, "-c", "print(2+3)"], text=True).strip()
    task = ts.ArenaTask(0, "formal_reasoning_math", (ts.ArenaSession("Compute 2+3"),), ("5",))
    memory = RecordingMemory()

    def policy(session, observation, context):
        if observation.get("step") == 0:
            return ts.ArenaAction(
                {
                    "type": "tool_result",
                    "tool": "coding",
                    "input": {"code": "print(2+3)"},
                    "result": output,
                }
            )
        return ts.ArenaAction({"type": "final", "answer": output}, session_complete=True)

    run = ts.run_arena(task, env, policy, memory, max_steps=2)
    assert run.completed_sessions == 1
    assert len(run.steps) == 2
    assert run.steps[0]["observation"]["tool_info"]["result"] == "5"
    assert run.steps[1]["observation"]["final"] == "5"
    assert "judge_result" not in repr(memory.inserted)
    assert env.history == []
