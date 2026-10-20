"""Pinned trajectory datasets and an explicit environment/action runner seam.

This module does not turn MemoryArena into static QA and does not claim a
native end-to-end benchmark implementation. No generation, downloads, SDK
imports, registry mutation, or automatic installation occur here. Callers own
isolated memories, environment/agent construction, tool execution, tokenizer
admission, budgets, tracing and native scoring. Keep data and receipts external.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class V2Case:
    id: str
    domain: str
    question_type: str
    question: str
    image: str | None
    gold: str
    eval_function: str
    trajectories: tuple[dict[str, Any], ...]
    root: Path


def _verified_path(root: Path, name: str, pins: Mapping[str, str]) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path escapes data root")
    expected = pins.get(name, "")
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"SHA256 pin required: {name}")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {name}")
    return path


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL rows must be objects")
                yield row


def load_v2(
    root: Path,
    *,
    pins: Mapping[str, str],
    tier: str = "small",
    question_ids: tuple[str, ...] | None = None,
) -> list[V2Case]:
    """Verify complete input bytes, stream trajectories, retain selected haystacks.

    All source trajectory fields, ordered states and image references survive.
    Loading text does not imply image assets or multimodal execution admission.
    """
    root = Path(root).resolve()
    if tier not in {"small", "medium"}:
        raise ValueError("tier must be small or medium")
    paths = {
        name: _verified_path(root, name, pins)
        for name in ("questions.jsonl", "trajectories.jsonl", f"haystacks/lme_v2_{tier}.json")
    }
    questions = list(_jsonl(paths["questions.jsonl"]))
    haystacks = json.loads(paths[f"haystacks/lme_v2_{tier}.json"].read_text(encoding="utf-8"))
    if question_ids is not None:
        questions = [q for q in questions if q["id"] in question_ids]
        if {q["id"] for q in questions} != set(question_ids):
            raise ValueError("unknown question IDs")
    question_keys = [q.get("id") for q in questions]
    if (
        not questions
        or any(not isinstance(k, str) or not k for k in question_keys)
        or len(set(question_keys)) != len(question_keys)
    ):
        raise ValueError("empty or duplicate/invalid questions")
    for q in questions:
        if q.get("domain") not in {"web", "enterprise"} or not isinstance(q.get("question"), str):
            raise ValueError("invalid question schema")
        tids = haystacks.get(q["id"])
        if (
            not isinstance(tids, list)
            or not tids
            or any(not isinstance(t, str) for t in tids)
            or len(set(tids)) != len(tids)
        ):
            raise ValueError("missing or duplicate/invalid haystack")
    wanted = {tid for q in questions for tid in haystacks[q["id"]]}
    trajectories = {}
    seen = set()
    for trajectory in _jsonl(paths["trajectories.jsonl"]):
        tid = trajectory.get("id")
        if not isinstance(tid, str) or not tid or tid in seen:
            raise ValueError("duplicate/invalid trajectory")
        seen.add(tid)
        if tid in wanted:
            states = trajectory.get("states")
            if (
                not isinstance(states, list)
                or not states
                or any(
                    not isinstance(s, dict)
                    or not isinstance(s.get("accessibility_tree"), str)
                    or (s.get("screenshot") is not None and not isinstance(s["screenshot"], str))
                    for s in states
                )
            ):
                raise ValueError("invalid trajectory states")
            trajectories[tid] = trajectory
    if wanted != set(trajectories):
        raise ValueError("missing trajectory")
    for q in questions:
        if any(trajectories[t]["domain"] != q["domain"] for t in haystacks[q["id"]]):
            raise ValueError("cross-domain haystack")
    return [
        V2Case(
            q["id"],
            q["domain"],
            q["question_type"],
            q["question"],
            q.get("image"),
            q["answer"],
            q["eval_function"],
            tuple(trajectories[t] for t in haystacks[q["id"]]),
            root,
        )
        for q in questions
    ]


@dataclass(frozen=True)
class V2MemoryResult:
    protocol: str
    context: Any


def run_v2_memory(
    case: V2Case,
    memory: Any,
    *,
    mode: str,
    reader_multimodal: bool = False,
    memory_multimodal: bool = False,
    image_pins: Mapping[str, str] | None = None,
) -> V2MemoryResult:
    """Insert a haystack into a FRESH isolated memory, then query without labels.

    Capability flags are operator attestations backed by external receipts, not
    tests of a model. Reader vision alone does not qualify memory ingestion.
    Native-input describes transport only, not official model reproduction.
    Text-only drops images: accessibility-tree/actions, NOT OCR or native vision.
    """
    from copy import deepcopy

    if mode not in {"text-only", "multimodal"}:
        raise ValueError("unsupported mode")
    images: dict[str, str] = {}
    if mode == "multimodal":
        if not reader_multimodal or not memory_multimodal:
            raise ValueError("reader AND memory multimodal qualification required")
        refs = [
            s["screenshot"] for t in case.trajectories for s in t["states"] if s.get("screenshot")
        ]
        if case.image:
            refs.append(case.image)
        # Validate the WHOLE selected set before the first mutating insertion.
        images = {ref: str(_verified_path(case.root, ref, image_pins or {})) for ref in refs}
    for trajectory in case.trajectories:
        adapted = deepcopy(trajectory)
        for state in adapted["states"]:
            if mode == "text-only":
                state.pop("screenshot", None)
            elif state.get("screenshot"):
                state["screenshot"] = images[state["screenshot"]]
        memory.insert(adapted)
    if mode == "multimodal":
        context = memory.query(
            case.question, query_image=images[case.image] if case.image else None
        )
        return V2MemoryResult("native-input:longmemeval-v2:multimodal@v1", context)
    return V2MemoryResult("adapted:longmemeval-v2:text-only@v1", memory.query(case.question))


ARENA_DOMAINS = (
    "bundled_shopping",
    "progressive_search",
    "group_travel_planner",
    "formal_reasoning_math",
    "formal_reasoning_phys",
)


@dataclass(frozen=True)
class ArenaSession:
    """Only allowed inputs; no IDs, paper names, gold or evaluator labels."""

    question: str
    background: str = ""


@dataclass(frozen=True)
class ArenaTask:
    id: int
    domain: str
    sessions: tuple[ArenaSession, ...]
    gold: tuple[Any, ...]


def load_arena(root: Path, *, domain: str, pins: Mapping[str, str]) -> list[ArenaTask]:
    """Parse the five upstream test configs as interdependent ordered tasks."""
    if domain not in ARENA_DOMAINS:
        raise ValueError("unknown MemoryArena domain")
    rows = _jsonl(_verified_path(Path(root), f"{domain}/data.jsonl", pins))
    tasks = []
    seen = set()
    for row in rows:
        tid = row.get("id")
        if type(tid) is not int or tid in seen:
            raise ValueError("invalid or duplicate task ID")
        seen.add(tid)
        questions, answers = row.get("questions"), row.get("answers")
        if (
            not isinstance(questions, list)
            or not questions
            or any(not isinstance(q, str) or not q.strip() for q in questions)
        ):
            raise ValueError("nonempty list of question strings required")
        if not isinstance(answers, list) or len(answers) != len(questions):
            raise ValueError("answer/session count mismatch")
        if domain.startswith("formal_reasoning"):
            backgrounds = row.get("backgrounds")
            if (
                not isinstance(backgrounds, list)
                or len(backgrounds) != len(questions)
                or any(not isinstance(b, str) for b in backgrounds)
            ):
                raise ValueError("aligned background strings required")
        elif domain == "group_travel_planner":
            backgrounds = [json.dumps(row["base_person"], ensure_ascii=False)] * len(questions)
        else:
            backgrounds = [""] * len(questions)
        tasks.append(
            ArenaTask(
                row["id"],
                domain,
                tuple(ArenaSession(q, b) for q, b in zip(questions, backgrounds, strict=True)),
                tuple(row["answers"]),
            )
        )
    if not tasks:
        raise ValueError("empty MemoryArena dataset")
    return tasks


@dataclass(frozen=True)
class ArenaAction:
    value: Any
    # Three-tuple math/travel environments have no done flag. Their caller
    # must explicitly mark a completed agent session; tool calls are not final.
    session_complete: bool = False


class ArenaIncomplete(RuntimeError):
    """Infrastructure/protocol failure with partial actions, never a zero score."""

    def __init__(self, message: str, completed_sessions: int, steps: list[dict[str, Any]]):
        super().__init__(message)
        from copy import deepcopy

        self.completed_sessions = completed_sessions
        self.steps = tuple(deepcopy(steps))


@dataclass(frozen=True)
class ArenaRun:
    protocol: str
    completed_sessions: int
    steps: tuple[dict[str, Any], ...]


_OBSERVATION_FIELDS = {
    "bundled_shopping": {
        "state",
        "instruction",
        "task_description",
        "purchases",
        "upstream_instruction_text",
    },
    "progressive_search": {"state", "agent_output", "tool_calls", "tool_results"},
    "group_travel_planner": {"step", "final", "state", "tool_results"},
    "formal_reasoning_math": {
        "step",
        "state",
        "tool_info",
        "tool_trace",
        "tool_calls",
        "tool_results",
        "final",
    },
    "formal_reasoning_phys": {
        "step",
        "state",
        "tool_info",
        "tool_trace",
        "tool_calls",
        "tool_results",
        "final",
    },
}


def _public_observation(domain: str, observation: Any) -> dict[str, Any]:
    """Do not pass native reset answers, shopping targets or judge feedback on."""
    from copy import deepcopy

    if not isinstance(observation, dict):
        raise ValueError("environment observation must be an object")
    return deepcopy({k: v for k, v in observation.items() if k in _OBSERVATION_FIELDS[domain]})


def run_arena(
    task: ArenaTask,
    environment: Any,
    policy: Any,
    memory: Any,
    *,
    max_steps: int,
    seed: int | None = None,
) -> ArenaRun:
    """Execute actual env.reset/step/close; never replace actions with QA pairs.

    Policy(session, public_observation, retrieved_context) -> ArenaAction. The
    policy must execute any agent-side tools itself (as upstream math/travel
    agents do). One fresh environment and one isolated memory per task. Memory
    needs query(text) and insert(experience); callers bind production ingestion.
    Judging is OFF: native scorers run separately over retained action traces.
    This is an adapted bridge, not the official multi-agent/evaluation runner.
    """
    from copy import deepcopy

    if task.domain not in ARENA_DOMAINS or type(max_steps) is not int or max_steps < 1:
        raise ValueError("known domain and positive max_steps required")
    steps = []
    completed = 0
    try:
        observation = _public_observation(task.domain, environment.reset(seed=seed))
        for session in task.sessions:
            query = (
                f"{session.background}\n\n{session.question}"
                if session.background
                else session.question
            )
            context = memory.query(query)
            session_steps = []
            for _ in range(max_steps):
                action = policy(session, deepcopy(observation), context)
                if not isinstance(action, ArenaAction):
                    raise TypeError("policy must return ArenaAction")
                result = environment.step(action.value, need_judge=False)
                if len(result) == 3:
                    raw, reward, info = result
                    done = info.get("done", False)
                elif len(result) == 4:
                    raw, reward, done, info = result
                else:
                    raise ValueError("environment step must return 3 or 4 values")
                observation = _public_observation(task.domain, raw)
                step = {"action": action.value, "observation": observation}
                steps.append(step)
                session_steps.append(step)
                if done or action.session_complete:
                    memory.insert(
                        {
                            "question": session.question,
                            "background": session.background,
                            "steps": session_steps,
                        }
                    )
                    completed += 1
                    if done and completed < len(task.sessions):
                        raise ArenaIncomplete(
                            "environment terminated before remaining sessions", completed, steps
                        )
                    break
            else:
                raise ArenaIncomplete(
                    "session action budget exhausted; incomplete, not a quality score",
                    completed,
                    steps,
                )
        return ArenaRun(
            "adapted:memoryarena:environment-bridge-unscored@v1", completed, tuple(steps)
        )
    except ArenaIncomplete:
        raise
    except Exception as exc:
        raise ArenaIncomplete(f"environment bridge failed: {exc}", completed, steps) from exc
    finally:
        try:
            environment.close()
        except Exception as exc:
            raise ArenaIncomplete(f"environment cleanup failed: {exc}", completed, steps) from exc
