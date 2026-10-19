"""Bounded pinned MathAgent/MathEnvironment seam; no live benchmark admission.

Safe track: ``python -m gluomem_bench.bench.arena_campaign smoke --native-root
CHECKOUT --out FRESH_DIRECTORY``. Only authored replies through MockTransport;
no model generation, Docker, generated code execution or provider defaults.
``rescore --bundle DIRECTORY`` uses saved source/steps, never a dataset/model.

The native tool loop, reasoning prompt, environment judge and metric functions
execute unchanged source from git objects at PIN. Transport constructors are
injected per private module, not globally patched. Native omitted output caps
are explicitly matched to the reader cap; native retries/fallback are suppressed
by a fail-closed BaseException boundary. This is bounded native-method execution,
NOT a reproduction of the default unbounded native actor/deployment. Pinned
source has no verify_solution function: MathEnvironment.judge is its verifier.
Tool capability can be admitted explicitly with FrozenToolCapability byte pins
for the operator's real one-attempt receipt and wire. This enables only the fixed
AUTHORED_LIVE_TASK through run_math_task (<=8 generations/120 seconds shared).
The CLI remains offline-only; no live constructor defaults or benchmark admission.
Live runs are synthetic integration, NOT benchmark evidence. Real task quality
remains closed pending a parent-frozen byte-bound job and real memory binding.
All five official domains remain unrun until full environment admission.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import logging
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx

from gluomem_bench.bench.error_privacy import safe_error

from .parallel_campaign import CampaignBudget, Job, JobContext, RequestLimiter
from .rack import FatalRackError, RackClient, RackProfile, SharedRackError
from .trajectory_suite import ARENA_DOMAINS, ArenaSession, ArenaTask

PIN = "6cd9de14b71915e39ac742a20dc33785e14b6aab"
SOURCE_PINS = {
    "agent/base.py": "3a7edf6b0f96e1793aef3d658e586c078ebc6a67b0ab0bc4239ba64370571b26",
    "agent/base_agent.py": "61eac895f6e8d4bc347898898bb7c431d8a6f7ee2f37377ede747a9620a6e28c",
    "agent/math.py": "33faff8e13f7c12d69d375f319fa151102a8268b8a52d1795304a325e2e85141",
    "env/env_systems/base_env.py": (
        "3c7a2dcd6595f3ec39c06e1fefcaaf8903d0848fb17e8bdf56daaf35c46a6278"
    ),
    "env/env_systems/math_env.py": (
        "c5d45b3354433ce188ada6d1e638d9cbb794f8c6e42de62c4d8cd1967f7e2d74"
    ),
    "env/env_systems/formal_reasoning_env/llm_backend.py": (
        "27c9898bd56becb732c2eed53036c3de289fe3cb0540ec73e53354abc8c64e67"
    ),
    "env/env_systems/formal_reasoning_env/eval.py": (
        "fa8ac267b397561eccd552a0fe05db5334d604ba3da3fb56eebc6bcaa700cf30"
    ),
}
EVAL = "env/env_systems/formal_reasoning_env/eval.py"
# Exact declared tool in pinned MathAgent._act_with_tools (not a replacement).
NATIVE_REASONING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "reasoning",
            "description": "Solve a math problem using step-by-step reasoning",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "The math problem to solve"}
                },
                "required": ["input"],
            },
        },
    }
]


# Only this authored synthetic case is admitted live. No real questions/gold may
# enter run_math_task until a parent-supplied byte-bound job approval is added.
AUTHORED_LIVE_TASK = ArenaTask(
    -1,
    "formal_reasoning_math",
    (
        ArenaSession(
            "Call the reasoning tool to compute 2+3. Then return the final answer. "
            "This is an authored integration exercise, not a benchmark question."
        ),
    ),
    ("5",),
)


class NativeAbort(BaseException):
    """Bypass upstream Exception retry/fallback loops; caught by campaign owner."""


def _json(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")


def _append(path, value):
    with path.open("ab") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")


def _sources(root):
    root = Path(root).resolve()
    # Use original LF git objects, not silently normalized Windows worktree bytes.
    source = {
        name: subprocess.check_output(["git", "-C", str(root), "show", PIN + ":" + name])
        for name in SOURCE_PINS
    }
    _verify_sources(source)
    return source


def _verify_sources(source):
    if set(source) != set(SOURCE_PINS) or any(
        hashlib.sha256(source[n]).hexdigest() != h for n, h in SOURCE_PINS.items()
    ):
        raise ValueError("pinned native source bytes mismatch")


@dataclass(frozen=True)
class FrozenToolCapability:
    """Operator-supplied byte pins, NOT a self-attested capability boolean.

    Freeze both files outside the repository before construction. This proves
    the recorded one-attempt wire contract only, not native task quality or
    production memory. Native requests are independently guarded on every call.
    """

    receipt_path: Path
    receipt_sha256: str
    wire_path: Path
    wire_sha256: str

    def validate(self, profile, *, exact_native=False):
        materials = []
        for path, expected in (
            (self.receipt_path, self.receipt_sha256),
            (self.wire_path, self.wire_sha256),
        ):
            if (
                not isinstance(expected, str)
                or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)
            ):
                raise ValueError("qualification requires exact lowercase SHA256 byte pins")
            raw = Path(path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("qualification SHA256 mismatch")
            materials.append(raw)
        try:
            receipt = json.loads(materials[0])
            wire = [json.loads(line) for line in materials[1].splitlines()]
            if len(wire) != 3 or any(r.get("status") != 200 or "error" in r for r in wire):
                raise ValueError("exact successful discovery/tokenizer/generation wire required")
            discovery, tokenizer, generation = wire
            base = profile.base_url
            if [(r["method"], r["url"]) for r in wire] != [
                ("GET", base + "/models"),
                ("POST", base.removesuffix("/v1") + "/tokenize"),
                ("POST", base + "/chat/completions"),
            ]:
                raise ValueError("endpoint/method parity mismatch")
            models = discovery["response"]["data"]
            if not any(
                m["id"] == profile.model and m["max_model_len"] == profile.context_window
                for m in models
            ):
                raise ValueError("served model/context mismatch")
            tp, gp = tokenizer["request"], generation["request"]
            if (
                gp["model"] != profile.model
                or gp["temperature"] != 0
                or gp["tool_choice"] != "auto"
                or gp.get("stream") is not False
                or gp.get("n") != 1
                or gp["chat_template_kwargs"] != {"enable_thinking": False}
                or tp.get("add_generation_prompt") is not True
                or any(
                    tp[k] != gp[k] for k in ("model", "messages", "tools", "chat_template_kwargs")
                )
                or tokenizer["response"]["max_model_len"] != profile.context_window
            ):
                raise ValueError("tokenizer/generation template parity mismatch")
            tools = gp["tools"]
            if exact_native and tools != NATIVE_REASONING_TOOLS:
                raise ValueError("exact native SDK tool schema required")
            if len(tools) != 1 or tools[0]["type"] != "function":
                raise ValueError("reasoning-only qualification required")
            function = tools[0]["function"]
            params = function["parameters"]
            if (
                function["name"] != "reasoning"
                or params["type"] != "object"
                or set(params["properties"]) != {"input"}
                or params["properties"]["input"]["type"] != "string"
                or params["required"] != ["input"]
                or (not exact_native and params["additionalProperties"] is not False)
            ):
                raise ValueError("reasoning input schema mismatch")
            response = generation["response"]
            count = tokenizer["response"]["count"]
            usage = response["usage"]
            if (
                response["model"] != profile.model
                or type(count) is not int
                or count <= 0
                or usage["prompt_tokens"] != count
                or type(usage["completion_tokens"]) is not int
                or not 0 < usage["completion_tokens"] <= gp["max_tokens"]
                or len(response["choices"]) != 1
            ):
                raise ValueError("actual model/usage mismatch")
            choice = response["choices"][0]
            calls = choice["message"]["tool_calls"]
            if choice["finish_reason"] != "tool_calls" or len(calls) != 1:
                raise ValueError("actual tool finish required")
            call = calls[0]
            args = json.loads(call["function"]["arguments"])
            if (
                call["type"] != "function"
                or not isinstance(call["id"], str)
                or not call["id"]
                or call["function"]["name"] != "reasoning"
                or set(args) != {"input"}
                or not isinstance(args["input"], str)
                or not args["input"].strip()
            ):
                raise ValueError("actual reasoning arguments invalid")
            jobs = receipt["jobs"]
            if (
                receipt["accounting"]["calls"] != 1
                or len(jobs) != 1
                or jobs[0]["status"] != "complete"
                or jobs[0]["calls"] != 1
            ):
                raise ValueError("one-attempt accounting mismatch")
            saved = jobs[0]["result"]
            expected = {
                "generation_attempts": 1,
                "prompt_tokens": count,
                "completion_tokens": usage["completion_tokens"],
                "finish_reason": "tool_calls",
                "tool_names": ["reasoning"],
                "tool_was_executed": False,
                "tools_sha256": hashlib.sha256(
                    json.dumps(tools, sort_keys=True).encode()
                ).hexdigest(),
            }
            if any(saved.get(k) != v for k, v in expected.items()):
                raise ValueError("receipt/wire mismatch")
            return {
                **expected,
                "receipt_sha256": self.receipt_sha256,
                "wire_sha256": self.wire_sha256,
            }
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError('tool qualification rejected: ' + safe_error(exc)) from exc


class ArenaTransport(RackClient):
    """Scoped SDK-shaped transport; declared reasoning only, one shared budget.

    Offline remains the default gate. Live requires separately frozen, validated
    one-attempt capability wire evidence and a tiny shared qualification budget.
    """

    def __init__(self, profile, *, receipt, budget, transport, tool_capability):
        if "max_tokens" not in profile.model_fields_set and profile.generation_budgets is None:
            raise ValueError("explicit declared output cap required; native omission is unbounded")
        self.live = type(tool_capability) is FrozenToolCapability
        self.live_transport_kind = "direct-http" if transport is None else "injected-test-transport"
        self.qualification = None
        self._identity_ready = False
        if self.live:
            self.qualification = tool_capability.validate(profile)
            if (
                type(budget) is not JobContext
                or budget.job.call_budget > 8
                or budget.job.seconds_budget > 120
                or budget.budget.max_calls > 8
                or budget.budget.max_seconds > 120
                or budget.calls != 0
                or budget.limiter.max_in_flight != 1
            ):
                raise ValueError(
                    "live qualification budget gate requires <=8 calls/120 seconds, sequential"
                )
        elif (
            not isinstance(transport, httpx.MockTransport) or tool_capability != "offline-authored"
        ):
            raise ValueError("live tool capability/environment qualification gate is closed")
        if budget is None or not callable(getattr(budget, "lease", None)):
            raise ValueError("shared request/token budget required")
        super().__init__(profile, receipt=receipt, budget=budget, transport=transport)

    def preflight(self):
        try:
            response = super().preflight()
            if self.live and not any(
                row["id"] == self.profile.model
                and row.get("max_model_len") == self.profile.context_window
                for row in response["data"]
            ):
                raise SharedRackError("live served model context differs from frozen qualification")
            self._identity_ready = True
            return response
        except BaseException as exc:
            self._latch(exc)
            raise NativeAbort(safe_error(exc)) from exc

    def create(self, role, **kwargs):
        try:
            self.check()
            if role not in {"actor", "reasoning", "judge"}:
                raise FatalRackError("undeclared native generation role")
            cap = self.profile.output_budget("rubric" if role == "judge" else "answer")
            if kwargs.get("model") != self.profile.model or kwargs.get("temperature", 0) != 0:
                raise FatalRackError("native model/temperature differs from declared profile")
            explicit = kwargs.get("max_tokens", kwargs.get("max_completion_tokens", cap))
            if explicit != cap:
                raise FatalRackError("native output cap must match explicit declared role cap")
            if set(kwargs) - {
                "model",
                "temperature",
                "messages",
                "tools",
                "tool_choice",
                "max_tokens",
                "max_completion_tokens",
            }:
                raise FatalRackError("undeclared native request option")
            messages = deepcopy(kwargs["messages"])
            pending = set()
            used_ids = set()
            for message in messages:
                calls = message.get("tool_calls") or []
                if message.get("role") == "tool":
                    tool_id = message.get("tool_call_id")
                    if (
                        tool_id not in pending
                        or message.get("name") != "reasoning"
                        or not isinstance(message.get("content"), str)
                        or not message["content"]
                    ):
                        raise FatalRackError("orphan/invalid tool message")
                    pending.remove(tool_id)
                elif pending:
                    raise FatalRackError("missing tool message before next turn")
                if calls:
                    if message.get("role") != "assistant":
                        raise FatalRackError("tool message calls require assistant role")
                    for call in calls:
                        tool_id = call.get("id")
                        if (
                            not isinstance(tool_id, str)
                            or not tool_id
                            or tool_id in used_ids
                            or call.get("type") != "function"
                            or call.get("function", {}).get("name") != "reasoning"
                        ):
                            raise FatalRackError("invalid tool message declaration")
                        args = json.loads(call["function"]["arguments"])
                        if (
                            not isinstance(args, dict)
                            or set(args) != {"input"}
                            or not isinstance(args["input"], str)
                            or not args["input"].strip()
                        ):
                            raise FatalRackError("invalid tool message arguments")
                        pending.add(tool_id)
                        used_ids.add(tool_id)
            if pending:
                raise FatalRackError("missing tool message results")
            tools = kwargs.get("tools")
            if (
                tools is not None
                and (
                    role != "actor"
                    or tools != NATIVE_REASONING_TOOLS
                    or kwargs.get("tool_choice") != "auto"
                )
            ) or (tools is None and "tool_choice" in kwargs):
                raise FatalRackError(
                    "only exact declared native reasoning tool/schema/auto choice admitted"
                )
            if self.live and not self._identity_ready:
                self.preflight()
            template = {
                "model": self.profile.model,
                "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if tools is not None:
                template["tools"] = tools
                template["tool_choice"] = "auto"
            tokenized = self._request("POST", "/tokenize", template)
            if self.live and tokenized.get("max_model_len") != self.profile.context_window:
                raise SharedRackError("live tokenizer context differs from frozen qualification")
            tokens = tokenized.get("count")
            self.admit_context(tokens, output_budget=cap, input_chars=len(json.dumps(template)))
            payload = {
                "model": self.profile.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": cap,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if tools is not None:
                payload.update(tools=tools, tool_choice="auto")
            with self.generation_lease(tokens + cap + self.profile.schema_reserve_tokens):
                self.budget.begin_call()
                response = self._request("POST", "/chat/completions", payload)
                self.check()
                if (
                    response.get("model") != self.profile.model
                    or len(response.get("choices", [])) != 1
                ):
                    raise SharedRackError("native response model/choices mismatch")
                choice = response["choices"][0]
                message = choice["message"]
                calls = message.get("tool_calls") or []
                if choice.get("finish_reason") != ("tool_calls" if calls else "stop"):
                    raise FatalRackError("incomplete native response")
                if calls:
                    if not tools:
                        raise FatalRackError("tool call without declaration")
                    seen = set()
                    for call in calls:
                        args = json.loads(call["function"]["arguments"])
                        if (
                            call.get("type") != "function"
                            or call["function"]["name"] != "reasoning"
                            or not isinstance(call.get("id"), str)
                            or not call["id"]
                            or call["id"] in seen
                            or not isinstance(args, dict)
                            or set(args) != {"input"}
                            or not isinstance(args["input"], str)
                            or not args["input"].strip()
                        ):
                            raise FatalRackError("undeclared/invalid reasoning tool call")
                        seen.add(call["id"])
                elif not isinstance(message.get("content"), str) or not message["content"].strip():
                    raise FatalRackError("empty native response")
                if role == "judge" and message["content"].strip().lower() not in {"yes", "no"}:
                    raise FatalRackError("native judge requires strict yes/no")
                usage = response.get("usage", {})
                if type(usage.get("prompt_tokens")) is not int or usage["prompt_tokens"] != tokens:
                    raise SharedRackError("served tokenizer/generation token count mismatch")
                if (
                    type(usage.get("completion_tokens")) is not int
                    or not 0 <= usage["completion_tokens"] <= cap
                ):
                    raise FatalRackError("actual completion usage outside declared cap")
                result_message = SimpleNamespace(
                    content=message.get("content"),
                    tool_calls=[
                        SimpleNamespace(id=c["id"], function=SimpleNamespace(**c["function"]))
                        for c in calls
                    ],
                )
                return SimpleNamespace(choices=[SimpleNamespace(message=result_message)])
        except BaseException as exc:
            self._latch(exc)
            raise NativeAbort(safe_error(exc)) from exc

    def sdk(self, role):
        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: self.create(role, **kwargs))
            )
        )


@contextmanager
def native_math(source, transport, *, mode="bounded-native-tools", max_tool_iterations=1):
    """Execute byte-pinned modules privately; never patch installed SDK globals."""
    _verify_sources(source)
    prefix = "_arena_" + uuid.uuid4().hex
    modules = {}

    def package(name):
        full = prefix + ("." + name if name else "")
        mod = ModuleType(full)
        mod.__path__ = []
        sys.modules[full] = mod
        modules[name] = mod

    for name in ["", "agent", "env", "env.env_systems", "env.env_systems.formal_reasoning_env"]:
        package(name)

    real_import = builtins.__import__

    def scoped_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "formal_reasoning_env.llm_backend":
            return modules["env.env_systems.formal_reasoning_env.llm_backend"]
        if name == "openai":
            return SimpleNamespace(OpenAI=lambda **kwargs: transport.sdk("actor"))
        if name == "sys":
            return SimpleNamespace(path=list(sys.path))  # isolate native sys.path.insert
        return real_import(name, globals, locals, fromlist, level)

    def load(name, filename):
        mod = ModuleType(prefix + "." + name)
        mod.__file__ = "pinned-git:" + PIN + "/" + filename
        mod.__package__ = mod.__name__.rpartition(".")[0]
        mod.__dict__["__builtins__"] = dict(vars(builtins), __import__=scoped_import)
        sys.modules[mod.__name__] = mod
        modules[name] = mod
        exec(compile(source[filename], mod.__file__, "exec"), mod.__dict__)
        return mod

    try:
        load("agent.base", "agent/base.py")
        load("agent.base_agent", "agent/base_agent.py")
        backend = load(
            "env.env_systems.formal_reasoning_env.llm_backend",
            "env/env_systems/formal_reasoning_env/llm_backend.py",
        )
        load("env.env_systems.base_env", "env/env_systems/base_env.py")
        agent_module = load("agent.math", "agent/math.py")
        env_module = load("env.env_systems.math_env", "env/env_systems/math_env.py")

        def backend_factory(role):
            def factory(**kwargs):
                obj = backend.OpenAIBackend.__new__(backend.OpenAIBackend)
                obj.client = transport.sdk(role)
                obj.model_name = transport.profile.model
                obj.temperature = 0
                obj.max_tokens = transport.profile.output_budget(
                    "rubric" if role == "judge" else "answer"
                )
                return obj

            return factory

        agent_module.create_backend = backend_factory("reasoning")
        env_module.create_backend = backend_factory("judge")
        agent = agent_module.MathAgent(
            model_name=transport.profile.model, max_tokens=transport.profile.output_budget("answer")
        )
        # Explicit bounded retry/iteration policy; ignore mutable process env.
        agent._env_int = lambda name, default, minimum=1: (
            max_tool_iterations if name == "MATH_AGENT_MAX_TOOL_ITERATIONS" else 1
        )
        if mode == "adapted-native-direct-reasoning":
            agent.backend_name = "scoped-direct-reasoning"
        elif mode != "bounded-native-tools":
            raise ValueError("unknown actor mode")
        env = env_module.MathEnvironment({"model_name": transport.profile.model})
        yield agent, env
    finally:
        for mod in modules.values():
            sys.modules.pop(mod.__name__, None)


def _official_metrics(source, rows, out):
    mod = ModuleType("_arena_metrics")
    exec(compile(source[EVAL], "pinned-git:" + PIN + "/" + EVAL, "exec"), mod.__dict__)
    logger = logging.getLogger(__name__)
    data = {"authored-task": (rows, len(rows))}
    papers, passed, counts, minimum = mod.get_passrate_at_k(data, logger=logger)
    mod.print_result(papers, passed, counts, minimum, str(out), logger=logger)
    return json.loads((out / "all_results.json").read_text(encoding="utf-8"))


def inventory_domains(data_root, pins):
    """Byte-verify all five configs; preserve task/session denominators as UNRUN."""
    from .trajectory_suite import load_arena

    inventory = {}
    for domain in ARENA_DOMAINS:
        tasks = load_arena(Path(data_root), domain=domain, pins=pins)
        inventory[domain] = {
            "status": "unrun",
            "tasks": len(tasks),
            "sessions": sum(len(task.sessions) for task in tasks),
            "data_sha256": pins[f"{domain}/data.jsonl"],
            "reason": "full benchmark environment qualification required",
        }
    return inventory


def document_experience(experience):
    """Document SourceEvent, not a user message; extraction remains mandatory.

    Core calls this SourceEvent(source_type='document'); there is no separate
    DocumentSourceEvent class in the installed API. The JSON document retains
    authored-experience provenance, not evaluator gold or hidden observations.
    """
    from gluomem.write.events import SourceEvent

    return SourceEvent(
        event_id=experience["source_ref"],
        source_ref=experience["source_ref"],
        source_type="document",
        content=_json(experience).decode("utf-8"),
        messages=[],
        kind=None,
    )


class AuthoredExperienceMemory:
    """Explicit authored trace fixture, NOT a memory product adapter."""

    def __init__(self, budget):
        self.budget = budget
        self.entries = []

    def query(self, question):
        self.budget.check()
        return "\n\n".join(entry["content"] for entry in self.entries)

    def insert(self, experience):
        self.budget.check()
        self.entries.append(deepcopy(experience))

    def close(self):
        pass


def run_math_task(
    task: ArenaTask,
    *,
    source,
    transport,
    memory,
    out,
    feedback_in_memory=False,
    mode="bounded-native-tools",
    max_tool_iterations=1,
) -> dict:
    """Run one complete ordered task. All actor/reasoning/judge/memory share budget.

    Production memory admission is intentionally closed until a real document
    SourceEvent extractor/readback adapter is qualified. Never relabel assistant
    experience as chat-user claims. Gold is passed ONLY to native env.step.
    """
    if (
        type(max_tool_iterations) is not int
        or not 1 <= max_tool_iterations <= 8
        or type(feedback_in_memory) is not bool
        or mode not in {"bounded-native-tools", "adapted-native-direct-reasoning"}
    ):
        raise ValueError("campaign admission requires explicit bounded actor/feedback policy")
    if transport.live and (
        task != AUTHORED_LIVE_TASK
        or mode != "bounded-native-tools"
        or max_tool_iterations != 1
        or feedback_in_memory
    ):
        raise ValueError(
            "live quality gate closed: parent-provided frozen task job approval required"
        )
    _verify_sources(source)
    if task.domain != "formal_reasoning_math":
        raise ValueError("full benchmark environment gate: other domains unrun")
    if not isinstance(memory, AuthoredExperienceMemory) or memory.budget is not transport.budget:
        raise ValueError("production DocumentSourceEvents/real extractors gate is closed")
    if not task.sessions or len(task.sessions) != len(task.gold) or len(task.sessions) > 20:
        raise ValueError("official math scorer requires 1..20 aligned sessions")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    for name, material in source.items():
        path = out / "native-source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(material)
    rows = []
    closed = False
    try:
        with native_math(source, transport, mode=mode, max_tool_iterations=max_tool_iterations) as (
            actor,
            env,
        ):
            try:
                env.reset(seed=0)
                transport.preflight()
                for index, (session, gold) in enumerate(zip(task.sessions, task.gold, strict=True)):
                    started = time.perf_counter()
                    transport.check()
                    query = actor.build_prompt(session.question, background=session.background)
                    context = memory.query(query)
                    prompt = actor.build_prompt(
                        session.question, background=session.background, memory_context=context
                    )
                    action = actor.act(prompt)
                    transport.check()
                    observation, reward, info = env.step(action, ground_truth=gold, need_judge=True)
                    # This is post-action official feedback; only native build_memory_entry
                    # selects allowed fields. Never send full judge observation to retrieval.
                    entry = actor.build_memory_entry(
                        session.question,
                        action,
                        observation,
                        reward=reward if feedback_in_memory else None,
                    )
                    experience = {
                        "content": entry,
                        "authorship": "agent",
                        "source_ref": f"memoryarena/{task.domain}/{task.id}/session/{index}",
                        "session_index": index,
                        "feedback_in_memory": feedback_in_memory,
                        "native_source_commit": PIN,
                    }
                    memory.insert(experience)
                    row = {
                        "query_id": index,
                        "response": observation["final"],
                        "is_correct": reward,
                        "memory_context": context,
                        "time": time.perf_counter() - started,
                        "action": action,
                        "observation": observation,
                        "experience": experience,
                    }
                    _append(out / "steps.jsonl", row)
                    rows.append(row)
            finally:
                env.close()
                memory.close()
                closed = env.history == [] and env._current_observation is None
        transport.check()
        metrics = _official_metrics(source, rows, out / "scores")
        result = {
            "status": "complete",
            "evidence_kind": (
                "synthetic-live-native-integration-NOT-benchmark"
                if transport.live
                else "authored-offline-integration-not-quality"
            ),
            "protocol": mode + ":memoryarena-math@v1",
            "native_source_commit": PIN,
            "source_sha256": SOURCE_PINS,
            "expected_sessions": len(task.sessions),
            "memory_system": "authored-experience-fixture-not-gluomem",
            "bounds": {
                "actor_max_tokens": transport.profile.output_budget("answer"),
                "reasoning_max_tokens": transport.profile.output_budget("answer"),
                "judge_max_tokens": transport.profile.output_budget("rubric"),
                "max_tool_iterations": max_tool_iterations,
                "reasoning_retries": 0,
                "sdk_retries": 0,
                "feedback_in_memory": feedback_in_memory,
            },
            "native_verifier": "MathEnvironment.judge",
            "deviations": [
                "explicit matched caps for native omitted max_tokens",
                "scoped transport injection; no SDK provider constructor",
                "bounded tool iterations and no native retry/fallback after guard failure",
                (
                    "live authored task; no benchmark-quality or production-memory qualification"
                    if transport.live
                    else "authored replies; no deployment or answer-quality qualification"
                ),
            ],
            "calls": transport.budget.calls,
            "native_closed": closed,
            "metrics": metrics,
            "coverage": {
                domain: {"status": "unrun", "reason": "no official dataset execution admitted"}
                for domain in ARENA_DOMAINS
            },
            "http_sha256": hashlib.sha256(transport.receipt.read_bytes()).hexdigest(),
            "steps_sha256": hashlib.sha256((out / "steps.jsonl").read_bytes()).hexdigest(),
        }
        if transport.live:
            result.update(
                tool_capability=transport.qualification,
                live_transport_kind=transport.live_transport_kind,
                timing_scope=(
                    "Separate <=8-call/120s qualification budget; "
                    "concurrent campaigns may be active; no controlled latency claim."
                ),
            )
        transport.check()
        (out / "result.json").write_bytes(_json(result))
        return result
    except BaseException as exc:
        transport._latch(exc)
        (out / "failure.json").write_bytes(
            _json(
                {
                    "status": "incomplete",
                    "calls": transport.budget.calls,
                    "completed_sessions": len(rows),
                    "native_closed": closed,
                    "error": safe_error(exc),
                }
            )
        )
        raise RuntimeError("MemoryArena incomplete; inspect failure.json") from exc


def rescore(bundle):
    """Dataset-free native scorer replay; source and complete-step verification."""
    bundle = Path(bundle)
    result = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    if (
        (bundle / "failure.json").exists()
        or result["status"] != "complete"
        or not result["native_closed"]
    ):
        raise ValueError("incomplete bundle")
    source = {name: (bundle / "native-source" / name).read_bytes() for name in SOURCE_PINS}
    _verify_sources(source)
    if hashlib.sha256((bundle / "http.jsonl").read_bytes()).hexdigest() != result["http_sha256"]:
        raise ValueError("HTTP evidence hash mismatch")
    raw = (bundle / "steps.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != result["steps_sha256"]:
        raise ValueError("saved steps hash mismatch")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != result["expected_sessions"] or [r["query_id"] for r in rows] != list(
        range(len(rows))
    ):
        raise ValueError("incomplete session coverage")
    wire = [json.loads(line) for line in (bundle / "http.jsonl").read_bytes().splitlines()]
    generations = [row for row in wire if row.get("url", "").endswith("/chat/completions")]
    judges = [
        row
        for row in generations
        if row["request"]["messages"][0].get("content")
        == (
            "You are a helpful assistant that judges the equivalence "
            "of two mathematical expressions."
        )
    ]
    if len(generations) != result["calls"] or len(judges) != len(rows):
        raise ValueError("native judge feedback/request coverage mismatch")
    for row, judge in zip(rows, judges, strict=True):
        verdict = judge["response"]["choices"][0]["message"]["content"].strip().lower()
        observation = row["observation"]
        if (
            verdict not in {"yes", "no"}
            or observation["judge_result"] != verdict
            or type(row["is_correct"]) is not bool
            or row["is_correct"] != (verdict == "yes")
            or observation["reward"] is not row["is_correct"]
            or observation["final"] != row["response"]
        ):
            raise ValueError("native judge feedback differs from saved HTTP/step")
    replay = bundle / ("rescore-" + uuid.uuid4().hex)
    metrics = _official_metrics(source, rows, replay)
    if metrics != result["metrics"]:
        raise ValueError("native metrics differ from saved scores")
    return metrics


def run_offline_smoke(*, native_root, out):
    """Two AUTHORED sessions; real native methods, fake HTTP at network boundary."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    source = _sources(native_root)
    profile = RackProfile(
        base_url="https://example.invalid/v1",
        model="authored-offline",
        max_tokens=128,
        max_generation_calls=8,
    )
    outer = CampaignBudget(max_calls=8, max_seconds=120)
    job = Job("authored-math", "authored-math", out, 8, 120)
    outer.reserve(job)
    budget = JobContext(job, outer, RequestLimiter(max_in_flight=1, max_in_flight_tokens=8192))
    counter = [0]

    def fake_http(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": profile.model}]})
        json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100})
        call = counter[0]
        counter[0] += 1
        if call % 4 == 0:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"reason-{call}",
                        "type": "function",
                        "function": {
                            "name": "reasoning",
                            "arguments": json.dumps({"input": "Compute the authored sum."}),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            content = ("no" if call == 3 else "yes") if call % 4 == 3 else "Authored result: 5"
            message = {"role": "assistant", "content": content}
            finish = "stop"
        return httpx.Response(
            200,
            json={
                "model": profile.model,
                "choices": [{"finish_reason": finish, "message": message}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    client = ArenaTransport(
        profile,
        receipt=out / "http.jsonl",
        budget=budget,
        transport=httpx.MockTransport(fake_http),
        tool_capability="offline-authored",
    )
    # Runner freshness checks precede any HTTP receipt; preflight model identity is
    # inside the first tokenizer-free dispatch in this safe mocked track.
    task = ArenaTask(
        0,
        "formal_reasoning_math",
        (ArenaSession("Compute 2+3."), ArenaSession("Use the preceding result.")),
        ("AUTHORED_GOLD_5", "AUTHORED_GOLD_5"),
    )
    try:
        return run_math_task(
            task, source=source, transport=client, memory=AuthoredExperienceMemory(budget), out=out
        )
    finally:
        client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--native-root", required=True, type=Path)
    smoke.add_argument("--out", required=True, type=Path)
    replay = sub.add_parser("rescore")
    replay.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args(argv)
    result = (
        run_offline_smoke(native_root=args.native_root, out=args.out)
        if args.command == "smoke"
        else rescore(args.bundle)
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
