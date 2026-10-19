"""Pinned native MemoryArena Travel: full freeze, approval, run and offline verification.

No benchmark generation is authorized by planning or offline smoke. Local tools
run in an explicitly supplied isolated interpreter, never the shared rack runtime.
Native actor prompts, memory entries, parser and six-slot PS/SPS/SR are unchanged.
The seven-slot environment reward is retained separately, never substituted for PS.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import re
import subprocess
import sys
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

from gluomem_bench.bench.error_privacy import safe_error, safe_kind

from . import arena_campaign as common
from . import arena_driver as formal

PIN = common.PIN
PREFIX = "env/env_systems/travel_planner_env/"
PROTOCOL = "memoryarena-full-travel-native@v1"
DOMAIN = "group_travel_planner"
SYSTEMS = formal.SYSTEMS
CORE_PINS = {
    "agent/base.py": common.SOURCE_PINS["agent/base.py"],
    "agent/base_agent.py": common.SOURCE_PINS["agent/base_agent.py"],
    "agent/travel_planner.py": "e96c4ef040156a2d9e6ac7c6598b685e4779adb44fb468bcbe48baa9f982b93f",
    "env/env_systems/base_env.py": common.SOURCE_PINS["env/env_systems/base_env.py"],
    "env/env_systems/travel_env.py": (
        "b189f76c4cd0db5984cce29a1ad19cba98ebdd033128679287106a054590797f"
    ),
    "run_travel.py": "3b35937f0caba4c2b614f42d1b0cd0d4f32242973e4942bca5eda89df82e5d5f",
}
TRAVEL_PINS = {
    "__init__.py": "4fefb9389cde377eef28caee57b4d027078794434c81fd7a960db67387cd2e45",
    "clients/__init__.py": "682dc7e5dc790a4d8dc79a391dccc559d44e0be3db7dea018e34e7a5e1da1c4f",
    "clients/anthropic_client.py": (
        "c14735141d231b062c64449f9d49c081fac79aa37a1da61a36198b73fd9ba8c1"
    ),
    "clients/base_client.py": "c0afc2ea343a981593e0373052d70d6b07e7dde339638a414289ac453c148a32",
    "clients/gemini_client.py": "22c308b9fde874ed52a66ea55ac87f5cbc0acbd5ba8ee9c5b442af9065f16ef4",
    "clients/openai_client.py": "8feca4760c6973f732abdab85c968b2f7e8f69f0586f7cc395303ada14b91f60",
    "combination.py": "e69ee0eef84ee6941d833a77db467179c37d61a7b544c4dc77b011f54d22fdae",
    "cost_tracker.py": "abc4d9d09abea2dba11e842d15f70600106d9e8f89104b510b33e2a59be737af",
    "data_loader.py": "8033cc39ab6514f9e79a7044ab84b4dffd1f6a72dcbe4a6281ea49766076c7af",
    "eval.py": "6b1da29d295207898401bf325684b83a0691ecd04cd5211ff5f1fca4d8555942",
    "prompts.py": "984d2c9dc8522ec750eb8ed4da41eb2397997ba80272faac918ba80fe7149ade",
    "tool_executor.py": "bfa838a63d2db4c886dcfdb2682210c6045f8f2b82c387d9a1ba6d4d2f9e0de7",
    "tool_schemas.py": "2dbc9bc0c84d0300a10ce008087cb868005cc1ec581dae7a539ebc524d3edffd",
    "tools/__init__.py": "79a2ad35f3e99728d310d16863ed3bcc2fa0e26db69426a514abb48b6e9b6d19",
    "tools/accommodations.py": "a0ef95e351a43ccd66c1dc75b9d01179742437811f88c35a43b52fe2b37d30e8",
    "tools/attractions.py": "450f40193ed0ee49c23d91a9add98bf9c5deefbe24aadd5c0638d755212d9bfc",
    "tools/cities.py": "007bc063ac04adfa0a1385eceba6ddfb29dad08c94094e567cbbbd69c455d3c9",
    "tools/distance_matrix.py": "096090321aa4ea426dc9615942652c9bd708321fddb1dd8682cb22384daa5147",
    "tools/flights.py": "eca58b6b205acccc1c229cbcfaf2cd14eaea89fb7462d6d1c544f1df21b63ac3",
    "tools/func.py": "93343b097f19f54fc5ac31b721b2565b1852989a7fb4e3784653b2e5ccb4d852",
    "tools/restaurants.py": "a7035ef66074561add879a6ba1af29405df3a9eba640951f856ac2e7c96ee11b",
}
SOURCE_PINS = {**CORE_PINS, **{PREFIX + k: v for k, v in TRAVEL_PINS.items()}}


def verify_sources(source):
    if set(source) != set(SOURCE_PINS) or any(
        hashlib.sha256(source[k]).hexdigest() != v for k, v in SOURCE_PINS.items()
    ):
        raise ValueError("pinned Travel native source mismatch")


def sources(root):
    source = {
        name: subprocess.check_output(["git", "-C", str(root), "show", PIN + ":" + name])
        for name in SOURCE_PINS
    }
    verify_sources(source)
    return source


@contextmanager
def native_travel(source, transport, rows, *, executor, max_steps):
    """Private native modules; no global SDK monkeypatch or provider constructor."""
    verify_sources(source)
    prefix = "_travel_" + uuid.uuid4().hex
    modules = {}
    real_import = builtins.__import__

    def scoped_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openai":
            return SimpleNamespace(OpenAI=lambda **kw: transport.sdk("actor"))
        if name == "datasets":
            return SimpleNamespace(
                load_dataset=lambda *a, **kw: (_ for _ in ()).throw(
                    RuntimeError("network dataset forbidden")
                )
            )
        if not level and (
            name == "env" or name.startswith("env.") or name == "agent" or name.startswith("agent.")
        ):
            return real_import(prefix + "." + name, globals, locals, fromlist, 0)
        return real_import(name, globals, locals, fromlist, level)

    def load(name, filename=None):
        mod = ModuleType(prefix + ("." + name if name else ""))
        mod.__package__ = mod.__name__.rpartition(".")[0]
        mod.__path__ = []
        mod.__dict__["__builtins__"] = dict(vars(builtins), __import__=scoped_import)
        sys.modules[mod.__name__] = mod
        modules[name] = mod
        if filename:
            mod.__file__ = "pinned-git:" + PIN + "/" + filename
            exec(compile(source[filename], mod.__file__, "exec"), mod.__dict__)
        return mod

    env = None
    try:
        for name in (
            "",
            "agent",
            "env",
            "env.env_systems",
            "env.env_systems.travel_planner_env",
            "env.env_systems.travel_planner_env.clients",
        ):
            load(name)
        load("agent.base", "agent/base.py")
        load("agent.base_agent", "agent/base_agent.py")
        load("env.env_systems.base_env", "env/env_systems/base_env.py")
        stem = "env.env_systems.travel_planner_env."
        loader = load(stem + "data_loader", PREFIX + "data_loader.py")
        converted = [loader._convert_row(deepcopy(row)) for row in rows]
        loader.load_travel_data = lambda: deepcopy(converted)
        prompts = load(stem + "prompts", PREFIX + "prompts.py")
        schemas = load(stem + "tool_schemas", PREFIX + "tool_schemas.py")
        load(stem + "clients.base_client", PREFIX + "clients/base_client.py")
        load(stem + "cost_tracker", PREFIX + "cost_tracker.py")
        client_module = load(stem + "clients.openai_client", PREFIX + "clients/openai_client.py")
        load(stem + "tool_executor").ToolExecutor = lambda **kwargs: executor
        actor_module = load("agent.travel_planner", "agent/travel_planner.py")
        # Pin the native OpenAI client irrespective of model-name substring heuristics.
        actor_module.TravelPlannerAgent._create_client = lambda self, model: (
            client_module.OpenAIClient(model_name=model)
        )
        actor = actor_module.TravelPlannerAgent(
            model_name=transport.profile.model, max_steps=max_steps
        )
        if hasattr(transport.profile, "output_budget"):
            original = actor.client.chat_with_tools
            actor.client.chat_with_tools = lambda messages, tools: original(
                messages,
                tools,
                temperature=0.0,
                max_tokens=transport.profile.output_budget("answer"),
            )
        env_module = load("env.env_systems.travel_env", "env/env_systems/travel_env.py")
        env = env_module.TravelPlannerEnvironment({"judgement_mode": "none"})
        scorer = load(stem + "eval", PREFIX + "eval.py")
        combination = load(stem + "combination", PREFIX + "combination.py")
        # Only unchanged utility function ASTs, never the upstream CLI/provider defaults.
        tree = ast.parse(source["run_travel.py"])
        utility = {"re": re}
        wanted = {"parse_all_plans", "format_person_plan"}
        tree.body = [
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        exec(compile(tree, "pinned-git:" + PIN + "/run_travel.py", "exec"), utility)
        yield SimpleNamespace(
            actor=actor,
            env=env,
            rows=converted,
            scorer=scorer,
            combination=combination,
            prompts=prompts,
            tools=schemas.TOOLS,
            **{k: utility[k] for k in wanted},
        )
    finally:
        primary = sys.exception()
        try:
            if env is not None:
                env.close()
        except BaseException as cleanup:
            if primary is None:
                raise
            primary.native_cleanup_diagnostics = exception_diagnostic(cleanup)
        finally:
            for mod in modules.values():
                sys.modules.pop(mod.__name__, None)


def tool_schema(source):
    verify_sources(source)
    namespace = {}
    exec(compile(source[PREFIX + "tool_schemas.py"], "pinned-travel-schemas", "exec"), namespace)
    return namespace["TOOLS"]


def sdk_response(response):
    message = response["choices"][0]["message"]
    return SimpleNamespace(
        usage=SimpleNamespace(**response.get("usage", {})),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=message.get("content"),
                    tool_calls=[
                        SimpleNamespace(id=c["id"], function=SimpleNamespace(**c["function"]))
                        for c in message.get("tool_calls") or []
                    ],
                )
            )
        ],
        to_dict=lambda: deepcopy(response),
    )


class ContextDoesNotFit(ValueError):
    """Only actual tokenizer overflow is a fit-only N/A; never a quality zero."""


@dataclass(frozen=True)
class FrozenTravelTask:
    """Real byte-bound approval, not a callback or self-attested capability boolean."""

    spec_path: Path
    approval_path: Path
    expected_sha256: str

    def check_approval(self):
        raw = Path(self.spec_path).read_bytes()
        approval = json.loads(Path(self.approval_path).read_bytes())
        if (
            hashlib.sha256(raw).hexdigest() != self.expected_sha256
            or approval.get("approved") is not True
            or approval.get("reviewer_kind") not in {"parent", "human"}
            or approval.get("spec_sha256") != self.expected_sha256
        ):
            raise ValueError("frozen Travel approval changed; dispatch forbidden")

    def validate(self, task, profile, max_steps):
        self.check_approval()
        spec = verify_spec(self.spec_path, self.approval_path, allow_started=True)
        if (
            spec["profile"] != profile.model_dump(mode="json")
            or spec["bounds"]["max_steps"] != max_steps
            or {"id": task["id"], "sessions": len(task["questions"]), "input_sha256": digest(task)}
            not in spec["tasks"]
        ):
            raise ValueError("Travel task/profile/steps differ from approved full input")
        return spec


class TravelTransport(formal.FullArenaTransport):
    """Common guarded SDK-shaped transport for the exact six native local tools."""

    def __init__(
        self,
        profile,
        *,
        receipt,
        budget,
        tools,
        transport=None,
        admission=None,
        task=None,
        max_steps=None,
    ):
        import httpx

        from .parallel_campaign import JobContext
        from .rack import RackClient

        if transport is not None and not isinstance(transport, httpx.MockTransport):
            raise ValueError("only offline MockTransport or approved direct HTTP")
        permit_types = (FrozenTravelTask, FrozenTravelContinuationTask)
        if transport is None and type(admission) not in permit_types:
            raise ValueError("independent frozen Travel approval permit required before clients")
        self.permit = admission
        if admission is not None:
            if type(admission) not in permit_types or task is None:
                raise ValueError("exact frozen Travel task approval permit required")
            spec = admission.validate(task, profile, max_steps)
            if (
                type(budget) is not JobContext
                or budget.calls != 0
                or budget.job.call_budget != spec["bounds"]["calls_per_task_method"]
                or budget.job.seconds_budget != spec["bounds"]["seconds_per_task_method"]
                or budget.limiter.max_in_flight != 1
            ):
                raise ValueError("approved whole-task sequential budget required")
            if type(admission) is FrozenTravelContinuationTask:
                remaining = admission.check_approval()["plan"]["remaining_allowance"]
                outer = budget.budget
                if (
                    outer.max_calls != remaining["calls"]
                    or outer.max_seconds != remaining["seconds"]
                    or outer.prior_calls
                    or outer.prior_elapsed_seconds
                ):
                    raise ValueError("approved reduced continuation outer guard required")
        if budget is None or not callable(getattr(budget, "lease", None)):
            raise ValueError("shared whole-task budget required")
        self.tools = deepcopy(tools)
        self.live = transport is None
        self._identity_ready = False
        self.qualification = None
        RackClient.__init__(self, profile, receipt=receipt, budget=budget, transport=transport)

    def check(self):
        from .rack import RackClient

        if self.permit is not None:
            authority = self.permit.check_approval()
            if type(self.permit) is FrozenTravelContinuationTask:
                remaining = authority["plan"]["remaining_allowance"]
                outer = self.budget.budget
                if (
                    outer.max_calls != remaining["calls"]
                    or outer.max_seconds != remaining["seconds"]
                    or outer.prior_calls
                    or outer.prior_elapsed_seconds
                    or outer.deadline != outer.started + remaining["seconds"]
                ):
                    raise ValueError("approved reduced continuation guard changed at dispatch")
        RackClient.check(self)

    def _latch(self, exc):
        # The shared RackClient classifies generation timeouts as job-local;
        # do not inherit the older formal adapter's broad HTTPError escalation.
        from .rack import RackClient

        RackClient._latch(self, exc)

    def create(self, role, **kwargs):
        from jsonschema import validate

        from .rack import FatalRackError, SharedRackError

        try:
            self.check()
            cap = self.profile.output_budget("answer")
            if (
                role != "actor"
                or kwargs.get("model") != self.profile.model
                or kwargs.get("temperature") != 0
                or kwargs.get("max_tokens") != cap
                or kwargs.get("tools") != self.tools
                or kwargs.get("tool_choice") != "auto"
                or set(kwargs)
                != {"model", "temperature", "max_tokens", "messages", "tools", "tool_choice"}
            ):
                raise FatalRackError("undeclared Travel model/role/cap/options/tool schema")
            schemas = {t["function"]["name"]: t["function"]["parameters"] for t in self.tools}

            def check_calls(calls):
                seen = set()
                for call in calls:
                    name = call.get("function", {}).get("name")
                    if (
                        call.get("type") != "function"
                        or name not in schemas
                        or not isinstance(call.get("id"), str)
                        or not call["id"]
                        or call["id"] in seen
                    ):
                        raise FatalRackError("undeclared local tool call")
                    args = json.loads(call["function"]["arguments"])
                    validate(args, schemas[name])
                    if set(args) - set(schemas[name].get("properties", {})):
                        raise FatalRackError("undeclared local tool arguments")
                    seen.add(call["id"])
                return seen

            messages = deepcopy(kwargs["messages"])
            pending, used = set(), set()
            for message in messages:
                if message.get("role") == "tool":
                    ident = message.get("tool_call_id")
                    if ident not in pending or not isinstance(message.get("content"), str):
                        raise FatalRackError("orphan tool result")
                    pending.remove(ident)
                else:
                    if pending:
                        raise FatalRackError("missing local tool results")
                    calls = message.get("tool_calls") or []
                    if calls and message.get("role") != "assistant":
                        raise FatalRackError("tool calls require assistant")
                    pending = check_calls(calls)
                    if used & pending:
                        raise FatalRackError("duplicate tool call IDs")
                    used |= pending
            if pending:
                raise FatalRackError("missing local tool results")
            if not self._identity_ready:
                self.preflight()
            template = {
                "model": self.profile.model,
                "messages": messages,
                "tools": self.tools,
                "tool_choice": "auto",
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            tokenized = self._request("POST", "/tokenize", template)
            tokens = tokenized.get("count")
            if (
                tokenized.get("max_model_len") != self.profile.context_window
                or type(tokens) is not int
                or tokens <= 0
            ):
                raise SharedRackError("native Travel tokenizer identity/count mismatch")
            if tokens + cap + self.profile.schema_reserve_tokens > self.profile.context_window:
                raise ContextDoesNotFit(
                    "actual served-tokenizer Travel context plus output reserve does not fit"
                )
            self.admit_context(
                tokens, output_budget=cap, input_chars=len(json.dumps(template, ensure_ascii=False))
            )
            payload = {k: v for k, v in template.items() if k != "add_generation_prompt"}
            payload.update(temperature=0, max_tokens=cap)
            with self.generation_lease(tokens + cap + self.profile.schema_reserve_tokens):
                self.budget.begin_call()
                response = self._request("POST", "/chat/completions", payload)
            self.check()
            if response.get("model") != self.profile.model or len(response.get("choices", [])) != 1:
                raise SharedRackError("Travel served generation identity mismatch")
            choice, usage = response["choices"][0], response.get("usage", {})
            message = choice["message"]
            calls = message.get("tool_calls") or []
            if choice.get("finish_reason") != ("tool_calls" if calls else "stop"):
                raise FatalRackError("incomplete native Travel response")
            check_calls(calls)
            if not calls and (
                not isinstance(message.get("content"), str) or not message["content"].strip()
            ):
                raise FatalRackError("empty native Travel response")
            if usage.get("prompt_tokens") != tokens or type(usage.get("prompt_tokens")) is not int:
                raise SharedRackError("Travel tokenizer/generation usage mismatch")
            if (
                type(usage.get("completion_tokens")) is not int
                or not 0 <= usage["completion_tokens"] <= cap
            ):
                raise SharedRackError("Travel completion accounting mismatch")
            return sdk_response(response)
        except BaseException as exc:
            self._latch(exc)
            raise common.NativeAbort(safe_error(exc)) from exc


def file_record(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def resource_inventory(readiness):
    report = json.loads(Path(readiness).read_bytes())
    if report["native_commit"] != PIN or report["status"] != "actual-native-local-tools-ready":
        raise ValueError("actual native local Travel readiness required")
    files = [file_record(readiness)]
    for record in report["source_files"]:
        observed = file_record(record["path"])
        if observed["sha256"] != record["sha256"] or observed["bytes"] != record["bytes"]:
            raise ValueError("native resource byte hash differs")
        if record["source"].endswith(".py") and observed["sha256"] != SOURCE_PINS.get(
            record["source"]
        ):
            raise ValueError("native runtime source differs from pinned commit")
        files.append(observed)
    runtime = Path(report["source_files"][0]["path"]).parent
    flight = file_record(runtime / "database/flights/clean_Flights_2022.csv")
    if (
        flight["sha256"] != report["flight"]["sha256"]
        or flight["bytes"] != report["flight"]["bytes"]
    ):
        raise ValueError("native flight CSV byte hash differs")
    files.append(flight)
    return {"native_commit": PIN, "files": files}


def exception_diagnostic(exc):
    """Stack locations and types only: never source lines, headers or locals."""
    import traceback

    result, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        result.append(
            {
                "type": safe_kind(exc),
                "stack": [
                    {
                        "file": Path(frame.filename).name,
                        "line": frame.lineno,
                        "function": frame.name,
                    }
                    for frame in traceback.extract_tb(exc.__traceback__)
                ],
            }
        )
        exc = exc.__cause__ or exc.__context__
    return result


class NativeTools:
    """Persistent isolated local-only worker; all waits spend the task deadline."""

    def __init__(self, *, python, helper, readiness, budget):
        import queue
        import threading
        import time

        self.budget = budget
        self._clock = time.monotonic
        self._deadline = self._clock() + budget.remaining()
        self._writer = None
        self._send_queue = None
        self._reader_failures = []
        self.queue = queue.Queue()
        self.native_closed = False
        self._failure = None
        self._cleanup = []
        # Leave reaping time inside (not in addition to) the original deadline.
        self._cleanup_reserve = min(0.1, max(0, budget.remaining()) / 10)
        budget.check()
        self.process = subprocess.Popen(
            [
                str(python),
                "-I",
                "-X",
                "utf8",
                str(Path(helper).resolve()),
                "--readiness",
                str(Path(readiness).resolve()),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1,
        )

        def reader():
            try:
                for line in self.process.stdout:
                    self.queue.put(line)
            except BaseException as exc:
                self._reader_failures.append(exc)
                self.queue.put(exc)
                if self._send_queue is not None:
                    self._send_queue.put(exc)
            finally:
                self.queue.put(None)

        self._stderr_tail = b""
        self._stderr_bytes = 0
        self._stderr_hash = hashlib.sha256()

        def stderr_reader():
            import codecs

            decoder = codecs.getincrementaldecoder("utf-8")("strict")
            try:
                while chunk := self.process.stderr.buffer.read1(4096):
                    self._stderr_bytes += len(chunk)
                    self._stderr_hash.update(chunk)
                    self._stderr_tail = (self._stderr_tail + chunk)[-16384:]
                    decoder.decode(chunk)
                decoder.decode(b"", final=True)
            except BaseException as exc:
                self._reader_failures.append(exc)
                self.queue.put(exc)
                if self._send_queue is not None:
                    self._send_queue.put(exc)

        self._reader = threading.Thread(target=reader, daemon=True)
        self._stderr_reader = threading.Thread(target=stderr_reader, daemon=True)
        self._stderr_reader.start()
        self._reader.start()
        try:
            if self._receive().get("ready") is not True:
                raise ValueError("native local tool initialization failed")
        except BaseException as exc:
            self._failure = exc
            self.close()
            exc.native_tools_diagnostics = self.diagnostics()
            raise

    def _remaining(self):
        # JobContext.remaining() is intentionally unavailable after a job latches.
        # Cleanup still owns the original (earliest job/outer) absolute deadline.
        return max(0, self._deadline - self._clock())

    def _send(self, line):
        import queue
        import threading

        sent = self._send_queue = queue.Queue()
        if self._reader_failures:
            raise self._reader_failures[0]

        def writer():
            try:
                self.process.stdin.write(line)
                self.process.stdin.flush()
            except BaseException as exc:
                sent.put(exc)
            else:
                sent.put(None)

        self._writer = threading.Thread(target=writer, daemon=True)
        self._writer.start()
        try:
            error = sent.get(timeout=max(0, self._remaining() - self._cleanup_reserve))
        except queue.Empty:
            raise TimeoutError("native local tool stdin deadline") from None
        if error is not None:
            raise error

    def _receive(self):
        import queue

        self.budget.check()
        if self._reader_failures:
            raise self._reader_failures[0]
        try:
            line = self.queue.get(timeout=max(0, self.budget.remaining() - self._cleanup_reserve))
        except queue.Empty:
            raise TimeoutError("native local tool task deadline") from None
        if self._reader_failures:
            raise self._reader_failures[0]
        if isinstance(line, BaseException):
            raise line
        if line is None:
            raise RuntimeError("native local tool worker exited")
        self.budget.check()
        return json.loads(line)

    def execute(self, name, args):
        if name not in {
            "FlightSearch",
            "RestaurantSearch",
            "AccommodationSearch",
            "AttractionSearch",
            "DistanceMatrix",
            "CitySearch",
        }:
            raise ValueError("only declared native local tools permitted")
        if self._failure is not None:
            raise self._failure
        try:
            self.budget.check()
            self._send(json.dumps({"name": name, "args": args}, ensure_ascii=False) + "\n")
            return self._receive()["result"]
        except BaseException as exc:
            self._failure = exc
            raise

    def diagnostics(self):
        # Legacy authored helper tracebacks are accepted read-only. Never publish
        # arbitrary stderr, exception messages, source lines or request/secret locals.
        tail = self._stderr_tail.decode("utf-8", errors="replace")
        worker = {
            "types": re.findall(
                r"^([A-Za-z_][A-Za-z0-9_]*): native local worker failed\r?$", tail, re.M
            )
            or re.findall(r"^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Exit)):", tail, re.M),
            "stack": [
                {"file": Path(filename).name, "line": int(line), "function": function}
                for filename, line, function in re.findall(
                    r'  File "([^"\r\n]+)", line (\d+), in ([A-Za-z_][A-Za-z0-9_.<>]*)', tail
                )
            ],
        }
        return {
            "primary": exception_diagnostic(self._failure)[0] if self._failure else None,
            "cleanup": list(self._cleanup),
            "worker": worker,
            "stderr_bytes": self._stderr_bytes,
            "stderr_retained_bytes": len(self._stderr_tail),
            "stderr_sha256": self._stderr_hash.hexdigest(),
            "returncode": self.process.poll(),
            "native_closed": self.native_closed,
        }

    def close(self):
        if self.native_closed:
            return
        active = sys.exception() or self._failure
        reader_error = self._reader_failures[0] if self._reader_failures else None
        if self._failure is None and reader_error is not None:
            self._failure = reader_error
        errors = []

        def attempt(action):
            try:
                action()
            except BaseException as exc:
                errors.append(exc)
                self._cleanup.extend(exception_diagnostic(exc))

        if self.process.poll() is None:
            if (
                active is None
                and reader_error is None
                and self._remaining() > self._cleanup_reserve
            ):
                attempt(lambda: self._send('{"close":true}\n'))
                attempt(
                    lambda: self.process.wait(
                        timeout=max(0, self._remaining() - self._cleanup_reserve)
                    )
                )
            elif self.process.stdin.closed:
                attempt(lambda: self._send('{"close":true}\n'))
            if self.process.poll() is None:
                attempt(self.process.kill)
                attempt(lambda: self.process.wait(timeout=self._remaining()))
        self._reader.join(timeout=self._remaining())
        self._stderr_reader.join(timeout=self._remaining())
        if self._writer is not None:
            self._writer.join(timeout=self._remaining())
        writer_alive = self._writer is not None and self._writer.is_alive()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if (
                (stream is self.process.stdout and self._reader.is_alive())
                or (stream is self.process.stderr and self._stderr_reader.is_alive())
                or (stream is self.process.stdin and writer_alive)
            ):
                continue  # Never take a TextIO lock from a blocked I/O thread.
            attempt(stream.close)
        self.native_closed = (
            self.process.poll() is not None
            and not self._reader.is_alive()
            and not self._stderr_reader.is_alive()
            and not writer_alive
            and all(
                s.closed for s in (self.process.stdin, self.process.stdout, self.process.stderr)
            )
        )
        if not self.native_closed:
            attempt(lambda: (_ for _ in ()).throw(RuntimeError("native local cleanup unverified")))
        if self.process.returncode != 0 and active is None:
            attempt(lambda: (_ for _ in ()).throw(RuntimeError("native local tool worker failed")))
        if active is None and self._reader_failures:
            self._failure = self._reader_failures[0]
            raise self._failure
        if errors and active is None:
            raise RuntimeError("native local tool cleanup failed") from errors[0]


def digest(value):
    return hashlib.sha256(common._json(value)).hexdigest()


def validate_raw(row):
    if (
        type(row.get("id")) is not int
        or not isinstance(row.get("questions"), list)
        or not row["questions"]
        or any(not isinstance(q, str) or not q.strip() for q in row["questions"])
        or len(row["questions"]) != len(row.get("answers", []))
        or not isinstance(row.get("base_person"), dict)
    ):
        raise ValueError("aligned native Travel task required")
    return row


def validate_submission(submission, converted):
    """Close official evaluator's intersection-only denominator loophole."""
    if len(submission) != len(converted) or [r["id"] for r in submission] != [
        r["id"] for r in converted
    ]:
        raise ValueError("exact complete submission task coverage required")
    for submitted, row in zip(submission, converted, strict=True):
        persons = submitted.get("persons", [])
        questions = row["questions"]
        if len(persons) != len(questions):
            raise ValueError("exact complete submission session coverage required")
        for person, question in zip(persons, questions, strict=True):
            if (
                person.get("person_idx") != question["round_idx"]
                or person.get("name") != question["name"]
                or person.get("query") != question["query"]
                or "plan" not in person
            ):
                raise ValueError("ordered native submission occurrence mismatch")


def run_task(raw, *, source, client, memory, executor, out, max_steps, system):
    """Run ordered native person sessions, retaining full budget and failure evidence."""
    import time

    validate_raw(raw)
    verify_sources(source)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows, attempted, closed, acquired = [], None, False, False
    cleanup_diagnostics = []
    for name, material in source.items():
        target = out / "native-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(material)
    (out / "input-task.json").write_bytes(common._json(raw))
    (out / "effective-profile.json").write_bytes(
        common._json(client.profile.model_dump(mode="json"))
    )
    try:
        if callable(memory):
            memory = memory(client.budget, out)
        acquired = True
        if memory.budget is not client.budget:
            raise ValueError(
                "production ingestion and native actor require same whole-task counter"
            )
        with native_travel(
            source, client, [raw], executor=executor, max_steps=max_steps
        ) as runtime:
            actor, env, converted = runtime.actor, runtime.env, runtime.rows[0]
            try:
                env.reset(seed=raw["id"])
                client.preflight()
                base = converted["base_person"]
                # Identical common native base context for EVERY method. Unlike the
                # reference no-memory branch, prior plans are never implicitly retained.
                actor.set_base_person(
                    base["name"],
                    base["query"],
                    runtime.format_person_plan(base["name"], base["daily_plans"]),
                )
                accumulated, all_results = "", []
                for index, question in enumerate(converted["questions"]):
                    attempted = index
                    started = time.perf_counter()
                    client.check()
                    context = memory.query(question["query"])
                    actor.prepare_for_person(
                        question["name"],
                        question["round_idx"],
                        include_previous_plans=False,
                        memory_context=context or None,
                        memory_system=None,
                    )
                    action = actor.act(question["query"])
                    if not actor.last_result.success or not action:
                        raise ValueError(
                            "native Travel actor exhausted step bound without submission"
                        )
                    client.check()
                    gold = {
                        "name": question["name"],
                        "daily_plans": converted["answers"][index]["daily_plans"],
                        "judgement_mode": "none",
                    }
                    observation, reward, info = env.step(action, ground_truth=gold, need_judge=True)
                    entry = actor.build_memory_entry(
                        question["query"], action, observation=observation, reward=reward
                    )
                    experience = {
                        "content": entry,
                        "authorship": "agent",
                        "source_ref": f"memoryarena/{DOMAIN}/{raw['id']}/session/{index}",
                        "session_index": index,
                        "feedback_in_memory": False,
                        "native_source_commit": PIN,
                    }
                    readback = memory.insert(experience)
                    if (
                        not isinstance(readback, dict)
                        or readback.get("readback_verified") is not True
                    ):
                        raise ValueError("committed native document readback required")
                    row = {
                        "query_id": index,
                        "action": action,
                        "observation": observation,
                        "reward": reward,
                        "info": info,
                        "memory_context": context,
                        "experience": experience,
                        "memory_readback": readback,
                        "native_result": asdict(actor.last_result),
                        "seconds": time.perf_counter() - started,
                    }
                    common._append(out / "steps.jsonl", row)
                    rows.append(row)
                    accumulated += ("\n\n" if accumulated else "") + action
                    all_results.append(
                        runtime.parse_all_plans(accumulated, converted["questions"][: index + 1])
                    )
                generated = {
                    f"{client.profile.model}_sole-planning_results": all_results[-1],
                    "all_results": all_results,
                }
                generated_dir = out / "generated"
                generated_dir.mkdir()
                (generated_dir / f"generated_plan_{raw['id']}.json").write_bytes(
                    common._json(generated)
                )
                # Neutral filename avoids model slashes becoming filesystem paths.
                generated["native_sole-planning_results"] = generated.pop(
                    f"{client.profile.model}_sole-planning_results"
                )
                (generated_dir / f"generated_plan_{raw['id']}.json").write_bytes(
                    common._json(generated)
                )
                native_submission = runtime.combination.combine(
                    "native", str(generated_dir), str(out / "combined")
                )
                submission = [
                    json.loads(line) for line in Path(native_submission).read_bytes().splitlines()
                ]
                validate_submission(submission, runtime.rows)
                (out / "submission.jsonl").write_bytes(
                    b"".join(common._json(r).replace(b"\n", b"") + b"\n" for r in submission)
                )
                metrics = runtime.scorer.evaluate(
                    str(out / "submission.jsonl"),
                    model_name=client.profile.model,
                    memory_system=system,
                )
            finally:
                primary, cleanup_errors = sys.exception(), []
                for owner, closer in (
                    ("environment", env.close),
                    ("memory", memory.close),
                    ("executor", getattr(executor, "close", None)),
                ):
                    if not callable(closer):
                        continue
                    try:
                        closer()
                        if owner == "memory":
                            acquired = False
                    except BaseException as cleanup:
                        cleanup_errors.append(cleanup)
                        cleanup_diagnostics.extend(exception_diagnostic(cleanup))
                closed = (
                    not cleanup_errors
                    and not acquired
                    and env.history == []
                    and env._current_observation is None
                    and getattr(executor, "native_closed", True) is True
                )
                if cleanup_errors and primary is None:
                    raise cleanup_errors[0]
                if not closed and primary is None:
                    raise RuntimeError("native Travel cleanup not verified")
        client.check()
        result = {
            "protocol": PROTOCOL,
            "status": "complete",
            "task_id": raw["id"],
            "system": system,
            "input_sha256": digest(raw),
            "expected_sessions": len(raw["questions"]),
            "completed_sessions": len(rows),
            "calls": client.budget.calls,
            "native_closed": closed,
            "metrics": metrics,
            "max_steps": max_steps,
            "transport": "live" if client.live else "offline-authored-only",
            "artifact_sha256": {
                str(p.relative_to(out)).replace("\\", "/"): file_record(p)["sha256"]
                for p in sorted(out.rglob("*"))
                if p.is_file()
            },
        }
        (out / "result.pending.json").write_bytes(common._json(result))
        verify_task(out, manifest_name="result.pending.json")
        client.check()
        (out / "result.pending.json").replace(out / "result.json")
        return result
    except BaseException as exc:
        cleanup_diagnostics.extend(getattr(exc, "native_cleanup_diagnostics", []))
        if acquired:
            try:
                memory.close()
            except BaseException as cleanup:
                cleanup_diagnostics.extend(exception_diagnostic(cleanup))
        for name in ("result.json", "result.pending.json"):
            (out / name).unlink(missing_ok=True)
        cause = exc
        while cause is not None and not isinstance(cause, ContextDoesNotFit):
            cause = cause.__cause__
        status = (
            "not_applicable"
            if system == "full_context" and isinstance(cause, ContextDoesNotFit)
            else "failed"
        )
        failure = {
            "status": status,
            "calls": client.budget.calls,
            "completed_sessions": len(rows),
            "expected_sessions": len(raw["questions"]),
            "native_closed": closed,
            "error": safe_kind(exc) + ': native Travel task failed',
            "diagnostics": {
                "exception_chain": exception_diagnostic(exc),
                "cleanup": cleanup_diagnostics,
                "native_tools": executor.diagnostics()
                if callable(getattr(executor, "diagnostics", None))
                else getattr(exc, "native_tools_diagnostics", None),
            },
            "sessions": [
                {
                    "id": i,
                    "status": "complete"
                    if i < len(rows)
                    else ("failed" if i == attempted else "unrun"),
                }
                for i in range(len(raw["questions"]))
            ],
        }
        (out / "failure.json").write_bytes(common._json(failure))
        if status == "not_applicable":
            return failure
        raise RuntimeError("native Travel task failed; retained spend in failure.json") from exc


def verify_task(out, *, manifest_name="result.json"):
    """Dataset/network-free exact native submission, reward and scorer replay."""
    from .rack import RackProfile

    out = Path(out)
    if manifest_name not in {"result.json", "result.pending.json"}:
        raise ValueError("invalid completion manifest")
    result = json.loads((out / manifest_name).read_bytes())
    if (
        result.get("status") != "complete"
        or result.get("protocol") != PROTOCOL
        or not result.get("native_closed")
        or (out / "failure.json").exists()
    ):
        raise ValueError("incomplete native Travel bundle")
    hashes = result["artifact_sha256"]
    required = {
        "input-task.json",
        "effective-profile.json",
        "steps.jsonl",
        "http.jsonl",
        "submission.jsonl",
    } | {"native-source/" + n for n in SOURCE_PINS}
    if not required <= hashes.keys():
        raise ValueError("required native evidence inventory missing")
    for name, expected in hashes.items():
        path = (out / name).resolve()
        if not path.is_relative_to(out.resolve()) or file_record(path)["sha256"] != expected:
            raise ValueError("artifact hash mismatch: " + name)
    source = {name: (out / "native-source" / name).read_bytes() for name in SOURCE_PINS}
    raw = validate_raw(json.loads((out / "input-task.json").read_bytes()))
    if digest(raw) != result["input_sha256"]:
        raise ValueError("input hash mismatch")
    rows = [json.loads(line) for line in (out / "steps.jsonl").read_bytes().splitlines()]
    if (
        len(rows) != len(raw["questions"])
        or len(rows) != result["completed_sessions"]
        or len(rows) != result["expected_sessions"]
        or [r["query_id"] for r in rows] != list(range(len(rows)))
    ):
        raise ValueError("native session occurrence coverage mismatch")
    wire = [json.loads(line) for line in (out / "http.jsonl").read_bytes().splitlines()]
    generations = [r for r in wire if r.get("url", "").endswith("/chat/completions")]
    production = (
        [json.loads(line) for line in (out / "production-raw.jsonl").read_bytes().splitlines()]
        if (out / "production-raw.jsonl").exists()
        else []
    )
    if (
        len(generations) + len(production) != result["calls"]
        or any(r.get("status") != 200 or "error" in r for r in generations)
        or any("error" in r or not r.get("response") for r in production)
    ):
        raise ValueError("native + production whole-task accounting mismatch")
    profile = RackProfile.model_validate(json.loads((out / "effective-profile.json").read_bytes()))
    cursor, pending_tools = [0], []

    def replay_create(**kwargs):
        if cursor[0] >= len(generations):
            raise ValueError("missing native actor wire replay")
        receipt = generations[cursor[0]]
        request, response = receipt["request"], receipt["response"]
        cursor[0] += 1
        if (
            any(request.get(k) != v for k, v in kwargs.items())
            or request.get("tools") != tool_schema(source)
            or request.get("chat_template_kwargs") != {"enable_thinking": False}
            or request.get("stream") is not False
            or request.get("n") != 1
            or len(response.get("choices", [])) != 1
            or response["choices"][0]["finish_reason"] not in {"stop", "tool_calls"}
        ):
            raise ValueError("native actor prompt/schema/response wire replay differs")
        return sdk_response(response)

    def replay_execute(name, args):
        if not pending_tools:
            raise ValueError("unexpected native tool wire replay")
        call, answer = pending_tools.pop(0)
        if (
            call["name"] != name
            or call["args"] != args
            or answer["name"] != name
            or answer["tool_call_id"] != call["id"]
        ):
            raise ValueError("native tool observation replay differs")
        return answer["result"]

    identity = SimpleNamespace(
        profile=profile,
        sdk=lambda role: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=replay_create))
        ),
    )
    with native_travel(
        source,
        identity,
        [raw],
        executor=SimpleNamespace(execute=replay_execute),
        max_steps=result["max_steps"],
    ) as runtime:
        accumulated = ""
        base = runtime.rows[0]["base_person"]
        runtime.actor.set_base_person(
            base["name"],
            base["query"],
            runtime.format_person_plan(base["name"], base["daily_plans"]),
        )
        for index, step in enumerate(rows):
            question = runtime.rows[0]["questions"][index]
            state = step["native_result"]
            pending_tools[:] = [
                (call, answer)
                for scratch in state["scratchpad"]
                for call, answer in zip(
                    scratch.get("tool_calls") or [], scratch.get("tool_results") or [], strict=True
                )
            ]
            runtime.actor.prepare_for_person(
                question["name"],
                question["round_idx"],
                include_previous_plans=False,
                memory_context=step["memory_context"] or None,
                memory_system=None,
            )
            replayed = runtime.actor.act(question["query"])
            if (
                replayed != step["action"]
                or asdict(runtime.actor.last_result) != state
                or pending_tools
            ):
                raise ValueError("native actor output/scratchpad wire replay differs")
            gt = {
                "name": question["name"],
                "daily_plans": raw["answers"][index],
                "judgement_mode": "none",
            }
            observation, reward, info = runtime.env.step(
                step["action"], ground_truth=gt, need_judge=True
            )
            if (
                step["observation"] != observation
                or step["reward"] is not reward
                or step["info"] != info
                or state["final_plan"] != step["action"]
                or state["success"] is not True
                or step["experience"]["content"]
                != runtime.actor.build_memory_entry(
                    question["query"], step["action"], observation, reward
                )
                or step["memory_readback"].get("readback_verified") is not True
            ):
                raise ValueError("native reward/entry/readback replay differs")
            accumulated += ("\n\n" if accumulated else "") + step["action"]
        if cursor[0] != len(generations):
            raise ValueError("unused native generation wire replay")
        expected = runtime.parse_all_plans(accumulated, runtime.rows[0]["questions"])
        submitted = [
            json.loads(line) for line in (out / "submission.jsonl").read_bytes().splitlines()
        ]
        validate_submission(submitted, runtime.rows)
        expected = [
            {**p, "plan": runtime.combination.parse_plan_text(p["result"])} for p in expected
        ]
        if [
            {k: p[k] for k in ("person_idx", "name", "query", "plan")} for p in expected
        ] != submitted[0]["persons"]:
            raise ValueError("native final accumulated submission differs")
        metrics = runtime.scorer.evaluate(str(out / "submission.jsonl"))
    if metrics != result["metrics"]:
        raise ValueError("official Travel scorer replay differs")
    return metrics


def load_rows(data, expected_sha256):
    record = file_record(data)
    if record["sha256"] != expected_sha256:
        raise ValueError("Travel dataset byte SHA256 mismatch")
    rows = [
        validate_raw(json.loads(line))
        for line in Path(data).read_bytes().splitlines()
        if line.strip()
    ]
    if (
        len(rows) != 270
        or len({r["id"] for r in rows}) != 270
        or sum(len(r["questions"]) for r in rows) != 1869
    ):
        raise ValueError("full 270 tasks / 1869 ordered session occurrences required")
    return rows


def build_spec(
    *,
    data,
    data_sha256,
    native_root,
    profile,
    output_root,
    embedding_url,
    readiness,
    tool_python,
    tool_helper,
    max_steps,
    campaign_seconds,
    production_memory_qualification=None,
    native_document_factories=None,
):
    rows = load_rows(data, data_sha256)
    sources(native_root)
    if (
        profile.profile_version != "role-budgets-v6"
        or not 1 <= max_steps <= 30
        or campaign_seconds <= 0
    ):
        raise ValueError("explicit v6 whole-task profile and native step/deadline bounds required")
    bindings = native_document_factories or {}
    if not set(bindings) <= {"mem0", "cognee"}:
        raise ValueError("only concrete native Mem0/Cognee document factories supported")
    return {
        "protocol": PROTOCOL,
        "quality_calls": 0,
        "data": file_record(data),
        "tasks": [
            {"id": r["id"], "sessions": len(r["questions"]), "input_sha256": digest(r)}
            for r in rows
        ],
        "native_root": str(Path(native_root).resolve()),
        "native_commit": PIN,
        "native_source_sha256": SOURCE_PINS,
        "profile": profile.model_dump(mode="json"),
        "output_root": str(Path(output_root).resolve()),
        "embedding_url": embedding_url,
        "systems": list(SYSTEMS) + sorted(bindings),
        "native_document_factories": bindings,
        "optional_coverage": {
            s: "qualified-factory-pending-verification"
            if s in bindings
            else "unqualified:no concrete qualified native document factory"
            for s in ("mem0", "cognee")
        },
        "bounds": {
            "calls_per_task_method": profile.max_generation_calls,
            "seconds_per_task_method": profile.max_run_seconds,
            "max_steps": max_steps,
            "campaign_seconds": campaign_seconds,
        },
        "actor_settings": {
            "temperature": 0,
            "step_memory": False,
            "include_previous_plans": False,
            "judgement_mode": "none",
            "common_native_base_person": True,
            "native_default_output_cap": 32768,
            "matched_explicit_output_cap": profile.output_budget("answer"),
        },
        "deviations": [
            "Native base-person prompt used identically for all methods; "
            "no implicit accumulated history even for no-memory.",
            "Native client default output cap 32768 replaced by declared matched actor cap; "
            "native prompts/entry/parser/reward/scorer unchanged.",
            "Local native tools in isolated read-only worker, no purchases, "
            "no generated-code execution.",
        ],
        "roles": {r: profile.model for r in ("actor", "extraction", "consolidation")},
        "production_memory_qualification": file_record(production_memory_qualification)
        if production_memory_qualification
        else None,
        "readiness": str(Path(readiness).resolve()),
        "resources": resource_inventory(readiness),
        "tool_python": file_record(tool_python),
        "tool_helper": file_record(tool_helper),
        "files": formal._code_inventory(),
    }


def verify_spec(spec_path, approval_path, *, allow_started=False):
    from .rack import RackProfile

    raw = Path(spec_path).read_bytes()
    spec, approval = json.loads(raw), json.loads(Path(approval_path).read_bytes())
    if (
        approval.get("approved") is not True
        or approval.get("reviewer_kind") not in {"parent", "human"}
        or approval.get("spec_sha256") != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError("independent byte-bound parent approval required")
    profile = RackProfile.model_validate(spec["profile"])
    rebuilt = build_spec(
        data=spec["data"]["path"],
        data_sha256=spec["data"]["sha256"],
        native_root=spec["native_root"],
        profile=profile,
        output_root=spec["output_root"],
        embedding_url=spec["embedding_url"],
        readiness=spec["readiness"],
        tool_python=spec["tool_python"]["path"],
        tool_helper=spec["tool_helper"]["path"],
        max_steps=spec["bounds"]["max_steps"],
        campaign_seconds=spec["bounds"]["campaign_seconds"],
        production_memory_qualification=spec["production_memory_qualification"]["path"]
        if spec.get("production_memory_qualification")
        else None,
        native_document_factories=spec.get("native_document_factories"),
    )
    if spec != rebuilt:
        raise ValueError("frozen full Travel source/data/resource/profile inventory differs")
    for system, binding in spec.get("native_document_factories", {}).items():
        if binding.get("kind") != "native-document-factory@v1" or binding.get("backend") != system:
            raise ValueError("qualified native document factory required")
        for stem in ("source", "qualification"):
            if file_record(binding[stem + "_path"])["sha256"] != binding[stem + "_sha256"]:
                raise ValueError("native document factory byte hash mismatch")
    out = Path(spec["output_root"])
    project = Path(__file__).resolve().parents[3]
    if out.resolve().is_relative_to(project):
        raise ValueError("external Travel output root required")
    if out.exists() and not allow_started:
        raise FileExistsError("fresh campaign root required")
    return spec


def retained_spend(target, calls):
    """Count each native/direct and production dispatch once; unknown usage stays unknown."""
    target = Path(target)
    roles = {}
    known = prompt = completion = 0
    for name in ("http.jsonl", "production-raw.jsonl"):
        path = target / name
        if not path.exists():
            continue
        for row in (json.loads(line) for line in path.read_bytes().splitlines()):
            if name == "http.jsonl" and not row.get("url", "").endswith("/chat/completions"):
                continue
            if name == "production-raw.jsonl" and "ordinal" not in row:
                continue
            role = "native_actor" if name == "http.jsonl" else row["task"]
            stats = roles.setdefault(
                role,
                {"calls": 0, "known_usage_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
            )
            stats["calls"] += 1
            usage = (row.get("response") or {}).get("usage", {})
            if all(
                type(usage.get(k)) is int and usage[k] >= 0
                for k in ("prompt_tokens", "completion_tokens")
            ):
                known += 1
                prompt += usage["prompt_tokens"]
                completion += usage["completion_tokens"]
                stats["known_usage_calls"] += 1
                stats["prompt_tokens"] += usage["prompt_tokens"]
                stats["completion_tokens"] += usage["completion_tokens"]
    if known > calls or sum(r["calls"] for r in roles.values()) > calls:
        raise ValueError("actual dispatch spend exceeds shared call accounting")
    return {
        "calls": calls,
        "known_usage_calls": known,
        "unknown_usage_calls": calls - known,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "roles": roles,
        "currency_cost": None,
    }


def _check_finalization_deadline(budget):
    """Keep saved-evidence finalization inside the original campaign clock.

    Admission may already be closed (including an unfit task envelope); that
    latch does not forbid an incomplete report while actual time remains.
    """
    from .parallel_campaign import CampaignStopped

    if budget is not None and budget.deadline - budget.clock() <= 0:
        raise CampaignStopped("campaign finalization deadline exhausted")


def execute_matrix(
    rows,
    *,
    systems,
    out,
    call_budget,
    seconds_budget,
    campaign_seconds,
    run_one,
    verify_one=verify_task,
    finalize=None,
    initialize=None,
):
    """Root-only authority: isolated failures, exact unrun coverage and actual spend."""
    from .parallel_campaign import CampaignBudget, CampaignStopped, Job, JobContext, RequestLimiter

    rows, systems, out = list(rows), list(systems), Path(out)
    if (
        not rows
        or len({r["id"] for r in rows}) != len(rows)
        or not systems
        or len(set(systems)) != len(systems)
    ):
        raise ValueError("unique complete task/system matrix required")
    outer = CampaignBudget(
        max_calls=len(rows) * len(systems) * call_budget, max_seconds=campaign_seconds
    )
    limiter = RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072)
    out.mkdir(parents=True, exist_ok=False)
    matrix = [
        {
            "task_id": r["id"],
            "system": s,
            "input_sha256": digest(r),
            "expected_sessions": len(r["questions"]),
            "completed_sessions": 0,
            "status": "unrun",
            "calls": 0,
        }
        for r in rows
        for s in systems
    ]
    (out / "frozen-matrix.json").write_bytes(common._json(matrix))
    (out / "task-statuses.json").write_bytes(common._json(matrix))
    if initialize is not None:
        initialize(out)
    for row in matrix:
        raw = next(r for r in rows if r["id"] == row["task_id"])
        target = out / str(row["task_id"]) / row["system"]
        context = None
        try:
            outer.check()
            if outer.remaining() < seconds_budget:
                outer.failed = True
                raise CampaignStopped("full whole-task deadline envelope cannot fit")
            job = Job(
                f"travel:{raw['id']}:{row['system']}",
                f"travel:{raw['id']}",
                target,
                call_budget,
                seconds_budget,
            )
            if not outer.reserve(job):
                raise CampaignStopped("whole-task envelope cannot be reserved")
            context = JobContext(job, outer, limiter)
            row.update(status="running", output=str(target))
            (out / "task-statuses.json").write_bytes(common._json(matrix))
            result = run_one(raw, row["system"], context, target)
            if result.get("status") == "not_applicable" and row["system"] == "full_context":
                row.update(status="not_applicable", reason="fit-only native context overflow")
                continue
            if result.get("status") != "complete" or result.get("completed_sessions") != len(
                raw["questions"]
            ):
                raise ValueError("incomplete native task cannot become root completion")
            verify_one(target)
            context.check()
            row.update(
                status="complete",
                completed_sessions=len(raw["questions"]),
                result_sha256=file_record(target / "result.json")["sha256"],
            )
        except CampaignStopped as exc:
            row.update(status="failed" if context is not None else "unrun", error=safe_error(exc))
        except (Exception, common.NativeAbort) as exc:
            row.update(status="failed", error=safe_error(exc))
            tool_failure = getattr(exc, "native_tools_diagnostics", None)
            if tool_failure is not None and not (target / "failure.json").exists():
                # Constructor failure precedes run_task, but is still a failed
                # infrastructure leaf with real spend, never a QA zero.
                target.mkdir(parents=True, exist_ok=True)
                (target / "failure.json").write_bytes(
                    common._json(
                        {
                            "status": "failed",
                            "calls": context.calls,
                            "completed_sessions": 0,
                            "expected_sessions": len(raw["questions"]),
                            "native_closed": tool_failure["native_closed"],
                            "error": safe_kind(exc) + ': native Travel worker startup failed',
                            "diagnostics": {
                                "exception_chain": exception_diagnostic(exc),
                                "cleanup": [],
                                "native_tools": tool_failure,
                            },
                            "sessions": [
                                {"id": i, "status": "unrun"} for i in range(len(raw["questions"]))
                            ],
                        }
                    )
                )
            if (target / "failure.json").exists():
                failure = json.loads((target / "failure.json").read_bytes())
                row["completed_sessions"] = failure["completed_sessions"]
        finally:
            if context is not None:
                row["calls"] = context.calls
                outer.release(context)
            try:
                row["spend"] = retained_spend(target, row["calls"])
            except Exception as exc:
                outer.failed = True
                row.update(status="failed", accounting_error=safe_error(exc))
            (out / "task-statuses.json").write_bytes(common._json(matrix))
    result = {
        "protocol": PROTOCOL,
        "completion_authority": "root-only",
        "quality_complete": all(r["status"] == "complete" for r in matrix),
        "tasks": matrix,
        "calls": outer.calls,
        "expected_sessions": sum(r["expected_sessions"] for r in matrix),
        "completed_sessions": sum(r["completed_sessions"] for r in matrix),
    }
    result["spend"] = {
        k: sum(r.get("spend", {}).get(k, 0) for r in matrix)
        for k in (
            "calls",
            "known_usage_calls",
            "unknown_usage_calls",
            "prompt_tokens",
            "completion_tokens",
        )
    }
    result["spend"]["currency_cost"] = None
    result["accounting_complete"] = not any("accounting_error" in r for r in matrix)
    if result["calls"] != sum(r["calls"] for r in matrix):
        raise RuntimeError("whole-task spend accounting unavailable")
    try:
        _check_finalization_deadline(outer)
        if finalize is not None:
            result = finalize(result, out)
        _check_finalization_deadline(outer)
        result["metrics"] = campaign_metrics(out, result, budget=outer)
        result["paired"] = paired_results(result, budget=outer)
        _check_finalization_deadline(outer)
        result["finalized"] = True
        result["elapsed_seconds"] = outer.snapshot()["elapsed_seconds"]
        pending = out / "campaign-result.pending.json"
        payload = common._json(result)
        _check_finalization_deadline(outer)
        pending.write_bytes(payload)
        _check_finalization_deadline(outer)
        if pending.read_bytes() != payload:
            raise RuntimeError("root candidate readback differs")
        _check_finalization_deadline(outer)
        verify_root(out, manifest_name=pending.name, budget=outer)
        _check_finalization_deadline(outer)
        if pending.read_bytes() != payload:
            raise RuntimeError("verified root candidate readback differs")
        _check_finalization_deadline(outer)
        pending.replace(out / "campaign-result.json")
        _check_finalization_deadline(outer)
        if (out / "campaign-result.json").read_bytes() != payload:
            raise RuntimeError("root completion readback differs")
        _check_finalization_deadline(outer)
    except BaseException as exc:
        (out / "campaign-result.json").unlink(missing_ok=True)
        (out / "campaign-result.pending.json").unlink(missing_ok=True)
        (out / "campaign-failure.json").write_bytes(
            common._json(
                {
                    **result,
                    "status": "incomplete",
                    "quality_complete": False,
                    "finalized": False,
                    "elapsed_seconds": outer.snapshot()["elapsed_seconds"],
                    "calls": outer.calls,
                    "error": safe_error(exc),
                    "completion_authority": "root-only",
                    "task_statuses": "task-statuses.json",
                }
            )
        )
        raise
    return result


def create_memory(system, *, spec, raw, client, budget, out, factories):
    """Concrete raw controls and unchanged production document ingestion, no stand-ins."""
    from gluomem.llm.modules import DspyExtractors

    from .rack import controlled_config

    out = Path(out)
    encoder = formal.BudgetedArenaEmbedder(spec["embedding_url"], budget=budget)
    if system in factories:
        return factories[system](
            profile=client.profile, client=client, budget=budget, embedder=encoder, out=out
        )
    if system != "gluomem":
        return formal.RawExperienceMemory(system, budget=budget, embedder=encoder, top_k=10)
    cfg = controlled_config(
        client.profile,
        data=Path(spec["data"]["path"]),
        sha256=spec["data"]["sha256"],
        embedder=encoder,
    )
    (out / "effective-config.json").write_bytes(common._json(cfg.model_dump(mode="json")))
    (out / "encoder.json").write_bytes(common._json(encoder.fingerprint.as_dict()))
    router = formal.ArenaProductionRouter(
        cfg, client=client, trace=out / "production-raw.jsonl", budget=budget
    )
    return formal.GluoExperienceMemory(
        cfg=cfg,
        extractors=DspyExtractors(cfg, router=router),
        embedder=encoder,
        budget=budget,
        db_path=out / "production.mind",
        evidence_path=out / "memory-events.jsonl",
        top_k=10,
    )


def execute_approved(spec_path, approval_path):
    """The real full run entry point, not a CLI interface or qualification substitute."""
    import importlib.util
    import zipfile

    from .rack import RackProfile

    spec = verify_spec(spec_path, approval_path)
    profile = RackProfile.model_validate(spec["profile"])
    formal.verify_memory_qualification(spec.get("production_memory_qualification"), profile)
    factories = {}
    for system, binding in spec.get("native_document_factories", {}).items():
        module_spec = importlib.util.spec_from_file_location(
            "_travel_document_" + system, binding["source_path"]
        )
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        module.verify_qualification(Path(binding["qualification_path"]), profile=profile)
        if not callable(getattr(module, "create_memory", None)):
            raise ValueError("concrete qualified native document factory missing")
        factories[system] = module.create_memory
    source = sources(spec["native_root"])
    rows = load_rows(spec["data"]["path"], spec["data"]["sha256"])
    root = Path(spec["output_root"])
    expected_spec = file_record(spec_path)["sha256"]

    def admission():
        approval = json.loads(Path(approval_path).read_bytes())
        if (
            file_record(spec_path)["sha256"] != expected_spec
            or approval.get("approved") is not True
            or approval.get("spec_sha256") != expected_spec
            or approval.get("reviewer_kind") not in {"parent", "human"}
            or formal._code_inventory() != spec["files"]
        ):
            raise ValueError("Travel runtime approval/source freeze changed; dispatch closed")

    def initialize(output):
        admission()
        (output / "execution-spec.json").write_bytes(Path(spec_path).read_bytes())
        (output / "execution-approval.json").write_bytes(Path(approval_path).read_bytes())
        with zipfile.ZipFile(output / "frozen-source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("inventory.json", common._json(spec["files"]))
            for index, record in enumerate(spec["files"]):
                archive.write(
                    record["path"], arcname=f"source/{index:06d}/" + Path(record["path"]).name
                )
            archive.write(spec["tool_helper"]["path"], arcname="travel_native_tool_worker.py")

    def run_one(raw, system, budget, target):
        admission()
        client = TravelTransport(
            profile,
            receipt=target / "http.jsonl",
            budget=budget,
            tools=tool_schema(source),
            admission=FrozenTravelTask(Path(spec_path), Path(approval_path), expected_spec),
            task=raw,
            max_steps=spec["bounds"]["max_steps"],
        )
        executor = None
        try:
            executor = NativeTools(
                python=spec["tool_python"]["path"],
                helper=spec["tool_helper"]["path"],
                readiness=spec["readiness"],
                budget=budget,
            )
            return run_task(
                raw,
                source=source,
                client=client,
                memory=lambda context, out: create_memory(
                    system,
                    spec=spec,
                    raw=raw,
                    client=client,
                    budget=context,
                    out=out,
                    factories=factories,
                ),
                executor=executor,
                out=target,
                max_steps=spec["bounds"]["max_steps"],
                system=system,
            )
        finally:
            if executor is not None:
                executor.close()
            client.close()

    def finalize(result, output):
        result.update(
            optional_coverage=spec["optional_coverage"],
            spec_sha256=expected_spec,
            coverage={
                "tasks": 270,
                "ordered_session_occurrences": 1869,
                "systems": spec["systems"],
            },
        )
        return result

    bounds = spec["bounds"]
    return execute_matrix(
        rows,
        systems=spec["systems"],
        out=root,
        call_budget=bounds["calls_per_task_method"],
        seconds_budget=bounds["seconds_per_task_method"],
        campaign_seconds=bounds["campaign_seconds"],
        run_one=run_one,
        finalize=finalize,
        initialize=initialize,
    )


def paired_results(result, *, budget=None):
    """Task-clustered wins/losses/ties only on root-verified complete intersections."""
    systems = list(dict.fromkeys(row["system"] for row in result["tasks"]))
    complete = {
        system: {
            row["task_id"]: row
            for row in result["tasks"]
            if row["system"] == system and row["status"] == "complete"
        }
        for system in systems
    }
    pairs = {}
    _check_finalization_deadline(budget)
    if "gluomem" not in complete:
        return pairs
    for system in systems:
        _check_finalization_deadline(budget)
        if system == "gluomem":
            continue
        ids = sorted(complete["gluomem"].keys() & complete[system].keys())
        scores = {metric: {"wins": 0, "losses": 0, "ties": 0} for metric in ("ps", "sps", "sr")}
        for tid in ids:
            _check_finalization_deadline(budget)
            left, right = complete["gluomem"][tid], complete[system][tid]
            if left["input_sha256"] != right["input_sha256"]:
                raise ValueError("paired native task input differs")
            lm = json.loads((Path(left["output"]) / "result.json").read_bytes())["metrics"]
            rm = json.loads((Path(right["output"]) / "result.json").read_bytes())["metrics"]
            for metric in scores:
                outcome = (
                    "wins"
                    if lm[metric] > rm[metric]
                    else ("losses" if lm[metric] < rm[metric] else "ties")
                )
                scores[metric][outcome] += 1
        pairs["gluomem_vs_" + system] = {
            "task_ids": ids,
            "complete_intersection": len(ids),
            **scores,
        }
    _check_finalization_deadline(budget)
    return pairs


def campaign_metrics(root, result, *, persist=True, budget=None):
    """Official completed-only metrics, never score failed/unrun as native zeros."""
    from .rack import RackProfile

    root = Path(root)
    metrics = {}
    for system in dict.fromkeys(r["system"] for r in result["tasks"]):
        _check_finalization_deadline(budget)
        planned = [r for r in result["tasks"] if r["system"] == system]
        complete = [r for r in planned if r["status"] == "complete"]
        record = {
            "planned_tasks": len(planned),
            "complete_tasks": len(complete),
            "full_metrics": None,
            "completed_only_metrics": None,
        }
        if complete:
            raw_rows, submission = [], []
            for row in complete:
                _check_finalization_deadline(budget)
                target = Path(row["output"])
                verify_task(target)
                _check_finalization_deadline(budget)
                raw = json.loads((target / "input-task.json").read_bytes())
                leaf = json.loads((target / "result.json").read_bytes())
                if raw["id"] != row["task_id"] or leaf["system"] != system:
                    raise ValueError("root-authoritative completed task identity differs")
                raw_rows.append(raw)
                submission.extend(
                    json.loads(line)
                    for line in (target / "submission.jsonl").read_bytes().splitlines()
                )
            target = Path(complete[0]["output"])
            source = {name: (target / "native-source" / name).read_bytes() for name in SOURCE_PINS}
            profile = RackProfile.model_validate(
                json.loads((target / "effective-profile.json").read_bytes())
            )
            with native_travel(
                source,
                SimpleNamespace(profile=profile, sdk=lambda role: None),
                raw_rows,
                executor=object(),
                max_steps=1,
            ) as runtime:
                _check_finalization_deadline(budget)
                validate_submission(submission, runtime.rows)
                directory = root / "scores" / system
                path = directory / "submission.jsonl"
                material = b"".join(
                    json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n" for r in submission
                )
                if persist:
                    _check_finalization_deadline(budget)
                    directory.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(material)
                elif path.read_bytes() != material:
                    raise ValueError("root aggregate submission bytes differ")
                _check_finalization_deadline(budget)
                record["completed_only_metrics"] = runtime.scorer.evaluate(
                    str(path), model_name=profile.model, memory_system=system
                )
                _check_finalization_deadline(budget)
                record["complete_sessions"] = sum(len(r["questions"]) for r in raw_rows)
                record["submission_sha256"] = file_record(path)["sha256"]
            if len(complete) == len(planned):
                record["full_metrics"] = record["completed_only_metrics"]
        metrics[system] = record
    _check_finalization_deadline(budget)
    return metrics


def verify_root(root, *, manifest_name="campaign-result.json", budget=None):
    """Root-authoritative, read-only saved-evidence verification; never opens dataset."""
    root = Path(root).resolve()
    if manifest_name not in {"campaign-result.json", "campaign-result.pending.json"}:
        raise ValueError("unsupported root manifest name")
    _check_finalization_deadline(budget)
    result = json.loads((root / manifest_name).read_bytes())
    statuses = json.loads((root / "task-statuses.json").read_bytes())
    frozen = json.loads((root / "frozen-matrix.json").read_bytes())
    if (
        result.get("protocol") != PROTOCOL
        or result.get("completion_authority") != "root-only"
        or result.get("finalized") is not True
        or result.get("tasks") != statuses
        or (root / "campaign-failure.json").exists()
        or len(statuses) != len(frozen)
    ):
        raise ValueError("root completion/coordination authority differs or is incomplete")
    for row, planned in zip(statuses, frozen, strict=True):
        _check_finalization_deadline(budget)
        if any(
            row[k] != planned[k] for k in ("task_id", "system", "input_sha256", "expected_sessions")
        ) or row["status"] not in {"complete", "failed", "unrun", "not_applicable"}:
            raise ValueError("root frozen task/method/occurrence inventory differs")
        target = root / str(row["task_id"]) / row["system"]
        if retained_spend(target, row["calls"]) != row.get("spend"):
            raise ValueError("root retained spend differs")
        if row["status"] == "complete":
            if (
                Path(row["output"]).resolve() != target
                or file_record(target / "result.json")["sha256"] != row["result_sha256"]
            ):
                raise ValueError("root complete leaf identity/hash differs")
            leaf = json.loads((target / "result.json").read_bytes())
            if (
                leaf["calls"] != row["calls"]
                or leaf["completed_sessions"] != row["expected_sessions"]
                or leaf["input_sha256"] != row["input_sha256"]
            ):
                raise ValueError("root complete occurrence/accounting differs")
            verify_task(target)
            _check_finalization_deadline(budget)
    if (
        result["calls"] != sum(r["calls"] for r in statuses)
        or result["expected_sessions"] != sum(r["expected_sessions"] for r in statuses)
        or result["completed_sessions"] != sum(r["completed_sessions"] for r in statuses)
        or result["quality_complete"] != all(r["status"] == "complete" for r in statuses)
    ):
        raise ValueError("root totals differ from exact planned coverage")
    if (root / "execution-spec.json").exists():
        raw = (root / "execution-spec.json").read_bytes()
        spec = json.loads(raw)
        approval = json.loads((root / "execution-approval.json").read_bytes())
        expected = [
            {
                "task_id": t["id"],
                "system": s,
                "input_sha256": t["input_sha256"],
                "expected_sessions": t["sessions"],
            }
            for t in spec["tasks"]
            for s in spec["systems"]
        ]
        actual = [
            {k: r[k] for k in ("task_id", "system", "input_sha256", "expected_sessions")}
            for r in statuses
        ]
        if (
            actual != expected
            or result.get("spec_sha256") != hashlib.sha256(raw).hexdigest()
            or approval.get("spec_sha256") != result["spec_sha256"]
            or approval.get("approved") is not True
            or approval.get("reviewer_kind") not in {"parent", "human"}
        ):
            raise ValueError("root frozen approval/full-data selection differs")
    if result.get("metrics") != campaign_metrics(root, result, persist=False, budget=budget):
        raise ValueError("root official native metrics differ")
    _check_finalization_deadline(budget)
    return {
        "protocol": PROTOCOL,
        "verified": True,
        "quality_complete": result["quality_complete"],
        "calls": result["calls"],
        "task_methods": len(statuses),
        "expected_sessions": result["expected_sessions"],
    }


def smoke(*, native_root, readiness, tool_python, tool_helper, out):
    """Fixed authored two-person case, mocked HTTP, actual six local native tools."""
    import httpx

    from .rack import RackProfile

    source = sources(native_root)
    resource_inventory(readiness)
    day = {
        "days": 1,
        **{
            k: "-"
            for k in (
                "current_city",
                "transportation",
                "breakfast",
                "attraction",
                "lunch",
                "dinner",
                "accommodation",
            )
        },
    }
    raw = {
        "id": -1,
        "base_person": {
            "name": "AuthoredBase",
            "query": "Authored integration base; NOT a benchmark question.",
            "daily_plans": [day],
        },
        "questions": [
            "I am AuthoredA.\nAuthored local tool integration only.",
            "I am AuthoredB.\nReuse authored experience only.",
        ],
        "answers": [[day], [day]],
    }
    profile = RackProfile(
        base_url="https://example.invalid/v1",
        model="authored-offline",
        max_tokens=128,
        max_generation_calls=4,
        max_run_seconds=120,
        max_input_chars=200000,
    )
    cases = [
        (
            "FlightSearch",
            {"origin": "New York", "destination": "Los Angeles", "date": "2022-03-01"},
        ),
        ("RestaurantSearch", {"city": "Rockford"}),
        ("AccommodationSearch", {"city": "Rockford"}),
        ("AttractionSearch", {"city": "Rockford"}),
        ("DistanceMatrix", {"origin": "New York", "destination": "Los Angeles", "mode": "driving"}),
        ("CitySearch", {"state": "California"}),
    ]

    def run_one(row, system, budget, target):
        def http(request):
            if request.url.path.endswith("/models"):
                return httpx.Response(
                    200,
                    json={"data": [{"id": profile.model, "max_model_len": profile.context_window}]},
                )
            if request.url.path == "/tokenize":
                return httpx.Response(
                    200, json={"count": 100, "max_model_len": profile.context_window}
                )
            if budget.calls == 1:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "local-" + str(i),
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                        for i, (name, args) in enumerate(cases)
                    ],
                }
                finish = "tool_calls"
            else:
                name = "AuthoredA" if budget.calls == 2 else "AuthoredB"
                message = {
                    "role": "assistant",
                    "content": f"=== {name}'s Plan ===\nDay 1:\nCurrent City: -\n",
                }
                finish = "stop"
            return httpx.Response(
                200,
                json={
                    "model": profile.model,
                    "choices": [{"finish_reason": finish, "message": message}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                },
            )

        executor = NativeTools(
            python=tool_python, helper=tool_helper, readiness=readiness, budget=budget
        )
        client = TravelTransport(
            profile,
            receipt=target / "http.jsonl",
            budget=budget,
            tools=tool_schema(source),
            transport=httpx.MockTransport(http),
        )
        try:
            return run_task(
                row,
                source=source,
                client=client,
                memory=formal.RawExperienceMemory(system, budget=budget),
                executor=executor,
                out=target,
                max_steps=3,
                system=system,
            )
        finally:
            executor.close()
            client.close()

    def finalize(result, root):
        result.update(
            evidence_kind="authored-offline-integration-NOT-benchmark",
            model_calls=0,
        )
        return result

    result = execute_matrix(
        [raw],
        systems=["full_context"],
        out=out,
        call_budget=4,
        seconds_budget=120,
        campaign_seconds=180,
        run_one=run_one,
        finalize=finalize,
    )
    if not result["quality_complete"]:
        raise RuntimeError("offline Travel smoke failed; inspect root-authoritative task statuses")
    return result


CONTINUATION_PROTOCOL = "memoryarena-travel-never-run-continuation@v1"
NEVER_RUN = "never_run_candidate_only"
REUSE = "verified_complete_reuse_conditional_on_continuation_authority"
CONTINUATION_DISPOSITIONS = {
    NEVER_RUN,
    REUSE,
    "quality_failed_latched_no_retry",
    "ambiguous_failure_infrastructure_hold_no_encoding_inference",
    "confirmed_worker_failure_boundary_infrastructure_hold_origin_not_established",
    "running_interrupted_infrastructure_hold",
}


def continuation_plan(ledger, original):
    """Pure full-scope admission projection. Reservation is NOT measured spend."""
    expected = [
        {
            "task_id": t["id"],
            "system": s,
            "input_sha256": t["input_sha256"],
            "expected_sessions": t["sessions"],
        }
        for t in original["tasks"]
        for s in original["systems"]
    ]
    rows = ledger["rows"]
    if (
        original.get("protocol") != PROTOCOL
        or len(original["tasks"]) != 270
        or len({t["id"] for t in original["tasks"]}) != 270
        or sum(t["sessions"] for t in original["tasks"]) != 1869
        or original["systems"] != list(SYSTEMS)
        or len(rows) != 1620
        or ledger["original_bounds"] != original["bounds"]
    ):
        raise ValueError("original full 270/1869/1620 scope and bounds required")
    dispatch, retained = [], []
    for i, (row, identity) in enumerate(zip(rows, expected, strict=True)):
        frozen = row["original_frozen_row"]
        if (
            row["matrix_index"] != i
            or any(frozen[k] != v for k, v in identity.items())
            or any(row["checkpoint_row"][k] != v for k, v in identity.items())
            or row["disposition"] not in CONTINUATION_DISPOSITIONS
        ):
            raise ValueError("ledger original ordered identity/disposition differs")
        disposition, checkpoint = row["disposition"], row["checkpoint_row"]
        calls = row["spend"]["calls"]
        if (
            type(calls) is not int
            or not 0 <= calls <= original["bounds"]["calls_per_task_method"]
            or type(checkpoint["calls"]) is not int
            or not 0 <= checkpoint["calls"] <= calls
        ):
            raise ValueError("prior attempts exceed original per-job guard; reserve unsafe")
        if disposition == NEVER_RUN and (
            checkpoint["status"] != "unrun"
            or calls != 0
            or row.get("failure") is not None
            or row.get("retained_leaf_path") is not None
            or row.get("checkpoint_call_delta") != 0
        ):
            raise ValueError("attempted/held identity cannot escalate into never-run dispatch")
        if disposition == REUSE and (
            checkpoint["status"] != "complete" or row.get("failure") is not None
        ):
            raise ValueError("reuse requires complete closed leaf")
        (dispatch if disposition == NEVER_RUN else retained).append(i)
    if len(dispatch) != 1511 or len(retained) != 109:
        raise ValueError("exact original 1511 never-run / 109 retained scope required")
    for key in (
        "calls",
        "known_usage_calls",
        "unknown_usage_calls",
        "prompt_tokens",
        "completion_tokens",
    ):
        if ledger["spend"][key] != sum(r["spend"].get(key, 0) for r in rows):
            raise ValueError("observed ledger spend totals differ")
    debit_calls = len(retained) * original["bounds"]["calls_per_task_method"]
    debit_seconds = len(retained) * original["bounds"]["seconds_per_task_method"] + 3600
    return {
        "scope": {"tasks": 270, "ordered_sessions_per_system": 1869, "rows": 1620},
        "dispatch_indices": dispatch,
        "retained_indices": retained,
        "allowance_debit": {"calls": debit_calls, "seconds": debit_seconds},
        "remaining_allowance": {
            "calls": len(rows) * original["bounds"]["calls_per_task_method"] - debit_calls,
            "seconds": original["bounds"]["campaign_seconds"] - debit_seconds,
        },
        "observed_prior": {
            k: ledger["spend"][k]
            for k in (
                "calls",
                "known_usage_calls",
                "unknown_usage_calls",
                "prompt_tokens",
                "completion_tokens",
            )
        },
        "unknown_prior_attempts": None,
        "measured_prior_elapsed_seconds": None,
    }


def _bound_json(record):
    raw = Path(record["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("continuation byte binding differs: " + Path(record["path"]).name)
    return json.loads(raw)


def _travel_normal_path(material, name):
    """AST evidence of normal-path parity, explicitly NOT a cleanup/failure certification."""
    import ast

    node = next(n for n in ast.parse(material).body if getattr(n, "name", None) == name)

    class NormalPath(ast.NodeTransformer):
        def visit_Try(self, value):
            value.handlers, value.finalbody = [], []
            return self.generic_visit(value)

        def visit_Assign(self, value):
            if any(
                isinstance(t, ast.Name) and t.id == "cleanup_diagnostics" for t in value.targets
            ):
                return None
            return self.generic_visit(value)

    return ast.dump(NormalPath().visit(node), include_attributes=False)


def _travel_semantic_bridge(origin):
    import ast
    import zipfile

    record = next(r for r in origin if Path(r["path"]).name == "frozen-source.zip")
    _verify_bound_file(record)
    with zipfile.ZipFile(record["path"]) as archive:
        inventory = json.loads(archive.read("inventory.json"))

        def old_source(name):
            index = next(i for i, r in enumerate(inventory) if Path(r["path"]).name == name)
            raw = archive.read(f"source/{index:06d}/" + name)
            if hashlib.sha256(raw).hexdigest() != inventory[index]["sha256"]:
                raise ValueError("old native producer archive member differs")
            return raw

        old, current = old_source("arena_travel_driver.py"), Path(__file__).read_bytes()
        exact = (
            "tool_schema",
            "sdk_response",
            "validate_raw",
            "validate_submission",
            "create_memory",
            "verify_task",
            "campaign_metrics",
            "paired_results",
        )
        normal = ("native_travel", "run_task")

        def definition(raw, name):
            node = next(n for n in ast.parse(raw).body if getattr(n, "name", None) == name)
            return ast.dump(node, include_attributes=False)

        if any(definition(old, n) != definition(current, n) for n in exact) or any(
            _travel_normal_path(old, n) != _travel_normal_path(current, n) for n in normal
        ):
            raise ValueError("native actor/input/scorer normal-path semantic bridge differs")
        old_memory, new_memory = old_source("arena_driver.py"), Path(formal.__file__).read_bytes()
        classes = [
            "RawExperienceMemory",
            "GluoExperienceMemory",
            "ArenaProductionRouter",
            "BudgetedArenaEmbedder",
        ]
        if any(definition(old_memory, n) != definition(new_memory, n) for n in classes):
            raise ValueError("native memory/production/embedding semantics changed")
        projections = {
            n: hashlib.sha256(_travel_normal_path(old, n).encode()).hexdigest() for n in normal
        }
    return {
        "normal_path_equal": True,
        "native_source_pins_equal": True,
        "exact_definitions": list(exact),
        "normal_path_ast_sha256": projections,
        "unchanged_memory_classes": classes,
        "old_archive_sha256": record["sha256"],
        "cleanup_and_failure_handlers_require_independent_review": True,
    }


def build_continuation_spec(*, runtime, ledger_path, ledger_sha256, reserve_proposal_path):
    """Build only; never writes approval or launches. Runtime is a fresh build_spec result.

    Parent reserve approval must independently review semantic_bridge, original
    per-job guard coverage and transport non-bypass; the builder cannot certify it.
    """
    ledger_record = file_record(ledger_path)
    if ledger_record["sha256"] != ledger_sha256:
        raise ValueError("independently supplied prior ledger byte SHA256 required")
    ledger = _bound_json(ledger_record)
    origin = [dict(r) for r in ledger["snapshot_bindings"]]
    if len({r["path"] for r in origin}) != len(origin):
        raise ValueError("duplicate original snapshot binding")
    original_record = next(r for r in origin if Path(r["path"]).name == "execution-spec.json")
    original = _bound_json(original_record)
    if original_record["sha256"] != ledger["original_spec_sha256"]:
        raise ValueError("original spec/ledger binding differs")
    plan = continuation_plan(ledger, original)
    frozen_record = next(r for r in origin if Path(r["path"]).name == "frozen-matrix.json")
    checkpoint_record = next(r for r in origin if Path(r["path"]).name == "task-statuses.json")
    if [r["original_frozen_row"] for r in ledger["rows"]] != _bound_json(frozen_record) or [
        r["checkpoint_row"] for r in ledger["rows"]
    ] != _bound_json(checkpoint_record):
        raise ValueError("ledger frozen matrix/checkpoint differs from bound original bytes")
    semantic_keys = (
        "data",
        "tasks",
        "systems",
        "native_commit",
        "native_source_sha256",
        "profile",
        "bounds",
        "actor_settings",
        "roles",
        "embedding_url",
        "native_document_factories",
        "production_memory_qualification",
        "optional_coverage",
        "deviations",
    )
    if any(runtime[k] != original[k] for k in semantic_keys):
        raise ValueError("unchanged ordered native input/scorer/actor/memory semantics required")
    output = Path(runtime["output_root"]).resolve()
    oldroot = Path(ledger["original_root"]).resolve()
    if (
        output == oldroot
        or output.is_relative_to(oldroot)
        or oldroot.is_relative_to(output)
        or output.is_relative_to(Path(__file__).resolve().parents[3])
    ):
        raise ValueError("fresh external disjoint continuation root required")
    proposal_record = file_record(reserve_proposal_path)
    proposal = _bound_json(proposal_record)
    p = proposal["proposal"]
    if (
        proposal["prior_ledger_sha256"] != ledger_sha256
        or proposal["original_spec_sha256"] != original_record["sha256"]
        or p["allowance_calls_debit_including_observed_prior_calls"]
        != plan["allowance_debit"]["calls"]
        or p["allowance_time_debit_seconds"] != plan["allowance_debit"]["seconds"]
        or p["never_run_rows_only"] != len(plan["dispatch_indices"])
        or p["retained_attempted_or_held_rows"] != len(plan["retained_indices"])
    ):
        raise ValueError("conservative allowance reserve proposal differs")
    return {
        "protocol": CONTINUATION_PROTOCOL,
        "runtime": deepcopy(runtime),
        "ledger": ledger_record,
        "origin": origin,
        "reserve_proposal": proposal_record,
        "plan": plan,
        "semantic_bridge": {
            "native_execution_projection": _travel_semantic_bridge(origin),
            "unchanged_fields": {k: original[k] for k in semantic_keys},
            "old_producer_inventory": original["files"],
            "new_producer_inventory": runtime["files"],
            "old_spec_sha256": original_record["sha256"],
            "requires_independent_semantic_review": True,
        },
    }


def verify_continuation_approval(spec_path, approval_path, reserve_approval_path):
    """Cheap byte-bound gate used again at every request; no approval self-generation."""
    spec_raw = Path(spec_path).read_bytes()
    spec = json.loads(spec_raw)
    sha = hashlib.sha256(spec_raw).hexdigest()
    approval = json.loads(Path(approval_path).read_bytes())
    reserve_raw = Path(reserve_approval_path).read_bytes()
    reserve = json.loads(reserve_raw)
    if (
        spec.get("protocol") != CONTINUATION_PROTOCOL
        or any(
            a.get("approved") is not True
            or a.get("reviewer_kind") not in {"parent", "human"}
            or a.get("spec_sha256") != sha
            for a in (approval, reserve)
        )
        or approval.get("reserve_approval_sha256") != hashlib.sha256(reserve_raw).hexdigest()
        or reserve.get("ledger_sha256") != spec["ledger"]["sha256"]
        or reserve.get("reserve_proposal_sha256") != spec["reserve_proposal"]["sha256"]
        or reserve.get("allowance_debit") != spec["plan"]["allowance_debit"]
        or reserve.get("semantic_bridge_sha256") != digest(spec["semantic_bridge"])
        or any(
            reserve.get(k) is not True
            for k in (
                "original_guards_verified",
                "no_transport_bypass",
                "unknown_attempts_bounded_by_original_guards",
                "semantic_bridge_verified",
            )
        )
    ):
        raise ValueError("separate exact-byte parent reserve/semantic/execution approval required")
    return spec


def verify_continuation_spec(
    spec_path, approval_path, reserve_approval_path, *, verify_runtime=True, budget=None
):
    """Full read-only origin/reuse verification; offline replay may omit runtime rebuild.

    Execution always requests verify_runtime=True. False is for post-exit saved
    evidence inspection only, never a permit or dispatch authorization.
    """
    _check_finalization_deadline(budget)
    spec = verify_continuation_approval(spec_path, approval_path, reserve_approval_path)
    rebuilt = build_continuation_spec(
        runtime=spec["runtime"],
        ledger_path=spec["ledger"]["path"],
        ledger_sha256=spec["ledger"]["sha256"],
        reserve_proposal_path=spec["reserve_proposal"]["path"],
    )
    if rebuilt != spec:
        raise ValueError("continuation exact plan/bridge rebuild differs")
    ledger = _bound_json(spec["ledger"])
    for record in spec["origin"]:
        _check_finalization_deadline(budget)
        _bound_json(record) if Path(record["path"]).suffix == ".json" else _verify_bound_file(
            record
        )
        _verify_bound_file({"path": record["snapshot"], "sha256": record["sha256"]})
    # Exact path inventory rejects post-ledger dispatch, even if every old hash
    # still matches. Databases are stat-only; never open them during reuse.
    expected_paths = {
        Path(r["path"]).resolve()
        for key in ("original_non_database_artifacts", "database_stat_only_not_opened")
        for r in ledger[key]
    }
    actual_paths = {p.resolve() for p in Path(ledger["original_root"]).rglob("*") if p.is_file()}
    if actual_paths != expected_paths:
        raise ValueError("original root exact artifact inventory changed after audit")
    for record in ledger["database_stat_only_not_opened"]:
        stat = Path(record["path"]).stat()
        if stat.st_size != record["bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise ValueError("original database stat-only inventory changed after audit")
    # Hash only immutable non-database evidence. Never open/copy/mutate old stores.
    for record in ledger["original_non_database_artifacts"]:
        _check_finalization_deadline(budget)
        _verify_bound_file(record)
    for row in ledger["rows"]:
        if row["disposition"] != REUSE:
            continue
        _check_finalization_deadline(budget)
        target = Path(row["retained_leaf_path"])
        checkpoint = row["checkpoint_row"]
        if (
            target.resolve()
            != (
                Path(ledger["original_root"]) / str(checkpoint["task_id"]) / checkpoint["system"]
            ).resolve()
            or file_record(target / "result.json")["sha256"] != checkpoint["result_sha256"]
            or verify_task(target) != row["verified_complete"]["metrics"]
        ):
            raise ValueError("original complete closed native leaf reuse differs")
        leaf = json.loads((target / "result.json").read_bytes())
        raw = json.loads((target / "input-task.json").read_bytes())
        if (
            digest(raw) != checkpoint["input_sha256"]
            or leaf["system"] != checkpoint["system"]
            or leaf["completed_sessions"] != checkpoint["expected_sessions"]
            or leaf["calls"] != row["spend"]["calls"]
        ):
            raise ValueError("old leaf ordered input/occurrence/spend differs")
    if verify_runtime:
        _check_finalization_deadline(budget)
        runtime = spec["runtime"]
        from .rack import RackProfile

        rebuilt_runtime = build_spec(
            data=runtime["data"]["path"],
            data_sha256=runtime["data"]["sha256"],
            native_root=runtime["native_root"],
            profile=RackProfile.model_validate(runtime["profile"]),
            output_root=runtime["output_root"],
            embedding_url=runtime["embedding_url"],
            readiness=runtime["readiness"],
            tool_python=runtime["tool_python"]["path"],
            tool_helper=runtime["tool_helper"]["path"],
            max_steps=runtime["bounds"]["max_steps"],
            campaign_seconds=runtime["bounds"]["campaign_seconds"],
            production_memory_qualification=(runtime["production_memory_qualification"] or {}).get(
                "path"
            ),
            native_document_factories=runtime.get("native_document_factories"),
        )
        if rebuilt_runtime != runtime:
            raise ValueError("continuation current full runtime inventory differs")
    _check_finalization_deadline(budget)
    return spec


def _verify_bound_file(record):
    if file_record(record["path"])["sha256"] != record["sha256"]:
        raise ValueError(
            "original/continuation file byte hash differs: " + Path(record["path"]).name
        )


def _continuation_rows(ledger):
    matrix = []
    for prior in ledger["rows"]:
        row = deepcopy(prior["checkpoint_row"])
        row.update(
            disposition=prior["disposition"],
            calls=prior["spend"]["calls"],
            spend=deepcopy(prior["spend"]),
            quality_metrics=None,
            provenance="original" if prior["disposition"] != NEVER_RUN else "continuation",
        )
        if prior["disposition"] == NEVER_RUN:
            row.update(
                status="unrun",
                completed_sessions=0,
                spend={
                    "calls": 0,
                    "known_usage_calls": 0,
                    "unknown_usage_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "roles": {},
                    "currency_cost": None,
                },
            )
        elif prior["disposition"] != REUSE:
            row["status"] = (
                "latched" if prior["disposition"] == "quality_failed_latched_no_retry" else "held"
            )
        matrix.append(row)
    return matrix


def _continuation_accounting(spec, matrix, budget):
    keys = (
        "calls",
        "known_usage_calls",
        "unknown_usage_calls",
        "prompt_tokens",
        "completion_tokens",
    )
    new = {
        k: sum(matrix[i].get("spend", {}).get(k, 0) for i in spec["plan"]["dispatch_indices"])
        for k in keys
    }
    # The live counter survives a trace/accounting/publication fault. Never zero it.
    new["calls"] = budget.calls
    new["unknown_usage_calls"] = budget.calls - new["known_usage_calls"]
    prior = spec["plan"]["observed_prior"]
    elapsed = budget.clock() - budget.started
    return {
        "observed_prior": prior,
        "new_actual": new,
        "observed_total_lower_bound": {k: prior[k] + new[k] for k in keys},
        "unknown_prior_attempts": None,
        "measured_prior_elapsed_seconds": None,
        "reserved_prior_allowance": spec["plan"]["allowance_debit"],
        "allowance_consumed_calls": spec["plan"]["allowance_debit"]["calls"] + budget.calls,
        "new_elapsed_seconds": elapsed,
        "allowance_consumed_seconds": spec["plan"]["allowance_debit"]["seconds"] + elapsed,
        "currency_cost": None,
    }


def _continuation_fit_evidence(target, row, runtime):
    """Replay the saved native tokenizer overflow, never trust a bare N/A label."""
    target = Path(target)
    failure = json.loads((target / "failure.json").read_bytes())
    profile = runtime["profile"]
    raw = json.loads((target / "input-task.json").read_bytes())
    receipts = [json.loads(line) for line in (target / "http.jsonl").read_bytes().splitlines()]
    tokenizers = [r for r in receipts if r.get("url", "").endswith("/tokenize")]
    response = (tokenizers[-1].get("response") or {}) if tokenizers else {}
    cap = profile["generation_budgets"]["reader"]
    if (
        row["system"] != "full_context"
        or failure.get("status") != "not_applicable"
        or failure.get("native_closed") is not True
        or failure.get("calls") != row["calls"]
        or failure.get("expected_sessions") != row["expected_sessions"]
        or digest(raw) != row["input_sha256"]
        or json.loads((target / "effective-profile.json").read_bytes()) != profile
        or not tokenizers
        or tokenizers[-1].get("status") != 200
        or response.get("max_model_len") != profile["context_window"]
        or type(response.get("count")) is not int
        or response["count"] + cap + profile["schema_reserve_tokens"] <= profile["context_window"]
    ):
        raise ValueError("native full-context fit-only overflow evidence differs")
    return {
        name: file_record(target / name)["sha256"]
        for name in ("failure.json", "input-task.json", "effective-profile.json", "http.jsonl")
    }


def _execute_continuation_matrix(spec, rows, *, budget, run_one, spec_paths, initialize=None):
    """Private executor seam; the public live entry fixes the native runner."""
    from .parallel_campaign import CampaignStopped, Job, JobContext, RequestLimiter

    bounds, plan = spec["runtime"]["bounds"], spec["plan"]
    if (
        budget.max_calls != plan["remaining_allowance"]["calls"]
        or budget.max_seconds != plan["remaining_allowance"]["seconds"]
        or budget.prior_calls
        or budget.prior_elapsed_seconds
    ):
        raise ValueError(
            "one reduced allowance guard required; reserved caps are not prior actual spend"
        )
    ledger = _bound_json(spec["ledger"])
    originals = _bound_json(
        next(r for r in spec["origin"] if Path(r["path"]).name == "execution-spec.json")
    )
    if continuation_plan(ledger, originals) != plan:
        raise ValueError("continuation dispatch subset differs")
    out = Path(spec["runtime"]["output_root"])
    out.mkdir(parents=True, exist_ok=False)
    matrix = _continuation_rows(ledger)
    result = {
        "protocol": CONTINUATION_PROTOCOL,
        "completion_authority": "root-only",
        "quality_complete": False,
        "finalized": False,
        "status": "incomplete",
        "tasks": matrix,
        "expected_sessions": sum(r["expected_sessions"] for r in matrix),
        "coverage": plan["scope"],
        "spec_sha256": file_record(spec_paths[0])["sha256"],
    }
    limiter = RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072)

    def checkpoint():
        (out / "task-statuses.json").write_bytes(common._json(matrix))

    def totals():
        result["accounting"] = _continuation_accounting(spec, matrix, budget)
        result["calls"] = result["accounting"]["observed_total_lower_bound"]["calls"]
        result["spend"] = result["accounting"]["observed_total_lower_bound"]
        result["completed_sessions"] = sum(r["completed_sessions"] for r in matrix)
        result["accounting_complete"] = not any("accounting_error" in r for r in matrix)

    try:
        for name, path in zip(
            ("execution-spec.json", "execution-approval.json", "reserve-approval.json"),
            spec_paths,
            strict=True,
        ):
            (out / name).write_bytes(Path(path).read_bytes())
        (out / "frozen-matrix.json").write_bytes(
            common._json([r["original_frozen_row"] for r in ledger["rows"]])
        )
        checkpoint()
        if initialize is not None:
            initialize(out)
        rows = rows() if callable(rows) else rows
        if [
            {"id": r["id"], "sessions": len(r["questions"]), "input_sha256": digest(r)}
            for r in rows
        ] != spec["runtime"]["tasks"]:
            raise ValueError("complete original ordered input matrix required")
        by_id = {r["id"]: r for r in rows}
        for index in plan["dispatch_indices"]:
            row = matrix[index]
            target = out / str(row["task_id"]) / row["system"]
            context = None
            try:
                budget.check()
                if budget.remaining() < bounds["seconds_per_task_method"]:
                    budget.failed = True
                    raise CampaignStopped("whole original task time envelope cannot fit")
                job = Job(
                    f"travel-continuation:{row['task_id']}:{row['system']}",
                    f"travel:{row['task_id']}",
                    target,
                    bounds["calls_per_task_method"],
                    bounds["seconds_per_task_method"],
                )
                if not budget.reserve(job):
                    budget.failed = True
                    raise CampaignStopped("whole original task call envelope cannot fit")

                def on_spend(ctx, row=row):
                    row["calls"] = ctx.calls
                    try:
                        checkpoint()
                    except BaseException:
                        budget.failed = True
                        raise

                context = JobContext(job, budget, limiter, on_spend=on_spend)
                row.update(status="running", output=str(target))
                checkpoint()
                leaf = run_one(by_id[row["task_id"]], row["system"], context, target)
                if leaf.get("status") == "not_applicable" and row["system"] == "full_context":
                    # Native fit overflow intentionally latches this job. It is
                    # still N/A only after closure + actual tokenizer replay.
                    budget.check()
                    if budget.clock() >= context.deadline:
                        raise CampaignStopped("native fit-only cleanup exceeded task deadline")
                    row["calls"] = context.calls
                    evidence = _continuation_fit_evidence(target, row, spec["runtime"])
                    row.update(
                        status="not_applicable",
                        reason="fit-only native context overflow",
                        fit_evidence_sha256=evidence,
                    )
                else:
                    if (
                        leaf.get("status") != "complete"
                        or leaf.get("completed_sessions") != row["expected_sessions"]
                    ):
                        raise ValueError("partial native task cannot become complete")
                    verify_task(target)
                    context.check()
                    row.update(
                        status="complete",
                        completed_sessions=row["expected_sessions"],
                        result_sha256=file_record(target / "result.json")["sha256"],
                    )
            except (Exception, common.NativeAbort) as exc:
                row.update(
                    status="failed" if context is not None else "unrun",
                    error=safe_error(exc),
                    quality_metrics=None,
                )
                if context is not None and (target / "failure.json").exists():
                    row["completed_sessions"] = json.loads((target / "failure.json").read_bytes())[
                        "completed_sessions"
                    ]
                diagnostic = getattr(exc, "native_tools_diagnostics", None)
                if diagnostic is not None and not (target / "failure.json").exists():
                    target.mkdir(parents=True, exist_ok=True)
                    (target / "failure.json").write_bytes(
                        common._json(
                            {
                                "status": "failed",
                                "calls": context.calls,
                                "completed_sessions": 0,
                                "expected_sessions": row["expected_sessions"],
                                "native_closed": diagnostic["native_closed"],
                                "quality_metrics": None,
                                "diagnostics": diagnostic,
                                "error": safe_kind(exc) + ': native worker startup failed',
                            }
                        )
                    )
            finally:
                if context is not None:
                    row["calls"] = context.calls
                    budget.release(context)
                try:
                    row["spend"] = retained_spend(target, row["calls"])
                except Exception as exc:
                    budget.failed = True
                    row.update(
                        status="failed", accounting_error=safe_error(exc), quality_metrics=None
                    )
                totals()
                checkpoint()
            if budget.failed:
                break
        totals()
        _check_finalization_deadline(budget)
        result["metrics"] = campaign_metrics(out, result, budget=budget)
        result["paired"] = paired_results(result, budget=budget)
        _check_finalization_deadline(budget)
        totals()
        result["finalized"] = True
        pending = out / "campaign-result.pending.json"
        payload = common._json(result)
        pending.write_bytes(payload)
        _check_finalization_deadline(budget)
        if pending.read_bytes() != payload:
            raise ValueError("continuation staged readback differs")
        verify_continuation_root(out, manifest_name=pending.name, budget=budget)
        _check_finalization_deadline(budget)
        if pending.read_bytes() != payload:
            raise ValueError("continuation verified readback differs")
        pending.replace(out / "campaign-result.json")
        _check_finalization_deadline(budget)
        if (out / "campaign-result.json").read_bytes() != payload:
            raise ValueError("continuation final readback differs")
        _check_finalization_deadline(budget)
    except BaseException as exc:
        totals()
        result.update(
            finalized=False,
            quality_complete=False,
            status="incomplete",
            error=safe_error(exc),
        )
        # Revoke authority before deletion: Windows can lock a promoted file.
        # The root verifier always rejects a surviving campaign-failure marker.
        try:
            (out / "campaign-failure.json").write_bytes(common._json(result))
        except BaseException:
            pass
        for name in ("campaign-result.pending.json", "campaign-result.json"):
            try:
                (out / name).unlink(missing_ok=True)
            except OSError as cleanup_exc:
                result.setdefault("publication_cleanup_errors", []).append(
                    safe_kind(cleanup_exc)
                )
        try:
            (out / "campaign-failure.json").write_bytes(common._json(result))
        except BaseException:
            # Disk failure cannot erase in-memory actual/reserved evidence.
            pass
        try:
            exc.continuation_receipt = result
        except (AttributeError, TypeError):
            pass
        raise
    return result


def verify_continuation_root(root, *, manifest_name="campaign-result.json", budget=None):
    """Full mixed-provenance verifier, separate from the unchanged Cartesian v1 verifier."""
    if manifest_name not in {"campaign-result.json", "campaign-result.pending.json"}:
        raise ValueError("unsupported continuation root manifest")
    root = Path(root).resolve()
    _check_finalization_deadline(budget)
    spec = verify_continuation_spec(
        root / "execution-spec.json",
        root / "execution-approval.json",
        root / "reserve-approval.json",
        verify_runtime=False,
        budget=budget,
    )
    if root != Path(spec["runtime"]["output_root"]).resolve():
        raise ValueError("continuation root identity differs")
    result = json.loads((root / manifest_name).read_bytes())
    statuses = json.loads((root / "task-statuses.json").read_bytes())
    ledger = _bound_json(spec["ledger"])
    initial = _continuation_rows(ledger)
    if (
        result.get("protocol") != CONTINUATION_PROTOCOL
        or result.get("completion_authority") != "root-only"
        or result.get("finalized") is not True
        or result.get("quality_complete") is not False
        or result.get("status") != "incomplete"
        or (root / "campaign-failure.json").exists()
        or statuses != result["tasks"]
        or len(statuses) != 1620
        or result.get("spec_sha256") != file_record(root / "execution-spec.json")["sha256"]
        or json.loads((root / "frozen-matrix.json").read_bytes())
        != [r["original_frozen_row"] for r in ledger["rows"]]
    ):
        raise ValueError("continuation root full coverage/authority differs")
    for i, (row, planned) in enumerate(zip(statuses, initial, strict=True)):
        _check_finalization_deadline(budget)
        if i in spec["plan"]["retained_indices"]:
            if row != planned:
                raise ValueError("retained reuse/latched/held row changed or retried")
            continue
        if (
            any(
                row[k] != planned[k]
                for k in (
                    "task_id",
                    "system",
                    "input_sha256",
                    "expected_sessions",
                    "disposition",
                    "provenance",
                )
            )
            or row["status"] not in {"complete", "failed", "unrun", "not_applicable"}
            or type(row["calls"]) is not int
            or not 0 <= row["calls"] <= spec["runtime"]["bounds"]["calls_per_task_method"]
            or type(row["completed_sessions"]) is not int
            or not 0 <= row["completed_sessions"] <= row["expected_sessions"]
            or (row["status"] != "complete" and row.get("quality_metrics") is not None)
        ):
            raise ValueError("continuation never-run task identity/status differs")
        target = root / str(row["task_id"]) / row["system"]
        if retained_spend(target, row["calls"]) != row["spend"]:
            raise ValueError("new actual retained spend differs")
        if row["status"] == "unrun" and (
            row["calls"] or row["completed_sessions"] or target.exists()
        ):
            raise ValueError("unrun task contains dispatch evidence")
        if row["status"] == "not_applicable":
            if row.get("fit_evidence_sha256") != _continuation_fit_evidence(
                target, row, spec["runtime"]
            ):
                raise ValueError("native fit-only evidence hashes differ")
        if row["status"] == "complete":
            if (
                Path(row["output"]).resolve() != target
                or file_record(target / "result.json")["sha256"] != row["result_sha256"]
            ):
                raise ValueError("new complete leaf output/hash differs")
            leaf = json.loads((target / "result.json").read_bytes())
            if (
                leaf["calls"] != row["calls"]
                or leaf["input_sha256"] != row["input_sha256"]
                or leaf["system"] != row["system"]
                or leaf["completed_sessions"] != row["expected_sessions"]
                or row["completed_sessions"] != row["expected_sessions"]
            ):
                raise ValueError("new complete leaf input/occurrence/accounting differs")
            verify_task(target)
    accounting = result["accounting"]
    new_calls = sum(statuses[i]["calls"] for i in spec["plan"]["dispatch_indices"])
    fake = SimpleNamespace(
        calls=new_calls, started=0, clock=lambda: accounting["new_elapsed_seconds"]
    )
    expected_accounting = _continuation_accounting(spec, statuses, fake)
    if (
        accounting != expected_accounting
        or result["spend"] != accounting["observed_total_lower_bound"]
        or result["calls"] != sum(r["calls"] for r in statuses)
        or result["expected_sessions"] != sum(r["expected_sessions"] for r in statuses)
        or result["completed_sessions"] != sum(r["completed_sessions"] for r in statuses)
        or result["coverage"] != spec["plan"]["scope"]
        or accounting["allowance_consumed_calls"]
        > 1620 * spec["runtime"]["bounds"]["calls_per_task_method"]
        or accounting["allowance_consumed_seconds"] > spec["runtime"]["bounds"]["campaign_seconds"]
        or result["accounting_complete"] is not True
    ):
        raise ValueError("continuation observed/reserved accounting or scope differs")
    if result["metrics"] != campaign_metrics(root, result, persist=False, budget=budget) or result[
        "paired"
    ] != paired_results(result, budget=budget):
        raise ValueError("mixed provenance official aggregate/paired metrics differ")
    _check_finalization_deadline(budget)
    return {
        "protocol": CONTINUATION_PROTOCOL,
        "verified": True,
        "quality_complete": False,
        "task_methods": 1620,
        "expected_sessions": result["expected_sessions"],
        "calls": result["calls"],
    }


@dataclass(frozen=True)
class FrozenTravelContinuationTask:
    """Explicit v1 never-run permit; neither subclass nor weakening of legacy full approval."""

    spec_path: Path
    approval_path: Path
    reserve_approval_path: Path
    expected_sha256: str
    approval_sha256: str
    reserve_approval_sha256: str
    system: str

    def check_approval(self):
        for path, sha in (
            (self.spec_path, self.expected_sha256),
            (self.approval_path, self.approval_sha256),
            (self.reserve_approval_path, self.reserve_approval_sha256),
        ):
            _verify_bound_file({"path": path, "sha256": sha})
        spec = verify_continuation_approval(
            self.spec_path, self.approval_path, self.reserve_approval_path
        )
        _verify_bound_file(spec["ledger"])
        _verify_bound_file(spec["reserve_proposal"])
        if formal._code_inventory() != spec["runtime"]["files"]:
            raise ValueError("continuation producer freeze changed; dispatch closed")
        return spec

    def validate(self, task, profile, max_steps):
        spec = self.check_approval()
        runtime = spec["runtime"]
        ledger = _bound_json(spec["ledger"])
        identity = {
            "task_id": task["id"],
            "system": self.system,
            "input_sha256": digest(task),
            "expected_sessions": len(task["questions"]),
        }
        permitted = [
            ledger["rows"][i]["original_frozen_row"] for i in spec["plan"]["dispatch_indices"]
        ]
        if (
            not any(all(row[k] == v for k, v in identity.items()) for row in permitted)
            or profile.model_dump(mode="json") != runtime["profile"]
            or max_steps != runtime["bounds"]["max_steps"]
        ):
            raise ValueError("exact original never-run task/system/profile permit required")
        return runtime


def execute_continuation(spec_path, approval_path, reserve_approval_path):
    """Execute only the approved original NEVER-RUN identities with the native driver.

    Operator: build_spec(...) for the *fresh frozen runtime*, pass that mapping to
    build_continuation_spec(...), save common._json bytes. Obtain separate parent
    reserve approval (schema in verify_continuation_approval), then execution approval
    binding its byte SHA. Call this API only after independent review/authorization.
    verify_continuation_root(new_root) is the post-exit mixed-provenance verifier.
    """
    import zipfile

    from .parallel_campaign import CampaignBudget
    from .rack import RackProfile

    # Only cheap authority parsing precedes the ONE guard. No preflight, tools or
    # native/source setup has happened; reserved prior caps are deducted once.
    spec = verify_continuation_approval(spec_path, approval_path, reserve_approval_path)
    remaining = spec["plan"]["remaining_allowance"]
    budget = CampaignBudget(max_calls=remaining["calls"], max_seconds=remaining["seconds"])
    runtime = spec["runtime"]
    paths = tuple(map(Path, (spec_path, approval_path, reserve_approval_path)))
    expected = tuple(file_record(p)["sha256"] for p in paths)
    state = {}

    def initialize(output):
        verified = verify_continuation_spec(*paths, budget=budget)
        if verified != spec:
            raise ValueError("continuation setup approval changed")
        profile = RackProfile.model_validate(runtime["profile"])
        formal.verify_memory_qualification(runtime.get("production_memory_qualification"), profile)
        # This protocol deliberately retains the original six systems only.
        if runtime.get("native_document_factories"):
            raise ValueError("optional native backends require a separate continuation protocol")
        state.update(profile=profile, source=sources(runtime["native_root"]))
        _check_finalization_deadline(budget)
        with zipfile.ZipFile(output / "frozen-source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("inventory.json", common._json(runtime["files"]))
            for i, record in enumerate(runtime["files"]):
                _check_finalization_deadline(budget)
                _verify_bound_file(record)
                archive.write(
                    record["path"], arcname=f"source/{i:06d}/" + Path(record["path"]).name
                )
            archive.write(runtime["tool_helper"]["path"], arcname="travel_native_tool_worker.py")
        _check_finalization_deadline(budget)

    def run_one(raw, system, context, target):
        permit = FrozenTravelContinuationTask(*paths, *expected, system=system)
        permit.validate(raw, state["profile"], runtime["bounds"]["max_steps"])
        context.check()
        client = TravelTransport(
            state["profile"],
            receipt=target / "http.jsonl",
            budget=context,
            tools=tool_schema(state["source"]),
            admission=permit,
            task=raw,
            max_steps=runtime["bounds"]["max_steps"],
        )
        executor = None
        try:
            executor = NativeTools(
                python=runtime["tool_python"]["path"],
                helper=runtime["tool_helper"]["path"],
                readiness=runtime["readiness"],
                budget=context,
            )
            return run_task(
                raw,
                source=state["source"],
                client=client,
                memory=lambda ctx, out: create_memory(
                    system, spec=runtime, raw=raw, client=client, budget=ctx, out=out, factories={}
                ),
                executor=executor,
                out=target,
                max_steps=runtime["bounds"]["max_steps"],
                system=system,
            )
        finally:
            try:
                if executor is not None:
                    executor.close()
            finally:
                client.close()

    # Ordered dataset reconstruction is guarded setup too. Matrix publication is
    # initialized before expensive verification, so setup faults retain all rows.
    def rows():
        return load_rows(runtime["data"]["path"], runtime["data"]["sha256"])

    return _execute_continuation_matrix(
        spec, rows, budget=budget, run_one=run_one, spec_paths=paths, initialize=initialize
    )


def main(argv=None):
    import argparse

    from .rack import RackProfile

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    offline = commands.add_parser(
        "smoke", help="authored HTTP fixture, actual six native tools, ZERO model calls"
    )
    for key in ("native-root", "readiness", "tool-python", "tool-helper", "out"):
        offline.add_argument("--" + key, type=Path, required=True)
    plan = commands.add_parser("plan", help="full-data freeze, zero quality calls")
    for key in (
        "data",
        "native-root",
        "profile",
        "output-root",
        "spec-out",
        "readiness",
        "tool-python",
        "tool-helper",
    ):
        plan.add_argument("--" + key, type=Path, required=True)
    plan.add_argument("--data-sha256", required=True)
    plan.add_argument("--embedding-url", required=True)
    plan.add_argument("--max-steps", type=int, required=True)
    plan.add_argument("--campaign-seconds", type=int, required=True)
    plan.add_argument("--production-memory-qualification", type=Path)
    plan.add_argument("--native-document-factories", type=Path)
    run = commands.add_parser("run", help="all tasks and methods; parent approval mandatory")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--approval", type=Path, required=True)
    verify = commands.add_parser(
        "verify-task", help="saved-source native scorer replay; no dataset/models"
    )
    verify.add_argument("--bundle", type=Path, required=True)
    root_verify = commands.add_parser(
        "verify-root", help="read-only root-authoritative coverage/spend/native scorer replay"
    )
    root_verify.add_argument("--bundle", type=Path, required=True)
    continuation = commands.add_parser(
        "plan-continuation",
        help="versioned full-scope NEVER-RUN executable freeze; no launch/approval",
    )
    for key in ("runtime-spec", "ledger", "reserve-proposal", "spec-out"):
        continuation.add_argument("--" + key, type=Path, required=True)
    continuation.add_argument("--ledger-sha256", required=True)
    continuation_run = commands.add_parser(
        "run-never-run",
        help="ONLY approved original never-run rows; separate parent reserve mandatory",
    )
    for key in ("spec", "approval", "reserve-approval"):
        continuation_run.add_argument("--" + key, type=Path, required=True)
    continuation_verify = commands.add_parser(
        "verify-continuation-root", help="read-only mixed provenance full-root native replay"
    )
    continuation_verify.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        print(
            json.dumps(
                smoke(
                    native_root=args.native_root,
                    readiness=args.readiness,
                    tool_python=args.tool_python,
                    tool_helper=args.tool_helper,
                    out=args.out,
                )
            )
        )
    elif args.command == "plan":
        kwargs = vars(args).copy()
        for key in ("command", "spec_out", "profile", "native_document_factories"):
            kwargs.pop(key)
        spec = build_spec(
            **kwargs,
            profile=RackProfile.model_validate(json.loads(args.profile.read_bytes())),
            native_document_factories=json.loads(args.native_document_factories.read_bytes())
            if args.native_document_factories
            else None,
        )
        with args.spec_out.open("xb") as stream:
            stream.write(common._json(spec))
        print(
            json.dumps(
                {
                    "spec_sha256": file_record(args.spec_out)["sha256"],
                    "tasks": len(spec["tasks"]),
                    "sessions": sum(t["sessions"] for t in spec["tasks"]),
                    "quality_calls": 0,
                }
            )
        )
    elif args.command == "plan-continuation":
        spec = build_continuation_spec(
            runtime=json.loads(args.runtime_spec.read_bytes()),
            ledger_path=args.ledger,
            ledger_sha256=args.ledger_sha256,
            reserve_proposal_path=args.reserve_proposal,
        )
        with args.spec_out.open("xb") as stream:
            stream.write(common._json(spec))
        print(
            json.dumps(
                {
                    "protocol": CONTINUATION_PROTOCOL,
                    "spec_sha256": file_record(args.spec_out)["sha256"],
                    "dispatch_rows": len(spec["plan"]["dispatch_indices"]),
                    "retained_rows": len(spec["plan"]["retained_indices"]),
                    "quality_calls": 0,
                }
            )
        )
    elif args.command == "run-never-run":
        result = execute_continuation(args.spec, args.approval, args.reserve_approval)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("protocol", "quality_complete", "calls", "accounting", "coverage")
                }
            )
        )
    elif args.command == "verify-continuation-root":
        print(json.dumps(verify_continuation_root(args.bundle)))
    elif args.command == "verify-task":
        print(json.dumps(verify_task(args.bundle)))
    elif args.command == "verify-root":
        print(json.dumps(verify_root(args.bundle)))
    else:
        verify_spec(args.spec, args.approval)
        print(json.dumps(execute_approved(args.spec, args.approval)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
