"""Full-scope native MemoryArena Shopping; no QA without independent approval."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

from gluomem_bench.bench.error_privacy import safe_error, safe_notes

from . import arena_campaign as native
from . import arena_driver as common
from .trajectory_suite import load_arena

PROTOCOL = "memoryarena-full-shopping-native@v1"
DOMAIN = "bundled_shopping"
SYSTEMS = common.SYSTEMS


DATA_SHA256 = "4411a2da528a33dc6aca519b49cc225895363f18b2d19b191fddb501200134ef"


def plan_dataset(data_root, pins):
    if pins.get(DOMAIN + "/data.jsonl") != DATA_SHA256:
        raise ValueError("released Shopping dataset SHA256 required")
    tasks = load_arena(Path(data_root), domain=DOMAIN, pins=pins)
    if len(tasks) != 150 or sum(len(t.sessions) for t in tasks) != 900:
        raise ValueError("full released 150 task / 900 ordered session scope required")
    return {
        "protocol": PROTOCOL,
        "quality_calls": 0,
        "tasks_count": len(tasks),
        "sessions_count": sum(len(t.sessions) for t in tasks),
        "systems": list(SYSTEMS),
        "optional_adapters": {"mem0": "unqualified", "cognee": "unqualified"},
        "tasks": [
            {"id": t.id, "sessions": len(t.sessions), "input_sha256": common.task_digest(t)}
            for t in tasks
        ],
        "occurrences": [
            {
                "task_id": t.id,
                "session_index": i,
                "input_sha256": hashlib.sha256(
                    native._json(
                        {"question": s.question, "background": s.background, "gold": t.gold[i]}
                    )
                ).hexdigest(),
            }
            for t in tasks
            for i, s in enumerate(t.sessions)
        ],
    }


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_runtime(external_root, out):
    """Copy original Git objects, never mutate upstream or install dependencies."""
    root, out = Path(external_root), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkout = root / "MemoryArena"
    names = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(checkout),
                "ls-tree",
                "-r",
                "--name-only",
                native.PIN,
                "--",
                "env/env_systems/web_shopping_env",
            ]
        )
        .decode()
        .splitlines()
    )
    names += [
        "env/env_systems/base_env.py",
        "env/env_systems/webshop_env.py",
        "agent/base.py",
        "agent/base_agent.py",
        "agent/webshop.py",
        "run_shopping.py",
    ]
    files = []
    for name in names:
        if "__pycache__" in name or ".ipynb_checkpoints" in name:
            continue
        material = subprocess.check_output(
            ["git", "-C", str(checkout), "show", native.PIN + ":" + name]
        )
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(material)
        files.append({"path": str(dest), "sha256": sha(dest), "source": name})
    worker = out / "shopping_native_worker.py"
    worker.write_bytes((root / "shopping_native_worker.py").read_bytes())
    readiness = json.loads((root / "arena-shopping-native-readiness.json").read_bytes())
    assets = json.loads((root / "arena-shopping-assets-provision.json").read_bytes())
    if readiness["status"] != "actual-native-webshop-reset-ready" or len(assets["files"]) != 89:
        raise ValueError("actual native readiness and all 89 product assets required")
    manifest = {
        "native_commit": native.PIN,
        "source_root": str(out / "source"),
        "asset_root": str((root / "native-assets/shopping").resolve()),
        "java_home": readiness["process_local_java_home"],
        "python": str((root / "shopping-environment/Scripts/python.exe").resolve()),
        "worker": str(worker),
        "worker_sha256": sha(worker),
        "files": files,
        "assets": [
            {"path": f["path"], "sha256": f["sha256"], "bytes": f["bytes"]} for f in assets["files"]
        ],
        "adaptation": (
            "in-process declared HTTP-route bridge; "
            "unchanged native agent/environment/Plus/reward; "
            "full products; one fixture reset goal; no network"
        ),
    }
    (out / "runtime.json").write_bytes(native._json(manifest))
    return out / "runtime.json"


class ShoppingRuntime:
    """Fixed isolated Python3.10 worker; bounded local JSON, not code execution."""

    OPERATIONS = {"open", "step", "close_session", "close"}

    def __init__(self, manifest_path, *, budget, out):
        self.manifest_path, self.budget, self.out = Path(manifest_path), budget, Path(out)
        self.budget.check()
        manifest = json.loads(self.manifest_path.read_bytes())
        for record in [
            *manifest["files"],
            {"path": manifest["worker"], "sha256": manifest["worker_sha256"]},
        ]:
            if sha(record["path"]) != record["sha256"]:
                raise ValueError("native runtime bytehash mismatch")
        self.budget.check()
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = (self.out / "runtime-stderr.log").open("wb")
        self.process = subprocess.Popen(
            [
                manifest["python"],
                "-X",
                "utf8",
                "-I",
                "-u",
                manifest["worker"],
                str(self.manifest_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            cwd=self.out,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.lines = queue.Queue()

        def pump():
            for line in self.process.stdout:
                self.lines.put(line)
            self.lines.put(None)

        self.thread = threading.Thread(target=pump, daemon=True)
        self.thread.start()
        self.sequence = 0

    def request(self, op, **fields):
        if op not in self.OPERATIONS:
            raise ValueError("undeclared native operation")
        cleanup = op in {"close", "close_session"}
        if not cleanup:
            self.budget.check()
        if op == "open":
            fields["task_path"] = str((self.out / f"session-{self.sequence}.json").resolve())
        self.sequence += 1
        message = {"op": op, **fields}
        self.process.stdin.write(json.dumps(message, ensure_ascii=False).encode("utf-8") + b"\n")
        self.process.stdin.flush()
        try:
            raw = self.lines.get(
                timeout=self._cleanup_timeout() if cleanup else max(0.01, self.budget.remaining())
            )
        except queue.Empty as exc:
            self.process.kill()
            raise TimeoutError("native shopping operation exceeded whole-task deadline") from exc
        if raw is None:
            raise RuntimeError("native worker exited; see runtime-stderr.log")
        response = json.loads(raw)
        native._append(
            self.out / "runtime.jsonl",
            {"sequence": self.sequence, "request": message, "response": response},
        )
        if not response["ok"]:
            raise RuntimeError(response["error"])
        if not cleanup:
            self.budget.check()
        return response["result"]

    def __enter__(self):
        return self

    def _cleanup_timeout(self):
        outer = getattr(self.budget, "budget", self.budget)
        return min(30, max(0.01, self.budget.deadline - outer.clock()))

    def __exit__(self, kind, value, traceback):
        errors = []
        try:
            proof = self.request("close")
            self.process.stdin.close()
            code = self.process.wait(timeout=self._cleanup_timeout())
            if code != 0 or not proof["native_closed"] or proof["raw_environment_active"]:
                raise ValueError("native cleanup verification failed")
            (self.out / "runtime-closed.json").write_bytes(
                native._json({**proof, "process_exit": code})
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            try:
                if self.process.poll() is None:
                    self.process.kill()
                    self.process.wait(timeout=self._cleanup_timeout())
            except BaseException as exc:
                errors.append(exc)
            for close in (
                self.log.close,
                self.process.stdout.close,
                lambda: self.thread.join(timeout=min(5, self._cleanup_timeout())),
            ):
                try:
                    close()
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            origin = value if value is not None else errors[0]
            for exc in errors:
                origin.add_note('Shopping runtime cleanup: ' + safe_error(exc))
            if value is None:
                raise origin


class ShoppingTransport(native.ArenaTransport):
    """Native WebShop uses text actions: its exact SDK tools schema is absent.

    Do not graft formal reasoning function schemas onto the native text actor.
    Common SDK/tokenizer/identity/usage guards remain active on every request.
    """

    @classmethod
    def offline(cls, profile, *, receipt, budget, transport):
        obj = cls.__new__(cls)
        native.ArenaTransport.__init__(
            obj,
            profile,
            receipt=receipt,
            budget=budget,
            transport=transport,
            tool_capability="offline-authored",
        )
        return obj

    def create(self, role, **kwargs):
        if role != "actor" or "tools" in kwargs or "tool_choice" in kwargs:
            error = ValueError("native shopping declares no SDK tools or non-actor roles")
            self._latch(error)
            raise native.NativeAbort(str(error))
        return super().create(role, **kwargs)

    def _request(self, method, path, payload=None):
        if path == "/chat/completions":
            payload = {**payload, "stream": False, "n": 1}
        return super()._request(method, path, payload)

    def _latch(self, exc):
        from .rack import SharedRackError

        super()._latch(exc)
        if isinstance(exc, SharedRackError) or getattr(exc, "shared_failure", False):
            self.budget.budget.failed = True


@contextmanager
def native_actor(runtime_path, client):
    manifest = json.loads(Path(runtime_path).read_bytes())
    prefix = "_shopping_" + uuid.uuid4().hex
    modules = []
    real_import = builtins.__import__

    def scoped_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openai":
            return SimpleNamespace(OpenAI=lambda **kwargs: client.sdk("actor"))
        return real_import(name, globals, locals, fromlist, level)

    try:
        package = ModuleType(prefix)
        package.__path__ = []
        sys.modules[prefix] = package
        modules.append(prefix)
        for name in ("base", "base_agent", "webshop"):
            record = next(r for r in manifest["files"] if r["source"] == "agent/" + name + ".py")
            if sha(record["path"]) != record["sha256"]:
                raise ValueError("native actor bytehash mismatch")
            mod = ModuleType(prefix + "." + name)
            mod.__package__ = prefix
            mod.__file__ = record["path"]
            mod.__dict__["__builtins__"] = dict(vars(builtins), __import__=scoped_import)
            sys.modules[mod.__name__] = mod
            modules.append(mod.__name__)
            exec(compile(Path(record["path"]).read_bytes(), mod.__file__, "exec"), mod.__dict__)
        yield mod.WebShopAgent(
            model_name=client.profile.model,
            temperature=0,
            max_tokens=client.profile.output_budget("answer"),
            api_key="local-endpoint-placeholder",
            base_url=client.profile.base_url,
        )
    finally:
        for name in reversed(modules):
            sys.modules.pop(name, None)


REWARD_SOURCES = {
    "env/env_systems/web_shopping_env/webshop_plus_client.py": (
        "6e0a07e371c7faca1fd84ad1113fadc3704ea31c299710ce2d551eab7775cd4a"
    ),
    "env/env_systems/web_shopping_env/runtime/reward_helpers.py": (
        "9dbf11debf3143c6080e68ad8eeeb8df6690d27f47c9e4560106d395ba0bc857"
    ),
}
REWARD_METRIC = "supplemental_plus_attribute_price_reward"


def replay_evaluator_reward(out, evidence):
    """Run the byte-pinned native scorer over evaluator-only saved inputs.

    Plus reward is optional feedback machinery in the pinned official runner,
    NOT its exact-ASIN success metric. This harness additionally requires it as
    a separately labelled diagnostic, without changing native binary fields.
    """
    import math

    if (
        not isinstance(evidence, dict)
        or evidence.get("metric") != REWARD_METRIC
        or evidence.get("enable_feedback") is not False
        or evidence.get("terminal_only") is not True
        or evidence.get("provenance")
        != {
            "native_commit": native.PIN,
            "entrypoint": "WebshopPlusEnvClient._compute_reward_entry",
            "source_sha256": REWARD_SOURCES,
        }
    ):
        raise ValueError("native evaluator reward provenance missing/invalid")
    for name, digest in REWARD_SOURCES.items():
        if sha(Path(out) / "native-source" / name) != digest:
            raise ValueError("native evaluator reward source mismatch")
    module = ModuleType("_reward_replay_" + uuid.uuid4().hex)
    sys.modules[module.__name__] = module
    try:
        path = Path(out) / "native-source" / next(reversed(REWARD_SOURCES))
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
        inputs = evidence["inputs"]
        replayed = module.compute_reward_for_step(
            inputs["step_result"], inputs["gt_step"], module.CatalogLookup(**inputs["catalog"])
        )
        value = replayed["reward"]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            or replayed != evidence.get("result")
        ):
            raise ValueError("native evaluator reward replay mismatch")
        return replayed
    finally:
        sys.modules.pop(module.__name__, None)


def verify_reward_coverage(out, row, steps, records):
    """Bind every terminal score to native task reconstruction and purchases."""
    import ast
    import re

    path = Path(out) / "native-source/env/env_systems/web_shopping_env/runtime/runner/task_files.py"
    if sha(path) != "1ed0b8e7d2ab825e69f7cc475f625b88a121e48ef3de3676c9c4dbec8fd40d8a":
        raise ValueError("native reward task reconstruction source mismatch")
    fn = next(
        n
        for n in ast.parse(path.read_bytes()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_reconstruct_task_def_from_hf_row"
    )
    scope = {"Dict": dict, "List": list, "re": re}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), scope)
    expected_steps = scope[fn.name](row)["steps"]
    if (
        len(records) != len(steps)
        or len(steps) != len(expected_steps)
        or [r.get("query_id") for r in records] != list(range(len(steps)))
    ):
        raise ValueError("native evaluator reward occurrence coverage mismatch")
    rewards = []
    for index, (step, record, expected_gt) in enumerate(
        zip(steps, records, expected_steps, strict=True)
    ):
        final = step["turns"][-1]["native_result"]
        evidence = record["evidence"]
        if (
            step["query_id"] != index
            or not final["done"]
            or evidence != final.get("evaluator_reward")
            or any(
                t["native_result"].get("evaluator_reward") is not None or t["native_result"]["done"]
                for t in step["turns"][:-1]
            )
        ):
            raise ValueError("native evaluator reward terminal binding mismatch")
        replayed = replay_evaluator_reward(out, evidence)
        inputs = evidence["inputs"]
        asins, prices = final["info"]["purchased_asins"], final["info"]["purchased_prices"]
        expected_input = {
            "step": expected_gt["step"],
            "target_asin": expected_gt["target_asin"],
            "purchased_asin": asins[0] if asins else None,
            "purchased_price": prices[0] if prices else None,
        }
        if (
            len(asins) > 1
            or len(asins) != len(prices)
            or inputs["gt_step"] != expected_gt
            or inputs["step_result"] != expected_input
        ):
            raise ValueError("native evaluator reward input/purchase binding mismatch")
        rewards.append(replayed["reward"])
    return {
        "metric": REWARD_METRIC,
        "official_success_metric": False,
        "sessions": len(rewards),
        "rewards": rewards,
        "mean_reward": sum(rewards) / len(rewards),
    }


def verify_public_experience(out, row, step):
    """Replay exact native record_turn/build_memory_entry with no evaluator input."""
    import ast
    from typing import Any, Optional

    path = Path(out) / "native-source/agent/webshop.py"
    if sha(path) != "b346bf2d7d4786770848c95c1159a87a00bea5cdb2724fd062c223311a7e1918":
        raise ValueError("native experience source mismatch")
    tree = ast.parse(path.read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WebShopAgent")
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in {"record_turn", "build_memory_entry"}
    ]
    scope = {"Dict": dict, "Any": Any, "Optional": Optional, "json": json}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope)
    actor = SimpleNamespace(turns=[], _last_raw_output=None)
    for turn in step["turns"]:
        observation = turn["native_result"]["public_observation"]
        if set(observation) != {"state"} or not isinstance(observation["state"], str):
            raise ValueError("native experience public projection mismatch")
        actor._last_raw_output = turn["raw_output"]
        scope["record_turn"](
            actor,
            turn_idx=turn["turn_idx"],
            prompt=turn["input_messages"][-1]["content"],
            action=turn["action"],
            observation=observation,
            reward=None,
            done=turn["native_result"]["done"],
            info={},
            raw_output=turn["raw_output"],
        )
    expected = scope["build_memory_entry"](
        actor, row["questions"][step["query_id"]], turn["action"], observation, reward=None
    )
    if json.loads(expected) != json.loads(step["experience"]["content"]):
        raise ValueError("native experience contains non-public/evaluator input")


def native_metrics(source_bytes, summaries):
    """Only the pinned upstream pure enrich_task_result; no provider imports."""
    import ast

    tree = ast.parse(source_bytes)
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "enrich_task_result"
    )
    scope = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "pinned:run_shopping.py", "exec"),
        {"Dict": dict, "Any": object},
        scope,
    )
    return scope["enrich_task_result"]({"total_steps": len(summaries), "steps": summaries})


def method_metrics(matrix, source_bytes):
    """Unmodified native global summary only for complete method cohorts."""
    import ast

    tree = ast.parse(source_bytes)
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_global_summary"
    )
    scope = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "pinned:run_shopping.py", "exec"),
        {"Dict": dict, "Any": object, "List": list},
        scope,
    )
    result = {}
    for system in dict.fromkeys(r["system"] for r in matrix):
        rows = [r for r in matrix if r["system"] == system]
        complete = all(r["status"] == "complete" for r in rows)
        result[system] = {
            "status": "complete" if complete else "incomplete-no-full-score",
            "expected_tasks": len(rows),
            "completed_tasks": sum(r["status"] == "complete" for r in rows),
        }
        if complete:
            result[system]["official_metrics"] = scope["build_global_summary"](
                [{**r["metrics"], "task_id": r["task_id"]} for r in rows]
            )
            rewards = [v for r in rows for v in r["supplemental_metrics"]["rewards"]]
            if len(rewards) != sum(r["metrics"]["total_steps"] for r in rows):
                raise ValueError("supplemental reward method coverage mismatch")
            result[system]["supplemental_metrics"] = {
                "metric": REWARD_METRIC,
                "official_success_metric": False,
                "sessions": len(rewards),
                "mean_reward": sum(rewards) / len(rewards),
            }
    return result


def run_task(row, *, runtime, client, memory, out, max_rounds):
    """Every ordered session, original local WebShop action loop and native memory.

    One task/method counter includes actor, embeddings, scoring and production
    extraction; cleanup/evidence failure forbids a completed task result.
    """
    import re
    from copy import deepcopy

    from .trajectory_suite import ArenaSession, ArenaTask

    out = Path(out)
    if out.exists():
        raise FileExistsError("fresh shopping task root required")
    if (
        not row["questions"]
        or len(row["questions"]) != len(row["answers"])
        or type(max_rounds) is not int
        or max_rounds < 1
    ):
        raise ValueError("aligned ordered native shopping sessions required")
    task = ArenaTask(
        row["id"], DOMAIN, tuple(ArenaSession(q) for q in row["questions"]), tuple(row["answers"])
    )
    if client.live:
        client.permit.validate(task, client.profile, max_rounds)
    out.mkdir(parents=True)
    manifest = json.loads(Path(runtime).read_bytes())
    (out / "input-task.json").write_bytes(native._json(row))
    (out / "effective-profile.json").write_bytes(
        native._json(client.profile.model_dump(mode="json"))
    )
    for record in manifest["files"]:
        target = out / "native-source" / record["source"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(record["path"]).read_bytes())
    rows, attempted, closed = [], None, False
    mem = None
    try:
        mem = memory(client.budget, out) if callable(memory) else memory
        if mem.budget is not client.budget:
            raise ValueError("shared whole-task memory budget required")
        with (
            ShoppingRuntime(runtime, budget=client.budget, out=out / "runtime") as worker,
            native_actor(runtime, client) as actor,
        ):
            client.preflight()
            for index, question in enumerate(row["questions"]):
                attempted = index
                client.check()
                actor.reset()
                opened = worker.request("open", row=row, index=index, max_rounds=max_rounds)
                conversation = deepcopy(opened["conversation_start"])
                context = mem.query(opened["public_observation"]["state"])
                observation = opened["public_observation"]
                turns, final = [], None
                try:
                    for turn in range(max_rounds):
                        client.check()
                        prompt = observation["state"]
                        if turn == 0 and context:
                            prompt = (
                                "Relevant past agent experiences:\n" + context + "\n\n" + prompt
                            )
                        inputs = [*conversation, {"role": "user", "content": prompt}]
                        action = actor.act_with_messages(inputs)
                        if not re.fullmatch(r"(?:search|click)\[[^\]\r\n]*\]", action):
                            raise native.NativeAbort(
                                "undeclared shopping action; generated-code execution forbidden"
                            )
                        final = worker.request("step", action=action)
                        observation = final["public_observation"]
                        # No hidden target/reward/info labels in native experience.
                        actor.record_turn(
                            turn_idx=turn + 1,
                            prompt=prompt,
                            action=action,
                            observation=observation,
                            reward=None,
                            done=final["done"],
                            info={},
                            raw_output=actor._last_raw_output,
                        )
                        conversation = [
                            *inputs,
                            {"role": "assistant", "content": actor._last_raw_output},
                        ]
                        trace = {
                            "turn_idx": turn + 1,
                            "input_messages": inputs,
                            "action": action,
                            "raw_output": actor._last_raw_output,
                            "native_result": final,
                        }
                        native._append(out / "turns.jsonl", {"query_id": index, **trace})
                        turns.append(trace)
                        if final["done"]:
                            break
                    if final is None or not final["done"]:
                        raise ValueError("native session incomplete at frozen action bound")
                    evaluator_reward = final.get("evaluator_reward")
                    replay_evaluator_reward(out, evaluator_reward)
                    native._append(
                        out / "evaluator-rewards.jsonl",
                        {"query_id": index, "evidence": evaluator_reward},
                    )
                    content = actor.build_memory_entry(question, action, observation, reward=None)
                    experience = {
                        "content": content,
                        "authorship": "agent",
                        "source_ref": f"memoryarena/{DOMAIN}/{row['id']}/session/{index}",
                        "session_index": index,
                        "feedback_in_memory": False,
                        "native_source_commit": native.PIN,
                    }
                    readback = mem.insert(experience)
                    if readback.get("readback_verified") is not True:
                        raise ValueError("memory committed readback proof missing")
                    post = mem.query(opened["public_observation"]["state"])
                    step = {
                        "query_id": index,
                        "memory_context": context,
                        "turns": turns,
                        "reward": final["reward"],
                        "summary": final["summary"],
                        "experience": experience,
                        "memory_readback": readback,
                        "post_insert_readback": post,
                    }
                    native._append(out / "steps.jsonl", step)
                    rows.append(step)
                finally:
                    origin = sys.exception()
                    try:
                        proof = worker.request("close_session")
                        if not proof["native_closed"] or proof["raw_environment_active"]:
                            raise ValueError("native session close readback failed")
                    except BaseException as cleanup:
                        if origin is None:
                            raise
                        origin.add_note(
                            'Shopping session cleanup: ' + safe_error(cleanup)
                        )
        mem.close()
        mem = None
        client.close()
        closed = True
        client.check()
        metrics = native_metrics(
            (out / "native-source/run_shopping.py").read_bytes(), [r["summary"] for r in rows]
        )
        result = {
            "protocol": PROTOCOL,
            "status": "complete",
            "task_id": row["id"],
            "input_sha256": hashlib.sha256(native._json(row)).hexdigest(),
            "completed_sessions": len(rows),
            "expected_sessions": len(row["questions"]),
            "calls": client.budget.calls,
            "native_closed": closed,
            "metrics": metrics,
            "supplemental_metrics": verify_reward_coverage(
                out,
                row,
                rows,
                [
                    {
                        "query_id": s["query_id"],
                        "evidence": s["turns"][-1]["native_result"]["evaluator_reward"],
                    }
                    for s in rows
                ],
            ),
            "transport": "live" if client.live else "offline-authored-NOT-quality",
            "artifact_sha256": {
                str(p.relative_to(out)).replace("\\", "/"): sha(p)
                for p in sorted(out.rglob("*"))
                if p.is_file()
            },
        }
        (out / "result.pending.json").write_bytes(native._json(result))
        verify_task(out, manifest_name="result.pending.json")
        client.check()
        (out / "result.pending.json").replace(out / "result.json")
        return json.loads((out / "result.json").read_bytes())
    except BaseException as exc:
        cleanup_error = None
        try:
            if mem is not None:
                mem.close()
            client.close()
        except BaseException as cleanup:
            cleanup_error = safe_error(cleanup)
        for name in ("result.pending.json", "result.json"):
            (out / name).unlink(missing_ok=True)
        failure = {
            "status": "failed",
            "completed_sessions": len(rows),
            "expected_sessions": len(row["questions"]),
            "calls": client.budget.calls,
            "native_closed": closed,
            "error": safe_error(exc),
            "cleanup_error": cleanup_error,
            "cleanup_diagnostics": safe_notes(exc),
            "sessions": [
                {
                    "id": i,
                    "status": "complete"
                    if i < len(rows)
                    else "failed"
                    if i == attempted
                    else "unrun",
                }
                for i in range(len(row["questions"]))
            ],
        }
        (out / "failure.json").write_bytes(native._json(failure))
        raise RuntimeError("shopping task failed; see failure.json") from exc


def verify_task(out, *, manifest_name="result.json"):
    """Dataset/network-free saved native wire, input, scoring and close verifier."""
    out = Path(out)
    if manifest_name not in {"result.json", "result.pending.json"}:
        raise ValueError("unsupported manifest")
    result = json.loads((out / manifest_name).read_bytes())
    if (
        result["protocol"] != PROTOCOL
        or result["status"] != "complete"
        or not result["native_closed"]
        or (out / "failure.json").exists()
    ):
        raise ValueError("incomplete native shopping bundle")
    if "evaluator-rewards.jsonl" not in result["artifact_sha256"]:
        raise ValueError("mandatory evaluator reward evidence missing")
    required = {
        "evaluator-rewards.jsonl",
        *("native-source/" + name for name in REWARD_SOURCES),
        "http.jsonl",
        "input-task.json",
        "effective-profile.json",
        "steps.jsonl",
        "turns.jsonl",
        "runtime/runtime.jsonl",
        "runtime/runtime-closed.json",
        "native-source/run_shopping.py",
        "native-source/agent/webshop.py",
    }
    if not required <= result["artifact_sha256"].keys():
        raise ValueError("mandatory shopping evidence missing")
    for name, digest in result["artifact_sha256"].items():
        path = (out / name).resolve()
        if not path.is_relative_to(out.resolve()) or sha(path) != digest:
            raise ValueError("shopping evidence bytehash mismatch")
    row = json.loads((out / "input-task.json").read_bytes())
    if hashlib.sha256(native._json(row)).hexdigest() != result["input_sha256"]:
        raise ValueError("shopping input mismatch")
    steps = [json.loads(line) for line in (out / "steps.jsonl").read_bytes().splitlines()]
    if (
        len(steps) != len(row["questions"])
        or result["completed_sessions"] != len(steps)
        or result["expected_sessions"] != len(steps)
        or [s["query_id"] for s in steps] != list(range(len(steps)))
    ):
        raise ValueError("ordered shopping occurrence coverage mismatch")
    reward_records = [
        json.loads(line) for line in (out / "evaluator-rewards.jsonl").read_bytes().splitlines()
    ]
    if verify_reward_coverage(out, row, steps, reward_records) != result.get(
        "supplemental_metrics"
    ):
        raise ValueError("native supplemental reward report replay mismatch")
    closed = json.loads((out / "runtime/runtime-closed.json").read_bytes())
    if closed != {"native_closed": True, "raw_environment_active": False, "process_exit": 0}:
        raise ValueError("native close proof mismatch")
    wire = [json.loads(line) for line in (out / "http.jsonl").read_bytes().splitlines()]
    generations = [w for w in wire if w.get("url", "").endswith("/chat/completions")]
    production = (
        [json.loads(line) for line in (out / "production-raw.jsonl").read_bytes().splitlines()]
        if (out / "production-raw.jsonl").exists()
        else []
    )
    if len(generations) + len(production) != result["calls"] or len(generations) != sum(
        len(s["turns"]) for s in steps
    ):
        raise ValueError("actual native/production generation accounting mismatch")
    from .rack import RackProfile

    profile = RackProfile.model_validate(json.loads((out / "effective-profile.json").read_bytes()))
    native_turns = [t for s in steps for t in s["turns"]]
    for gen, turn in zip(generations, native_turns, strict=True):
        position = wire.index(gen)
        if position < 1:
            raise ValueError("native tokenizer wire missing")
        tokenized = wire[position - 1]
        request, response = gen["request"], gen["response"]
        tokens = tokenized.get("response", {}).get("count")
        message = response["choices"][0]["message"]
        if (
            gen.get("status") != 200
            or "error" in gen
            or not tokenized["url"].endswith("/tokenize")
            or tokenized.get("status") != 200
            or type(tokens) is not int
            or tokens < 1
            or request.get("model") != profile.model
            or response.get("model") != profile.model
            or request.get("stream") is not False
            or request.get("n") != 1
            or request.get("temperature") != 0
            or request.get("max_tokens") != profile.output_budget("answer")
            or "tools" in request
            or "tool_choice" in request
            or request.get("chat_template_kwargs") != {"enable_thinking": False}
            or any(
                tokenized["request"].get(k) != request.get(k)
                for k in ("messages", "model", "chat_template_kwargs")
            )
            or response["choices"][0].get("finish_reason") != "stop"
            or message.get("tool_calls")
            or message.get("content") != turn["raw_output"]
            or response.get("usage", {}).get("prompt_tokens") != tokens
            or type(response.get("usage", {}).get("completion_tokens")) is not int
            or not 0 <= response["usage"]["completion_tokens"] <= profile.output_budget("answer")
        ):
            raise ValueError("native shopping wire/model/usage/SDK schema mismatch")
        messages = request["messages"]
        if messages[1:] != turn["input_messages"] or messages[0]["role"] != "system":
            raise ValueError("native actor prompt wire differs from saved turn")
    for record in production:
        response = record.get("response", {})
        if (
            "error" in record
            or response.get("model") != profile.model
            or not response.get("usage")
            or any(c.get("finish_reason") != "stop" for c in response.get("choices", []))
        ):
            raise ValueError("production ingestion usage/wire incomplete")
    for step in steps:
        verify_public_experience(out, row, step)
        final = step["turns"][-1]["native_result"]
        if (
            not final["done"]
            or final["summary"] != step["summary"]
            or step["reward"] != final["reward"]
            or step["memory_readback"].get("readback_verified") is not True
        ):
            raise ValueError("native shopping result/readback mismatch")
        expected = [row["answers"][step["query_id"]]["target_asin"].upper()]
        purchased = final["info"]["purchased_asins"]
        if type(step["reward"]) is not bool or step["reward"] != (purchased == expected):
            raise ValueError("native exact ASIN judgement mismatch")
        content = json.loads(step["experience"]["content"])
        if (
            content["reward"] is not None
            or content["final_observation"] != final["public_observation"]
            or content["task"] != row["questions"][step["query_id"]]
            or any(t["reward"] is not None or t["info"] != {} for t in content["turns"])
        ):
            raise ValueError("hidden evaluator feedback in native experience")
    if (out / "production.mind").exists():
        verify_memory_store(out, steps)
    metrics = native_metrics(
        (out / "native-source/run_shopping.py").read_bytes(), [s["summary"] for s in steps]
    )
    if metrics != result["metrics"]:
        raise ValueError("native shopping metrics replay mismatch")
    return result


def verify_memory_store(out, steps):
    """Verify acknowledged processing, not permanent plaintext retention.

    Reconstruct canonical scoped events independently via _validate_rows and
    bind them to construction-time readbacks. Only production lifecycle's
    frame-local scrub shape may replace an acknowledged span with NULL.
    Quiescent immutable readback never creates SQLite WAL sidecars; this does
    not grant reuse/completion authority to a previously failed bundle.
    """
    import sqlite3

    from .beam_ingest import _validate_rows

    out = Path(out)
    saved = [json.loads(line) for line in (out / "memory-events.jsonl").read_bytes().splitlines()]
    events = [native.document_experience(step["experience"]) for step in steps]
    conn = sqlite3.connect(
        (out / "production.mind").resolve().as_uri() + "?mode=ro&immutable=1", uri=True
    )
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM source_events ORDER BY id")]
        commits, _ = _validate_rows(rows, events)
        if conn.execute("SELECT COUNT(*) FROM memory_provenance").fetchone()[0] != sum(
            len(step["memory_readback"].get("provenance", [])) for step in steps
        ):
            raise ValueError("production document provenance coverage mismatch")
        for step, envelope, event, commit, source_row in zip(
            steps, saved, events, commits, rows, strict=True
        ):
            if (
                envelope["event"] != event.model_dump(mode="json")
                or envelope["receipt"] != step["memory_readback"]
                or any(
                    type(step["memory_readback"].get(k)) is not type(v)
                    or step["memory_readback"].get(k) != v
                    for k, v in commit.items()
                )
                or step["memory_readback"].get("readback_verified") is not True
                or step["memory_readback"].get("content_sha256")
                != hashlib.sha256(step["experience"]["content"].encode()).hexdigest()
            ):
                raise ValueError("production source envelope/commit readback mismatch")
            provenance = [
                dict(r)
                for r in conn.execute(
                    "SELECT frame_id, source_event_id, source_type, source_ref, "
                    "message_id, verbatim_span FROM memory_provenance "
                    "WHERE source_event_id=? ORDER BY frame_id",
                    (commit["row_id"],),
                )
            ]
            expected_provenance = step["memory_readback"]["provenance"]
            if (
                len(provenance) != len(expected_provenance)
                or type(step["memory_readback"].get("provenance_count")) is not int
                or step["memory_readback"]["provenance_count"] != len(provenance)
                or not {f["id"] for f in step["memory_readback"]["frames"]}
                <= {p["frame_id"] for p in provenance}
            ):
                raise ValueError("production document provenance mismatch")
            for observed, expected in zip(provenance, expected_provenance, strict=True):
                frame = conn.execute(
                    "SELECT tenant_id, principal_id, namespace, status, content, folded_content "
                    "FROM memory_frames WHERE id=?",
                    (observed["frame_id"],),
                ).fetchone()
                scoped = conn.execute(
                    "SELECT tenant_id, principal_id, namespace FROM memory_provenance "
                    "WHERE frame_id=? AND source_event_id=?",
                    (observed["frame_id"], commit["row_id"]),
                ).fetchone()
                if (
                    frame is None
                    or scoped is None
                    or any(
                        record[k] != commit[k]
                        for record in (frame, scoped)
                        for k in ("tenant_id", "principal_id", "namespace")
                    )
                ):
                    raise ValueError("production document provenance scope/frame mismatch")
                # Lifecycle scrubs spans only for the affected frames, not every
                # frame sharing a redacted SourceEvent. Keep identity and all
                # retained spans exact; a NULL alone is not privacy evidence.
                comparable = dict(observed)
                if observed["verbatim_span"] is None and expected["verbatim_span"] is not None:
                    if (
                        frame["status"] not in {"superseded", "forgotten", "tombstoned"}
                        or frame["content"] != "[forgotten]"
                        or frame["folded_content"] != ""
                        or json.loads(source_row["payload_json"]) != {}
                    ):
                        raise ValueError(
                            "production document provenance privacy transition mismatch"
                        )
                    comparable["verbatim_span"] = expected["verbatim_span"]
                if (
                    comparable != expected
                    or any(
                        type(expected.get(k)) is not int for k in ("frame_id", "source_event_id")
                    )
                    or observed["source_type"] != "document"
                    or observed["source_ref"] != event.source_ref
                    or observed["message_id"] is not None
                ):
                    raise ValueError("production document provenance mismatch")
    finally:
        conn.close()


def build_spec(
    *,
    data_root,
    pins,
    runtime,
    profile,
    output_root,
    embedding_url,
    calls_per_task,
    seconds_per_task,
    campaign_seconds,
    max_rounds,
    production_memory_qualification=None,
    native_document_factories=None,
):
    plan = plan_dataset(data_root, pins)
    manifest = json.loads(Path(runtime).read_bytes())
    return {
        **plan,
        "data_root": str(Path(data_root).resolve()),
        "data_pins": pins,
        "runtime_path": str(Path(runtime).resolve()),
        "runtime_sha256": sha(runtime),
        "runtime": manifest,
        "profile": profile.model_dump(mode="json"),
        "output_root": str(Path(output_root).resolve()),
        "embedding_url": embedding_url,
        "bounds": {
            "calls_per_task_method": calls_per_task,
            "seconds_per_task_method": seconds_per_task,
            "campaign_seconds": campaign_seconds,
            "max_rounds": max_rounds,
            "feedback_in_memory": False,
            "include_history": False,
        },
        "production_memory_qualification": production_memory_qualification,
        "native_document_factories": native_document_factories or {},
        "files": common._code_inventory(),
    }


def verify_spec(spec_path, approval_path, *, allow_started=False):
    from .rack import RackProfile

    raw = Path(spec_path).read_bytes()
    approval = json.loads(Path(approval_path).read_bytes())
    if (
        approval.get("approved") is not True
        or approval.get("reviewer_kind") not in {"parent", "human"}
        or approval.get("spec_sha256") != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError("independent byte-bound parent approval required")
    spec = json.loads(raw)
    plan = plan_dataset(spec["data_root"], spec["data_pins"])
    if any(spec.get(k) != v for k, v in plan.items()):
        raise ValueError("full released shopping scope/occurrence coverage mismatch")
    p = RackProfile.model_validate(spec["profile"])
    bounds = spec["bounds"]
    if (
        "qwen" not in p.model.lower()
        or p.temperature != 0
        or bounds["calls_per_task_method"] != p.max_generation_calls
        or bounds["seconds_per_task_method"] != p.max_run_seconds
        or type(bounds["max_rounds"]) is not int
        or not 1 <= bounds["max_rounds"] <= 100
        or bounds["campaign_seconds"] <= 0
        or bounds["feedback_in_memory"] is not False
        or bounds["include_history"] is not False
    ):
        raise ValueError("declared Qwen equal whole-task budget contract required")
    if spec["protocol"] != PROTOCOL or spec["files"] != common._code_inventory():
        raise ValueError("frozen shopping producer inventory mismatch")
    if (
        sha(spec["runtime_path"]) != spec["runtime_sha256"]
        or json.loads(Path(spec["runtime_path"]).read_bytes()) != spec["runtime"]
        or spec["runtime"]["native_commit"] != native.PIN
    ):
        raise ValueError("frozen native runtime mismatch")
    for record in [
        *spec["runtime"]["files"],
        {"path": spec["runtime"]["worker"], "sha256": spec["runtime"]["worker_sha256"]},
    ]:
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("frozen native source mismatch")
    for system, binding in spec["native_document_factories"].items():
        if (
            system not in {"mem0", "cognee"}
            or binding.get("kind") != "native-document-factory@v1"
            or binding.get("backend") != system
        ):
            raise ValueError("real native document factory binding required")
        for key in ("source", "qualification"):
            if (
                not Path(binding[key + "_path"]).is_absolute()
                or sha(binding[key + "_path"]) != binding[key + "_sha256"]
            ):
                raise ValueError("real native document factory hash mismatch")
    root = Path(spec["output_root"]).resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("external shopping output required")
    if not allow_started and root.exists():
        raise FileExistsError("fresh shopping campaign root required")
    return spec


class ShoppingPermit:
    def __init__(self, spec_path, approval_path):
        self.spec_path, self.approval_path = Path(spec_path), Path(approval_path)

    def validate(self, task, profile, max_rounds):
        spec = verify_spec(self.spec_path, self.approval_path, allow_started=True)
        if (
            task.domain != DOMAIN
            or spec["profile"] != profile.model_dump(mode="json")
            or spec["bounds"]["max_rounds"] != max_rounds
            or {
                "id": task.id,
                "sessions": len(task.sessions),
                "input_sha256": common.task_digest(task),
            }
            not in spec["tasks"]
        ):
            raise ValueError("task/profile differs from independently frozen shopping input")
        return spec


def execute_shopping(spec_path, approval_path):
    """Full Shopping execution only. Admission precedes all clients/writes."""
    spec = verify_spec(spec_path, approval_path)
    from .rack import RackProfile

    profile = RackProfile.model_validate(spec["profile"])
    common.verify_memory_qualification(spec["production_memory_qualification"], profile)
    # Full product bytes are checked once at campaign admission, not per actor turn.
    for asset in spec["runtime"]["assets"]:
        if (
            Path(asset["path"]).stat().st_size != asset["bytes"]
            or sha(asset["path"]) != asset["sha256"]
        ):
            raise ValueError("pinned product asset mismatch")
    return _execute_approved(spec, spec_path, approval_path)


def verify_root(out, *, payload=None, verifier=None, expected_matrix=None, expected_rows=None):
    """Read-only root reconciliation; never opens native SDKs or dataset assets."""
    out = Path(out)
    result = (
        payload if payload is not None else json.loads((out / "campaign-result.json").read_bytes())
    )
    frozen = json.loads((out / "frozen-matrix.json").read_bytes())
    rows = json.loads((out / "task-statuses.json").read_bytes())
    if expected_rows is not None and rows != expected_rows:
        raise ValueError("root statuses differ from actual admitted job accounting")
    if expected_matrix is not None and frozen != expected_matrix:
        raise ValueError("frozen root matrix differs from admitted matrix")
    if (
        result.get("protocol") != PROTOCOL
        or result.get("tasks") != rows
        or len(rows) != len(frozen)
    ):
        raise ValueError("root task statuses/coverage mismatch")
    identity = ("task_id", "domain", "system", "input_sha256", "expected_sessions")
    spec_path = out / "execution-spec.json"
    spec = json.loads(spec_path.read_bytes()) if spec_path.exists() else None
    if spec is not None:
        selected = [*spec["systems"], *sorted(spec["native_document_factories"])]
        planned = [
            {
                "task_id": t["id"],
                "domain": DOMAIN,
                "system": s,
                "input_sha256": t["input_sha256"],
                "expected_sessions": t["sessions"],
            }
            for t in spec["tasks"]
            for s in selected
        ]
        if [{k: r[k] for k in identity} for r in frozen] != planned:
            raise ValueError("root matrix differs from saved frozen spec")
        metadata = {
            "spec_sha256": sha(spec_path),
            "scope": (
                "all released 150 tasks / 900 ordered session occurrences per admitted method; "
                "local simulated WebShop only"
            ),
            "optional_coverage": [
                {
                    "task_id": t["id"],
                    "system": s,
                    "expected_sessions": t["sessions"],
                    "status": "qualified-selected"
                    if s in spec["native_document_factories"]
                    else "unqualified",
                    "calls": 0,
                }
                for t in spec["tasks"]
                for s in ("mem0", "cognee")
            ],
        }
        if any(result.get(k) != v for k, v in metadata.items()):
            raise ValueError("root frozen spec metadata mismatch")
        approval_path = out / "execution-approval.json"
        if approval_path.exists():
            approval = json.loads(approval_path.read_bytes())
            if (
                approval.get("approved") is not True
                or approval.get("reviewer_kind") not in {"parent", "human"}
                or approval.get("spec_sha256") != sha(spec_path)
            ):
                raise ValueError("root frozen spec approval mismatch")
    seen = set()
    for planned, row in zip(frozen, rows, strict=True):
        key = (row["task_id"], row["system"])
        if key in seen or any(row.get(k) != planned.get(k) for k in identity):
            raise ValueError("root frozen matrix identity/order mismatch")
        seen.add(key)
        if (
            row["status"] not in {"complete", "failed", "unrun"}
            or type(row["calls"]) is not int
            or row["calls"] < 0
            or type(row["completed_sessions"]) is not int
            or not 0 <= row["completed_sessions"] <= row["expected_sessions"]
        ):
            raise ValueError("root task status/counter mismatch")
        if row["status"] == "unrun" and (row["calls"] or row["completed_sessions"]):
            raise ValueError("unrun root row carries spend/completion")
        if row["status"] != "complete":
            if any(k in row for k in ("metrics", "supplemental_metrics", "result_sha256")):
                raise ValueError("incomplete root row carries completed score")
            failure = out / str(row["task_id"]) / row["system"] / "failure.json"
            if row["status"] == "failed" and failure.exists():
                saved = json.loads(failure.read_bytes())
                if any(
                    saved.get(k) != row[k]
                    for k in ("calls", "completed_sessions", "expected_sessions")
                ):
                    raise ValueError("root failed leaf spend/progress mismatch")
            continue
        target = out / str(row["task_id"]) / row["system"]
        if Path(row["output"]).resolve() != target.resolve():
            raise ValueError("root leaf output mismatch")
        leaf = (verifier or verify_task)(target)
        if (target / "input-task.json").exists():
            from .trajectory_suite import ArenaSession, ArenaTask

            saved = json.loads((target / "input-task.json").read_bytes())
            task = ArenaTask(
                saved["id"],
                DOMAIN,
                tuple(ArenaSession(q) for q in saved["questions"]),
                tuple(saved["answers"]),
            )
            if saved["id"] != row["task_id"] or common.task_digest(task) != row["input_sha256"]:
                raise ValueError("root native leaf frozen input mismatch")
        if (
            leaf.get("status") != "complete"
            or row["result_sha256"] != sha(target / "result.json")
            or row["completed_sessions"] != row["expected_sessions"]
            or any(
                leaf.get(k) != row.get(k)
                for k in ("calls", "completed_sessions", "metrics", "supplemental_metrics")
            )
        ):
            raise ValueError("root verified leaf digest/metrics/spend mismatch")
    expected = {
        "quality_complete": all(r["status"] == "complete" for r in rows),
        "calls": sum(r["calls"] for r in rows),
        "expected_sessions": sum(r["expected_sessions"] for r in rows),
        "completed_sessions": sum(r["completed_sessions"] for r in rows),
    }
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError("root totals/completion mismatch")
    if "method_metrics" in result or spec is not None:
        sources = [
            out / str(r["task_id"]) / r["system"] / "native-source/run_shopping.py"
            for r in rows
            if r["status"] == "complete"
        ]
        if sources:
            source = sources[0].read_bytes()
        else:
            import zipfile

            with zipfile.ZipFile(out / "frozen-source.zip") as archive:
                source = archive.read("runtime/source/run_shopping.py")
        if spec is not None and "runtime" in spec:
            pin = next(
                r["sha256"] for r in spec["runtime"]["files"] if r["source"] == "run_shopping.py"
            )
            if hashlib.sha256(source).hexdigest() != pin:
                raise ValueError("root native metric source mismatch")
        if result.get("method_metrics") != method_metrics(rows, source):
            raise ValueError("root official/supplemental method metrics mismatch")
    return result


def execute_matrix(
    tasks,
    *,
    systems,
    out,
    call_budget,
    seconds_budget,
    campaign_seconds,
    run_one,
    verifier=verify_task,
    publish=None,
    root_metadata=None,
):
    from .parallel_campaign import CampaignBudget, CampaignStopped, Job, JobContext, RequestLimiter

    tasks, systems, out = list(tasks), list(systems), Path(out)
    if (
        not tasks
        or len({t.id for t in tasks}) != len(tasks)
        or any(t.domain != DOMAIN for t in tasks)
        or not systems
        or len(set(systems)) != len(systems)
        or not set(systems) <= set(SYSTEMS) | {"mem0", "cognee"}
    ):
        raise ValueError("unique shopping-only paired matrix required")
    outer = CampaignBudget(
        max_calls=len(tasks) * len(systems) * call_budget, max_seconds=campaign_seconds
    )
    limiter = RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072)
    matrix = [
        {
            "task_id": t.id,
            "domain": DOMAIN,
            "system": s,
            "input_sha256": common.task_digest(t),
            "expected_sessions": len(t.sessions),
            "completed_sessions": 0,
            "calls": 0,
            "status": "unrun",
        }
        for t in tasks
        for s in systems
    ]
    out.mkdir(parents=True, exist_ok=False)
    (out / "frozen-matrix.json").write_bytes(native._json(matrix))
    frozen_matrix = json.loads(native._json(matrix))
    (out / "task-statuses.json").write_bytes(native._json(matrix))
    for row in matrix:
        task = next(t for t in tasks if t.id == row["task_id"])
        target = out / str(task.id) / row["system"]
        job = Job(
            f"shopping:{task.id}:{row['system']}",
            f"shopping:{task.id}",
            target,
            call_budget,
            seconds_budget,
        )
        context = None
        try:
            outer.check()
            if not outer.reserve(job):
                raise CampaignStopped("whole-task campaign reservation unavailable")
            context = JobContext(job, outer, limiter)
            row.update(status="running", output=str(target))
            (out / "task-statuses.json").write_bytes(native._json(matrix))
            value = run_one(task, row["system"], context, target)
            verified = verifier(target)
            context.check()
            if (
                verified != value
                or value.get("status") != "complete"
                or value.get("completed_sessions") != len(task.sessions)
                or value.get("calls") != context.calls
            ):
                raise ValueError("leaf evidence/result/spend differs; root refuses completion")
            row.update(
                status="complete",
                completed_sessions=len(task.sessions),
                result_sha256=sha(target / "result.json"),
                metrics=value.get("metrics"),
                supplemental_metrics=value.get("supplemental_metrics"),
            )
        except (Exception, native.NativeAbort) as exc:
            row.update(
                status="failed" if context is not None else "unrun",
                error=safe_error(exc),
            )
            if (target / "failure.json").exists():
                failure = json.loads((target / "failure.json").read_bytes())
                row["completed_sessions"] = failure["completed_sessions"]
        finally:
            if context is not None:
                row["calls"] = context.calls
                outer.release(context)
            (out / "task-statuses.json").write_bytes(native._json(matrix))
    result = {
        "protocol": PROTOCOL,
        "authority": "campaign-result.json only; leaf manifests never override failed root jobs",
        "tasks": matrix,
        "quality_complete": all(r["status"] == "complete" for r in matrix),
        "calls": outer.calls,
        "expected_sessions": sum(r["expected_sessions"] for r in matrix),
        "completed_sessions": sum(r["completed_sessions"] for r in matrix),
        **(root_metadata or {}),
    }
    if sum(r["calls"] for r in matrix) != outer.calls:
        raise ValueError("root actual spend reconciliation mismatch")
    pending = out / "campaign-result.pending.json"
    accounted_rows = json.loads(native._json(matrix))

    def check_finalization():
        # A shared job failure still permits an explicitly incomplete receipt.
        if outer.clock() >= outer.deadline:
            raise CampaignStopped("campaign deadline exhausted during root finalization")
        if not outer.failed or result["quality_complete"]:
            outer.check()

    try:
        check_finalization()
        if publish is None:
            pending.write_bytes(native._json(result))
        else:
            publish(pending, result)
        check_finalization()
        if json.loads(pending.read_bytes()) != result:
            raise ValueError("root result readback mismatch")
        verify_root(
            out,
            payload=result,
            verifier=verifier,
            expected_matrix=frozen_matrix,
            expected_rows=accounted_rows,
        )
        if result["calls"] != outer.calls:
            raise ValueError("root actual spend reconciliation mismatch")
        check_finalization()
        pending.replace(out / "campaign-result.json")
        check_finalization()
        saved = json.loads((out / "campaign-result.json").read_bytes())
        if saved != result:
            raise ValueError("root final readback mismatch")
        check_finalization()
        return saved
    except BaseException:
        pending.unlink(missing_ok=True)
        (out / "campaign-result.json").unlink(missing_ok=True)
        raise


def _execute_approved(
    spec, spec_path, approval_path, *, _matrix_executor=None, _permit_factory=None
):
    import importlib.util
    import zipfile

    from gluomem.llm.modules import DspyExtractors

    from .rack import RackClient, RackProfile, controlled_config

    profile = RackProfile.model_validate(spec["profile"])
    factories = {}
    for system, binding in spec["native_document_factories"].items():
        module_spec = importlib.util.spec_from_file_location(
            "_shopping_factory_" + system, binding["source_path"]
        )
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        module.verify_qualification(Path(binding["qualification_path"]), profile=profile)
        if not callable(getattr(module, "create_memory", None)):
            raise ValueError("real qualified document factory required")
        factories[system] = module.create_memory
    tasks = load_arena(Path(spec["data_root"]), domain=DOMAIN, pins=spec["data_pins"])
    rows = {
        r["id"]: r
        for r in map(
            json.loads, (Path(spec["data_root"]) / DOMAIN / "data.jsonl").read_bytes().splitlines()
        )
    }
    permit = ShoppingPermit(spec_path, approval_path)
    root = Path(spec["output_root"])

    def run_one(task, system, budget, target):
        task_permit = _permit_factory(system) if _permit_factory is not None else permit
        task_permit.validate(task, profile, spec["bounds"]["max_rounds"])
        if not (root / "execution-spec.json").exists():
            (root / "execution-spec.json").write_bytes(Path(spec_path).read_bytes())
            (root / "execution-approval.json").write_bytes(Path(approval_path).read_bytes())
            with zipfile.ZipFile(root / "frozen-source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
                for index, item in enumerate(spec["files"]):
                    archive.write(
                        item["path"], arcname=f"producer/{index}/" + Path(item["path"]).name
                    )
                archive.write(spec["runtime_path"], arcname="runtime/runtime.json")
                archive.write(
                    spec["runtime"]["worker"], arcname="runtime/shopping_native_worker.py"
                )
                for item in spec["runtime"]["files"]:
                    archive.write(item["path"], arcname="runtime/source/" + item["source"])
        client = ShoppingTransport.__new__(ShoppingTransport)
        client.live, client._identity_ready, client.qualification = True, False, None
        client.live_transport_kind, client.permit = "direct-http", task_permit
        RackClient.__init__(client, profile, receipt=target / "http.jsonl", budget=budget)
        encoder = common.BudgetedArenaEmbedder(spec["embedding_url"], budget=budget)

        def memory(context, output):
            if system in factories:
                return factories[system](
                    profile=profile, client=client, budget=context, embedder=encoder, out=output
                )
            if system != "gluomem":
                return common.RawExperienceMemory(
                    system, budget=context, embedder=encoder, top_k=10
                )
            cfg = controlled_config(
                profile,
                data=Path(spec["data_root"]) / DOMAIN / "data.jsonl",
                sha256=spec["data_pins"][DOMAIN + "/data.jsonl"],
                embedder=encoder,
            )
            (output / "effective-config.json").write_bytes(
                native._json(cfg.model_dump(mode="json"))
            )
            (output / "encoder.json").write_bytes(native._json(encoder.fingerprint.as_dict()))
            router = common.ArenaProductionRouter(
                cfg, client=client, trace=output / "production-raw.jsonl", budget=context
            )
            return common.GluoExperienceMemory(
                cfg=cfg,
                extractors=DspyExtractors(cfg, router=router),
                embedder=encoder,
                budget=context,
                db_path=output / "production.mind",
                evidence_path=output / "memory-events.jsonl",
            )

        try:
            return run_task(
                rows[task.id],
                runtime=spec["runtime_path"],
                client=client,
                memory=memory,
                out=target,
                max_rounds=spec["bounds"]["max_rounds"],
            )
        finally:
            origin = sys.exception()
            try:
                client.close()
            except BaseException as cleanup:
                if origin is None:
                    raise
                origin.add_note(
                    'Shopping client cleanup: ' + safe_error(cleanup)
                )

    bounds = spec["bounds"]

    def publish_root(path, result):
        source = Path(spec["runtime"]["source_root"]) / "run_shopping.py"
        result["method_metrics"] = method_metrics(result["tasks"], source.read_bytes())
        path.write_bytes(native._json(result))

    return (_matrix_executor or execute_matrix)(
        tasks,
        systems=[*SYSTEMS, *sorted(factories)],
        out=root,
        call_budget=bounds["calls_per_task_method"],
        seconds_budget=bounds["seconds_per_task_method"],
        campaign_seconds=bounds["campaign_seconds"],
        run_one=run_one,
        publish=publish_root,
        root_metadata={
            "spec_sha256": sha(spec_path),
            "optional_coverage": [
                {
                    "task_id": t.id,
                    "system": s,
                    "expected_sessions": len(t.sessions),
                    "status": "qualified-selected" if s in factories else "unqualified",
                    "calls": 0,
                }
                for t in tasks
                for s in ("mem0", "cognee")
            ],
            "scope": (
                "all released 150 tasks / 900 ordered session occurrences per admitted method; "
                "local simulated WebShop only"
            ),
        },
    )


CONTINUATION_PROTOCOL = "memoryarena-shopping-never-run-continuation@v1"
NEVER_RUN = "never_run_candidate_only"
REUSE = "verified_complete_reuse_conditional"
CONTINUATION_DISPOSITIONS = {
    NEVER_RUN: 849,
    REUSE: 8,
    "harness_held_privacy_incompatible_guard_not_QA_zero": 8,
    "retained_context_nonadmission_NA_no_retry": 15,
    "ambiguous_HTTP400_hold_origin_body_unavailable": 16,
    "quality_failed_latched_no_retry": 3,
    "interrupted_poststop_hold": 1,
}


def continuation_plan(ledger, original):
    """Original ordered full matrix; conservative allowance is not actual spend."""
    from collections import Counter

    rows = ledger["rows"]
    bounds = original["bounds"]
    expected = [
        dict(
            task_id=t["id"],
            domain=DOMAIN,
            system=s,
            input_sha256=t["input_sha256"],
            expected_sessions=t["sessions"],
        )
        for t in original["tasks"]
        for s in original["systems"]
    ]
    if (
        original["protocol"] != PROTOCOL
        or len(original["tasks"]) != 150
        or len({t["id"] for t in original["tasks"]}) != 150
        or sum(t["sessions"] for t in original["tasks"]) != 900
        or original["systems"] != list(SYSTEMS)
        or len(rows) != 900
        or Counter(r["disposition"] for r in rows) != CONTINUATION_DISPOSITIONS
        or bounds
        != dict(
            calls_per_task_method=512,
            seconds_per_task_method=21600,
            campaign_seconds=19443600,
            max_rounds=25,
            feedback_in_memory=False,
            include_history=False,
        )
    ):
        raise ValueError("original full Shopping scope/bounds/dispositions required")
    dispatch, retained = [], []
    for i, (row, identity) in enumerate(zip(rows, expected, strict=True)):
        cp, spend = row["checkpoint_row"], row["spend"]
        calls = spend["calls_lower_bound"]
        if (
            row["matrix_index"] != i
            or any(
                row["original_frozen_row"].get(k) != v or cp.get(k) != v
                for k, v in identity.items()
            )
            or type(calls) is not int
            or not 0 <= calls <= 512
            or type(cp["calls"]) is not int
            or not 0 <= cp["calls"] <= calls
            or row["checkpoint_additional_receipted_calls"] != calls - cp["calls"]
        ):
            raise ValueError("ledger original identity/order/spend differs")
        if row["disposition"] == NEVER_RUN:
            if (
                cp != row["original_frozen_row"]
                or cp["status"] != "unrun"
                or calls
                or cp["completed_sessions"]
                or row["failure"] is not None
                or row["retained_leaf_path"] is not None
            ):
                raise ValueError("attempted/held rows cannot become never-run")
            dispatch.append(i)
        else:
            if cp["status"] == "unrun":
                raise ValueError("never-run row cannot acquire prior disposition")
            if row["disposition"] == REUSE and (
                cp["status"] != "complete"
                or row["failure"] is not None
                or cp["system"] != "no_memory"
                or cp["task_id"] not in range(8)
            ):
                raise ValueError("only original closed no-memory completes eligible for reuse")
            retained.append(i)
        for key in (
            "known_usage_calls",
            "unknown_usage_calls_lower_bound",
            "prompt_tokens",
            "completion_tokens",
        ):
            if type(spend.get(key, 0)) is not int or spend.get(key, 0) < 0:
                raise ValueError("invalid observed prior cost")
        if (
            spend.get("known_usage_calls", 0) + spend.get("unknown_usage_calls_lower_bound", 0)
            != calls
        ):
            raise ValueError("prior usage/calls differ")
    keys = (
        "calls_lower_bound",
        "known_usage_calls",
        "unknown_usage_calls_lower_bound",
        "prompt_tokens",
        "completion_tokens",
    )
    if any(ledger["spend"][k] != sum(r["spend"].get(k, 0) for r in rows) for k in keys):
        raise ValueError("observed ledger cost totals differ")
    return dict(
        dispatch_indices=dispatch,
        retained_indices=retained,
        allowance_debit=dict(calls=len(retained) * 512, seconds=len(retained) * 21600 + 3600),
        remaining_allowance=dict(
            calls=len(dispatch) * 512, seconds=19443600 - len(retained) * 21600 - 3600
        ),
        observed_prior={k: ledger["spend"][k] for k in keys},
        unknown_prior_attempts=None,
        measured_prior_elapsed_seconds=None,
    )


def _shopping_record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def _shopping_bound(record):
    raw = Path(record["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("Shopping continuation byte binding differs: " + Path(record["path"]).name)
    return raw


def _shopping_original(ledger, name):
    record = next(r for r in ledger["snapshot_bindings"] if Path(r["path"]).name == name)
    return json.loads(_shopping_bound(record))


def _shopping_old_source(original, name):
    import zipfile

    index, record = next(
        (i, r) for i, r in enumerate(original["files"]) if Path(r["path"]).name == name
    )
    with zipfile.ZipFile(Path(original["output_root"]) / "frozen-source.zip") as archive:
        raw = archive.read(f"producer/{index}/" + name)
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("old Shopping producer archive member differs")
    return raw


def _shopping_bridge(original):
    import ast

    def definitions(raw):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(raw).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }

    before, after = (
        definitions(_shopping_old_source(original, "arena_shopping_driver.py")),
        definitions(Path(__file__).read_bytes()),
    )
    exact = (
        "ShoppingTransport",
        "native_actor",
        "native_metrics",
        "method_metrics",
        "replay_evaluator_reward",
        "verify_reward_coverage",
        "verify_public_experience",
        "verify_task",
    )

    class NormalPath(ast.NodeTransformer):
        def visit_Try(self, node):
            node.handlers, node.finalbody = [], []
            return self.generic_visit(node)

    def normal(raw):
        node = next(n for n in ast.parse(raw).body if getattr(n, "name", None) == "run_task")
        return ast.dump(NormalPath().visit(node), include_attributes=False)

    if normal(_shopping_old_source(original, "arena_shopping_driver.py")) != normal(
        Path(__file__).read_bytes()
    ):
        raise ValueError("Shopping normal actor/input/scoring execution changed")

    def runtime_projection(raw):
        node = next(n for n in ast.parse(raw).body if getattr(n, "name", None) == "ShoppingRuntime")
        node.body = [
            n for n in node.body if getattr(n, "name", None) not in {"__exit__", "_cleanup_timeout"}
        ]

        class CleanupTimeout(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name == "__init__":
                    node.body = [n for n in node.body if ast.unparse(n) != "self.budget.check()"]
                return self.generic_visit(node)

            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "_cleanup_timeout":
                    return ast.Constant(value=30)
                return self.generic_visit(node)

        return ast.dump(CleanupTimeout().visit(node), include_attributes=False)

    if runtime_projection(
        _shopping_old_source(original, "arena_shopping_driver.py")
    ) != runtime_projection(Path(__file__).read_bytes()):
        raise ValueError("Shopping native runtime non-cleanup execution changed")

    def wiring(raw):
        node = next(
            n for n in ast.parse(raw).body if getattr(n, "name", None) == "_execute_approved"
        )
        node.args.kwonlyargs, node.args.kw_defaults = [], []

        class Wiring(NormalPath):
            def visit_Assign(self, value):
                if any(isinstance(t, ast.Name) and t.id == "task_permit" for t in value.targets):
                    return None
                return self.generic_visit(value)

            def visit_Name(self, value):
                if value.id == "task_permit":
                    value.id = "permit"
                return value

            def visit_Call(self, value):
                if (
                    isinstance(value.func, ast.BoolOp)
                    and isinstance(value.func.op, ast.Or)
                    and isinstance(value.func.values[0], ast.Name)
                    and value.func.values[0].id == "_matrix_executor"
                ):
                    value.func = ast.Name(id="execute_matrix", ctx=ast.Load())
                return self.generic_visit(value)

        return ast.dump(Wiring().visit(node), include_attributes=False)

    old_source, new_source = (
        _shopping_old_source(original, "arena_shopping_driver.py"),
        Path(__file__).read_bytes(),
    )
    if wiring(old_source) != wiring(new_source):
        raise ValueError("Shopping native actor/production/memory runner wiring changed")
    if any(before[n] != after[n] for n in exact):
        raise ValueError("Shopping actor/input/scorer normal execution semantics changed")
    memory_classes = (
        "RawExperienceMemory",
        "GluoExperienceMemory",
        "ArenaProductionRouter",
        "BudgetedArenaEmbedder",
    )
    old_memory = definitions(_shopping_old_source(original, "arena_driver.py"))
    current_memory = definitions(Path(common.__file__).read_bytes())
    if any(old_memory[n] != current_memory[n] for n in memory_classes):
        raise ValueError("Shopping memory/encoder/production algorithm changed")
    from . import rack

    old_rack, new_rack = (
        definitions(_shopping_old_source(original, "rack.py")),
        definitions(Path(rack.__file__).read_bytes()),
    )
    if any(old_rack[n] != new_rack[n] for n in ("RackProfile", "GenerationBudgets")):
        raise ValueError("old/new native profile semantics differ")
    return dict(
        exact_definitions=list(exact),
        unchanged_memory_classes=list(memory_classes),
        privacy_guard_changed_requires_review=before["verify_memory_store"]
        != after["verify_memory_store"],
        old_privacy_guard_ast_sha256=hashlib.sha256(
            before["verify_memory_store"].encode()
        ).hexdigest(),
        new_privacy_guard_ast_sha256=hashlib.sha256(
            after["verify_memory_store"].encode()
        ).hexdigest(),
        old_producer_inventory=original["files"],
        normal_task_ast_sha256=hashlib.sha256(normal(new_source).encode()).hexdigest(),
        native_runner_wiring_ast_sha256=hashlib.sha256(wiring(new_source).encode()).hexdigest(),
        runtime_noncleanup_ast_sha256=hashlib.sha256(
            runtime_projection(new_source).encode()
        ).hexdigest(),
        changed_cleanup_and_guard_definitions={
            name: {
                "old_ast_sha256": hashlib.sha256(before[name].encode()).hexdigest(),
                "new_ast_sha256": hashlib.sha256(after[name].encode()).hexdigest(),
            }
            for name in ("run_task", "ShoppingRuntime", "_execute_approved")
        },
        guard_nonbypass_and_cleanup_require_independent_review=True,
    )


def build_continuation_spec(*, runtime, ledger_path, ledger_sha256, reserve_proposal_path):
    """Freeze NEVER-RUN protocol; never authors approval or launches.

    Supply a fresh normal build_spec with original ordered inputs/profile/bounds.
    Obtain separate parent reserve and final execution approval bytes. The proposal
    remains execution_approved=false: it is not itself execution authority.
    """
    from copy import deepcopy

    record = _shopping_record(ledger_path)
    if record["sha256"] != ledger_sha256:
        raise ValueError("independently supplied ledger SHA256 differs")
    ledger = json.loads(_shopping_bound(record))
    original = _shopping_original(ledger, "execution-spec.json")
    if sha(Path(ledger["original_root"]) / "execution-spec.json") != ledger["spec_sha256"]:
        raise ValueError("original spec ledger binding differs")
    plan = continuation_plan(ledger, original)
    if [r["original_frozen_row"] for r in ledger["rows"]] != _shopping_original(
        ledger, "frozen-matrix.json"
    ) or [r["checkpoint_row"] for r in ledger["rows"]] != _shopping_original(
        ledger, "task-statuses.json"
    ):
        raise ValueError("ledger differs from original ordered checkpoint bytes")
    semantic = (
        "protocol",
        "tasks",
        "occurrences",
        "systems",
        "data_pins",
        "profile",
        "bounds",
        "embedding_url",
        "production_memory_qualification",
        "native_document_factories",
    )
    if any(runtime[k] != original[k] for k in semantic) or runtime["native_document_factories"]:
        raise ValueError("original Shopping scope/native inputs/profile/scorer parity required")

    def native_projection(manifest):
        return dict(
            native_commit=manifest["native_commit"],
            worker_sha256=manifest["worker_sha256"],
            files=[(r["source"], r["sha256"]) for r in manifest["files"]],
            assets=[(r["bytes"], r["sha256"]) for r in manifest["assets"]],
        )

    if native_projection(runtime["runtime"]) != native_projection(original["runtime"]):
        raise ValueError("Shopping native runtime actor/tool/scorer/asset semantics differ")
    old, new = Path(ledger["original_root"]).resolve(), Path(runtime["output_root"]).resolve()
    if (
        new == old
        or new.is_relative_to(old)
        or old.is_relative_to(new)
        or new.is_relative_to(Path(__file__).resolve().parents[3])
    ):
        raise ValueError("disjoint external fresh Shopping root required")
    proposal_record = _shopping_record(reserve_proposal_path)
    proposal = json.loads(_shopping_bound(proposal_record))
    p = proposal["proposal"]
    if (
        proposal["prior_ledger_sha256"] != ledger_sha256
        or proposal["original_spec_sha256"] != ledger["spec_sha256"]
        or p
        != dict(
            retained_attempted_or_held_rows=51,
            never_run_rows_only=849,
            allowance_calls_debit_total_including_observed=26112,
            allowance_time_debit_seconds=1105200,
            remaining_call_allowance=434688,
            remaining_time_allowance_seconds=18338400,
        )
        or proposal["observed_prior_actual"]
        != dict(
            generation_attempts_lower_bound=plan["observed_prior"]["calls_lower_bound"],
            known_prompt_tokens=plan["observed_prior"]["prompt_tokens"],
            known_completion_tokens=plan["observed_prior"]["completion_tokens"],
            exact_elapsed_seconds=None,
            unpersisted_inflight_attempts_excluded=False,
        )
    ):
        raise ValueError("conservative reserve proposal differs; reservation is not actual spend")
    return dict(
        protocol=CONTINUATION_PROTOCOL,
        runtime=deepcopy(runtime),
        ledger=record,
        reserve_proposal=proposal_record,
        origin=deepcopy(ledger["snapshot_bindings"]),
        plan=plan,
        semantic_bridge=_shopping_bridge(original),
    )


def _shopping_deadline(budget):
    if budget is not None and budget.clock() >= budget.deadline:
        from .parallel_campaign import CampaignStopped

        raise CampaignStopped("Shopping continuation deadline exhausted during finalization")


def verify_continuation_origin(spec, *, budget=None):
    """Actual archived native verifier for eight old no-memory leaves; no DB opens."""
    import ast

    ledger = json.loads(_shopping_bound(spec["ledger"]))
    original = _shopping_original(ledger, "execution-spec.json")
    for rec in ledger["snapshot_bindings"]:
        _shopping_deadline(budget)
        _shopping_bound(rec)
        _shopping_bound(dict(path=rec["snapshot"], sha256=rec["sha256"]))
    records, databases = ledger["original_non_database_artifacts"], ledger["database_stat_only"]
    actual = {p.resolve() for p in Path(ledger["original_root"]).rglob("*") if p.is_file()}
    if actual != {Path(r["path"]).resolve() for r in [*records, *databases]}:
        raise ValueError("original Shopping exact non-DB/DB membership changed")
    for rec in databases:
        stat = Path(rec["path"]).stat()
        if stat.st_size != rec["bytes"] or stat.st_mtime_ns != rec["mtime_ns"]:
            raise ValueError("original Shopping DB stat changed")
    for rec in records:
        _shopping_deadline(budget)
        _shopping_bound(rec)
    for rec in ledger["source_parity"]:
        _shopping_bound(rec)
    names = {
        "sha",
        "verify_task",
        "replay_evaluator_reward",
        "verify_reward_coverage",
        "verify_public_experience",
        "native_metrics",
    }
    tree = ast.parse(_shopping_old_source(original, "arena_shopping_driver.py"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in names
        or isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in {"PROTOCOL", "REWARD_SOURCES", "REWARD_METRIC"}
            for t in n.targets
        )
    ]
    scope = dict(globals())

    def no_database(*args, **kwargs):
        raise ValueError("old Shopping reuse must never open a database")

    scope["verify_memory_store"] = no_database
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "<pinned-old-shopping-verifier>", "exec"),
        scope,
    )
    tasks = sessions = 0
    from .trajectory_suite import ArenaSession, ArenaTask

    seen_responses = set()
    for row in ledger["rows"]:
        cp = row["checkpoint_row"]
        target = Path(ledger["original_root"]) / str(cp["task_id"]) / cp["system"]
        generations, known, prompt_tokens, completion_tokens = 0, 0, 0, 0
        for name in ("http.jsonl", "production-raw.jsonl"):
            path = target / name
            if not path.exists():
                continue
            for line in path.read_bytes().splitlines():
                wire = json.loads(line)
                if name == "http.jsonl" and not wire.get("url", "").endswith("/chat/completions"):
                    continue
                if name == "production-raw.jsonl" and "ordinal" not in wire:
                    continue
                response = wire.get("response") or {}
                rid = response.get("id")
                if rid and rid in seen_responses:
                    continue
                if rid:
                    seen_responses.add(rid)
                generations += 1
                usage = response.get("usage") or {}
                if all(
                    type(usage.get(k)) is int and usage[k] >= 0
                    for k in ("prompt_tokens", "completion_tokens")
                ):
                    known += 1
                    prompt_tokens += usage["prompt_tokens"]
                    completion_tokens += usage["completion_tokens"]
        calls = max(cp["calls"], (row["failure"] or {}).get("calls", 0), generations)
        costs = dict(
            calls_lower_bound=calls,
            known_usage_calls=known,
            unknown_usage_calls_lower_bound=calls - known,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            receipted_generations=generations,
        )
        if any(row["spend"].get(k, 0) != v for k, v in costs.items()):
            raise ValueError(
                "original Shopping spend differs from actual byte-bound native/production receipts"
            )
        if row["disposition"] != REUSE:
            continue
        _shopping_deadline(budget)
        cp = row["checkpoint_row"]
        target = Path(ledger["original_root"]) / str(cp["task_id"]) / cp["system"]
        if (
            target.resolve() != Path(row["retained_leaf_path"]).resolve()
            or sha(target / "result.json") != cp["result_sha256"]
        ):
            raise ValueError("old complete reuse digest/path differs")
        value = scope["verify_task"](target)
        raw_task = json.loads((target / "input-task.json").read_bytes())
        task = ArenaTask(
            raw_task["id"],
            DOMAIN,
            tuple(ArenaSession(q) for q in raw_task["questions"]),
            tuple(raw_task["answers"]),
        )
        if (
            common.task_digest(task) != cp["input_sha256"]
            or json.loads((target / "effective-profile.json").read_bytes()) != original["profile"]
            or value["calls"] != row["spend"]["calls_lower_bound"]
            or any(
                value[k] != cp[k]
                for k in (
                    "metrics",
                    "supplemental_metrics",
                    "completed_sessions",
                    "expected_sessions",
                )
            )
        ):
            raise ValueError("old native reuse input/profile/score/spend parity differs")
        tasks += 1
        sessions += value["completed_sessions"]
    return dict(reused_tasks=tasks, reused_sessions=sessions, database_opens=0)


def verify_continuation_approval(spec_path, approval_path, reserve_approval_path):
    spec = json.loads(Path(spec_path).read_bytes())
    approval = json.loads(Path(approval_path).read_bytes())
    reserve = json.loads(Path(reserve_approval_path).read_bytes())
    if (
        spec.get("protocol") != CONTINUATION_PROTOCOL
        or any(
            a.get("approved") is not True
            or a.get("reviewer_kind") not in {"parent", "human"}
            or a.get("spec_sha256") != sha(spec_path)
            for a in (approval, reserve)
        )
        or Path(approval_path).resolve() == Path(reserve_approval_path).resolve()
        or approval.get("reserve_approval_sha256") != sha(reserve_approval_path)
        or reserve.get("ledger_sha256") != spec["ledger"]["sha256"]
        or reserve.get("reserve_proposal_sha256") != spec["reserve_proposal"]["sha256"]
        or reserve.get("allowance_debit") != spec["plan"]["allowance_debit"]
        or reserve.get("semantic_bridge_sha256")
        != hashlib.sha256(native._json(spec["semantic_bridge"])).hexdigest()
        or any(
            reserve.get(k) is not True
            for k in (
                "original_guards_verified",
                "no_transport_bypass",
                "unknown_attempts_bounded_by_original_guards",
                "semantic_bridge_verified",
                "privacy_guard_repair_reviewed",
            )
        )
    ):
        raise ValueError("separate exact-byte parent reserve/semantic/execution approval required")
    if not reserve.get("independent_review") or not reserve.get("guard_proof"):
        raise ValueError("independent exact-byte guard/nonbypass review required")
    review = json.loads(_shopping_bound(reserve["independent_review"]))
    proof = json.loads(_shopping_bound(reserve["guard_proof"]))
    ledger = json.loads(_shopping_bound(spec["ledger"]))
    if (
        review.get("approved") is not True
        or review.get("reviewer_kind") != "independent"
        or review.get("blockers") != []
        or any(
            review.get(k) != reserve[k]
            for k in (
                "spec_sha256",
                "ledger_sha256",
                "reserve_proposal_sha256",
                "semantic_bridge_sha256",
            )
        )
        or review.get("guard_proof_sha256") != reserve["guard_proof"]["sha256"]
        or proof.get("original_spec_sha256") != ledger["spec_sha256"]
        or proof.get("original_source_archive_sha256") != ledger["source_archive_sha256"]
        or proof.get("original_bounds") != spec["runtime"]["bounds"]
        or any(
            proof.get(k) is not True
            for k in (
                "original_guards_verified",
                "no_transport_bypass",
                "unknown_attempts_bounded_by_original_guards",
            )
        )
    ):
        raise ValueError("independent original-guard/nonbypass review differs")
    return spec


def verify_continuation_spec(
    spec_path, approval_path, reserve_approval_path, *, verify_runtime=True, budget=None
):
    """Full exact-bound review/origin verification. Offline mode never grants dispatch."""
    from .rack import RackProfile

    _shopping_deadline(budget)
    spec = verify_continuation_approval(spec_path, approval_path, reserve_approval_path)
    rebuilt = build_continuation_spec(
        runtime=spec["runtime"],
        ledger_path=spec["ledger"]["path"],
        ledger_sha256=spec["ledger"]["sha256"],
        reserve_proposal_path=spec["reserve_proposal"]["path"],
    )
    if rebuilt != spec:
        raise ValueError("Shopping continuation exact plan/semantic bridge rebuild differs")
    verify_continuation_origin(spec, budget=budget)
    if verify_runtime:
        runtime = spec["runtime"]
        bounds = runtime["bounds"]
        fresh = build_spec(
            data_root=runtime["data_root"],
            pins=runtime["data_pins"],
            runtime=runtime["runtime_path"],
            profile=RackProfile.model_validate(runtime["profile"]),
            output_root=runtime["output_root"],
            embedding_url=runtime["embedding_url"],
            calls_per_task=bounds["calls_per_task_method"],
            seconds_per_task=bounds["seconds_per_task_method"],
            campaign_seconds=bounds["campaign_seconds"],
            max_rounds=bounds["max_rounds"],
            production_memory_qualification=runtime["production_memory_qualification"],
            native_document_factories=runtime["native_document_factories"],
        )
        if fresh != runtime or runtime["runtime"]["native_commit"] != native.PIN:
            raise ValueError("Shopping continuation full runtime producer/input inventory differs")
        for rec in [
            *runtime["runtime"]["files"],
            *runtime["runtime"]["assets"],
            dict(path=runtime["runtime"]["worker"], sha256=runtime["runtime"]["worker_sha256"]),
        ]:
            _shopping_deadline(budget)
            _shopping_bound(rec)
    _shopping_deadline(budget)
    return spec


class ShoppingContinuationPermit:
    """Task AND method scoped authority; never widens legacy Cartesian/pilot gates."""

    def __init__(self, spec_path, approval_path, reserve_approval_path, *, system):
        self.paths = tuple(map(Path, (spec_path, approval_path, reserve_approval_path)))
        self.expected = tuple(sha(p) for p in self.paths)
        self.spec = verify_continuation_approval(*self.paths)
        reserve = json.loads(self.paths[2].read_bytes())
        self.review_records = [reserve["independent_review"], reserve["guard_proof"]]
        self.system = system

    def check(self):
        if tuple(sha(p) for p in self.paths) != self.expected:
            from .rack import SharedRackError

            raise SharedRackError("Shopping continuation authority bytes changed")
        try:
            for record in self.review_records:
                _shopping_bound(record)
        except (OSError, ValueError) as exc:
            from .rack import SharedRackError

            raise SharedRackError("Shopping continuation reviewed authority bytes changed") from exc

    def validate(self, task, profile, max_rounds):
        self.check()
        runtime = self.spec["runtime"]
        identity = dict(
            id=task.id, sessions=len(task.sessions), input_sha256=common.task_digest(task)
        )
        indices = self.spec["plan"]["dispatch_indices"]
        task_index = next((i for i, row in enumerate(runtime["tasks"]) if row == identity), None)
        if (
            task.domain != DOMAIN
            or runtime["profile"] != profile.model_dump(mode="json")
            or max_rounds != runtime["bounds"]["max_rounds"]
            or task_index is None
            or self.system not in runtime["systems"]
            or task_index * len(runtime["systems"]) + runtime["systems"].index(self.system)
            not in indices
        ):
            raise ValueError("typed Shopping never-run task/method/profile permit required")
        return runtime


def _shopping_continuation_rows(ledger):
    from copy import deepcopy

    matrix = []
    for prior in ledger["rows"]:
        row = deepcopy(prior["checkpoint_row"])
        row.update(
            disposition=prior["disposition"],
            provenance="continuation" if prior["disposition"] == NEVER_RUN else "original",
        )
        row["calls"] = prior["spend"]["calls_lower_bound"]
        if prior["disposition"] not in {NEVER_RUN, REUSE}:
            row["status"] = (
                "latched" if prior["disposition"] == "quality_failed_latched_no_retry" else "held"
            )
        if row["status"] != "complete":
            row.update(metrics=None, supplemental_metrics=None)
            row.pop("result_sha256", None)
        matrix.append(row)
    return matrix


def _shopping_accounting(spec, calls, elapsed):
    plan = spec["plan"]
    return dict(
        observed_prior=plan["observed_prior"],
        new_actual_calls=calls,
        observed_total_calls_lower_bound=plan["observed_prior"]["calls_lower_bound"] + calls,
        reserved_prior_allowance=plan["allowance_debit"],
        allowance_consumed_calls=plan["allowance_debit"]["calls"] + calls,
        new_elapsed_seconds=elapsed,
        allowance_consumed_seconds=plan["allowance_debit"]["seconds"] + elapsed,
        unknown_prior_attempts=None,
        measured_prior_elapsed_seconds=None,
        currency_cost=None,
    )


def _execute_continuation_matrix(
    spec, tasks, *, budget, run_one, initialize, verify_saved=None, permit_factory=None
):
    """Private authored-test seam. Public execution fixes native actor and verifier."""
    from copy import deepcopy

    from .parallel_campaign import Job, JobContext, RequestLimiter

    remaining = spec["plan"]["remaining_allowance"]
    if (
        budget.max_calls != remaining["calls"]
        or budget.max_seconds != remaining["seconds"]
        or budget.prior_calls
        or budget.prior_elapsed_seconds
    ):
        raise ValueError(
            "one reduced continuation guard required; do not double-debit actual prior calls"
        )
    ledger = json.loads(_shopping_bound(spec["ledger"]))
    matrix = _shopping_continuation_rows(ledger)
    frozen = deepcopy(matrix)
    out = Path(spec["runtime"]["output_root"])
    out.mkdir(parents=True, exist_ok=False)
    pending, final = out / "campaign-result.pending.json", out / "campaign-result.json"
    limiter = RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072)

    def save_status():
        (out / "task-statuses.json").write_bytes(native._json(matrix))

    def receipt():
        return dict(
            protocol=CONTINUATION_PROTOCOL,
            spec_sha256=sha(out / "continuation-spec.json")
            if (out / "continuation-spec.json").exists()
            else None,
            tasks=deepcopy(matrix),
            quality_complete=all(r["status"] == "complete" for r in matrix),
            calls=budget.calls,
            accounting=_shopping_accounting(spec, budget.calls, budget.clock() - budget.started),
            expected_sessions=sum(r["expected_sessions"] for r in matrix),
            completed_sessions=sum(
                r["completed_sessions"] for r in matrix if r["status"] == "complete"
            ),
        )

    try:
        (out / "frozen-matrix.json").write_bytes(native._json(frozen))
        save_status()
        budget.check()
        initialize(out)
        budget.check()
        actual_tasks = list(tasks() if callable(tasks) else tasks)
        if [
            dict(id=t.id, sessions=len(t.sessions), input_sha256=common.task_digest(t))
            for t in actual_tasks
        ] != spec["runtime"]["tasks"]:
            raise ValueError("actual scoped native ordered input matrix differs")
        task_map = {t.id: t for t in actual_tasks}
        for index in spec["plan"]["dispatch_indices"]:
            row = matrix[index]
            context = None
            target = out / str(row["task_id"]) / row["system"]
            # Failure latches are never reset. Whole-task time AND calls admission.
            if (
                budget.failed
                or budget.clock() + spec["runtime"]["bounds"]["seconds_per_task_method"]
                > budget.deadline
            ):
                break
            job = Job(
                f"shopping:{row['task_id']}:{row['system']}",
                f"shopping:{row['task_id']}",
                target,
                spec["runtime"]["bounds"]["calls_per_task_method"],
                spec["runtime"]["bounds"]["seconds_per_task_method"],
            )
            if not budget.reserve(job):
                break
            try:
                context = JobContext(job, budget, limiter)
                if permit_factory is not None:
                    permit = permit_factory(row["system"])
                    original_check = context.check

                    def checked(check=original_check, authority=permit):
                        authority.check()
                        check()

                    context.check = checked
                context.check()
                row.update(status="running", output=str(target))
                save_status()
                value = run_one(task_map[row["task_id"]], row["system"], context, target)
                verified = verify_task(target)
                context.check()
                if (
                    verified != value
                    or value["status"] != "complete"
                    or value["calls"] != context.calls
                    or value["completed_sessions"] != row["expected_sessions"]
                ):
                    raise ValueError("continuation native leaf evidence/spend differs")
                row.update(
                    status="complete",
                    completed_sessions=value["completed_sessions"],
                    metrics=value["metrics"],
                    supplemental_metrics=value["supplemental_metrics"],
                    result_sha256=sha(target / "result.json"),
                )
            except (Exception, native.NativeAbort) as exc:
                row.update(
                    status="failed",
                    error=safe_error(exc),
                    metrics=None,
                    supplemental_metrics=None,
                    cleanup_diagnostics=safe_notes(exc),
                )
                row.pop("result_sha256", None)
                failure = target / "failure.json"
                if failure.exists():
                    row["completed_sessions"] = json.loads(failure.read_bytes())[
                        "completed_sessions"
                    ]
                # Only evidence-based explicit shared failure cancels unrelated jobs.
                if getattr(exc, "shared_failure", False):
                    budget.failed = True
            finally:
                if context is not None:
                    row["calls"] = context.calls
                    budget.release(context)
                save_status()
        _shopping_deadline(budget)
        result = receipt()
        result["bindings"] = {
            n: sha(out / n)
            for n in (
                "continuation-spec.json",
                "execution-spec.json",
                "execution-approval.json",
                "reserve-approval.json",
                "frozen-matrix.json",
                "frozen-source.zip",
            )
            if (out / n).exists()
        }
        source = Path(spec["runtime"]["runtime"]["source_root"]) / "run_shopping.py"
        result["method_metrics"] = method_metrics(matrix, source.read_bytes())
        if sum(matrix[i]["calls"] for i in spec["plan"]["dispatch_indices"]) != budget.calls:
            raise ValueError("continuation actual spend reconciliation differs")
        pending.write_bytes(native._json(result))
        _shopping_deadline(budget)
        saved = json.loads(pending.read_bytes())
        if saved != result:
            raise ValueError("staged Shopping root readback differs")
        (verify_saved or verify_continuation_root)(
            out, payload=saved, budget=budget, expected_rows=deepcopy(matrix)
        )
        _shopping_deadline(budget)
        pending.replace(final)
        _shopping_deadline(budget)
        if json.loads(final.read_bytes()) != result:
            raise ValueError("final Shopping root readback differs")
        _shopping_deadline(budget)
        return result
    except BaseException as exc:
        budget.failed = True
        for path in (pending, final):
            try:
                path.unlink(missing_ok=True)
            except BaseException as cleanup:
                exc.add_note(
                    'Shopping authority revocation: ' + safe_error(cleanup)
                )
                try:
                    if path.exists():
                        path.replace(out / ("revoked-" + path.name))
                except BaseException as rename_error:
                    exc.add_note("Shopping revoked-artifact move: " + str(rename_error))
        try:
            failure = receipt()
        except BaseException as accounting_error:
            exc.add_note("Shopping failure readback: " + str(accounting_error))
            failure = dict(
                protocol=CONTINUATION_PROTOCOL,
                tasks=deepcopy(matrix),
                calls=budget.calls,
                accounting=_shopping_accounting(
                    spec, budget.calls, budget.clock() - budget.started
                ),
            )
        failure.update(
            quality_complete=False,
            method_metrics=None,
            error=safe_error(exc),
            cleanup_diagnostics=safe_notes(exc),
        )
        exc.receipt = failure
        try:
            (out / "campaign-failure.json").write_bytes(native._json(failure))
        except BaseException as persistence_error:
            exc.add_note(
                'Shopping failure receipt unavailable; retained on exception.receipt: '
                + safe_error(persistence_error)
            )
        raise


def _shopping_archive(out, runtime, *, budget=None):
    import zipfile

    with zipfile.ZipFile(Path(out) / "frozen-source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for i, record in enumerate(runtime["files"]):
            _shopping_deadline(budget)
            archive.writestr(f"producer/{i}/" + Path(record["path"]).name, _shopping_bound(record))
        archive.writestr(
            "runtime/runtime.json",
            _shopping_bound(dict(path=runtime["runtime_path"], sha256=runtime["runtime_sha256"])),
        )
        archive.writestr(
            "runtime/shopping_native_worker.py",
            _shopping_bound(
                dict(path=runtime["runtime"]["worker"], sha256=runtime["runtime"]["worker_sha256"])
            ),
        )
        for record in runtime["runtime"]["files"]:
            _shopping_deadline(budget)
            archive.writestr("runtime/source/" + record["source"], _shopping_bound(record))


def verify_continuation_root(out, *, payload=None, budget=None, expected_rows=None):
    """Full mixed-provenance root authority; saved evidence only, never old DBs."""
    import math
    import zipfile

    from .trajectory_suite import ArenaSession, ArenaTask

    out = Path(out)
    if (out / "campaign-failure.json").exists():
        raise ValueError("Shopping continuation failed publication latch revokes root authority")
    _shopping_deadline(budget)
    spec_path, approval, reserve = (
        out / n
        for n in ("continuation-spec.json", "execution-approval.json", "reserve-approval.json")
    )
    spec = verify_continuation_spec(
        spec_path, approval, reserve, verify_runtime=False, budget=budget
    )
    if Path(spec["runtime"]["output_root"]).resolve() != out.resolve():
        raise ValueError("continuation root output identity differs")
    result = json.loads(
        (
            out
            / ("campaign-result.pending.json" if payload is not None else "campaign-result.json")
        ).read_bytes()
    )
    if payload is not None and result != payload:
        raise ValueError("staged continuation root exact payload differs")
    ledger = json.loads(_shopping_bound(spec["ledger"]))
    frozen = _shopping_continuation_rows(ledger)
    rows = json.loads((out / "task-statuses.json").read_bytes())
    if (
        json.loads((out / "frozen-matrix.json").read_bytes()) != frozen
        or result.get("protocol") != CONTINUATION_PROTOCOL
        or result.get("spec_sha256") != sha(spec_path)
        or result.get("tasks") != rows
        or len(rows) != len(frozen)
        or expected_rows is not None
        and rows != expected_rows
    ):
        raise ValueError("continuation full ordered matrix/status/root differs")
    if json.loads((out / "execution-spec.json").read_bytes()) != spec["runtime"]:
        raise ValueError("continuation saved runtime differs")
    required = (
        "continuation-spec.json",
        "execution-spec.json",
        "execution-approval.json",
        "reserve-approval.json",
        "frozen-matrix.json",
        "frozen-source.zip",
    )
    if result.get("bindings") != {n: sha(out / n) for n in required}:
        raise ValueError("continuation root source/authority bindings differ")
    runtime = spec["runtime"]
    with zipfile.ZipFile(out / "frozen-source.zip") as archive:
        members = {
            f"producer/{i}/" + Path(r["path"]).name: r["sha256"]
            for i, r in enumerate(runtime["files"])
        }
        members.update(
            {"runtime/source/" + r["source"]: r["sha256"] for r in runtime["runtime"]["files"]}
        )
        members.update(
            {
                "runtime/runtime.json": runtime["runtime_sha256"],
                "runtime/shopping_native_worker.py": runtime["runtime"]["worker_sha256"],
            }
        )
        if len(archive.namelist()) != len(members) or set(archive.namelist()) != set(members):
            raise ValueError("continuation frozen source membership differs")
        for name, pin in members.items():
            _shopping_deadline(budget)
            if hashlib.sha256(archive.read(name)).hexdigest() != pin:
                raise ValueError("continuation frozen producer archive bytes differ")
        source = archive.read("runtime/source/run_shopping.py")
    dispatch = set(spec["plan"]["dispatch_indices"])
    identity = (
        "task_id",
        "system",
        "domain",
        "input_sha256",
        "expected_sessions",
        "disposition",
        "provenance",
    )
    for i, (prior, row) in enumerate(zip(frozen, rows, strict=True)):
        _shopping_deadline(budget)
        if i not in dispatch:
            if row != prior:
                raise ValueError("prior held/failed/reuse disposition or digest promoted/changed")
            continue  # Actual old verifier already replayed by verify_continuation_spec.
        if (
            any(row.get(k) != prior[k] for k in identity)
            or row.get("status") not in {"unrun", "failed", "complete"}
            or type(row["calls"]) is not int
            or not 0 <= row["calls"] <= 512
            or type(row["completed_sessions"]) is not int
            or not 0 <= row["completed_sessions"] <= row["expected_sessions"]
        ):
            raise ValueError("continuation row identity/status/spend differs")
        target = out / str(row["task_id"]) / row["system"]
        if row["status"] == "unrun":
            if row["calls"] or row["completed_sessions"] or target.exists():
                raise ValueError("unrun continuation has spend/artifacts")
        elif Path(row.get("output", "")).resolve() != target.resolve():
            raise ValueError("continuation leaf path differs")
        if row["status"] != "complete":
            if (
                row.get("metrics") is not None
                or row.get("supplemental_metrics") is not None
                or "result_sha256" in row
            ):
                raise ValueError("incomplete continuation cannot carry scores")
            failure = target / "failure.json"
            if failure.exists():
                saved = json.loads(failure.read_bytes())
                if any(
                    saved[k] != row[k] for k in ("calls", "completed_sessions", "expected_sessions")
                ):
                    raise ValueError("continuation failure accounting differs")
            continue
        leaf = verify_task(target)
        raw = json.loads((target / "input-task.json").read_bytes())
        task = ArenaTask(
            raw["id"],
            DOMAIN,
            tuple(ArenaSession(q) for q in raw["questions"]),
            tuple(raw["answers"]),
        )
        if (
            common.task_digest(task) != row["input_sha256"]
            or raw["id"] != row["task_id"]
            or json.loads((target / "effective-profile.json").read_bytes()) != runtime["profile"]
            or row["result_sha256"] != sha(target / "result.json")
            or row["completed_sessions"] != row["expected_sessions"]
            or any(
                leaf[k] != row[k]
                for k in (
                    "calls",
                    "metrics",
                    "supplemental_metrics",
                    "completed_sessions",
                    "expected_sessions",
                )
            )
        ):
            raise ValueError("continuation native leaf input/profile/digest/score/spend differs")
    calls = sum(rows[i]["calls"] for i in dispatch)
    elapsed = result["accounting"]["new_elapsed_seconds"]
    if (
        type(elapsed) not in (int, float)
        or not math.isfinite(elapsed)
        or not 0 <= elapsed < spec["plan"]["remaining_allowance"]["seconds"]
    ):
        raise ValueError("continuation elapsed accounting differs")
    expected = dict(
        calls=calls,
        accounting=_shopping_accounting(spec, calls, elapsed),
        quality_complete=all(r["status"] == "complete" for r in rows),
        expected_sessions=sum(r["expected_sessions"] for r in rows),
        completed_sessions=sum(r["completed_sessions"] for r in rows if r["status"] == "complete"),
        method_metrics=method_metrics(rows, source),
    )
    if (
        any(result.get(k) != v for k, v in expected.items())
        or budget is not None
        and budget.calls != calls
    ):
        raise ValueError(
            "continuation root official/supplemental metrics or actual/reserved accounting differs"
        )
    _shopping_deadline(budget)
    return result


def _execute_shopping_continuation(spec, spec_path, approval_path, reserve_approval_path):
    from .parallel_campaign import CampaignBudget
    from .rack import RackProfile

    remaining = spec["plan"]["remaining_allowance"]
    # This is the sole guard, started BEFORE setup/native/JVM/client creation.
    budget = CampaignBudget(max_calls=remaining["calls"], max_seconds=remaining["seconds"])
    runtime = spec["runtime"]
    paths = tuple(map(Path, (spec_path, approval_path, reserve_approval_path)))
    expected = tuple(sha(p) for p in paths)
    state = {}

    def permits(system):
        if tuple(sha(p) for p in paths) != expected:
            from .rack import SharedRackError

            raise SharedRackError("Shopping continuation authority changed during setup")
        return ShoppingContinuationPermit(*paths, system=system)

    def initialize(root):
        if verify_continuation_spec(*paths, budget=budget) != spec:
            raise ValueError("Shopping continuation setup spec changed")
        common.verify_memory_qualification(
            runtime["production_memory_qualification"],
            RackProfile.model_validate(runtime["profile"]),
        )
        for path, name in zip(
            paths,
            ("continuation-spec.json", "execution-approval.json", "reserve-approval.json"),
            strict=True,
        ):
            (root / name).write_bytes(path.read_bytes())
        (root / "execution-spec.json").write_bytes(native._json(runtime))
        _shopping_archive(root, runtime, budget=budget)
        budget.check()

        def capture(tasks, **kwargs):
            state.update(tasks=tasks, run_one=kwargs["run_one"])

        # Same unmodified actor, native inputs, production factories and leaf scorer.
        # Only typed permit and ordered dispatch-subset coordinator are substituted.
        _execute_approved(
            runtime, spec_path, approval_path, _matrix_executor=capture, _permit_factory=permits
        )
        budget.check()

    def run_one(task, system, context, target):
        return state["run_one"](task, system, context, target)

    return _execute_continuation_matrix(
        spec,
        lambda: state["tasks"],
        budget=budget,
        run_one=run_one,
        initialize=initialize,
        permit_factory=permits,
    )


def execute_continuation(spec_path, approval_path, reserve_approval_path):
    """Run only approved original NEVER-RUN tasks through native Shopping paths.

    Seal/review order (all JSON must be saved as explicit UTF-8 bytes):
      1. Freeze a fresh private runtime with build_spec, preserving original bounds,
         ordered inputs, six methods, actor/production/scorer and native asset pins.
      2. build_continuation_spec binds ledger, original checkpoint/source/archive,
         conservative proposal and the privacy/cleanup semantic bridge. Save it;
         its byte SHA is the independently reviewed spec, never an approval.
      3. An external guard_proof JSON binds original_spec_sha256,
         original_source_archive_sha256 and original_bounds, and establishes true
         original_guards_verified/no_transport_bypass/
         unknown_attempts_bounded_by_original_guards with actual review evidence.
      4. Independent review JSON: approved=true, reviewer_kind=independent,
         blockers=[], spec_sha256, ledger_sha256, reserve_proposal_sha256,
         semantic_bridge_sha256 (SHA256 of native._json(semantic_bridge)) and
         guard_proof_sha256. Review the changed privacy/cleanup/guard handlers too.
      5. Separate parent/human reserve approval binds those same four digests,
         allowance_debit, guard_proof and independent_review {path,sha256} records,
         plus all five true review flags enforced by verify_continuation_approval.
      6. Final separate parent/human execution approval: approved=true,
         reviewer_kind, spec_sha256 and reserve_approval_sha256 of final bytes.

    CLI: plan-continuation -> verify-continuation-spec -> run-never-run;
    post-exit authority: verify-continuation-root --bundle NEW_ROOT. No API here
    writes approvals. The original proposal remains false, old roots immutable,
    failed/held rows unpromoted, and unknown prior elapsed/attempts remain unknown.
    """
    spec = verify_continuation_approval(spec_path, approval_path, reserve_approval_path)
    return _execute_shopping_continuation(spec, spec_path, approval_path, reserve_approval_path)


def main(argv=None):
    import argparse

    from .rack import RackProfile

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-runtime", help="copy pinned source; zero model calls")
    prepare.add_argument("--external-root", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    freeze = commands.add_parser(
        "freeze", help="all 150 tasks / 900 sessions; never grants QA permission"
    )
    for key in ("data-root", "pins", "runtime", "profile", "output-root", "spec-out"):
        freeze.add_argument("--" + key, type=Path, required=True)
    for key in ("calls-per-task", "seconds-per-task", "campaign-seconds", "max-rounds"):
        freeze.add_argument("--" + key, type=int, required=True)
    freeze.add_argument("--embedding-url", required=True)
    freeze.add_argument("--production-memory-qualification", type=Path)
    freeze.add_argument("--native-document-factories", type=Path)
    template = commands.add_parser(
        "approval-template", help="unapproved template for independent parent/human review"
    )
    template.add_argument("--spec", type=Path, required=True)
    template.add_argument("--out", type=Path, required=True)
    for name in ("run", "verify-spec"):
        command = commands.add_parser(name)
        command.add_argument("--spec", type=Path, required=True)
        command.add_argument("--approval", type=Path, required=True)
    verify = commands.add_parser("verify-task", help="offline saved-native-evidence verification")
    verify.add_argument("--bundle", type=Path, required=True)
    continuation = commands.add_parser(
        "plan-continuation",
        help="freeze versioned full ordered Shopping NEVER-RUN protocol; zero QA calls",
    )
    for key in ("runtime-spec", "ledger", "reserve-proposal", "spec-out"):
        continuation.add_argument("--" + key, type=Path, required=True)
    continuation.add_argument("--ledger-sha256", required=True)
    for name in ("run-never-run", "verify-continuation-spec"):
        command = commands.add_parser(
            name, help="separate reviewed guard/reserve and parent execution approval required"
        )
        for key in ("spec", "approval", "reserve-approval"):
            command.add_argument("--" + key, type=Path, required=True)
    root_verify = commands.add_parser(
        "verify-continuation-root",
        help="read-only full mixed-provenance native replay; no model/encoder/JVM",
    )
    root_verify.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan-continuation":
        spec = build_continuation_spec(
            runtime=json.loads(args.runtime_spec.read_bytes()),
            ledger_path=args.ledger,
            ledger_sha256=args.ledger_sha256,
            reserve_proposal_path=args.reserve_proposal,
        )
        with args.spec_out.open("xb") as stream:
            stream.write(native._json(spec))
        print(
            json.dumps(
                dict(
                    spec_sha256=sha(args.spec_out),
                    dispatch_rows=len(spec["plan"]["dispatch_indices"]),
                    retained_rows=len(spec["plan"]["retained_indices"]),
                    quality_calls=0,
                )
            )
        )
    elif args.command == "run-never-run":
        print(json.dumps(execute_continuation(args.spec, args.approval, args.reserve_approval)))
    elif args.command == "verify-continuation-spec":
        verify_continuation_spec(args.spec, args.approval, args.reserve_approval)
        print(json.dumps(dict(verified=True, quality_calls=0)))
    elif args.command == "verify-continuation-root":
        print(json.dumps(verify_continuation_root(args.bundle)))
    elif args.command == "prepare-runtime":
        print(
            json.dumps(
                {"runtime": str(prepare_runtime(args.external_root, args.out)), "model_calls": 0}
            )
        )
    elif args.command == "freeze":
        binding = (
            None
            if args.production_memory_qualification is None
            else {
                "path": str(args.production_memory_qualification.resolve()),
                "sha256": sha(args.production_memory_qualification),
            }
        )
        factories = (
            {}
            if args.native_document_factories is None
            else json.loads(args.native_document_factories.read_bytes())
        )
        spec = build_spec(
            data_root=args.data_root,
            pins=json.loads(args.pins.read_bytes()),
            runtime=args.runtime,
            profile=RackProfile.model_validate(json.loads(args.profile.read_bytes())),
            output_root=args.output_root,
            embedding_url=args.embedding_url,
            calls_per_task=args.calls_per_task,
            seconds_per_task=args.seconds_per_task,
            campaign_seconds=args.campaign_seconds,
            max_rounds=args.max_rounds,
            production_memory_qualification=binding,
            native_document_factories=factories,
        )
        with args.spec_out.open("xb") as stream:
            stream.write(native._json(spec))
        print(
            json.dumps(
                {
                    "spec_sha256": sha(args.spec_out),
                    "tasks": spec["tasks_count"],
                    "sessions": spec["sessions_count"],
                    "quality_calls": 0,
                }
            )
        )
    elif args.command == "approval-template":
        with args.out.open("xb") as stream:
            stream.write(
                native._json(
                    {
                        "approved": False,
                        "reviewer_kind": "parent",
                        "spec_sha256": sha(args.spec),
                        "note": (
                            "Independent parent/human must review exact frozen bytes "
                            "and set approved=true. This command does not approve execution."
                        ),
                    }
                )
            )
        print(json.dumps({"approved": False, "spec_sha256": sha(args.spec)}))
    elif args.command == "verify-task":
        result = verify_task(args.bundle)
        print(json.dumps({"status": result["status"], "calls": result["calls"]}))
    elif args.command == "verify-spec":
        spec = verify_spec(args.spec, args.approval)
        print(
            json.dumps(
                {
                    "tasks": spec["tasks_count"],
                    "sessions": spec["sessions_count"],
                    "quality_calls": 0,
                }
            )
        )
    else:
        result = execute_shopping(args.spec, args.approval)
        print(
            json.dumps(
                {
                    "quality_complete": result["quality_complete"],
                    "calls": result["calls"],
                    "completed_sessions": result["completed_sessions"],
                    "expected_sessions": result["expected_sessions"],
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
