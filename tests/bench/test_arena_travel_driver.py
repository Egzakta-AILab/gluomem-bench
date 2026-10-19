"""Offline tests execute actual pinned Travel sources; no endpoint calls."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gluomem_bench.bench import arena_travel_driver as travel

ROOT = Path(
    'tests/fixtures/native/operator-supplied'
)
NATIVE = ROOT / "MemoryArena"


def authored():
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
    return {
        "id": -1,
        "base_person": {"name": "Base", "query": "Authored base.", "daily_plans": [day]},
        "questions": ["I am Test.\nAuthored integration only."],
        "answers": [[day]],
    }


def test_actual_native_prompt_entry_environment_and_official_scorer(tmp_path):
    source = travel.sources(NATIVE)
    with travel.native_travel(
        source,
        SimpleNamespace(profile=SimpleNamespace(model="authored"), sdk=lambda role: None),
        [authored()],
        executor=object(),
        max_steps=2,
    ) as runtime:
        row = runtime.rows[0]
        actor, env = runtime.actor, runtime.env
        assert row["questions"][0]["name"] == "Test"
        assert actor.system_prompt == runtime.prompts.AGENT_SYSTEM_PROMPT
        text = runtime.format_person_plan("Test", row["answers"][0]["daily_plans"])
        obs, reward, info = env.step(
            text,
            ground_truth={"name": "Test", "daily_plans": row["answers"][0]["daily_plans"]},
            need_judge=True,
        )
        assert reward is True and info == {}
        entry = actor.build_memory_entry("Authored task", text, obs, reward=reward)
        assert json.loads(entry) == {"task": "Authored task", "action": text}
        submission = tmp_path / "submission.jsonl"
        submission.write_text(
            json.dumps(
                {"id": -1, "persons": [{"person_idx": 1, "plan": row["answers"][0]["daily_plans"]}]}
            ),
            encoding="utf-8",
        )
        assert runtime.scorer.evaluate(str(submission)) == {"ps": 100.0, "sps": 0, "sr": 100.0}
    assert env.history == [] and env._current_observation is None


def budget_for(tmp_path, calls=8):
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        Job,
        JobContext,
        RequestLimiter,
    )

    job = Job("authored", "authored", tmp_path, calls, 120)
    outer = CampaignBudget(max_calls=calls, max_seconds=120)
    outer.reserve(job)
    return JobContext(job, outer, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))


def test_actual_native_actor_uses_scoped_sdk_and_exact_tool_schema(tmp_path):
    import httpx

    from gluomem_bench.bench.rack import RackProfile

    source, budget = travel.sources(NATIVE), budget_for(tmp_path)
    requests = []

    def http(request):
        payload = json.loads(request.content) if request.content else None
        requests.append((request.url.path, payload))
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "authored", "max_model_len": 131072}]})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100, "max_model_len": 131072})
        message = {"role": "assistant", "content": "Authored final plan"}
        if budget.calls == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "local-1",
                        "type": "function",
                        "function": {"name": "CitySearch", "arguments": '{"state":"California"}'},
                    }
                ],
            }
        return httpx.Response(
            200,
            json={
                "model": "authored",
                "choices": [
                    {
                        "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                        "message": message,
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    profile = RackProfile(
        base_url="https://example.invalid/v1",
        model="authored",
        max_tokens=128,
        context_window=131072,
    )
    client = travel.TravelTransport(
        profile,
        receipt=tmp_path / "http.jsonl",
        budget=budget,
        tools=travel.tool_schema(source),
        transport=httpx.MockTransport(http),
    )
    observed = []
    executor = SimpleNamespace(
        execute=lambda name, args: observed.append((name, args)) or "Authored local city result"
    )
    with travel.native_travel(
        source, client, [authored()], executor=executor, max_steps=3
    ) as runtime:
        runtime.actor.prepare_for_person("Test", 1, include_previous_plans=False)
        assert runtime.actor.act("Authored query") == "Authored final plan"
        entry = json.loads(
            runtime.actor.build_memory_entry("Authored query", runtime.actor.last_result.final_plan)
        )
        assert entry["scratchpad"][0]["tool_results"][0]["result"] == "Authored local city result"
        assert observed == [("CitySearch", {"state": "California"})]
    generations = [p for url, p in requests if url.endswith("/chat/completions")]
    assert budget.calls == 2
    assert all(p["tools"] == travel.tool_schema(source) for p in generations)
    assert generations[1]["messages"][-1]["role"] == "tool"
    assert generations[0]["max_tokens"] == 128
    assert generations[0]["stream"] is False and generations[0]["n"] == 1
    with pytest.raises(travel.common.NativeAbort):
        client.create(
            "actor",
            model="authored",
            messages=[],
            tools=[{"type": "function", "function": {"name": "purchase"}}],
            tool_choice="auto",
            temperature=0,
            max_tokens=128,
        )
    assert budget.calls == 2
    client.close()


class LocalIOBudget:
    def __init__(self, seconds=3):
        import time

        self.end = time.monotonic() + seconds
        self.calls = 0

    def remaining(self):
        import time

        return self.end - time.monotonic()

    def check(self):
        if self.remaining() <= 0:
            raise TimeoutError("authored local I/O deadline")


def authored_worker(tmp_path, body):
    """Only authored local tools; run the actual configured isolated interpreter."""
    import hashlib

    runtime = tmp_path / "travel_planner_env"
    (runtime / "database/flights").mkdir(parents=True)
    code = (
        "class ToolExecutor:\n"
        " def __init__(self, **kwargs): pass\n"
        " def execute(self, name, args):\n"
    )
    module = runtime / "tool_executor.py"
    module.write_bytes((code + body).encode("utf-8"))
    flight = runtime / "database/flights/clean_Flights_2022.csv"
    flight.write_bytes(b"authored,fixture\n")
    readiness = tmp_path / "readiness.json"
    readiness.write_bytes(
        json.dumps(
            {
                "native_commit": travel.PIN,
                "source_files": [
                    {
                        "path": str(module),
                        "sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
                        "bytes": module.stat().st_size,
                    }
                ],
                "flight": {"sha256": hashlib.sha256(flight.read_bytes()).hexdigest()},
            }
        ).encode("utf-8")
    )
    return readiness


def local_tools(readiness, budget=None, helper=None):
    return travel.NativeTools(
        python=ROOT / "native-environment/Scripts/python.exe",
        helper=helper or ROOT / "travel_native_tool_worker.py",
        readiness=readiness,
        budget=budget or LocalIOBudget(),
    )


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("ascii", "fixture result"),
        ("cp1252", "fixture\u2019s result"),
        ("unencodable", "fixture \u4e2d result"),
    ],
)
def test_unchanged_authored_worker_fixtures_roundtrip_utf8(tmp_path, mode, expected):
    # Same authored ToolExecutor source bytes as the retained pre-repair repro;
    # materialize outside frozen assets instead of depending on a scratch lifetime.
    ready = authored_worker(tmp_path, "  return " + repr(expected) + "\n")
    tools = local_tools(ready)
    try:
        assert tools.execute("CitySearch", {}) == expected
    finally:
        tools.close()
    assert tools.process.poll() == 0


def test_isolated_child_and_parent_all_streams_utf8(tmp_path):
    ready = authored_worker(
        tmp_path,
        "  import sys\n"
        "  print('diagnostic \U0001f9ed \u4e2d', file=sys.stderr)\n"
        "  return [args, [s.encoding for s in (sys.stdin, sys.stdout, sys.stderr)]]\n",
    )
    tools = local_tools(ready)
    try:
        answer, encodings = tools.execute(
            "CitySearch", {"state": "quote\u2019 \U0001f9ed \u65e5\u672c\u8a9e"}
        )
        assert answer == {"state": "quote\u2019 \U0001f9ed \u65e5\u672c\u8a9e"}
        assert all(s.lower().replace("-", "") == "utf8" for s in encodings)
        assert all(
            s.encoding == "utf-8"
            for s in (tools.process.stdin, tools.process.stdout, tools.process.stderr)
        )
    finally:
        tools.close()
    assert tools.budget.calls == 0


@pytest.mark.parametrize(
    "body,error",
    [
        ("  import os\n  os._exit(7)\n", RuntimeError),
        ("  import os\n  os.write(1, b'\\x92\\n')\n  return 'unreachable'\n", UnicodeDecodeError),
    ],
)
def test_reader_fault_reaches_waiter_immediately(tmp_path, body, error):
    import time

    tools = local_tools(authored_worker(tmp_path, body))
    started = time.monotonic()
    try:
        with pytest.raises(error):
            tools.execute("CitySearch", {})
        assert time.monotonic() - started < 1
    finally:
        if tools.process.poll() is None:
            tools.process.kill()
        tools.process.wait(timeout=1)
        for stream in (tools.process.stdin, tools.process.stdout, tools.process.stderr):
            try:
                stream.close()
            except OSError:
                pass


def test_closed_stdin_is_failed_without_masking_and_double_close_is_safe(tmp_path):
    tools = local_tools(authored_worker(tmp_path, "  return 'ok'\n"))
    tools.process.stdin.close()
    with pytest.raises(ValueError, match="closed"):
        tools.execute("CitySearch", {})
    tools.close()
    tools.close()
    assert tools.native_closed is True
    assert tools.process.poll() is not None
    assert tools.diagnostics()["primary"]["type"] == "ValueError"
    assert tools.diagnostics()["cleanup"]


def test_deadline_cleanup_is_bounded_and_verified(tmp_path):
    import time

    budget = LocalIOBudget(0.7)
    tools = local_tools(
        authored_worker(tmp_path, "  import time\n  time.sleep(30)\n  return 'late'\n"), budget
    )
    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            tools.execute("CitySearch", {})
    finally:
        tools.close()
        tools.close()
    assert time.monotonic() - start < 0.8
    assert tools.process.poll() is not None
    assert tools.native_closed is True
    assert tools.budget.calls == 0


def test_worker_stderr_is_drained_bounded_and_safe(tmp_path):
    ready = authored_worker(
        tmp_path,
        "  import sys\n"
        "  print('Authorization: Bearer authored-secret\\n' * 10000, file=sys.stderr)\n"
        "  raise ValueError('authored tool origin with secret-local-value')\n",
    )
    tools = local_tools(ready)
    try:
        with pytest.raises(RuntimeError, match="worker exited"):
            tools.execute("CitySearch", {})
    finally:
        tools.close()
    info = tools.diagnostics()
    assert info["stderr_bytes"] > 100000
    assert info["stderr_retained_bytes"] <= 16384
    assert info["worker"]["types"] == ["ValueError"]
    assert any(frame["function"] == "execute" for frame in info["worker"]["stack"])
    assert "authored-secret" not in json.dumps(info)
    assert "secret-local-value" not in json.dumps(info)
    assert tools.native_closed is True
    tools.close()


def test_worker_itself_forces_utf8_even_without_parent_flags(tmp_path):
    import subprocess

    ready = authored_worker(
        tmp_path,
        "  import sys\n"
        "  print('diagnostic \U0001f9ed \u4e2d', file=sys.stderr)\n"
        "  return [args, [s.encoding for s in (sys.stdin, sys.stdout, sys.stderr)]]\n",
    )
    result = subprocess.run(
        [
            str(ROOT / "native-environment/Scripts/python.exe"),
            "-I",
            str(ROOT / "travel_native_tool_worker.py"),
            "--readiness",
            str(ready),
        ],
        input=(
            json.dumps({"name": "CitySearch", "args": {"text": "curly\u2019 \U0001f9ed \u4e2d"}})
            + '\n{"close":true}\n'
        ).encode("utf-8"),
        capture_output=True,
        timeout=3,
    )
    assert result.returncode == 0
    rows = [json.loads(line) for line in result.stdout.decode("utf-8").splitlines()]
    assert rows[1]["result"][0] == {"text": "curly\u2019 \U0001f9ed \u4e2d"}
    assert all(s.lower().replace("-", "") == "utf8" for s in rows[1]["result"][1])
    assert "diagnostic \U0001f9ed \u4e2d" in result.stderr.decode("utf-8")


@pytest.mark.parametrize("error_type", ["ValueError", "KeyboardInterrupt"])
def test_worker_failure_stderr_has_stack_not_secret_locals_or_messages(tmp_path, error_type):
    ready = authored_worker(
        tmp_path,
        "  secret_local = 'authored-private-local'\n"
        f"  raise {error_type}('Authorization: Bearer authored-secret')\n",
    )
    tools = local_tools(ready)
    try:
        with pytest.raises(RuntimeError, match="worker exited"):
            tools.execute("CitySearch", {})
    finally:
        tools.close()
    assert tools.diagnostics()["worker"]["types"] == [error_type]
    assert b"authored-secret" not in tools._stderr_tail
    assert b"authored-private-local" not in tools._stderr_tail
    assert b"secret_local" not in tools._stderr_tail
    assert tools.native_closed


@pytest.mark.parametrize("origin", ["memory", "worker"])
def test_failed_leaf_preserves_origin_cleanup_stacks_spend_and_no_quality_zero(tmp_path, origin):
    import httpx

    from gluomem_bench.bench.rack import RackProfile

    source, budget = travel.sources(NATIVE), budget_for(tmp_path)
    profile = RackProfile(base_url="https://example.invalid/v1", model="authored", max_tokens=128)
    target = tmp_path / "task"

    def http(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"data": [{"id": "authored", "max_model_len": profile.context_window}]}
            )
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100, "max_model_len": profile.context_window})
        message = {"role": "assistant", "content": "=== Test's Plan ===\nDay 1:\nCurrent City: -"}
        if origin == "worker":
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "local-1",
                        "type": "function",
                        "function": {"name": "CitySearch", "arguments": '{"state":"California"}'},
                    }
                ],
            }
        return httpx.Response(
            200,
            json={
                "model": "authored",
                "choices": [
                    {
                        "finish_reason": "tool_calls" if origin == "worker" else "stop",
                        "message": message,
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    client = travel.TravelTransport(
        profile,
        receipt=target / "http.jsonl",
        budget=budget,
        tools=travel.tool_schema(source),
        transport=httpx.MockTransport(http),
    )
    memory = travel.formal.RawExperienceMemory("no_memory", budget=budget)
    if origin == "memory":

        def insert(experience):
            raise LookupError("Authorization: Bearer authored-origin-secret")

        memory.insert = insert

        def close():
            raise OSError("authored-cleanup-secret")

        executor = SimpleNamespace(close=close)
    else:
        executor = local_tools(
            authored_worker(tmp_path, "  raise ValueError('authored originating tool failure')\n"),
            budget,
        )
        close = executor.process.stdin.close

        def faulty_close():
            close()
            executor.process.stdin.close = close
            raise OSError("authored-cleanup-secret")

        executor.process.stdin.close = faulty_close
    with pytest.raises(RuntimeError, match="retained spend") as raised:
        travel.run_task(
            authored(),
            source=source,
            client=client,
            memory=memory,
            executor=executor,
            out=target,
            max_steps=2,
            system="no_memory",
        )
    failure = json.loads((target / "failure.json").read_bytes())
    assert failure["calls"] == budget.calls == 1
    assert failure["status"] == "failed" and failure["completed_sessions"] == 0
    assert failure["sessions"] == [{"id": 0, "status": "failed"}]
    assert "metrics" not in failure and not (target / "result.json").exists()
    diagnostics = failure["diagnostics"]
    if origin == "memory":
        assert isinstance(raised.value.__cause__, LookupError)
        assert diagnostics["exception_chain"][0]["type"] == "LookupError"
        assert diagnostics["cleanup"][0]["type"] == "OSError"
        assert diagnostics["cleanup"][0]["stack"]
        assert failure["native_closed"] is False
    else:
        assert diagnostics["native_tools"]["worker"]["types"] == ["ValueError"]
        assert diagnostics["native_tools"]["cleanup"][0]["type"] == "OSError"
        assert diagnostics["native_tools"]["cleanup"][0]["stack"]
        assert failure["native_closed"] is True
        executor.close()  # The production outer finally performs this second close.
    assert "authored-origin-secret" not in json.dumps(failure)
    assert "authored-cleanup-secret" not in json.dumps(failure)
    client.close()


def test_latched_job_budget_still_reaps_worker_inside_original_deadline(tmp_path):
    budget = budget_for(tmp_path)
    tools = local_tools(authored_worker(tmp_path, "  return 'ok'\n"), budget)
    budget.failed = True
    tools.close()
    tools.close()
    assert tools.native_closed is True
    assert tools.process.poll() is not None


def test_blocked_stdin_write_spends_same_deadline_and_reaps(tmp_path):
    import threading
    import time

    helper = tmp_path / "authored_blocked_stdin.py"
    helper.write_bytes(b"import time\nprint('{\"ready\":true}', flush=True)\ntime.sleep(30)\n")
    tools = local_tools(tmp_path / "unused-readiness.json", LocalIOBudget(0.7), helper)
    watchdog_stop = threading.Event()
    watchdog_fired = threading.Event()

    def watchdog():
        if not watchdog_stop.wait(1.2):
            watchdog_fired.set()
            tools.process.kill()

    thread = threading.Thread(target=watchdog, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            tools.execute("CitySearch", {"state": "x" * 1000000})
    finally:
        tools.close()
        watchdog_stop.set()
        thread.join(timeout=1)
    assert not watchdog_fired.is_set()
    assert time.monotonic() - started < 0.8
    assert tools.native_closed is True
    assert tools.process.poll() is not None


@pytest.mark.parametrize(
    "body,error",
    [
        ("  import os, time\n  os.close(1)\n  time.sleep(30)\n", RuntimeError),
        ("  import os, time\n  os.write(2, b'\\x92\\n')\n  time.sleep(30)\n", UnicodeDecodeError),
    ],
)
def test_pipe_eof_and_stderr_decode_error_do_not_wait_for_worker_exit(tmp_path, body, error):
    import time

    tools = local_tools(authored_worker(tmp_path, body))
    started = time.monotonic()
    try:
        with pytest.raises(error):
            tools.execute("CitySearch", {})
        assert time.monotonic() - started < 1
    finally:
        tools.close()
        tools.close()
    assert tools.native_closed and tools.process.poll() is not None


def test_real_dead_worker_broken_pipe_preserves_origin_and_closes_twice(tmp_path):
    tools = local_tools(authored_worker(tmp_path, "  return 'ok'\n"))
    tools.process.kill()
    tools.process.wait(timeout=1)
    try:
        # Fill the pipe: a small write can succeed against a just-dead Windows
        # pipe before EOF is observed, so it is not a deterministic EPIPE probe.
        with pytest.raises(OSError) as raised:
            tools.execute("CitySearch", {"state": "x" * 1000000})
    finally:
        tools.close()
        tools.close()
    assert tools.native_closed
    assert tools.diagnostics()["primary"]["type"] == type(raised.value).__name__


def test_worker_startup_failure_is_closed_failed_leaf_not_quality_zero(tmp_path):
    ready = authored_worker(tmp_path, "  return 'ok'\n")
    report = json.loads(ready.read_bytes())
    report["native_commit"] = "authored-wrong-pin"
    ready.write_bytes(json.dumps(report).encode("utf-8"))
    root = tmp_path / "matrix"

    def run_one(raw, system, budget, target):
        local_tools(ready, budget)
        raise AssertionError("startup should fail before any generation")

    result = travel.execute_matrix(
        [authored()],
        systems=["no_memory"],
        out=root,
        call_budget=3,
        seconds_budget=5,
        campaign_seconds=10,
        run_one=run_one,
    )
    failure = json.loads((root / "-1/no_memory/failure.json").read_bytes())
    assert failure["native_closed"] is True
    assert failure["calls"] == result["calls"] == 0
    assert failure["status"] == "failed"
    assert failure["sessions"] == [{"id": 0, "status": "unrun"}]
    assert failure["diagnostics"]["native_tools"]["worker"]["types"] == ["ValueError"]
    assert result["tasks"][0]["status"] == "failed" and result["quality_complete"] is False
    assert result["metrics"]["no_memory"]["completed_only_metrics"] is None
    assert result["metrics"]["no_memory"]["full_metrics"] is None
    assert "metrics" not in failure


def test_late_stderr_decoder_failure_cannot_be_published_as_success(tmp_path):
    import time

    ready = authored_worker(
        tmp_path,
        "  import os, threading, time\n"
        "  def later():\n"
        "   time.sleep(0.1)\n"
        "   os.write(2, b'\\x92\\n')\n"
        "  threading.Thread(target=later).start()\n"
        "  return 'ok'\n",
    )
    tools = local_tools(ready)
    assert tools.execute("CitySearch", {}) == "ok"
    time.sleep(0.2)
    with pytest.raises(UnicodeDecodeError):
        tools.close()
    tools.close()
    assert tools.native_closed
    assert tools.diagnostics()["primary"]["type"] == "UnicodeDecodeError"


def test_native_context_cleanup_preserves_primary_and_removes_private_modules():
    source = travel.sources(NATIVE)
    before = {name for name in sys.modules if name.startswith("_travel_")}
    with pytest.raises(LookupError) as caught:
        with travel.native_travel(
            source,
            SimpleNamespace(profile=SimpleNamespace(model="authored"), sdk=lambda role: None),
            [authored()],
            executor=object(),
            max_steps=2,
        ) as runtime:

            def close():
                raise OSError("authored cleanup secret")

            runtime.env.close = close
            raise LookupError("authored original secret")
    assert caught.value.native_cleanup_diagnostics[0]["type"] == "OSError"
    assert {name for name in sys.modules if name.startswith("_travel_")} == before


def test_actual_six_local_tools_in_isolated_native_runtime(tmp_path):
    readiness = ROOT / "arena-travel-native-readiness.json"
    resources = travel.resource_inventory(readiness)
    assert resources["native_commit"] == travel.PIN
    budget = budget_for(tmp_path)
    tools = travel.NativeTools(
        python=ROOT / "native-environment/Scripts/python.exe",
        helper=ROOT / "travel_native_tool_worker.py",
        readiness=readiness,
        budget=budget,
    )
    try:
        cases = [
            (
                "FlightSearch",
                {"origin": "New York", "destination": "Los Angeles", "date": "2022-03-01"},
            ),
            ("RestaurantSearch", {"city": "Rockford"}),
            ("AccommodationSearch", {"city": "Rockford"}),
            ("AttractionSearch", {"city": "Rockford"}),
            (
                "DistanceMatrix",
                {"origin": "New York", "destination": "Los Angeles", "mode": "driving"},
            ),
            ("CitySearch", {"state": "California"}),
        ]
        for name, args in cases:
            answer = tools.execute(name, args)
            assert isinstance(answer, str) and answer and not answer.startswith("Error")
        with pytest.raises(ValueError, match="local"):
            tools.execute("Purchase", {})
    finally:
        tools.close()
    assert tools.process.poll() == 0
    assert budget.calls == 0


def test_complete_task_native_submission_scoring_memory_and_offline_tamper_guard(tmp_path):
    import httpx

    from gluomem_bench.bench.rack import RackProfile

    raw = authored()
    raw["questions"].append("I am Next.\nUse earlier authored trip.")
    raw["answers"].append(raw["answers"][0])
    source, budget = travel.sources(NATIVE), budget_for(tmp_path)
    profile = RackProfile(base_url="https://example.invalid/v1", model="authored", max_tokens=128)
    texts = [
        "=== Test's Plan ===\nDay 1:\nCurrent City: -\n",
        "=== Next's Plan ===\nDay 1:\nCurrent City: -\n",
    ]

    def http(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"data": [{"id": "authored", "max_model_len": profile.context_window}]}
            )
        if req.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100, "max_model_len": profile.context_window})
        return httpx.Response(
            200,
            json={
                "model": "authored",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": texts[budget.calls - 1], "role": "assistant"},
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    target = tmp_path / "task"
    client = travel.TravelTransport(
        profile,
        receipt=target / "http.jsonl",
        budget=budget,
        tools=travel.tool_schema(source),
        transport=httpx.MockTransport(http),
    )
    memory = formal_memory = travel.formal.RawExperienceMemory("full_context", budget=budget)
    result = travel.run_task(
        raw,
        source=source,
        client=client,
        memory=memory,
        executor=object(),
        out=target,
        max_steps=3,
        system="full_context",
    )
    assert result["completed_sessions"] == 2 and result["calls"] == 2
    assert result["metrics"] == {"ps": 100.0, "sps": 0, "sr": 100.0}
    steps = [json.loads(line) for line in (target / "steps.jsonl").read_bytes().splitlines()]
    assert steps[0]["memory_context"] == ""
    assert json.loads(steps[0]["experience"]["content"])["name"] == "Test"
    assert steps[0]["experience"]["content"] in steps[1]["memory_context"]
    assert travel.verify_task(target) == result["metrics"]
    assert formal_memory.budget is client.budget
    campaign = {
        "tasks": [
            {"task_id": -1, "system": "full_context", "status": "complete", "output": str(target)},
            {"task_id": 99, "system": "full_context", "status": "failed"},
        ]
    }
    metrics = travel.campaign_metrics(tmp_path, campaign)
    assert metrics["full_context"]["complete_tasks"] == 1
    assert metrics["full_context"]["full_metrics"] is None
    assert metrics["full_context"]["completed_only_metrics"] == result["metrics"]
    wire_original = (target / "http.jsonl").read_bytes()
    manifest_original = (target / "result.json").read_bytes()
    wires = [json.loads(line) for line in wire_original.splitlines()]
    generation = next(w for w in wires if w["url"].endswith("/chat/completions"))
    generation["response"]["choices"][0]["message"]["content"] = "Tampered wire output"
    (target / "http.jsonl").write_bytes(b"".join(json.dumps(w).encode() + b"\n" for w in wires))
    manifest = json.loads(manifest_original)
    manifest["artifact_sha256"]["http.jsonl"] = travel.file_record(target / "http.jsonl")["sha256"]
    (target / "result.json").write_bytes(travel.common._json(manifest))
    with pytest.raises(ValueError, match="wire|replay"):
        travel.verify_task(target)
    (target / "http.jsonl").write_bytes(wire_original)
    (target / "result.json").write_bytes(manifest_original)
    (target / "submission.jsonl").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="hash"):
        travel.verify_task(target)
    client.close()


def test_full_data_freeze_and_actual_cli_refuses_unapproved_run(tmp_path):
    import subprocess

    from gluomem_bench.bench.rack import GenerationBudgets, RackProfile

    data = ROOT / "memoryarena-data/group_travel_planner/data.jsonl"
    profile = RackProfile(
        base_url="https://example.invalid/v1",
        model="operator-qwen",
        profile_version="role-budgets-v6",
        generation_budgets=GenerationBudgets(
            extraction=4096, consolidation=4096, reader=4096, judge=4096
        ),
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_bytes(travel.common._json(profile.model_dump(mode="json")))
    spec_path = tmp_path / "spec.json"
    command = [
        sys.executable,
        "-m",
        "gluomem_bench.bench.arena_travel_driver",
        "plan",
        "--data",
        str(data),
        "--data-sha256",
        travel.file_record(data)["sha256"],
        "--native-root",
        str(NATIVE),
        "--profile",
        str(profile_path),
        "--output-root",
        str(tmp_path / "not-created"),
        "--spec-out",
        str(spec_path),
        "--embedding-url",
        "http://example.invalid/v1",
        "--readiness",
        str(ROOT / "arena-travel-native-readiness.json"),
        "--tool-python",
        str(ROOT / "native-environment/Scripts/python.exe"),
        "--tool-helper",
        str(ROOT / "travel_native_tool_worker.py"),
        "--max-steps",
        "30",
        "--campaign-seconds",
        "3600",
    ]
    plan = subprocess.run(command, capture_output=True, text=True)
    assert plan.returncode == 0, plan.stderr
    spec = json.loads(spec_path.read_bytes())
    assert len(spec["tasks"]) == 270 and sum(t["sessions"] for t in spec["tasks"]) == 1869
    assert spec["systems"] == list(travel.SYSTEMS)
    approval = tmp_path / "approval.json"
    approval.write_bytes(
        travel.common._json(
            {
                "approved": True,
                "reviewer_kind": "parent",
                "spec_sha256": travel.file_record(spec_path)["sha256"],
            }
        )
    )
    assert travel.verify_spec(spec_path, approval)["tasks"] == spec["tasks"]
    original = spec_path.read_bytes()
    altered = json.loads(original)
    altered["tasks"][-1]["sessions"] += 1
    spec_path.write_bytes(travel.common._json(altered))
    approval.write_bytes(
        travel.common._json(
            {
                "approved": True,
                "reviewer_kind": "parent",
                "spec_sha256": travel.file_record(spec_path)["sha256"],
            }
        )
    )
    with pytest.raises(ValueError, match="inventory"):
        travel.verify_spec(spec_path, approval)
    spec_path.write_bytes(original)
    approval.write_bytes(b'{"approved":false}')
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "gluomem_bench.bench.arena_travel_driver",
            "run",
            "--spec",
            str(spec_path),
            "--approval",
            str(approval),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0 and "approval" in run.stderr
    assert not (tmp_path / "not-created").exists()


def test_root_authority_isolates_failed_middle_job_retains_spend(tmp_path, monkeypatch):

    # Scheduling-only authored rows; full native publication is tested below.
    monkeypatch.setattr(travel, "campaign_metrics", lambda *a, **k: {})
    monkeypatch.setattr(travel, "verify_root", lambda *a, **k: None)

    rows = [authored() for _ in range(3)]
    for i, row in enumerate(rows):
        row["id"] = i
    verified = []

    def run_one(raw, system, budget, target):
        budget.begin_call()
        if raw["id"] == 1:
            raise TimeoutError("single request deadline")
        target.mkdir(parents=True)
        (target / "result.json").write_bytes(b'{"status":"complete"}')
        return {"status": "complete", "completed_sessions": 1}

    def verify_one(target):
        verified.append(target)

    result = travel.execute_matrix(
        rows,
        systems=["no_memory"],
        out=tmp_path / "matrix",
        call_budget=2,
        seconds_budget=10,
        campaign_seconds=60,
        run_one=run_one,
        verify_one=verify_one,
    )
    assert [r["status"] for r in result["tasks"]] == ["complete", "failed", "complete"]
    assert result["calls"] == 3 and len(verified) == 2
    assert result["spend"]["unknown_usage_calls"] == 3
    saved = json.loads((tmp_path / "matrix/campaign-result.json").read_bytes())
    assert saved == result and saved["completion_authority"] == "root-only"


def test_real_production_memory_factory_reuses_scoped_router_and_budget(tmp_path):
    import httpx
    from gluomem.llm.modules import DspyExtractors

    from gluomem_bench.bench.rack import GenerationBudgets, RackProfile

    profile = RackProfile(
        base_url="https://example.invalid/v1",
        model="operator-qwen",
        profile_version="role-budgets-v6",
        generation_budgets=GenerationBudgets(
            extraction=4096, consolidation=4096, reader=4096, judge=4096
        ),
    )
    budget = budget_for(tmp_path)
    source = travel.sources(NATIVE)
    client = travel.TravelTransport(
        profile,
        receipt=tmp_path / "http.jsonl",
        budget=budget,
        tools=travel.tool_schema(source),
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(
                AssertionError("no network during factory construction")
            )
        ),
    )
    raw = authored()
    spec = {
        "embedding_url": "http://example.invalid/v1",
        "data": {"path": str(tmp_path / "data.jsonl"), "sha256": "0" * 64},
    }
    memory = travel.create_memory(
        "gluomem", spec=spec, raw=raw, client=client, budget=budget, out=tmp_path, factories={}
    )
    try:
        assert type(memory) is travel.formal.GluoExperienceMemory
        assert type(memory.extractors) is DspyExtractors
        assert memory.budget is client.budget
        assert memory.query("Authored query") == ""
        assert budget.calls == 0
    finally:
        memory.close()
        client.close()


def test_execution_requires_real_qualified_production_binding_before_scheduling(
    tmp_path, monkeypatch
):
    from gluomem_bench.bench.rack import GenerationBudgets, RackProfile

    profile = RackProfile(
        base_url="https://example.invalid/v1",
        model="operator-qwen",
        profile_version="role-budgets-v6",
        generation_budgets=GenerationBudgets(
            extraction=4096, consolidation=4096, reader=4096, judge=4096
        ),
    )
    monkeypatch.setattr(
        travel,
        "verify_spec",
        lambda *a, **k: {
            "profile": profile.model_dump(mode="json"),
            "production_memory_qualification": None,
        },
    )
    with pytest.raises(ValueError, match="qualification"):
        travel.execute_approved(tmp_path / "spec.json", tmp_path / "approval.json")


@pytest.mark.parametrize("failure_stage", ["verify", "finalize", "root_write"])
def test_root_finalization_failures_never_promote_leaf_success(
    tmp_path, monkeypatch, failure_stage
):
    # Isolate the historical leaf/finalizer seam from full native replay below.
    monkeypatch.setattr(travel, "campaign_metrics", lambda *a, **k: {})
    monkeypatch.setattr(travel, "verify_root", lambda *a, **k: None)
    root = tmp_path / "root"

    def run_one(raw, system, budget, target):
        budget.begin_call()
        target.mkdir(parents=True)
        (target / "result.json").write_bytes(b'{"status":"complete"}')
        return {"status": "complete", "completed_sessions": 1}

    def verify(target):
        if failure_stage == "verify":
            raise ValueError("independent verification failed")

    def finalize(result, root):
        if failure_stage == "finalize":
            raise OSError("official scorer finalization failed")
        return result

    original = Path.write_bytes

    def write(path, raw):
        if failure_stage == "root_write" and path.name == "campaign-result.pending.json":
            raise OSError("root result write failed")
        return original(path, raw)

    monkeypatch.setattr(Path, "write_bytes", write)
    kwargs = dict(
        rows=[authored()],
        systems=["no_memory"],
        out=root,
        call_budget=2,
        seconds_budget=10,
        campaign_seconds=30,
        run_one=run_one,
        verify_one=verify,
        finalize=finalize,
    )
    if failure_stage == "verify":
        result = travel.execute_matrix(**kwargs)
        assert result["tasks"][0]["status"] == "failed"
        assert result["calls"] == 1 and result["quality_complete"] is False
    else:
        with pytest.raises(OSError):
            travel.execute_matrix(**kwargs)
        assert not (root / "campaign-result.json").exists()
    assert (root / "-1/no_memory/result.json").exists()
    statuses = json.loads((root / "task-statuses.json").read_bytes())
    assert statuses[0]["calls"] == 1


@pytest.mark.parametrize(
    "fault,shared",
    [
        ("timeout", False),
        ("identity", True),
        ("length", False),
        ("overflow", False),
        ("usage", True),
    ],
)
def test_actual_transport_failure_scope_no_retry_or_fallback(tmp_path, fault, shared):
    import httpx

    from gluomem_bench.bench.rack import GenerationBudgets, RackProfile

    profile = RackProfile(
        base_url="https://example.invalid/v1",
        model="authored",
        profile_version="role-budgets-v6",
        generation_budgets=GenerationBudgets(
            extraction=4096, consolidation=4096, reader=4096, judge=4096
        ),
    )
    budget = budget_for(tmp_path)

    def http(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"data": [{"id": "authored", "max_model_len": profile.context_window}]}
            )
        if request.url.path == "/tokenize":
            return httpx.Response(
                200,
                json={
                    "count": 131072 if fault == "overflow" else 100,
                    "max_model_len": profile.context_window,
                },
            )
        if fault == "timeout":
            raise httpx.ReadTimeout("one generation timed out", request=request)
        return httpx.Response(
            200,
            json={
                "model": "wrong" if fault == "identity" else "authored",
                "choices": [
                    {
                        "finish_reason": "length" if fault == "length" else "stop",
                        "message": {"role": "assistant", "content": "Authored response"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 99 if fault == "usage" else 100,
                    "completion_tokens": 10,
                },
            },
        )

    client = travel.TravelTransport(
        profile,
        receipt=tmp_path / "http.jsonl",
        budget=budget,
        tools=travel.tool_schema(travel.sources(NATIVE)),
        transport=httpx.MockTransport(http),
    )
    kw = dict(
        model=profile.model,
        messages=[{"role": "user", "content": "Authored"}],
        tools=client.tools,
        tool_choice="auto",
        temperature=0,
        max_tokens=4096,
    )
    with pytest.raises(travel.common.NativeAbort):
        client.create("actor", **kw)
    assert budget.calls == (0 if fault == "overflow" else 1)
    assert budget.budget.failed is shared
    prior = budget.calls
    with pytest.raises(travel.common.NativeAbort):
        client.create("actor", **kw)
    assert budget.calls == prior
    client.close()


def test_native_tool_cleanup_failure_withholds_completion_and_retains_calls(tmp_path):
    import httpx

    from gluomem_bench.bench.rack import RackProfile

    budget = budget_for(tmp_path)
    profile = RackProfile(base_url="https://example.invalid/v1", model="authored", max_tokens=128)

    def http(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"data": [{"id": "authored", "max_model_len": profile.context_window}]}
            )
        if req.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 100, "max_model_len": profile.context_window})
        return httpx.Response(
            200,
            json={
                "model": "authored",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "=== Test's Plan ===\nDay 1:\nCurrent City: -",
                            "role": "assistant",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    source = travel.sources(NATIVE)
    target = tmp_path / "task"
    client = travel.TravelTransport(
        profile,
        receipt=target / "http.jsonl",
        budget=budget,
        tools=travel.tool_schema(source),
        transport=httpx.MockTransport(http),
    )

    def close():
        raise OSError("native worker cleanup fault")

    with pytest.raises(RuntimeError, match="retained spend"):
        travel.run_task(
            authored(),
            source=source,
            client=client,
            memory=travel.formal.RawExperienceMemory("no_memory", budget=budget),
            executor=SimpleNamespace(close=close),
            out=target,
            max_steps=2,
            system="no_memory",
        )
    assert not (target / "result.json").exists()
    failure = json.loads((target / "failure.json").read_bytes())
    assert failure["calls"] == 1 and failure["status"] == "failed"
    client.close()


def test_actual_cli_offline_smoke_exercises_all_six_native_tools_and_rescores(tmp_path):
    import subprocess

    out = tmp_path / "smoke"
    command = [
        sys.executable,
        "-m",
        "gluomem_bench.bench.arena_travel_driver",
        "smoke",
        "--native-root",
        str(NATIVE),
        "--readiness",
        str(ROOT / "arena-travel-native-readiness.json"),
        "--tool-python",
        str(ROOT / "native-environment/Scripts/python.exe"),
        "--tool-helper",
        str(ROOT / "travel_native_tool_worker.py"),
        "--out",
        str(out),
    ]
    run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 0, run.stderr + "\n" + run.stdout[-3000:]
    root = json.loads((out / "campaign-result.json").read_bytes())
    assert root["model_calls"] == 0 and root["calls"] == 3 and root["quality_complete"] is True
    leaf = out / "-1/full_context"
    steps = [json.loads(line) for line in (leaf / "steps.jsonl").read_bytes().splitlines()]
    assert len(steps) == 2 and len(steps[0]["native_result"]["scratchpad"][0]["tool_calls"]) == 6
    assert all(
        not r["result"].startswith("Error")
        for r in steps[0]["native_result"]["scratchpad"][0]["tool_results"]
    )
    verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "gluomem_bench.bench.arena_travel_driver",
            "verify-task",
            "--bundle",
            str(leaf),
        ],
        capture_output=True,
        text=True,
    )
    assert verify.returncode == 0, verify.stderr
    frozen = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    check = subprocess.run(
        [
            sys.executable,
            "-m",
            "gluomem_bench.bench.arena_travel_driver",
            "verify-root",
            "--bundle",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr
    assert frozen == {
        str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()
    }
    statuses = json.loads((out / "task-statuses.json").read_bytes())
    statuses[0]["status"] = "failed"
    (out / "task-statuses.json").write_bytes(travel.common._json(statuses))
    with pytest.raises(ValueError, match="root|coordination"):
        travel.verify_root(out)


def test_spend_reconciles_actual_native_and_production_usage_with_unknown_failed_calls(tmp_path):
    travel.common._append(
        tmp_path / "http.jsonl",
        {
            "url": "https://example.invalid/v1/chat/completions",
            "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        },
    )
    travel.common._append(
        tmp_path / "http.jsonl", {"url": "https://example.invalid/v1/models", "response": {}}
    )
    travel.common._append(
        tmp_path / "production-raw.jsonl",
        {
            "task": "event_claims",
            "ordinal": 2,
            "response": {"usage": {"prompt_tokens": 20, "completion_tokens": 4}},
        },
    )
    spend = travel.retained_spend(tmp_path, 3)
    assert (
        spend["calls"] == 3
        and spend["known_usage_calls"] == 2
        and spend["unknown_usage_calls"] == 1
    )
    assert (
        spend["prompt_tokens"] == 30
        and spend["completion_tokens"] == 9
        and spend["currency_cost"] is None
    )
    assert spend["roles"]["native_actor"]["calls"] == 1
    assert spend["roles"]["event_claims"]["prompt_tokens"] == 20


def test_pairing_uses_only_root_verified_exact_task_intersection(tmp_path):
    tasks = []
    for system, values in [("gluomem", [(1, 100), (2, 0)]), ("bm25", [(1, 0), (2, 0)])]:
        for tid, value in values:
            path = tmp_path / system / str(tid)
            path.mkdir(parents=True)
            (path / "result.json").write_bytes(
                travel.common._json({"metrics": {"ps": value, "sps": value, "sr": value}})
            )
            tasks.append(
                {
                    "task_id": tid,
                    "system": system,
                    "status": "complete",
                    "output": str(path),
                    "input_sha256": str(tid),
                }
            )
    tasks.append({"task_id": 3, "system": "bm25", "status": "failed"})
    pairs = travel.paired_results({"tasks": tasks})
    assert pairs["gluomem_vs_bm25"]["task_ids"] == [1, 2]
    assert pairs["gluomem_vs_bm25"]["ps"] == {"wins": 1, "losses": 0, "ties": 1}


def test_whole_task_time_envelope_is_required_before_job_admission(tmp_path):
    admitted = []

    def run_one(*args):
        admitted.append(args)
        raise AssertionError("must not construct native clients or tools")

    result = travel.execute_matrix(
        [authored()],
        systems=["no_memory"],
        out=tmp_path / "root",
        call_budget=2,
        seconds_budget=10,
        campaign_seconds=1,
        run_one=run_one,
    )
    assert not admitted and result["calls"] == 0
    assert result["tasks"][0]["status"] == "unrun"


@pytest.mark.parametrize("entry", ["cli", "approved"])
def test_actual_full_run_cli_preserves_all_unrun_coverage_without_spending_or_client_construction(
    tmp_path,
    monkeypatch,
    entry,
):
    """Real qualification replay + real full CLI, deliberately non-admitted time envelope."""
    import shutil
    import subprocess

    from gluomem_bench.bench.rack import RackProfile

    proof = json.loads((ROOT / "arena-full-authored-memory/qualification.json").read_bytes())
    copied = tmp_path / "qualified-task"
    shutil.copytree(proof["task_bundle"], copied)
    proof["task_bundle"] = str(copied)
    for stem in ("receipt", "wire"):
        path = Path(proof["capability"][stem + "_path"])
        dest = tmp_path / path.name
        shutil.copy2(path, dest)
        proof["capability"][stem + "_path"] = str(dest)
    qualification = tmp_path / "qualification.json"
    qualification.write_bytes(travel.common._json(proof))
    profile = RackProfile.model_validate(
        {**proof["profile"], "max_generation_calls": 512, "max_run_seconds": 3600}
    )
    data = ROOT / "memoryarena-data/group_travel_planner/data.jsonl"
    out = tmp_path / "full-run"
    spec = travel.build_spec(
        data=data,
        data_sha256=travel.file_record(data)["sha256"],
        native_root=NATIVE,
        profile=profile,
        output_root=out,
        embedding_url="http://example.invalid/v1",
        readiness=ROOT / "arena-travel-native-readiness.json",
        tool_python=ROOT / "native-environment/Scripts/python.exe",
        tool_helper=ROOT / "travel_native_tool_worker.py",
        max_steps=30,
        # Insufficient for any 3600s task, sufficient for full root reporting.
        campaign_seconds=120,
        production_memory_qualification=qualification,
    )
    sp, ap = tmp_path / "spec.json", tmp_path / "approval.json"
    sp.write_bytes(travel.common._json(spec))
    ap.write_bytes(
        travel.common._json(
            {
                "approved": True,
                "reviewer_kind": "parent",
                "spec_sha256": travel.file_record(sp)["sha256"],
                "scope": "offline non-admission fixture; no generations",
            }
        )
    )
    if entry == "approved":
        events = []
        verify, replace = travel.verify_root, Path.replace

        def checked(root, **kwargs):
            assert not (root / "campaign-result.json").exists()
            result = verify(root, **kwargs)
            events.append("verified")
            return result

        def promoted(path, target):
            if path.name == "campaign-result.pending.json":
                assert events == ["verified"]
                events.append("promoted")
            return replace(path, target)

        with monkeypatch.context() as patch:
            patch.setattr(travel, "verify_root", checked)
            patch.setattr(Path, "replace", promoted)
            result = travel.execute_approved(sp, ap)
        assert events == ["verified", "promoted"]
        assert result["finalized"]
    else:
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "gluomem_bench.bench.arena_travel_driver",
                "run",
                "--spec",
                str(sp),
                "--approval",
                str(ap),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert run.returncode == 0, run.stderr + "\n" + run.stdout[-3000:]
    result = json.loads((out / "campaign-result.json").read_bytes())
    assert len(result["tasks"]) == 1620 and result["expected_sessions"] == 11214
    assert result["calls"] == 0 and all(r["status"] == "unrun" for r in result["tasks"])
    assert result["quality_complete"] is False
    assert result["paired"]["gluomem_vs_no_memory"]["complete_intersection"] == 0
    assert all(m["full_metrics"] is None for m in result["metrics"].values())
    assert (out / "execution-spec.json").read_bytes() == sp.read_bytes()
    assert (out / "execution-approval.json").read_bytes() == ap.read_bytes()
    assert (out / "frozen-source.zip").exists()
    assert travel.verify_root(out)["quality_complete"] is False


def test_live_transport_rejects_self_attested_callback_admission(tmp_path):
    from gluomem_bench.bench.rack import RackProfile

    with pytest.raises(ValueError, match="approval|permit"):
        travel.TravelTransport(
            RackProfile(
                base_url="https://example.invalid/v1", model="operator-qwen", max_tokens=128
            ),
            receipt=tmp_path / "http.jsonl",
            budget=budget_for(tmp_path),
            tools=travel.tool_schema(travel.sources(NATIVE)),
            admission=lambda: None,
        )
    assert not (tmp_path / "http.jsonl").exists()


def test_finalizer_actual_elapsed_exhaustion_withholds_root_and_retains_spend(tmp_path):
    import time

    from gluomem_bench.bench.parallel_campaign import CampaignStopped

    root = tmp_path / "deadline"

    def run(raw, system, budget, target):
        budget.begin_call()
        target.mkdir(parents=True)
        (target / "result.json").write_bytes(b'{"status":"complete"}')
        return {"status": "complete", "completed_sessions": 1}

    def finalize(result, output):
        time.sleep(0.25)
        return result

    with pytest.raises(CampaignStopped, match="deadline"):
        travel.execute_matrix(
            [authored()],
            systems=["no_memory"],
            out=root,
            call_budget=2,
            seconds_budget=0.01,
            campaign_seconds=0.2,
            run_one=run,
            verify_one=lambda target: None,
            finalize=finalize,
        )
    assert not (root / "campaign-result.json").exists()
    assert not (root / "campaign-result.pending.json").exists()
    failure = json.loads((root / "campaign-failure.json").read_bytes())
    assert failure["calls"] == 1 and failure["elapsed_seconds"] >= 0.25
    assert failure["quality_complete"] is False and failure["finalized"] is False
    assert failure["spend"]["unknown_usage_calls"] == 1
    assert (root / "-1/no_memory/result.json").exists()


@pytest.fixture(scope="module")
def native_root_leaf(tmp_path_factory):
    """Actual six native tools and saved SDK mock receipts, never real models."""
    root = tmp_path_factory.mktemp("travel-finalization-native")
    travel.smoke(
        native_root=NATIVE,
        readiness=ROOT / "arena-travel-native-readiness.json",
        tool_python=ROOT / "native-environment/Scripts/python.exe",
        tool_helper=ROOT / "travel_native_tool_worker.py",
        out=root / "smoke",
    )
    return root / "smoke/-1/full_context"


def native_matrix_kwargs(native_root_leaf, root):
    import shutil

    raw = json.loads((native_root_leaf / "input-task.json").read_bytes())

    def run(raw, system, budget, target):
        shutil.copytree(native_root_leaf, target)
        leaf = json.loads((target / "result.json").read_bytes())
        for _ in range(leaf["calls"]):
            budget.begin_call()
        return leaf

    return dict(
        rows=[raw],
        systems=["full_context"],
        out=root,
        call_budget=4,
        seconds_budget=10,
        campaign_seconds=120,
        run_one=run,
    )


def test_full_root_verifier_must_accept_staged_candidate_before_sole_promotion(
    tmp_path,
    monkeypatch,
    native_root_leaf,
):
    original = travel.verify_root
    events = []
    root = tmp_path / "root"

    def verify(output, **kwargs):
        assert not (output / "campaign-result.json").exists()
        assert kwargs["manifest_name"] == "campaign-result.pending.json"
        checked = original(output, **kwargs)
        events.append("verified")
        return checked

    replace = Path.replace

    def promote(path, target):
        if path.name == "campaign-result.pending.json":
            assert events == ["verified"]
            events.append("promoted")
        return replace(path, target)

    monkeypatch.setattr(travel, "verify_root", verify)
    monkeypatch.setattr(Path, "replace", promote)
    result = travel.execute_matrix(**native_matrix_kwargs(native_root_leaf, root))
    assert events == ["verified", "promoted"]
    assert result["quality_complete"] and result["finalized"]
    assert original(root)["verified"] is True
    assert not (root / "campaign-result.pending.json").exists()


@pytest.mark.parametrize(
    "stage",
    [
        "metrics",
        "verify_root",
        "write_before",
        "write_after",
        "candidate_read",
        "verified_read",
        "replace_before",
        "replace_after",
        "final_read",
    ],
)
@pytest.mark.parametrize("fault", ["exception", "deadline"])
def test_native_root_publication_faults_retain_spend_and_allow_fresh_retry(
    tmp_path,
    monkeypatch,
    native_root_leaf,
    stage,
    fault,
):
    root = tmp_path / "fault"
    kwargs = native_matrix_kwargs(native_root_leaf, root)
    run = kwargs["run_one"]
    outer = []
    fired = []

    def run_one(raw, system, budget, target):
        outer.append(budget.budget)
        return run(raw, system, budget, target)

    kwargs["run_one"] = run_one

    def trip():
        fired.append(stage)
        if fault == "deadline":
            outer[0].deadline = outer[0].clock() - 0.001
        else:
            raise OSError("injected " + stage)

    with monkeypatch.context() as patch:
        if stage in {"metrics", "verify_root"}:
            original = getattr(travel, "campaign_metrics" if stage == "metrics" else stage)

            def wrapped(*args, **kw):
                result = original(*args, **kw)
                trip()
                return result

            patch.setattr(travel, "campaign_metrics" if stage == "metrics" else stage, wrapped)
        elif stage.startswith("write"):
            original = Path.write_bytes

            def write(path, raw):
                if path == root / "campaign-result.pending.json" and stage == "write_before":
                    trip()
                result = original(path, raw)
                if path == root / "campaign-result.pending.json" and stage == "write_after":
                    trip()
                return result

            patch.setattr(Path, "write_bytes", write)
        elif stage.startswith("replace"):
            original = Path.replace

            def replace(path, target):
                if path == root / "campaign-result.pending.json" and stage == "replace_before":
                    trip()
                result = original(path, target)
                if path == root / "campaign-result.pending.json" and stage == "replace_after":
                    trip()
                return result

            patch.setattr(Path, "replace", replace)
        else:
            original = Path.read_bytes
            reads = []

            def read(path):
                result = original(path)
                if path == root / "campaign-result.pending.json":
                    reads.append(path)
                    # Read 2 is the full verifier's own load; read 3 is the
                    # exact byte check after that verifier has accepted it.
                    if (stage == "candidate_read" and len(reads) == 1) or (
                        stage == "verified_read" and len(reads) == 3
                    ):
                        trip()
                if path == root / "campaign-result.json" and stage == "final_read":
                    trip()
                return result

            patch.setattr(Path, "read_bytes", read)
        with pytest.raises(RuntimeError if fault == "deadline" else OSError):
            travel.execute_matrix(**kwargs)

    assert fired == [stage]
    assert not (root / "campaign-result.json").exists()
    assert not (root / "campaign-result.pending.json").exists()
    failure = json.loads((root / "campaign-failure.json").read_bytes())
    statuses = json.loads((root / "task-statuses.json").read_bytes())
    assert failure["status"] == "incomplete" and failure["finalized"] is False
    assert failure["quality_complete"] is False and failure["calls"] == 3
    assert failure["spend"]["calls"] == 3
    assert failure["spend"]["prompt_tokens"] == 300
    assert failure["spend"]["completion_tokens"] == 30
    assert failure["spend"]["unknown_usage_calls"] == 0
    assert failure["elapsed_seconds"] >= 0 and statuses[0]["calls"] == 3
    assert statuses[0]["status"] == "complete"
    assert travel.verify_task(root / "-1/full_context")["ps"] == 100.0
    with pytest.raises((ValueError, FileNotFoundError)):
        travel.verify_root(root)
    fresh = tmp_path / "fresh"
    healthy = travel.execute_matrix(**native_matrix_kwargs(native_root_leaf, fresh))
    assert healthy["quality_complete"] and travel.verify_root(fresh)["verified"]


@pytest.mark.parametrize("tamper", ["totals", "metrics", "frozen", "approval"])
def test_staged_candidate_runs_full_root_checks_not_only_leaf_replay(
    tmp_path,
    monkeypatch,
    native_root_leaf,
    tamper,
):
    root = tmp_path / "root"
    original = travel.verify_root
    checks = []

    def verify(output, **kwargs):
        path = output / kwargs["manifest_name"]
        candidate = json.loads(path.read_bytes())
        if tamper == "totals":
            candidate["calls"] += 1
        elif tamper == "metrics":
            candidate["metrics"]["full_context"]["full_metrics"]["ps"] = -123
        elif tamper == "frozen":
            frozen = json.loads((output / "frozen-matrix.json").read_bytes())
            frozen[0]["expected_sessions"] += 1
            (output / "frozen-matrix.json").write_bytes(travel.common._json(frozen))
        else:
            # Exact inventory is valid, but the saved approval is not.
            spec = {
                "tasks": [
                    {"id": -1, "sessions": 2, "input_sha256": candidate["tasks"][0]["input_sha256"]}
                ],
                "systems": ["full_context"],
            }
            (output / "execution-spec.json").write_bytes(travel.common._json(spec))
            (output / "execution-approval.json").write_bytes(b'{"approved":false}')
        path.write_bytes(travel.common._json(candidate))
        with pytest.raises(ValueError) as caught:
            original(output, **kwargs)
        checks.append(str(caught.value))
        raise caught.value

    monkeypatch.setattr(travel, "verify_root", verify)
    with pytest.raises(ValueError):
        travel.execute_matrix(**native_matrix_kwargs(native_root_leaf, root))
    assert len(checks) == 1
    assert not (root / "campaign-result.json").exists()
    assert not (root / "campaign-result.pending.json").exists()
    assert travel.verify_task(root / "-1/full_context")["ps"] == 100.0
