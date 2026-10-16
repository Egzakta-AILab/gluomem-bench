"""Production-path rack tracing and whole-history fail-closed budgets."""

from __future__ import annotations

import json
import time
from pathlib import Path

import dspy
from gluomem.llm.router import LMRouter


class RackLM(dspy.LM):
    """Real DSPy LM, guarded before generation and before adapter parsing.

    Never globally patches DSPy. Raw content may contain dataset text; traces
    belong only in the operator-selected external benchmark artifact directory.
    """

    def __init__(self, *args, client, task, trace: Path, budget, **kwargs):
        super().__init__(*args, **kwargs)
        self.rack_client = client
        self.rack_task = task
        self.trace = trace
        self.budget = budget

    def __call__(self, prompt=None, *, messages=None, **kwargs):
        self.rack_client.check()
        self.budget.check()
        profile = self.rack_client.profile
        row = {
            "task": self.rack_task,
            "model": self.model,
            "messages": messages,
            "output_budget": profile.output_budget(self.rack_task),
        }
        started = time.monotonic()
        try:
            self.budget.check()
            allowed = {"response_format", "max_tokens", "temperature", "n"}
            if set(kwargs) - allowed or kwargs.get("n", 1) != 1:
                raise ValueError("rack production transport/model override is forbidden")
            row["generation_kwargs"] = {
                key: value.model_json_schema()
                if isinstance(value, type) and hasattr(value, "model_json_schema")
                else value
                for key, value in kwargs.items()
            }
            if (
                self.model != "openai/" + profile.model
                or self.kwargs.get("api_base") != profile.base_url
                or kwargs.get("max_tokens", self.kwargs.get("max_tokens")) != row["output_budget"]
                or kwargs.get("temperature", self.kwargs.get("temperature")) != 0
            ):
                raise ValueError("rack production model/output/temperature mismatch")
            if prompt is not None or not messages:
                raise ValueError("rack production requires exact chat messages")
            tokens = self.rack_client.prompt_token_count(messages)
            row["prompt_token_count"] = tokens
            self.rack_client.admit_context(
                tokens,
                output_budget=row["output_budget"],
                input_chars=len(json.dumps(messages, ensure_ascii=False)),
            )
            self.budget.begin_call()
            row["ordinal"] = self.budget.calls
            self.kwargs["timeout"] = min(profile.timeout_seconds, self.budget.remaining())
            prior = len(self.history)
            outputs = super().__call__(prompt, messages=messages, **kwargs)
            if len(self.history) <= prior:
                raise ValueError("incomplete production response: missing raw history")
            raw = self.history[-1].get("response")
            row["response"] = raw.model_dump(mode="json") if hasattr(raw, "model_dump") else raw
            row["outputs"] = outputs
            response = row["response"]
            if not isinstance(response, dict):
                raise ValueError("incomplete production response: missing raw response")
            reasons = [c.get("finish_reason") for c in response.get("choices", [])]
            row["finish_reasons"] = reasons
            if not reasons or any(r != "stop" for r in reasons):
                raise ValueError("incomplete production response; refusing repaired/truncated JSON")
            if response.get("model") != profile.model:
                raise ValueError("rack production response model mismatch")
            self.budget.check()
            return outputs
        except Exception as exc:
            self.rack_client._latch(exc)
            self.budget.failed = True
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row["seconds"] = time.monotonic() - started
            self.trace.parent.mkdir(parents=True, exist_ok=True)
            with self.trace.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


class RackRouter(LMRouter):
    def __init__(self, cfg, *, client, trace: Path, budget):
        super().__init__(cfg)
        self.client = client
        self.trace = trace
        self.budget = budget

    def lm_for(self, task):
        original = super().lm_for(task)
        return RackLM(
            original.model,
            client=self.client,
            task=task,
            trace=self.trace,
            budget=self.budget,
            cache=False,
            num_retries=0,
            **original.kwargs,
        )


class WorkloadBudget:
    def __init__(self, *, max_calls=128, max_seconds=900, clock=time.monotonic):
        if type(max_calls) is not int or not 1 <= max_calls <= 256 or not 1 <= max_seconds <= 900:
            raise ValueError("workload bounds require <=256 calls and <=900 seconds")
        self.max_calls = max_calls
        self.max_seconds = max_seconds
        self.clock = clock
        self.started = clock()
        self.calls = 0
        self.failed = False

    def check(self):
        if self.failed or self.clock() - self.started >= self.max_seconds:
            self.failed = True
            raise RuntimeError("whole-history workload budget exhausted; incomplete")

    def remaining(self):
        self.check()
        return self.max_seconds - (self.clock() - self.started)

    def begin_call(self):
        self.check()
        if self.calls >= self.max_calls:
            self.failed = True
            raise RuntimeError("whole-history workload call budget exhausted; incomplete")
        self.calls += 1
