"""Production-path rack tracing and whole-history fail-closed budgets."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path

import dspy
from gluomem.llm.router import LMRouter
from gluomem_bench.bench.parallel_campaign import guard_stopped


def failure_diagnostic(exc):
    """Bounded causal projection; never format exceptions or copy opaque payloads."""
    import math
    from gluomem_bench.bench.embedding_runtime import safe_failure_diagnostic
    from gluomem_bench.bench.cognee_native_boundary import clean_diagnostic
    from gluomem_bench.bench.lme_first_execute import GUARD_REASONS, GUARD_OPERATIONS
    result = safe_failure_diagnostic(exc)
    pending, seen = [exc], set()
    while pending and len(seen) < 12:
        item = pending.pop(0)
        if not isinstance(item, BaseException) or id(item) in seen:
            continue
        seen.add(id(item))
        fields = item.__dict__
        value = fields.get('guard_refusal')
        if ('guard_refusal' not in result and type(value) is dict
                and type(value.get('protocol')) is str
                and value['protocol'] == 'native-job-guard-refusal@v1'):
            row = {'protocol': 'native-job-guard-refusal@v1'}
            for key, allowed in (('reason', GUARD_REASONS), ('source_operation', GUARD_OPERATIONS)):
                v = value.get(key)
                row[key] = v if type(v) is str and v in allowed else 'unknown'
            for key in ('clock_monotonic', 'job_deadline_monotonic', 'campaign_deadline_monotonic'):
                v = value.get(key)
                row[key] = v if type(v) in (int, float) and 0 <= v < 1e15 and math.isfinite(v) else None
            v = value.get('call_ordinal')
            row['call_ordinal'] = v if type(v) is int and 0 <= v < 2**63 else None
            v = value.get('job_sha256')
            row['job_sha256'] = v if type(v) is str and len(v) == 64 and all(c in '0123456789abcdef' for c in v) else None
            result['guard_refusal'] = row
        if 'native_diagnostic' not in result and type(fields.get('native_diagnostic')) is dict:
            result['native_diagnostic'] = clean_diagnostic(fields['native_diagnostic'])
        pending.extend((BaseException.__cause__.__get__(item), BaseException.__context__.__get__(item)))
    return result


class RackLM(dspy.LM):
    """Real DSPy LM, guarded before generation and before adapter parsing.

    Never globally patches DSPy. Raw content may contain dataset text; traces
    belong only in the operator-selected external benchmark artifact directory.
    """

    def __init__(self, *args, client, task, trace: Path, budget, schema_binding=None, **kwargs):
        if schema_binding:
            from gluomem_bench.bench.rack_schema import schema_binding as derive_binding
            if (schema_binding != derive_binding(client.profile, schema_binding)
                    or kwargs.get("engine") != "lm15"):
                raise ValueError("invalid declared schema binding")
        self.rack_schema_binding = schema_binding
        self.rack_expected_model = (schema_binding["lm_model"] if schema_binding
                                    else "openai/" + client.profile.model)
        super().__init__(*args, **kwargs)
        self.rack_client = client
        self.rack_task = task
        self.trace = trace
        self.budget = budget

    def update_history(self, entry):
        """Retain only this task LM's latest raw response, never global/caller copies.

        DSPy's response processing and usage tracker still run unchanged. The
        native recorder also writes process-global and caller-module histories,
        which are not permitted sinks for scoped production memory inputs.
        """
        self.history[:] = [entry]

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
        dispatched = False
        original_error = None
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
                self.model != self.rack_expected_model
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
            with self.budget.lease(
                tokens + row["output_budget"] + profile.schema_reserve_tokens
            ):
                self.budget.begin_call()
                row["ordinal"] = self.budget.calls
                self.kwargs["timeout"] = min(profile.timeout_seconds, self.budget.remaining())
                prior = self.history[-1] if self.history else None
                dispatched = True
                try:
                    outputs = super().__call__(prompt, messages=messages, **kwargs)
                except Exception:
                    # Latch through the existing categorical job interface before
                    # the lease's legacy failure formatter can inspect the reason.
                    # Keep the original exception and traceback unchanged.
                    self.budget.failed = True
                    context = getattr(self.budget, "outer_budget", None)
                    if context is not None and hasattr(context, "job"):
                        context.failed = True
                    raise
            if not self.history or self.history[-1] is prior:
                raise ValueError("incomplete production response: missing raw history")
            raw = self.history[-1].get("response")
            if self.rack_schema_binding is not None:
                from gluomem_bench.bench.rack_schema import project_native_response
                row["response"], row["native_response"] = project_native_response(raw)
            else:
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
                from gluomem_bench.bench.rack import SharedRackError
                raise SharedRackError("rack production response model mismatch")
            self.budget.check()
            return outputs
        except Exception as exc:
            original_error = exc
            # A dispatched call without a response proves a failed attempt, not
            # an outage. DSPy's typed connection failure is shared evidence;
            # LMTimeoutError (including SDK read timeouts) remains job-local.
            from dspy.utils.exceptions import LMTransportError
            if dispatched and isinstance(exc, LMTransportError):
                exc.__dict__['shared_failure'] = True
            # Seed the scoped client's categorical latch before its legacy
            # recorder can format the original reason. Pass the original type
            # through for unchanged shared/local failure classification.
            if self.rack_client._failure is None:
                self.rack_client._failure = "incomplete production operation_failed"
            self.rack_client._latch(exc)
            self.budget.failed = True
            row["error"] = {"type": "Exception", "message": "operation_failed"}
            try:
                row["error"] = failure_diagnostic(exc)
            except Exception:
                pass  # Diagnostic collection must not replace the initiating error.
            raise
        finally:
            row["seconds"] = time.monotonic() - started
            try:
                self.trace.parent.mkdir(parents=True, exist_ok=True)
                with self.trace.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            except Exception:
                if original_error is None:
                    raise


class RackRouter(LMRouter):
    def __init__(self, cfg, *, client, trace: Path, budget, schema_transport=None):
        super().__init__(cfg)
        self.client = client
        self.trace = trace
        self.budget = budget
        self.schema_binding = None
        if schema_transport is not None:
            from gluomem_bench.bench.rack_schema import TRANSPORT, register_schema
            if schema_transport != TRANSPORT:
                raise ValueError("unknown declared schema transport")
            self.schema_binding = register_schema(client.profile, cfg)

    def lm_for(self, task):
        original = super().lm_for(task)
        options = dict(original.kwargs)
        options.pop("num_retries", None)
        if self.schema_binding:
            profile = self.client.profile
            if (original.model != "openai/" + profile.model
                    or options.get("api_base") != profile.base_url
                    or options.get("temperature") != 0
                    or options.get("max_tokens") != profile.output_budget(task)):
                raise ValueError("declared schema source model/config mismatch")
            template = {"enable_thinking": False}
            if options.pop("extra_body", None) != {"chat_template_kwargs": template}:
                raise ValueError("declared schema chat template mismatch")
            options["chat_template_kwargs"] = template
            options["engine"] = "lm15"
        return RackLM(
            self.schema_binding["lm_model"] if self.schema_binding else original.model,
            client=self.client,
            task=task,
            trace=self.trace,
            budget=self.budget,
            schema_binding=self.schema_binding,
            cache=False,
            num_retries=0,
            **options,
        )


class WorkloadBudget:
    def __init__(self, *, max_calls=128, max_seconds=900, clock=time.monotonic,
                 policy="legacy-v1", outer_budget=None):
        if policy not in {"legacy-v1", "role-budgets-v2", "role-budgets-v4", "role-budgets-v5", "role-budgets-v6", "role-budgets-v7"}:
            raise ValueError("unknown workload budget policy")
        calls_limit, seconds_limit = (16384, 21600) if policy in {"role-budgets-v5", "role-budgets-v6", "role-budgets-v7"} else (256, 900)
        if (type(max_calls) is not int or not 1 <= max_calls <= calls_limit
                or not 1 <= max_seconds <= seconds_limit):
            raise ValueError(
                f"workload bounds require <={calls_limit} calls and <={seconds_limit} seconds"
            )
        self.policy = policy
        self.outer_budget = outer_budget
        self.max_calls = max_calls
        self.max_seconds = max_seconds
        self.clock = clock
        self.started = clock()
        self.calls = 0
        self.failed = False

    def check(self):
        if self.outer_budget is not None:
            try:
                self.outer_budget.check()
            except Exception:
                self.failed = True
                raise
        if self.failed or self.clock() - self.started >= self.max_seconds:
            reason = 'job-failed' if self.failed else 'job-deadline'
            self.failed = True
            raise guard_stopped(RuntimeError("whole-history workload budget exhausted; incomplete"), reason)

    def fail_shared(self):
        """Latch the shared campaign, never reset a failed local history budget."""
        self.failed = True
        shared = getattr(self.outer_budget, 'budget', None)
        if shared is not None:
            shared.failed = True

    def remaining(self):
        self.check()
        local = self.max_seconds - (self.clock() - self.started)
        return min(local, self.outer_budget.remaining()) if self.outer_budget is not None else local

    def lease(self, tokens):
        """Delegate concurrency only; begin_call still accounts exactly once."""
        lease = getattr(self.outer_budget, "lease", None)
        return lease(tokens) if callable(lease) else nullcontext()

    def begin_call(self):
        self.check()
        if self.calls >= self.max_calls:
            self.failed = True
            raise guard_stopped(RuntimeError("whole-history workload call budget exhausted; incomplete"), 'job-call-cap')
        if self.outer_budget is not None:
            try:
                self.outer_budget.begin_call()
            except Exception:
                self.failed = True
                raise
        self.calls += 1
