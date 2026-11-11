"""Production-path rack tracing and whole-history fail-closed budgets."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import dspy
from gluomem.llm.router import LMRouter

from gluomem_bench.bench.parallel_campaign import guard_stopped
from gluomem_bench.bench.rack_provider import POOL_VERSION, PRODUCTION_PROVIDER_VERSIONS


def failure_diagnostic(exc):
    """Bounded causal projection; never format exceptions or copy opaque payloads."""
    import math

    from gluomem_bench.bench.cognee_native_boundary import clean_diagnostic
    from gluomem_bench.bench.embedding_runtime import safe_failure_diagnostic
    from gluomem_bench.bench.lme_first_execute import GUARD_OPERATIONS, GUARD_REASONS
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
            from gluomem_bench.bench.rack_schema import transport_binding as derive_binding
            if (schema_binding != derive_binding(client.profile, schema_binding,
                    capture_policy=schema_binding.get("extraction_contract", {}).get(
                        "capture_policy"))
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
        if client.profile.native_diagnostic is not None:
            from gluomem_bench.bench.rack_diagnostic import sampling_options
            self.kwargs.pop("temperature", None)
            self.kwargs.update(sampling_options(client.profile))

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
            if (self.rack_task in {"identity_alignment", "consolidate"}
                    and profile.semantic_producer_contract):
                from gluomem_bench.bench.rack_observed import (
                    observed_invocation as producer_invocation,
                )
                key = ("identity_invocation" if self.rack_task == "identity_alignment"
                       else "consolidation_invocation")
                row[key] = producer_invocation(self.rack_task)
            if self.rack_task == "event_claims" and self.rack_schema_binding is not None:
                from gluomem_bench.bench.rack_extraction import (
                    observed_invocation as extraction_invocation,
                )
                observed = extraction_invocation()
                if observed is not None:
                    row["extraction_invocation"] = observed
                elif "extraction_contract" in self.rack_schema_binding:
                    raise ValueError("strict extraction requires observed invocation binding")
            if (self.rack_task == "consolidate" and self.rack_schema_binding is not None
                    and not profile.semantic_producer_contract):
                from gluomem_bench.bench.rack_projection import observed_invocation
                invocation = observed_invocation()
                if invocation is not None:
                    row["consolidation_invocation"] = invocation
                elif "consolidation_contract" in self.rack_schema_binding:
                    raise ValueError("projected production requires observed invocation binding")
            self.budget.check()
            allowed = {"response_format", "max_tokens", "temperature", "n"}
            if profile.native_diagnostic is not None:
                allowed.add("top_p")
            if set(kwargs) - allowed or kwargs.get("n", 1) != 1:
                raise ValueError("rack production transport/model override is forbidden")
            if profile.native_diagnostic is not None:
                from gluomem_bench.bench.rack_diagnostic import validate_sampling_overrides
                validate_sampling_overrides(profile, kwargs)
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
                or kwargs.get("temperature", self.kwargs.get("temperature")) != (
                    profile.native_diagnostic.temperature if profile.native_diagnostic else 0)
            ):
                raise ValueError("rack production model/output/temperature mismatch")
            if prompt is not None or not messages:
                raise ValueError("rack production requires exact chat messages")
            from gluomem_bench.bench.rack_provider import validate_lm_policy
            validate_lm_policy(profile, self)
            tokens = self.rack_client.prompt_token_count(messages)
            row["prompt_token_count"] = tokens
            self.rack_client.admit_context(
                tokens,
                output_budget=row["output_budget"],
                input_chars=len(json.dumps(messages, ensure_ascii=False)),
            )
            if profile.profile_version == POOL_VERSION or (
                    profile.profile_version == "observed-provider-v13"
                    and profile.openrouter_pool is not None):
                from gluomem_bench.bench.rack_pool import run_pool_call
                (outputs, raw), serving_profile = run_pool_call(
                    self.rack_client, self.rack_task, tokens, row["output_budget"], row,
                    lambda single, seconds, attempt: self._pool_dispatch(
                        single, seconds, attempt, messages, kwargs),
                )
                dispatched = True
                self.history[:] = [{"response": raw}]
                prior = None
            else:
                row.update(self.rack_client.pricing_fields())
                dispatched = True
                outputs, prior = self._single_dispatch(prompt, messages, kwargs, tokens, row)
                serving_profile = profile
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
            from gluomem_bench.bench.rack_provider import response_model_matches
            if not response_model_matches(profile, response.get("model")):
                from gluomem_bench.bench.rack import SharedRackError
                raise SharedRackError("rack production response model mismatch")
            from gluomem_bench.bench.rack_provider import is_openrouter, response_evidence
            provider_response = (row["native_response"]["canonical"]["provider_data"]
                                 if is_openrouter(profile) else response)
            if (profile.native_diagnostic is not None
                    and profile.native_diagnostic.response_identity_policy is not None
                    and response.get("model") != provider_response.get("model")):
                from gluomem_bench.bench.rack import SharedRackError
                raise SharedRackError("diagnostic raw/SDK returned model values differ")
            row.update(response_evidence(serving_profile, provider_response, tokens,
                                         output_budget=row["output_budget"]))
            if profile.semantic_producer_contract is not None:
                from gluomem_bench.bench.rack_observed import validate_observed_response
                validate_observed_response(row, profile)
            if profile.native_diagnostic is not None:
                from gluomem_bench.bench.rack_diagnostic import validate_local_schema
                validate_local_schema(row)
            self.budget.check()
            return outputs
        except Exception as exc:
            original_error = exc
            from dspy.utils.exceptions import LMTransportError
            if dispatched and isinstance(exc, LMTransportError):
                exc.__dict__['shared_failure'] = True
            if self.rack_client._failure is None:
                self.rack_client._failure = "incomplete production operation_failed"
            self.rack_client._latch(exc)
            self.budget.failed = True
            row["error"] = {"type": "Exception", "message": "operation_failed"}
            try:
                row["error"] = failure_diagnostic(exc)
            except Exception:
                pass
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

    def _single_dispatch(self, prompt, messages, kwargs, tokens, row):
        profile = self.rack_client.profile
        with self.budget.lease(
                tokens + row["output_budget"] + profile.schema_reserve_tokens
            ):
                self.budget.begin_call()
                row["ordinal"] = self.budget.calls
                if profile.profile_version == "observed-provider-v13":
                    root_ordinal = getattr(
                        getattr(self.budget, "outer_budget", None), "ordinal", None)
                    if type(root_ordinal) is int:
                        row["root_ordinal"] = root_ordinal
                self.kwargs["timeout"] = min(profile.timeout_seconds, self.budget.remaining())
                prior = self.history[-1] if self.history else None
                try:
                    if (profile.native_diagnostic is not None
                            or profile.profile_version == "observed-provider-v13"):
                        from gluomem_bench.bench.rack_diagnostic import (
                            capture_native_attempt,
                            native_request,
                        )
                        from gluomem_bench.bench.rack_schema import project_native_response
                        class SingleNativeLM(dspy.LM):
                            def update_history(self, entry):
                                self.history[:] = [entry]

                        # The outer synchronous guard refuses forward/async bypasses.
                        # DSPy treats such overrides as a legacy engine, so dispatch
                        # through a plain real native LM, as the qualified pool does.
                        lm = SingleNativeLM(self.model, cache=False, num_retries=0,
                            engine="lm15", **self.kwargs)
                        try:
                            with capture_native_attempt(profile, row, self.kwargs["timeout"]):
                                raw = lm(native_request(self, row))
                            self.history[:] = lm.history[-1:]
                        finally:
                            lm.close()
                        view, _ = project_native_response(raw)
                        outputs = [view["choices"][0]["message"]["content"]]
                    elif profile.semantic_producer_contract is not None:
                        class ObservedNativeLM(dspy.LM):
                            def update_history(self, entry):
                                self.history[:] = [entry]

                        # The public forward/async routes are closed. Keep actual
                        # synchronous native serialization in a real plain SDK LM
                        # inside the one original lease/debit, not a fallback.
                        lm = ObservedNativeLM(self.model, cache=False, num_retries=0,
                            engine="lm15", **self.kwargs)
                        try:
                            outputs = lm(prompt, messages=messages, **kwargs)
                            self.history[:] = lm.history[-1:]
                        finally:
                            lm.close()
                    else:
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
        return outputs, prior

    def _pool_dispatch(self, single, seconds, attempt, messages, kwargs):
        from gluomem_bench.bench.rack_pool import capture_native_attempt, digest
        from gluomem_bench.bench.rack_provider import response_evidence
        from gluomem_bench.bench.rack_schema import _register_native, project_native_response

        binding = dict(self.rack_schema_binding)
        binding["lm_model"] = "rack-pool-attempt-" + digest(
            [binding, single.openrouter_endpoint.model_dump(mode="json")]) + "/" + single.model
        _register_native(single, binding, response_schema=True)

        class PhysicalLM(dspy.LM):
            def update_history(self, entry):
                self.history[:] = [entry]

        options = dict(self.kwargs, timeout=seconds)
        lm = PhysicalLM(binding["lm_model"], cache=False, num_retries=0, engine="lm15", **options)
        try:
            with capture_native_attempt(attempt, seconds):
                if single.profile_version == "observed-provider-v13":
                    from gluomem_bench.bench.rack_diagnostic import native_request
                    raw = lm(native_request(lm, {"messages": messages,
                        "output_budget": single.output_budget(self.rack_task),
                        "generation_kwargs": kwargs}, profile=single))
                    message = raw.provider_data.get("choices", [{}])[0].get("message", {})
                    if message.get("refusal") is not None:
                        raise ValueError("observed native refusal cannot be successful")
                    view, _ = project_native_response(raw)
                    outputs = [view["choices"][0]["message"]["content"]]
                else:
                    outputs = lm(messages=messages, **kwargs)
            raw = lm.history[-1]["response"]
            view, native = project_native_response(raw)
            attempt["response"], attempt["native_response"] = view, native
            response_evidence(single, native["canonical"]["provider_data"],
                self.rack_client.prompt_token_count(messages),
                output_budget=single.output_budget(self.rack_task))
            schema = kwargs.get("response_format", {}).get("json_schema", {}).get("schema")
            if schema is not None:
                from gluomem.llm.structured_adapter import _object
                from jsonschema import Draft202012Validator

                try:
                    payload = json.loads(view["choices"][0]["message"]["content"],
                                         object_pairs_hook=_object)
                    Draft202012Validator(schema).validate(payload)
                except Exception:
                    raise ValueError(
                        "native pool output schema validation failed; no semantic retry"
                    ) from None
            return outputs, raw
        finally:
            lm.close()


class PoolRackLM(RackLM):
    """Only the qualified synchronous __call__ pool path is admitted.

    Keep these overrides off legacy RackLM: DSPy detects class-level forward
    overrides as a legacy custom engine, even when they delegate unchanged.
    """

    def forward(self, prompt=None, messages=None, **kwargs):
        raise ValueError("unqualified pool forward bypass; use the guarded call boundary")

    async def aforward(self, prompt=None, messages=None, **kwargs):
        raise ValueError("unqualified asynchronous pool dispatch")

    async def acall(self, prompt=None, messages=None, **kwargs):
        raise ValueError("unqualified asynchronous pool dispatch")


class RackRouter(LMRouter):
    def __init__(self, cfg, *, client, trace: Path, budget, schema_transport=None):
        if client.profile.semantic_producer_contract is not None and (
                budget is None or client.budget is not budget):
            raise ValueError("observed producer requires one shared history budget")
        if (client.profile.semantic_producer_contract is not None
                or client.profile.native_diagnostic is not None
                or client.profile.profile_version == POOL_VERSION
                or client.profile.judge_profile is not None
                and client.profile.judge_profile.profile_version == POOL_VERSION):
            if (cfg.dspy.validation_retries or cfg.dspy.retry.max_attempts != 1
                    or any((t.refine or {}).get("enabled") for t in cfg.dspy.tasks.values())):
                raise ValueError("pool native factory forbids semantic regeneration/retry")
        super().__init__(cfg)
        self.client = client
        self.trace = trace
        self.budget = budget
        self.schema_binding = None
        self.judge_schema_binding = None
        if client.profile.provider is not None and schema_transport is None:
            from gluomem_bench.bench.rack_schema import TEXT_TRANSPORT, TRANSPORT
            if cfg.dspy.schema_mode not in {"required", "typed-text"}:
                raise ValueError("provider-aware native route requires declared schema transport")
            schema_transport = TEXT_TRANSPORT if cfg.dspy.schema_mode == "typed-text" else TRANSPORT
        if schema_transport is not None:
            from gluomem_bench.bench.rack_schema import (
                TEXT_TRANSPORT,
                TRANSPORT,
                register_schema,
                register_typed_text,
            )
            if schema_transport not in {TRANSPORT, TEXT_TRANSPORT}:
                raise ValueError("unknown declared schema transport")
            register = (
                register_typed_text if schema_transport == TEXT_TRANSPORT else register_schema
            )
            self.schema_binding = register(client.profile, cfg)
            judge_client = client.client_for("rubric")
            if judge_client is not client:
                judge_cfg = cfg.model_copy(deep=True)
                judge_cfg.write.writer_contract = None
                judge_cfg.write.semantic_producer_contract = None
                if client.profile.semantic_producer_contract is not None:
                    judge_cfg.dspy.tasks.pop("identity_alignment", None)
                    judge_cfg.dspy.output_field_limits.pop("identity_alignment", None)
                if judge_client.profile.production_json_transport is None:
                    if judge_cfg.dspy.schema_mode != "required":
                        judge_cfg.dspy.schema_mode = "required"
                        judge_cfg.dspy.schema_strict = True
                    if (
                        client.profile.native_schema_strict is False
                        or client.profile.native_diagnostic is not None
                    ):
                        judge_cfg.dspy.schema_strict = True
                    self.judge_schema_binding = register_schema(judge_client.profile, judge_cfg)
                else:
                    judge_cfg.dspy.schema_mode = "typed-text"
                    judge_cfg.dspy.schema_strict = False
                    self.judge_schema_binding = register_typed_text(judge_client.profile, judge_cfg)

    @contextmanager
    def bind(self, task):
        if task == "identity_alignment" and self.client.profile.semantic_producer_contract is None:
            raise ValueError("legacy rack profiles cannot admit identity alignment")
        with super().bind(task):
            client = self.client.client_for(task)
            binding = (
                self.judge_schema_binding if client is not self.client else self.schema_binding
            )
            if task == "event_claims" and binding:
                from gluomem_bench.bench.rack_extraction import ObservedSourceQuoteAdapter
                with dspy.context(adapter=ObservedSourceQuoteAdapter(
                        dspy.settings.adapter, client.profile,
                        self._cfg.write.source_capture_policy)):
                    yield
            elif task == "identity_alignment" and binding:
                from gluomem_bench.bench.rack_observed import ObservedAdapter
                with dspy.context(adapter=ObservedAdapter(
                        dspy.settings.adapter, client.profile, task)):
                    yield
            elif task == "consolidate" and binding:
                from gluomem_bench.bench.rack_projection import ObservedConsolidationAdapter

                with dspy.context(
                    adapter=ObservedConsolidationAdapter(dspy.settings.adapter, client.profile)
                ):
                    yield
            elif binding and (binding["schema_mode"] != self._cfg.dspy.schema_mode
                              or binding["schema_strict"] != self._cfg.dspy.schema_strict):
                from gluomem.llm.structured_adapter import make_adapter

                adapter = make_adapter(
                    self._cfg.dspy.adapter,
                    canonicalize=self._cfg.dspy.canonicalize,
                    schema_mode=binding["schema_mode"],
                    schema_strict=binding["schema_strict"],
                    validation_retries=self._cfg.dspy.validation_retries,
                )
                with dspy.context(adapter=adapter):
                    yield
            else:
                yield

    def lm_for(self, task):
        if self.client.profile.profile_version == "observed-provider-v13" and task not in {
                "event_claims", "identity_alignment", "consolidate", "answer", "rubric"}:
            raise ValueError("unqualified v13 native task")
        original = super().lm_for(task)
        client = self.client.client_for(task)
        binding = self.judge_schema_binding if client is not self.client else self.schema_binding
        return self._guarded_lm(original, task, client, binding)

    def lm_for_model(self, name, *, temperature=None, max_tokens=None):
        if self.client.profile.provider is None:
            return super().lm_for_model(name, temperature=temperature, max_tokens=max_tokens)
        if name != self._cfg.dspy.tasks["rubric"].model:
            raise ValueError(
                "provider-aware named-model route is restricted to the configured judge"
            )
        client = self.client.client_for("rubric")
        maximum = client.profile.output_budget("rubric")
        if temperature not in (None, 0) or max_tokens not in (None, maximum):
            raise ValueError("provider-aware named judge parameter mismatch")
        return self.lm_for("rubric")

    def _guarded_lm(self, original, task, client, binding):
        options = dict(original.kwargs)
        options.pop("num_retries", None)
        if binding:
            profile = client.profile
            if (original.model != "openai/" + profile.model
                    or options.get("api_base") != profile.base_url
                    or options.get("temperature") != (
                        profile.native_diagnostic.temperature if profile.native_diagnostic else 0)
                    or options.get("max_tokens") != profile.output_budget(task)):
                raise ValueError("declared schema source model/config mismatch")
            from gluomem_bench.bench.rack_provider import generation_options
            body = generation_options(profile)
            if options.pop("extra_body", None) != body:
                raise ValueError("declared schema chat template mismatch")
            if "chat_template_kwargs" in body:
                options["chat_template_kwargs"] = body["chat_template_kwargs"]
            else:
                options["reasoning"] = body["reasoning"]
            if profile.native_diagnostic is not None:
                from gluomem_bench.bench.rack_diagnostic import sampling_options
                options.pop("temperature", None)
                options.update(sampling_options(profile))
            options["engine"] = "lm15"
        lm_type = PoolRackLM if (client.profile.semantic_producer_contract is not None
            or client.profile.profile_version == POOL_VERSION
            or client.profile.native_diagnostic is not None) else RackLM
        return lm_type(
            binding["lm_model"] if binding else original.model,
            client=client,
            task=task,
            trace=self.trace,
            budget=self.budget,
            schema_binding=binding,
            cache=False,
            num_retries=0,
            **options,
        )


class WorkloadBudget:
    def __init__(self, *, max_calls=128, max_seconds=900, clock=time.monotonic,
                 policy="legacy-v1", outer_budget=None):
        if policy not in {"legacy-v1", "role-budgets-v2", "role-budgets-v4", "role-budgets-v5",
                "role-budgets-v6", "role-budgets-v7",
                "observed-producer-v12", "observed-provider-v13",
                *PRODUCTION_PROVIDER_VERSIONS}:
            raise ValueError("unknown workload budget policy")
        calls_limit, seconds_limit = (16384, 21600) if policy in {
            "role-budgets-v5", "role-budgets-v6", "role-budgets-v7",
            "observed-provider-v13", *PRODUCTION_PROVIDER_VERSIONS} else (256, 900)
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
