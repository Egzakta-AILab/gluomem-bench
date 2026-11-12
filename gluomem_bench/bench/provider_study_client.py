"""Versioned study post-response guards at the rack client/native factory seams."""

from gluomem_bench.bench.provider_study_evidence import key, validate_response
from gluomem_bench.bench.rack import FatalRackError, RackClient
from gluomem_bench.bench.rack_provider import generation_options, is_openrouter, selected_endpoint


def currency_for(budget):
    seen = set()
    while budget is not None and id(budget) not in seen:
        seen.add(id(budget))
        currency = getattr(budget, "currency", None)
        if currency is not None:
            return currency
        budget = getattr(budget, "outer_budget", None)
    return None


def validate_currency_metadata(profile, metadata, budget):
    """Recheck exact selected endpoint prices against the predispatch upper bound."""
    from decimal import Decimal

    currency = currency_for(budget)
    if not is_openrouter(profile) or currency is None:
        return
    try:
        raw = metadata.get("provider_metadata", metadata)["data"]
        endpoints = raw if isinstance(raw, list) else raw["endpoints"]
        selected = [e for e in endpoints if e.get("tag") == selected_endpoint(profile).tag]
        if len(selected) != 1:
            raise ValueError("exact paid price metadata missing")
        pricing = selected[0]["pricing"]
        for name in ("prompt", "completion"):
            value = pricing[name]
            if type(value) is not str:
                raise ValueError("explicit decimal provider prices required")
            price = Decimal(value)
            if not price.is_finite() or price < 0 or price * Decimal(1000000) > currency.rate:
                raise ValueError("provider price exceeds frozen currency reservation rate")
        if pricing.get("request", "0") not in ("0", "0.0", 0):
            raise ValueError("unreserved per-request provider price unsupported")
    except Exception:
        currency.failed = True
        raise ValueError("provider price/currency admission failed") from None


class ProviderStudyClient(RackClient):
    """Check actual usage immediately, before the next role or source mutation."""

    def preflight(self):
        result = super().preflight()
        try:
            validate_currency_metadata(self.profile, result, self.budget)
        except Exception as exc:
            exc.__dict__["shared_failure"] = True
            self._latch(exc)
            raise
        return result

    def generate_messages(self, task, messages, **kwargs):
        active = self.client_for(task)
        try:
            result = super().generate_messages(task, messages, **kwargs)
        except Exception:
            # The legacy client may format a private transport exception. Keep its
            # typed causal evidence for safe diagnostics, never its text publicly.
            raise FatalRackError("provider-study generation operation failed") from None
        profile = active.profile
        try:
            validate_response(
                profile,
                {
                    "model": profile.model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", profile.temperature),
                    "max_tokens": kwargs.get("max_tokens", profile.output_budget(task)),
                    **generation_options(profile),
                },
                result["response"],
                output=kwargs.get("max_tokens", profile.output_budget(task)),
                counts={key(messages): result["admitted_prompt_tokens"]},
            )
        except Exception as exc:
            self._latch(exc)
            raise
        return result

    def prompt_token_count(self, messages):
        import threading

        count = super().prompt_token_count(messages)
        if not hasattr(self, "_count_local"):
            self._count_local = threading.local()
        self._count_local.last = (key(messages), count)
        return count


def make_study_router(cfg, *, client, trace, budget):
    """Bind the same declared native factory plus immediate raw usage validation."""
    from gluomem_bench.bench.rack_schema import TEXT_TRANSPORT, TRANSPORT, project_native_response
    from gluomem_bench.bench.rack_trace import RackLM, RackRouter

    class CheckedNativeLM(RackLM):
        def __call__(self, prompt=None, *, messages=None, **kwargs):
            result = super().__call__(prompt, messages=messages, **kwargs)
            try:
                _, native = project_native_response(self.history[-1]["response"])
                raw = native["canonical"]["provider_data"]
                profile = self.rack_client.profile
                counted = getattr(getattr(self.rack_client, "_count_local", None), "last", None)
                if counted is None or counted[0] != key(messages):
                    raise ValueError("native exact token admission missing")
                validate_response(
                    profile,
                    {
                        "model": profile.model,
                        "messages": messages,
                        "temperature": 0,
                        "max_tokens": profile.output_budget(self.rack_task),
                        **generation_options(profile),
                    },
                    raw,
                    output=profile.output_budget(self.rack_task),
                    counts={counted[0]: counted[1]},
                )
            except Exception as exc:
                self.rack_client._latch(exc)
                raise
            return result

    class CheckedRouter(RackRouter):
        def _guarded_lm(self, original, task, client, binding):
            guarded = super()._guarded_lm(original, task, client, binding)
            return CheckedNativeLM(
                guarded.model,
                client=client,
                task=task,
                trace=self.trace,
                budget=self.budget,
                schema_binding=binding,
                cache=False,
                num_retries=0,
                engine="lm15",
                **guarded.kwargs,
            )

    transport = TEXT_TRANSPORT if cfg.dspy.schema_mode == "typed-text" else TRANSPORT
    return CheckedRouter(cfg, client=client, trace=trace, budget=budget, schema_transport=transport)
