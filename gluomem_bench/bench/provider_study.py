"""Explicit v8 and opt-in v10 LME-S bindings over existing scheduler/history factories.

This module supplies version-specific admission and currency to comprehensive_driver;
it is not a generic benchmark runner. The legacy closed-producer contracts are untouched.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from decimal import Decimal
from pathlib import Path

from gluomem_bench.bench.rack import RackClient

_PHYSICAL_CONTEXT = ContextVar("provider_study_physical_context", default=None)
_PHYSICAL_LOCK = threading.RLock()

PROTOCOL = "longmemeval-provider-study@v8"
QUALIFICATION = "provider-native-qualification-128@v1"
POOL_PROTOCOL = "longmemeval-provider-study@v10"
POOL_QUALIFICATION = "provider-pool-qualification-128@v1"
POOL_SUPPORTED = ("gluomem", "no_memory", "bm25", "dense", "hybrid", "full_context")
POOL_TAGS = (
    "nebius/fp8",
    "parasail/fp8",
    "crusoe/fp8",
    "atlas-cloud/fp8",
    "morph/fp8",
    "deepinfra/fp8",
)
POOL_UNSUPPORTED = [
    {"family": "longmemeval_s", "system": system, "reason": reason}
    for system, reason in (
        ("mem0", "native-pool-factory-unqualified"),
        ("cognee", "native-pool-factory-unqualified"),
        ("qwen", "retained-quality-contract-failure-no-retry"),
    )
] + [
    {"family": family, "system": "all", "reason": "v10-family-admission-unqualified"}
    for family in ("memoryagentbench", "beam", "memoryarena")
]


def money(value):
    if type(value) not in (str, int):
        raise ValueError("currency values require explicit decimal strings")
    result = Decimal(value)
    if not result.is_finite() or result <= 0:
        raise ValueError("currency values must be positive finite decimals")
    return result


class CurrencyReservations:
    """Append-only worst-price token reservations; no success-only or retry refunds.

    Every call, including local Qwen, reserves the paid maximum token price. This
    deliberately over-reserves shared-role work, avoids guessing role from tokens,
    and covers prompt+output+schema before physical dispatch at existing lease seam.
    It is a conservative allowance debit, NEVER a claim of actual billed cost.
    """

    def __init__(self, *, max_usd, max_rate_per_million, receipt, max_prompt_rate_per_million=None):
        self.max_usd, self.rate = money(max_usd), money(max_rate_per_million)
        self.prompt_rate = (
            money(max_prompt_rate_per_million) if max_prompt_rate_per_million is not None else None
        )
        self.receipt = Path(receipt)
        self.receipt.parent.mkdir(parents=True, exist_ok=True)
        self.receipt.open("xb").close()
        self.lock = threading.RLock()
        self.reserved = Decimal(0)
        self.sequence = 0
        self.failed = False
        self.first_dispatch_monotonic = self.first_dispatch_wall = None

    def append(self, row):
        with self.lock, self.receipt.open("ab") as stream:
            stream.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())

    def reserve(self, tokens, job_id, *, prepared=None, active=None):
        with self.lock:
            if type(tokens) is not int or tokens < 1:
                raise ValueError("positive full-context currency reservation required")
            value = self.rate * Decimal(tokens) / Decimal(1000000)
            extra = {}
            if self.prompt_rate is not None:
                if prepared is None or active is None:
                    raise ValueError("pool physical reservation needs exact logical admission")
                prompt, output = prepared["prompt_tokens"], prepared["output_tokens"]
                if tokens != prompt + output + prepared["schema_reserve_tokens"]:
                    raise ValueError("pool lease differs from original logical admission")
                prompt += prepared["schema_reserve_tokens"]
                value = (self.prompt_rate * prompt + self.rate * output) / Decimal(1000000)
                extra = {
                    "prompt_reserved_tokens": prompt,
                    "output_reserved_tokens": output,
                    "endpoint": active["attempt"]["endpoint"],
                    "intent": active["attempt"]["intent"],
                    "logical_ordinal": active["attempt"]["logical_ordinal"],
                    "logical_deadline": active["logical_deadline"],
                }
            if self.failed or self.reserved + value > self.max_usd:
                self.failed = True
                raise RuntimeError("shared currency reservation cap exhausted")
            self.reserved += value
            self.sequence += 1
            row = {
                "kind": "reservation",
                "reservation_id": self.sequence,
                "job_id": job_id,
                "tokens": tokens,
                "reserved_usd": str(value),
                "cumulative_reserved_usd": str(self.reserved),
                "policy": "all-calls-worst-price-no-refund@v1",
                **extra,
            }
            if self.prompt_rate is not None:
                row["policy"] = "pool-componentwise-worst-price-no-refund@v1"
            try:
                self.append(row)
            except BaseException:
                self.failed = True
                raise
            return row

    def snapshot(self):
        with self.lock:
            result = {
                "reserved_usd": str(self.reserved),
                "max_usd": str(self.max_usd),
                "max_rate_per_million": str(self.rate),
                "reservations": self.sequence,
                "billed_usd": None,
                "policy": "all-calls-worst-price-no-refund@v1",
            }
            if self.prompt_rate is not None:
                result.update(
                    max_prompt_rate_per_million=str(self.prompt_rate),
                    policy="pool-componentwise-worst-price-no-refund@v1",
                    first_dispatch_monotonic=self.first_dispatch_monotonic,
                    first_dispatch_wall=self.first_dispatch_wall,
                )
            return result


class CostedJobContext:
    """Decorate JobContext; original aggregate calls/deadline/leases stay authoritative."""

    def __init__(self, context, currency):
        self.context, self.currency = context, currency
        self._local = threading.local()

    def __getattr__(self, name):
        return getattr(self.context, name)

    @property
    def failed(self):
        return self.context.failed

    @failed.setter
    def failed(self, value):
        self.context.failed = value

    @contextmanager
    def lease(self, tokens):
        self.check()
        if getattr(self._local, "reservation", None) is not None:
            raise RuntimeError("nested currency generation lease forbidden")
        token = None
        try:
            with self.context.lease(tokens):
                from gluomem_bench.bench.rack_pool import current_pool_attempt

                active = current_pool_attempt() if self.currency.prompt_rate is not None else None
                self._local.reservation = self.currency.reserve(
                    tokens,
                    self.job.job_id,
                    prepared=getattr(self._local, "prepared", None),
                    active=active,
                )
                if active is not None:
                    token = _PHYSICAL_CONTEXT.set(self)
                yield
        except BaseException:
            self.context.failed = True
            if self.currency.failed:
                self.budget.failed = True
            raise
        finally:
            if token is not None:
                _PHYSICAL_CONTEXT.reset(token)
            self._local.reservation = None

    def prepare(self, profile, tokens, output):
        self._local.prepared = {
            "profile": profile,
            "prompt_tokens": tokens,
            "output_tokens": output,
            "schema_reserve_tokens": profile.schema_reserve_tokens,
        }

    @property
    def ordinal(self):
        return self.budget.calls

    @property
    def root(self):
        # Pool health/logical ordinals belong to the very same campaign that
        # owns calls and physical slots, not a new state per job or role.
        return self.budget

    def physical_before(self, url, raw):
        """Durable send intent in the same currency journal, before either transport."""
        import time

        from gluomem_bench.bench.rack_pool import current_pool_attempt
        from gluomem_bench.bench.rack_provider import estimated_prompt_tokens, generation_options

        self.check()
        active, prepared, reservation = (
            current_pool_attempt(),
            self._local.prepared,
            self._local.reservation,
        )
        request = json.loads(raw)
        profile = active["profile"]
        if (
            url != profile.base_url + "/chat/completions"
            or reservation.get("physical_sent")
            or not reservation.get("dispatched")
            or reservation["intent"] != active["attempt"]["intent"]
            or request.get("model") != profile.model
            or request.get("temperature") != 0
            or request.get("max_tokens") != prepared["output_tokens"]
            or estimated_prompt_tokens(request["messages"]) != prepared["prompt_tokens"]
            or any(request.get(k) != v for k, v in generation_options(profile).items())
            or "service_tier" in request
        ):
            self.currency.failed = self.budget.failed = True
            raise ValueError("pool physical request escapes exact original admission")
        seconds = min(
            profile.timeout_seconds, self.remaining(), active["logical_deadline"] - time.monotonic()
        )
        if seconds <= 0:
            raise TimeoutError("pool original deadline exhausted")
        with self.budget.lock:
            if self.currency.first_dispatch_monotonic is None:
                self.currency.first_dispatch_monotonic = time.monotonic()
                self.currency.first_dispatch_wall = time.time()
            row = {
                "kind": "physical-send",
                "reservation_id": reservation["reservation_id"],
                "job_id": self.job.job_id,
                "campaign_call_ordinal": self.budget.calls,
                "call_ordinal": self.calls,
                "intent": reservation["intent"],
                "request": request,
                "endpoint": active["attempt"]["endpoint"],
                "logical_deadline": active["logical_deadline"],
                "job_deadline": self.deadline,
                "campaign_deadline": self.budget.deadline,
                "campaign_started": self.budget.started,
                "started_monotonic": time.monotonic(),
                "first_dispatch_monotonic": self.currency.first_dispatch_monotonic,
                "first_dispatch_wall": self.currency.first_dispatch_wall,
                "timeout_seconds": seconds,
            }
            try:
                self.currency.append(row)
            except BaseException:
                self.currency.failed = self.budget.failed = True
                raise
            reservation["physical_sent"] = True
        return row, seconds

    def physical_after(self, sent, status, raw):
        import time

        try:
            response = json.loads(raw)
        except (ValueError, TypeError):
            response = None
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        observed = {
            k: usage.get(k) if type(usage.get(k)) is int and usage[k] >= 0 else None
            for k in ("prompt_tokens", "completion_tokens", "cost")
        }
        row = {
            **sent,
            "kind": "physical-response",
            "status": status,
            "response": response,
            "completed_monotonic": time.monotonic(),
            "usage": observed,
            "actual_cost_usd": None,
        }
        try:
            self.currency.append(row)
        except BaseException:
            self.currency.failed = self.budget.failed = True
            raise

    def begin_call(self):
        reservation = getattr(self._local, "reservation", None)
        if reservation is None or reservation.get("dispatched"):
            self.budget.failed = True
            raise RuntimeError("physical dispatch requires a fresh currency reservation")
        try:
            # Keep the original campaign ordinal and durable dispatch intent in one
            # critical section; HTTP still runs outside this accounting lock.
            with self.budget.lock:
                self.context.begin_call()
                self.currency.append(
                    {
                        "kind": "dispatch-intent",
                        "reservation_id": reservation["reservation_id"],
                        "job_id": self.job.job_id,
                        "call_ordinal": self.calls,
                        "campaign_call_ordinal": self.budget.calls,
                        "reserved_usd": reservation["reserved_usd"],
                    }
                )
        except BaseException:
            self.currency.failed = True
            self.budget.failed = True
            self.context.failed = True
            raise
        reservation["dispatched"] = True

    def check(self):
        self.context.check()
        if self.currency.failed:
            self.budget.failed = True
            raise RuntimeError("shared currency reservation guard failed")


class PoolStudyClient(RackClient):
    """Bind existing pool dispatches to the existing costed JobContext."""

    def __init__(self, profile, **kwargs):
        super().__init__(profile, **kwargs)
        if self._judge_client is not None:
            self._judge_client.close()
            self._judge_client = PoolStudyClient(
                profile.judge_profile,
                receipt=self.receipt.with_name(self.receipt.stem + "-judge" + self.receipt.suffix),
                budget=self.budget,
                transport=kwargs.get("transport"),
            )

    def admit_context(self, tokens, *, output_budget=None, input_chars=None):
        super().admit_context(tokens, output_budget=output_budget, input_chars=input_chars)
        outer = getattr(self.budget, "outer_budget", None)
        if not isinstance(outer, CostedJobContext) or outer.currency.prompt_rate is None:
            raise ValueError("pool study requires original shared componentwise currency authority")
        outer.prepare(self.profile, tokens, output_budget or self.profile.max_tokens)


@contextmanager
def capture_pool_physical():
    """Observe both actual synchronous transports, never count wrapper receipts twice."""
    from dataclasses import replace

    import httpx
    from dspy._vendor.lm15.transports import StdlibTransport, TransportResponse

    with _PHYSICAL_LOCK:
        original_send, original_stream = httpx.Client.send, StdlibTransport.stream

        def send(client, request, **kwargs):
            context = _PHYSICAL_CONTEXT.get()
            if context is None:
                return original_send(client, request, **kwargs)
            sent, seconds = context.physical_before(str(request.url), request.content)
            request.extensions["timeout"] = {
                k: seconds for k in ("connect", "read", "write", "pool")
            }
            try:
                response = original_send(client, request, **kwargs)
                response.read()
            except BaseException:
                context.physical_after(sent, None, b"")
                raise
            context.physical_after(sent, response.status_code, response.content)
            return response

        def stream(transport, request):
            context = _PHYSICAL_CONTEXT.get()
            if context is None:
                return original_stream(transport, request)
            sent, seconds = context.physical_before(request.url, request.body)
            request = replace(
                request, connect_timeout=seconds, read_timeout=seconds, write_timeout=seconds
            )
            chunks, status = [], None
            try:
                with original_stream(transport, request) as response:
                    status = response.status
                    size = 0
                    for chunk in response:
                        chunks.append(chunk)
                        size += len(chunk)
                        if size > 8_000_000:
                            raise ValueError("pool response byte bound")
                    content = b"".join(chunks)
                    context.physical_after(sent, status, content)
                    return TransportResponse(
                        status=response.status,
                        reason=response.reason,
                        headers=response.headers,
                        http_version=response.http_version,
                        chunks=iter([content]),
                        release=lambda consumed: None,
                    )
            except BaseException:
                context.physical_after(sent, status, b"".join(chunks))
                raise

        httpx.Client.send, StdlibTransport.stream = send, stream
        try:
            yield
        finally:
            httpx.Client.send, StdlibTransport.stream = original_send, original_stream


SUPPORTED = ("gluomem", "mem0", "cognee", "no_memory", "bm25", "dense", "hybrid", "full_context")
UNSUPPORTED = [
    {
        "family": "memoryagentbench",
        "system": "cognee",
        "reason": "legacy-full-occurrence-matrix-required",
    },
    {"family": "beam", "system": "cognee", "reason": "no-admitted-native-factory"},
    {
        "family": "memoryarena",
        "system": "mem0/cognee",
        "reason": "native-experience-factories-unqualified",
    },
    {"family": "longmemeval_s", "system": "mem0-source-clock", "reason": "2.2.1-native-clock-only"},
    {
        "family": "longmemeval_s",
        "system": "cognee-final-answer",
        "reason": "selected-graph-evidence-common-reader-only",
    },
]


def _sha(path):
    from gluomem_bench.bench.artifacts import sha256_file

    return sha256_file(path)


def _save(path, value):
    from gluomem_bench.bench.parallel_campaign import _atomic_write

    _atomic_write(Path(path), value)


def _profiles(spec):
    from gluomem_bench.bench.rack import RackProfile

    return {name: RackProfile.model_validate(value) for name, value in spec["profiles"].items()}


def build_spec(
    *,
    dataset,
    judge_source,
    profiles,
    sample_ids,
    systems,
    output_root,
    authority,
    job_call_budget,
    job_seconds_budget,
    evidence_token_budget=4096,
    execution_kind="operator-authorized",
    embedding_url=None,
    embedding_recipe=None,
    cognee_python=None,
    config_overrides=None,
    study_protocol=PROTOCOL,
):
    """Freeze caller-selected IDs; never select on gold/outcomes or create an approval."""
    from gluomem_bench.bench.comprehensive_driver import frozen_inventory
    from gluomem_bench.bench.lme_projection import PROJECTION_POLICY

    root, source, judge = (
        Path(output_root).resolve(),
        Path(dataset).resolve(),
        Path(judge_source).resolve(),
    )
    if study_protocol not in {PROTOCOL, POOL_PROTOCOL}:
        raise ValueError("explicit supported provider study version required")
    pool = study_protocol == POOL_PROTOCOL
    spec = {
        "protocol": study_protocol,
        "dataset": {"name": "longmemeval_s", "path": str(source), "sha256": _sha(source)},
        "judge_source": str(judge),
        "profiles": profiles,
        "sample_ids": list(sample_ids),
        "systems": list(systems),
        "output_root": str(root),
        "authority": authority,
        "execution_kind": execution_kind,
        "source_projection": PROJECTION_POLICY,
        "evidence_token_budget": evidence_token_budget,
        "embedding_url": embedding_url,
        "embedding_recipe": embedding_recipe,
        "cognee_python": cognee_python,
        "config_overrides": config_overrides or {},
        "unsupported_cells": POOL_UNSUPPORTED if pool else UNSUPPORTED,
        "jobs": [],
    }
    for track in ("deepseek",) if pool else ("qwen", "deepseek"):
        for system in systems:
            for index, sample_id in enumerate(sample_ids, 1):
                job_id = f"{track}-{system}-{index:06d}"
                spec["jobs"].append(
                    {
                        "job_id": job_id,
                        "track": track,
                        "system": system,
                        "history_id": sample_id,
                        "sample_id": sample_id,
                        "output": str(root / "jobs" / job_id),
                        "call_budget": job_call_budget,
                        "seconds_budget": job_seconds_budget,
                    }
                )
    validate_spec(spec)
    spec["files"] = frozen_inventory(source, judge)
    if cognee_python:
        spec["files"].append(
            {"path": str(Path(cognee_python).resolve()), "sha256": _sha(cognee_python)}
        )
    return spec


def validate_spec(spec, *, judge_source=None):
    from urllib.parse import urlsplit

    from gluomem_bench.bench.lme_projection import PROJECTION_POLICY
    from gluomem_bench.bench.parallel_campaign import CampaignPolicy, Job
    from gluomem_bench.bench.upstream_scoring import load_longmemeval_prompt

    pool = spec.get("protocol") == POOL_PROTOCOL
    if (
        spec.get("protocol") not in {PROTOCOL, POOL_PROTOCOL}
        or spec["dataset"]["name"] != "longmemeval_s"
    ):
        raise ValueError("explicit versioned LongMemEval-S provider study required")
    authority = spec["authority"]
    authorities = (
        {POOL_QUALIFICATION, "provider-development-authority@v10"}
        if pool
        else {QUALIFICATION, "provider-development-authority@v8"}
    )
    if authority["protocol"] not in authorities:
        raise ValueError("new provider study authority required; legacy permits remain closed")
    if authority["retries"] != 0:
        raise ValueError("provider study forbids retries")
    cap, rate = money(authority["max_usd"]), money(authority["max_rate_per_million"])
    if authority["protocol"] in {QUALIFICATION, POOL_QUALIFICATION} and (
        authority["max_calls"] > 128 or authority["max_seconds"] > 7200 or cap > Decimal("5")
    ):
        raise ValueError("qualification exceeds original 128/$5/7200 authority")
    if rate < Decimal("0.39437"):
        raise ValueError("paid worst-price reservation below frozen endpoint snapshot")
    if pool and (
        rate != Decimal("1.2") or money(authority["max_prompt_rate_per_million"]) != Decimal("0.3")
    ):
        raise ValueError("v10 requires frozen componentwise worst-price reservation rates")
    CampaignPolicy(**policy_for(spec))
    if not Path(authority["claim_path"]).is_absolute():
        raise ValueError("absolute one-use authority claim path required")
    root = Path(spec["output_root"]).resolve()
    project = Path(__file__).resolve().parents[2]
    if any(root.is_relative_to(p) for p in (project, project.parent / "GluoMem")):
        raise ValueError("external provider study output required")
    if (
        spec["source_projection"] != PROJECTION_POLICY
        or spec["unsupported_cells"] != (POOL_UNSUPPORTED if pool else UNSUPPORTED)
        or not spec["systems"]
        or len(set(spec["systems"])) != len(spec["systems"])
        or not set(spec["systems"]) <= set(POOL_SUPPORTED if pool else SUPPORTED)
        or not spec["sample_ids"]
        or len(set(spec["sample_ids"])) != len(spec["sample_ids"])
    ):
        raise ValueError("unsupported cells/projection or duplicate study selection")
    if spec["execution_kind"] not in {"authored-loopback", "operator-authorized"}:
        raise ValueError("explicit generation provenance required")
    if pool and spec.get("embedding_url") and spec["execution_kind"] == "authored-loopback":
        if urlsplit(spec["embedding_url"]).hostname not in {"127.0.0.1", "::1"}:
            raise ValueError("authored fixture refuses non-loopback encoder endpoint")
    profiles = _profiles(spec)
    if pool and set(profiles) != {"deepseek"}:
        raise ValueError("v10 admits one explicit DeepSeek track only; Qwen remains failed")
    if not pool and set(profiles) != {"qwen", "deepseek"}:
        raise ValueError("both explicit matched model tracks required")
    common = None
    for name, profile in profiles.items():
        if pool:
            from gluomem_bench.bench.rack_provider import POOL_VERSION

            judge = profile.judge_profile
            if (
                profile.profile_version != POOL_VERSION
                or profile.provider != "openrouter"
                or judge is None
                or judge.profile_version != POOL_VERSION
                or judge.judge_profile is not None
                or judge.production_json_transport is not None
                or judge.consolidation_contract is not None
                or judge.extraction_contract is not None
                or profile.production_json_transport is not None
                or profile.extraction_contract != "strict-source-quotes-v1"
                or profile.consolidation_contract != "evolving-per-claim-v1"
            ):
                raise ValueError(
                    "v10 strict native primary and separate flat common pool judge required"
                )
            declared = profile.openrouter_pool
            if (
                tuple(e.tag for e in declared.endpoints) != POOL_TAGS
                or declared.max_attempts_per_logical_call != 6
                or declared.logical_deadline_seconds != 300
            ):
                raise ValueError(
                    "v10 ordered standard FP8 pool and original 6-attempt/300s bound required"
                )
            for endpoint in declared.endpoints:
                if Decimal(endpoint.pricing.prompt) > Decimal("0.0000003") or Decimal(
                    endpoint.pricing.completion
                ) > Decimal("0.0000012"):
                    raise ValueError("v10 endpoint price exceeds frozen componentwise allowance")
        elif (
            profile.profile_version != "provider-roles-v8"
            or profile.judge_profile is None
            or profile.provider != ("qwen-vllm" if name == "qwen" else "openrouter-baidu-fp8")
        ):
            raise ValueError("both V8 primary profiles and separate common judge required")
        judge = profile.judge_profile
        if common is not None and judge.model_dump() != common:
            raise ValueError("exact common DeepSeek judge policy differs between tracks")
        common = judge.model_dump()
        for p in (profile, judge):
            if spec["execution_kind"] == "authored-loopback" and urlsplit(
                p.base_url
            ).hostname not in {"127.0.0.1", "::1"}:
                raise ValueError("authored fixture refuses non-loopback provider endpoint")
    if any(s in {"gluomem", "mem0", "cognee", "dense", "hybrid"} for s in spec["systems"]):
        if not spec.get("embedding_url") or not spec.get("embedding_recipe"):
            raise ValueError("common explicit shared encoder required for native/dense systems")
    if "cognee" in spec["systems"] and not spec.get("cognee_python"):
        raise ValueError("actual isolated Cognee interpreter required")
    ids = [r["job_id"] for r in spec["jobs"]]
    matrix = [(r["track"], r["system"], r["sample_id"]) for r in spec["jobs"]]
    wanted = [
        (t, s, q)
        for t in (("deepseek",) if pool else ("qwen", "deepseek"))
        for s in spec["systems"]
        for q in spec["sample_ids"]
    ]
    if matrix != wanted or len(set(ids)) != len(ids):
        raise ValueError("exact provider study matrix required")
    for r in spec["jobs"]:
        Job(r["job_id"], r["history_id"], Path(r["output"]), r["call_budget"], r["seconds_budget"])
        if (
            Path(r["output"]).resolve() != root / "jobs" / r["job_id"]
            or r["history_id"] != r["sample_id"]
            or r["call_budget"] > authority["max_calls"]
            or r["seconds_budget"] >= authority["max_seconds"]
        ):
            raise ValueError("provider job escapes original authority")
    if spec["evidence_token_budget"] is not None and (
        type(spec["evidence_token_budget"]) is not int or spec["evidence_token_budget"] < 1
    ):
        raise ValueError("positive evidence ceiling required")
    if pool:
        if spec["evidence_token_budget"] is None:
            raise ValueError(
                "v10 requires an explicit replayable evidence ceiling; full context remains exempt"
            )
        from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe, RemoteEmbedder

        encoder = (
            RemoteEmbedder(
                spec["embedding_url"], recipe=EmbeddingRecipe(**spec["embedding_recipe"])
            )
            if spec.get("embedding_recipe")
            else None
        )
        for profile in profiles.values():
            config_for(spec, profile, embedder=encoder)
    load_longmemeval_prompt(Path(judge_source or spec["judge_source"]))
    return spec


def policy_for(spec):
    a = spec["authority"]
    return {
        k: a[k]
        for k in (
            "max_workers",
            "max_in_flight",
            "max_in_flight_tokens",
            "max_calls",
            "max_seconds",
        )
    } | {"stop_on_failure": True}


def verify_admission(spec, raw, approval):
    import hashlib

    from gluomem_bench.bench.comprehensive_driver import frozen_inventory

    validate_spec(spec)
    if (
        approval.get("approved") is not True
        or approval.get("reviewer_kind") not in {"human", "AI-assisted"}
        or approval.get("spec_sha256") != hashlib.sha256(raw).hexdigest()
        or approval.get("authority_protocol") != spec["authority"]["protocol"]
        or (
            spec["execution_kind"] == "authored-loopback"
            and approval.get("fixture_only") is not True
        )
        or (
            spec["execution_kind"] == "operator-authorized"
            and approval.get("fixture_only") is not False
        )
    ):
        raise ValueError("exact independent provider study approval required")
    expected = frozen_inventory(spec["dataset"]["path"], spec["judge_source"])
    if spec.get("cognee_python"):
        expected.append(
            {
                "path": str(Path(spec["cognee_python"]).resolve()),
                "sha256": _sha(spec["cognee_python"]),
            }
        )
    if spec.get("files") != expected:
        raise ValueError("complete provider runtime frozen inventory differs")
    if Path(spec["output_root"]).exists() or Path(spec["authority"]["claim_path"]).exists():
        raise FileExistsError("fresh root and unused original provider authority required")
    return spec


def config_for(spec, profile, *, data=None, embedder=None):
    from gluomem_bench.bench.provider_study_evidence import validate_config
    from gluomem_bench.bench.rack import controlled_config

    cfg = controlled_config(
        profile,
        data=Path(data or spec["dataset"]["path"]),
        sha256=spec["dataset"]["sha256"],
        embedder=embedder,
    )
    cfg = cfg.with_overrides(
        {
            "benchmark.datasets.longmemeval_s": {
                "url": "",
                "file": str(data or spec["dataset"]["path"]),
                "sha256": spec["dataset"]["sha256"],
            },
            **spec["config_overrides"],
        }
    )
    if spec.get("protocol") == POOL_PROTOCOL:
        dataset = cfg.model_dump(mode="json")["benchmark"]["datasets"]["longmemeval_s"]
        if (
            dataset.get("file") != str(data or spec["dataset"]["path"])
            or dataset.get("sha256") != spec["dataset"]["sha256"]
            or dataset.get("url") != ""
        ):
            raise ValueError("v10 effective dataset escapes the frozen complete source")
    validate_config(cfg.model_dump(mode="json"), profile)
    return cfg


def retriever_factory_for(spec, record, *, profile, context, encoder):
    """Only existing native graph/raw factories; never replace native algorithms."""
    system, out = record["system"], Path(record["output"])
    if system == "cognee":
        import time

        from gluomem_bench.bench.cognee_retriever import CogneeRetrieverFactory
        from gluomem_bench.bench.comparison_backends import CogneeSettings

        deadline = time.monotonic() + context.remaining()
        settings = CogneeSettings(
            spec["cognee_python"],
            out / "graph/native",
            "lme-provider-study",
            profile.base_url,
            profile.model,
            "guard-owned-placeholder",
            encoder.base_url,
            encoder.recipe.model,
            encoder.dim,
            "local-encoder-placeholder",
            max_completion_tokens=profile.output_budget("cognify"),
            timeout_seconds=context.remaining(),
            request_timeout_seconds=profile.timeout_seconds,
            operation_deadline=deadline,
            native_llm_timeout_seconds=min(profile.timeout_seconds, context.remaining()),
            cleanup_reserve_seconds=min(5, context.remaining() / 10),
        )
        return CogneeRetrieverFactory(
            settings=settings, profile=profile, outer_budget=context, out=out / "graph"
        )
    if system in {"bm25", "dense", "hybrid"}:
        from dataclasses import asdict

        from gluomem_bench.bench.evidence_retriever import RawCampaignRetriever

        def factory(cfg, conv, answer):
            retriever = RawCampaignRetriever(
                cfg, conv, answer, mode=system, embedder=encoder, chunk_chars=4096
            )
            _save(out / "raw-backend.json", retriever.manifest_extra())
            _save(out / "raw-inputs.json", [asdict(item) for item in retriever.chunks])
            return retriever

        return factory
    return None


def check_publication_deadline(budget):
    """Finalization shares the original campaign clock, including root readback."""
    if budget.clock() >= budget.deadline:
        from gluomem_bench.bench.parallel_campaign import CampaignStopped

        budget.failed = True
        raise CampaignStopped("original campaign finalization deadline exhausted")


def execute_spec(spec_path, approval_path, spec):
    import hashlib
    from dataclasses import asdict
    from types import SimpleNamespace

    from gluomem_bench.bench.comprehensive_driver import preserve_frozen_inputs
    from gluomem_bench.bench.embedding_runtime import (
        EmbeddingRecipe,
        RemoteEmbedder,
        safe_failure_diagnostic,
    )
    from gluomem_bench.bench.lme_projection import ExactLongMemEvalAdapter
    from gluomem_bench.bench.parallel_campaign import CampaignBudget, CampaignPolicy, Job, run_jobs
    from gluomem_bench.bench.suite_campaign import run_history

    pool = spec.get("protocol") == POOL_PROTOCOL
    if pool:
        raw, approved_raw = Path(spec_path).read_bytes(), Path(approval_path).read_bytes()
        if json.loads(raw) != spec:
            raise ValueError("v10 exact admitted execution specification differs")
        verify_admission(spec, raw, json.loads(approved_raw))
    root = Path(spec["output_root"])
    budget = CampaignBudget(
        max_calls=spec["authority"]["max_calls"], max_seconds=spec["authority"]["max_seconds"]
    )
    raw = raw if pool else Path(spec_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    claim = Path(spec["authority"]["claim_path"])
    claim.parent.mkdir(parents=True, exist_ok=True)
    with claim.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "protocol": spec["authority"]["protocol"],
                "spec_sha256": digest,
                "output_root": str(root),
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    root.mkdir(parents=True, exist_ok=False)
    (root / "execution-spec.json").write_bytes(raw)
    (root / "independent-approval.json").write_bytes(
        approved_raw if pool else Path(approval_path).read_bytes()
    )
    currency = CurrencyReservations(
        max_usd=spec["authority"]["max_usd"],
        max_rate_per_million=spec["authority"]["max_rate_per_million"],
        receipt=root / "currency.jsonl",
        max_prompt_rate_per_million=spec["authority"].get("max_prompt_rate_per_million")
        if pool
        else None,
    )
    rows = [
        {"job_id": r["job_id"], "history_id": r["history_id"], "status": "unrun", "calls": 0}
        for r in spec["jobs"]
    ]
    result = {
        "protocol": spec["protocol"],
        "complete": False,
        "spec_sha256": digest,
        "jobs": rows,
        "scope": "LongMemEval-S matched provider study only",
        "unsupported_cells": spec["unsupported_cells"],
        "completion_authority": "root-status.json",
        "execution_kind": spec["execution_kind"],
    }
    _save(root / "root-status.json", {**result, "publication": "running"})
    try:
        result["frozen_inputs"] = preserve_frozen_inputs(spec, root)
        encoder = (
            RemoteEmbedder(
                spec["embedding_url"], recipe=EmbeddingRecipe(**spec["embedding_recipe"])
            )
            if spec["embedding_recipe"]
            else None
        )
        if "mem0" in spec["systems"]:
            if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"):
                raise ValueError(
                    "native Mem0 requires explicit private iteration credential variable, "
                    "not ambient SDK keys"
                )
            os.environ["MEM0_TELEMETRY"] = "False"
            os.environ["MEM0_DIR"] = str(root / "native-mem0-home")
        profiles = _profiles(spec)
        configs = {name: config_for(spec, p, embedder=encoder) for name, p in profiles.items()}
        adapter = ExactLongMemEvalAdapter(configs["deepseek" if pool else "qwen"], "longmemeval_s")
        adapter.verify()
        samples = {s.id: s for s in adapter.load()}
        if not set(spec["sample_ids"]) <= samples.keys():
            raise ValueError("selected IDs differ from pinned LongMemEval-S asset")
        convs = {c.conv_id: c for c in adapter.conversations() if c.conv_id in spec["sample_ids"]}
        selected = SimpleNamespace(
            name=adapter.name,
            sha=adapter.sha,
            path=adapter.path,
            verify=adapter.verify,
            load=lambda: [samples[q] for q in spec["sample_ids"]],
            conversations=lambda: list(convs.values()),
        )
        bindings = {r["job_id"]: r for r in spec["jobs"]}
        jobs = [
            Job(r["job_id"], r["history_id"], r["output"], r["call_budget"], r["seconds_budget"])
            for r in spec["jobs"]
        ]

        def run_job(job, original_context):
            r = bindings[job.job_id]
            context = CostedJobContext(original_context, currency)
            profile, cfg, sample = (
                profiles[r["track"]],
                configs[r["track"]],
                samples[r["sample_id"]],
            )
            factory = retriever_factory_for(
                spec, r, profile=profile, context=context, encoder=encoder
            )
            result = run_history(
                cfg=cfg,
                adapter=selected,
                samples=[sample],
                profile=profile,
                out=job.output,
                system=r["system"],
                judge_source=Path(spec["judge_source"]),
                embedder=encoder,
                outer_budget=context,
                evidence_token_budget=spec["evidence_token_budget"],
                retriever_factory=factory,
            )
            return {
                "quality_status": "complete_verified",
                "summary": result,
                "sample_sha256": hashlib.sha256(
                    json.dumps(asdict(sample), sort_keys=True).encode()
                ).hexdigest(),
            }

        with capture_pool_physical() if pool else nullcontext():
            coordinated = run_jobs(
                jobs,
                policy=CampaignPolicy(**policy_for(spec)),
                out=root / "coordination",
                run_job=run_job,
                budget=budget,
            )
        result["jobs"] = coordinated["jobs"]
    except Exception as exc:
        budget.failed = True
        result["failure"] = safe_failure_diagnostic(exc)
        progress = root / "coordination/progress.json"
        if progress.exists():
            result["jobs"] = json.loads(progress.read_bytes())["jobs"]
        for row in result["jobs"]:
            if row["status"] in {"queued", "running"}:
                row["status"] = "failed" if row.get("calls") else "unrun"
    result["accounting"] = budget.snapshot()
    result["currency"] = currency.snapshot()
    for row, record in zip(result["jobs"], spec["jobs"], strict=True):
        row.setdefault("calls", 0)
        leaf = Path(record["output"])
        row["artifact_sha256"] = {
            p.relative_to(root).as_posix(): _sha(p) for p in leaf.rglob("*") if p.is_file()
        }
    result["complete"] = all(
        r["status"] == "complete"
        and r.get("result", {}).get("quality_status") == "complete_verified"
        for r in result["jobs"]
    )
    result["publication"] = "verified"
    pending = root / "root.pending.json"
    try:
        _save(pending, result)
        check_publication_deadline(budget)
        verify_root(root, candidate_path=pending)
        check_publication_deadline(budget)
        result["accounting"] = budget.snapshot()
        _save(pending, result)
        _save(root / "campaign-result.json", result)
        pending.replace(root / "root-status.json")
        if json.loads((root / "root-status.json").read_bytes()) != result:
            raise ValueError("root publication readback differs")
        check_publication_deadline(budget)
    except BaseException:
        # A complete leaf never outranks failed root publication.
        result.update(complete=False, publication="incomplete")
        try:
            _save(root / "root-incomplete.json", result)
        finally:
            (root / "root-status.json").unlink(missing_ok=True)
        raise
    return result


def verify_root(root, *, candidate_path=None):
    import hashlib
    from dataclasses import asdict

    from gluomem_bench.bench.artifacts import verify_artifact_hashes
    from gluomem_bench.bench.lme_projection import ExactLongMemEvalAdapter
    from gluomem_bench.bench.provider_study_evidence import read_json, read_lines
    from gluomem_bench.bench.suite_campaign import verify_history

    root = Path(root)
    if (root / "root-incomplete.json").exists():
        raise ValueError("root publication incomplete; leaf success is not authority")
    result = read_json(candidate_path or root / "root-status.json")
    spec_raw = (root / "execution-spec.json").read_bytes()
    spec, approval = json.loads(spec_raw), read_json(root / "independent-approval.json")
    pool = spec.get("protocol") == POOL_PROTOCOL
    if (
        spec.get("protocol") not in {PROTOCOL, POOL_PROTOCOL}
        or result.get("protocol") != spec.get("protocol")
        or result.get("spec_sha256") != hashlib.sha256(spec_raw).hexdigest()
        or approval.get("spec_sha256") != result["spec_sha256"]
        or approval.get("approved") is not True
        or result.get("publication") != "verified"
        or result.get("unsupported_cells") != spec["unsupported_cells"]
        or [r["job_id"] for r in result["jobs"]] != [r["job_id"] for r in spec["jobs"]]
    ):
        raise ValueError("root authority/matrix/publication differs")
    snapshots = {}
    for record in result.get("frozen_inputs", []):
        target = root / record["snapshot"]
        if not target.resolve().is_relative_to(root.resolve()) or _sha(target) != record["sha256"]:
            raise ValueError("root frozen input snapshot differs")
        snapshots[record["path"]] = target
    dataset, judge = snapshots.get(spec["dataset"]["path"]), snapshots.get(spec["judge_source"])
    if dataset is None or judge is None or _sha(dataset) != spec["dataset"]["sha256"]:
        raise ValueError("root native dataset/scorer snapshots missing")
    if pool:
        validate_spec(spec, judge_source=judge)
        if (
            approval.get("authority_protocol") != spec["authority"]["protocol"]
            or approval.get("reviewer_kind") not in {"human", "AI-assisted"}
            or approval.get("fixture_only") is not (spec["execution_kind"] == "authored-loopback")
            or [{k: r[k] for k in ("path", "sha256")} for r in result["frozen_inputs"]]
            != spec["files"]
            or len(snapshots) != len(spec["files"])
        ):
            raise ValueError("root exact approval/complete frozen runtime inventory differs")
    profiles = _profiles(spec)
    from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe, RemoteEmbedder

    encoder = (
        RemoteEmbedder(spec["embedding_url"], recipe=EmbeddingRecipe(**spec["embedding_recipe"]))
        if spec["embedding_recipe"]
        else None
    )
    cfg = config_for(spec, profiles["deepseek" if pool else "qwen"], data=dataset, embedder=encoder)
    adapter = ExactLongMemEvalAdapter(cfg, "longmemeval_s")
    samples = {s.id: s for s in adapter.load()}
    conversations = {c.conv_id: c for c in adapter.conversations()}
    total = 0
    for row, record in zip(result["jobs"], spec["jobs"], strict=True):
        if (
            row["status"] not in {"complete", "failed", "unrun", "cancelled"}
            or type(row.get("calls")) is not int
        ):
            raise ValueError("root failure/unrun coverage differs")
        verify_artifact_hashes(root, row["artifact_sha256"], canonical_names=True)
        total += row["calls"]
        if row["status"] == "complete":
            leaf = Path(record["output"])
            expected = {
                "dataset": "longmemeval_s",
                "system": record["system"],
                "sample_ids": [record["sample_id"]],
            }
            active_profile = profiles[record["track"]]
            active_cfg = config_for(spec, active_profile, embedder=encoder)
            if read_json(leaf / "effective-config.json") != active_cfg.model_dump(mode="json"):
                raise ValueError(
                    "root effective configuration differs from frozen profile/settings"
                )
            if pool:
                from gluomem_bench.bench.provider_study_evidence import (
                    source_projection,
                    validate_pool_source,
                )

                conv = conversations[record["sample_id"]]
                if read_json(leaf / "source-projection.json") != source_projection(conv):
                    raise ValueError("root complete source roles/dates/text/opaque IDs differ")
                validate_pool_source(leaf, record["system"], conv)
            if record["system"] == "gluomem":
                from gluomem_bench.bench.rack_schema import production_expectation
                from gluomem_bench.bench.suite_campaign import make_source_events

                expected.update(production_expectation("gluomem", active_profile, active_cfg))
                source = [
                    e.model_copy(
                        update={
                            "capture_policy": "attributed-interaction",
                            "created_at": e.event_time,
                        }
                    )
                    for e in make_source_events(conversations[record["sample_id"]], "longmemeval_s")
                ]
                if read_json(leaf / "source-events.json") != [
                    e.model_dump(mode="json") for e in source
                ]:
                    raise ValueError("root exact allowed source roles/text/opaque IDs/clock differ")
            if record["system"] == "mem0":
                import ast

                from gluomem_bench.bench.mem0 import _mem0_digest, _session_messages

                pins = [
                    v
                    for k, v in snapshots.items()
                    if k.replace("\\", "/").endswith("/mem0/configs/prompts.py")
                ]
                if len(pins) != 1:
                    raise ValueError("unique frozen native Mem0 prompt source required")
                tree = ast.parse(pins[0].read_bytes())
                literals = [
                    ast.literal_eval(n.value)
                    for n in tree.body
                    if isinstance(n, ast.Assign)
                    and any(
                        isinstance(t, ast.Name) and t.id == "ADDITIVE_EXTRACTION_PROMPT"
                        for t in n.targets
                    )
                ]
                if len(literals) != 1 or not isinstance(literals[0], str):
                    raise ValueError("native Mem0 prompt literal differs")
                original = [
                    {
                        "session_index": s.index,
                        "session_date": s.date_time.isoformat(),
                        "messages": _session_messages(s),
                    }
                    for s in conversations[record["sample_id"]].sessions
                ]
                expected.update(
                    mem0_source_inputs_sha256=_mem0_digest(original),
                    mem0_extractor_system_sha256=hashlib.sha256(literals[0].encode()).hexdigest(),
                    profile_sha256=hashlib.sha256(
                        active_profile.model_dump_json().encode()
                    ).hexdigest(),
                    dataset_sha256=spec["dataset"]["sha256"],
                )
            if record["system"] == "cognee":
                from gluomem_bench.bench.cognee_retriever import session_documents

                if read_json(leaf / "graph/native-ledger.json")["source_inputs"] != [
                    asdict(d) for d in session_documents(conversations[record["sample_id"]])
                ]:
                    raise ValueError(
                        "root native Cognee source roles/dates/whole-history coverage differs"
                    )
            summary = verify_history(
                leaf, expected_job=expected, require_real=True, judge_source=judge
            )
            if (
                summary != row["result"]["summary"]
                or read_json(leaf / "usage.json")["generation_calls"] != row["calls"]
                or read_json(leaf / "predictions.json")[0]["sample"]
                != json.loads(json.dumps(asdict(samples[record["sample_id"]])))
            ):
                raise ValueError("root native sample/summary/accounting differs")
        elif row["status"] in {"unrun", "cancelled"} and row["calls"] != 0:
            raise ValueError("unrun row contains attempted spend")
    currency_rows = read_lines(root / "currency.jsonl")
    reserved = [r for r in currency_rows if r["kind"] == "reservation"]
    intents = [r for r in currency_rows if r["kind"] == "dispatch-intent"]
    if (
        len(intents) != total
        or total != result["accounting"]["calls"]
        or [r["campaign_call_ordinal"] for r in intents] != list(range(1, total + 1))
        or len({r["reservation_id"] for r in intents}) != total
    ):
        raise ValueError("root all-attempt accounting differs from predispatch intents")
    cumulative = Decimal(0)
    for index, row in enumerate(reserved, 1):
        value = (
            money(spec["authority"]["max_rate_per_million"])
            * Decimal(row["tokens"])
            / Decimal(1000000)
        )
        if pool:
            value = (
                money(spec["authority"]["max_prompt_rate_per_million"])
                * row["prompt_reserved_tokens"]
                + money(spec["authority"]["max_rate_per_million"]) * row["output_reserved_tokens"]
            ) / Decimal(1000000)
        cumulative += value
        if (
            row["reservation_id"] != index
            or Decimal(row["reserved_usd"]) != value
            or Decimal(row["cumulative_reserved_usd"]) != cumulative
        ):
            raise ValueError("root paid reservation arithmetic differs")
    if (
        cumulative > money(spec["authority"]["max_usd"])
        or str(cumulative) != result["currency"]["reserved_usd"]
        or result["currency"]["billed_usd"] is not None
    ):
        raise ValueError("root paid cost reservation evidence differs")
    complete = all(
        r["status"] == "complete"
        and r.get("result", {}).get("quality_status") == "complete_verified"
        for r in result["jobs"]
    )
    if result["complete"] != complete:
        raise ValueError("root completion differs from exact verified matrix")
    if pool:
        verify_pool_currency(root, spec, result, currency_rows, profiles["deepseek"], cfg, judge)
    return result


def verify_pool_currency(root, spec, result, ledger, profile, cfg, judge):
    """Independent root physical receipts, clocks, reservations and pool-history replay."""
    import copy
    import math

    from gluomem_bench.bench.provider_study_evidence import pool_bundle_calls, read_json
    from gluomem_bench.bench.rack_pool_replay import validate_pool_history
    from gluomem_bench.bench.rack_provider import estimated_prompt_tokens
    from gluomem_bench.bench.rack_schema import production_expectation

    kinds = {"reservation", "dispatch-intent", "physical-send", "physical-response"}
    if any(r.get("kind") not in kinds for r in ledger):
        raise ValueError("unknown pool physical journal row")
    reservations = {r["reservation_id"]: r for r in ledger if r["kind"] == "reservation"}
    intents = [r for r in ledger if r["kind"] == "dispatch-intent"]
    sends = [r for r in ledger if r["kind"] == "physical-send"]
    responses = [r for r in ledger if r["kind"] == "physical-response"]

    def unique(rows, key):
        values = {r[key]: r for r in rows}
        if len(values) != len(rows):
            raise ValueError("duplicate pool physical journal identity")
        return values

    sent_by = unique(sends, "campaign_call_ordinal")
    received_by = unique(responses, "campaign_call_ordinal")
    if set(received_by) != set(sent_by) or len(sends) != len(intents):
        raise ValueError("pool physical send/receive/intent coverage differs")
    first = sends[0] if sends else None
    deadlines = {}
    for intent in intents:
        ordinal = intent["campaign_call_ordinal"]
        sent, received, reserved = (
            sent_by[ordinal],
            received_by[ordinal],
            reservations[intent["reservation_id"]],
        )
        record = next(r for r in spec["jobs"] if r["job_id"] == intent["job_id"])
        request = sent["request"]
        prompt = estimated_prompt_tokens(request["messages"]) + profile.schema_reserve_tokens
        output = request["max_tokens"]
        endpoint = next(
            (
                e.model_dump(mode="json")
                for e in profile.openrouter_pool.endpoints
                if [e.tag] == request.get("provider", {}).get("only")
            ),
            None,
        )
        immutable = (
            "reservation_id",
            "job_id",
            "campaign_call_ordinal",
            "call_ordinal",
            "intent",
            "request",
            "endpoint",
            "logical_deadline",
            "job_deadline",
            "campaign_deadline",
            "campaign_started",
            "first_dispatch_monotonic",
            "first_dispatch_wall",
            "started_monotonic",
            "timeout_seconds",
        )
        numbers = [
            sent[k]
            for k in (
                "started_monotonic",
                "first_dispatch_monotonic",
                "first_dispatch_wall",
                "logical_deadline",
                "job_deadline",
                "campaign_deadline",
                "campaign_started",
                "timeout_seconds",
            )
        ]
        if (
            any(type(v) not in (int, float) or not math.isfinite(v) for v in numbers)
            or any(sent[k] != received.get(k) for k in immutable)
            or any(sent[k] != intent[k] for k in ("reservation_id", "job_id", "call_ordinal"))
            or reserved.get("intent") != sent["intent"]
            or reserved.get("endpoint") != endpoint
            or sent["endpoint"] != endpoint
            or reserved["prompt_reserved_tokens"] != prompt
            or reserved["output_reserved_tokens"] != output
            or reserved["tokens"] != prompt + output
            or reserved.get("logical_deadline") != sent["logical_deadline"]
            or not sent["campaign_started"]
            <= sent["first_dispatch_monotonic"]
            <= sent["started_monotonic"]
            or sent["first_dispatch_monotonic"] != first["first_dispatch_monotonic"]
            or sent["first_dispatch_wall"] != first["first_dispatch_wall"]
            or sent["campaign_started"] != first["campaign_started"]
            or sent["campaign_deadline"]
            != sent["campaign_started"] + spec["authority"]["max_seconds"]
            or not 0
            < sent["timeout_seconds"]
            <= min(
                profile.timeout_seconds,
                sent["job_deadline"] - sent["started_monotonic"],
                sent["logical_deadline"] - sent["started_monotonic"],
                sent["campaign_deadline"] - sent["started_monotonic"],
            )
            + 0.001
            or sent["job_deadline"] - sent["started_monotonic"] > record["seconds_budget"]
            or deadlines.setdefault(record["job_id"], sent["job_deadline"]) != sent["job_deadline"]
        ):
            raise ValueError("pool physical currency/first-clock/original deadline binding differs")
        if result["complete"] and not sent["started_monotonic"] <= received[
            "completed_monotonic"
        ] < min(sent["job_deadline"], sent["logical_deadline"], sent["campaign_deadline"]):
            raise ValueError("pool physical completion exceeded original deadline")
        if received.get("actual_cost_usd") is not None:
            raise ValueError(
                "pool physical unknown actual cost cannot be replaced by a reservation"
            )
    if first and (
        result["currency"].get("first_dispatch_monotonic") != first["first_dispatch_monotonic"]
        or result["currency"].get("first_dispatch_wall") != first["first_dispatch_wall"]
    ):
        raise ValueError("pool physical first dispatch clock differs from shared currency")
    # Normalize local workload ordinals ONLY after independently binding them to
    # the journal's original campaign ordinals. Saved traces remain untouched.
    native, attempts, requests = [], [], {}
    for row, record in zip(result["jobs"], spec["jobs"], strict=True):
        if row["status"] != "complete":
            continue
        leaf = Path(record["output"])
        ns, ats, _, reqs = pool_bundle_calls(
            leaf,
            profile,
            read_json(leaf / "predictions.json"),
            read_json(leaf / "protocol.json"),
            judge_source=judge,
        )
        mapping = {}
        for attempt in ats:
            matches = [
                r
                for r in sends
                if r["job_id"] == record["job_id"]
                and r["call_ordinal"] == attempt["pool_attempt"]["ordinal"]
            ]
            if len(matches) != 1:
                raise ValueError("pool physical root/local ordinal coverage differs")
            sent = matches[0]
            ordinal = sent["campaign_call_ordinal"]
            received = received_by[ordinal]
            if (
                attempt["pool_attempt"].get("root_ordinal") != ordinal
                or attempt["pool_attempt"]["intent"] != sent["intent"]
                or any(attempt.get(k) != received.get(k) for k in ("request", "response", "status"))
            ):
                raise ValueError("pool physical root receipt differs from leaf attempt")
            mapping[attempt["pool_attempt"]["ordinal"]] = ordinal
            normalized = copy.deepcopy(attempt)
            normalized["pool_attempt"]["ordinal"] = ordinal
            attempts.append(normalized)
        for item in copy.deepcopy(ns):
            item["ordinal"] = mapping[item["ordinal"]]
            for attempt in item["pool_attempts"]:
                attempt["ordinal"] = mapping[attempt["ordinal"]]
            native.append(item)
        if set(requests) & set(reqs):
            raise ValueError("pool repeated shared logical invocation")
        requests.update(reqs)
    if result["complete"]:
        attempts.sort(key=lambda r: r["pool_attempt"]["ordinal"])
        binding = production_expectation("gluomem", profile, cfg)
        validate_pool_history(
            native,
            attempts,
            profile,
            cfg.model_dump(mode="json"),
            binding,
            binding,
            physical_rows=[received_by[i] for i in sorted(received_by)],
            expected_requests=requests,
        )
