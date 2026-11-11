"""Client-owned FP8 attempts at the existing lease/native SDK seams.

No account ledger lives here. Each attempt debits WorkloadBudget.begin_call and
its original outer authority. Wire capture composes with the operator's existing
physical capture; it neither admits nor refunds anything.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from email.utils import parsedate_to_datetime

from gluomem_bench.bench.rack_provider import pool_attempt_profile

_CURRENT = ContextVar("rack_fp8_pool_attempt", default=None)
_NATIVE_LOCK = threading.RLock()
_STATE_LOCK = threading.RLock()
RETRY_STATUSES = frozenset({429, 502, 503, 504})


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def current_pool_attempt():
    """Invocation-local binding for an operator's physical pre-send checks."""
    return _CURRENT.get()


def pool_key(profile):
    return digest(
        {
            "base_url": profile.base_url,
            "model": profile.model,
            "pool": profile.openrouter_pool.model_dump(mode="json"),
            **({'pricing_policy': profile.pricing_policy.model_dump(mode='json')}
               if profile.pricing_policy is not None else {}),
        }
    )


def _authority(budget):
    seen = set()
    while id(budget) not in seen:
        seen.add(id(budget))
        parent = getattr(budget, "outer_budget", None) or getattr(budget, "root", None)
        if parent is None:
            return budget
        budget = parent
    raise ValueError("cyclic pool allowance")


class PoolState:
    def __init__(self):
        self.logical_calls = 0
        self.cooldowns = {}
        self.failures = {}
        self.lock = threading.RLock()


def pool_state(budget, profile):
    if budget is None:
        raise ValueError("pool requires the original shared workload allowance")
    authority = _authority(budget)
    with _STATE_LOCK:
        if not hasattr(authority, "_rack_fp8_pools"):
            authority._rack_fp8_pools = {}
        return authority._rack_fp8_pools.setdefault(pool_key(profile), PoolState())


def retryable_wire(wire, profile=None):
    """Positive HTTP evidence only; never infer capacity from error message prose."""
    if type(wire) is not dict or wire.get("status") not in RETRY_STATUSES:
        return False
    body = wire.get("response")
    error = body.get("error") if type(body) is dict else None
    if type(error) is not dict:
        return False
    if profile is not None and any(
        key in body and body[key] != expected
        for key, expected in (
            ("model", profile.model),
            ("provider", profile.openrouter_endpoint.provider_name),
            ("quantization", profile.openrouter_endpoint.quantization),
        )
    ):
        return False
    metadata = error.get("metadata")
    if (
        type(metadata) is dict
        and "limit_source" in metadata
        and metadata["limit_source"]
        not in {
            "upstream_provider_shared_pool",
            "tpm_rate_limit_exceeded",
            "rpm_rate_limit_exceeded",
            "provider_capacity",
            "rate_limit_exceeded",
        }
    ):
        return False
    code = error.get("code")
    # Unknown codes stay closed, including auth, credit, billing and quota errors
    # wrapped as HTTP429. Numeric status or explicit capacity names only.
    return (type(code) is int and code == wire["status"]) or (
        type(code) is str
        and code
        in {
            "rate_limit_exceeded",
            "temporarily_unavailable",
            "overloaded",
            "capacity_exceeded",
            "server_error",
            "bad_gateway",
            "gateway_timeout",
        }
    )


def wire_record(url, raw, status, body, headers):
    """Private selected trace: exact JSON body, never authentication headers."""
    row = {
        "url": url,
        "request": json.loads(raw),
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "status": status,
        "response_sha256": hashlib.sha256(body).hexdigest(),
    }
    try:
        row["response"] = json.loads(body)
    except ValueError:
        row["response_text"] = body.decode("utf-8", errors="replace")
    active = current_pool_attempt()
    if active is not None and active["profile"].profile_version == "observed-provider-v13":
        row["request_raw_utf8"] = raw.decode("utf-8")
        row["response_raw_utf8"] = body.decode("utf-8")
    # No arbitrary response header or exception-text sink.
    row["retry_after"] = next((v for k, v in headers if k.lower() == "retry-after"), None)
    row["received_wall"] = time.time()
    return row


def retry_after_seconds(value, *, wall=time.time):
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (ValueError, TypeError):
        try:
            number = max(0.0, parsedate_to_datetime(value).timestamp() - wall())
        except (ValueError, TypeError, OverflowError):
            raise ValueError("invalid Retry-After; retry refused") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError("invalid Retry-After; retry refused")
    return number


@contextmanager
def capture_native_attempt(attempt, seconds):
    """One fresh native connection; zero idle-socket re-sends and SDK retries.

    Synchronous-only qualified scope. Context-local interception leaves unrelated
    threads alone. The saved stream may be the operator's accounting wrapper.
    """
    from dspy._vendor.lm15.transports import StdlibTransport, TransportResponse

    with _NATIVE_LOCK:
        original = StdlibTransport.stream

        def stream(transport, request):
            active = current_pool_attempt()
            if active is None or active["attempt"] is not attempt:
                return original(transport, request)
            if "wire" in attempt:
                raise ValueError("unaccounted native SDK transport retry refused")
            owned = StdlibTransport(
                connect_timeout=seconds,
                read_timeout=seconds,
                write_timeout=seconds,
                pool_timeout=seconds,
                max_connections=1,
                trust_env=False,
            )
            request = replace(
                request, connect_timeout=seconds, read_timeout=seconds, write_timeout=seconds
            )
            timer = threading.Timer(seconds, owned.close)
            timer.daemon = True
            timer.start()
            chunks, status, headers = [], 0, []
            try:
                with original(owned, request) as response:
                    status, headers = response.status, response.headers
                    size = 0
                    for chunk in response:
                        chunks.append(chunk)
                        size += len(chunk)
                        if size > 8_000_000:
                            raise ValueError("pool response byte bound")
                    body = b"".join(chunks)
                    attempt["wire"] = wire_record(request.url, request.body, status, body, headers)
                    return TransportResponse(
                        status=status,
                        reason=response.reason,
                        headers=headers,
                        http_version=response.http_version,
                        chunks=iter([body]),
                        release=lambda consumed: None,
                    )
            finally:
                if "wire" not in attempt:
                    attempt["wire"] = wire_record(
                        request.url, request.body, status, b"".join(chunks), headers
                    )
                timer.cancel()
                owned.close()

        StdlibTransport.stream = stream
        try:
            yield
        finally:
            StdlibTransport.stream = original


def run_pool_call(client, task, tokens, output, row, dispatch):
    """Retry before the existing caller's final latch, never clear a failed flag."""
    profile, budget = client.profile, client.budget
    state, pool = pool_state(budget, profile), profile.openrouter_pool
    clock = getattr(budget, "clock", time.monotonic)
    started = clock()
    row.update(client.pricing_fields())
    deadline = started + min(pool.logical_deadline_seconds, budget.remaining())
    with state.lock:
        state.logical_calls += 1
        logical = state.logical_calls
        row.update(
            declared_pool=pool.model_dump(mode="json"),
            pool_key=pool_key(profile),
            logical_ordinal=logical,
            logical_started=started,
            logical_deadline=deadline,
            pool_attempts=[],
        )
        for index in range(pool.max_attempts_per_logical_call):
            client.check()
            if clock() >= deadline:
                raise TimeoutError("pool logical deadline exhausted")
            available = [e for e in pool.endpoints if state.cooldowns.get(e.tag, 0) <= clock()]
            if not available:
                delay = min(state.cooldowns.values()) - clock()
                if delay >= min(deadline - clock(), budget.remaining()):
                    raise TimeoutError("pool cooldown exceeds original deadline")
                time.sleep(delay)
                client.check()
                available = [e for e in pool.endpoints if state.cooldowns.get(e.tag, 0) <= clock()]
            endpoint = available[0]
            single = pool_attempt_profile(profile, endpoint.tag)
            attempt = {
                "attempt": index + 1,
                "logical_ordinal": logical,
                "endpoint": endpoint.model_dump(mode="json"),
                "pool_key": pool_key(profile),
                "intent": digest([pool_key(profile), logical, index + 1]),
                "logical_deadline": deadline,
            }
            context = {
                "profile": single,
                "endpoint_tag": endpoint.tag,
                "attempt": attempt,
                "logical_deadline": deadline,
                "pool_key": pool_key(profile),
            }
            if profile.pricing_policy is not None:
                attempt['pricing_observation'] = client.pricing_fields(
                    endpoint.tag)['pricing_observation']
            token = _CURRENT.set(context)
            error, result = None, None
            try:
                with budget.lease(tokens + output + profile.schema_reserve_tokens):
                    budget.begin_call()
                    row["pool_attempts"].append(attempt)
                    attempt["ordinal"] = budget.calls
                    row["ordinal"] = budget.calls
                    attempt["started_monotonic"] = clock()
                    seconds = min(
                        profile.timeout_seconds,
                        budget.remaining(),
                        deadline - attempt["started_monotonic"],
                    )
                    if seconds <= 0:
                        raise TimeoutError("pool logical deadline exhausted")
                    attempt["timeout_seconds"] = seconds
                    root_ordinal = getattr(getattr(budget, "outer_budget", None), "ordinal", None)
                    if type(root_ordinal) is int:
                        attempt["root_ordinal"] = root_ordinal
                    try:
                        result = dispatch(single, seconds, attempt)
                    except Exception as exc:
                        # Catch inside the lease: its formatter/failure latch must
                        # not turn an eligible infrastructure retry into job failure.
                        error = exc
                    finally:
                        attempt["completed_monotonic"] = clock()
            finally:
                _CURRENT.reset(token)
            if error is None:
                try:
                    budget.check()
                    if clock() >= deadline:
                        raise TimeoutError("pool logical deadline exhausted")
                except Exception:
                    attempt["outcome"] = "terminal-deadline"
                    persist_pool_attempt(client, task, row, attempt)
                    raise
                attempt["outcome"] = "success"
                persist_pool_attempt(client, task, row, attempt)
                return result, single
            retry = retryable_wire(attempt.get("wire"), single)
            attempt["outcome"] = "infrastructure-retry" if retry else "terminal-failure"
            attempt["error"] = {"type": "Exception", "message": "operation_failed"}
            if retry:
                state.failures[endpoint.tag] = state.failures.get(endpoint.tag, 0) + 1
                backoff = min(
                    pool.max_backoff_seconds,
                    pool.cooldown_seconds * 2 ** min(state.failures[endpoint.tag] - 1, 20),
                )
                jitter = min(
                    pool.max_backoff_seconds - backoff,
                    backoff * pool.jitter_fraction * random.random(),
                )
                try:
                    retry_after = retry_after_seconds(
                        attempt["wire"].get("retry_after"),
                        wall=lambda received=attempt["wire"]["received_wall"]: received,
                    )
                except ValueError:
                    attempt["outcome"] = "terminal-failure"
                    persist_pool_attempt(client, task, row, attempt)
                    raise
                state.cooldowns[endpoint.tag] = attempt["completed_monotonic"] + max(
                    retry_after, backoff + jitter
                )
                attempt.update(
                    retry_after_seconds=retry_after,
                    jitter_seconds=jitter,
                    cooldown_until=state.cooldowns[endpoint.tag],
                )
            if not retry or index + 1 == pool.max_attempts_per_logical_call:
                if retry:
                    attempt["outcome"] = "infrastructure-exhausted"
                persist_pool_attempt(client, task, row, attempt)
                raise error
            persist_pool_attempt(client, task, row, attempt)
    raise RuntimeError("pool exhausted")


def persist_pool_attempt(client, task, row, attempt):
    client.receipt.parent.mkdir(parents=True, exist_ok=True)
    wire = attempt.get("wire", {})
    record = {
        **wire,
        "method": "POST",
        "task": task,
        "pool_attempt": attempt,
        "declared_pool": row["declared_pool"],
        "pool_key": row["pool_key"],
        "logical_ordinal": row["logical_ordinal"],
        "logical_started": row["logical_started"],
        "logical_deadline": row["logical_deadline"],
        **({'pricing_evidence': row['pricing_evidence']} if 'pricing_evidence' in row else {}),
    }
    with client.receipt.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
