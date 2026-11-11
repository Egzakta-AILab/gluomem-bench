"""Opt-in local MLflow tracking with explicit client calls, never autologging.

Disabled tracking imports no optional SDK and preserves returns and exceptions.
The client API avoids MLflow's process-global active-run state.
"""

from __future__ import annotations

import inspect
import os
import time
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any

from gluomem_bench.tracking_identity import code_identities, experiment_params
from gluomem_bench.tracking_projection import STATUSES, digest, project_result
from gluomem_bench.tracking_receipts import aggregate_children, collect_record, metric_values
from gluomem_bench.tracking_server import default_root, validate_tracking_uri

_PARENT: ContextVar[Any] = ContextVar("gluomem_tracking_parent", default=None)
_CHILD: ContextVar[Any] = ContextVar("gluomem_tracking_child", default=None)


@dataclass
class _Campaign:
    client: Any
    experiment_id: str
    run_id: str
    family: str
    seen: set = field(default_factory=set)
    outcomes: list = field(default_factory=list)
    code: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    records: dict = field(default_factory=dict)


def _log_params(parent, run_id, values):
    from mlflow.entities import Param

    current = parent.params.setdefault(run_id, {})
    new = {key: str(value) for key, value in values.items() if key not in current}
    if new:
        entries = list(new.items())
        for offset in range(0, len(entries), 100):
            parent.client.log_batch(
                run_id, params=[Param(key, value) for key, value in entries[offset : offset + 100]]
            )
        current.update(new)


def _log_metrics(parent, run_id, values):
    from mlflow.entities import Metric

    entries = list(values.items())
    timestamp = int(time.time() * 1000)
    for offset in range(0, len(entries), 1000):
        parent.client.log_batch(
            run_id,
            metrics=[
                Metric(key, value, timestamp, 0) for key, value in entries[offset : offset + 1000]
            ],
        )


def _row_identity(value):
    if not isinstance(value, dict):
        return None
    if value.get("job_id") is not None:
        return digest(["job_id", str(value["job_id"])])
    for key in ("output", "path"):
        if isinstance(value.get(key), (str, Path)):
            return digest(["output", str(Path(value[key]).resolve())])
    for key in ("group_id", "task_id", "history_id", "question_id"):
        if value.get(key) is not None:
            return digest(
                [
                    key,
                    str(value[key]),
                    value.get("system", value.get("method")),
                    value.get("domain"),
                    value.get("tier"),
                ]
            )
    return None


def _bound_identities(bound):
    values = []
    job = bound.get("job")
    if getattr(job, "job_id", None) is not None:
        values.append(_row_identity({"job_id": job.job_id}))
    for key in ("out", "out_dir", "target"):
        if isinstance(bound.get(key), (str, Path)):
            values.append(_row_identity({"output": bound[key]}))
    for key in ("group", "task"):
        obj = bound.get(key)
        if getattr(obj, "id", None) is not None:
            values.append(
                _row_identity(
                    {
                        key + "_id": obj.id,
                        "system": bound.get("system", bound.get("method")),
                        "domain": getattr(obj, "domain", None),
                        "tier": getattr(obj, "tier", None),
                    }
                )
            )
    return [value for value in values if value is not None]


def _bound_identity(bound):
    return next(iter(_bound_identities(bound)), None)


def _outcome(result, exception=None):
    if exception is not None:
        return "failed"
    if isinstance(result, dict):
        for key in ("status", "quality_status"):
            value = result.get(key)
            if value == "verified_pending_post_exit":
                return "weak"
            if isinstance(value, str) and value in STATUSES:
                if value not in {"complete", "complete_verified", "success", "ok"}:
                    return value
    return "failed" if _failed(result) else "complete"


def _publish_pending(parent, result):
    if not isinstance(result, dict):
        return
    for name in ("jobs", "tasks", "rows", "matrix", "conditions"):
        rows = result.get(name, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            identity = _row_identity(row)
            if identity is None or identity in parent.seen:
                continue
            parent.seen.add(identity)
            outcome = _outcome(row)
            run_id = parent.client.create_run(
                parent.experiment_id,
                tags={
                    "gluomem.kind": "condition",
                    "gluomem.family": parent.family,
                    "mlflow.parentRunId": parent.run_id,
                    "gluomem.outcome": outcome,
                    "gluomem.execution": "retained_planned_row",
                },
                run_name="planned-condition",
            ).info.run_id
            parent.client.log_param(run_id, "condition_sha256", identity)
            parent.client.log_dict(run_id, project_result(row), "tracking/result.json")
            parent.client.set_terminated(
                run_id,
                status="FINISHED"
                if outcome == "complete"
                else ("FAILED" if outcome == "failed" else "KILLED"),
            )
            parent.outcomes.append(outcome)


def _validate_artifact_location(location):
    from urllib.parse import unquote, urlsplit

    value = urlsplit(location)
    if (
        value.netloc
        or value.query
        or value.fragment
        or value.scheme not in {"file", "mlflow-artifacts"}
    ):
        raise ValueError(
            "local MLflow artifact storage is required; remote artifact locations are refused"
        )
    if value.scheme == "file":
        # file:///C:/... on Windows; the leading URL slash is not a drive root.
        text = unquote(value.path)
        if os.name == "nt" and len(text) > 3 and text[0] == "/" and text[2] == ":":
            text = text[1:]
        root = Path(text).resolve()
        repo = Path(__file__).resolve().parents[1]
        if not Path(text).is_absolute() or any(
            root.is_relative_to(p) for p in (repo, repo.parent / "GluoMem")
        ):
            raise ValueError(
                "local MLflow artifacts must be absolute and outside source repositories"
            )


def _client_campaign(
    family, *, tracking_uri=None, experiment_name=None, artifact_root=None, run_name=None
):
    uri = validate_tracking_uri(
        tracking_uri or os.environ.get("GLUOMEM_MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    name = experiment_name or os.environ.get("GLUOMEM_MLFLOW_EXPERIMENT", "gluomem-bench")
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        root = Path(
            artifact_root
            or os.environ.get("GLUOMEM_MLFLOW_ARTIFACT_ROOT", str(default_root() / "artifacts"))
        )
        location = root.resolve().as_uri()
        _validate_artifact_location(location)
        experiment_id = client.create_experiment(name, artifact_location=location)
    else:
        _validate_artifact_location(experiment.artifact_location)
        experiment_id = experiment.experiment_id
    run = client.create_run(
        experiment_id,
        tags={"gluomem.kind": "campaign", "gluomem.family": family},
        run_name=run_name or family,
    )
    return _Campaign(client, experiment_id, run.info.run_id, family, code=code_identities())


def _failed(result):
    if isinstance(result, list):
        return any(_failed(row) for row in result)
    if not isinstance(result, dict):
        return False
    if any(
        result.get(key) is False
        for key in ("complete", "matrix_complete", "quality_complete", "invariants_passed")
    ):
        return True
    for key in ("status", "quality_status"):
        value = result.get(key)
        if (
            isinstance(value, str)
            and value
            not in {"complete", "completed", "complete_verified", "success", "ok", "verified"}
            and (value in STATUSES or value.startswith(("incomplete", "not_", "failed")))
        ):
            return True
    failures = result.get("failures", {})
    if isinstance(failures, dict) and any(bool(value) for value in failures.values()):
        return True
    return any(
        _failed(result.get(key))
        for key in ("jobs", "tasks", "rows", "questions", "result", "conditions", "matrix")
    )


def track_experiment(family: str, *, kind: str):
    """Wrap a campaign/condition boundary; legacy behavior is the default."""
    if kind not in {"campaign", "condition"}:
        raise ValueError("tracking kind must be campaign or condition")

    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def tracked(*args, **kwargs):
            if _PARENT.get() is None and os.environ.get(
                "GLUOMEM_MLFLOW_ENABLED", ""
            ).lower() not in {"1", "true"}:
                return function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs).arguments
            parent = _PARENT.get()
            if parent is not None and ((kind == "campaign") or _CHILD.get() is not None):
                run_id = _CHILD.get() or parent.run_id
                _log_params(
                    parent,
                    run_id,
                    experiment_params(
                        function, signature.bind(*args, **kwargs).arguments, code=parent.code
                    ),
                )
                return function(*args, **kwargs)
            owns_parent = parent is None
            if owns_parent:
                parent = _client_campaign(family)
            parent_token = _PARENT.set(parent)
            child_token = None
            run_id = parent.run_id
            if kind == "condition":
                run_id = parent.client.create_run(
                    parent.experiment_id,
                    tags={
                        "gluomem.kind": "condition",
                        "gluomem.family": family,
                        "mlflow.parentRunId": parent.run_id,
                    },
                    run_name=function.__name__,
                ).info.run_id
                child_token = _CHILD.set(run_id)
            status = "FAILED"
            result, exception = None, None
            record, outcome = None, "failed"
            started = time.perf_counter()
            try:
                values = experiment_params(function, bound, code=parent.code)
                inherited = {
                    key: value
                    for key, value in parent.params.get(parent.run_id, {}).items()
                    if key.startswith(("profile.", "config.", "code.")) or key == "data.sha256"
                }
                _log_params(parent, run_id, {**inherited, **values})
                if owns_parent and run_id != parent.run_id:
                    _log_params(parent, parent.run_id, values)
                result = function(*args, **kwargs)
                status = "FAILED" if _failed(result) else "FINISHED"
                return result
            except BaseException as exc:
                exception = exc
                raise
            finally:
                try:
                    record = collect_record(
                        bound,
                        result,
                        exception=exception,
                        wall_seconds=time.perf_counter() - started,
                    )
                    if exception is not None:
                        parent.client.set_tag(
                            run_id, "gluomem.exception_type", type(exception).__name__
                        )
                    outcome = _outcome(result, exception)
                    if outcome == "complete" and (
                        "failure.json" in record["receipts"]
                        or _failed(record["result"])
                        or any(_failed(view) for view in record["receipts"].values())
                    ):
                        outcome = "failed"
                    status = "FINISHED" if outcome == "complete" else "FAILED"
                    parent.client.set_tag(run_id, "gluomem.outcome", outcome)
                    if kind == "condition":
                        identity = _bound_identity(bound)
                        if identity is not None:
                            parent.seen.update(_bound_identities(bound))
                            parent.client.log_param(run_id, "condition_sha256", identity)
                        parent.outcomes.append(outcome)
                        parent.records[run_id] = record
                        if outcome in {
                            "unsupported",
                            "not_supported",
                            "not_admitted",
                            "not_applicable",
                            "not_applicable_context_overflow",
                            "context_refused",
                            "unrun",
                            "cancelled",
                        }:
                            status = "KILLED"
                    else:
                        _publish_pending(parent, result)
                        aggregate_children(record, list(parent.records.values()))
                        record["coverage"].update(
                            planned_conditions=len(parent.outcomes),
                            completed_conditions=parent.outcomes.count("complete"),
                            failed_conditions=parent.outcomes.count("failed"),
                        )
                        if any(item != "complete" for item in parent.outcomes):
                            status = "FAILED"
                    parent.client.log_dict(run_id, record, "tracking/result.json")
                    _log_metrics(parent, run_id, metric_values(record, parent.family))
                except Exception as tracking_error:
                    status = "FAILED"
                    if exception is None:
                        raise RuntimeError(
                            "MLflow finalization failed; experiment is not fully tracked"
                        ) from tracking_error
                    exception.add_note(
                        "MLflow finalization also failed; no raw exception text uploaded"
                    )
                finally:
                    try:
                        parent.client.set_terminated(run_id, status=status)
                        if owns_parent and run_id != parent.run_id:
                            if record is not None:
                                parent.client.log_dict(
                                    parent.run_id, record, "tracking/result.json"
                                )
                                _log_metrics(
                                    parent, parent.run_id, metric_values(record, parent.family)
                                )
                            parent.client.set_tag(parent.run_id, "gluomem.outcome", outcome)
                            parent.client.set_terminated(
                                parent.run_id,
                                status=("FINISHED" if status == "FINISHED" else "FAILED"),
                            )
                    finally:
                        if child_token is not None:
                            _CHILD.reset(child_token)
                        _PARENT.reset(parent_token)

        return tracked

    return decorate


class MlflowParent:
    """Explicit operator-owned campaign scope around original entrypoint calls.

    Calling this context is itself opt-in; no process-global MLflow state or
    environment mutation is used. Nested campaign decorators reuse this parent.
    """

    def __init__(
        self,
        family="campaign",
        *,
        experiment_name=None,
        tracking_uri=None,
        artifact_root=None,
        run_name=None,
        metadata=None,
    ):
        self.family = family
        self.options = dict(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            artifact_root=artifact_root,
            run_name=run_name,
        )
        self.metadata = metadata or {}
        self._campaign = None
        self._token = None
        self._owner = False
        self._result = {}

    @property
    def run_id(self):
        return self._campaign.run_id if self._campaign is not None else None

    @property
    def client(self):
        return self._campaign.client if self._campaign is not None else None

    def __enter__(self):
        self._started = time.perf_counter()
        existing = _PARENT.get()
        self._owner = existing is None
        self._campaign = existing or _client_campaign(self.family, **self.options)
        if self._owner:
            _log_params(
                self._campaign,
                self.run_id,
                experiment_params(type(self), self.metadata, code=self._campaign.code),
            )
        self._token = _PARENT.set(self._campaign)
        return self

    def record_result(self, result):
        """Retain the authoritative root receipt; never reinterpret a zero exit."""
        if not isinstance(result, dict):
            raise TypeError("MlflowParent.record_result requires a driver result dictionary")
        self._result = result
        return result

    set_result = record_result

    def __exit__(self, exc_type, exception, traceback):
        try:
            if not self._owner:
                return False
            parent = self._campaign
            assert parent is not None
            status = "FAILED"
            try:
                record = collect_record(
                    {},
                    self._result,
                    exception=exception,
                    wall_seconds=time.perf_counter() - self._started,
                )
                _publish_pending(parent, self._result)
                aggregate_children(record, list(parent.records.values()))
                record["coverage"].update(
                    planned_conditions=len(parent.outcomes),
                    completed_conditions=parent.outcomes.count("complete"),
                    failed_conditions=parent.outcomes.count("failed"),
                )
                outcome = _outcome(self._result, exception)
                if any(value != "complete" for value in parent.outcomes):
                    outcome = "failed"
                status = "FINISHED" if outcome == "complete" else "FAILED"
                parent.client.set_tag(parent.run_id, "gluomem.outcome", outcome)
                if exception is not None:
                    parent.client.set_tag(
                        parent.run_id, "gluomem.exception_type", type(exception).__name__
                    )
                parent.client.log_dict(parent.run_id, record, "tracking/result.json")
                _log_metrics(parent, parent.run_id, metric_values(record, self.family))
            except Exception:
                status = "FAILED"
                if exception is None:
                    raise
                exception.add_note("MLflow parent finalization also failed; no raw text uploaded")
            finally:
                parent.client.set_terminated(parent.run_id, status=status)
                if parent.client.get_run(parent.run_id).info.status != status:
                    raise RuntimeError("MLflow parent final-state readback mismatch")
            return False
        finally:
            if self._token is not None:
                _PARENT.reset(self._token)


@track_experiment("campaign", kind="condition")
def _execute_job(function, job, context):
    return function(job, context)


def submit_condition(pool, function, job, context):
    """Carry only the current run context into the existing worker submission."""
    return pool.submit(copy_context().run, _execute_job, function, job, context)
