"""Local MLflow server launch helpers; no benchmark/model dependencies required."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


def default_root():
    if os.name == "nt":
        return (
            Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
            / "gluomem-evaluations/mlflow"
        )
    return Path.home() / ".local/share/gluomem-evaluations/mlflow"


def validate_tracking_uri(uri):
    """Reject credentials, remote endpoints, and relative/ambiguous databases."""
    parsed = urlsplit(uri)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("local tracking URI must not contain credentials, query or fragment")
    if parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
        if parsed.path not in ("", "/") or parsed.port is None or not 1 <= parsed.port <= 65535:
            raise ValueError("local loopback tracking requires an explicit port and no path")
        return uri
    if uri.startswith("sqlite:///") and not parsed.netloc:
        path = Path(unquote(uri[len("sqlite:///") :]))
        if path.is_absolute() and not str(path).startswith(("\\\\", "//")):
            return uri
        raise ValueError("local SQLite tracking requires an absolute database path")
    raise ValueError("only loopback HTTP or absolute local SQLite tracking is allowed")


def server_command(*, root=None, port=5000, python=None):
    """Foreground, one-worker command; caller owns the service process lifetime."""
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("local server port must be an integer in 1..65535")
    root = Path(root or default_root()).resolve()
    repo = Path(__file__).resolve().parents[1]
    if root.is_relative_to(repo) or str(root).startswith(("\\\\", "//")):
        raise ValueError("local MLflow durable store must be outside the source repository")
    return [
        str(python or sys.executable),
        "-m",
        "mlflow",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
        "--backend-store-uri",
        "sqlite:///" + (root / "mlflow.db").as_posix(),
        "--default-artifact-root",
        (root / "artifacts").as_uri(),
        "--no-serve-artifacts",
        "--allowed-hosts",
        f"127.0.0.1:{port},localhost:{port}",
    ]


def server_status(uri="http://127.0.0.1:5000"):
    """Bounded health AND tracking-API read probe (health alone is insufficient)."""
    import json
    from urllib.request import Request, urlopen

    validate_tracking_uri(uri)
    if not uri.startswith("http://"):
        raise ValueError("server status requires a loopback HTTP URI")
    result = {"health_ok": False, "api_ok": False, "ready": False}
    for key, path in (("health_ok", "/health"), ("api_ok", "/api/2.0/mlflow/experiments/search")):
        request = Request(
            uri.rstrip("/") + path,
            data=b'{"max_results":1}' if key == "api_ok" else None,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=3) as response:
                body = response.read(1024 * 1024)
                result[key] = response.status == 200 and (
                    key == "health_ok" or isinstance(json.loads(body), dict)
                )
        except Exception as exc:
            result[key + "_error_type"] = type(exc).__name__
    result["ready"] = result["health_ok"] and result["api_ok"]
    return result


def smoke(*, uri="http://127.0.0.1:5000", artifact_root=None):
    """Actually log and reread a parent/child metric and safe JSON artifact."""
    import json
    import tempfile
    import uuid

    from mlflow import MlflowClient

    validate_tracking_uri(uri)
    root = Path(artifact_root or default_root() / "artifacts").resolve()
    root.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=uri)
    name = "tracking-smoke-" + uuid.uuid4().hex
    eid = client.create_experiment(name, artifact_location=(root / name).as_uri())
    ids = []
    scope = "tracking-plumbing-only-not-benchmark-quality"
    try:
        parent = client.create_run(eid, tags={"gluomem.kind": "campaign", "gluomem.scope": scope})
        ids.append(parent.info.run_id)
        child = client.create_run(
            eid,
            tags={
                "gluomem.kind": "condition",
                "gluomem.scope": scope,
                "mlflow.parentRunId": ids[0],
            },
        )
        ids.append(child.info.run_id)
        client.log_param(ids[1], "probe", "explicit-local-log-readback")
        client.log_metric(ids[1], "tracking.probe", 1.0)
        payload = {"scope": scope, "no_model_calls": True}
        client.log_dict(ids[1], payload, "tracking/probe.json")
        for run_id in ids:
            client.set_terminated(run_id, status="FINISHED")
        readback = client.get_run(ids[1])
        assert readback.info.status == "FINISHED"
        assert client.get_run(ids[0]).info.status == "FINISHED"
        assert readback.data.tags["mlflow.parentRunId"] == ids[0]
        assert readback.data.params["probe"] == "explicit-local-log-readback"
        assert readback.data.metrics["tracking.probe"] == 1.0
        with tempfile.TemporaryDirectory(prefix="mlflow-readback-", dir=root) as target:
            path = client.download_artifacts(ids[1], "tracking/probe.json", target)
            assert json.loads(Path(path).read_text(encoding="utf-8")) == payload
        return {"verified": True, "experiment_id": eid, "run_ids": ids, "scope": scope}
    except BaseException:
        for run_id in ids:
            client.set_terminated(run_id, status="FAILED")
        raise


def add_commands(parser):
    commands = parser.add_subparsers(dest="tracking_command", required=True)
    serve = commands.add_parser("serve", help="foreground local SQLite-backed MLflow server")
    serve.add_argument("--root", type=Path, default=default_root())
    serve.add_argument("--port", type=int, default=5000)
    serve.add_argument(
        "--python", default=sys.executable, help="interpreter with full mlflow installed"
    )
    serve.add_argument("--dry-run", action="store_true")
    status = commands.add_parser("status", help="bounded health and tracking API readiness")
    status.add_argument("--uri", default="http://127.0.0.1:5000")
    probe = commands.add_parser(
        "smoke", help="write/readback a labelled infrastructure-only experiment"
    )
    probe.add_argument("--uri", default="http://127.0.0.1:5000")
    probe.add_argument("--artifact-root", type=Path, default=default_root() / "artifacts")


def dispatch(args):
    import json
    import shlex
    import subprocess

    if args.tracking_command == "serve":
        command = server_command(root=args.root, port=args.port, python=args.python)
        if args.dry_run:
            print(shlex.join(command))
            return 0
        Path(args.root).mkdir(parents=True, exist_ok=True)
        return subprocess.run(command, check=False).returncode
    result = (
        server_status(args.uri)
        if args.tracking_command == "status"
        else smoke(uri=args.uri, artifact_root=args.artifact_root)
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("ready", result.get("verified", False)) else 1


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_commands(parser)
    return dispatch(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
