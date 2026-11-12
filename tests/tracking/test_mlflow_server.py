"""Local-only server and tracking configuration guards."""

from __future__ import annotations

import importlib.util

import pytest


def test_server_command_is_loopback_sqlite_single_worker_with_exact_hosts(tmp_path):
    assert importlib.util.find_spec("gluomem_bench.tracking_server") is not None, "launcher missing"
    from gluomem_bench.tracking_server import server_command

    command = server_command(root=tmp_path, port=5009, python="python-fixture")
    assert command[:4] == ["python-fixture", "-m", "mlflow", "server"]
    for flag, value in {
        "--host": "127.0.0.1",
        "--port": "5009",
        "--workers": "1",
        "--allowed-hosts": "127.0.0.1:5009,localhost:5009",
    }.items():
        assert command[command.index(flag) + 1] == value
    uri = command[command.index("--backend-store-uri") + 1]
    assert uri == "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    assert str(tmp_path) in " ".join(command) or tmp_path.as_uri() in " ".join(command)
    assert "--disable-security-middleware" not in command
    assert "*" not in " ".join(command)


def test_tracking_settings_reject_remote_or_credentialed_uris_before_execution(
    tmp_path, monkeypatch
):
    from gluomem_bench import tracking

    assert hasattr(tracking, "validate_tracking_uri"), "local-only URI guard missing"
    monkeypatch.setenv("GLUOMEM_MLFLOW_ENABLED", "1")
    executed = []

    @tracking.track_experiment("beam", kind="condition")
    def driver():
        executed.append(True)
        return {"complete": True}

    for uri in (
        "https://remote.example",
        "http://127.0.0.1:5000/?token=secret",
        "http://user:password@localhost:5000",
        "file:///somewhere",
        "sqlite:///relative.db",
    ):
        monkeypatch.setenv("GLUOMEM_MLFLOW_TRACKING_URI", uri)
        with pytest.raises(ValueError, match="local|loopback|credential|absolute"):
            driver()
    assert not executed


def test_existing_remote_artifact_experiment_is_refused_before_work(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from mlflow import MlflowClient

    from gluomem_bench import tracking

    assert hasattr(tracking, "_validate_artifact_location"), "remote artifact guard missing"
    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    client = MlflowClient(tracking_uri=uri)
    client.create_experiment("unsafe-artifacts", artifact_location="s3://do-not-contact/tracking")
    for key, value in {
        "ENABLED": "1",
        "TRACKING_URI": uri,
        "EXPERIMENT": "unsafe-artifacts",
    }.items():
        monkeypatch.setenv("GLUOMEM_MLFLOW_" + key, value)
    executed = []

    @tracking.track_experiment("beam", kind="condition")
    def work():
        executed.append(True)
        return {"complete": True}

    with pytest.raises(ValueError, match="local.*artifact"):
        work()
    assert not executed


def test_server_status_checks_mlflow_api_not_only_health():
    from gluomem_bench import tracking_server

    assert hasattr(tracking_server, "server_status"), "API readiness probe missing"
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 403)
            self.end_headers()
            self.wfile.write(b"OK" if self.path == "/health" else b"Invalid Host header")

        do_POST = do_GET

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status = tracking_server.server_status(f"http://127.0.0.1:{server.server_port}")
        assert status["health_ok"] is True
        assert status["api_ok"] is False
        assert status["ready"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_operational_smoke_verifies_actual_parent_child_metrics_and_artifact(tmp_path):
    pytest.importorskip("mlflow")
    from gluomem_bench import tracking_server

    assert hasattr(tracking_server, "smoke"), "actual API log/readback probe missing"
    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    result = tracking_server.smoke(uri=uri, artifact_root=tmp_path / "artifacts")
    assert result["verified"] is True
    assert len(result["run_ids"]) == 2
    assert result["scope"] == "tracking-plumbing-only-not-benchmark-quality"


def test_tracking_launch_command_works_without_importing_benchmark_config(capsys, tmp_path):
    from gluomem_bench import tracking_server

    assert hasattr(tracking_server, "main"), "standalone server command missing"
    assert (
        tracking_server.main(["serve", "--root", str(tmp_path), "--port", "5009", "--dry-run"]) == 0
    )
    assert "127.0.0.1:5009,localhost:5009" in capsys.readouterr().out
