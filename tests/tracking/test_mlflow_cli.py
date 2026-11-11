"""The existing CLI opts in without changing the legacy default."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path


def test_existing_cli_exposes_tracking_service_commands(tmp_path, capsys):
    from gluomem_bench.cli import main

    assert main(["tracking", "serve", "--root", str(tmp_path), "--port", "5009", "--dry-run"]) == 0
    assert "--allowed-hosts 127.0.0.1:5009,localhost:5009" in capsys.readouterr().out


def test_existing_cli_tracking_flags_scope_only_explicit_settings(monkeypatch):
    from gluomem_bench import cli

    monkeypatch.delenv("GLUOMEM_MLFLOW_ENABLED", raising=False)
    monkeypatch.setenv("GLUOMEM_MLFLOW_EXPERIMENT", "previous-experiment")
    captured = {}

    def dispatch(args, parser):
        for key in ("ENABLED", "TRACKING_URI", "EXPERIMENT"):
            captured[key] = os.environ.get("GLUOMEM_MLFLOW_" + key)
        return 0

    monkeypatch.setattr(cli, "_dispatch", dispatch)
    assert (
        cli.main(
            [
                "--mlflow",
                "--mlflow-tracking-uri",
                "http://127.0.0.1:5009",
                "--mlflow-experiment",
                "explicit-campaign",
                "components",
            ]
        )
        == 0
    )
    assert captured == {
        "ENABLED": "1",
        "TRACKING_URI": "http://127.0.0.1:5009",
        "EXPERIMENT": "explicit-campaign",
    }
    assert "GLUOMEM_MLFLOW_ENABLED" not in os.environ
    assert os.environ["GLUOMEM_MLFLOW_EXPERIMENT"] == "previous-experiment"


def test_optional_client_and_server_dependencies_are_explicit_and_version_matched():
    project = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    extras = project["project"]["optional-dependencies"]
    assert extras.get("tracking-client") == ["mlflow-skinny==3.17.2"]
    assert extras.get("tracking") == ["mlflow==3.17.2"]
    assert (
        project["project"]["scripts"].get("gluomem-bench-mlflow")
        == "gluomem_bench.tracking_server:main"
    )
