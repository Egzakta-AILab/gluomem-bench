"""Actual CLI refuses old output before constructing a network client."""

import hashlib
import json

import httpx
import pytest

from gluomem_bench.bench import beam_cli
from gluomem_bench.bench.rack import RackClient, RackProfile
from gluomem_bench.cli import main
from tests.bench.test_beam import beam_data


@pytest.mark.parametrize("command", ["run", "ingest"])
def test_cli_requires_operator_profile_before_constructing_client(tmp_path, monkeypatch, command):
    constructed = []
    monkeypatch.setattr(beam_cli, "RackClient", lambda *a, **k: constructed.append(True))
    args = ["beam", command, "--data", str(tmp_path / "data.json"), "--sha256", "a" * 64]
    args += (
        ["--out", str(tmp_path / "run")]
        if command == "run"
        else ["--db-dir", str(tmp_path / "stores"), "--conv-id", "beam-1"]
    )
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert constructed == []


@pytest.mark.parametrize("conflict", ["prior-run", "later-history", "trace-only", "failure-only"])
def test_cli_ingest_refuses_entire_output_scope_before_preflight(
    tmp_path, monkeypatch, conflict, capsys
):
    cfg, data, rows = beam_data(tmp_path)
    rows.append({**rows[0], "conversation_id": "2"})
    data.write_text(json.dumps(rows), encoding="utf-8")
    sha = hashlib.sha256(data.read_bytes()).hexdigest()
    stores = tmp_path / "stores"
    stores.mkdir()
    existing = {
        "prior-run": ["beam-1.mind", "beam-1.ingest.json", "calls.jsonl", "failure.json"],
        "later-history": ["beam-2.mind", "beam-2.ingest.json", "beam-2.ingest.failure.json"],
        "trace-only": ["calls.jsonl"],
        "failure-only": ["beam-1.ingest.failure.json"],
    }[conflict]
    for name in existing:
        (stores / name).write_bytes(b"original evidence\r\n\x00" + name.encode())
    profile_path = tmp_path / "operator.json"
    profile_path.write_text(
        RackProfile(
            base_url="http://localhost:8000/v1", model="example/served-model"
        ).model_dump_json()
    )
    before = {p.name: p.read_bytes() for p in stores.iterdir()}
    requests = []
    constructed = []

    def serve(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": RackProfile(
                            base_url="http://localhost:8000/v1", model="example/served-model"
                        ).model
                    }
                ]
            },
        )

    def client_factory(profile, **kwargs):
        constructed.append(True)
        return RackClient(profile, transport=httpx.MockTransport(serve), **kwargs)

    monkeypatch.setattr(beam_cli, "RackClient", client_factory)
    assert (
        main(
            [
                "beam",
                "ingest",
                "--profile",
                str(profile_path),
                "--data",
                str(data),
                "--sha256",
                sha,
                "--db-dir",
                str(stores),
                "--conv-id",
                "beam-1",
                "--conv-id",
                "beam-2",
            ]
        )
        == 1
    )
    assert "fresh ingestion required" in capsys.readouterr().err
    assert {p.name: p.read_bytes() for p in stores.iterdir()} == before
    assert requests == []
    assert constructed == []
