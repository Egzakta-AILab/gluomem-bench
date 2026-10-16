"""Workload failures keep an honest measured terminal receipt, not a score."""

import json

import pytest

from gluomem_bench.bench.beam_eval import run_beam
from gluomem_bench.bench.rack import RackClient, RackProfile
from gluomem_bench.bench.rack_trace import WorkloadBudget
from tests.bench.test_beam import beam_data


def test_failed_run_keeps_actual_counts_wall_and_error(tmp_path):
    cfg, _, _ = beam_data(tmp_path)
    budget = WorkloadBudget()
    budget.failed = True
    client = RackClient(
        RackProfile(base_url="http://localhost:8000/v1", model="example/served-model"),
        receipt=tmp_path / "run/calls.jsonl",
        budget=budget,
    )
    try:
        with pytest.raises(RuntimeError, match="workload"):
            run_beam(cfg, client=client, out=tmp_path / "run", retriever="bm25")
    finally:
        client.close()
    failure = json.loads((tmp_path / "run/failure.json").read_text())
    assert failure["complete"] is False
    assert failure["generation_calls"] == 0
    assert failure["predictions"] == failure["judgments"] == 0
    assert failure["wall_seconds"] >= 0
    assert "workload" in failure["error"]
    assert not (tmp_path / "run/manifest.json").exists()
