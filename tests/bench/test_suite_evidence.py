"""Stronger offline bundle admission; no dataset, model or network access."""

import json

import pytest


def test_evidence_bundle_rejects_empty_hash_inventory(tmp_path):
    from gluomem_bench.bench.suite_evidence import validate_bundle

    manifest = {"artifact_sha256": {}, "sample_ids": ["q"],
                "dataset": "memoryagentbench_conflict", "system": "no_memory"}
    with pytest.raises(ValueError, match="required evidence"):
        validate_bundle(tmp_path, manifest)


def test_evidence_bundle_requires_frozen_plan_identity(tmp_path):
    from gluomem_bench.bench.suite_evidence import validate_bundle

    names = ["protocol.json", "effective-config.json", "profile.json", "predictions.json",
             "usage.json", "calls.jsonl", "scores.json"]
    import hashlib

    for name in names:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    manifest = {"artifact_sha256": {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
                                   for n in names}, "sample_ids": ["forged"],
                "dataset": "memoryagentbench_conflict", "system": "no_memory"}
    with pytest.raises(ValueError, match="frozen plan"):
        validate_bundle(tmp_path, manifest, expected={"dataset": "memoryagentbench_conflict",
                                                     "system": "no_memory", "sample_ids": ["q"]})
    assert json.loads((tmp_path / "protocol.json").read_text()) == {}
