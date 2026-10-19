"""Exercise safe diagnostics at actual coordinator persistence boundaries."""

import json

import pytest

from gluomem_bench.bench import parallel_campaign as campaign
from gluomem_bench.bench.embedding_runtime import safe_failure_diagnostic
from tests.bench.test_encoder_failure_privacy import (
    assert_absent,
    privacy_native_history_scenario,
)


def test_safe_failure_preserves_allowlisted_builtin_type_without_formatting():
    assert safe_failure_diagnostic(OSError("private")) == {
        "type": "OSError", "message": "operation_failed"
    }

    class UntrustedError(OSError):
        def __str__(self):
            raise AssertionError("exception formatting is forbidden")

    assert safe_failure_diagnostic(UntrustedError()) == {
        "type": "Exception", "message": "operation_failed"
    }


@pytest.mark.parametrize("outer_form", ["context", "workload"])
def test_coordinator_does_not_serialize_original_http_reason(
    tmp_path, bench_cfg, monkeypatch, outer_form
):
    result, secrets = privacy_native_history_scenario(
        tmp_path, bench_cfg, monkeypatch, outer_form, "reason"
    )
    assert_absent(json.dumps(result).encode(), secrets)
    assert_absent((tmp_path / "coordination/progress.json").read_bytes(), secrets)


def test_accounting_failure_receipt_does_not_format_untrusted_exception(
    tmp_path, monkeypatch
):
    secret = "authored-private-accounting-value"
    original = campaign._atomic_write

    def fail_progress(path, payload):
        if path.name == "progress.json":
            raise OSError(secret)
        return original(path, payload)

    monkeypatch.setattr(campaign, "_atomic_write", fail_progress)
    policy = campaign.CampaignPolicy(
        max_workers=1,
        max_in_flight=1,
        max_in_flight_tokens=100,
        max_calls=2,
        max_seconds=60,
        stop_on_failure=False,
    )
    job = campaign.Job(
        job_id="authored",
        history_id="authored",
        output=tmp_path / "job",
        call_budget=1,
        seconds_budget=10,
    )
    with pytest.raises(campaign.CampaignReceiptError) as caught:
        campaign.run_jobs(
            [job],
            out=tmp_path / "coordination",
            policy=policy,
            run_job=lambda *_: {"complete": True},
        )
    assert_absent(json.dumps(caught.value.receipt).encode(), [secret])
