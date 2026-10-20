"""Outer native verification must carry independent adapted Mem0 authority."""
import sys
from datetime import UTC, datetime
from types import ModuleType, SimpleNamespace

import pytest
from gluomem_bench.bench import lme_first_execute as native
from gluomem_bench.bench import mem0, suite_campaign
from gluomem_bench.bench.rack import RackProfile


@pytest.mark.parametrize("forged", [False, True])
def test_native_mem0_verifier_carries_original_binding(monkeypatch, tmp_path, forged):
    prompt = ModuleType("mem0.configs.prompts")
    prompt.ADDITIVE_EXTRACTION_PROMPT = "Authored fixture prompt, not SDK parity evidence."
    monkeypatch.setitem(sys.modules, "mem0.configs.prompts", prompt)
    session = SimpleNamespace(index=0, date_time=datetime(2020, 1, 1, tzinfo=UTC),
                              turns=[SimpleNamespace(speaker="user", text=" Authored source.\n")])
    conv = SimpleNamespace(conv_id="authored", sessions=[session])
    adapter = SimpleNamespace(name="longmemeval_s", sha="a" * 64, conversations=lambda: [conv])
    sample = SimpleNamespace(id="authored-question", conv_id=conv.conv_id)
    cfg = SimpleNamespace(benchmark=SimpleNamespace(mem0=SimpleNamespace(max_batch_input_tokens=2048)))
    profile = RackProfile(base_url="http://fixture.invalid/v1", model="fixture/model")
    originals = [{"session_index": 0, "session_date": session.date_time.isoformat(),
                  "messages": mem0._session_messages(session)}]
    captured = {}

    def binding(actual_cfg, inputs, *, scope):
        assert actual_cfg is cfg and inputs == originals and scope == conv.conv_id
        return {}, {"mem0_original_source_inputs_sha256": "c" * 64 if forged else mem0._mem0_digest(originals),
                    "mem0_source_inputs_sha256": "b" * 64}

    class ReachedVerifier(Exception):
        pass

    def verify(*args, **kwargs):
        captured.update(kwargs)
        raise ReachedVerifier()

    monkeypatch.setattr(mem0, "mem0_adapted_binding", binding)
    monkeypatch.setattr(suite_campaign, "verify_history", verify)
    job = SimpleNamespace(system="mem0", output=tmp_path / "unused")
    with pytest.raises(ValueError if forged else ReachedVerifier):
        native.verify_native_job(job=job, context=SimpleNamespace(calls=0), cfg=cfg, adapter=adapter,
                                 sample=sample, profile=profile, judge_source=tmp_path / "unused.py")
    if forged:
        assert not captured
    else:
        assert "mem0_cfg" in captured, "outer verifier discarded independent adapted authority"
        assert captured["mem0_cfg"] is cfg
        assert captured["mem0_original_inputs"] == originals
        assert captured["mem0_scope"] == conv.conv_id
        assert captured["expected_job"]["mem0_original_source_inputs_sha256"] == mem0._mem0_digest(originals)
        assert captured["expected_job"]["mem0_source_inputs_sha256"] == "b" * 64
