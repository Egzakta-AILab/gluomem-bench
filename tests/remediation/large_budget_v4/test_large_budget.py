"""Offline versioned allowance contracts; no deployed endpoints."""
import pytest
from pydantic import ValidationError
from gluomem_bench.bench.rack import GenerationBudgets, RackProfile


def material(**updates):
    value = dict(base_url='http://owned.invalid/v1', model='authored/qwen',
        profile_version='role-budgets-v7', timeout_seconds=1800,
        max_generation_calls=16384, max_run_seconds=21600,
        generation_budgets=dict(extraction=16384, consolidation=32768, reader=4096, judge=4096))
    value.update(updates)
    return value


def test_opt_in_large_profile_round_trips_without_changing_other_roles():
    profile = RackProfile.model_validate(material())
    assert RackProfile.model_validate_json(profile.model_dump_json()) == profile
    assert [profile.output_budget(t) for t in ('event_claims', 'consolidate', 'contradiction', 'answer', 'rubric')] == [16384, 32768, 32768, 4096, 4096]
    assert profile.timeout_seconds == 1800
    assert profile.context_window == 131072 and profile.schema_reserve_tokens == 2048
    assert profile.temperature == 0


def test_operator_profile_changes_only_version_consolidation_and_timeout():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parent
    old = json.loads((root.parent/'full-study-v3/profile.json').read_bytes())
    configured = json.loads((root/'profile.json').read_bytes())
    expected = {**old, 'profile_version': 'role-budgets-v7', 'timeout_seconds': 1800,
        'generation_budgets': {**old['generation_budgets'], 'consolidation': 32768}}
    assert configured == expected
    actual = RackProfile.model_validate(configured)
    assert actual.max_run_seconds == 21600 and actual.max_generation_calls == 16384
    assert actual.generation_budgets.extraction == 16384
    assert actual.generation_budgets.reader == actual.generation_budgets.judge == 4096


@pytest.mark.parametrize('role', ['extraction', 'consolidation', 'reader', 'judge'])
def test_v7_each_role_is_context_bound_not_a_new_fixed_cap(role):
    values = material()
    values['generation_budgets'][role] = 131072 - 2048
    assert getattr(RackProfile.model_validate(values).generation_budgets, role) == 129024
    values['generation_budgets'][role] += 1
    with pytest.raises(ValidationError, match='schema reserve'):
        RackProfile.model_validate(values)


@pytest.mark.parametrize('version', ['role-budgets-v2', 'role-budgets-v4', 'role-budgets-v5', 'role-budgets-v6'])
def test_legacy_exact_role_and_timeout_caps(version):
    values = material(profile_version=version, timeout_seconds=120,
        max_generation_calls=128, max_run_seconds=900)
    values['generation_budgets'].update(consolidation=4096, judge=1024)
    assert RackProfile.model_validate(values)
    for role, limit in [('extraction', 16384), ('consolidation', 4096), ('reader', 4096),
                        ('judge', 4096 if version == 'role-budgets-v6' else 1024)]:
        invalid = {**values, 'generation_budgets': {**values['generation_budgets'], role: limit + 1}}
        with pytest.raises(ValidationError):
            RackProfile.model_validate(invalid)
    with pytest.raises(ValidationError):
        RackProfile.model_validate({**values, 'timeout_seconds': 601 if version == 'role-budgets-v6' else 121})
    if version == 'role-budgets-v6':
        assert RackProfile.model_validate({**values, 'timeout_seconds': 600})


def test_standalone_generation_budgets_retains_legacy_caps():
    with pytest.raises(ValidationError):
        GenerationBudgets(extraction=16384, consolidation=32768, reader=4096, judge=4096)


@pytest.mark.parametrize('updates', [dict(timeout_seconds=21601), dict(timeout_seconds=float('inf')),
    dict(timeout_seconds=float('nan')), dict(max_run_seconds=1799), dict(generation_budgets=None),
    dict(generation_budgets=dict(extraction=16384, consolidation=32768, reader=4096)),
    dict(generation_budgets=dict(extraction=16384, consolidation=32768, reader=4096, judge=4096, unknown=1)),
    dict(generation_budgets=dict(extraction=16384, consolidation=31, reader=4096, judge=4096)),
    dict(schema_reserve_tokens=131073)])
def test_v7_invalid_role_request_and_context_inputs_rejected(updates):
    with pytest.raises(ValidationError):
        RackProfile.model_validate(material(**updates))


def test_legacy_defaults_unchanged():
    profile = RackProfile(base_url='http://owned.invalid/v1', model='authored/qwen')
    assert (profile.profile_version, profile.timeout_seconds, profile.max_tokens,
            profile.max_generation_calls, profile.max_run_seconds) == ('legacy-v1', 60, 512, 128, 900)


@pytest.mark.parametrize('tokens', [None, True, -1, '20'])
def test_v7_tokenizer_fault_is_shared_and_latched(tmp_path, tokens):
    import httpx
    from gluomem_bench.bench.rack import RackClient, SharedRackError
    profile = RackProfile.model_validate(material())
    client = RackClient(profile, receipt=tmp_path/'http.jsonl',
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={'count': tokens})))
    try:
        with pytest.raises(SharedRackError):
            client.prompt_token_count([{'role': 'user', 'content': 'authored'}])
        assert client._failure is not None
    finally:
        client.close()


def test_v7_shared_failure_reaches_outer_budget(tmp_path):
    import httpx
    from gluomem_bench.bench.rack import RackClient, SharedRackError
    class Budget:
        failed = False
        shared = False
        def fail_shared(self):
            self.shared = True
    budget = Budget()
    client = RackClient(RackProfile.model_validate(material()), receipt=tmp_path/'http.jsonl',
        budget=budget, transport=httpx.MockTransport(lambda req: httpx.Response(200, json={})))
    try:
        client._latch(SharedRackError('authored failure'))
        assert budget.failed and budget.shared
    finally:
        client.close()


def test_v7_native_judge_authorized_parameters_preserved(tmp_path):
    import httpx
    from gluomem_bench.bench.rack import RackClient
    seen = []
    def handler(request):
        import json
        seen.append(json.loads(request.content))
        if request.url.path == '/tokenize':
            return httpx.Response(200, json={'count': 20})
        return httpx.Response(200, json={'model': 'authored/qwen', 'choices': [
            {'finish_reason': 'stop', 'message': {'content': 'yes'}}]})
    client = RackClient(RackProfile.model_validate(material()), receipt=tmp_path/'http.jsonl',
        transport=httpx.MockTransport(handler))
    try:
        assert client.generate_messages('rubric', [{'role': 'user', 'content': 'authored'}],
            temperature=0.1, max_tokens=10)['text'] == 'yes'
        assert seen[-1]['temperature'] == 0.1 and seen[-1]['max_tokens'] == 10
    finally:
        client.close()
