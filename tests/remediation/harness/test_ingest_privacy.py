import json
from types import SimpleNamespace
import pytest
from pydantic import ValidationError
from gluomem.llm.outputs import ExtractedClaimSpec
from gluomem_bench.bench import beam_ingest as b


@pytest.mark.parametrize('payload, expected', [
    ({'text': 'PRIVATE', 'kind': 'PRIVATE'}, 'literal_error'),
    ({'text': 'PRIVATE', 'event_date': 'PRIVATE'}, 'datetime_from_date_parsing'),
    ({'text': 'PRIVATE', 'relations': [{'source': 'PRIVATE', 'target': 'PRIVATE'}]}, 'missing'),
])
def test_nested_schema_receipt_keeps_only_type_and_static_field_location(payload, expected):
    try:
        ExtractedClaimSpec.model_validate(payload)
    except ValidationError as cause:
        outer = RuntimeError('PRIVATE wrapped output')
        outer.__cause__ = cause
    report = b._failure_diagnostic(outer, 'native-ingest')
    assert report['category'] == 'validation-error'
    assert report['stage'] == 'native-ingest'
    assert report['validation_errors'][0]['type'] == expected
    assert 'PRIVATE' not in json.dumps(report)
    assert all(set(row) == {'type', 'loc'} for row in report['validation_errors'])


@pytest.mark.parametrize('publication_failure', [False, True])
def test_actual_ingest_failure_never_formats_error_and_preserves_partial_counters(tmp_path, monkeypatch, publication_failure):
    class PrivateError(Exception):
        def __str__(self):
            pytest.fail('unrestricted exception formatting')
    error = PrivateError()
    rows = [{'payload_json': '{}'}, {'payload_json': '{"messages":[{}]}'}]
    class DB:
        def __init__(self, *args, **kwargs): pass
        def close(self): pass
    monkeypatch.setattr(b, 'MindDB', DB)
    monkeypatch.setattr(b, 'FrameStore', lambda *args, **kwargs: object())
    monkeypatch.setattr(b, 'KnowledgeGraph', lambda *args: object())
    monkeypatch.setattr(b, '_read_rows', lambda path: rows)
    def fail(*args, **kwargs): raise error
    monkeypatch.setattr(b, 'ingest_source_event', fail)
    if publication_failure:
        def broken_write(*args): raise OSError('PRIVATE write failure')
        monkeypatch.setattr(b, '_write_meta', broken_write)
    cfg = SimpleNamespace(storage=SimpleNamespace(vector_dim=1), chunking=None, lifecycle=None)
    event = SimpleNamespace(messages=[object()])
    with pytest.raises(PrivateError) as found:
        b._ingest_conversation(cfg, None, tmp_path, object(), object(),
            SimpleNamespace(conv_id='authored'), None, source_events=[event])
    assert found.value is error
    if publication_failure:
        assert not (tmp_path / 'authored.ingest.failure.json').exists()
        return
    report = json.loads((tmp_path / 'authored.ingest.failure.json').read_bytes())
    assert report['error'] == 'production_source_event_failed'
    assert report['failure_diagnostic']['stage'] == 'native-ingest'
    assert report['persisted_event_rows'] == 2
    assert report['redacted_events'] == 1 and report['retained_plaintext_messages'] == 1
    assert report['committed_receipts'] == [] and not report['complete']
    assert not (tmp_path / 'authored.ingest.json').exists()

def test_dynamic_validation_keys_and_messages_are_not_diagnostic_fields():
    from pydantic_core import PydanticCustomError
    error = ValidationError.from_exception_data('PRIVATE title', [{
        'type': PydanticCustomError('PRIVATE_type', 'PRIVATE message', {'PRIVATE': 'PRIVATE'}),
        'loc': ('claims', index, 'properties', 'PRIVATE key'), 'input': 'PRIVATE input',
    } for index in range(40)])
    wrapper = RuntimeError('PRIVATE wrapper')
    wrapper.__cause__ = error
    wrapper.__context__ = wrapper
    report = b._failure_diagnostic(wrapper, 'native-ingest')
    assert len(report['validation_errors']) == 16
    assert all(row['type'] == 'unknown' for row in report['validation_errors'])
    assert report['validation_errors'][0]['loc'] == ['claims', 0, 'properties', 'unknown']
    assert 'PRIVATE' not in json.dumps(report)
