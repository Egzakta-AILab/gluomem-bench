"""Offline native logging privacy: pinned algorithms, no service or database fixture."""
import ast
import hashlib
import io
import json
import logging
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from gluomem_bench.bench.mem0 import NativeMem0Diagnostics, SharedMem0Embedding
from gluomem_bench.bench.rack import FatalRackError
from tests.bench.test_encoder_failure_privacy import assert_absent







def test_emit_never_formats_untrusted_record_or_phase(tmp_path):
    class Hostile:
        def __str__(self):
            pytest.fail('diagnostic invoked untrusted str')
        def __repr__(self):
            pytest.fail('diagnostic invoked untrusted repr')
        def __hash__(self):
            pytest.fail('diagnostic invoked untrusted hash')
    class Record(logging.LogRecord):
        def getMessage(self):
            pytest.fail('diagnostic invoked getMessage')
    observer = NativeMem0Diagnostics(tmp_path / 'events.json')
    try:
        observer.phase = Hostile()
        record = Record('mem0.private', logging.WARNING, '', 1, Hostile(), (), None)
        record.name = record.levelname = record.levelno = Hostile()
        observer.emit(record)
        event = json.loads(observer.path.read_bytes())[0]
        assert event == {'logger': 'mem0', 'phase': 'unknown', 'level': 'UNKNOWN',
                         'message': 'native Mem0 runtime diagnostic'}
        with pytest.raises(FatalRackError):
            observer.check()
    finally:
        observer.close()


def test_overlapping_captures_restore_only_after_last_close_and_cover_new_children(tmp_path):
    import concurrent.futures
    original = logging.Logger.callHandlers
    first = NativeMem0Diagnostics(tmp_path / 'first.json')
    second = NativeMem0Diagnostics(tmp_path / 'second.json')
    root = logging.getLogger()
    sink = logging.FileHandler(tmp_path / 'forwarded.log')
    root.addHandler(sink)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda n: logging.getLogger('mem0.both.child.' + str(n)).warning(
                'opaque-private-message'), range(20)))
        first.close()
        first.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda n: logging.getLogger('mem0.new.child.' + str(n)).warning(
                'opaque-private-message'), range(20)))
        assert len(json.loads(first.path.read_bytes())) == 20
        assert len(json.loads(second.path.read_bytes())) == 40
        with pytest.raises(FatalRackError):
            second.check()
        logging.getLogger('unrelated.fixture').warning('unrelated remains enabled')
        sink.flush()
        assert 'unrelated remains enabled' in (tmp_path / 'forwarded.log').read_text()
        assert_absent((tmp_path / 'forwarded.log').read_bytes(), ['opaque-private-message'])
    finally:
        second.close()
        root.removeHandler(sink)
        sink.close()
    assert logging.Logger.callHandlers is original


def test_sdk_failure_trace_uses_shared_categorical_api_and_original_exception(tmp_path):
    from gluomem_bench.bench.embedding_runtime import safe_failure_diagnostic
    from gluomem_bench.bench.mem0 import GuardedMem0SDK
    text = 'authored-native-sdk-' + 'private-reason'
    error = urllib.error.HTTPError('https://fixture.invalid', 400, text, {}, io.BytesIO(b'opaque'))
    cause = ValueError('authored cause')
    error.__cause__ = cause
    class Rack:
        def check(self):
            raise error
        def _latch(self, exc):
            assert exc is error
    sdk = GuardedMem0SDK(Rack(), trace=tmp_path / 'native.jsonl', expected_system='authored')
    with pytest.raises(urllib.error.HTTPError) as caught:
        sdk.create()
    assert caught.value is error and error.__cause__ is cause and error.reason == text
    saved = json.loads(sdk.trace.read_bytes())
    assert_absent(sdk.trace.read_bytes(), [text])
    assert saved['error'] == safe_failure_diagnostic(error)


@pytest.mark.parametrize('failure', ['diagnostic', 'write'])
def test_sdk_diagnostic_failure_preserves_original_exception(tmp_path, monkeypatch, failure):
    from gluomem_bench.bench import embedding_runtime
    from gluomem_bench.bench.mem0 import GuardedMem0SDK
    error = ValueError('original exception identity')
    class Rack:
        def check(self):
            raise error
        def _latch(self, exc):
            assert exc is error
    def fail(*args, **kwargs):
        raise OSError('diagnostic unavailable')
    sdk = GuardedMem0SDK(Rack(), trace=tmp_path / 'native.jsonl', expected_system='authored')
    if failure == 'diagnostic':
        monkeypatch.setattr(embedding_runtime, 'safe_failure_diagnostic', fail)
    else:
        monkeypatch.setattr(Path, 'open', fail)
    with pytest.raises(ValueError) as caught:
        sdk.create()
    assert caught.value is error


def test_emit_never_calls_untrusted_record_dictionary(tmp_path):
    class HostileDict(dict):
        def get(self, *args):
            pytest.fail('untrusted record dictionary executed')
    record = logging.LogRecord('mem0.private', logging.WARNING, '', 1, 'private', (), None)
    record.__dict__ = HostileDict(record.__dict__)
    observer = NativeMem0Diagnostics(tmp_path / 'events.json')
    try:
        observer.emit(record)
        assert json.loads(observer.path.read_bytes())[0]['level'] == 'UNKNOWN'
    finally:
        observer.close()


def test_native_memory_close_failure_restores_capture_and_preserves_exception(tmp_path):
    from gluomem_bench.bench.mem0 import close_mem0_memory
    original = logging.Logger.callHandlers
    observer = NativeMem0Diagnostics(tmp_path / 'events.json')
    error = ValueError('original native close exception')
    closed = []
    def fail():
        raise error
    memory = SimpleNamespace(close=fail, vector_store=SimpleNamespace(
        client=SimpleNamespace(close=lambda: closed.append(True))),
        _benchmark_diagnostics=observer)
    with pytest.raises(ValueError) as caught:
        close_mem0_memory(memory)
    assert caught.value is error
    assert closed == [True]
    assert logging.Logger.callHandlers is original
    observer.close()


def test_close_restores_after_exception_and_leaves_preexisting_handlers(tmp_path):
    original = logging.Logger.callHandlers
    child = logging.getLogger('mem0.fixture.restoration')
    stream = io.StringIO()
    sink = logging.StreamHandler(stream)
    child.addHandler(sink)
    state = (tuple(child.handlers), tuple(child.filters), child.level, child.propagate)
    observer = NativeMem0Diagnostics(tmp_path / 'events.json')
    error = ValueError('authored original exception')
    try:
        try:
            raise error
        finally:
            observer.close()
    except ValueError as caught:
        assert caught is error
    assert logging.Logger.callHandlers is original
    assert (tuple(child.handlers), tuple(child.filters), child.level, child.propagate) == state
    try:
        child.warning('logging restored')
        assert stream.getvalue().strip() == 'logging restored'
    finally:
        child.removeHandler(sink)
        sink.close()
