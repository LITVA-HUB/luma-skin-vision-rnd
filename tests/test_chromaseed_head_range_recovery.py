import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_transient_windows_denial_retries_same_payload(tmp_path):
    from chromaseed_head_range_recovery import retry_writer

    calls, waits, events = [], [], []
    payload = {'step': 128, 'loss': [1., 2.]}
    path = tmp_path / 'progress.json'

    def writer(p, value):
        calls.append((p, value))
        if len(calls) < 3:
            raise PermissionError('Windows reader holds destination')
        p.write_text(json.dumps(value))

    wrapped = retry_writer(writer, path, delays=(.01, .02), sleep=waits.append, report=events.append)
    wrapped(path, payload)
    assert calls == [(path, payload)]*3 and all(v is payload for _, v in calls)
    assert waits == [.01, .02] and json.loads(path.read_text()) == payload
    assert events[-1]['outcome'] == 'retried' and events[-1]['retries'] == 2


def test_only_exhausted_progress_is_nonfatal_and_critical_writes_still_fail(tmp_path):
    from chromaseed_head_range_recovery import retry_writer

    events = []

    def denied(path, value):
        raise PermissionError('busy')

    progress = tmp_path / 'progress.json'
    wrapped = retry_writer(denied, progress, delays=(), report=events.append)
    assert wrapped(progress, {'step': 128}) is None
    assert events[-1]['outcome'] == 'telemetry_skipped'
    with pytest.raises(PermissionError):
        wrapped(tmp_path / 'receipt.json', {'models': 'required'})
    assert events[-1]['outcome'] == 'critical_failed'


def test_non_permission_failures_are_never_swallowed(tmp_path):
    from chromaseed_head_range_recovery import retry_writer

    def invalid(path, value):
        raise ValueError('nonfinite result')

    progress = tmp_path / 'progress.json'
    with pytest.raises(ValueError, match='nonfinite result'):
        retry_writer(invalid, progress)(progress, {})


@pytest.mark.skipif(sys.platform != 'win32', reason='Real Windows file sharing semantics')
def test_original_writer_reproduces_sharing_failure_then_recovers(tmp_path):
    from chromaseed_head_range_recovery import retry_writer
    from skin_local_search_train import write_json

    path = tmp_path / 'progress.json'
    write_json(path, {'old': 1})
    handle = path.open('r', encoding='utf-8')
    try:
        with pytest.raises(PermissionError):
            write_json(path, {'new': 2})
        events = []
        wrapped = retry_writer(write_json, path, delays=(.01,), sleep=lambda _: handle.close(), report=events.append)
        wrapped(path, {'new': 2})
        assert json.loads(path.read_text()) == {'new': 2}
        assert events[-1]['retries'] == 1
    finally:
        handle.close()
