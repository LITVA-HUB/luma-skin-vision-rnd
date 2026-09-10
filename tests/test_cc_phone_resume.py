"""Retry policy must respect server throttling and preserve other failures."""
import importlib.util
from pathlib import Path
from urllib.error import HTTPError

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/cc_phone_resume.py'
    assert path.exists(), 'Polite resume wrapper missing'
    spec = importlib.util.spec_from_file_location('cc_phone_resume', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rate_limit_is_retried_after_server_requested_delay():
    mod = module()
    calls, waits = [], []
    def response(request, **kwargs):
        calls.append((request, kwargs))
        if len(calls) == 1:
            raise HTTPError('https://example.test', 429, 'slow down', {'Retry-After': '120'}, None)
        return b'verified-response'
    opener = mod.polite_opener(response, waits.append)
    assert opener('same-range', timeout=60) == b'verified-response'
    assert calls[0] == calls[1]
    assert sum(waits) >= 120


def test_missing_file_is_not_retried_or_hidden():
    mod = module()
    calls = []
    def response(*args, **kwargs):
        calls.append(1)
        raise HTTPError('https://example.test', 404, 'missing', {}, None)
    with pytest.raises(HTTPError) as exc:
        mod.polite_opener(response, lambda seconds: None)('range')
    assert exc.value.code == 404 and len(calls) == 1
