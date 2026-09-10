"""Independent scoring and artifact-integrity regression checks."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def reporter():
    path = Path(__file__).resolve().parents[1] / 'scripts/cc_v5_report.py'
    assert path.exists(), 'V5 independent scorer missing'
    spec = importlib.util.spec_from_file_location('cc_v5_report', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_physical_error_and_integer_coverage():
    r = reporter()
    pred = np.array([[1., 1., 1.], [2., 1., 1.]])
    gt = np.array([[2., 1., 1.], [2., 1., 1.]])
    error = r.reproduction_degrees(pred, gt)
    assert error.tolist() == pytest.approx([19.47122063449, 0.], abs=1e-10)
    result = r.selective(np.array([1., 2., 9., 20., 30.]), np.array([3., 2., 1., 4., 5.]))
    assert result['fixed']['80']['accepted'] == 4
    assert result['fixed']['60']['mean'] == 4.
    assert len(result['curve']) == 5
    with pytest.raises(ValueError):
        r.reproduction_degrees(np.zeros((1, 3)), np.ones((1, 3)))


def test_manifest_rejects_modified_payload_and_parent_paths(tmp_path):
    r = reporter()
    (tmp_path / 'payload').write_bytes(b'abc')
    (tmp_path / 'artifact_manifest.json').write_text(json.dumps({'sha256': {
        'payload': 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'}}))
    r.verify_manifest(tmp_path)
    (tmp_path / 'payload').write_bytes(b'abd')
    with pytest.raises(ValueError):
        r.verify_manifest(tmp_path)
    (tmp_path / 'artifact_manifest.json').write_text(json.dumps({'sha256': {'../outside': 'bad'}}))
    with pytest.raises(ValueError):
        r.verify_manifest(tmp_path)
