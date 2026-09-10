import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
SPEC = importlib.util.spec_from_file_location('phone_bench', Path(__file__).parents[1]/'scripts/cc_phone_benchmark.py')
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_coverage_does_not_accept_refused_rows_or_hide_reference_exclusions():
    result = bench.risk_table(np.array([1., 3., 20.]), np.array([.2, .1, .0]),
                              np.array([True, True, False]), ['a', 'b', 'c'], planned=4)
    assert result['fixed']['100']['accepted'] == 2
    assert result['fixed']['100']['mean'] == 2
    assert result['fixed']['100']['coverage_scorable'] == 2/3
    assert result['fixed']['100']['coverage_planned'] == .5


def test_no_acceptable_prediction_is_explicitly_empty():
    result = bench.risk_table(np.array([50.]), np.array([1.]), np.array([False]), ['a'], 2)
    assert result['fixed']['80']['accepted'] == 0
    assert result['fixed']['80']['mean'] is None
    assert result['curve'] == []


def test_data_hash_failure_precedes_numerical_file_read(tmp_path):
    path = tmp_path/'x.h5'
    path.write_bytes(b'bad')
    with pytest.raises(ValueError, match='digest'):
        bench.verified_path(tmp_path, 'x.h5', {'x.h5': {'sha256': '0'*64}})
    with pytest.raises(ValueError, match='outside'):
        bench.verified_path(tmp_path, '../escape', {})


def test_primary_scene_bootstrap_preserves_phone_pairing():
    draws = bench.scene_draws(['a','a','b','b'], repeats=5, seed=17)
    for ix in draws:
        assert list(ix).count(0) == list(ix).count(1)
        assert list(ix).count(2) == list(ix).count(3)
