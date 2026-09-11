import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from skin_teacher_readout import fit_blocks, transform_blocks, site_weights, source_permutation


def test_scaler_uses_fit_columns_and_equalizes_block_energy():
    x = np.array([[1., 10], [3., 10]])
    scaler = fit_blocks([x])
    z = transform_blocks([x], scaler)
    np.testing.assert_allclose(z, [[-1/np.sqrt(2), 0], [1/np.sqrt(2), 0]])
    v = transform_blocks([np.array([[101., 20.]])], scaler)
    np.testing.assert_allclose(v, [[99/np.sqrt(2), 10/np.sqrt(2)]])
    assert scaler[0]['mean'].tolist() == [2, 10]


def test_site_weight_does_not_multiply_repeated_views():
    w = site_weights(np.array(['a', 'a', 'a', 'b']))
    assert w.mean() == pytest.approx(1)
    assert w[:3].sum() == pytest.approx(w[3])
    np.testing.assert_allclose(w, [2/3, 2/3, 2/3, 2])


def test_shuffles_stay_within_subset_and_are_repeatable():
    p = source_permutation(17, 11, 'fit')
    np.testing.assert_array_equal(np.sort(p), np.arange(17))
    np.testing.assert_array_equal(p, source_permutation(17, 11, 'fit'))
    assert not np.array_equal(p, source_permutation(17, 11, 'selection'))
    with pytest.raises(ValueError):
        source_permutation(17, 11, 'test')
