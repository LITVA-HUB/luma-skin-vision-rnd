import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from skin_correspondence import site_decomposition, unique_references


def test_shared_bias_and_view_variation_are_distinguished():
    target = np.array([[50., 0, 0]] * 3 + [[60., 0, 0]])
    prediction = target + np.array([[1., 0, 0], [2., 0, 0], [3., 0, 0], [4., 0, 0]])
    result = site_decomposition(prediction, target, np.array(['a', 'a', 'a', 'b']))
    assert result['image_mse_lab'] == pytest.approx(7.5)
    assert result['image_shared_bias_squared'] == pytest.approx(7.)
    assert result['image_within_site_variance'] == pytest.approx(.5)
    assert result['site_mse_lab'] == pytest.approx((14/3 + 16)/2)
    assert result['site_shared_bias_squared'] == pytest.approx(10.)
    assert result['site_within_site_variance'] == pytest.approx(1/3)


def test_reference_copies_count_once_and_conflicts_fail():
    reps = np.array([[[49., 0, 0], [50., 0, 0], [51., 0, 0]]] * 3)
    rows = unique_references(reps, np.array(['a', 'a', 'b']), np.array(['p', 'p', 'q']))
    assert len(rows) == 2
    assert rows[0]['images'] == 2
    corrupt = reps.copy(); corrupt[1, 0, 0] += 1
    with pytest.raises(ValueError, match='reference'):
        unique_references(corrupt, np.array(['a', 'a', 'b']), np.array(['p', 'p', 'q']))
    with pytest.raises(ValueError, match='patient'):
        unique_references(reps, np.array(['a', 'a', 'b']), np.array(['p', 'q', 'q']))


def test_mismatched_site_targets_cannot_be_interpreted_as_view_error():
    target = np.array([[50., 0, 0], [51., 0, 0]])
    with pytest.raises(ValueError, match='target'):
        site_decomposition(target, target, np.array(['a', 'a']))
