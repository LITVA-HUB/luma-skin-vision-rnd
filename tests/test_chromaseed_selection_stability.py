import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_people_not_images_or_prediction_ensemble():
    from chromaseed_selection_stability import person_losses

    from luma_skin_vision.color import delta_e00

    target = np.tile([50., 0., 0.], (4, 1))
    prediction = np.stack([target.copy(), target.copy()])
    prediction[0, :, 0] += [10., 10., 10., 0.]
    prediction[1, :, 0] -= [10., 10., 10., 0.]
    actual = person_losses(prediction, target, np.array([7, 7, 7, 9]))
    expected_first = np.mean([delta_e00(prediction[s, :1], target[:1])[0] for s in range(2)])
    np.testing.assert_allclose(actual, [expected_first, 0.], rtol=0, atol=1e-12)
    assert actual.mean() > 0.  # Averaged predictions would falsely give zero.
    assert actual.mean() != pytest.approx(expected_first * .75)


def test_stratified_counts_preserve_camera_people_and_are_reproducible():
    from chromaseed_selection_stability import bootstrap_counts

    camera = np.array(['a', 'b', 'b', 'a', 'b'])
    counts = bootstrap_counts(camera, 1000, 19)
    assert counts.shape == (1000, 5)
    assert counts.dtype.kind in 'iu'
    np.testing.assert_array_equal(counts[:, camera == 'a'].sum(axis=1), 2)
    np.testing.assert_array_equal(counts[:, camera == 'b'].sum(axis=1), 3)
    np.testing.assert_array_equal(counts, bootstrap_counts(camera, 1000, 19))
    assert np.any(counts > 1)


def configs():
    return [dict(steps=4, alpha=.01, width_factor=1.),
            dict(steps=0, alpha=.1, width_factor=2.),
            dict(steps=0, alpha=.01, width_factor=2.),
            dict(steps=0, alpha=.01, width_factor=1.)]


def test_exact_ties_follow_registered_order():
    from chromaseed_selection_stability import winner_indices

    scores = np.ones((2, 4))
    scores[1, 0] -= 1e-12  # Do not collapse near ties.
    np.testing.assert_array_equal(winner_indices(scores, configs()), [3, 0])


def test_hand_computable_deletion_bootstrap_and_paired_gaps():
    from chromaseed_selection_stability import diagnose

    losses = np.array([[0., 0., 6.], [3., 3., 3.]])
    cfg = [dict(steps=0, alpha=.01, width_factor=1.), dict(steps=1, alpha=.1, width_factor=1.)]
    counts = np.array([[3, 0, 0], [0, 0, 3], [1, 1, 1], [0, 3, 0]])
    result, trace = diagnose(losses, cfg, counts, parent_index=1)
    assert result['original_index'] == 0
    np.testing.assert_array_equal(trace['deletion_winners'], [0, 0, 0])
    np.testing.assert_array_equal(trace['bootstrap_winners'], [0, 1, 0, 0])
    np.testing.assert_array_equal(trace['bootstrap_original_ranks'], [1, 2, 1, 1])
    np.testing.assert_array_equal(trace['gap_vs_parent'], [-3., 3., -1., -3.])
    assert result['bootstrap_original_frequency'] == .75
    assert result['bootstrap_weak_alpha_frequency'] == .75
    assert result['deletion_switches'] == 0
    assert result['paired_vs_parent']['mean'] == -1.


def test_one_person_can_change_selection_without_refitting():
    from chromaseed_selection_stability import diagnose

    losses = np.array([[0., 5., 5.], [4., 3., 3.]])
    cfg = [dict(steps=0, alpha=.01, width_factor=1.), dict(steps=1, alpha=.1, width_factor=1.)]
    result, trace = diagnose(losses, cfg, np.array([[1, 1, 1]]), parent_index=1)
    assert result['original_index'] == 0
    assert result['deletion_switches'] == 1
    np.testing.assert_array_equal(trace['deletion_winners'], [1, 0, 0])


@pytest.mark.parametrize('bad', [np.array([[np.nan, 2.], [1., 2.]]), np.ones((2, 1)), np.ones((1, 3))])
def test_invalid_scores_fail(bad):
    from chromaseed_selection_stability import diagnose

    with pytest.raises(ValueError):
        diagnose(bad, configs()[:2], np.ones((2, bad.shape[1]), dtype=int), parent_index=0)
