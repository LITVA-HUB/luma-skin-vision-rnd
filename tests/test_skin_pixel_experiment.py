import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_confusion_matrix_and_imbalanced_accuracy():
    from skin_pixel_experiment import metrics
    m = metrics(np.array([1, 1, 0, 0, 0, 0]), np.array([.9, .2, .6, .1, .2, .3]), .5)
    assert m['tp'] == 1 and m['fn'] == 1 and m['fp'] == 1 and m['tn'] == 3
    assert m['accuracy'] == pytest.approx(4/6)
    assert m['balanced_accuracy'] == pytest.approx(.625)
    assert m['skin_precision'] == pytest.approx(.5)
    assert m['skin_recall'] == pytest.approx(.5)


def test_winner_uses_validation_balanced_accuracy_only():
    from skin_pixel_experiment import choose
    items = [dict(id='a', validation=dict(balanced_accuracy=.8), test=dict(accuracy=1.0)),
             dict(id='b', validation=dict(balanced_accuracy=.9), test=dict(accuracy=0.0))]
    assert choose(items)['id'] == 'b'


def test_metric_boundary_rejects_bad_probabilities():
    from skin_pixel_experiment import metrics
    with pytest.raises(ValueError):
        metrics(np.array([0, 1]), np.array([np.nan, .8]), .5)
    with pytest.raises(ValueError):
        metrics(np.array([0, 1]), np.array([0, 1.1]), .5)
