import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
SPEC = importlib.util.spec_from_file_location(
    'phone_prepare', Path(__file__).parents[1]/'scripts/cc_phone_prepare.py')
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def patch(rgb, saturated=0):
    rgb = np.array(rgb, dtype=float)
    return {'median_rgb': rgb.tolist(), 'sample_count': 100,
            'saturated_pixel_fraction': saturated}


def test_neutral_fallback_is_reference_only_and_preserves_chromaticity():
    patches = {'20': patch([.4, 1, .6], 1), '21': patch([.2, .4, .3]),
               '22': patch([.1, .2, .15])}
    result = prepare.reference_from_patches(patches)
    assert result['valid'] and result['selected_patch'] == 21
    np.testing.assert_allclose(result['gt'], np.array([2, 4, 3])/np.sqrt(29))


def test_dark_or_disagreeing_reference_is_not_scored():
    p = {'20': patch([.2, .4, .3]), '21': patch([.01, .02, .015]),
         '22': patch([.005, .01, .0075])}
    result = prepare.reference_from_patches(p)
    assert not result['valid'] and result['gt'] is None
    p['21'] = patch([.4, .2, .3])
    result = prepare.reference_from_patches(p)
    assert not result['valid'] and 'disagreement' in result['reason']


def test_brightest_allowed_gray_selected_consistently():
    p = {str(i): patch(np.array([.2, .4, .3])/(i-19)) for i in (20, 21, 22)}
    result = prepare.reference_from_patches(p)
    assert result['valid'] and result['selected_patch'] == 20
    assert result['max_pairwise_degrees'] < 1e-6
