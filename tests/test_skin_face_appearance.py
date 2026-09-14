import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_color_reference_depends_only_on_masked_pixels_and_empty_is_missing():
    from skin_face_appearance import appearance
    rgb = np.zeros((8,8,3),np.uint8)
    rgb[:4] = [180,120,100]
    mask = np.zeros((8,8),bool)
    mask[:4] = True
    a = appearance(rgb,mask)
    rgb[4:] = [10,250,70]
    np.testing.assert_array_equal(a['median_lab'],appearance(rgb,mask)['median_lab'])
    np.testing.assert_array_equal(a['mean_lab'],appearance(rgb,mask)['mean_lab'])
    assert appearance(rgb,np.zeros_like(mask)) is None


def test_surrogate_color_error_is_zero_for_matching_masks():
    from skin_face_appearance import appearance, color_error
    rng = np.random.default_rng(123)
    rgb = rng.integers(0,256,(10,10,3),dtype=np.uint8)
    a = appearance(rgb,np.ones((10,10),bool))
    assert color_error(a,a)['median_delta_e00'] == 0
    assert color_error(a,a)['mean_delta_e00'] == 0
