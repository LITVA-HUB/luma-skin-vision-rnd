import numpy as np
import pytest
from scripts.skin_issa_color import spectral_xyz, source_lab


def test_percent_reflectance_scale_and_white_are_explicit():
    cmf = np.array([[.2,.1,.7],[.5,.8,.3],[.1,.4,.2]])
    spd = np.array([20.,70.,10.])
    white = spectral_xyz(np.full((1,3), 100.), cmf, spd)[0]
    xyz = spectral_xyz(np.full((1,3), 50.), cmf, spd)
    np.testing.assert_allclose(xyz[0], .5*white, atol=1e-12)
    np.testing.assert_allclose(source_lab(xyz,white)[0], [116*np.cbrt(.5)-16,0,0], atol=1e-12)


def test_missing_spectral_samples_cannot_silently_be_zero_filled():
    with pytest.raises(ValueError, match='finite'):
        spectral_xyz(np.array([[2.,np.nan]]),np.ones((2,3)),np.ones(2))
    np.testing.assert_allclose(source_lab(np.zeros((1,3)),np.ones(3)*100),np.zeros((1,3)),atol=1e-12)
