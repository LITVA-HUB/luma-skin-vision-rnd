import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_interpolated_integration_matches_direct_curve_and_retains_high_values():
    from skin_spectral_palette import integration_matrix, xyz_lab
    knots=np.array([400.,500.,600.])
    wave=np.array([380.,400.,450.,500.,550.,600.,630.])
    cmf=np.arange(21).reshape(7,3)/20+.1
    spd=np.linspace(1,2,7)
    r=np.array([.2,1.4,.8])
    matrix=integration_matrix(knots,wave,cmf,spd)
    direct=np.sum(np.interp(wave,knots,r)[:,None]*cmf*spd[:,None],axis=0)/np.sum(cmf[:,1]*spd)
    np.testing.assert_allclose(r@matrix,direct,rtol=0,atol=1e-14)
    white=matrix.sum(0)
    np.testing.assert_allclose(xyz_lab(white,white),[100,0,0],atol=1e-12)
    assert xyz_lab(white*1.4,white)[0]>100
    with pytest.raises(ValueError):
        integration_matrix(knots[::-1],wave,cmf,spd)


def test_sampling_is_deterministic_without_replacement_and_stays_inside_mask():
    from skin_spectral_palette import sample_indices
    mask=np.zeros((60,70),bool)
    mask[10:50,5:60]=True
    a=sample_indices(mask,'a'*64)
    b=sample_indices(mask,'a'*64)
    np.testing.assert_array_equal(a,b)
    assert len(a)==len(np.unique(a))==1024 and mask.ravel()[a].all()
    assert not np.array_equal(a,sample_indices(mask,'b'*64))
    assert len(sample_indices(np.zeros_like(mask),'a'*64))==0
    assert len(sample_indices(np.ones((3,5),bool),'a'*64))==0
    assert len(sample_indices(np.ones((4,4),bool),'a'*64))==16


def test_qualified_mask_uses_interior_and_source_support():
    from skin_spectral_palette import qualified_mask
    logits=np.full((12,12),10.)
    foreground=np.ones((24,24),bool)
    foreground[12:14,12:14]=False
    selected=qualified_mask(logits,foreground)
    assert not selected[:4].any() and not selected[-4:].any()
    assert not selected[:, :4].any() and not selected[:, -4:].any()
    assert not selected[12:14,12:14].any()
    assert selected[6:10,6:10].all()
