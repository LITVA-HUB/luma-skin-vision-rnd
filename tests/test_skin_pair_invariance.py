import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from skin_pair_invariance import pair_indices,nuisance_transform,vicreg_loss


def test_pairs_never_cross_sites_and_use_distinct_views_when_available():
    sites=np.array(['a','a','b','b','b','c'])
    a,b=pair_indices(sites,30,np.random.default_rng(17))
    np.testing.assert_array_equal(sites[a],sites[b])
    assert np.all(a[sites[a]!='c']!=b[sites[a]!='c'])
    assert set(np.r_[a,b])<=set(range(6))


def test_nuisance_projection_removes_known_capture_direction_but_preserves_color():
    # First coordinate changes within each site; second is constant skin signal.
    tokens=np.array([[[u,v]] for v in [-2,0,3] for u in [-1,1]],dtype=float)
    sites=np.repeat(['a','b','c'],2)
    center,scale,q,spectrum=nuisance_transform(tokens,sites,1)
    transformed=((tokens-center)/scale)@q
    np.testing.assert_allclose(transformed[::2],transformed[1::2],atol=1e-12)
    assert np.ptp(transformed[:,:,1])>1
    np.testing.assert_allclose(q@q,q,atol=1e-12)


def test_vicreg_penalizes_collapsed_representation_and_has_finite_gradients():
    a=torch.zeros(8,4,requires_grad=True);b=torch.zeros(8,4,requires_grad=True)
    loss=vicreg_loss(a,b)
    assert loss.item()>20
    loss.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
