import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from skin_patch_likelihood import bag_loglik,component_update


def test_single_gaussian_patch_scatter_cancels_across_color_candidates():
    rng=np.random.default_rng(17);x=rng.normal(size=(3,8,3));means=rng.normal(size=(5,1,3))
    a=bag_loglik(x,means,np.eye(3)[None],np.ones(1));b=bag_loglik(x.mean(1,keepdims=True),means,np.eye(3)[None],np.ones(1))
    np.testing.assert_allclose(a-a[:,:1],b-b[:,:1],atol=1e-12)


def test_non_gaussian_bag_contains_information_absent_from_mean():
    x=np.array([[[-1.],[1.]]]);means=np.array([[[-1.],[1.]],[[0.],[0.]]]);cov=np.array([[[.05]],[[.05]]]);pi=np.ones(2)/2
    a=bag_loglik(x,means,cov,pi);b=bag_loglik(x.mean(1,keepdims=True),means,cov,pi)
    assert a.argmax(1)[0]==0 and b.argmax(1)[0]==1
    np.testing.assert_allclose(a,bag_loglik(x[:,::-1],means,cov,pi),atol=1e-12)


def test_one_component_mean_fit_matches_image_weighted_ridge():
    rng=np.random.default_rng(8);d=np.column_stack([np.ones(10),rng.normal(size=(10,3))]);x=rng.normal(size=(10,7,3))
    w,_,_=component_update(d,x,np.ones((10,7,1)),1.)
    penalty=np.eye(4);penalty[0,0]=0
    expected=np.linalg.solve(d.T@d+penalty,d.T@x.mean(1))
    np.testing.assert_allclose(w[0],expected,rtol=1e-12,atol=1e-12)
