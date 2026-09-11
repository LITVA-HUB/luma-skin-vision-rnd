"""Local support controls: uniform limit, stable mass, held-target exclusion."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_uniform_local_affine_is_global_ridge_with_unpenalized_intercept():
    from skin_local_reference import local_affine
    rng=np.random.default_rng(199);x=rng.normal(size=(17,4));y=rng.normal(size=(17,3));q=rng.normal(size=4)
    xc=x-x.mean(0);yc=y-y.mean(0)
    coef=np.linalg.solve(xc.T@xc+np.eye(4),xc.T@yc)
    expected=y.mean(0)+(q-x.mean(0))@coef
    np.testing.assert_allclose(local_affine(x,y,q,np.ones(17)),expected,atol=1e-14,rtol=1e-12)
    shift=np.array([40.,-5.,20.])
    np.testing.assert_allclose(local_affine(x,y+shift,q,np.ones(17)),expected+shift,atol=1e-13,rtol=1e-12)


def test_kernel_mass_survives_extreme_distances_and_is_scale_invariant():
    from skin_local_reference import normalized_weights
    logs=np.array([-1000001.,-1000000.,-1001000.])
    w=normalized_weights(logs)
    assert np.isfinite(w).all() and (w>=0).all()
    np.testing.assert_allclose(w.sum(),3,atol=1e-15)
    np.testing.assert_allclose(w,normalized_weights(logs+1000000),atol=1e-14,rtol=1e-12)


def test_predictions_do_not_require_query_color_or_identifiers():
    from skin_local_reference import predict_bank
    rng=np.random.default_rng(62);x=rng.normal(size=(15,4));y=rng.normal(size=(15,3))+np.array([50,10,15]);q=rng.normal(size=(3,4))
    got=predict_bank(x,y,q)
    assert set(got['predictions'])=={'global_ridge','global_mean','appearance_mean','appearance_affine','color_mean','color_affine'}
    for v in got['predictions'].values():assert v.shape==(3,3) and np.isfinite(v).all()
    reverse=predict_bank(x[::-1],y[::-1],q)
    for name,v in got['predictions'].items():np.testing.assert_allclose(v,reverse['predictions'][name],atol=1e-12,rtol=1e-11)
    assert got['weights_appearance'].shape==(3,15)
    np.testing.assert_allclose(got['weights_color'].sum(1),15,atol=1e-13)
