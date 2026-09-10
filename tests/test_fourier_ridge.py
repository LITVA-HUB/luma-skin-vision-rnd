import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from cc_fourier_ridge import fit_filter, predict_score


def test_fourier_solution_matches_real_linear_system():
    rng = np.random.default_rng(8)
    hist = rng.normal(size=(12,2,4,4))
    target = rng.normal(size=(12,4,4))
    weight = fit_filter(hist,target,.1)
    h = np.fft.fft2(hist).transpose(0,2,3,1)
    a = np.concatenate([h,np.ones((*h.shape[:-1],1))],axis=-1)
    y = np.fft.fft2(target)
    residual = np.einsum("nhwc,hwc->nhw",a,weight)-y
    gradient = np.einsum("nhwc,nhw->hwc",a.conj(),residual)/len(hist)+.1*weight
    np.testing.assert_allclose(gradient,0,atol=1e-12)
    score = predict_score(hist,weight)
    assert score.shape == target.shape and np.isfinite(score).all()


def test_zero_ridge_recovers_known_circular_operator():
    rng = np.random.default_rng(3)
    hist = rng.normal(size=(10,2,4,4))
    true = np.fft.fft2(rng.normal(size=(3,4,4))).transpose(1,2,0)
    target = predict_score(hist,true)
    fitted = fit_filter(hist,target,1e-12)
    unseen = rng.normal(size=(3,2,4,4))
    np.testing.assert_allclose(predict_score(unseen,fitted),predict_score(unseen,true),atol=1e-9)


def test_bias_ablation_removes_prior_coefficient_exactly():
    from cc_fourier_ridge_v2 import fit_filter as fit_v2
    rng=np.random.default_rng(14)
    hist=rng.normal(size=(10,2,4,4))
    target=rng.normal(size=(10,4,4))
    weight=fit_v2(hist,target,.01,bias=False)
    np.testing.assert_array_equal(weight[...,2],0)
    assert np.isfinite(predict_score(hist,weight)).all()


def test_bias_on_is_exact_original_fit():
    from cc_fourier_ridge_v2 import fit_filter as fit_v2
    rng=np.random.default_rng(15)
    hist=rng.normal(size=(10,2,4,4))
    target=rng.normal(size=(10,4,4))
    np.testing.assert_array_equal(fit_filter(hist,target,.01),fit_v2(hist,target,.01,bias=True))
