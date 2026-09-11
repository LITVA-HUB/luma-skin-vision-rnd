import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from skin_appearance_inverse import design,fit_map,predict_map,posterior,decisions


def test_polynomial_design_and_affine_recovery():
    x=np.array([[1.,2.],[3.,4.]])
    np.testing.assert_array_equal(design(x,2),[[1,1,2,1,2,4],[1,3,4,9,12,16]])
    rng=np.random.default_rng(7);x=rng.normal(size=(40,3));y=x@rng.normal(size=(3,2))+2
    model=fit_map(x,y,1,0.)
    np.testing.assert_allclose(predict_map(model,x),y,atol=1e-12)


def test_marginal_likelihood_keeps_gaussian_normalizer_and_uniform_atom_prior():
    # Two colors and two nuisance modes with different variances.
    means=np.array([[[0.],[2.]],[[1.],[3.]]]);cov=np.array([[[1.]],[[4.]]])
    x=np.array([[0.],[2.]])
    p,log_evidence=posterior(x,means,cov)
    density=np.empty((2,2))
    for i in range(2):
        for a in range(2):
            density[i,a]=sum(np.exp(-.5*(x[i,0]-means[k,a,0])**2/cov[k,0,0])/np.sqrt(2*np.pi*cov[k,0,0]) for k in range(2))/2
    np.testing.assert_allclose(p,density/density.sum(1,keepdims=True),rtol=1e-13)
    np.testing.assert_allclose(log_evidence,np.log(density.mean(1)),rtol=1e-13)
    assert np.allclose(p.sum(1),1)


def test_uniform_likelihood_recovers_prior_and_extreme_inputs_remain_finite():
    means=np.zeros((2,4,3));cov=np.stack([np.eye(3),np.eye(3)*2])
    p,logp=posterior(np.ones((2,3))*1e5,means,cov)
    np.testing.assert_allclose(p,.25,atol=1e-6)
    assert np.isfinite(logp).all()


def test_color_decision_minimizes_actual_expected_palette_error():
    palette=np.array([[40.,10.,20.],[55.,12.,21.],[70.,15.,23.]])
    p=np.array([[.2,.7,.1],[1.,0.,0.]])
    out=decisions(p,palette)
    from luma_skin_vision.color import delta_e00
    cost=np.array([[float(delta_e00(x,y)) for y in palette] for x in palette])
    np.testing.assert_array_equal(out['medoid'],palette[np.argmin(p@cost,axis=1)])
    np.testing.assert_allclose(out['mean'],p@palette)
    np.testing.assert_allclose(out['medoid_risk'],np.min(p@cost,axis=1))
