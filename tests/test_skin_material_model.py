import numpy as np
import torch
from scripts.skin_material_prior import material_value_jacobian
from scripts.skin_material_model import MaterialImage, ARMS


def prior():
    rng=np.random.default_rng(1);mu=np.linspace(-3.,-.2,31);basis=rng.normal(size=(31,8))*.1
    matrix=rng.uniform(.001,.1,size=(31,3));matrix/=matrix[:,1].sum();white=matrix.sum(0)
    base,jac=material_value_jacobian(mu,basis,matrix,white)
    return dict(mu=mu,basis=basis,matrix=matrix,white=white,base=base,jacobian=jac)


def test_all_arms_have_matched_states_and_finite_gradients():
    states=[]
    for arm in ARMS:
        torch.manual_seed(9);m=MaterialImage(arm,prior(),np.zeros(3),np.ones(3))
        states.append(m.state_dict())
        p,g,h,r=m(torch.ones(2,64,18)*.2)
        assert p.shape==(2,3) and h.shape==(2,4,3)
        (p.square().mean()+g.square().mean()).backward()
        assert all(torch.isfinite(q.grad).all() for q in m.parameters() if q.grad is not None)
    assert all(torch.equal(a,states[0][k]) for s in states[1:] for k,a in s.items())


def test_tangent_matches_material_value_and_derivative_at_origin():
    p=prior();m=MaterialImage('material',p,np.zeros(3),np.ones(3)).double()
    z=torch.zeros(8,dtype=torch.double,requires_grad=True)
    value=m.material_decode(z)
    jac=torch.autograd.functional.jacobian(m.material_decode,z).T
    # Buffers originate as deployment float32; comparison uses their actual values.
    expected,j=material_value_jacobian(m.mu.numpy(),m.basis.numpy(),m.matrix.numpy(),m.white.numpy())
    np.testing.assert_allclose(value.detach().numpy(),expected,atol=1e-10)
    np.testing.assert_allclose(jac.detach().numpy(),j,atol=1e-10)
