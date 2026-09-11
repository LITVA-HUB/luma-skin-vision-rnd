"""Relational degeneracy, matched controls and excluded-person regression."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_additive_reference_bank_and_cycle_projection_are_not_new_information():
    from skin_relational_probe import anchor_average,complete_potential
    f=np.array([[1.,2.,3.],[4.,6.,8.],[-2.,1.,5.]])
    target=f+np.array([[.1,.2,.3],[.5,-.2,1.],[-.4,2.,0.]])
    x=np.array([[10.,4.,2.],[.2,3.,4.]])
    np.testing.assert_allclose(anchor_average(x[:,None]-f[None],target),x+(target-f).mean(0))
    edges=f[:,None]-f[None]
    potential,residual=complete_potential(edges)
    np.testing.assert_allclose(potential,f-f.mean(0),atol=1e-15)
    np.testing.assert_allclose(residual,0,atol=1e-15)
    cycle=np.array([[0,1,-1],[-1,0,1],[1,-1,0]])[:,:,None]*np.array([1.,2.,3.])
    p,r=complete_potential(edges+cycle)
    np.testing.assert_allclose(p,potential,atol=1e-15)
    np.testing.assert_allclose(r,cycle,atol=1e-15)


def test_controls_preserve_person_capture_and_exclude_reference_site():
    from skin_relational_probe import matched_controls
    z={'patient':np.array(['p']*6+['q']*2),'site':np.array(['a','a','b','b','c','c','d','d']),
       'mode':np.array(['x','y']*4),'image_type':np.array(['clinical']*8),
       'target':np.array([[50.,10.,15.]]*2+[[51.,10.,15.]]*2+[[65.,10.,15.]]*2+[[50.,10.,15.]]*2)}
    r=matched_controls(z)
    assert r['total_positive_pairs']==4 and len(r['positive'])==3 and r['unmatched']==1
    for (a,b),u,n in zip(r['positive'],r['uniform'],r['near']):
        for c in (u,n):
            assert z['patient'][a]==z['patient'][c]
            assert z['site'][a]!=z['site'][c]
            assert z['mode'][b]==z['mode'][c]
            assert z['image_type'][b]==z['image_type'][c]
    assert r['near'][0]==3


def test_excluded_person_regression_cannot_use_held_targets():
    from skin_relational_probe import loo_ridge
    rng=np.random.default_rng(51);x=rng.normal(size=(12,4));y=rng.normal(size=(12,3));person=np.repeat(np.arange(4),3)
    p,folds=loo_ridge(x,y,person)
    changed=y.copy();changed[person==2]+=1e5
    q,_=loo_ridge(x,changed,person)
    np.testing.assert_array_equal(p[person==2],q[person==2])
    assert len(folds)==4
    for f in folds:
        assert not set(f['train_indices'])&set(f['held_indices'])
        assert len(set(person[f['held_indices']]))==1
