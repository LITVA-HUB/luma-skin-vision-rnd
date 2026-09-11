import numpy as np
from scripts.skin_issa_material import fit_basis, encode, decode, subject_weights


def test_subject_weighting_resists_record_multiplicity():
    x=np.array([[1.,0.,2.],[2.,1.,3.],[4.,3.,1.],[5.,1.,2.]])
    people=np.array(['a','a','b','c'])
    mean, basis=fit_basis(x,subject_weights(people))
    more=np.concatenate([x,x[:2]])
    m2,b2=fit_basis(more,subject_weights(np.concatenate([people,people[:2]])))
    np.testing.assert_allclose(m2,mean,atol=1e-12)
    np.testing.assert_allclose(basis@basis.T,b2@b2.T,atol=1e-12)
    np.testing.assert_allclose((x-mean)@basis@basis.T+mean,x,atol=1e-12)


def test_physical_transforms_roundtrip_without_training_on_evaluation_data():
    x=np.array([[.01,.3,.8],[.6,.9,.99]])
    for method in ('reflectance','density','logit'):
        np.testing.assert_allclose(decode(encode(x,method),method),x,atol=1e-14)
