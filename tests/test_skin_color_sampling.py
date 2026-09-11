import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from skin_color_sampling import sampling_distribution,draw_indices


def example():
    return {'site':np.array(['a','a','b','c','c','c']), 'patient':np.array(['p','p','p','q','q','q']),
            'target':np.array([[40,10,15],[40,10,15],[42,11,15],[75,4,8],[75,4,8],[75,4,8]],float)}


def test_site_and_person_mass_are_independent_of_photo_duplicates():
    d=example();q,_,_=sampling_distribution(d,'site')
    np.testing.assert_allclose([q[d['site']==s].sum() for s in ('a','b','c')],[1/3]*3)
    q,_,_=sampling_distribution(d,'person_site')
    np.testing.assert_allclose([q[d['patient']==p].sum() for p in ('p','q')],[.5,.5])
    np.testing.assert_allclose([q[d['site']==s].sum() for s in ('a','b','c')],[.25,.25,.5])


def test_importance_correction_recovers_site_objective_exactly():
    d=example();site,_,_=sampling_distribution(d,'site');q,w,_=sampling_distribution(d,'color_ipw')
    np.testing.assert_allclose(q*w,site,rtol=1e-14)
    assert w.max()<=2 and q.min()>0 and abs(q.sum()-1)<1e-14
    assert q[d['site']=='c'].sum()>q[d['site']=='a'].sum()
    loss=np.array([1,2,3,8,4,7.]);assert abs(np.dot(q,w*loss)-np.dot(site,loss))<1e-14


def test_image_draws_match_existing_control_and_weights_do_not_change_draws():
    d=example();q,_,_=sampling_distribution(d,'image')
    np.testing.assert_array_equal(draw_indices(q,'image',17,1),np.random.default_rng(17001).integers(0,6,(31,32),dtype=np.int64))
    a,_,_=sampling_distribution(d,'color');b,_,_=sampling_distribution(d,'color_ipw')
    np.testing.assert_array_equal(draw_indices(a,'color',29,2),draw_indices(b,'color_ipw',29,2))
