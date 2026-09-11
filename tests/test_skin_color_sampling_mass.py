import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from skin_color_sampling import sampling_distribution as original
from skin_color_sampling_mass import sampling_distribution


def test_person_mass_preserved_when_color_association_removed():
    d={'site':np.array(['a','a','b','c','d']), 'patient':np.array(['p','p','p','q','q']),
       'target':np.array([[40,10,15],[40,10,15],[41,11,15],[42,9,16],[80,4,10]],float)}
    base,_,info=original(d,'color')
    for arm in ('person_mass','within_person_shuffle'):
        q,w,meta=sampling_distribution(d,arm,17)
        assert np.all(q>0) and abs(q.sum()-1)<1e-14
        for p in ('p','q'):np.testing.assert_allclose(q[d['patient']==p].sum(),base[d['patient']==p].sum(),rtol=1e-14)
        if arm=='within_person_shuffle':np.testing.assert_allclose(np.sort(meta['site_probability']),np.sort(info['site_probability']),rtol=1e-14)
        else:
            np.testing.assert_allclose(q[d['site']=='a'].sum(),q[d['site']=='b'].sum(),rtol=1e-14)
            np.testing.assert_allclose(q[d['site']=='c'].sum(),q[d['site']=='d'].sum(),rtol=1e-14)
