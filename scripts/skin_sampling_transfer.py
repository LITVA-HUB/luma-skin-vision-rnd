"""Combine person coverage and within-person native-color allocation."""
import numpy as np
from skin_color_sampling import sampling_distribution as base_distribution,draw_indices

ARMS=('image','person_site','site','color','color_ipw','person_color')


def sampling_distribution(data,arm):
    if arm!='person_color':return base_distribution(data,arm)
    q,w,info=base_distribution(data,'color');people=np.unique(data['patient'])
    for person in people:
        ix=data['patient']==person;q[ix]/=q[ix].sum()*len(people)
    mass=np.bincount(info['inverse'],weights=q,minlength=info['site_count'])
    return q,np.ones(len(q)),info|{'site_probability':mass}


def protocols(tr,va):
    if not set(tr['patient']).isdisjoint(va['patient']):raise ValueError('Patient leakage')
    def sub(d,mask):return {k:v[mask] for k,v in d.items()}
    yield 'mixed',tr,{'known':va}
    for camera in ('SLR','ipod'):
        t=sub(tr,tr['device']==camera);known=sub(va,va['device']==camera);unseen=sub(va,va['device']!=camera)
        if not all(len(x['patient']) for x in (t,known,unseen)):raise ValueError('Empty protocol')
        yield 'from_'+camera,t,{'known':known,'unseen':unseen}
