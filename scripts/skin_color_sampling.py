"""Selected-TRAIN native-color allocation; no evaluation labels or camera input."""
import numpy as np
from luma_skin_vision.color import delta_e00

ARMS=('image','person_site','site','color','color_ipw')


def sampling_distribution(data,arm):
    if arm not in ARMS:raise ValueError('Unknown sampler')
    sites,first,inverse,counts=np.unique(data['site'],return_index=True,return_inverse=True,return_counts=True)
    target=data['target'][first];owner=data['patient'][first];n=len(inverse);s=len(sites)
    if not np.array_equal(target[inverse],data['target']):raise ValueError('Inconsistent site reference')
    if not np.array_equal(owner[inverse],data['patient']):raise ValueError('Inconsistent site person')
    site=np.ones(s)/s;density=np.ones(s)
    if arm=='image':q=np.ones(n)/n
    else:
        if arm=='person_site':
            people,pi,pc=np.unique(owner,return_inverse=True,return_counts=True)
            site=1/(len(people)*pc[pi])
        elif arm in ('color','color_ipw'):
            de=delta_e00(target[:,None],target[None])
            density=np.exp(-.5*(de/5)**2).mean(1)
            weight=np.clip(np.median(density)/density,1/3,3)
            site=.5/s+.5*weight/weight.sum()
        q=site[inverse]/counts[inverse]
    correction=(1/(s*counts[inverse]))/q if arm=='color_ipw' else np.ones(n)
    if not np.isfinite(q).all() or np.any(q<=0) or not np.isclose(q.sum(),1,rtol=0,atol=1e-12):raise ValueError('Invalid distribution')
    return q,correction,{'site_probability':np.bincount(inverse,weights=q,minlength=s),'density':density,'inverse':inverse,'site_count':s}


def draw_indices(q,arm,seed,epoch):
    rng=np.random.default_rng(seed*1000+epoch)
    if arm=='image':return rng.integers(0,len(q),(31,32),dtype=np.int64)
    cdf=np.cumsum(q);cdf[-1]=1
    return np.searchsorted(cdf,rng.random((31,32)),side='right').astype(np.int64)
