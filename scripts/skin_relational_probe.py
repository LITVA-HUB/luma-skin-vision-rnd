"""Control pairs and additive-relational degeneracy; native instrument targets."""
import itertools
import numpy as np
from luma_skin_vision.color import delta_e00


def anchor_average(relative,reference):
    return (relative+reference[None]).mean(1)


def complete_potential(edges):
    if edges.ndim!=3 or edges.shape[0]!=edges.shape[1]:raise ValueError('Complete square graph required')
    if not np.allclose(edges,-edges.swapaxes(0,1),atol=1e-12,rtol=1e-12):raise ValueError('Antisymmetric comparisons required')
    potential=edges.mean(1)
    return potential,edges-(potential[:,None]-potential[None])


def matched_controls(z):
    rng=np.random.default_rng(69117);positive=[];uniform=[];near=[];counts=[];total=0;missing=0
    for site in np.unique(z['site']):
        for a,b in itertools.combinations(np.flatnonzero(z['site']==site),2):
            total+=1
            candidates=np.flatnonzero((z['patient']==z['patient'][a])&(z['site']!=site)&
                (z['mode']==z['mode'][b])&(z['image_type']==z['image_type'][b]))
            if len(candidates)==0:missing+=1;continue
            de=delta_e00(z['target'][a],z['target'][candidates])
            positive.append((a,b));uniform.append(rng.choice(candidates));near.append(candidates[np.argmin(de)]);counts.append(len(candidates))
    return {'positive':np.array(positive,dtype=np.int64).reshape(-1,2),'uniform':np.array(uniform,dtype=np.int64),
            'near':np.array(near,dtype=np.int64),'candidate_counts':np.array(counts,dtype=np.int64),
            'total_positive_pairs':total,'unmatched':missing}


def loo_ridge(features,target,person):
    x=np.asarray(features,dtype=np.float64);y=np.asarray(target,dtype=np.float64)
    result=np.empty_like(y);folds=[]
    for identity in np.unique(person):
        tr=np.flatnonzero(person!=identity);ev=np.flatnonzero(person==identity)
        if not len(tr):raise ValueError('At least two people required')
        mean=x[tr].mean(0);std=np.maximum(x[tr].std(0),1e-6);center=y[tr].mean(0)
        xx=(x[tr]-mean)/std;coef=np.linalg.solve(xx.T@xx+np.eye(x.shape[1]),xx.T@(y[tr]-center))
        result[ev]=(x[ev]-mean)/std@coef+center
        folds.append({'train_indices':tr,'held_indices':ev,'mean':mean,'std':std,'center':center,'coef':coef})
    return result,folds


def excluded_standardize(x,person):
    x=np.asarray(x,dtype=np.float64);out=np.empty_like(x)
    for identity in np.unique(person):
        held=person==identity;other=x[~held];out[held]=(x[held]-other.mean(0))/np.maximum(other.std(0),1e-6)
    return out


def paired_preference(positive,negative,person):
    wins=(positive<negative).astype(float)+.5*(positive==negative)
    return {'pairs':len(wins),'pooled_preference':float(wins.mean()),
            'person_mean_preference':float(np.mean([wins[person==p].mean() for p in np.unique(person)])),
            'people':len(np.unique(person)),'mean_positive_distance':float(positive.mean()),'mean_negative_distance':float(negative.mean())}
