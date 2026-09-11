"""Standard local-reference regressions; no query references or camera identity."""
import numpy as np
from luma_skin_vision.color import delta_e00

METHODS=('global_ridge','global_mean','appearance_mean','appearance_affine','color_mean','color_affine')


def normalized_weights(log_weights):
    z=np.asarray(log_weights,dtype=np.float64)
    if z.ndim!=1 or not len(z) or not np.isfinite(z).all():raise ValueError('Finite vector required')
    w=np.exp(z-z.max());return w*(len(w)/w.sum())


def local_affine(x,y,query,weights):
    weights=np.asarray(weights,dtype=float);mass=weights.sum()
    if mass<=0 or np.any(weights<0):raise ValueError('Positive mass required')
    xm=weights@x/mass;ym=weights@y/mass;xc=x-xm;yc=y-ym
    coef=np.linalg.solve(xc.T@(weights[:,None]*xc)+np.eye(x.shape[1]),xc.T@(weights[:,None]*yc))
    return ym+(query-xm)@coef


def predict_bank(bank_features,bank_target,query_features):
    x=np.asarray(bank_features,dtype=np.float64);y=np.asarray(bank_target,dtype=np.float64);q=np.asarray(query_features,dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(q).all():raise ValueError('Nonfinite inputs')
    mean=x.mean(0);std=np.maximum(x.std(0),1e-6);center=y.mean(0);xx=(x-mean)/std;qq=(q-mean)/std
    coef=np.linalg.solve(xx.T@xx+np.eye(x.shape[1]),xx.T@(y-center))
    base=qq@coef+center;bank_pred=xx@coef+center
    pred={'global_ridge':base,'global_mean':np.broadcast_to(center,(len(q),3)).copy()}
    for name in METHODS[2:]:pred[name]=np.empty((len(q),3))
    wa=[];wc=[];risk=[];uniform_gap=0.
    for j,query in enumerate(qq):
        d2=np.mean((xx-query)**2,1);a=normalized_weights(-.5*d2)
        d=delta_e00(base[j],bank_pred);c=normalized_weights(-.5*(d/5.)**2)
        wa.append(a);wc.append(c);risk.append(float(np.sqrt(d2.min())))
        for affinity,w in [('appearance',a),('color',c)]:
            pred[affinity+'_mean'][j]=w@y/w.sum()
            pred[affinity+'_affine'][j]=local_affine(xx,y,query,w)
        uniform_gap=max(uniform_gap,float(np.max(abs(local_affine(xx,y,query,np.ones(len(x)))-base[j]))))
    return {'predictions':pred,'weights_appearance':np.array(wa),'weights_color':np.array(wc),'risk':np.array(risk),
            'mean':mean,'std':std,'center':center,'coef':coef,'bank_prediction':bank_pred,'uniform_limit_max_gap':uniform_gap}
