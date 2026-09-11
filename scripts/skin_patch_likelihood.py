"""Latent conditional patch distributions without camera or capture-mode input."""
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from skin_appearance_inverse import design


def bag_loglik(x,means,cov,pi):
    likelihood=[]
    for k in range(len(pi)):
        precision=np.linalg.inv(cov[k]);sign,logdet=np.linalg.slogdet(cov[k]);assert sign==1
        mu=means[:,k];xp=x@precision
        square=np.sum(x*xp,axis=-1)[:,:,None]+np.sum((mu@precision)*mu,axis=1)[None,None]
        square-=2*np.einsum('npd,ad->npa',xp,mu)
        likelihood.append(-.5*(square+logdet+x.shape[-1]*np.log(2*np.pi))+np.log(pi[k]))
    return logsumexp(np.stack(likelihood,axis=-1),axis=-1).mean(1)


def component_update(d,x,r,alpha):
    n,patches,channels=x.shape;k=r.shape[-1];weights=[];cov=[]
    penalty=np.eye(d.shape[1])*alpha;penalty[0,0]=0
    mass=r.sum(1)/patches;response=np.einsum('npk,npd->nkd',r,x)/patches
    for c in range(k):
        w=np.linalg.solve(d.T@(d*mass[:,c,None])+penalty,d.T@response[:,c]);weights.append(w)
        error=x-(d@w)[:,None]
        moment=np.einsum('np,npd,npe->de',r[:,:,c],error,error)/max(float(r[:,:,c].sum()),1e-12)
        cov.append(.9*moment+.1*np.diag(np.diag(moment))+np.eye(channels)*1e-3)
    pi=(r.sum((0,1))+1)/(n*patches+k)
    return np.stack(weights),np.stack(cov),pi


def fit_patch(data,representation,degree,components,seed):
    x=data['tokens'].astype(np.float64)[:,:,9:12]
    if representation=='mean':x=x.mean(1,keepdims=True)
    elif representation!='bag':raise ValueError('Unknown representation')
    y=data['target'];yc=y.mean(0);ys=np.maximum(y.std(0),1e-6)
    xc=x.mean((0,1));xs=np.maximum(x.std((0,1)),1e-6);z=(x-xc)/xs;d=design((y-yc)/ys,degree)
    r=np.ones((*z.shape[:2],1));initial,_,_=component_update(d,z,r,1.)
    if components>1:
        residual=z-(d@initial[0])[:,None]
        labels=KMeans(n_clusters=components,n_init=1,max_iter=100,random_state=seed).fit_predict(residual.reshape(-1,3))
        r=np.eye(components)[labels].reshape(*z.shape[:2],components)*.99+.01/components
    history=[]
    for iteration in range(35):
        weight,cov,pi=component_update(d,z,r,1.);ll=[]
        for k in range(components):
            error=z-(d@weight[k])[:,None];precision=np.linalg.inv(cov[k]);sign,logdet=np.linalg.slogdet(cov[k]);assert sign==1
            ll.append(-.5*(np.einsum('npd,df,npf->np',error,precision,error)+logdet+3*np.log(2*np.pi))+np.log(pi[k]))
        ll=np.stack(ll,axis=-1);normalizer=logsumexp(ll,axis=-1)
        r=np.exp(ll-normalizer[:,:,None]);assert np.isfinite(r).all()
        history.append({'iteration':iteration+1,'mean_training_patch_loglik':float(normalizer.mean()),'smallest_component_prior':float(pi.min())})
    _,first=np.unique(data['site'],return_index=True);palette=y[first].copy()
    return {'yc':yc,'ys':ys,'xc':xc,'xs':xs,'weight':weight,'cov':cov,'pi':pi,'degree':np.array(degree),
        'representation':np.array(representation),'palette':palette},history


def likelihood(model,tokens,collapse=False):
    x=tokens.astype(np.float64)[:,:,9:12]
    if str(model['representation'])=='mean' or collapse:x=x.mean(1,keepdims=True)
    x=(x-model['xc'])/model['xs'];d=design((model['palette']-model['yc'])/model['ys'],int(model['degree']))
    means=np.einsum('aq,kqd->akd',d,model['weight'])
    return np.concatenate([bag_loglik(x[i:i+8],means,model['cov'],model['pi']) for i in range(0,len(x),8)])


def color_output(loglik,palette,strength):
    logits=loglik*strength;p=np.exp(logits-logsumexp(logits,axis=1)[:,None]);p/=p.sum(1,keepdims=True)
    return p@palette,p
