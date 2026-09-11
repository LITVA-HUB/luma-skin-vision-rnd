"""Small conditional appearance likelihoods; actual native skin colors as atoms."""
import numpy as np
from scipy.special import logsumexp
from luma_skin_vision.color import delta_e00


def design(x,degree):
    parts=[np.ones((len(x),1)),x]
    if degree==2:parts.append(np.stack([x[:,i]*x[:,j] for i in range(x.shape[1]) for j in range(i,x.shape[1])],axis=1))
    elif degree!=1:raise ValueError('Degree must be 1 or 2')
    return np.concatenate(parts,axis=1)


def fit_map(x,y,degree,alpha):
    xc=x.mean(0);xs=np.maximum(x.std(0),1e-6);yc=y.mean(0);ys=np.maximum(y.std(0),1e-6)
    d=design((x-xc)/xs,degree);penalty=np.eye(d.shape[1])*alpha;penalty[0,0]=0
    weight=np.linalg.solve(d.T@d+penalty,d.T@((y-yc)/ys))
    return {'xc':xc,'xs':xs,'yc':yc,'ys':ys,'weight':weight,'degree':np.array(degree)}


def predict_map(model,x):
    return (design((x-model['xc'])/model['xs'],int(model['degree']))@model['weight'])*model['ys']+model['yc']


def posterior(x,means,cov):
    # means K x A x D, covariance K x D x D. All arithmetic is float64.
    likelihood=[]
    for mu,c in zip(means,cov):
        sign,logdet=np.linalg.slogdet(c)
        if sign!=1:raise ValueError('Nonpositive covariance')
        error=x[:,None]-mu[None];precision=np.linalg.inv(c)
        q=np.einsum('nad,df,naf->na',error,precision,error)
        likelihood.append(-.5*(q+logdet+x.shape[1]*np.log(2*np.pi)))
    marginal=logsumexp(np.stack(likelihood),axis=0)-np.log(len(means))
    logtotal=logsumexp(marginal,axis=1)
    p=np.exp(marginal-logtotal[:,None])
    # Renormalize after exponentiation for extreme floating-point inputs.
    p/=p.sum(1,keepdims=True)
    return p,logtotal-np.log(means.shape[1])


def decisions(p,palette):
    cost=delta_e00(palette[:,None],palette[None])
    expected=p@cost;index=np.argmin(expected,axis=1);mean=p@palette
    return {'mean':mean,'medoid':palette[index],'index':index,
        'mean_risk':np.sum(p*delta_e00(mean[:,None],palette[None]),axis=1),
        'medoid_risk':expected[np.arange(len(p)),index]}


def image_features(data,kind):
    x=data['tokens'].astype(np.float64).mean(1)
    if kind=='rgb':return x[:,9:12]
    if kind=='stats':return x
    raise ValueError('Unknown feature representation')


def fit_forward(data,kind,grouping,degree,alpha):
    x=image_features(data,kind);y=data['target'];xc=x.mean(0);xs=np.maximum(x.std(0),1e-6)
    labels=data['mode'] if grouping=='mode' else np.zeros(len(x),dtype=int)
    modes=np.unique(labels);models=[];covariances=[];oof=np.empty_like(x);fold_count=0
    for mode in modes:
        ix=labels==mode
        models.append(fit_map(y[ix],x[ix],degree,alpha))
        for person in np.unique(data['patient'][ix]):
            train=ix & (data['patient']!=person);held=ix & (data['patient']==person)
            assert train.sum()>degree+1 and not np.any(train & held)
            fit=fit_map(y[train],x[train],degree,alpha)
            oof[held]=predict_map(fit,y[held]);fold_count+=1
        error=(x[ix]-oof[ix])/xs
        second=error.T@error/len(error)
        covariances.append(.8*second+.2*np.diag(np.diag(second))+np.eye(len(xs))*1e-4)
    unique,first=np.unique(data['site'],return_index=True);palette=y[first].copy()
    for site,ref in zip(unique,palette):assert np.all(y[data['site']==site]==ref)
    result={'xc':xc,'xs':xs,'cov':np.stack(covariances),'palette':palette,'groups':np.array(len(modes)),
        'oof_prediction':oof,'oof_folds':np.array(fold_count)}
    for k,model in enumerate(models):
        for key,value in model.items():result[f'map{k}_{key}']=value
    return result


def forward_means(model):
    means=[]
    for k in range(int(model['groups'])):
        prefix=f'map{k}_';mapping={key[len(prefix):]:value for key,value in model.items() if key.startswith(prefix)}
        means.append((predict_map(mapping,model['palette'])-model['xc'])/model['xs'])
    return np.stack(means)


def infer_forward(model,x):
    p,evidence=posterior((x-model['xc'])/model['xs'],forward_means(model),model['cov'])
    return p,evidence,decisions(p,model['palette'])
