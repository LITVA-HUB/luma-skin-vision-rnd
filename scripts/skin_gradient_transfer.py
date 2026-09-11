"""TRAIN-only group-gradient falsifier, not a new accuracy experiment."""
from contextlib import contextmanager
import numpy as np
import torch

SHUFFLES=(51871,51872,51873)
STEPS=(1e-4,1e-3)


def partitions(person):
    _,group=np.unique(person,return_inverse=True)
    return [('person',group)]+[(f'shuffle{s}',np.random.default_rng(s).permutation(group)) for s in SHUFFLES]


def group_means(values,group):
    return np.stack([values[group==j].mean(0) for j in range(int(group.max())+1)])


def gradient_statistics(gram,weights):
    norm=np.sqrt(np.maximum(np.diag(gram),0));denom=norm[:,None]*norm[None,:]
    cosine=np.divide(gram,denom,out=np.zeros_like(gram),where=denom>0)
    off=~np.eye(len(gram),dtype=bool)
    return {'mean_cosine':float(cosine[off].mean()),'negative_pair_fraction':float((gram[off]<0).mean()),
            'mean_group_norm':float(norm.mean()),'pooled_norm':float(np.sqrt(max(0.,weights@gram@weights))),
            'weighted_group_squared_norm':float(weights@np.diag(gram))}


@contextmanager
def transient_step(model,gradient,length):
    saved={k:v.detach().clone() for k,v in model.state_dict().items()}
    norm=gradient.norm()
    if not torch.isfinite(norm) or norm<=0:raise ValueError('Nonfinite or zero direction')
    try:
        start=0
        with torch.no_grad():
            for p in model.parameters():
                end=start+p.numel();p.add_(gradient[start:end].reshape_as(p),alpha=-length/float(norm));start=end
        if start!=gradient.numel():raise ValueError('Direction size mismatch')
        yield
    finally:model.load_state_dict(saved)


def component_gradients(model,x,y,mode,indices,batch_size=32):
    params=tuple(model.parameters());n=len(indices)
    gs=[torch.zeros(sum(p.numel() for p in params),dtype=x.dtype,device=x.device) for _ in range(2)]
    for batch in np.array_split(indices,np.arange(batch_size,n,batch_size)):
        ix=torch.as_tensor(batch,device=x.device);pred,logits,_=model(x[ix],None)
        loss=[(pred-y[ix]).square().mean()*len(ix)/n,
              torch.nn.functional.cross_entropy(logits,mode[ix])*len(ix)/n]
        for j,l in enumerate(loss):
            parts=torch.autograd.grad(l,params,retain_graph=j==0,allow_unused=True)
            gs[j]+=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(params,parts)])
    if not all(torch.isfinite(g).all() for g in gs):raise ValueError('Nonfinite gradient')
    return gs
