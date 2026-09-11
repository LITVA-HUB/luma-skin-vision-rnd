"""Native-color inference, interpretable risk features and immutable bindings."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from skin_mskcc_data import ROOT,MANIFEST,sha
from skin_mskcc_vote import PatchVotes

PROTOCOL=ROOT/'docs/research/skin_mskcc_selective_protocol_v1.md'
OUT=ROOT/'docs/benchmarks/skin_mskcc_selective_v1'
RUN=ROOT/'experiments/runs/skin_mskcc_selective_v1'
SEEDS=[17,29,43]


def patient_folds(patients):
    unique=sorted(set(patients),key=lambda p:hashlib.sha256(('LumaMSKCCRISKv1|'+str(p)).encode()).hexdigest())
    if len(unique)!=24:raise ValueError('Expected24TRAIN people')
    return {p:i//4 for i,p in enumerate(unique)}


def density(fit,query):
    scale=StandardScaler().fit(fit)
    return NearestNeighbors(n_neighbors=5).fit(scale.transform(fit)).kneighbors(scale.transform(query))[0].mean(1)


def plain_predict(path,data):
    state=torch.load(path,weights_only=True,map_location='cpu')
    model=PatchVotes(state['target_std'],0).cuda().eval();model.load_state_dict(state['state'])
    mean,std=state['target_mean'].cuda(),state['target_std'].cuda();preds=[];witnesses=[]
    with torch.no_grad():
        for offset in range(0,len(data['tokens']),64):
            x=torch.from_numpy(data['tokens'][offset:offset+64]).float().cuda()
            h=model.local(x);context=model.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
            v=model.votes(torch.cat([h,context[:,None].expand(-1,64,-1)],-1))
            w=v[...,3].softmax(1);estimate=(v[...,:3]*w[...,None]).sum(1)*std+mean
            physical=v[...,:3]*std+mean
            residual=torch.linalg.vector_norm(physical-estimate[:,None],dim=-1)
            order=x[...,3:6].mean(-1).argsort(1)
            sorted_votes=physical.gather(1,order[...,None].expand(-1,-1,3))
            features=torch.stack([(residual.square()*w).sum(1).sqrt(),torch.quantile(residual,.9,dim=1),residual.amax(1),
                -(w*w.clamp_min(1e-12).log()).sum(1)/np.log(64),w.amax(1),
                torch.linalg.vector_norm(physical.mean(1)-estimate,dim=1),
                torch.linalg.vector_norm(sorted_votes[:,32:].mean(1)-sorted_votes[:,:32].mean(1),dim=1),
                1/(64*w.square().sum(1))],1)
            preds.append(estimate.cpu().numpy());witnesses.append(features.cpu().numpy())
    return np.concatenate(preds),np.concatenate(witnesses)


def designs(data,predictions,witnesses,dist):
    p=np.asarray(predictions,dtype=np.float64);w=np.asarray(witnesses,dtype=np.float64)
    result={}
    for i,name in enumerate(['single17','single29','single43','ensemble']):
        if i<3:pred=p[i];wit=w[i];dis=np.zeros(len(pred))
        else:
            pred=p.mean(0);wit=w.mean(0);dis=np.sqrt(np.mean(np.sum((p-pred)**2,axis=2),axis=0))
        base=np.column_stack([data['color'],pred,dist])
        proposed=np.column_stack([base,wit,dis])
        assert base.shape[1]==40 and proposed.shape[1]==49
        result[name]={'prediction':pred,'C_plus':base,'Proposed':proposed}
    return result


def deployment_designs(data,train):
    pred=[];wit=[]
    for seed in SEEDS:
        p,w=plain_predict(ROOT/f'experiments/runs/skin_mskcc_pixels_v1/votes_mean_seed{seed}/best.pt',data)
        pred.append(p);wit.append(w)
    return designs(data,pred,wit,density(train['color'],data['color']))


def binding(paths):
    return {str(Path(p).resolve().relative_to(ROOT)):sha(p) for p in sorted(set(map(str,paths)))}


def verify_lock(path,expected_sha,stage):
    path=Path(path)
    if sha(path)!=expected_sha:raise ValueError('Lock hash differs')
    record=json.loads(path.read_bytes())
    if record['stage']!=stage:raise ValueError('Wrong lock stage')
    for name,digest in record['bindings'].items():
        target=(ROOT/name).resolve()
        if not target.is_relative_to(ROOT):raise ValueError('Binding outside repository')
        if sha(target)!=digest:raise ValueError('Bound artifact changed: '+name)
    return record
