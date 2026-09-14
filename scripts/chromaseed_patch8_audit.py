"""Independent P8 lineage, local reductions, selection and actual consumers."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
from chromaseed_affine_audit import summaries
from chromaseed_gate_stability_audit import transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_audit import SETTINGS
from chromaseed_kernel_audit import js,nz
from chromaseed_local_denoise_audit import close,full_metrics,verify_normalizers
from chromaseed_long_training_run import ND,NP
from chromaseed_neural_prefix_audit import inspect_export
from chromaseed_patch8_numpy import Predictor
from chromaseed_patch8_run import OUT,ROOT,RUN,bank_path,check_map,load_data
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from chromaseed_refine_train import setup
from skin_local_search_train import folds_for,roles,sha,write_json

SEEDS=(17,29,43)
TIMES=(0,512,2048,8192,32768)
RATES=(.0001,.0003,.001)
ARMS=("stats","patch8")
SLOTS=[dict(seed=s,arm=a,lr=r) for s in SEEDS for a in ARMS for r in RATES]
EXTRA={"u","e","g","t_mean","t_std"}


def direct(model,x,tokens):
    xx=((np.asarray(x,np.float32)-model["x_mean"])/model["x_std"]).astype(float)
    hidden=np.sum(xx[:,:,None]*model["w0"].astype(float)[None],axis=1)+model["b0"]
    if "u" in model:
        tn=((np.asarray(tokens,np.float32)-model["t_mean"])/model["t_std"]).astype(float)
        local=np.stack([np.maximum(np.sum(tn*model["u"][:,i].astype(float),axis=2)+model["e"][i],0) for i in range(8)],axis=2)
        mean=local.mean(1)
        spread=np.sqrt(np.mean((local-mean[:,None,:])**2,axis=1)+1e-6)
        pool=np.concatenate((mean,spread,np.max(local,axis=1)),axis=1)
        hidden=hidden+np.sum(pool[:,:,None]*model["g"].astype(float)[None],axis=1)
    normalized=np.sum(np.maximum(hidden,0)[:,:,None]*model["v0"].astype(float)[None],axis=1)+model["c0"]
    return normalized*model["y_std"]+model["y_mean"]


def transform_local(tokens,dose,anchor):
    v=np.asarray(tokens,np.float32).astype(float)
    for col in range(18):
        if col<12:
            v[:,:,col]=v[:,:,col]+dose*(anchor[col%3]-v[:,:,col])
        else:
            v[:,:,col]=v[:,:,col]+dose*(-v[:,:,col])
    return v.astype(np.float32)


def gpu_parity(models,x,tokens,expected):
    first=models[0];patch=models[3]
    xx=(np.asarray(x,np.float32)-first["x_mean"])/first["x_std"]
    tn=(np.asarray(tokens,np.float32)-patch["t_mean"])/patch["t_std"]
    with torch.no_grad():
        xt=torch.tensor(np.repeat(xx[None],18,axis=0),device="cuda")
        tt=torch.tensor(np.repeat(tn.reshape(-1,18)[None],18,axis=0),device="cuda")
        def stack(key,shape):
            return torch.tensor(np.stack([m.get(key,np.zeros(shape,np.float32)) for m in models]),device="cuda")
        local=torch.relu(torch.bmm(tt,stack("u",(18,8)))+stack("e",(8,))[:,None,:]).reshape(18,len(x),64,8)
        mean=local.mean(2)
        pool=torch.cat((mean,((local-mean[:,:,None,:]).square().mean(2)+1e-6).sqrt(),local.amax(2)),dim=-1)
        hidden=torch.relu((torch.bmm(xt,stack("w0",(36,16)))+stack("b0",(16,))[:,None,:])+torch.bmm(pool,stack("g",(24,16))))
        actual=(torch.bmm(hidden,stack("v0",(16,3)))+stack("c0",(3,))[:,None,:]).cpu().numpy()
    actual=actual*first["y_std"]+first["y_mean"]
    maximum=float(np.max(abs(actual-expected)))
    assert maximum<=.002,maximum
    return maximum


def exact(a,b):
    assert set(a)==set(b)
    for k in a:
        np.testing.assert_array_equal(a[k],b[k],err_msg=k)


def inspect_bank(data,role,fold,source,artifacts):
    mask,held=roles(data["patient"],data["device"])[role]
    ix=np.flatnonzero(mask)
    query=np.flatnonzero(held)
    if fold is not None:
        assignment=folds_for(data["patient"][ix],data["device"][ix])
        query,ix=ix[assignment==fold],ix[assignment!=fold]
    path=bank_path(role,fold)
    r=js(path/"receipt.json")
    assert r["source_lock_sha256"]==source and r["selection_sha256"]==(None if fold is not None else sha(RUN/"selections.json"))
    assert r["slots"]==SLOTS and r["steps"]==r["schedule_horizon"]==32768 and r["checkpoints"]==list(TIMES)
    assert r["trajectory_count"]==18 and r["device"]=="cuda" and r["engine"]=="cuda_graph"
    check_map(r["warm_parent_sha256"])
    for n,digest in r["files"].items():
        assert sha(path/n)==digest
        artifacts[(path/n).relative_to(ROOT).as_posix()]=digest
    artifacts[(path/"receipt.json").relative_to(ROOT).as_posix()]=sha(path/"receipt.json")
    rows=nz(path/"rows.npz")
    np.testing.assert_array_equal(rows["fit_rows"],ix)
    np.testing.assert_array_equal(rows["query_rows"],query if fold is not None else np.array([],np.int64))
    assert not set(data["patient"][ix]) & set(data["patient"][query])
    token=data["tokens"][ix]
    assert hashlib.sha256(token.tobytes()).hexdigest()==r["fit_tokens_sha256"]
    flat=token.astype(float).reshape(-1,18)
    mean,std=flat.mean(0).astype(np.float32),np.maximum(flat.std(0),1e-6).astype(np.float32)
    np.testing.assert_array_equal(np.array(r["token_mean"],np.float32),mean)
    np.testing.assert_array_equal(np.array(r["token_std"],np.float32),std)
    p=balanced(data["patient"][ix],data["site"][ix]);p/=p.sum()
    sampled=np.stack([np.random.default_rng(s+9001).choice(len(ix),(32768,64),p=p) for s in SEEDS])
    assert hashlib.sha256(sampled.tobytes()).hexdigest()==r["sampling_sha256"]
    np.testing.assert_array_equal(np.array(r["final_learning_rates"],np.float32),np.array([s["lr"] for s in SLOTS],np.float32)*np.float32(.1))
    prior=js(NP/"selections.json")["roles"][role]["blind4"]
    j=prior["policies"]["quality"]["prefix"]
    warm=nz(path/"warm_models.npz")
    baseline=nz(path/"models_0.npz")
    for si,seed in enumerate(SEEDS):
        w=model_from(warm,str(si))
        verify_normalizers(w,data["color"][ix],data["target"][ix])
        if fold is None:
            exact(w,nz(NP/"selected"/role/f"blind4_j{j}_s{seed}.npz"))
        else:
            oldpath=ND/"inner"/role/"blind4"/f"fold{fold}"
            oldrows=nz(oldpath/"rows.npz")
            np.testing.assert_array_equal(oldrows["fit_rows"],ix)
            np.testing.assert_array_equal(oldrows["query_rows"],query)
            raw=model_from(nz(oldpath/f"models_{prior['step']}.npz"),str(2*si+prior["lr_index"]))
            inspect_export(raw,w,j)
        for offset in range(6):
            m=model_from(baseline,str(si*6+offset))
            exact({k:v for k,v in m.items() if k not in EXTRA},w)
            if offset>=3:
                expected=np.random.default_rng(seed+770003).uniform(-1/np.sqrt(18),1/np.sqrt(18),(18,8)).astype(np.float32)
                np.testing.assert_array_equal(m["u"],expected)
                assert np.count_nonzero(m["e"])==np.count_nonzero(m["g"])==0
                np.testing.assert_array_equal(m["t_mean"],mean);np.testing.assert_array_equal(m["t_std"],std)
    return ix,query,warm
