"""Convex Fourier correlation-filter spike; not an author FFCC reproduction."""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def design(hist):
    h = np.fft.fft2(np.asarray(hist,dtype=np.float64)).transpose(0,2,3,1)
    return np.concatenate([h,np.ones((*h.shape[:-1],1),dtype=np.complex128)],axis=-1)


def fit_filter(hist, target, ridge, bias=True):
    if ridge <= 0:
        raise ValueError("Positive ridge required")
    a = design(hist)
    if not bias:
        a[...,2] = 0
    y = np.fft.fft2(np.asarray(target,dtype=np.float64))
    gram = np.einsum("nhwc,nhwd->hwcd",a.conj(),a)/len(a)
    rhs = np.einsum("nhwc,nhw->hwc",a.conj(),y)/len(a)
    return np.linalg.solve(gram+ridge*np.eye(3),rhs[...,None])[...,0]


def predict_score(hist,weight):
    spectrum = np.einsum("nhwc,hwc->nhw",design(hist),weight)
    score = np.fft.ifft2(spectrum)
    if abs(score.imag).max() > 1e-8:
        raise ValueError("Non-real filter reconstruction")
    return score.real


def run(out):
    import torch
    from cc_v2_statistics import read_npz_rows
    from cc_v3_experiment import DATA_HASHES, source_rows
    from cc_v3_ffcc import decode, featurize, target_distribution
    from cc_v4_experiment import risk_summary
    from threadpoolctl import threadpool_limits

    from luma_skin_vision.cc.core import reproduction, summarize
    from luma_skin_vision.data import sha256
    from luma_skin_vision.experiment import write_json

    root = Path(__file__).resolve().parents[1]
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Existing immutable spike output")
    data = root/"data/processed/cc128"
    if {k:sha256(data/k) for k in DATA_HASHES} != DATA_HASHES:
        raise ValueError("Changed frozen source data")
    rows = json.loads((data/"cube_manifest.json").read_text())
    selected,train,val = source_rows(rows)
    assert (len(train),len(val)) == (1126,119)
    out.mkdir(parents=True)
    bindings = ["scripts/cc_fourier_ridge_v2.py","scripts/cc_v3_ffcc.py","docs/research/fourier_ridge_v2_protocol.md"]
    write_json(out/"config.json",{"data_sha256":DATA_HASHES,"source_sha256":{p:sha256(root/p) for p in bindings},"train_ids":[rows[selected[i]]["id"] for i in train],"validation_ids":[rows[selected[i]]["id"] for i in val],"protocol":"18 adaptive development candidates, bias ablation; prior16 preserved; no author reproduction"})
    torch.set_num_threads(2)
    images = read_npz_rows(data/"cube.npz","images",selected,2234).astype(np.float32)
    gt = read_npz_rows(data/"cube.npz","gt",selected,2234).astype(np.float64)
    hist,average = [],[]
    started = time.perf_counter()
    for start in range(0,len(images),16):
        x = torch.from_numpy(images[start:start+16]).double()
        x = x/x.amax((1,2,3),keepdim=True).clamp_min(1e-12)
        f = featurize(x)
        hist.append(f["hist"].numpy())
        average.append(f["average_rgb"].numpy())
    hist,average = np.concatenate(hist),np.concatenate(average)
    prep_seconds = time.perf_counter()-started
    target = target_distribution(torch.from_numpy(gt[train])).numpy()
    distance = np.minimum(np.arange(64),64-np.arange(64))
    squared = distance[:,None]**2+distance[None,:]**2
    results=[]
    with threadpool_limits(limits=2):
        for sigma,bias in [(s,b) for s in (2.,4.,8.) for b in (True,False)]:
            kernel = np.exp(-squared/(2*sigma*sigma))
            kernel /= kernel.sum()
            smoothed = np.fft.ifft2(np.fft.fft2(target)*np.fft.fft2(kernel)).real
            for ridge in (.00001,.0001,.001):
                started = time.perf_counter()
                weight = fit_filter(hist[train],smoothed,ridge,bias)
                score = predict_score(hist[val],weight)
                pmf = torch.from_numpy(score*200).flatten(1).softmax(-1).reshape(-1,64,64)
                elapsed = time.perf_counter()-started
                for mode in ("gray_world",):
                    name = f"sigma{sigma}_ridge{ridge}_{mode}_bias{bias}"
                    output = decode(pmf,average_rgb=torch.from_numpy(average[val]),unwrap_mode=mode)
                    pred = output["pred"].numpy()
                    valid = output["mean_valid"].numpy()
                    risk = 1-output["confidence"].numpy()
                    error = reproduction(pred,gt[val])
                    metrics = {"id":name,"sigma":sigma,"ridge":ridge,"unwrap":mode,"reproduction":summarize(error),"raw_risk":risk_summary(error,risk),"valid":int(valid.sum()),"n":len(val),"bias":bias,"parameters_real":12288 if bias else 8192,"fit_and_validation_seconds":elapsed}
                    np.savez_compressed(out/(name+".npz"),weight=weight,pred=pred,risk=risk,valid=valid,gt=gt[val],ids=np.array([rows[selected[i]]["id"] for i in val]))
                    results.append(metrics)
                    print(json.dumps({"id":name,"mean":metrics["reproduction"]["mean"],"valid":metrics["valid"]}),flush=True)
    winner = min(results,key=lambda r:r["reproduction"]["mean"])
    write_json(out/"results.json",{"status":"MEASURED EXPLORATORY SOURCE VALIDATION ONLY","histogram_preparation_seconds":prep_seconds,"winner_by_validation_mean":winner["id"],"candidates":results})
    write_json(out/"manifest.json",{"sha256":{p.name:sha256(p) for p in out.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True)
    run(parser.parse_args().out)
