"""Additional frozen Fourier/classical source-only selectors for V7 transfer."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from cc_fourier_ridge import predict_score
from cc_v3_experiment import DATA_HASHES
from cc_v3_ffcc import decode, featurize
from cc_v7_risk import fit_heads, read_in_role_order
from cc_v7_verify import verify_manifest
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

from luma_skin_vision.cc.core import EXPERT_NAMES, angular, reproduction
from luma_skin_vision.cc.v2 import risk_features_invariant
from luma_skin_vision.cc.v2_experiment import split_indices
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT=Path(__file__).resolve().parents[1]
FILTER=ROOT/"experiments/runs/fourier_ridge_v1/sigma2.0_ridge0.001_gray_world.npz"


@torch.no_grad()
def fourier_predictions(images,weight):
    arrays={k:[] for k in ("pred","context","cheap","valid","raw_risk")}
    for start in range(0,len(images),16):
        x=torch.from_numpy(images[start:start+16]).double()
        normalized=x/x.amax((1,2,3),keepdim=True).clamp_min(1e-12)
        features=featurize(normalized)
        score=predict_score(features["hist"].numpy(),weight)
        pmf=torch.from_numpy(score*200).flatten(1).softmax(-1).reshape(-1,64,64)
        decoded=decode(pmf,average_rgb=features["average_rgb"],unwrap_mode="gray_world")
        context=F.adaptive_avg_pool2d(pmf[:,None],(8,8)).flatten(1)*64
        standard=risk_features_invariant(x,decoded["pred"],context)
        output={"pred":decoded["pred"],"context":context,"cheap":standard["cheap"],"valid":standard["valid"]&decoded["mean_valid"],"raw_risk":1-decoded["confidence"]}
        for key,value in output.items():
            arrays[key].append(value.numpy())
    return {k:np.concatenate(v) for k,v in arrays.items()}


def classical_disagreement(pred,experts):
    return np.mean([angular(pred,experts[:,i]) for i in range(4)],axis=0)


def calibrated_raw(scores,errors):
    if not np.isfinite(scores).all() or np.any(scores<0):
        raise ValueError("Raw confidence must be finite and nonnegative")
    scale=max(1e-8,float(errors.mean()/max(1e-8,float(scores.mean()))))
    return {"scale":scale,"cal_scores":(scores*scale).tolist(),"cal_errors":errors.tolist(),"calibration":"Source CAL positive scalar only; preserves ranking, no target calibration"}


def run(out):
    out=Path(out).resolve()
    if out.exists():
        raise ValueError("Immutable source controls already exist")
    verify_manifest(FILTER.parent,"manifest.json")
    data=ROOT/"data/processed/cc128"
    if {k:sha256(data/k) for k in DATA_HASHES}!=DATA_HASHES:
        raise ValueError("Source cache changed")
    rows=json.loads((data/"cube_manifest.json").read_text())
    roles=split_indices(rows,"official")
    selected=np.r_[roles["risk"],roles["cal"]]
    if (len(roles["risk"]),len(roles["cal"]))!=(259,268):
        raise ValueError("Wrong source fitting roles")
    source_rows=[rows[i] for i in selected]
    images=read_in_role_order(data/"cube.npz","images",selected).astype(np.float32)
    gt=read_in_role_order(data/"cube.npz","gt",selected).astype(np.float64)
    experts=read_in_role_order(data/"cube.npz","experts",selected).astype(np.float64)
    with np.load(FILTER) as saved:
        weight=saved["weight"]
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    write_json(out/"config.json",{"scope":"Source RISK/CAL only; no target pixels/GT","source_indices":selected.tolist(),"source_ids":[r["id"] for r in source_rows],"data_sha256":DATA_HASHES,"fourier_weight":FILTER.relative_to(ROOT).as_posix(),"fourier_weight_sha256":sha256(FILTER),"source_sha256":{p:sha256(ROOT/p) for p in ("scripts/cc_v7_external_controls.py","scripts/cc_v7_risk.py","scripts/cc_v2_select.py","scripts/cc_v3_ffcc.py","scripts/cc_fourier_ridge.py")},"protocol_sha256":sha256(ROOT/"docs/research/cc_v7_external_methods_protocol.md")})
    with threadpool_limits(limits=2):
        values=fourier_predictions(images,weight)
        target=out/"fourier_ridge_combined"
        target.mkdir()
        heads=fit_heads(values,gt,source_rows,len(roles["risk"]),target,sha256(FILTER))
        np.savez_compressed(target/"risk_cal_predictions.npz",**values,gt=gt,ids=np.array([r["id"] for r in source_rows]))
        n=len(roles["risk"])
        calibration={"fourier_ridge_raw":calibrated_raw(values["raw_risk"][n:],reproduction(values["pred"][n:],gt[n:]))}
        for index,name in enumerate(EXPERT_NAMES):
            pred=experts[n:,index]
            scores=classical_disagreement(pred,experts[n:])
            calibration[name]=calibrated_raw(scores,reproduction(pred,gt[n:]))
        write_json(out/"raw_calibration.json",{"cal_ids":[r["id"] for r in source_rows[n:]],"methods":calibration})
    write_json(out/"manifest.json",{"sha256":{p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}})
    print(json.dumps({"source_rows":len(selected),"fourier_heads":heads,"raw_calibrated_methods":list(calibration)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True)
    run(parser.parse_args().out)
