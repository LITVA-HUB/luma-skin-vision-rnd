"""Matched V7 error heads fitted only to held-out source RISK/CAL predictions."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from cc_v2_select import estimator, raw_predict
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES
from cc_v7_verify import verify_manifest
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from luma_skin_vision.cc.core import reproduction, selective_curve
from luma_skin_vision.cc.v2 import CompactResidualCC, risk_features_invariant
from luma_skin_vision.cc.v2_experiment import split_indices
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT=Path(__file__).resolve().parents[1]


def read_in_role_order(path,key,selected):
    ordered=np.sort(selected)
    values=read_npz_rows(path,key,ordered,2234)
    return values[np.searchsorted(ordered,selected)]


@torch.no_grad()
def predict(model,x,batch=32):
    arrays={k:[] for k in ("pred","context","cheap","valid")}
    for start in range(0,len(x),batch):
        xb=x[start:start+batch]
        pred,context=model(xb)
        output={"pred":pred,**risk_features_invariant(xb,pred,context)}
        for key in arrays:
            arrays[key].append(output[key].numpy())
    return {k:np.concatenate(v) for k,v in arrays.items()}


def fit_heads(values,gt,rows,n,out,checkpoint_sha):
    if not values["valid"].all():
        raise ValueError("Invalid RISK/CAL input requires an explicit protocol revision")
    if [r["subset"] for r in rows]!=["risk"]*n+["cal"]*(len(rows)-n):
        raise ValueError("Risk fitting received a wrong role or row ordering")
    fit_groups={r["group"] for r in rows[:n]}
    if fit_groups & {r["group"] for r in rows[n:]}:
        raise ValueError("Risk/calibration group overlap")
    errors=reproduction(values["pred"].astype(np.float64),gt)
    groups=np.array([r["group"] for r in rows[:n]])
    ids=[r["id"] for r in rows[:n]]
    folds=list(GroupKFold(n_splits=5).split(np.arange(n),groups=groups))
    report={"scope":"Only held-out source RISK and disjoint CAL; no test/phone labels or teacher features","selection":"Five standard candidate heads per block; minimum pooled 5-group OOF risk80 then AURC","calibration":"Positive scalar mean-error/mean-prediction on separate CAL; empirical, not a shift guarantee","checkpoint_sha256":checkpoint_sha,"fit_ids":ids,"cal_ids":[r["id"] for r in rows[n:]],"folds":[{"train":tr.tolist(),"validation":va.tolist()} for tr,va in folds],"heads":{}}
    for block in ("context","cheap","combined"):
        feature=np.concatenate([values["context"],values["cheap"]],axis=-1) if block=="combined" else values[block]
        feature=feature.astype(np.float64)
        candidates=[]
        for name in ("ridge1","ridge10","ridge100","hgb3","hgb7"):
            oof=np.zeros(n)
            for train,val in folds:
                model=estimator(name)
                model.fit(feature[train],np.log1p(errors[train]))
                oof[val]=raw_predict(model,feature[val])
            curve=selective_curve(errors[:n],oof,ids)
            candidates.append({"name":name,"risk80":curve["fixed"]["80"]["mean"],"aurc":curve["aurc"],"oof_scores":oof.tolist()})
        selected=min(candidates,key=lambda c:(c["risk80"],c["aurc"]))
        model=estimator(selected["name"])
        model.fit(feature[:n],np.log1p(errors[:n]))
        raw_cal=raw_predict(model,feature[n:])
        scale=max(1e-8,float(errors[n:].mean()/raw_cal.mean()))
        joblib.dump({"model":model,"scale":scale,"block":block},out/(block+".joblib"))
        report["heads"][block]={"selected":selected["name"],"features":feature.shape[1],"candidates":candidates,"scale":scale,"cal_scores":(raw_cal*scale).tolist(),"cal_errors":errors[n:].tolist(),"artifact_sha256":sha256(out/(block+".joblib"))}
    write_json(out/"selection.json",report)
    return {k:v["selected"] for k,v in report["heads"].items()}


def run(folder,out):
    folder,out=Path(folder).resolve(),Path(out).resolve()
    if out.exists():
        raise ValueError("Immutable risk output exists")
    verify_manifest(folder,"artifact_manifest.json")
    config=json.loads((folder/"config.json").read_text())
    data=ROOT/"data/processed/cc128"
    if {k:sha256(data/k) for k in DATA_HASHES}!=DATA_HASHES:
        raise ValueError("Source digest changed")
    rows=json.loads((data/"cube_manifest.json").read_text())
    roles=split_indices(rows,"official")
    if (len(roles["risk"]),len(roles["cal"]))!=(259,268):
        raise ValueError("Source risk roles changed")
    if config["train_ids"]!=[rows[i]["id"] for i in roles["train"]]:
        raise ValueError("Wrong estimator training population")
    selected=np.r_[roles["risk"],roles["cal"]]
    source_rows=[rows[i] for i in selected]
    torch.set_num_threads(2)
    x=torch.from_numpy(read_in_role_order(data/"cube.npz","images",selected).astype(np.float32))
    gt=read_in_role_order(data/"cube.npz","gt",selected).astype(np.float64)
    out.mkdir(parents=True)
    started=time.perf_counter()
    numerical=[ROOT/"scripts"/p for p in ("cc_v7_risk.py","cc_v2_select.py","cc_v2_statistics.py")]
    numerical += [ROOT/"src/luma_skin_vision/cc"/p for p in ("core.py","v2.py","v2_experiment.py")]
    write_json(out/"config.json",{"training_run":str(folder),"training_config_sha256":sha256(folder/"config.json"),"data_sha256":DATA_HASHES,"source_indices":selected.tolist(),"source_ids":[r["id"] for r in source_rows],"primary_estimator_run":config["primary"],"numerical_sha256":{p.relative_to(ROOT).as_posix():sha256(p) for p in numerical},"primary_risk_block":"combined; context and cheap are ablations, not test-selected alternatives"})
    results={}
    with threadpool_limits(limits=2):
        for arm in config["arms"]:
            target=out/arm
            target.mkdir()
            checkpoint=folder/arm/"best.pt"
            state=torch.load(checkpoint,map_location="cpu",weights_only=True)
            model=CompactResidualCC("direct","large").eval()
            model.load_state_dict({k:v for k,v in state.items() if not k.startswith("teacher_projection.")},strict=True)
            values=predict(model,x)
            np.savez_compressed(target/"risk_cal_predictions.npz",**values,gt=gt,ids=np.array([r["id"] for r in source_rows]))
            results[arm]=fit_heads(values,gt,source_rows,len(roles["risk"]),target,sha256(checkpoint))
            print(json.dumps({"arm":arm,"selected_heads":results[arm],"seconds":round(time.perf_counter()-started,1)}),flush=True)
    write_json(out/"result.json",{"status":"COMPLETE SOURCE-ONLY RISK FITTING","arms":results,"seconds":time.perf_counter()-started})
    write_json(out/"manifest.json",{"sha256":{p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    run(args.run,args.out)
