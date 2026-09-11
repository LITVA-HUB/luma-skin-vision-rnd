"""Predict and score the frozen29-method real-camera benchmark; no fitting."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from cc_v2_select import features, raw_predict, selective_result
from cc_v2_statistics import predict_arrays
from cc_v7_external_controls import classical_disagreement, fourier_predictions
from cc_v7_external_data import CACHE
from cc_v7_external_lock import BENCH, LOCK, ROOT, checked_lock, load
from threadpoolctl import threadpool_limits

from luma_skin_vision.cc.v2 import CompactResidualCC, input_validity, risk_features_invariant
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json


def check_cache(digest):
    checked_lock(digest)
    prepared=load(CACHE/"preparation.json")
    if prepared["method_lock_sha256"]!=digest or prepared["status"]!="COMPLETE LOCKED EVALUATION CACHES":
        raise ValueError("Evaluation preparation/lock mismatch")
    for name,expected in prepared["sha256"].items():
        if sha256(CACHE/name)!=expected:
            raise ValueError("Evaluation cache changed")
    return prepared


@torch.no_grad()
def method_prediction(method,images,experts,raw_calibration):
    kind=method["kind"]
    if kind=="classical":
        pred=experts[:,method["expert_index"]].astype(np.float64)
        score=classical_disagreement(pred,experts)*raw_calibration[method["id"]]["scale"]
        values={"pred":pred,"valid":np.ones(len(images),dtype=bool)}
    elif kind=="statistics":
        values=predict_arrays(joblib.load(ROOT/method["weight"]),images)
    elif kind=="fourier":
        with np.load(ROOT/method["weight"]) as saved:
            weight=saved["weight"]
        values=fourier_predictions(images,weight)
        values["cheap_features"]=values.pop("cheap")
        if method["selector"]=="raw":
            score=values["raw_risk"]*raw_calibration[method["id"]]["scale"]
    else:
        model=CompactResidualCC(method["mode"],method["backbone"]).eval()
        state=torch.load(ROOT/method["weight"],map_location="cpu",weights_only=True)
        state=state["state"] if kind=="v2" else {k:v for k,v in state.items() if not k.startswith("teacher_projection.")}
        model.load_state_dict(state,strict=True)
        values={key:[] for key in ("pred","context","cheap_features","valid")}
        for start in range(0,len(images),16):
            x=torch.from_numpy(images[start:start+16])
            pred,context=model(x)
            feature=risk_features_invariant(x,pred,context)
            for key,value in (("pred",pred),("context",context),("cheap_features",feature["cheap"]),("valid",feature["valid"])):
                values[key].append(value.numpy())
        values={k:np.concatenate(v) for k,v in values.items()}
    if "head" in method:
        payload=joblib.load(ROOT/method["head"])
        score=raw_predict(payload["model"],features(values,"combined"))*payload["scale"]
    valid=values["valid"].astype(bool)&input_validity(torch.from_numpy(images)).numpy()
    pred=values["pred"].astype(np.float64)
    if pred.shape!=(len(images),3) or score.shape!=(len(images),):
        raise ValueError("Unaligned method outputs")
    if not np.isfinite(pred).all() or np.any(pred<=0) or not np.isfinite(score).all():
        raise ValueError("Nonfinite/nonpositive method output")
    pred=np.where(valid[:,None],pred,np.ones_like(pred)/np.sqrt(3))
    return {"pred":pred,"scores":score,"valid":valid}


def predict(digest):
    lock=checked_lock(digest)
    check_cache(digest)
    out=BENCH/"predictions"
    if out.exists():
        raise ValueError("Immutable target predictions exist")
    out.mkdir(parents=True)
    raw=load(ROOT/lock["raw_calibration"])["methods"]
    torch.set_num_threads(2)
    outputs={}
    with threadpool_limits(limits=2):
        for dataset in ("external","source_test"):
            rows=load(CACHE/(dataset+"_manifest.json"))
            with np.load(CACHE/(dataset+".npz"),allow_pickle=False) as values:
                # Ground-truth arrays are deliberately not deserialized here.
                images=values["images"].astype(np.float32)
                experts=values["experts"].astype(np.float64)
            for method in lock["methods"]:
                prediction=method_prediction(method,images,experts,raw)
                name=dataset+"__"+method["id"]+".npz"
                np.savez_compressed(out/name,**prediction,ids=np.array([r["id"] for r in rows]))
                outputs[name]=sha256(out/name)
                print(json.dumps({"dataset":dataset,"method":method["id"],"rows":len(rows),"unsupported":int((~prediction["valid"]).sum())}),flush=True)
    write_json(out/"manifest.json",{"method_lock_sha256":sha256(LOCK),"preparation_sha256":sha256(CACHE/"preparation.json"),"script_sha256":sha256(Path(__file__)),"outputs_sha256":outputs,"scope":"Images and cached classical estimates only; no label fitting or evaluation in prediction"})


def populations(rows,dataset):
    all_indices=np.arange(len(rows))
    if dataset=="source_test":
        return [("known_camera_official_test",all_indices)]
    primary=np.array([r["primary"] for r in rows],dtype=bool)
    result=[("unseen_primary",all_indices[primary]),("unseen_sensitivity",all_indices)]
    for camera in sorted({r["camera"] for r in rows}):
        match=np.array([r["camera"]==camera for r in rows])
        result += [("unseen_primary/"+camera,all_indices[primary&match]),("unseen_sensitivity/"+camera,all_indices[match])]
    return result


def evaluate(digest):
    lock=checked_lock(digest)
    check_cache(digest)
    manifest=load(BENCH/"predictions/manifest.json")
    if manifest["method_lock_sha256"]!=digest or manifest["preparation_sha256"]!=sha256(CACHE/"preparation.json"):
        raise ValueError("Prediction bindings differ")
    out=BENCH/"evaluation"
    if out.exists():
        raise ValueError("Immutable evaluation exists")
    out.mkdir(parents=True)
    raw=load(ROOT/lock["raw_calibration"])["methods"]
    records=[]
    for dataset in ("external","source_test"):
        rows=load(CACHE/(dataset+"_manifest.json"))
        with np.load(CACHE/(dataset+".npz")) as values:
            gt=values["gt"].astype(np.float64)
        np.savez_compressed(out/(dataset+"_references.npz"),gt=gt,ids=np.array([r["id"] for r in rows]))
        for method in lock["methods"]:
            name=dataset+"__"+method["id"]+".npz"
            path=BENCH/"predictions"/name
            if sha256(path)!=manifest["outputs_sha256"][name]:
                raise ValueError("Prediction changed")
            with np.load(path) as stored:
                if stored["ids"].tolist()!=[r["id"] for r in rows]:
                    raise ValueError("Reference/prediction ID mismatch")
                pred,scores,valid=(stored[k].copy() for k in ("pred","scores","valid"))
            if "selection" in method:
                cal=np.array(load(ROOT/method["selection"])["heads"]["combined"]["cal_scores"])
            else:
                cal=np.array(raw[method["id"]]["cal_scores"])
            for population,indices in populations(rows,dataset):
                selected_ids=[rows[i]["id"] for i in indices]
                result=selective_result(pred[indices],gt[indices],scores[indices],selected_ids,cal,valid[indices])
                for key in ("pred","gt","scores","errors","ids"):
                    result.pop(key)
                records.append({"method":method["id"],"family":method["family"],"seed":method.get("seed"),"dataset":dataset,"population":population,"indices":indices.tolist(),"metrics":result})
    write_json(out/"results.json",{"method_lock_sha256":digest,"prediction_manifest_sha256":sha256(BENCH/"predictions/manifest.json"),"records":records,"script_sha256":sha256(Path(__file__)),"metric":"Reproduction/recovery angular degrees; no surface DeltaE","invalid_policy":"Unsupported rows stay in counts; full mean includes neutral fallback diagnostics; selective curves use valid rows only"})
    print(json.dumps({"method_population_records":len(records)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("predict","evaluate"))
    parser.add_argument("--digest",required=True)
    args=parser.parse_args()
    (predict if args.action=="predict" else evaluate)(args.digest)
