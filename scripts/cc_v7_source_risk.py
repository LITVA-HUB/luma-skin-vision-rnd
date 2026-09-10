"""Evaluate frozen standard risk heads on reused source validation, never fit there."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from cc_v2_select import raw_predict, selective_result
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES, source_rows
from cc_v7_verify import verify_manifest
from threadpoolctl import threadpool_limits

from luma_skin_vision.cc.v2 import risk_features_invariant
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT=Path(__file__).resolve().parents[1]


def run(folder,risk,out):
    folder,risk,out=(Path(p).resolve() for p in (folder,risk,out))
    if out.exists():
        raise ValueError("Immutable source-risk evaluation exists")
    verify_manifest(folder,"artifact_manifest.json")
    verify_manifest(risk,"manifest.json")
    config=json.loads((folder/"config.json").read_text())
    risk_config=json.loads((risk/"config.json").read_text())
    if risk_config["training_config_sha256"]!=sha256(folder/"config.json"):
        raise ValueError("Wrong training/risk binding")
    data=ROOT/"data/processed/cc128"
    if {k:sha256(data/k) for k in DATA_HASHES}!=DATA_HASHES:
        raise ValueError("Source changed")
    rows=json.loads((data/"cube_manifest.json").read_text())
    selected,_,val=source_rows(rows)
    indices=selected[val]
    ids=[rows[i]["id"] for i in indices]
    if ids!=config["validation_ids"] or set(ids)&set(risk_config["source_ids"]):
        raise ValueError("Validation/risk role mismatch")
    x=torch.from_numpy(read_npz_rows(data/"cube.npz","images",indices,2234).astype(np.float32))
    gt=read_npz_rows(data/"cube.npz","gt",indices,2234).astype(np.float32).astype(np.float64)
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    records=[]
    with torch.no_grad(),threadpool_limits(limits=2):
        for arm in config["arms"]:
            with np.load(folder/arm/"best_validation.npz") as stored:
                pred,context,valid=(stored[k].copy() for k in ("pred","context","valid"))
                np.testing.assert_array_equal(gt,stored["gt"])
            features=risk_features_invariant(x,torch.from_numpy(pred),torch.from_numpy(context))
            selection=json.loads((risk/arm/"selection.json").read_text())
            if selection["checkpoint_sha256"]!=sha256(folder/arm/"best.pt"):
                raise ValueError("Risk head bound to wrong checkpoint")
            arms={}
            for block in ("context","cheap","combined"):
                path=risk/arm/(block+".joblib")
                if selection["heads"][block]["artifact_sha256"]!=sha256(path):
                    raise ValueError("Risk artifact changed")
                payload=joblib.load(path)
                scores=raw_predict(payload["model"],features[block].numpy().astype(np.float64))*payload["scale"]
                cal_scores=np.array(selection["heads"][block]["cal_scores"])
                arms[block]=selective_result(pred.astype(np.float64),gt,scores,ids,cal_scores,valid)
            records.append({"arm":arm,"heads":arms})
            print(json.dumps({"arm":arm,"primary_combined_risk80":arms["combined"]["selective"]["fixed"]["80"]["mean"]}),flush=True)
    write_json(out/"results.json",{"scope":"REUSED SOURCE DEVELOPMENT VALIDATION; all heads fitted elsewhere, estimator checkpoints selected here; not new test evidence","primary_estimator_run":config["primary"],"primary_head":"combined","training_config_sha256":sha256(folder/"config.json"),"risk_manifest_sha256":sha256(risk/"manifest.json"),"validation_ids":ids,"records":records,"script_sha256":sha256(Path(__file__))})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    parser.add_argument("--risk",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    run(args.run,args.risk,args.out)
