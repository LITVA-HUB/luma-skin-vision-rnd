"""Fixed virtual-sensor diagnostics; never substitute for real camera evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES, source_rows
from cc_v7_verify import verify_manifest

from luma_skin_vision.cc.core import angular, reproduction, summarize, unit
from luma_skin_vision.cc.v2 import CompactResidualCC
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT=Path(__file__).resolve().parents[1]
MATRICES={"identity":np.eye(3),"gain_red":np.diag([2,1,.5]),"gain_blue":np.diag([.5,1,2]),"mix_small":np.array([[.9,.05,.05],[.05,.9,.05],[.05,.05,.9]]),"mix_train_range":np.array([[.65,.25,.10],[.10,.70,.20],[.20,.10,.70]]),"mix_extrapolation":np.array([[.4,.4,.2],[.2,.5,.3],[.3,.2,.5]])}


@torch.no_grad()
def prediction(model,x):
    return torch.cat([model(batch)[0] for batch in x.split(32)]).numpy()


def run(folder,out):
    folder,out=Path(folder).resolve(),Path(out).resolve()
    if out.exists():
        raise ValueError("Immutable diagnostics already exist")
    verify_manifest(folder,"artifact_manifest.json")
    config=json.loads((folder/"config.json").read_text())
    seed=config["arguments"]["seed"]
    data=ROOT/"data/processed/cc128"
    if {k:sha256(data/k) for k in DATA_HASHES}!=DATA_HASHES:
        raise ValueError("Wrong source cache")
    rows=json.loads((data/"cube_manifest.json").read_text())
    selected,_,val=source_rows(rows)
    ix=selected[val]
    if config["validation_ids"]!=[rows[i]["id"] for i in ix]:
        raise ValueError("Validation identities changed")
    x=torch.from_numpy(read_npz_rows(data/"cube.npz","images",ix,2234).astype(np.float32))
    gt=read_npz_rows(data/"cube.npz","gt",ix,2234).astype(np.float64)
    models=[(arm+"_"+kind,"direct",folder/arm/(kind+".pt"),False) for arm in config["arms"] for kind in ("best","last")]
    models += [("historical_v2_"+mode,mode,ROOT/f"experiments/runs/ccv2_{mode}_large_g0_s{seed}/model.pt",True) for mode in ("direct","sog")]
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    records=[]
    for name,mode,path,old_format in models:
        state=torch.load(path,map_location="cpu",weights_only=True)
        state=state["state"] if old_format else {k:v for k,v in state.items() if not k.startswith("teacher_projection.")}
        model=CompactResidualCC(mode,"large").eval()
        model.load_state_dict(state,strict=True)
        native=prediction(model,x)
        arrays={"ids":np.array([rows[i]["id"] for i in ix]),"native_prediction":native}
        metrics={}
        for label,matrix in MATRICES.items():
            transformed=torch.einsum("ij,njhw->nihw",torch.from_numpy(matrix.astype(np.float32)),x)
            pred=prediction(model,transformed)
            target=unit(gt@matrix.T)
            expected=unit(native.astype(np.float64)@matrix.T)
            error=reproduction(pred.astype(np.float64),target)
            consistency=angular(pred.astype(np.float64),expected)
            metrics[label]={"reproduction":summarize(error),"equivariance_recovery_degrees":summarize(consistency)}
            arrays[label+"_pred"]=pred
            arrays[label+"_gt"]=target
            if mode=="sog" and label in ("identity","gain_red","gain_blue") and consistency.mean()>.001:
                raise ValueError("Diagonal-equivariant control failed")
        np.savez_compressed(out/(name+".npz"),**arrays)
        records.append({"model":name,"checkpoint_sha256":sha256(path),"metrics":metrics})
        print(json.dumps({"model":name,"native_mean":metrics["identity"]["reproduction"]["mean"],"extrapolation_mean":metrics["mix_extrapolation"]["reproduction"]["mean"]}),flush=True)
    write_json(out/"results.json",{"scope":"TRANSFORMED SOURCE VALIDATION, NOT REAL NEW CAMERAS; original GT float64 before M","primary_estimator_run":config["primary"],"source_training_config_sha256":sha256(folder/"config.json"),"matrices":{k:v.tolist() for k,v in MATRICES.items()},"records":records,"script_sha256":sha256(Path(__file__))})
    write_json(out/"manifest.json",{"sha256":{p.name:sha256(p) for p in out.iterdir() if p.is_file()}})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    run(args.run,args.out)
