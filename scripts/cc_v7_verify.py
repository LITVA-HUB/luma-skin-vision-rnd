"""CPU checkpoint/teacher replay and independent FP64 rescoring of V7 artifacts."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED","1")

import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES, source_rows
from cc_v7_core import SemanticColorNet, teacher_render
from cc_v7_experiment import evaluate
from torch.nn import functional as F

from luma_skin_vision.cc.v2 import CompactResidualCC
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT=Path(__file__).resolve().parents[1]


def verify_manifest(folder,name):
    manifest=json.loads((folder/name).read_text())
    for relative,digest in manifest["sha256"].items():
        if sha256(folder/relative)!=digest:
            raise ValueError("Artifact changed: "+relative)


def score(pred,gt):
    ratio=gt.astype(np.float64)/pred.astype(np.float64)
    cosine=ratio.sum(-1)/(np.sqrt(3)*np.linalg.norm(ratio,axis=-1))
    return np.degrees(np.arccos(np.clip(cosine,-1,1)))


def teacher_replay(folder):
    verify_manifest(folder,"manifest.json")
    config=json.loads((folder/"config.json").read_text())
    data=ROOT/"data/processed/cc128"
    rows=json.loads((data/"cube_manifest.json").read_text())
    # Fixed first/middle/last TRAIN positions, never external validation/test.
    positions=np.array([0,len(config["train_indices"])//2,len(config["train_indices"])-1])
    indices=np.asarray(config["train_indices"])[positions]
    if any(rows[i]["subset"]!="train" for i in indices):
        raise ValueError("Teacher replay crossed training scope")
    x=torch.from_numpy(read_npz_rows(data/"cube.npz","images",indices,2234).astype(np.float32))
    gt=torch.from_numpy(read_npz_rows(data/"cube.npz","gt",indices,2234).astype(np.float32))
    teacher_root=ROOT/"artifacts/teachers/dinov2-7764ea0f912e"
    sys.path.insert(0,str(teacher_root/"source"))
    from dinov2.hub.backbones import dinov2_vits14
    teacher=dinov2_vits14(pretrained=False).eval()
    weight=teacher_root/"dinov2_vits14_pretrain.pth"
    if sha256(weight)!=config["teacher_weight_sha256"]:
        raise ValueError("Wrong teacher weights")
    teacher.load_state_dict(torch.load(weight,map_location="cpu",weights_only=True))
    records=[]
    with torch.no_grad():
        for kind in ("raw","canonical"):
            stored=np.load(folder/(kind+".npy"),allow_pickle=False,mmap_mode="r")
            for flip in (0,1):
                rendered=teacher_render(x.flip(-1) if flip else x,gt if kind=="canonical" else None)
                tokens=teacher.forward_features(rendered)["x_norm_patchtokens"]
                grid=tokens.transpose(1,2).reshape(len(x),384,16,16)
                pooled=F.adaptive_avg_pool2d(grid,(4,4)).flatten(2).transpose(1,2)
                actual=F.normalize(pooled,dim=-1).numpy()
                difference=float(abs(actual-stored[positions,flip]).max())
                if difference>2e-5:
                    raise ValueError("Teacher cache replay changed")
                records.append({"kind":kind,"flip":flip,"max_abs_difference":difference,"positions":positions.tolist()})
    return records


def run(folder,out,teacher_only):
    folder=Path(folder).resolve()
    out=Path(out).resolve()
    if out.exists():
        raise ValueError("Immutable verification output exists")
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    data=ROOT/"data/processed/cc128"
    if {k:sha256(data/k) for k in DATA_HASHES}!=DATA_HASHES:
        raise ValueError("Data digest changed")
    if teacher_only:
        records=teacher_replay(folder)
        write_json(out,{"status":"TRAIN-ONLY TEACHER REPLAY PASSED","records":records,"script_sha256":sha256(Path(__file__))})
        print(json.dumps({"verified":len(records),"kind":"teacher"}))
        return
    verify_manifest(folder,"artifact_manifest.json")
    config=json.loads((folder/"config.json").read_text())
    for name in ("cc_v7_core.py","cc_v7_experiment.py","cc_v7_teacher.py"):
        if sha256(ROOT/"scripts"/name)!=config["scripts_sha256"][name]:
            raise ValueError("Executed numerical source changed")
    rows=json.loads((data/"cube_manifest.json").read_text())
    selected,_,val=source_rows(rows)
    indices=selected[val]
    if config["validation_ids"]!=[rows[i]["id"] for i in indices]:
        raise ValueError("Wrong validation identities")
    x=torch.from_numpy(read_npz_rows(data/"cube.npz","images",indices,2234).astype(np.float32))
    gt=read_npz_rows(data/"cube.npz","gt",indices,2234).astype(np.float64)
    records=[]
    with torch.no_grad():
        for arm in config["arms"]:
            for checkpoint in ("best","last"):
                model=SemanticColorNet().eval()
                path=folder/arm/(checkpoint+".pt")
                state=torch.load(path,map_location="cpu",weights_only=True)
                model.load_state_dict(state)
                _,actual=evaluate(model,x,torch.from_numpy(gt.astype(np.float32)),16)
                with np.load(folder/arm/(checkpoint+"_validation.npz")) as stored:
                    difference=max(float(abs(actual[k]-stored[k]).max()) for k in ("pred","context"))
                    if difference>1e-3:
                        raise ValueError("Checkpoint replay differs")
                    error=score(stored["pred"],gt.astype(np.float32))
                    error_difference=float(abs(error-stored["reproduction"]).max())
                    if error_difference>1e-7:
                        raise ValueError("Independent error differs")
                    rounded=float(abs(error-score(stored["pred"],gt)).max())
                plain=CompactResidualCC("direct","large").eval()
                deployment={k:v for k,v in state.items() if not k.startswith("teacher_projection.")}
                plain.load_state_dict(deployment,strict=True)
                plain_prediction=plain(x[:16])[0]
                student_prediction=model(x[:16])["pred"]
                if not torch.equal(plain_prediction,student_prediction):
                    raise ValueError("Removing training-only head changed inference")
                records.append({"arm":arm,"checkpoint":checkpoint,"max_cpu_prediction_context_replay_difference":difference,"max_independent_reproduction_error_difference":error_difference,"max_source_gt_rounding_degrees":rounded,"deployment_head_removal_bitwise_equal":True,"checkpoint_sha256":sha256(path)})
    write_json(out,{"status":"ALL V7 CHECKPOINTS AND INDEPENDENT ERRORS PASSED","primary":config["primary"],"records":records,"script_sha256":sha256(Path(__file__))})
    print(json.dumps({"verified":len(records),"primary":config["primary"]}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--teacher-only",action="store_true")
    args=parser.parse_args()
    run(args.run,args.out,args.teacher_only)
