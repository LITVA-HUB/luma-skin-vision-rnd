"""Train-only frozen DINOv2 patch targets; no network access or inference teacher."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED","1")

import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES
from cc_v7_core import teacher_render
from torch.nn import functional as F

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT=Path(__file__).resolve().parents[1]
WEIGHT_SHA="b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def training_indices(rows,*,allow_official_test_group_overlap=False):
    selected=[i for i,r in enumerate(rows) if r["subset"]=="train"]
    if not selected or len({r["id"] for r in rows})!=len(rows):
        raise ValueError("Unique rows and nonempty training set required")
    groups={rows[i]["group"] for i in selected}
    held_out=[r for r in rows if r["subset"]!="train"]
    if allow_official_test_group_overlap:
        held_out=[r for r in held_out if not (r["subset"]=="test" and r.get("official_split")=="test")]
    if groups & {r["group"] for r in held_out}:
        raise ValueError("Train group overlaps held-out roles")
    return np.asarray(selected,dtype=np.int64)


def run(out,batch):
    out=Path(out).resolve()
    if out.exists():
        raise ValueError("Immutable teacher cache already exists")
    data=ROOT/"data/processed/cc128"
    if {k:sha256(data/k) for k in DATA_HASHES}!=DATA_HASHES:
        raise ValueError("Frozen source data changed")
    rows=json.loads((data/"cube_manifest.json").read_text())
    ix=training_indices(rows,allow_official_test_group_overlap=True)
    if len(ix)!=1126:
        raise ValueError("Wrong train role count")
    teacher_root=ROOT/"artifacts/teachers/dinov2-7764ea0f912e"
    acquisition=json.loads((teacher_root/"acquisition.json").read_text())
    for name,r in acquisition["files"].items():
        if sha256(teacher_root/"source"/name)!=r["sha256"]:
            raise ValueError("Original teacher source digest changed")
    weight=teacher_root/"dinov2_vits14_pretrain.pth"
    if sha256(weight)!=WEIGHT_SHA:
        raise ValueError("Teacher weight digest mismatch")
    sys.path.insert(0,str(teacher_root/"source"))
    from dinov2.hub.backbones import dinov2_vits14
    torch.set_num_threads(2)
    model=dinov2_vits14(pretrained=False).eval()
    model.load_state_dict(torch.load(weight,map_location="cpu",weights_only=True),strict=True)
    model.requires_grad_(False)
    x=read_npz_rows(data/"cube.npz","images",ix,2234).astype(np.float32)
    gt=read_npz_rows(data/"cube.npz","gt",ix,2234).astype(np.float32)
    out.mkdir(parents=True)
    config={"teacher_weight_sha256":WEIGHT_SHA,"teacher_source_commit":acquisition["commit"],"data_sha256":DATA_HASHES,"train_indices":ix.tolist(),"train_ids":[rows[i]["id"] for i in ix],"device":"cpu","batch":batch,"torch":torch.__version__,"script_sha256":sha256(Path(__file__)),"core_sha256":sha256(ROOT/"scripts/cc_v7_core.py"),"teacher_parameters":sum(p.numel() for p in model.parameters()),"scope":"TRAIN ONLY; two view types and two horizontal orientations; no held-out images/GT"}
    write_json(out/"config.json",config)
    train_groups={rows[i]["group"] for i in ix}
    overlaps={role:sorted(train_groups & {r["group"] for r in rows if r["subset"]==role}) for role in ("val","risk","cal","test")}
    write_json(out/"role_audit.json",{"training_rows":len(ix),"training_groups":len(train_groups),"train_group_overlap":overlaps,"scope":"Official SimpleCube test overlaps dates; no test image/GT extraction. Source fitting roles remain date-disjoint. This screen does not evaluate the official test."})
    started=time.perf_counter()
    with torch.no_grad():
        for kind in ("raw","canonical"):
            cache=np.empty((len(ix),2,16,384),dtype=np.float32)
            for flip in (0,1):
                for start in range(0,len(x),batch):
                    xb=torch.from_numpy(x[start:start+batch])
                    if flip:
                        xb=xb.flip(-1)
                    gb=torch.from_numpy(gt[start:start+batch]) if kind=="canonical" else None
                    rendered=teacher_render(xb,gb)
                    tokens=model.forward_features(rendered)["x_norm_patchtokens"]
                    grid=tokens.transpose(1,2).reshape(len(xb),384,16,16)
                    pooled=F.adaptive_avg_pool2d(grid,(4,4)).flatten(2).transpose(1,2)
                    normalized=F.normalize(pooled,dim=-1).numpy()
                    if not np.isfinite(normalized).all():
                        raise ValueError("Nonfinite teacher target")
                    cache[start:start+len(xb),flip]=normalized
                    if start%(batch*16)==0 or start+batch>=len(x):
                        print(json.dumps({"kind":kind,"flip":flip,"done":min(start+batch,len(x)),"total":len(x),"seconds":round(time.perf_counter()-started,1)}),flush=True)
            np.save(out/(kind+".npy"),cache)
    write_json(out/"manifest.json",{"status":"COMPLETE TRAIN-ONLY TEACHER TARGETS","elapsed_seconds":time.perf_counter()-started,"sha256":{p.name:sha256(p) for p in out.iterdir() if p.is_file()}})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True)
    parser.add_argument("--batch",type=int,default=8)
    args=parser.parse_args()
    run(args.out,args.batch)
