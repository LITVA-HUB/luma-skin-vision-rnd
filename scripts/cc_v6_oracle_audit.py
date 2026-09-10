"""Post-hoc correction of unused V6 oracle diagnostics; training outputs stay immutable."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import source_rows
from cc_v4_experiment import action_rgb

from luma_skin_vision.cc.core import reproduction
from luma_skin_vision.cc.v2 import channel_anchor
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json


def corrected_oracle(point,trajectory,anchor,gt):
    offsets=np.array([(a,b) for a in np.linspace(-1,1,5) for b in np.linspace(-1,1,5)])
    candidates=[]
    for step,radius in enumerate((.24,.06,.03,.015)):
        center=point if step==0 else trajectory[:,step-1]
        candidates.append(np.clip(center[:,None]+radius*offsets,-2,2)+anchor[:,None])
        if step:
            candidates.append((point+anchor)[:,None])
    candidates=np.concatenate(candidates,1)
    pred=action_rgb(candidates)
    target=np.broadcast_to(gt[:,None],pred.shape)
    error=reproduction(pred.reshape(-1,3),target.reshape(-1,3)).reshape(len(gt),-1)
    return error[:,:51].min(-1),error.min(-1),candidates


def run(seed):
    root=Path(__file__).resolve().parents[1]
    folder=root/f"experiments/runs/ccv6_sog_s{seed}"
    if not (folder/"result.json").exists():
        raise ValueError("Wait for complete immutable run")
    data=root/"data/processed/cc128"
    rows=json.loads((data/"cube_manifest.json").read_text())
    selected,_,val=source_rows(rows)
    x=torch.from_numpy(read_npz_rows(data/"cube.npz","images",selected[val],2234).astype(np.float32))
    gt=read_npz_rows(data/"cube.npz","gt",selected[val],2234).astype(np.float32).astype(np.float64)
    anchor=channel_anchor(x/x.amax((1,2,3),keepdim=True),p=6).log()
    anchor=(anchor[:,[0,2]]-anchor[:,1:2]).numpy().astype(np.float64)
    records=[]
    for path in folder.glob("*/*_validation.npz"):
        with np.load(path) as stored:
            point=stored["point_action"].astype(np.float64)-anchor
            trajectory=stored["trajectory_actions"].astype(np.float64)-anchor[:,None]
            a,b,_=corrected_oracle(point,trajectory,anchor,gt)
            records.append({"file":path.relative_to(folder).as_posix(),"input_sha256":sha256(path),"corrected_oracle2":a.tolist(),"corrected_oracle4":b.tolist(),"old_oracle2_max_difference":float(abs(a-stored["oracle2"]).max()),"old_oracle4_max_difference":float(abs(b-stored["oracle4"]).max())})
    out=root/f"docs/benchmarks/cc_v6/seed{seed}_oracle_correction.json"
    if out.exists():
        raise ValueError("Existing audit receipt")
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,{"status":"CORRECTED UNUSED DIAGNOSTIC; original artifacts preserved","reason":"Old V4 oracle helper clipped absolute coordinates; V6 policy clips residual coordinates. Training/checkpoint selection/predicted errors/risk never consume oracle metrics.","script_sha256":sha256(Path(__file__)),"records":records})
    print(json.dumps({"records":len(records),"max_oracle2_change":max(r["old_oracle2_max_difference"] for r in records)}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    run(p.parse_args().seed)
