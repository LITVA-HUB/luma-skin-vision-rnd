"""Matched source-only compact semantic/sensor experiment; teacher is train-only."""
import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")

import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES, source_rows
from cc_v5_experiment import capture_state, restore_state, state_digest
from cc_v7_core import SemanticColorNet, sensor_matrix, sensor_transform
from cc_v7_teacher import training_indices
from torch.nn import functional as F

from luma_skin_vision.cc.core import angular, reproduction, summarize
from luma_skin_vision.cc.v2_experiment import source_snapshot
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT=Path(__file__).resolve().parents[1]
ARMS=("gt_native","gt_sensor","raw_teacher_sensor","canonical_teacher_sensor","canonical_teacher_native")


def augment_batch(x,gt,aug_rng,sensor_rng,with_sensor):
    flip=torch.rand(len(x),device=x.device,generator=aug_rng)<.5
    exposure=(torch.rand(len(x),1,1,1,device=x.device,generator=aug_rng)-.5).exp()
    image=torch.where(flip[:,None,None,None],x.flip(-1),x)*exposure
    matrix=sensor_matrix(len(x),generator=sensor_rng,device=x.device,dtype=x.dtype)
    if with_sensor:
        image,gt=sensor_transform(image,gt,matrix)
    return image,gt,flip,matrix


def distillation_loss(student,teacher):
    a=F.normalize(student,dim=-1)
    b=F.normalize(teacher.detach(),dim=-1)
    return (1-(a*b).sum(-1)).mean()


def point_loss(pred,gt):
    # Same smoothed physical angle as V5, directly in positive RGB: avoid
    # re-entering V5's bounded query API through a normalize/log roundtrip.
    ratio=gt/pred
    ratio=ratio/ratio.amax(-1,keepdim=True)
    red,green,blue=ratio.unbind(-1)
    squared=(red-green).square()+(green-blue).square()+(blue-red).square()
    return (torch.atan2((squared+1e-16).sqrt(),ratio.sum(-1))*180/math.pi).mean()


def load_teacher(folder,rows,selected,train_ix,device):
    folder=Path(folder)
    manifest=json.loads((folder/"manifest.json").read_text())
    if manifest["status"]!="COMPLETE TRAIN-ONLY TEACHER TARGETS":
        raise ValueError("Incomplete teacher cache")
    for name,digest in manifest["sha256"].items():
        if sha256(folder/name)!=digest:
            raise ValueError("Teacher artifact hash changed")
    config=json.loads((folder/"config.json").read_text())
    expected=training_indices(rows,allow_official_test_group_overlap=True)
    if config["data_sha256"]!=DATA_HASHES or config["train_indices"]!=expected.tolist():
        raise ValueError("Wrong teacher role or source identity")
    if config["train_ids"]!=[rows[i]["id"] for i in expected] or not np.array_equal(selected[train_ix],expected):
        raise ValueError("Unaligned training feature rows")
    if config["core_sha256"]!=sha256(ROOT/"scripts/cc_v7_core.py"):
        raise ValueError("Teacher render bytes changed")
    targets={}
    for kind in ("raw","canonical"):
        array=np.load(folder/(kind+".npy"),allow_pickle=False)
        if array.shape!=(len(expected),2,16,384) or not np.isfinite(array).all():
            raise ValueError("Invalid teacher targets")
        if not np.allclose(np.linalg.norm(array,axis=-1),1,atol=1e-5):
            raise ValueError("Teacher targets are not unit length")
        targets[kind]=torch.zeros(len(selected),2,16,384,device=device)
        targets[kind][torch.as_tensor(train_ix,device=device)]=torch.from_numpy(array).to(device)
    return targets,sha256(folder/"manifest.json")


def train_epoch(model,optimizer,scheduler,x,gt,targets,train_ix,epoch,seed,batch,arm):
    model.train()
    device=x.device
    order_rng=torch.Generator(device=device).manual_seed(seed*100000+epoch*10)
    aug_rng=torch.Generator(device=device).manual_seed(seed*100000+epoch*10+1)
    sensor_rng=torch.Generator(device=device).manual_seed(seed*100000+epoch*10+2)
    order=train_ix[torch.randperm(len(train_ix),device=device,generator=order_rng)]
    total=np.zeros(2)
    seen=0
    for ix in order.split(batch):
        xb,gb,flip,_=augment_batch(x[ix],gt[ix],aug_rng,sensor_rng,arm.endswith("_sensor"))
        optimizer.zero_grad(set_to_none=True)
        output=model(xb)
        point=point_loss(output["pred"],gb)
        kd=point.new_zeros(())
        if "teacher" in arm:
            kind="canonical" if arm.startswith("canonical") else "raw"
            # Teacher views always derive from untransformed TRAIN images; the
            # flip index selects an actual teacher forward, not token reversal.
            target=targets[kind][ix,flip.long()]
            kd=distillation_loss(output["teacher_features"],target)
        loss=point+kd
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {arm} epoch{epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        optimizer.step()
        total+=np.array([point.item(),kd.item()])*len(ix)
        seen+=len(ix)
    scheduler.step()
    return dict(zip(("reproduction_loss","distillation_loss"),(total/seen).tolist()))


@torch.no_grad()
def evaluate(model,x,gt,batch):
    model.eval()
    arrays={k:[] for k in ("pred","context","valid")}
    for start in range(0,len(x),batch):
        output=model(x[start:start+batch])
        for key in arrays:
            arrays[key].append(output[key].cpu().numpy())
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    label=gt.cpu().numpy().astype(np.float64)
    arrays["gt"]=label
    arrays["reproduction"]=reproduction(arrays["pred"].astype(np.float64),label)
    arrays["recovery"]=angular(arrays["pred"].astype(np.float64),label)
    return {"reproduction":summarize(arrays["reproduction"]),"recovery":summarize(arrays["recovery"]),"valid_fraction":float(arrays["valid"].mean()),"risk":"NOT TRAINED OR MEASURED"},arrays


def run(args):
    out=Path(args.out).resolve()
    if out.exists() or args.epochs<1 or args.batch<2:
        raise ValueError("Require a new run directory, positive epochs and batch>=2")
    data=ROOT/"data/processed/cc128"
    if {k:sha256(data/k) for k in DATA_HASHES}!=DATA_HASHES:
        raise ValueError("Source digest changed")
    rows=json.loads((data/"cube_manifest.json").read_text())
    selected,train_ix,val_ix=source_rows(rows)
    if (len(train_ix),len(val_ix))!=(1126,119):
        raise ValueError("Wrong source roles")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(args.device)
    targets,teacher_hash=load_teacher(args.teacher,rows,selected,train_ix,device)
    x=torch.from_numpy(read_npz_rows(data/"cube.npz","images",selected,2234).astype(np.float32)).to(device)
    gt=torch.from_numpy(read_npz_rows(data/"cube.npz","gt",selected,2234).astype(np.float32)).to(device)
    if not torch.isfinite(x).all() or (x<0).any() or not (torch.isfinite(gt)&(gt>0)).all():
        raise ValueError("Invalid source pixels or GT")
    out.mkdir(parents=True)
    identity,snapshot=source_snapshot(out)
    scripts=[p for p in (ROOT/"scripts").glob("cc_*.py")]
    hashes={}
    for path in scripts:
        dest=out/"source_snapshot/scripts"/path.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        hashes[path.name]=sha256(dest)
    shutil.copyfile(ROOT/"docs/research/cc_v7_semantic_sensor_protocol.md",out/"protocol.md")
    config={"arguments":vars(args),"arms":ARMS,"primary":(args.epochs,args.batch)==(120,32),"source_identity":identity,"snapshot":snapshot,"scripts_sha256":hashes,"data_sha256":DATA_HASHES,"teacher_manifest_sha256":teacher_hash,"torch":torch.__version__,"gpu":torch.cuda.get_device_name() if device.type=="cuda" else None,"train_ids":[rows[i]["id"] for i in selected[train_ix]],"validation_ids":[rows[i]["id"] for i in selected[val_ix]],"source_cache_mib":(x.numel()*x.element_size()+gt.numel()*gt.element_size())/2**20,"teacher_cache_mib":sum(t.numel()*t.element_size() for t in targets.values())/2**20,"limitation":"Reused source validation, no test/camera/risk/skin accuracy evidence. Official test overlaps capture dates, not decoded."}
    write_json(out/"config.json",config)
    train_ix=torch.as_tensor(train_ix,device=device)
    val_ix=torch.as_tensor(val_ix,device=device)
    model=SemanticColorNet().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=.00002)
    initial=capture_state(model,optimizer,scheduler)
    initial_hash=state_digest(initial)
    receipts=[]
    # Replay the first epoch of EVERY arm, including teacher/sensor paths.
    for arm in ARMS:
        values=[]
        for _ in range(2):
            restore_state(initial,model,optimizer,scheduler)
            stats=train_epoch(model,optimizer,scheduler,x,gt,targets,train_ix,1,args.seed,args.batch,arm)
            values.append({"state_sha256":state_digest(capture_state(model,optimizer,scheduler)),"training":stats})
        if values[0]!=values[1]:
            raise RuntimeError("Exact first-epoch replay failed: "+arm)
        receipts.append({"arm":arm,"bitwise_equal":True,**values[0]})
    write_json(out/"first_epoch_replay.json",{"initial_state_sha256":initial_hash,"records":receipts})
    results={}
    for arm in ARMS:
        folder=out/arm
        folder.mkdir()
        restore_state(initial,model,optimizer,scheduler)
        if state_digest(capture_state(model,optimizer,scheduler))!=initial_hash:
            raise RuntimeError("Unequal initial state")
        if device.type=="cuda":
            torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        best=math.inf
        for epoch in range(1,args.epochs+1):
            stats=train_epoch(model,optimizer,scheduler,x,gt,targets,train_ix,epoch,args.seed,args.batch,arm)
            metrics,arrays=evaluate(model,x[val_ix],gt[val_ix],args.batch)
            if metrics["valid_fraction"]<.99:
                raise ValueError("Invalid source predictions")
            key=metrics["reproduction"]["mean"]
            if key<best:
                best,best_epoch=key,epoch
                torch.save(model.state_dict(),folder/"best.pt")
                np.savez_compressed(folder/"best_validation.npz",**arrays)
                write_json(folder/"best_metrics.json",{"epoch":epoch,"arm":arm,**metrics})
            with (folder/"history.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps({"epoch":epoch,"training":stats,"validation":metrics})+"\n")
            if epoch%10==0 or epoch==args.epochs:
                print(json.dumps({"arm":arm,"epoch":epoch,"val_repro":key,"best":best,"training":stats,"seconds":round(time.perf_counter()-started,1)}),flush=True)
        torch.save(model.state_dict(),folder/"last.pt")
        np.savez_compressed(folder/"last_validation.npz",**arrays)
        write_json(folder/"last_metrics.json",{"epoch":args.epochs,"arm":arm,**metrics})
        results[arm]={"best_mean_reproduction":best,"best_epoch":best_epoch,"last_mean_reproduction":key,"initial_state_sha256":initial_hash,"training_parameters":sum(p.numel() for p in model.parameters()),"deployment_parameters":sum(p.numel() for n,p in model.named_parameters() if not n.startswith("teacher_projection.")),"best_checkpoint_sha256":sha256(folder/"best.pt"),"best_checkpoint_bytes_including_training_projection":(folder/"best.pt").stat().st_size,"elapsed_seconds":time.perf_counter()-started,"timing_scope":"Training plus epoch validation/checkpoint I/O; no teacher extraction; not inference latency","training_peak_mib_including_data_teacher_caches_and_initial_state":torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else None}
        write_json(folder/"result.json",results[arm])
        if state_digest(initial)!=initial_hash:
            raise RuntimeError("Mutated initial state")
    write_json(out/"result.json",{"status":"complete","arms":results})
    write_json(out/"artifact_manifest.json",{"sha256":{p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}})
    print(json.dumps({"status":"complete","arms":results}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True)
    parser.add_argument("--teacher",default="experiments/runs/ccv7_teacher_train_v1")
    parser.add_argument("--epochs",type=int,default=120)
    parser.add_argument("--batch",type=int,default=32)
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    arguments=parser.parse_args()
    existed=Path(arguments.out).exists()
    try:
        run(arguments)
    except Exception as exc:
        if not existed and Path(arguments.out).is_dir():
            write_json(Path(arguments.out)/"failure.json",{"type":type(exc).__name__,"message":str(exc)})
        raise
