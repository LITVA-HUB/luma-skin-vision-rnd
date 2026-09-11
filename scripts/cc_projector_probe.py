"""TRAIN-only subspace invariance and GT-assisted positive color-cone diagnostics."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES
from cc_v7_teacher import training_indices
from scipy.optimize import nnls
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

from luma_skin_vision.cc.core import angular, reproduction, summarize, unit
from luma_skin_vision.cc.v2 import CompactResidualCC
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT=Path(__file__).resolve().parents[1]


def projector_features(x,probes):
    if x.ndim!=3 or x.shape[-1]!=3 or probes.ndim!=2 or probes.shape[0]!=x.shape[1]:
        raise ValueError("Expected BxNx3 colors and NxK fixed probes")
    if not torch.isfinite(x).all() or not torch.isfinite(probes).all():
        raise ValueError("Finite geometry required")
    values=x.double()
    singular=torch.linalg.svdvals(values)
    condition=singular[:,0]/singular[:,-1]
    valid=(singular[:,0]>0)&(condition<=1e4)&torch.isfinite(condition)
    features=values.new_zeros(len(x),x.shape[1],probes.shape[1])
    if valid.any():
        q,_=torch.linalg.qr(values[valid],mode="reduced")
        features[valid]=q@(q.transpose(1,2)@probes.double())
    return {"features":features,"condition":condition,"valid":valid}


def cone_oracle(x,gt):
    x,gt=np.asarray(x,dtype=np.float64),np.asarray(gt,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=3 or gt.shape!=(3,) or not np.isfinite(x).all() or np.any(x<0) or not np.isfinite(gt).all() or np.any(gt<=0):
        raise ValueError("Expected finite nonnegative colors and positive GT")
    norms=np.linalg.norm(x,axis=-1)
    active=norms>1e-12
    if not active.any():
        raise ValueError("Empty positive cone")
    a=(x[active]/norms[active,None]).T
    b=gt/np.linalg.norm(gt)
    weights,residual_norm=nnls(a,b,maxiter=max(3000,30*a.shape[1]))
    projection=a@weights
    residual=projection-b
    dual=a.T@residual
    # Necessary and sufficient convex NNLS optimality conditions: nonnegative
    # primal/dual and complementary slackness. No black-box oracle assertion.
    violation=max(float(max(0,-weights.min())),float(max(0,-dual.min())),float(np.abs(weights*dual).max()))
    orthogonality=float(abs(projection@residual))
    if violation>1e-7 or orthogonality>1e-7 or np.linalg.norm(projection)<=1e-12:
        raise ValueError("Cone projection failed KKT verification")
    pred=unit(projection)
    positive=bool((pred>1e-12).all())
    return {"pred":pred,"recovery":float(angular(pred,b)),"feasible_reproduction":float(reproduction(pred,b)) if positive else None,"positive_prediction":positive,"kkt_violation":violation,"orthogonality":orthogonality,"projection_residual":float(residual_norm)}


def run(out):
    out=Path(out).resolve()
    if out.exists():
        raise ValueError("Immutable probe output exists")
    data=ROOT/"data/processed/cc128"
    if {k:sha256(data/k) for k in DATA_HASHES}!=DATA_HASHES:
        raise ValueError("Source identity changed")
    rows=json.loads((data/"cube_manifest.json").read_text())
    ix=training_indices(rows,allow_official_test_group_overlap=True)
    if len(ix)!=1126:
        raise ValueError("Wrong TRAIN population")
    torch.set_num_threads(2)
    images=torch.from_numpy(read_npz_rows(data/"cube.npz","images",ix,2234).astype(np.float32))
    gt=read_npz_rows(data/"cube.npz","gt",ix,2234).astype(np.float64)
    out.mkdir(parents=True)
    write_json(out/"config.json",{"scope":"TRAIN ONLY; GT-assisted oracle, no new learned model or held-out accuracy","train_indices":ix.tolist(),"train_ids":[rows[i]["id"] for i in ix],"data_sha256":DATA_HASHES,"script_sha256":sha256(Path(__file__)),"probe_seed":17,"grids":[4,8,16],"condition_limit":1e4,"protocol_sha256":sha256(ROOT/"docs/research/projector_cone_probe_protocol.md")})
    started=time.perf_counter()
    records=[]
    with torch.no_grad(),threadpool_limits(limits=2):
        checkpoint=ROOT/"experiments/runs/ccv7_semantic_s17/gt_native/best.pt"
        if sha256(checkpoint)!="62c71edde258893a77af8a6fc60c8ffd6f0b70169cc44ed0d527bd0aa42379eb":
            raise ValueError("Wrong frozen training comparator")
        state=torch.load(checkpoint,map_location="cpu",weights_only=True)
        model=CompactResidualCC("direct","large").eval()
        model.load_state_dict({k:v for k,v in state.items() if not k.startswith("teacher_projection.")},strict=True)
        pred=torch.cat([model(x)[0] for x in images.split(32)]).numpy().astype(np.float64)
        reference={"checkpoint_sha256":sha256(checkpoint),"recovery":summarize(angular(pred,gt)),"reproduction":summarize(reproduction(pred,gt)),"scope":"TRAIN errors of a previously fitted comparator; not validation/test"}
        np.savez_compressed(out/"training_comparator.npz",pred=pred,gt=gt,ids=np.array([rows[i]["id"] for i in ix]))
        matrices={"gain":torch.diag(torch.tensor([2.,1.,.5],dtype=torch.float64)),"mix":torch.tensor([[.65,.25,.10],[.10,.70,.20],[.20,.10,.70]],dtype=torch.float64),"strong_mix":torch.tensor([[.4,.4,.2],[.2,.5,.3],[.3,.2,.5]],dtype=torch.float64)}
        for grid in (4,8,16):
            colors=F.adaptive_avg_pool2d(images,(grid,grid)).flatten(2).transpose(1,2).double()
            # Matrix multiplication is applied to the pooled FP64 colors. This
            # checks the representation's algebra, not end-to-end ISP fidelity.
            probes=torch.randn(grid*grid,16,generator=torch.Generator().manual_seed(17),dtype=torch.float64)/(grid*grid)**.5
            base=projector_features(colors,probes)
            invariance={}
            for name,matrix in matrices.items():
                transformed=projector_features(colors@matrix.T,probes)
                valid=base["valid"]&transformed["valid"]
                difference=(base["features"][valid]-transformed["features"][valid]).abs().amax((1,2))
                invariance[name]={"jointly_supported":int(valid.sum()),"support_changed":int((base["valid"]!=transformed["valid"]).sum()),"max_absolute_feature_difference":float(difference.max()) if len(difference) else None}
            predictions=np.zeros((len(colors),3))
            recovery=np.full(len(colors),np.nan)
            feasible=np.full(len(colors),np.nan)
            kkt=np.full(len(colors),np.nan)
            residual=np.full(len(colors),np.nan)
            failures=[]
            for i,x in enumerate(colors.numpy()):
                try:
                    value=cone_oracle(x,gt[i])
                    predictions[i]=value["pred"]
                    recovery[i]=value["recovery"]
                    feasible[i]=value["feasible_reproduction"] if value["feasible_reproduction"] is not None else np.nan
                    kkt[i]=value["kkt_violation"]
                    residual[i]=value["projection_residual"]
                except (ValueError,RuntimeError) as exc:
                    failures.append({"id":rows[ix[i]]["id"],"error":str(exc)})
            if failures:
                # Preserve failure rows and do not report an all-population bound.
                status="NUMERICAL FAILURES; conditional summaries only"
            else:
                status="ALL CONE PROJECTIONS KKT VERIFIED"
            condition=base["condition"][torch.isfinite(base["condition"])].numpy()
            condition_summary={"n":len(condition),"median":float(np.median(condition)),"p95":float(np.percentile(condition,95)),"max":float(condition.max())}
            result={"grid":grid,"status":status,"population":len(colors),"projector_supported":int(base["valid"].sum()),"condition_finite":condition_summary,"invariance":invariance,"oracle_minimum_recovery":summarize(recovery[np.isfinite(recovery)]),"oracle_feasible_reproduction_NOT_lower_bound":summarize(feasible[np.isfinite(feasible)]),"nonpositive_or_failed_reproduction_rows":int((~np.isfinite(feasible)).sum()),"outside_cone_fraction":float(np.mean(residual>1e-7)) if not failures else None,"max_kkt_violation":float(np.nanmax(kkt)),"failures":failures}
            np.savez_compressed(out/f"grid{grid}.npz",pred=predictions,gt=gt,recovery=recovery,feasible_reproduction=feasible,kkt=kkt,residual=residual,condition=base["condition"].numpy(),supported=base["valid"].numpy(),ids=np.array([rows[i]["id"] for i in ix]))
            records.append(result)
            print(json.dumps({"grid":grid,"minimum_recovery_mean":result["oracle_minimum_recovery"]["mean"],"feasible_reproduction_mean":result["oracle_feasible_reproduction_NOT_lower_bound"]["mean"],"outside_fraction":result["outside_cone_fraction"],"failures":len(failures)}),flush=True)
    write_json(out/"results.json",{"scope":"TRAIN-ONLY REPRESENTATION DIAGNOSTIC; oracles use GT; no deployable accuracy result","comparator":reference,"records":records,"seconds":time.perf_counter()-started})
    write_json(out/"manifest.json",{"sha256":{p.name:sha256(p) for p in out.iterdir() if p.is_file()}})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True)
    run(parser.parse_args().out)
