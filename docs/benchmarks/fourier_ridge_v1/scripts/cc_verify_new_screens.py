"""Independent saved-weight replay for V6 and the Fourier representation spike."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from cc_fourier_ridge import predict_score
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES, source_rows
from cc_v3_ffcc import decode, featurize
from cc_v6_evaluate import evaluate
from cc_v6_model import CanonicalEvidenceNet
from threadpoolctl import threadpool_limits

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT=Path(__file__).resolve().parents[1]


def independently_error(pred,gt):
    ratio=gt.astype(np.float64)/pred.astype(np.float64)
    cosine=ratio.sum(-1)/(np.sqrt(3)*np.linalg.norm(ratio,axis=-1))
    return np.degrees(np.arccos(np.clip(cosine,-1,1)))


def verify_manifest(run,name):
    receipt=json.loads((run/name).read_text())
    for file,digest in receipt["sha256"].items():
        if sha256(run/file)!=digest:
            raise ValueError("Artifact hash mismatch: "+file)


def run(seed,fourier):
    torch.set_num_threads(2)
    data=ROOT/"data/processed/cc128"
    assert {k:sha256(data/k) for k in DATA_HASHES}==DATA_HASHES
    rows=json.loads((data/"cube_manifest.json").read_text())
    selected,_,val=source_rows(rows)
    ix=selected[val]
    x=read_npz_rows(data/"cube.npz","images",ix,2234).astype(np.float32)
    gt=read_npz_rows(data/"cube.npz","gt",ix,2234).astype(np.float64)
    receipt=[]
    with torch.no_grad(),threadpool_limits(limits=2):
        if seed:
            folder=ROOT/f"experiments/runs/ccv6_sog_s{seed}"
            verify_manifest(folder,"artifact_manifest.json")
            config=json.loads((folder/"config.json").read_text())
            for name,digest in config["scripts_sha256"].items():
                assert sha256(ROOT/"scripts"/name)==digest
            for arm in config["arms"]:
                for checkpoint in ("best","last"):
                    model=CanonicalEvidenceNet(mode="posterior" if arm=="point" else arm.split("_")[0],frame="sog").eval()
                    path=folder/arm/(checkpoint+".pt")
                    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
                    _,actual=evaluate(model,torch.from_numpy(x),torch.from_numpy(gt.astype(np.float32)),16)
                    with np.load(folder/arm/(checkpoint+"_validation.npz")) as stored:
                        delta=max(float(abs(actual[k]-stored[k]).max()) for k in ("pred","base_pred","risk"))
                        assert delta<1e-3
                        pred=stored["base_pred" if arm=="point" else "pred"]
                        error=independently_error(pred,gt.astype(np.float32))
                        source_precision_delta=float(abs(error-independently_error(pred,gt)).max())
                        expected=stored["base_reproduction" if arm=="point" else "reproduction"]
                        error_delta=float(abs(error-expected).max())
                        assert error_delta<1e-7
                    receipt.append({"arm":arm,"checkpoint":checkpoint,"max_replay_abs_difference":delta,"max_error_rescore_difference":error_delta,"source_gt_float32_rounding_degrees":source_precision_delta,"checkpoint_sha256":sha256(path)})
            destination=ROOT/f"docs/benchmarks/cc_v6/seed{seed}_verification.json"
        else:
            folder=ROOT/"experiments/runs/fourier_ridge_v1"
            verify_manifest(folder,"manifest.json")
            batch=torch.from_numpy(x).double()
            features=featurize(batch/batch.amax((1,2,3),keepdim=True))
            results=json.loads((folder/"results.json").read_text())
            for candidate in results["candidates"]:
                with np.load(folder/(candidate["id"]+".npz")) as stored:
                    scores=predict_score(features["hist"].numpy(),stored["weight"])
                    pmf=torch.from_numpy(scores*200).flatten(1).softmax(-1).reshape(-1,64,64)
                    actual=decode(pmf,average_rgb=features["average_rgb"],unwrap_mode=candidate["unwrap"])
                    delta=float(abs(actual["pred"].numpy()-stored["pred"]).max())
                    assert delta<1e-10
                    error=independently_error(stored["pred"],gt)
                    error_delta=abs(float(error.mean())-candidate["reproduction"]["mean"])
                    assert error_delta<1e-8
                    np.testing.assert_array_equal(gt,stored["gt"])
                    np.testing.assert_array_equal(stored["ids"],np.array([rows[i]["id"] for i in ix]))
                receipt.append({"candidate":candidate["id"],"max_prediction_difference":delta,"mean_error_difference":error_delta})
            destination=ROOT/"docs/benchmarks/fourier_ridge_v1/verification.json"
    if destination.exists():
        raise ValueError("Existing verification receipt")
    destination.parent.mkdir(parents=True,exist_ok=True)
    write_json(destination,{"status":"ALL HASHES AND REPLAYS PASSED","records":receipt,"script_sha256":sha256(Path(__file__))})
    print(json.dumps({"verified":len(receipt),"output":str(destination)}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed",type=int)
    g.add_argument("--fourier",action="store_true")
    args=p.parse_args()
    run(args.seed,args.fourier)
