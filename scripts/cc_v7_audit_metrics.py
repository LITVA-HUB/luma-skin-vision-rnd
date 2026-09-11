"""Independent scalar/ranking audit of recorded V7 source and virtual diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from cc_v7_verify import score, verify_manifest

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT=Path(__file__).resolve().parents[1]


def run(seed):
    base=ROOT/"docs/benchmarks/cc_v7"
    out=base/f"seed{seed}_metric_audit.json"
    if out.exists():
        raise ValueError("Immutable audit exists")
    source_path=base/f"seed{seed}_source_risk/results.json"
    records=json.loads(source_path.read_text())["records"]
    differences=[]
    checks=0
    for record in records:
        selection=json.loads((ROOT/f"experiments/runs/ccv7_semantic_risk_s{seed}"/record["arm"]/"selection.json").read_text())
        for block,result in record["heads"].items():
            errors=score(np.asarray(result["pred"]),np.asarray(result["gt"]))
            differences.append(float(abs(errors-np.asarray(result["errors"])).max()))
            scores=np.array(result["scores"])
            ids=result["ids"]
            order=np.array(sorted(range(len(ids)),key=lambda i:(scores[i],hashlib.sha256(ids[i].encode()).hexdigest())))
            np.testing.assert_array_equal(order,result["selective"]["order"])
            curve=np.cumsum(errors[order])/np.arange(1,len(errors)+1)
            differences.append(float(abs(curve-result["selective"]["risk"]).max()))
            cal=np.array(selection["heads"][block]["cal_scores"])
            for c in (100,95,90,80,70,60):
                fixed=result["selective"]["fixed"][str(c)]
                n=int(len(errors)*c//100)
                differences.append(abs(float(errors[order[:n]].mean())-fixed["mean"]))
                accepted=np.ones(len(scores),dtype=bool) if c==100 else scores<=np.quantile(cal,c/100)
                frozen=result["frozen_source_thresholds"][str(c)]
                if frozen["n"]!=int(accepted.sum()) or frozen["coverage"]!=float(accepted.mean()):
                    raise ValueError("Source threshold coverage changed")
                if accepted.any():
                    differences.append(abs(float(errors[accepted].mean())-frozen["mean_reproduction"]))
                checks+=1
    folder=base/f"seed{seed}_stress"
    verify_manifest(folder,"manifest.json")
    stress=json.loads((folder/"results.json").read_text())
    for record in stress["records"]:
        with np.load(folder/(record["model"]+".npz")) as values:
            for name,metrics in record["metrics"].items():
                error=score(values[name+"_pred"],values[name+"_gt"])
                differences.append(abs(float(error.mean())-metrics["reproduction"]["mean"]))
                differences.append(abs(float(np.percentile(error,95))-metrics["reproduction"]["p95"]))
                if float(np.mean(error>10))!=metrics["reproduction"]["over10_fraction"]:
                    raise ValueError("Stress tail count differs")
                checks+=1
    maximum=max(differences)
    if maximum>1e-7:
        raise ValueError(f"Scalar/ranking metric mismatch {maximum}")
    write_json(out,{"status":"SOURCE FIXED COVERAGE, FULL CURVES, THRESHOLDS AND STRESS ERROR AUDIT PASSED","checks":checks,"max_error_difference_degrees":maximum,"source_risk_sha256":sha256(source_path),"stress_manifest_sha256":sha256(folder/"manifest.json"),"script_sha256":sha256(Path(__file__))})
    print(json.dumps({"checks":checks,"max_difference":maximum}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True)
    run(parser.parse_args().seed)
