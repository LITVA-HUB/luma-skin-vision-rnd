"""Independent scalar XYZ audit and bounded real-skin calibration report."""
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
from cc_v7_external_audit import percentile
from skin_he_xyz import BENCH, DATA, MODELS, ROOT, inputs, read, sha, write

output = BENCH / "evaluation"
receipt_path = BENCH / "independent_audit.json"
if receipt_path.exists():
    raise ValueError("Immutable audit/report exists")
lock = read(BENCH / "model_lock.json")
for name, expected in lock["files"].items():
    assert sha(ROOT / name) == expected
manifest = read(output / "manifest.json")
for name, expected in manifest["sha256"].items():
    assert sha(output / name) == expected
test = read(DATA / "test.json")["rows"]
train = read(DATA / "train.json")["rows"]
assert not {r["subject"] for r in test} & {r["subject"] for r in train}
with np.load(output / "references.npz") as refs:
    gt = refs["gt"].copy()
    assert refs["ids"].tolist() == [r["id"] for r in test]
    np.testing.assert_array_equal(gt, np.array([r["xyz"] for r in test]))
results = read(output / "results.json")["records"]
checks, differences, replay_differences = 0, [], []


def scalar_stats(pred, target):
    difference = [[float(a)-float(b) for a,b in zip(p,g)] for p,g in zip(pred,target)]
    norms = [math.sqrt(math.fsum(v*v for v in row)) for row in difference]
    return {"xyz_rmse":math.sqrt(math.fsum(v*v for row in difference for v in row)/(3*len(difference))),
            "channel_rmse":[math.sqrt(math.fsum(row[c]**2 for row in difference)/len(difference)) for c in range(3)],
            "channel_mae":[math.fsum(abs(row[c]) for row in difference)/len(difference) for c in range(3)],
            "median_euclidean_xyz":percentile(norms,.5),"p95_euclidean_xyz":percentile(norms,.95)}


for record in results:
    fmt, method = record["format"], record["method"]
    with np.load(output / (fmt+"__"+method+".npz")) as values:
        pred, scores = values["pred"].copy(), values["scores"].copy()
        assert values["ids"].tolist() == [r["id"] for r in test]
    x = np.array([r[fmt] for r in test])
    replay = joblib.load(MODELS/(fmt+"__"+method+".joblib")).predict(inputs(x,method))
    replay_differences.append(float(abs(replay-pred).max()))
    training = np.array([r[fmt] for r in train])
    scales = training.std(axis=0)
    independent_scores = []
    for row in x:
        ds = sorted(math.fsum(((float(row[c])-float(t[c]))/float(scales[c]))**2 for c in range(3))/3 for t in training)
        independent_scores.append(math.sqrt(math.fsum(ds[:5])/5))
    differences.extend(abs(a-b) for a,b in zip(independent_scores,scores))
    order = sorted(range(100),key=lambda i:(scores[i],hashlib.sha256(test[i]["id"].encode()).hexdigest()))
    cases = [(list(range(100)),record["full"])]
    cases += [(order[:c],record["fixed"][str(c)]) for c in (100,95,90,80,70,60)]
    cases += [([i for i,r in enumerate(test) if r["site"]==site],stats) for site,stats in record["sites"].items()]
    for ix, observed in cases:
        assert observed["n"] == len(ix)
        expected = scalar_stats(pred[ix],gt[ix])
        for key,value in expected.items():
            differences.append(float(np.max(abs(np.asarray(value)-observed[key]))))
        checks += 1
assert max(differences) < 1e-10 and max(replay_differences) < 1e-10
write(receipt_path,{"status":"PASS: all14 model replays, independent scalar XYZ metrics and support scores", "metric_cases":checks,"max_scalar_difference":max(differences),"max_prediction_replay_difference":max(replay_differences),"model_lock_sha256":sha(BENCH/"model_lock.json"),"result_sha256":sha(output/"results.json"),"script_sha256":sha(Path(__file__))})
source = read(BENCH / "source_fit.json")
lines = ["# Real paired skin-color calibration: He 2021 pilot", "",
         "Measured on 100 facial sites from20 held-out people. Training used200 sites",
         "from40 different people. Both reference XYZ and corresponding regional RAW/JPG",
         "RGB originate in the author's CC BY4.0 workbook. These are source-calibrated",
         "ordinary regressors, not the V7 image model or a proposed new architecture.", "",
         "All14 controls were frozen before numeric TEST extraction. A5-subject-group",
         "cross-validation selected RAW poly3 and JPG poly2 before the test. No test",
         "tuning, demographics, site identity, camera labels or full images were inputs.", "",
         "|Input|Control|Selected by source CV|Source OOF XYZ RMSE|Test XYZ RMSE|Test80% XYZ RMSE|",
         "|---|---|---|---:|---:|---:|"]
for r in results:
    original=next(s for s in source["records"] if s["format"]==r["format"] and s["method"]==r["method"])
    lines.append(f"|{r['format']}|{r['method']}|{r['selected_by_source_oof']}|{original['oof']['xyz_rmse']:.6f}|{r['full']['xyz_rmse']:.6f}|{r['fixed']['80']['xyz_rmse']:.6f}|")
lines += ["", "XYZ RMSE is the square root of the average squared discrepancy across all",
          "three original instrument-coordinate channels. It is neither DeltaE nor",
          "angular illumination error, nor a percentage accuracy. It has no established",
          "cosmetics acceptance interpretation here. The measured reference-white",
          "convention is not numerically verified, so DeltaE00/DeltaE76 are NOT CALCULATED.", ""]
for r in results:
    if r["selected_by_source_oof"]:
        ci = r["subject_cluster_bootstrap_rmse95"]
        lines.append(f"{r['format']} source-selected control: XYZ RMSE {r['full']['xyz_rmse']:.6f}, 95% subject-cluster bootstrap interval [{ci[0]:.6f}, {ci[1]:.6f}].")
        lines.append("Fixed100/95/90/80/70/60% XYZ RMSE: "+", ".join(f"{r['fixed'][str(c)]['xyz_rmse']:.6f}" for c in (100,95,90,80,70,60))+".")
        lines.append("")
lines += ["The shared5-nearest-training-RGB distance is an exploratory ranking, not a",
          "calibrated expected-error estimate. Lower coverage does not monotonically",
          "lower measured risk. Do not infer reliable production rejection from it.", "",
          "Five MLP convergence warnings across source fits are retained in source_fit.json.",
          "No rerun or optimizer retuning followed test exposure. The MLP uses sklearn",
          "L-BFGS and is not a reproduction of MATLAB Bayesian-regularized trainbr.", "",
          "Limits: one controlled Canon6D MarkII capture setup with polarization, no",
          "unseen phone/lighting test, no full-image normalization or region extraction",
          "evaluation, no repeated capture noise floor and no validated cosmetic decision.",
          "Only skin regional RGB-to-measured-XYZ calibration was tested. Skin-accuracy",
          "claims for the Luma image pipeline and its V7 weights remain unmeasured.", "",
          "[Original author data](https://zenodo.org/records/5532176),",
          "[paper](https://doi.org/10.1002/col.22737),",
          "[frozen protocol](../../research/skin_he_xyz_protocol_v1.md),",
          "[all metrics](evaluation/results.json), [independent audit](independent_audit.json)."]
(BENCH / "report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps({"cases":checks,"max_difference":max(differences),"max_replay_difference":max(replay_differences)}))
