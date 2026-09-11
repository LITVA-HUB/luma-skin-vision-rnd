"""Locked source-only controls for real paired facial RGB/XYZ, not DeltaE."""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/skin_he_xyz_v1"
BENCH = ROOT / "docs/benchmarks/skin_he_xyz_v1"
MODELS = ROOT / "experiments/runs/skin_he_xyz_v1"
METHODS = ("constant", "linear3", "affine4", "poly2", "poly3", "root2", "mlp_5_25_5")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def inputs(x, name):
    if name == "root2":
        return np.c_[x, np.sqrt(x[:, 0]*x[:, 1]), np.sqrt(x[:, 0]*x[:, 2]), np.sqrt(x[:, 1]*x[:, 2])]
    return x


def model(name):
    if name == "constant":
        return DummyRegressor(strategy="mean")
    if name.startswith("poly"):
        return make_pipeline(PolynomialFeatures(int(name[-1]), include_bias=True), LinearRegression(fit_intercept=False))
    if name.startswith("mlp"):
        return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(5,25,5), activation="tanh", solver="lbfgs", alpha=1, max_iter=2000, random_state=17)), transformer=StandardScaler())
    return LinearRegression(fit_intercept=name == "affine4")


def metrics(pred, gt):
    difference = pred-gt
    distance = np.linalg.norm(difference, axis=1)
    return {"n": len(gt), "xyz_rmse": float(np.sqrt(np.mean(difference**2))),
            "channel_rmse": np.sqrt(np.mean(difference**2, axis=0)).tolist(),
            "channel_mae": np.mean(abs(difference), axis=0).tolist(),
            "median_euclidean_xyz": float(np.median(distance)),
            "p95_euclidean_xyz": float(np.percentile(distance, 95)),
            "nonpositive_prediction_n": int(np.any(pred <= 0, axis=1).sum())}


def fit():
    if MODELS.exists() or (BENCH/"model_lock.json").exists():
        raise ValueError("Immutable source fit exists")
    source = read(DATA/"train.json")
    assert source["role"] == "train"
    rows = source["rows"]
    assert len(rows) == 200 and len({r["subject"] for r in rows}) == 40
    gt = np.array([r["xyz"] for r in rows])
    groups = np.array([r["subject"] for r in rows])
    folds = list(GroupKFold(5).split(gt, groups=groups))
    for train, val in folds:
        assert not set(groups[train]) & set(groups[val])
    MODELS.mkdir(parents=True)
    BENCH.mkdir(parents=True, exist_ok=True)
    records, files = [], {}
    with threadpool_limits(limits=2):
        for format_name in ("raw", "jpg"):
            x = np.array([r[format_name] for r in rows])
            for name in METHODS:
                xx = inputs(x, name)
                predictions = np.zeros_like(gt)
                messages = []
                for train, val in folds:
                    estimator = model(name)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        estimator.fit(xx[train], gt[train])
                    messages.extend(str(w.message) for w in caught)
                    predictions[val] = estimator.predict(xx[val])
                fitted = model(name)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fitted.fit(xx, gt)
                messages.extend(str(w.message) for w in caught)
                key = format_name+"__"+name
                destination = MODELS/(key+".joblib")
                joblib.dump(fitted, destination)
                files[str(destination.relative_to(ROOT))] = sha(destination)
                np.savez_compressed(MODELS/(key+"_oof.npz"), pred=predictions, gt=gt, ids=np.array([r["id"] for r in rows]))
                record = {"format": format_name, "method": name, "oof": metrics(predictions, gt), "warnings": messages}
                records.append(record)
                print(json.dumps({"format": format_name, "method": name, "oof_xyz_rmse": record["oof"]["xyz_rmse"], "warnings": len(messages)}), flush=True)
    selected = {fmt: min((r for r in records if r["format"] == fmt), key=lambda r:r["oof"]["xyz_rmse"])["method"] for fmt in ("raw", "jpg")}
    write(BENCH/"source_fit.json", {"records": records, "selected_by_source_oof": selected, "folds": [{"train":t.tolist(), "validation":v.tolist()} for t,v in folds], "fit_people":40, "metric":"Original XYZ coordinate error, not perceptual DeltaE"})
    for relative in ("scripts/skin_he_xyz.py", "scripts/skin_he_data.py", "docs/research/skin_he_xyz_protocol_v1.md", "data/processed/skin_he_xyz_v1/train.json", "docs/benchmarks/skin_he_xyz_v1/source_fit.json"):
        files[relative] = sha(ROOT/relative)
    write(BENCH/"model_lock.json", {"status":"ALL14 CONTROLS FROZEN BEFORE SKIN TEST NUMERIC EXTRACTION", "files":files, "selected":selected, "methods":METHODS})
    print(json.dumps({"lock_sha256":sha(BENCH/"model_lock.json"), "selected":selected}), flush=True)


def evaluate(lock_digest):
    lock_path = BENCH/"model_lock.json"
    if sha(lock_path) != lock_digest:
        raise ValueError("Expected exact pre-test lock")
    lock = read(lock_path)
    for name, expected in lock["files"].items():
        if sha(ROOT/name) != expected:
            raise ValueError("Source/model binding mismatch")
    test = read(DATA/"test.json")
    train = read(DATA/"train.json")["rows"]
    assert test["role"] == "test" and test["model_lock_sha256"] == lock_digest
    rows = test["rows"]
    assert len(rows) == 100 and not {r["subject"] for r in rows} & {r["subject"] for r in train}
    output = BENCH/"evaluation"
    output.mkdir(exist_ok=False)
    gt = np.array([r["xyz"] for r in rows])
    ids = [r["id"] for r in rows]
    np.savez_compressed(output/"references.npz", gt=gt, ids=np.array(ids), subjects=np.array([r["subject"] for r in rows]))
    records = []
    for fmt in ("raw", "jpg"):
        x = np.array([r[fmt] for r in rows])
        training = np.array([r[fmt] for r in train])
        scale = training.std(axis=0)
        if np.any(scale <= 0):
            raise ValueError("Degenerate training input")
        squared = (((x[:,None,:]-training[None,:,:])/scale)**2).mean(axis=2)
        scores = np.sqrt(np.sort(squared, axis=1)[:,:5].mean(axis=1))
        order = np.array(sorted(range(len(rows)), key=lambda i:(scores[i],hashlib.sha256(ids[i].encode()).hexdigest())))
        for name in METHODS:
            estimator = joblib.load(MODELS/(fmt+"__"+name+".joblib"))
            pred = estimator.predict(inputs(x,name))
            if pred.shape != gt.shape or not np.isfinite(pred).all():
                raise ValueError("Invalid prediction")
            np.savez_compressed(output/(fmt+"__"+name+".npz"),pred=pred,scores=scores,ids=np.array(ids))
            record = {"format":fmt,"method":name,"selected_by_source_oof":lock["selected"][fmt]==name,"full":metrics(pred,gt),"fixed":{str(c):metrics(pred[order[:c]],gt[order[:c]]) for c in (100,95,90,80,70,60)},"sites":{site:metrics(pred[[r["site"]==site for r in rows]],gt[[r["site"]==site for r in rows]]) for site in sorted({r["site"] for r in rows})},"risk_score":"Fixed5-nearest-training-RGB distance; no expected-error calibration"}
            if record["selected_by_source_oof"]:
                subjects=sorted({r["subject"] for r in rows})
                members=[np.flatnonzero([r["subject"]==s for r in rows]) for s in subjects]
                rng=np.random.default_rng(20260911)
                boot=[metrics(pred[ix],gt[ix])["xyz_rmse"] for ix in (np.concatenate([members[g] for g in rng.integers(0,len(subjects),len(subjects))]) for _ in range(2000))]
                record["subject_cluster_bootstrap_rmse95"]=np.percentile(boot,[2.5,97.5]).tolist()
            records.append(record)
    write(output/"results.json", {"method_lock_sha256":lock_digest,"test_data_sha256":sha(DATA/"test.json"),"records":records,"delta_e00":"NOT CALCULATED: original reference white unverified","scope":"Controlled single-camera regional skin calibration, not an image-pipeline/camera-generalization test"})
    write(output/"manifest.json", {"sha256":{p.name:sha(p) for p in output.iterdir() if p.is_file()},"method_lock_sha256":lock_digest})
    print(json.dumps({"tested_methods":len(records),"selected_results":[r for r in records if r["selected_by_source_oof"]]}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("fit","evaluate"))
    parser.add_argument("--lock-digest")
    args=parser.parse_args()
    fit() if args.action == "fit" else evaluate(args.lock_digest)
