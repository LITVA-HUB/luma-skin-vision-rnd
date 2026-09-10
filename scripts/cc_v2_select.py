"""Fit source-only risk selectors, then evaluate frozen predictions independently.

No target labels enter fit(): even source GT arrays are sliced to risk/cal first.
All candidates receive identical date-group CV; physical DeltaE is unavailable.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from luma_skin_vision.cc.benchmark import result
from luma_skin_vision.cc.core import reproduction, selective_curve, summarize
from luma_skin_vision.cc.v2_experiment import split_indices, verify_run
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json


def estimator(name):
    if name.startswith("ridge"):
        return make_pipeline(StandardScaler(), Ridge(alpha=float(name[5:])))
    return HistGradientBoostingRegressor(
        max_leaf_nodes=int(name[3:]),
        max_iter=100,
        min_samples_leaf=20,
        l2_regularization=10,
        learning_rate=0.05,
        early_stopping=False,
        random_state=17,
    )


def features(values, block):
    if block == "context":
        return values["context"].astype(np.float64)
    if block == "cheap":
        return values["cheap_features"].astype(np.float64)
    return np.concatenate([values["context"], values["cheap_features"]], axis=1).astype(np.float64)


def raw_predict(model, x):
    return np.expm1(np.clip(model.predict(x), 0, np.log(181))) + 1e-8


def verify_predictions(run, data):
    config, checkpoint = verify_run(run, data)
    path = run / "predictions_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if any(manifest.get(key) != checkpoint[key] for key in ("checkpoint_sha256", "config_sha256")):
        raise ValueError("Prediction checkpoint/config binding mismatch")
    if (
        manifest.get("training_source_hash") != config["source_hash"]
        or manifest.get("prediction_source_identity", {}).get("source_hash")
        != config["source_hash"]
    ):
        raise ValueError("Prediction source binding mismatch")
    for name, record in manifest["outputs"].items():
        if name not in ("predictions.npz", "sony_predictions.npz", "external_predictions.npz"):
            raise ValueError("Unexpected prediction output")
        if sha256(run / name) != record["sha256"]:
            raise ValueError("Prediction output hash mismatch")
        for key, field in (("npz", "input_npz"), ("manifest", "input_manifest")):
            if sha256(Path(record[field])) != record["input_hashes"][key]:
                raise ValueError("Prediction input hash mismatch")
    if "predictions.npz" not in manifest["outputs"]:
        raise ValueError("Missing source predictions")
    return manifest


def selective_result(pred, gt, scores, ids, calibration_scores, valid):
    record = result(pred, gt, scores, ids, calibration_scores)
    valid = np.asarray(valid, dtype=bool)
    record["valid"] = valid.tolist()
    record["invalid_n"] = int((~valid).sum())
    record["max_supported_coverage"] = float(valid.mean())
    if not valid.all():
        # Keep full-population angular metrics as explicit fallback diagnostics.
        errors = np.asarray(record["errors"])
        for item in record["frozen_source_thresholds"].values():
            accepted = valid & (True if item["threshold"] is None else scores <= item["threshold"])
            item.update(
                n=int(accepted.sum()),
                coverage=float(accepted.mean()),
                mean_reproduction=float(errors[accepted].mean()) if accepted.any() else None,
            )
        if valid.any():
            eligible = selective_curve(
                errors[valid], scores[valid], np.asarray(ids)[valid].tolist()
            )
            eligible["coverage"] = (np.asarray(eligible["coverage"]) * valid.mean()).tolist()
            order = np.flatnonzero(valid)[eligible["order"]]
            eligible["order"] = order.tolist()
            for percent in (100, 95, 90, 80, 70, 60):
                n = max(1, int(np.floor(len(valid) * percent / 100)))
                eligible["fixed"][str(percent)] = (
                    {**summarize(errors[order[:n]]), "coverage": n / len(valid), "attainable": True}
                    if n <= len(order)
                    else {
                        "mean": None,
                        "n": len(order),
                        "coverage": float(valid.mean()),
                        "attainable": False,
                    }
                )
            eligible["aurc_scope"] = "mean risk over attainable valid coverages only"
            record["selective"] = eligible
        else:
            record["selective"] = {
                "fixed": {
                    str(p): {"mean": None, "n": 0, "coverage": 0, "attainable": False}
                    for p in (100, 95, 90, 80, 70, 60)
                },
                "coverage": [],
                "risk": [],
                "order": [],
                "aurc": None,
            }
    return record


def fit(run, data):
    config, checkpoint = verify_run(run, data)
    prediction_manifest = verify_predictions(run, data)
    source_input = prediction_manifest["outputs"]["predictions.npz"]
    if (
        Path(source_input["input_npz"]).resolve() != (data / "cube.npz").resolve()
        or Path(source_input["input_manifest"]).resolve() != (data / "cube_manifest.json").resolve()
    ):
        raise ValueError("Source prediction input differs from fitting dataset")
    out = run / "risk_v2"
    if out.exists():
        raise FileExistsError(out)
    rows = json.loads((data / "cube_manifest.json").read_text(encoding="utf-8"))
    ix = split_indices(rows, config["protocol"])
    values = np.load(run / "predictions.npz", allow_pickle=False)
    if values["ids"].tolist() != [r["id"] for r in rows]:
        raise ValueError("Prediction/manifest ID mismatch")
    selected = np.r_[ix["risk"], ix["cal"]]
    with np.load(data / "cube.npz", allow_pickle=False) as cache:
        gt = cache["gt"][selected]
    pred = values["pred"][selected]
    valid = values["valid"][selected]
    if not valid.all():
        raise ValueError("Unsupported source risk/cal rows require explicit protocol revision")
    errors = reproduction(pred, gt)
    n = len(ix["risk"])
    groups = np.array([rows[i]["group"] for i in ix["risk"]])
    ids = [rows[i]["id"] for i in ix["risk"]]
    folds = list(GroupKFold(n_splits=5).split(np.arange(n), groups=groups))
    out.mkdir()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "risk-fit and calibration source capture groups ONLY; no test or external errors",
        "selection": "lowest pooled 5-date-group-OOF risk at80%; AURC tie break",
        "target": "log1p reproduction angular error in degrees; no physical surface DeltaE",
        "calibration": "positive scalar mean-error/mean-predicted-error on separate cal; preserves ranking",
        "fit_ids": ids,
        "cal_ids": [rows[i]["id"] for i in ix["cal"]],
        "folds": [{"train": a.tolist(), "validation": b.tolist()} for a, b in folds],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "prediction_sha256": sha256(run / "predictions.npz"),
        "predictions_manifest_sha256": sha256(run / "predictions_manifest.json"),
        "script_sha256": sha256(Path(__file__)),
        "heads": {},
    }
    for block in ("context", "cheap", "combined"):
        x = features(values, block)[selected]
        candidates = []
        for name in ("ridge1", "ridge10", "ridge100", "hgb3", "hgb7"):
            oof = np.zeros(n)
            for tr, va in folds:
                model = estimator(name)
                model.fit(x[tr], np.log1p(errors[tr]))
                oof[va] = raw_predict(model, x[va])
            curve = selective_curve(errors[:n], oof, ids)
            candidates.append(
                {
                    "name": name,
                    "risk80": curve["fixed"]["80"]["mean"],
                    "aurc": curve["aurc"],
                    "oof_scores": oof.tolist(),
                }
            )
        best = min(candidates, key=lambda c: (c["risk80"], c["aurc"]))
        model = estimator(best["name"])
        model.fit(x[:n], np.log1p(errors[:n]))
        raw_cal = raw_predict(model, x[n:])
        scale = max(1e-8, float(errors[n:].mean() / raw_cal.mean()))
        payload = {"model": model, "scale": scale, "block": block}
        joblib.dump(payload, out / (block + ".joblib"))
        report["heads"][block] = {
            "selected": best["name"],
            "features": x.shape[1],
            "candidates": candidates,
            "scale": scale,
            "cal_scores": (raw_cal * scale).tolist(),
            "cal_errors": errors[n:].tolist(),
            "cal_mae": float(np.abs(raw_cal * scale - errors[n:]).mean()),
            "artifact_sha256": sha256(out / (block + ".joblib")),
        }
    write_json(out / "selection.json", report)
    print(
        json.dumps(
            {
                "run": run.name,
                "heads": {
                    k: {
                        "selected": v["selected"],
                        "risk80": next(
                            c["risk80"] for c in v["candidates"] if c["name"] == v["selected"]
                        ),
                    }
                    for k, v in report["heads"].items()
                },
            }
        ),
        flush=True,
    )


def evaluate(run, data, external, output):
    config, _ = verify_run(run, data)
    manifest = verify_predictions(run, data)
    state = json.loads((run / "risk_v2/selection.json").read_text(encoding="utf-8"))
    if sha256(run / "predictions_manifest.json") != state["predictions_manifest_sha256"]:
        raise ValueError("Prediction manifest changed since selector fitting")
    if sha256(run / "predictions.npz") != state["prediction_sha256"]:
        raise ValueError("Source predictions changed since selector fitting")
    if output.exists():
        raise FileExistsError(output)
    rows = json.loads((data / "cube_manifest.json").read_text(encoding="utf-8"))
    test = split_indices(rows, config["protocol"])["test"]
    domains = [
        (
            "source_regression",
            data / "cube.npz",
            data / "cube_manifest.json",
            run / "predictions.npz",
            test,
        )
    ]
    if (run / "sony_predictions.npz").exists():
        domains.append(
            (
                "sony30_regression",
                data / "sony.npz",
                data / "sony_manifest.json",
                run / "sony_predictions.npz",
                None,
            )
        )
    if external:
        domains.append(
            (
                "fresh_all",
                Path(external[0]),
                Path(external[1]),
                run / "external_predictions.npz",
                None,
            )
        )
    evaluations = {}
    for domain, npz_path, manifest_path, prediction_path, subset in domains:
        bound = manifest["outputs"].get(prediction_path.name)
        if (
            bound is None
            or Path(bound["input_npz"]).resolve() != npz_path.resolve()
            or Path(bound["input_manifest"]).resolve() != manifest_path.resolve()
        ):
            raise ValueError("Evaluation dataset differs from prediction binding")
        cache = np.load(npz_path, allow_pickle=False)
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        values = np.load(prediction_path, allow_pickle=False)
        if values["ids"].tolist() != [r["id"] for r in rows]:
            raise ValueError("Prediction/manifest IDs differ")
        if subset is None:
            subset = np.arange(len(rows))
        slices = {domain: subset}
        if domain == "fresh_all":
            slices.update(
                {
                    "fresh_" + camera: np.array(
                        [i for i, r in enumerate(rows) if r["camera"] == camera]
                    )
                    for camera in sorted({r["camera"] for r in rows})
                }
            )
        for block, head in state["heads"].items():
            path = run / "risk_v2" / (block + ".joblib")
            if sha256(path) != head["artifact_sha256"]:
                raise ValueError("Risk artifact changed")
            payload = joblib.load(path)  # Only locally produced, hash-bound artifacts.
            scores = raw_predict(payload["model"], features(values, block)) * payload["scale"]
            scores[~values["valid"]] = 1e6
            for name, idx in slices.items():
                record = selective_result(
                    values["pred"][idx],
                    cache["gt"][idx],
                    scores[idx],
                    [rows[i]["id"] for i in idx],
                    np.asarray(head["cal_scores"]),
                    values["valid"][idx],
                )
                record["mandatory_refusal_note"] = (
                    "Invalid inputs excluded from every acceptance mask and selective curve. Full-population angular metrics include finite fallback predictions only as diagnostics."
                )
                record["groups"] = [rows[i]["group"] for i in idx]
                record["cameras"] = [rows[i]["camera"] for i in idx]
                evaluations.setdefault(name, {})[block] = record
    write_json(
        output,
        {
            "run": str(run),
            "config": config,
            "selector_sha256": sha256(run / "risk_v2/selection.json"),
            "evaluation_source": sha256(Path(__file__)),
            "domains": evaluations,
        },
    )
    print(
        json.dumps(
            {
                "run": run.name,
                "domains": {
                    d: {
                        b: {
                            "mean": v["reproduction"]["mean"],
                            "risk80": v["selective"]["fixed"]["80"]["mean"],
                            "aurc": v["selective"]["aurc"],
                        }
                        for b, v in records.items()
                    }
                    for d, records in evaluations.items()
                },
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fit", "evaluate"])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/cc128"))
    parser.add_argument("--external", nargs=2)
    parser.add_argument("--output-root", type=Path, default=Path("docs/benchmarks/cc_v2/runs"))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        for value in args.runs:
            run = Path(value)
            if args.action == "fit":
                fit(run, args.data)
            else:
                args.output_root.mkdir(parents=True, exist_ok=True)
                evaluate(run, args.data, args.external, args.output_root / (run.name + ".json"))


if __name__ == "__main__":
    main()
