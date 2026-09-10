"""Frozen v1 comparators and source-only confidence-head upgrade, no retraining.

The original checkpoints/results remain untouched. Only source risk/cal labels
fit new heads; fresh camera labels are opened solely by the evaluate action.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from cc_v2_select import estimator, raw_predict, selective_result
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from luma_skin_vision.cc.benchmark import apply_risk, data_hashes, indices, predict, risk_features
from luma_skin_vision.cc.core import EXPERT_NAMES, reproduction, selective_curve
from luma_skin_vision.cc.model import CompactCC
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import source_identity, write_json


def verify_legacy(run, data):
    evidence = json.loads(
        subprocess.check_output(
            ["git", "show", "7637d6d:docs/benchmarks/public_evidence_manifest.json"]
        )
    )
    registered = [item for item in evidence if item["run"] == run.name]
    if len(registered) != 1 or sha256(run / "model.pt") != registered[0]["model_sha256"]:
        raise ValueError("Unregistered or modified frozen legacy checkpoint")
    for name in ("config.json", "risk_heads.json"):
        original = subprocess.check_output(
            ["git", "show", f"7637d6d:docs/benchmarks/public_runs/{run.name}/{name}"]
        )
        # Git stores LF; the original Windows experiment writer stored CRLF.
        if (run / name).read_bytes().replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n"):
            raise ValueError("Frozen legacy config/selector changed")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if config["data_hashes"] != data_hashes(data):
        raise ValueError("Original source data changed")
    for name in ("benchmark", "core", "model", "data"):
        path = f"src/luma_skin_vision/cc/{name}.py"
        original = subprocess.check_output(["git", "show", "7637d6d:" + path])
        if original != Path(path).read_bytes():
            raise ValueError("Legacy inference source changed: " + name)
    return config


def model_load(run, device="cuda"):
    saved = torch.load(run / "model.pt", weights_only=True, map_location=device)
    model = CompactCC(saved["mixture"]).to(device).eval()
    model.load_state_dict(saved["state"])
    return model


def predictions(model, cache):
    return predict(
        model,
        torch.tensor(cache["images"].astype(np.float32), device="cuda"),
        torch.tensor(cache["experts"], dtype=torch.float32, device="cuda"),
    )


def fit(run, data, out):
    config = verify_legacy(run, data)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    rows = json.loads((data / "cube_manifest.json").read_text(encoding="utf-8"))
    ix = indices(rows, config["protocol"])
    selected = np.r_[ix["risk"], ix["cal"]]
    cache = np.load(data / "cube.npz", allow_pickle=False)
    # Only selected rows enter neural prediction or any target computation.
    x = cache["images"][selected].astype(np.float32)
    ex = cache["experts"][selected]
    gt = cache["gt"][selected]
    model = model_load(run, device="cpu")
    pred, context = predict(model, torch.tensor(x), torch.tensor(ex, dtype=torch.float32))
    errors, n = reproduction(pred, gt), len(ix["risk"])
    groups = np.array([rows[i]["group"] for i in ix["risk"]])
    ids = [rows[i]["id"] for i in ix["risk"]]
    folds = list(GroupKFold(5).split(np.arange(n), groups=groups))
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_run": str(run.resolve()),
        "checkpoint_sha256": sha256(run / "model.pt"),
        "legacy_config_sha256": sha256(run / "config.json"),
        "legacy_risk_heads_sha256": sha256(run / "risk_heads.json"),
        "source_hashes": config["data_hashes"],
        "fit_ids": ids,
        "cal_ids": [rows[i]["id"] for i in ix["cal"]],
        "selection": "five risk-date-group OOF folds; minimumrisk80 thenAURC; standard positive multiplicative calibration on separatecal",
        "scope": "source risk/cal labels ONLY; no external or regression target errors; no estimator retraining",
        "source_feature_device": "CPU FP32; avoid concurrent GPU load during estimator training",
        "heads": {},
        "script_sha256": sha256(Path(__file__)),
        "helper_sha256": sha256(Path(__file__).with_name("cc_v2_select.py")),
    }
    payloads = {}
    for block in ("context", "combined"):
        x = risk_features(context, pred, ex, block)
        candidates = []
        for name in ("ridge1", "ridge10", "ridge100", "hgb3", "hgb7"):
            oof = np.empty(n)
            for tr, va in folds:
                head = estimator(name).fit(x[tr], np.log1p(errors[tr]))
                oof[va] = raw_predict(head, x[va])
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
        head = estimator(best["name"]).fit(x[:n], np.log1p(errors[:n]))
        cal_raw = raw_predict(head, x[n:])
        scale = max(1e-8, float(errors[n:].mean() / cal_raw.mean()))
        payloads[block] = {"model": head, "scale": scale}
        metadata["heads"][block] = {
            "selected": best["name"],
            "candidates": candidates,
            "cal_scores": (cal_raw * scale).tolist(),
            "scale": scale,
        }
    joblib.dump(payloads, out / "heads.joblib")
    metadata["artifact_sha256"] = sha256(out / "heads.joblib")
    write_json(out / "selection.json", metadata)
    print(
        json.dumps(
            {"legacy": run.name, "heads": {b: h["selected"] for b, h in metadata["heads"].items()}}
        ),
        flush=True,
    )


def evaluate(run, data, out, external, destination):
    config = verify_legacy(run, data)
    meta = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    if meta["script_sha256"] != sha256(Path(__file__)) or meta["helper_sha256"] != sha256(
        Path(__file__).with_name("cc_v2_select.py")
    ):
        raise ValueError("Legacy evaluation source differs from selector fitting source")
    for key, path in (
        ("checkpoint_sha256", run / "model.pt"),
        ("legacy_config_sha256", run / "config.json"),
        ("legacy_risk_heads_sha256", run / "risk_heads.json"),
        ("artifact_sha256", out / "heads.joblib"),
    ):
        if meta[key] != sha256(path):
            raise ValueError("Legacy bound artifact changed")
    if destination.exists():
        raise FileExistsError(destination)
    model = model_load(run)
    payloads = joblib.load(out / "heads.joblib")
    original_heads = {
        k: {
            key: np.asarray(value) if isinstance(value, list) else value
            for key, value in state.items()
        }
        for k, state in json.loads((run / "risk_heads.json").read_text()).items()
    }
    cube = np.load(data / "cube.npz", allow_pickle=False)
    rows = json.loads((data / "cube_manifest.json").read_text(encoding="utf-8"))
    ix = indices(rows, config["protocol"])
    cp, cf = predictions(model, cube)
    cex = cube["experts"]
    original_cal = {}
    for name, state in original_heads.items():
        if name in EXPERT_NAMES:
            p, mode = cex[:, EXPERT_NAMES.index(name)], "combined"
        else:
            p, mode = cp, name.rsplit("_", 1)[-1]
        original_cal[name] = apply_risk(risk_features(cf, p, cex, mode), state)[ix["cal"]]
    domains = [("source_regression", cube, rows, cp, cf, ix["test"])]
    input_hashes = {}
    for domain, cache_path, manifest_path in [
        ("sony30_regression", data / "sony.npz", data / "sony_manifest.json"),
        ("fresh_all", Path(external[0]), Path(external[1])),
    ]:
        cache = np.load(cache_path, allow_pickle=False)
        rr = json.loads(manifest_path.read_text(encoding="utf-8"))
        p, f = predictions(model, cache)
        domains.append((domain, cache, rr, p, f, np.arange(len(rr))))
        input_hashes[domain] = {"cache": sha256(cache_path), "manifest": sha256(manifest_path)}
    evaluations = {}
    for domain, cache, rows, pred, context, subset in domains:
        ex, gt = cache["experts"], cache["gt"]
        valid = (cache["images"].mean(axis=(2, 3), dtype=np.float32) > 1e-8).all(axis=1)
        methods = {}
        for name, state in original_heads.items():
            p = ex[:, EXPERT_NAMES.index(name)] if name in EXPERT_NAMES else pred
            mode = "combined" if name in EXPERT_NAMES else name.rsplit("_", 1)[-1]
            score = apply_risk(risk_features(context, p, ex, mode), state)
            methods["v1_" + name] = (p, score, original_cal[name])
        for block, payload in payloads.items():
            score = (
                raw_predict(payload["model"], risk_features(context, pred, ex, block))
                * payload["scale"]
            )
            methods["upgraded_" + block] = (
                pred,
                score,
                np.asarray(meta["heads"][block]["cal_scores"]),
            )
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
        for name, idx in slices.items():
            for method, (p, score, cal) in methods.items():
                record = selective_result(
                    p[idx],
                    gt[idx],
                    score[idx],
                    [rows[i]["id"] for i in idx],
                    cal,
                    valid[idx],
                )
                record["groups"] = [rows[i]["group"] for i in idx]
                record["cameras"] = [rows[i]["camera"] for i in idx]
                evaluations.setdefault(name, {})[method] = record
    write_json(
        destination,
        {
            "legacy_run": str(run),
            "source_identity": source_identity(),
            "script_sha256": sha256(Path(__file__)),
            "selector_sha256": sha256(out / "selection.json"),
            "input_hashes": input_hashes,
            "domains": evaluations,
        },
    )
    print(
        json.dumps(
            {
                "legacy": run.name,
                "domains": {
                    d: {
                        m: {
                            "mean": v["reproduction"]["mean"],
                            "risk80": v["selective"]["fixed"]["80"]["mean"],
                        }
                        for m, v in records.items()
                    }
                    for d, records in evaluations.items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fit", "evaluate"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/cc128"))
    parser.add_argument("--external", nargs=2)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        if args.action == "fit":
            fit(args.run, args.data, args.out)
        else:
            if not args.external or not args.destination:
                parser.error("Evaluation requires external and destination")
            evaluate(args.run, args.data, args.out, args.external, args.destination)
