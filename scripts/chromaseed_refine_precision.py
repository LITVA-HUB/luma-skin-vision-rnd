"""Storage-only encodings; inference remains decoded float32."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chromaseed_refine_audit import error_summary, read_npz
from chromaseed_refine_numpy import NumpyRefiner
from chromaseed_refine_train import atomic_npz
from skin_local_search_train import CACHE_HASH, roles, sha, write_json

from luma_skin_vision.color import delta_e00


def encode(payload, precision):
    result = {k: np.array(v, copy=True) for k, v in payload.items() if k != "theta"}
    result["precision"] = np.asarray(precision)
    if precision == "fp16_storage":
        result["theta"] = payload["theta"].astype(np.float16)
    elif precision == "int8_channels":
        model = NumpyRefiner(payload)
        result["layer_order"] = np.asarray(list(model.layers))
        for name, (weight, bias) in model.layers.items():
            magnitude = np.max(np.abs(weight), axis=0)
            scale = np.where(magnitude == 0, 1., magnitude / 127.).astype(np.float32)
            result[f"{name}_q"] = np.clip(np.rint(weight / scale), -127, 127).astype(np.int8)
            result[f"{name}_scale"] = scale
            if bias is not None:
                result[f"{name}_bias"] = bias.copy()
    else:
        raise ValueError("unknown storage precision")
    return result


def decode(encoded):
    precision = str(encoded["precision"])
    keys = ("family", "x_mean", "x_std", "t_mean", "t_std", "y_mean", "y_std", "anchor", "exit_threshold")
    payload = {key: encoded[key] for key in keys if key in encoded}
    if precision == "fp16_storage":
        payload["theta"] = encoded["theta"].astype(np.float32)
    elif precision == "int8_channels":
        blocks = []
        for name in encoded["layer_order"]:
            blocks.append((encoded[f"{name}_q"].astype(np.float32) * encoded[f"{name}_scale"]).ravel())
            if f"{name}_bias" in encoded:
                blocks.append(encoded[f"{name}_bias"])
        payload["theta"] = np.concatenate(blocks).astype(np.float32)
    else:
        raise ValueError("unknown storage precision")
    return payload


def evaluate(source_run, cache, run, output):
    if cache.name != "train.npz" or sha(cache) != CACHE_HASH:
        raise ValueError("only original TRAIN allowed")
    files = ("scripts/chromaseed_refine_precision.py", "scripts/chromaseed_refine_numpy.py", "scripts/chromaseed_refine_audit.py", "docs/research/chromaseed_refine_precision_protocol.md")
    lock = {"sources": {p: sha(ROOT / p) for p in files}, "primary_source_lock_sha256": sha(source_run / "source_lock.json"),
            "primary_results_sha256": sha(source_run / "results.json"), "cache_sha256": CACHE_HASH,
            "precisions": ["fp16_storage", "int8_channels"], "arithmetic": "decode to float32 before inference",
            "evidence": "post-hoc storage evaluation on reused exploratory rows; no training or threshold retuning"}
    lock_path = run / "source_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != lock:
        raise ValueError("changed precision source lock")
    write_json(lock_path, lock)
    with np.load(cache, allow_pickle=False) as z:
        data = {key: z[key] for key in ("color", "tokens", "target", "patient", "site", "device")}
    records = []
    for role, (_, held) in roles(data["patient"], data["device"]).items():
        idx = np.flatnonzero(held)
        for path in sorted((source_run / "final" / role).glob("*/seed*.npz")):
            original = read_npz(path)
            base_model = NumpyRefiner(original)
            base_outputs, base_steps, base_counts = [], [], []
            for row in idx:
                prediction, steps, count = base_model.predict(data["color"][row], data["tokens"][row])
                base_outputs.append(prediction)
                base_steps.append(steps)
                base_counts.append(count)
            base_outputs = np.stack(base_outputs)
            base_metrics, _ = error_summary(base_outputs, data["target"][idx], data["patient"][idx], data["site"][idx])
            for precision in lock["precisions"]:
                encoded = encode(original, precision)
                target_path = run / role / path.parent.name / f"{path.stem}_{precision}.npz"
                atomic_npz(target_path, **encoded)
                model = NumpyRefiner(decode(read_npz(target_path)))
                outputs, step_changes, connection_changes = [], [], []
                for k, row in enumerate(idx):
                    prediction, steps, count = model.predict(data["color"][row], data["tokens"][row])
                    outputs.append(prediction)
                    step_changes.append(steps != base_steps[k])
                    shared = min(len(count), len(base_counts[k]))
                    connection_changes.append(bool(np.any(count[:shared] != base_counts[k][:shared])))
                outputs = np.stack(outputs)
                evaluated, _ = error_summary(outputs, data["target"][idx], data["patient"][idx], data["site"][idx])
                drift = delta_e00(outputs, base_outputs)
                record = {"role": role, "family": path.parent.name, "seed": int(path.stem.removeprefix("seed")),
                          "precision": precision, "numeric_bytes": sum(v.nbytes for v in encoded.values() if np.issubdtype(v.dtype, np.number)),
                          "archive_bytes": target_path.stat().st_size, "sha256": sha(target_path), "source_weight_sha256": sha(path),
                          "metrics": evaluated, "person_error_change": evaluated["person_mean"] - base_metrics["person_mean"],
                          "max_prediction_delta_e00": float(drift.max()), "p95_prediction_delta_e00": float(np.quantile(drift, .95)),
                          "mean_prediction_delta_e00": float(drift.mean()), "changed_exit_fraction": float(np.mean(step_changes)),
                          "changed_connection_count_fraction": float(np.mean(connection_changes))}
                record["engineering_guard_passed"] = record["max_prediction_delta_e00"] <= .1 and record["person_error_change"] <= .02
                records.append(record)
        print(f"PRECISION EVALUATED {role}", flush=True)
    result = {"source_lock_sha256": sha(lock_path), "evidence": lock["evidence"], "records": records,
              "arithmetic": lock["arithmetic"], "guard": "per-model max prediction DeltaE00 <=0.1 and person error increase <=0.02; not a validated product tolerance"}
    write_json(run / "results.json", result)
    write_json(output / "precision.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.source_run, args.cache, args.run, args.output)


if __name__ == "__main__":
    main()
