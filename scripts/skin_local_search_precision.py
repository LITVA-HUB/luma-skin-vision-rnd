"""Freeze and evaluate storage-only precision variants of final local-search models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import skin_local_search_train as runner

ROOT = Path(__file__).resolve().parents[1]
CACHE_HASH = runner.CACHE_HASH
FORMATS = ("fp32_reference", "fp16", "int8")
PROTOCOLS = ("mixed", "slr_to_ipod", "ipod_to_slr")
METHOD_SEEDS = {
    "ridge": (17,),
    "krr": (17,),
    "random_rbf": (17, 29, 43),
    "guided_rbf": (17, 29, 43),
    "mlp": (17, 29, 43),
}
SCALER_KEYS = frozenset(("x_mean", "x_std", "y_mean", "y_std"))
MODEL_NUMERIC_KEYS = {
    "ridge": frozenset((*SCALER_KEYS, "beta")),
    "krr": frozenset((*SCALER_KEYS, "centers", "widths", "beta")),
    "random_rbf": frozenset((*SCALER_KEYS, "centers", "widths", "beta")),
    "guided_rbf": frozenset((*SCALER_KEYS, "centers", "widths", "beta")),
    "mlp": frozenset(
        (*SCALER_KEYS, "hidden_w", "hidden_b", "out_w", "out_b", "skip_w")
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _method(model):
    value = np.asarray(model.get("method"))
    if value.shape != ():
        raise ValueError("method must be a scalar string")
    method = str(value)
    if method not in MODEL_NUMERIC_KEYS:
        raise ValueError(f"unsupported method: {method}")
    return method


def _validate_source(model):
    method = _method(model)
    expected = MODEL_NUMERIC_KEYS[method] | {"method"}
    if set(model) != expected:
        raise ValueError(f"unexpected {method} model keys")
    for key in MODEL_NUMERIC_KEYS[method]:
        value = np.asarray(model[key])
        if value.dtype != np.float32:
            raise ValueError(f"source {key} must be float32")
        if not np.isfinite(value).all():
            raise ValueError(f"source {key} must be finite")
    if np.any(model["x_std"] <= 0) or np.any(model["y_std"] <= 0):
        raise ValueError("normalization standard deviations must be positive")
    if "widths" in model and np.any(model["widths"] <= 0):
        raise ValueError("RBF widths must be positive")
    return method


def _quantize_symmetric(value, rule):
    array = np.asarray(value, dtype=np.float32)
    if rule in ("columns", "features"):
        maximum = np.max(np.abs(array), axis=0)
        scale = np.where(maximum > 0, maximum / 127.0, 1.0).astype(np.float32)
        quantized = np.rint(array / scale[None, :])
    elif rule == "rows":
        maximum = np.max(np.abs(array), axis=1)
        scale = np.where(maximum > 0, maximum / 127.0, 1.0).astype(np.float32)
        quantized = np.rint(array / scale[:, None])
    elif rule == "single":
        maximum = float(np.max(np.abs(array)))
        scale = np.asarray(maximum / 127.0 if maximum > 0 else 1.0, dtype=np.float32)
        quantized = np.rint(array / scale)
    else:
        raise ValueError(f"unsupported quantization rule: {rule}")
    return np.clip(quantized, -127, 127).astype(np.int8), scale


def _int8_rule(method, key):
    if key == "beta":
        return "columns"
    if key == "centers":
        return "features"
    if method == "mlp" and key.endswith("_w"):
        return "rows"
    return "single"


def pack_model(model, storage_format):
    """Convert one canonical FP32 model to a fixed storage representation."""
    method = _validate_source(model)
    if storage_format not in FORMATS:
        raise ValueError(f"unsupported storage format: {storage_format}")
    payload = {
        "method": np.asarray(method),
        "storage_format": np.asarray(storage_format),
    }
    for key in MODEL_NUMERIC_KEYS[method]:
        value = model[key]
        if storage_format == "fp32_reference" or (
            storage_format == "int8" and key in SCALER_KEYS
        ):
            payload[key] = value.astype(np.float32)
        elif storage_format == "fp16":
            with np.errstate(over="ignore", under="ignore"):
                payload[key] = value.astype(np.float16)
            if not np.isfinite(payload[key]).all():
                raise ValueError(f"{key} is not representable in float16")
        else:
            payload[key], payload[f"{key}_scale"] = _quantize_symmetric(
                value, _int8_rule(method, key)
            )
    return payload


def dequantize_model(payload):
    """Materialize an FP32 inference model from one stored representation."""
    method = _method(payload)
    storage_format = str(np.asarray(payload.get("storage_format")))
    if storage_format not in FORMATS:
        raise ValueError(f"unsupported storage format: {storage_format}")
    model = {"method": np.asarray(method)}
    for key in MODEL_NUMERIC_KEYS[method]:
        if key not in payload:
            raise ValueError(f"missing payload array: {key}")
        value = np.asarray(payload[key])
        if storage_format != "int8" or key in SCALER_KEYS:
            model[key] = value.astype(np.float32)
            continue
        scale_key = f"{key}_scale"
        if scale_key not in payload:
            raise ValueError(f"missing quantization scale: {scale_key}")
        scale = np.asarray(payload[scale_key], dtype=np.float32)
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError(f"invalid quantization scale: {scale_key}")
        if _int8_rule(method, key) in ("columns", "features"):
            restored = value.astype(np.float32) * scale[None, :]
        elif _int8_rule(method, key) == "rows":
            restored = value.astype(np.float32) * scale[:, None]
        else:
            restored = value.astype(np.float32) * scale
        model[key] = restored.astype(np.float32)
    _validate_source(model)
    return model


def payload_stats(payload):
    numeric = [np.asarray(value) for value in payload.values() if np.asarray(value).dtype.kind in "fiu"]
    return {
        "numeric_scalars": int(sum(value.size for value in numeric)),
        "numeric_bytes": int(sum(value.nbytes for value in numeric)),
        "int8_scalars": int(sum(value.size for value in numeric if value.dtype == np.int8)),
        "float16_scalars": int(sum(value.size for value in numeric if value.dtype == np.float16)),
        "float32_scalars": int(sum(value.size for value in numeric if value.dtype == np.float32)),
        "quantization_scale_scalars": int(
            sum(np.asarray(value).size for key, value in payload.items() if key.endswith("_scale"))
        ),
    }


def _expected_sources(run):
    sources = []
    for protocol in PROTOCOLS:
        for method, seeds in METHOD_SEEDS.items():
            for seed in seeds:
                sources.append(run / protocol / "final" / f"{method}_s{seed}.npz")
    if len(sources) != 33:
        raise AssertionError("fixed primary-model inventory changed")
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise ValueError(f"missing final primary model: {missing[0]}")
    return sources


def _same_payload(left, right):
    return set(left) == set(right) and all(np.array_equal(left[key], right[key]) for key in left)


def _save_payload_once(path, payload):
    if path.exists():
        if not _same_payload(load_npz(path), payload):
            raise ValueError(f"existing precision artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _write_once(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"frozen output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _verify_manifest(run):
    folder = run / "precision"
    manifest_path = folder / "manifest.json"
    receipt_path = folder / "manifest.sha256"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise ValueError("precision manifest is not frozen")
    expected = receipt_path.read_text(encoding="ascii").strip()
    if sha256(manifest_path) != expected:
        raise ValueError("precision manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_model_count") != 33 or len(manifest.get("models", [])) != 33:
        raise ValueError("precision manifest must contain 33 primary models")
    for entry in manifest["models"]:
        source = run / entry["source_model"]
        if sha256(source) != entry["source_sha256"]:
            raise ValueError(f"source model hash mismatch: {source}")
        for storage_format in FORMATS:
            item = entry["formats"][storage_format]
            artifact = run / item["artifact"]
            if sha256(artifact) != item["sha256"]:
                raise ValueError(f"precision artifact hash mismatch: {artifact}")
    return manifest, expected


def prepare(run):
    """Freeze three storage formats without opening any data or evaluation arrays."""
    run = Path(run)
    manifest_path = run / "precision" / "manifest.json"
    if manifest_path.exists():
        return _verify_manifest(run)[0]
    records = []
    for source in _expected_sources(run):
        model = load_npz(source)
        method = _validate_source(model)
        protocol = source.parents[1].name
        formats = {}
        for storage_format in FORMATS:
            payload = pack_model(model, storage_format)
            artifact = (
                run / "precision" / protocol / source.stem / storage_format / "quantized.npz"
            )
            _save_payload_once(artifact, payload)
            formats[storage_format] = {
                "artifact": artifact.relative_to(run).as_posix(),
                "sha256": sha256(artifact),
                "archive_bytes": artifact.stat().st_size,
                **payload_stats(payload),
            }
        records.append(
            {
                "protocol": protocol,
                "method": method,
                "seed": int(source.stem.rsplit("s", 1)[1]),
                "source_model": source.relative_to(run).as_posix(),
                "source_sha256": sha256(source),
                "source_numeric_scalars": payload_stats(model)["numeric_scalars"],
                "source_numeric_bytes": payload_stats(model)["numeric_bytes"],
                "formats": formats,
            }
        )
    manifest = {
        "status": "PREPARED_WITHOUT_OUTER_EVALUATION",
        "formats": list(FORMATS),
        "inference": "All reduced formats are dequantized to FP32; no native INT8 speed claim.",
        "source_model_count": len(records),
        "models": records,
    }
    content = _json_bytes(manifest)
    _write_once(manifest_path, content)
    _write_once(
        run / "precision" / "manifest.sha256",
        (hashlib.sha256(content).hexdigest() + "\n").encode("ascii"),
    )
    return _verify_manifest(run)[0]


def _deviation(prediction, reference, person):
    l2 = np.linalg.norm(prediction.astype(np.float64) - reference.astype(np.float64), axis=1)
    de00 = runner.delta_e00(prediction, reference)
    return {
        "max_abs_lab_component": float(np.max(np.abs(prediction - reference))),
        "l2_mean": float(l2.mean()),
        "l2_max": float(l2.max()),
        "delta_e00_mean": float(de00.mean()),
        "delta_e00_max": float(de00.max()),
        "person_mean_delta_e00": float(
            np.mean([de00[person == value].mean() for value in np.unique(person)])
        ),
    }


def evaluate(run, cache):
    """Evaluate every frozen format on all outer roles, without choosing a format."""
    run, cache = Path(run), Path(cache)
    manifest, manifest_sha = _verify_manifest(run)
    if sha256(cache) != CACHE_HASH:
        raise ValueError("unauthorized cache or changed data")
    with np.load(cache, allow_pickle=False) as archive:
        data = {
            key: archive[key]
            for key in ("color", "target", "patient", "site", "device")
        }
    records = []
    for entry in manifest["models"]:
        protocol = entry["protocol"]
        with np.load(run / protocol / "roles.npz", allow_pickle=False) as roles:
            held = roles["held"]
        if held.dtype != np.bool_ or held.shape != (len(data["color"]),):
            raise ValueError(f"invalid outer roles: {protocol}")
        x, target = data["color"][held], data["target"][held]
        person, site = data["patient"][held], data["site"][held]
        reference_payload = load_npz(run / entry["formats"]["fp32_reference"]["artifact"])
        reference = runner.predict(dequantize_model(reference_payload), x)
        for storage_format in FORMATS:
            item = entry["formats"][storage_format]
            payload = load_npz(run / item["artifact"])
            prediction = runner.predict(dequantize_model(payload), x)
            measured = runner.metrics(prediction, target, person, site)
            records.append(
                {
                    "protocol": protocol,
                    "method": entry["method"],
                    "seed": entry["seed"],
                    "format": storage_format,
                    "stored_numeric_scalars": item["numeric_scalars"],
                    "stored_numeric_bytes": item["numeric_bytes"],
                    "int8_scale_scalars": item["quantization_scale_scalars"],
                    "person_delta_e": measured["person_mean"],
                    "outer_metrics": measured,
                    "deviation_from_fp32": _deviation(prediction, reference, person),
                }
            )
    result = {
        "evidence": "Exploratory original-TRAIN outer roles; overlapping and historically exposed.",
        "precision_manifest_sha256": manifest_sha,
        "formats": list(FORMATS),
        "format_was_selected_on_outer": False,
        "inference": "Reduced payloads were dequantized to FP32; no native INT8 speed claim.",
        "models": records,
    }
    _write_once(run / "precision" / "evaluation.json", _json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", type=Path, default=ROOT / "experiments/runs/skin_local_search_v1"
    )
    parser.add_argument("--stage", choices=("prepare", "evaluate"), required=True)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare(args.run)
        print(json.dumps({"stage": "prepare", "models": result["source_model_count"]}))
    else:
        if args.cache is None:
            parser.error("--cache is required for evaluate")
        result = evaluate(args.run, args.cache)
        print(json.dumps({"stage": "evaluate", "records": len(result["models"])}))


if __name__ == "__main__":
    main()
