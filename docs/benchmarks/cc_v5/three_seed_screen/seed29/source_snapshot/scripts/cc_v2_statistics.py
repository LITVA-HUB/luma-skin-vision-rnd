"""Conventional color-statistics regression: source-only screen, no novelty claim.

Only official train/validation rows are numerically decoded during the screen.
Other NPY rows are skipped as uninterpreted bytes inside the compressed archive.
Prediction is a separate frozen action and never reads illuminant labels.
"""

import argparse
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from luma_skin_vision.cc.core import reproduction, summarize
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import source_identity, write_json

SCHEMA = "cc-statistics-v1"
EPS = 1e-12
POWERS = (1, 2, 4, 6, 10)
QUANTILES = (10, 25, 50, 75, 90, 95, 99)
NAMES = ("ridge1", "ridge10", "ridge100", "hgb7", "hgb15", "extratrees128")
STAT_NAMES = [f"minkowski_p{p}" for p in POWERS] + [f"quantile{q}" for q in QUANTILES]
FEATURE_COLUMNS = (
    [f"log_{stat}_{channel}" for stat in STAT_NAMES for channel in ("r", "g", "b")]
    + [
        f"patch2x2_{row}{col}_{channel}_over_global_gw"
        for channel in ("r", "g", "b")
        for row in range(2)
        for col in range(2)
    ]
    + [f"spatial_std_{channel}_over_global_gw" for channel in ("r", "g", "b")]
)


def read_npz_rows(path, key, selected, expected_rows=None):
    """Read selected C-order NPY rows without deserializing any excluded row."""
    selected = np.asarray(selected)
    if (
        selected.ndim != 1
        or selected.dtype.kind not in "iu"
        or len(selected) == 0
        or np.any(selected[1:] <= selected[:-1])
    ):
        raise ValueError("Selected rows must be sorted unique nonempty integer indices")
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("Unsupported NPY header version")
        if (
            fortran
            or dtype.kind != "f"
            or len(shape) < 2
            or (expected_rows is not None and shape[0] != expected_rows)
        ):
            raise ValueError("Expected matching C-order floating image/GT rows")
        if selected[0] < 0 or selected[-1] >= shape[0]:
            raise ValueError("Selected row outside cache")
        row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        result = np.empty((len(selected), *shape[1:]), dtype=dtype)
        cursor = 0
        for destination, index in enumerate(selected):
            skip = (int(index) - cursor) * row_bytes
            while skip:
                content = stream.read(min(skip, 8 * 1024 * 1024))
                if not content:
                    raise ValueError("Truncated excluded NPY rows")
                skip -= len(content)
            content = stream.read(row_bytes)
            if len(content) != row_bytes:
                raise ValueError("Truncated selected NPY row")
            result[destination] = np.frombuffer(content, dtype=dtype).reshape(shape[1:])
            cursor = int(index) + 1
    return result


def featurize(images, mode):
    """51 fixed per-image features; no learned normalization or camera metadata.

    Global statistics include common masked zeros and use natural logs. Direct
    statistics use a common scalar (mean of the three GW anchors); residual-mode
    statistics use unnormalized per-channel GW anchors. Both append the same
    channel-normalized spatial ratios. Quantiles use NumPy linear interpolation.
    """
    if mode not in ("direct", "gw"):
        raise ValueError("Mode must be direct or gw")
    x = np.asarray(images, dtype=np.float64)
    if (
        x.ndim != 4
        or x.shape[0] < 1
        or x.shape[1] != 3
        or min(x.shape[2:]) < 2
        or not np.isfinite(x).all()
        or np.any(x < 0)
    ):
        raise ValueError("Expected finite nonnegative NCHW thumbnails >=2x2")
    raw_anchor = x.mean(axis=(2, 3))
    valid = (raw_anchor > EPS).all(axis=1)
    anchor = np.maximum(raw_anchor, EPS)
    normalizer = anchor if mode == "gw" else anchor.mean(axis=1, keepdims=True)
    z = x / normalizer[:, :, None, None]
    maximum = np.maximum(z.max(axis=(2, 3)), EPS)
    scaled = z / maximum[:, :, None, None]
    stats = [maximum * np.mean(scaled**p, axis=(2, 3)) ** (1 / p) for p in POWERS]
    stats += list(np.quantile(z, np.asarray(QUANTILES) / 100, axis=(2, 3), method="linear"))
    global_features = np.log(np.maximum(np.stack(stats, axis=1), EPS)).reshape(len(x), -1)
    if mode == "gw":
        # The valid-channel p1/anchor ratio is exactly one. Roundoff here would
        # be amplified by a fitted scaler into a spurious non-invariant feature.
        global_features[:, :3] = np.where(raw_anchor > EPS, 0.0, global_features[:, :3])
    relative = x / anchor[:, :, None, None]
    patch = np.empty((len(x), 3, 2, 2), dtype=np.float64)
    for row, ys in enumerate(np.array_split(np.arange(x.shape[2]), 2)):
        for col, xs in enumerate(np.array_split(np.arange(x.shape[3]), 2)):
            patch[:, :, row, col] = relative[:, :, ys][:, :, :, xs].mean(axis=(2, 3))
    feature = np.concatenate(
        [global_features, patch.reshape(len(x), -1), relative.std(axis=(2, 3))], axis=1
    )
    return feature, anchor, valid


def target_ratios(gt, anchor, mode):
    gt = np.asarray(gt, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if (
        mode not in ("direct", "gw")
        or gt.ndim != 2
        or gt.shape[1] != 3
        or gt.shape != anchor.shape
        or not np.isfinite(gt).all()
        or np.any(gt <= 0)
        or not np.isfinite(anchor).all()
        or np.any(anchor <= 0)
    ):
        raise ValueError("Expected paired positive Nx3 illuminants and anchors")
    log_gt = np.log(gt)
    result = log_gt[:, [0, 2]] - log_gt[:, 1:2]
    if mode == "gw":
        log_anchor = np.log(anchor)
        result -= log_anchor[:, [0, 2]] - log_anchor[:, 1:2]
    return result


def decode_prediction(raw, anchor, mode):
    raw, anchor = np.asarray(raw, dtype=np.float64), np.asarray(anchor, dtype=np.float64)
    if (
        mode not in ("direct", "gw")
        or raw.shape != (len(anchor), 2)
        or not np.isfinite(raw).all()
        or not np.isfinite(anchor).all()
        or np.any(anchor <= 0)
    ):
        raise ValueError("Invalid predicted ratios or anchor")
    # Fixed symmetric bound applies to learned output, never after anchor fusion.
    log_rgb = np.zeros((len(raw), 3))
    log_rgb[:, [0, 2]] = np.clip(raw, -8, 8)
    if mode == "gw":
        log_rgb += np.log(anchor)
    rgb = np.exp(log_rgb - log_rgb.max(axis=1, keepdims=True))
    return rgb / np.linalg.norm(rgb, axis=1, keepdims=True)


def model_spec(name, seed, threads):
    if name not in NAMES:
        raise ValueError("Unknown statistics model")
    if name.startswith("ridge"):
        return {
            "family": "Ridge",
            "alpha": float(name[5:]),
            "standard_scaler": "fit on train only",
            "features": 51,
            "outputs": 2,
        }
    if name.startswith("hgb"):
        return {
            "family": "MultiOutput HistGradientBoosting",
            "max_leaf_nodes": int(name[3:]),
            "max_iter": 100,
            "min_samples_leaf": 20,
            "l2_regularization": 10,
            "learning_rate": 0.1,
            "early_stopping": False,
            "random_state": seed,
            "features": 51,
            "outputs": 2,
        }
    return {
        "family": "ExtraTreesRegressor",
        "n_estimators": 128,
        "min_samples_leaf": 5,
        "max_features": 1.0,
        "random_state": seed,
        "n_jobs": threads,
        "features": 51,
        "outputs": 2,
    }


def estimator(name, seed=17, threads=4):
    model_spec(name, seed, threads)
    if name.startswith("ridge"):
        return make_pipeline(StandardScaler(), Ridge(alpha=float(name[5:])))
    if name.startswith("hgb"):
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_leaf_nodes=int(name[3:]),
                max_iter=100,
                min_samples_leaf=20,
                l2_regularization=10,
                learning_rate=0.1,
                early_stopping=False,
                random_state=seed,
            ),
            n_jobs=1,
        )
    return ExtraTreesRegressor(
        n_estimators=128, min_samples_leaf=5, max_features=1.0, random_state=seed, n_jobs=threads
    )


def _feature_batches(images, mode, batch=32):
    parts = [featurize(images[i : i + batch], mode) for i in range(0, len(images), batch)]
    return tuple(np.concatenate([part[j] for part in parts]) for j in range(3))


def screen(data, out, summary, *, seed=17, threads=4, names=None, expected_counts=(1126, 119)):
    data, out, summary = Path(data).resolve(), Path(out).resolve(), Path(summary).resolve()
    if out.exists() or summary.exists():
        raise FileExistsError("Source screen output and summary must both be new")
    if not 1 <= threads <= 4 or not 0 <= seed < 2**32:
        raise ValueError("Use 1–4 CPU threads and a uint32 seed")
    names = list(NAMES if names is None else names)
    if not names or len(set(names)) != len(names) or any(name not in NAMES for name in names):
        raise ValueError("Invalid candidate grid")
    rows = json.loads((data / "cube_manifest.json").read_text(encoding="utf-8"))
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Expected unique image IDs")
    ix = {
        name: np.array([i for i, row in enumerate(rows) if row["subset"] == name], dtype=np.int64)
        for name in ("train", "val")
    }
    counts = (len(ix["train"]), len(ix["val"]))
    if counts != tuple(expected_counts) or min(counts) < 2:
        raise ValueError(f"Unexpected immutable source split sizes: {counts}")
    if {rows[i]["group"] for i in ix["train"]} & {rows[i]["group"] for i in ix["val"]}:
        raise ValueError("Train/validation capture-group leakage")
    selected = np.sort(np.concatenate(list(ix.values())))
    train_ix, val_ix = np.searchsorted(selected, ix["train"]), np.searchsorted(selected, ix["val"])
    hashes = {name: sha256(data / name) for name in ("cube.npz", "cube_manifest.json")}
    images = read_npz_rows(data / "cube.npz", "images", selected, expected_rows=len(rows))
    gt = read_npz_rows(data / "cube.npz", "gt", selected, expected_rows=len(rows)).astype(
        np.float64
    )
    if gt.shape != (len(selected), 3) or not np.isfinite(gt).all() or np.any(gt <= 0):
        raise ValueError("Invalid source-only illuminant labels")
    out.mkdir(parents=True, exist_ok=False)
    script_bytes = Path(__file__).read_bytes()
    (out / "script_snapshot.py").write_bytes(script_bytes)
    identity = source_identity()
    report = {
        "schema_version": SCHEMA,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "SOURCE TRAIN/VALIDATION ONLY",
        "evidence_kind": "MEASURED SOURCE VALIDATION; model-selection evidence, not unbiased test performance",
        "novelty_claim": False,
        "selection_rule": "lowest source-validation mean reproduction error; all candidates retained; no target/risk/calibration errors",
        "source_ids": {name: [rows[i]["id"] for i in idx] for name, idx in ix.items()},
        "source_groups": {
            name: sorted({rows[i]["group"] for i in idx}) for name, idx in ix.items()
        },
        "source_counts": {name: len(idx) for name, idx in ix.items()},
        "data_root": str(data),
        "data_hashes": hashes,
        "source_identity": identity,
        "script_sha256": sha256(out / "script_snapshot.py"),
        "seed": seed,
        "threads": threads,
        "libraries": {
            "numpy": np.__version__,
            "scikit_learn": sklearn_version,
            "joblib": joblib.__version__,
        },
        "validation_ground_truth": gt[val_ix].tolist(),
        "features": {
            "count": 51,
            "columns": FEATURE_COLUMNS,
            "powers": list(POWERS),
            "quantiles": list(QUANTILES),
            "quantile_method": "linear, includes masked zeros",
            "normalization": "direct: common mean-GW scalar; gw: unnormalized per-channel GW; spatial ratios always channel normalized",
            "learned_log_output_clip": [-8, 8],
        },
        "candidates": [],
    }
    with threadpool_limits(limits=threads):
        for mode in ("direct", "gw"):
            start = time.perf_counter()
            x, anchors, valid = _feature_batches(images, mode)
            feature_seconds = time.perf_counter() - start
            if not valid.all():
                raise ValueError(
                    "Invalid source anchors require explicit protocol revision; no silent filtering"
                )
            target = target_ratios(gt[train_ix], anchors[train_ix], mode)
            for name in names:
                candidate = mode + "_" + name
                model = estimator(name, seed, threads)
                start = time.perf_counter()
                model.fit(x[train_ix], target)
                train_seconds = time.perf_counter() - start
                raw = model.predict(x[val_ix])
                prediction = decode_prediction(raw, anchors[val_ix], mode)
                error = reproduction(prediction, gt[val_ix])
                file = candidate + ".joblib"
                joblib.dump(
                    {
                        "model": model,
                        "mode": mode,
                        "name": name,
                        "schema_version": SCHEMA,
                        "feature_columns": FEATURE_COLUMNS,
                    },
                    out / file,
                )
                record = {
                    "candidate": candidate,
                    "mode": mode,
                    "model": name,
                    "parameters": model_spec(name, seed, threads),
                    "train_seconds": train_seconds,
                    "source_feature_seconds": feature_seconds,
                    "model_file": file,
                    "model_bytes": (out / file).stat().st_size,
                    "model_sha256": sha256(out / file),
                    "val_mean_reproduction": float(error.mean()),
                    "val_reproduction": summarize(error),
                    "val_predictions": prediction.tolist(),
                    "val_errors": error.tolist(),
                    "val_learned_output_clipped_fraction": float(
                        np.mean(np.any(np.abs(raw) > 8, axis=1))
                    ),
                }
                report["candidates"].append(record)
                write_json(out / "screen_progress.json", report)
                print(
                    json.dumps(
                        {
                            "candidate": candidate,
                            "val_mean_reproduction": record["val_mean_reproduction"],
                            "train_seconds": train_seconds,
                        }
                    ),
                    flush=True,
                )
    report["selected_by_mode"] = {
        mode: min(
            (r for r in report["candidates"] if r["mode"] == mode),
            key=lambda r: r["val_mean_reproduction"],
        )["candidate"]
        for mode in ("direct", "gw")
    }
    report["selected_overall"] = min(
        report["candidates"], key=lambda r: r["val_mean_reproduction"]
    )["candidate"]
    if (
        report["script_sha256"] != sha256(Path(__file__))
        or identity["source_hash"] != source_identity()["source_hash"]
    ):
        raise ValueError(
            "Source changed during screen; preserve progress but do not declare frozen result"
        )
    if hashes != {name: sha256(data / name) for name in hashes}:
        raise ValueError("Data changed during source screen")
    write_json(out / "screen.json", report)
    summary.parent.mkdir(parents=True, exist_ok=True)
    write_json(summary, report)
    return report


def predict_arrays(payload, images):
    """Frozen conventional estimator output plus context for later source-only risk fitting."""
    x, anchor, valid = _feature_batches(images, payload["mode"])
    prediction = decode_prediction(payload["model"].predict(x), anchor, payload["mode"])
    invariant, _, _ = _feature_batches(images, "gw")
    residual = target_ratios(prediction, anchor, "gw")
    return {
        "pred": prediction,
        "context": x,
        "cheap_features": np.column_stack([residual, invariant]),
        "valid": valid,
    }


def predict_frozen(run, candidate, input_npz, input_manifest, output, *, threads=4):
    run, input_npz, input_manifest, output = map(Path, (run, input_npz, input_manifest, output))
    output_manifest = output.with_suffix(".manifest.json")
    if output.exists() or output_manifest.exists():
        raise FileExistsError("Frozen prediction output already exists")
    if not 1 <= threads <= 4:
        raise ValueError("Use 1–4 CPU threads")
    report = json.loads((run / "screen.json").read_text(encoding="utf-8"))
    matches = [record for record in report["candidates"] if record["candidate"] == candidate]
    if report["schema_version"] != SCHEMA or len(matches) != 1:
        raise ValueError("Unknown frozen statistics candidate")
    record = matches[0]
    path = run / record["model_file"]
    if sha256(path) != record["model_sha256"]:
        raise ValueError("Model artifact hash changed")
    if (
        sha256(Path(__file__)) != report["script_sha256"]
        or sha256(run / "script_snapshot.py") != report["script_sha256"]
        or source_identity()["source_hash"] != report["source_identity"]["source_hash"]
    ):
        raise ValueError("Statistics/source implementation changed")
    input_hashes = {"npz": sha256(input_npz), "manifest": sha256(input_manifest)}
    rows = json.loads(input_manifest.read_text(encoding="utf-8"))
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Prediction manifest requires unique IDs")
    images = read_npz_rows(input_npz, "images", np.arange(len(rows)), expected_rows=len(rows))
    payload = joblib.load(path)  # Local hash-checked sklearn artifacts only.
    with threadpool_limits(limits=threads):
        values = predict_arrays(payload, images)
    values["ids"] = np.array([row["id"] for row in rows])
    if input_hashes != {"npz": sha256(input_npz), "manifest": sha256(input_manifest)}:
        raise ValueError("Prediction input changed during inference")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(stream, **values)
    write_json(
        output_manifest,
        {
            "schema_version": SCHEMA,
            "candidate": candidate,
            "screen_sha256": sha256(run / "screen.json"),
            "model_sha256": record["model_sha256"],
            "script_sha256": report["script_sha256"],
            "input_npz": str(input_npz.resolve()),
            "input_manifest": str(input_manifest.resolve()),
            "input_hashes": input_hashes,
            "prediction_sha256": sha256(output),
            "context_columns": FEATURE_COLUMNS,
            "cheap_feature_columns": ["pred_over_gw_log_r_minus_g", "pred_over_gw_log_b_minus_g"]
            + FEATURE_COLUMNS,
            "invalid_policy": "valid=False must be refused independently of learned risk",
            "gt_read_or_errors_computed": False,
        },
    )
    return output_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "predict"])
    parser.add_argument("--data", type=Path, default=Path("data/processed/cc128"))
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/ccv2_statistics"))
    parser.add_argument(
        "--summary", type=Path, default=Path("docs/benchmarks/cc_v2/statistics_source_screen.json")
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--candidate")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prediction-output", type=Path)
    args = parser.parse_args()
    if args.action == "train":
        report = screen(args.data, args.out, args.summary, seed=args.seed, threads=args.threads)
        print(
            json.dumps(
                {
                    "summary": str(args.summary),
                    "selected_by_mode": report["selected_by_mode"],
                    "selected_overall": report["selected_overall"],
                }
            ),
            flush=True,
        )
    else:
        if any(
            value is None
            for value in (args.candidate, args.input, args.manifest, args.prediction_output)
        ):
            parser.error("predict requires --candidate, --input, --manifest, --prediction-output")
        result = predict_frozen(
            args.out,
            args.candidate,
            args.input,
            args.manifest,
            args.prediction_output,
            threads=args.threads,
        )
        print(json.dumps({"prediction_manifest": str(result)}), flush=True)


if __name__ == "__main__":
    main()
