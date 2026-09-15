"""Deterministic, fixed-recipe C_REBASE_V1 OOF training. No model selection.

The CLI refuses to fit unless the input files and trainer match the pre-training
freeze receipt. It never reads historical predictions or reserved data. Every
fold starts from initialization and no holdout metric is computed before epoch
100. A new output directory is required, including for fold reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from luma_skin_vision.color import delta_e00  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def state_sha256(state):
    """Hash tensor contents, independent of torch.save ZIP/storage metadata."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def verify_freeze(paths, freeze_path):
    freeze = json.loads(Path(freeze_path).read_text())
    require(freeze.get("experiment") == "C_REBASE_V1", "Wrong freeze experiment")
    require(freeze.get("phase") == "BEFORE_ANY_OOF_RESULTS", "Missing pre-result freeze declaration")
    entries = freeze["files"]
    hashes = {}
    for role, path in paths.items():
        require(role in entries, f"Freeze receipt lacks {role}")
        expected = entries[role]["sha256"] if isinstance(entries[role], dict) else entries[role]
        actual = sha256(path)
        require(actual == expected, f"Frozen {role} SHA256 mismatch")
        hashes[role] = actual
    for role, entry in entries.items():
        if role in hashes:
            continue
        require(isinstance(entry, dict) and "path" in entry, f"Additional frozen role {role} needs path")
        path = Path(entry["path"])
        if not path.is_absolute():
            path = REPO / path
        actual = sha256(path)
        require(actual == entry["sha256"], f"Frozen {role} SHA256 mismatch")
        hashes[role] = actual
    hashes["freeze"] = sha256(freeze_path)
    return hashes


def validate_config(config):
    require(config["experiment"] == "C_REBASE_V1", "Wrong experiment")
    require(config["architecture"] == [36, 64, 3], "Only frozen 36-64-3 architecture supported")
    require(config["activation"] == "ReLU", "Expected ReLU")
    require(config["initialization"] == "torch_nn_Linear_default", "Unexpected initialization")
    require(config["epochs"] == 100 and config["phase_boundary"] == 50, "Expected 50+50 epochs")
    require(config["optimizer_boundary"] == "continue_state_and_rng", "Optimizer must continue at boundary")
    require(config["scheduler"] is None, "No scheduler in fixed recipe")
    require(config["device"] == "cpu" and config["num_threads"] == 1, "Frozen runtime is one CPU thread")
    require(config["dtype"] == "float32", "Unexpected training dtype")
    require(config["loss"] == "mean_squared_error_standardized_Lab", "Unexpected loss")
    require(config["optimizer"]["name"] == "AdamW", "Expected AdamW")
    require(config["normalization"]["method"] == "training_rows_population_mean_std_float64", "Normalization contract mismatch")
    require(config["shuffle"] == "numpy_RandomState_seed_plus_fold_permutation_each_epoch", "Shuffle contract mismatch")
    require(config["sampling"] == "uniform_images_no_weights", "Sampling contract mismatch")
    require(int(config["batch_size"]) > 0 and config["gradient_clip_norm"] > 0, "Invalid batch or clipping")


def load_inputs(config_path, mapping_path, features_path, freeze_path):
    paths = {
        "config": config_path, "mapping": mapping_path, "features": features_path,
        "trainer": Path(__file__), "color": REPO / "src/luma_skin_vision/color.py",
    }
    hashes = verify_freeze(paths, freeze_path)
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    mapping = json.loads(Path(mapping_path).read_text())
    require(mapping["experiment"] == config["experiment"], "Mapping experiment mismatch")
    with np.load(features_path, allow_pickle=False) as z:
        data = {k: np.asarray(z[k]) for k in z.files}
    n = len(data["row_index"])
    require(n == 966 and len(np.unique(data["patient"])) == 24, "Expected exactly 966 TRAIN images / 24 people")
    require(np.array_equal(data["row_index"], np.arange(n)), "Features must be original row order")
    require(data["color36"].shape == (n, 36) and data["target"].shape == (n, 3), "Unexpected X/Y shape")
    require(np.isfinite(data["color36"]).all() and np.isfinite(data["target"]).all(), "Nonfinite X/Y")
    require(len(np.unique(data["image"])) == n, "Duplicate image pseudonyms")
    people = {str(p) for p in data["patient"]}
    p_to_f = {str(row["patient"]): int(row["fold"]) for row in mapping["people"]}
    require(len(p_to_f) == len(mapping["people"]), "Duplicate people in mapping")
    require(set(p_to_f) == people, "Mapping does not exactly cover TRAIN people")
    fold = np.asarray([p_to_f[str(p)] for p in data["patient"]], dtype=np.int64)
    require(np.array_equal(data["row_index"], mapping["row_index"]), "Mapping row index mismatch")
    require(np.array_equal(fold, mapping["fold_by_row"]), "Row mapping disagrees with person mapping")
    require(set(fold) == set(range(6)), "Expected six folds, numbered 0..5")
    for f in range(6):
        require(len(np.unique(data["patient"][fold == f])) == 4, "Each fold must hold out four people")
        require(not set(data["patient"][fold == f]) & set(data["patient"][fold != f]), "Patient overlap")
    for site in np.unique(data["site"]):
        require(len(np.unique(fold[data["site"] == site])) == 1, "Site spans folds")
    data["fold"] = fold
    versions = config["versions"]
    actual = {"python": platform.python_version(), "pytorch": torch.__version__, "numpy": np.__version__}
    try:
        import sklearn
        actual["sklearn"] = sklearn.__version__
    except ImportError:
        actual["sklearn"] = None
    for key in actual:
        require(versions[key] == actual[key], f"Runtime version differs from frozen recipe: {key}")
    return config, mapping, data, hashes


def fit_normalization(x, y, train_indices, std_floor):
    """Population moments fit only on fold training images, calculated in float64."""
    x_train = np.asarray(x[train_indices], dtype=np.float64)
    y_train = np.asarray(y[train_indices], dtype=np.float64)
    return {
        "x_mean": x_train.mean(0), "x_std": np.maximum(x_train.std(0, ddof=0), std_floor),
        "y_mean": y_train.mean(0), "y_std": np.maximum(y_train.std(0, ddof=0), std_floor),
    }


def summarize(pred, target, people):
    error = delta_e00(pred, target)
    residual = pred - target
    means = [error[people == p].mean() for p in np.unique(people)]
    medians = [np.median(error[people == p]) for p in np.unique(people)]
    weight = np.array([1. / np.sum(people == p) for p in people])
    order = np.argsort(error, kind="stable")
    midpoint = np.searchsorted(np.cumsum(weight[order]), weight.sum() / 2)
    return {
        "images": len(error), "people": len(means), "mean": float(error.mean()),
        "median": float(np.median(error)), "p90": float(np.quantile(error, .9)),
        "p95": float(np.quantile(error, .95)), "max": float(error.max()),
        "fraction_gt_5": float(np.mean(error > 5)), "fraction_gt_10": float(np.mean(error > 10)),
        "mae_Lab": np.mean(np.abs(residual), 0).tolist(),
        "rmse_Lab": np.sqrt(np.mean(residual ** 2, 0)).tolist(),
        "person_balanced_mean": float(np.mean(means)),
        "mean_of_person_medians": float(np.mean(medians)),
        "equal_person_weighted_median": float(error[order[midpoint]]),
    }


def deterministic_runtime(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        require(torch.get_num_interop_threads() == threads, "Interop thread setting changed")
    torch.use_deterministic_algorithms(True)


def create_model():
    return nn.Sequential(nn.Linear(36, 64), nn.ReLU(), nn.Linear(64, 3)).float()


def train_fold(config, data, fold, output, hashes):
    """Fit a single fold from scratch. No resume path or checkpoint input exists."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed = int(config["seed"])
    deterministic_runtime(seed, int(config["num_threads"]))
    train = np.flatnonzero(data["fold"] != fold)
    holdout = np.flatnonzero(data["fold"] == fold)
    require(len(train) and len(holdout), "Empty train/holdout")
    require(not set(data["patient"][train]) & set(data["patient"][holdout]), "Patient leakage")
    require(not set(data["site"][train]) & set(data["site"][holdout]), "Site leakage")
    norm = fit_normalization(data["color36"], data["target"], train, config["normalization"]["std_floor"])
    # No holdout target is transformed, passed to the optimizer or used for selection.
    tx = torch.from_numpy(((data["color36"][train] - norm["x_mean"]) / norm["x_std"]).astype(np.float32))
    ty = torch.from_numpy(((data["target"][train] - norm["y_mean"]) / norm["y_std"]).astype(np.float32))
    model = create_model()
    initial_hash = state_sha256(model.state_dict())
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = config["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt["lr"]),
                                 betas=tuple(opt["betas"]), eps=float(opt["eps"]),
                                 weight_decay=float(opt["weight_decay"]),
                                 amsgrad=False, foreach=False, fused=False)
    rng = np.random.RandomState(seed + int(fold))
    batch_size = int(config["batch_size"])
    epoch_log = []
    started = time.monotonic()
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        perm = rng.permutation(len(train))
        total_loss = 0.
        max_gradient = 0.
        for offset in range(0, len(train), batch_size):
            ids = perm[offset:offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tx[ids])
            loss = torch.mean((prediction - ty[ids]) ** 2)
            require(torch.isfinite(loss).item(), "Nonfinite training loss")
            loss.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
                    "Missing or nonfinite gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"],
                                                      error_if_nonfinite=True, foreach=False)
            max_gradient = max(max_gradient, float(grad_norm))
            optimizer.step()
            total_loss += float(loss.detach()) * len(ids)
        epoch_log.append({"epoch": epoch, "train_standardized_mse": total_loss / len(train),
                          "maximum_preclip_gradient_norm": max_gradient, "finite_gradients": True,
                          "optimizer_steps": int(next(iter(optimizer.state.values()))["step"].item())})
        if epoch in (int(config["phase_boundary"]), int(config["epochs"])):
            random_state = rng.get_state()
            checkpoint = {
                "experiment": config["experiment"], "fold": int(fold), "epoch": epoch,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "normalization": {k: torch.from_numpy(v.copy()) for k, v in norm.items()},
                "train_row_index": torch.from_numpy(train.copy()),
                "holdout_row_index": torch.from_numpy(holdout.copy()),
                "hashes": hashes, "initialization_sha256": initial_hash,
                "state_sha256": state_sha256(model.state_dict()), "config": config,
                "torch_rng_state": torch.get_rng_state(),
                "numpy_shuffle_state": {"algorithm": random_state[0], "keys": random_state[1].tolist(),
                                        "position": random_state[2], "has_gauss": random_state[3],
                                        "cached_gaussian": random_state[4]},
                "phase_boundary_behavior": "continuous_optimizer_and_shuffle_rng_no_restart",
            }
            torch.save(checkpoint, output / f"epoch_{epoch:03d}.pt")
            write_json(output / "training_log.json", epoch_log)
            print(json.dumps({"fold": int(fold), "epoch": epoch,
                              "train_standardized_mse": epoch_log[-1]["train_standardized_mse"],
                              "seconds": time.monotonic() - started}), flush=True)
    changed = {k: not torch.equal(initial[k], v) for k, v in model.state_dict().items()}
    require(all(changed.values()), "An intended trainable parameter did not change")
    model.eval()
    with torch.no_grad():
        hx = torch.from_numpy(((data["color36"][holdout] - norm["x_mean"]) / norm["x_std"]).astype(np.float32))
        pred = model(hx).numpy().astype(np.float64) * norm["y_std"] + norm["y_mean"]
    require(np.isfinite(pred).all(), "Nonfinite holdout prediction")
    checkpoint_path = output / f"epoch_{config['epochs']:03d}.pt"
    # Saved checkpoint must load into a fresh model and reproduce outputs exactly.
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reload_model = create_model()
    reload_model.load_state_dict(loaded["model_state_dict"], strict=True)
    reload_model.eval()
    with torch.no_grad():
        pred_reload = reload_model(hx).numpy().astype(np.float64) * norm["y_std"] + norm["y_mean"]
    require(np.array_equal(pred, pred_reload), "Save/load prediction mismatch")
    result = summarize(pred, data["target"][holdout], data["patient"][holdout])
    evidence = {
        "experiment": config["experiment"], "fold": int(fold), "hashes": hashes,
        "train_images": len(train), "holdout_images": len(holdout),
        "train_people": len(np.unique(data["patient"][train])),
        "holdout_people": sorted(np.unique(data["patient"][holdout]).tolist()),
        "initialization_sha256": initial_hash, "final_state_sha256": state_sha256(model.state_dict()),
        "parameter_count": sum(p.numel() for p in model.parameters()), "parameter_changed": changed,
        "checkpoint_sha256": sha256(checkpoint_path), "save_load_predictions_bitwise_equal": True,
        "normalization_fit_row_indices": train.tolist(), "holdout_evaluated_at_epoch": config["epochs"],
        "wall_seconds": time.monotonic() - started, "metrics": result,
    }
    write_json(output / "fold_result.json", evidence)
    np.savez_compressed(output / "predictions.npz", row_index=data["row_index"][holdout],
                        image=data["image"][holdout], patient=data["patient"][holdout],
                        site=data["site"][holdout], fold=data["fold"][holdout],
                        target=data["target"][holdout], predicted_lab=pred,
                        delta_e00=delta_e00(pred, data["target"][holdout]))
    return holdout, pred, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", default="all", choices=["all", "0", "1", "2", "3", "4", "5"])
    args = parser.parse_args()
    config, mapping, data, hashes = load_inputs(args.config, args.mapping, args.features, args.freeze)
    require(not args.output.exists(), "Output already exists; choose a new directory to preserve prior run")
    args.output.mkdir(parents=True)
    write_json(args.output / "input_receipt.json", {"experiment": config["experiment"], "hashes": hashes,
                                                    "selected_fold": args.fold,
                                                    "no_historical_or_reserved_data_used": True})
    folds = range(6) if args.fold == "all" else [int(args.fold)]
    predictions = np.full((len(data["target"]), 3), np.nan)
    counts = np.zeros(len(predictions), dtype=np.int64)
    results = []
    for fold in folds:
        rows, pred, result = train_fold(config, data, fold, args.output / f"fold_{fold}", hashes)
        predictions[rows] = pred
        counts[rows] += 1
        results.append(result)
    if args.fold == "all":
        require(np.all(counts == 1), "Every image must receive exactly one holdout prediction")
        require(np.isfinite(predictions).all(), "Incomplete OOF predictions")
        private = args.output / "experiment/private_review"
        private.mkdir(parents=True)
        oof = private / "c_rebase_v1_oof.npz"
        np.savez_compressed(oof, **data, predicted_lab=predictions,
                            delta_e00=delta_e00(predictions, data["target"]),
                            experiment=np.array("C_REBASE_V1"), config_sha256=np.array(hashes["config"]),
                            mapping_sha256=np.array(hashes["mapping"]),
                            feature_sha256=np.array(hashes["features"]))
        report = {"experiment": config["experiment"], "scope": "TRAIN-only person-held-out OOF; clinical/dermoscopic MSKCC",
                  "hashes": hashes, "oof_sha256": sha256(oof),
                  "overall": summarize(predictions, data["target"], data["patient"]),
                  "folds": [r["metrics"] | {"fold": r["fold"]} for r in results],
                  "people": {str(p): summarize(predictions[data["patient"] == p], data["target"][data["patient"] == p],
                                                data["patient"][data["patient"] == p]) for p in np.unique(data["patient"])},
                  "historical_C": "HISTORICAL_NOT_REPRODUCIBLE; not a controlled comparator",
                  "independent_reproduction_status": "PENDING_SEPARATE_PROCESS_FOLD_RERUN"}
        write_json(args.output / "oof_metrics.json", report)
        write_json(private / "c_rebase_v1_oof.sha256.json", {"file": oof.name, "sha256": sha256(oof)})
        print(json.dumps({"oof_sha256": report["oof_sha256"], "overall": report["overall"]}), flush=True)


if __name__ == "__main__":
    main()
