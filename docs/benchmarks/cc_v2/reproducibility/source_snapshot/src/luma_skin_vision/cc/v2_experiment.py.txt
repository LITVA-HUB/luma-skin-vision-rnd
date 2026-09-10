"""Additive v2 source-only fitting and frozen prediction export.

Training never evaluates test/Sony errors. Prediction writes per-image model
outputs without fitting selectors, evaluating errors, or choosing a method.
NPZ is a compressed array format: numpy materializes an array when indexing;
only selected source rows are retained or transferred to the training device.
"""

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import source_identity, write_json

from .benchmark import indices
from .core import reproduction
from .model import reproduction_loss
from .v2 import (
    CHEAP_FEATURE_COLUMNS,
    FEATURE_UNITS,
    CompactResidualCC,
    gain_augment,
    input_validity,
    risk_features_invariant,
    validate_image,
)

SCHEMA = "cc-v2-1"
STRESS_GAINS = [[2, 1, 0.5], [0.5, 1, 2], [1, 2, 0.5], [0.7, 0.8, 1.6]]


def data_fingerprints(data):
    data = Path(data)
    names = ["cube.npz", "cube_manifest.json"]
    sony = [(data / name).is_file() for name in ("sony.npz", "sony_manifest.json")]
    if any(sony) and not all(sony):
        raise ValueError("Sony cache and manifest must both exist or both be absent")
    if all(sony):
        names += ["sony.npz", "sony_manifest.json"]
    return {name: sha256(data / name) for name in names}


def verify_data(data, expected):
    if data_fingerprints(data) != expected:
        raise ValueError("Data fingerprint changed since training")


def split_indices(rows, protocol):
    """Reuse v1 partitions; additionally enforce disjoint source fitting roles."""
    if protocol not in ("official", "camera"):
        raise ValueError("Unsupported split protocol")
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Manifest requires nonempty unique image IDs")
    result = {key: value.astype(np.int64) for key, value in indices(rows, protocol).items()}
    if any(len(v) == 0 for v in result.values()) or len(result["train"]) < 2:
        raise ValueError("All split roles must be populated; training needs >=2 images")
    source = ["train", "val", "risk", "cal"]
    used_rows, used_groups = set(), set()
    for name in source:
        row_set = set(result[name].tolist())
        groups = {rows[i]["group"] for i in row_set}
        if row_set & used_rows or groups & used_groups:
            raise ValueError("Source fitting roles overlap in rows or capture groups")
        used_rows |= row_set
        used_groups |= groups
    if used_rows & set(result["test"].tolist()):
        raise ValueError("Test image overlaps source fitting")
    if protocol == "camera" and used_groups & {rows[i]["group"] for i in result["test"]}:
        raise ValueError("Held-out camera test overlaps source dates")
    return result


def source_snapshot(out):
    """Save exact bytes with the same ordered hash construction as source_identity."""
    root = Path(__file__).resolve().parents[3]
    identity = source_identity()
    paths = sorted((root / "src").rglob("*.py")) + [root / "pyproject.toml", root / "uv.lock"]
    destination = Path(out) / "source_snapshot"
    destination.mkdir(exist_ok=False)
    digest, files = hashlib.sha256(), {}
    for path in paths:
        if not path.exists():
            continue
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        digest.update(relative.encode())
        digest.update(content)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        files[relative] = hashlib.sha256(content).hexdigest()
    if digest.hexdigest() != identity["source_hash"]:
        raise ValueError("Source changed while snapshotting; retry from a stable source tree")
    return identity, {
        "directory": "source_snapshot",
        "source_hash": digest.hexdigest(),
        "files": files,
    }


def _device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name not in ("cpu", "cuda") or (name == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested device unavailable")
    return torch.device(name)


@torch.no_grad()
def predict_tensor(model, x, batch):
    model.eval()
    predictions, contexts = [], []
    for start in range(0, len(x), batch):
        pred, context = model(x[start : start + batch])
        predictions.append(pred)
        contexts.append(context)
    return torch.cat(predictions), torch.cat(contexts)


def _validate_args(args):
    if args.mode not in ("direct", "gw", "sog") or args.backbone not in ("small", "large"):
        raise ValueError("Invalid architecture")
    if args.epochs < 1 or args.batch < 2 or args.gain_aug not in (0, 0.7):
        raise ValueError("Need epochs>=1, batch>=2, gain-aug 0 or 0.7")
    if not isinstance(args.seed, int) or not 0 <= args.seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")


def train(args):
    _validate_args(args)
    out, data = Path(args.out).resolve(), Path(args.data).resolve()
    if out.exists():
        raise FileExistsError(f"Training requires a new output directory: {out}")
    device = _device(args.device)
    hashes = data_fingerprints(data)
    rows = json.loads((data / "cube_manifest.json").read_text(encoding="utf-8"))
    ix = split_indices(rows, args.protocol)
    selected = np.sort(np.concatenate([ix["train"], ix["val"]]))
    with np.load(data / "cube.npz", allow_pickle=False) as cache:
        if cache["images"].shape[0] != len(rows) or cache["gt"].shape != (len(rows), 3):
            raise ValueError("Cube cache/manifest row mismatch")
        # No test/risk/cal/Sony rows survive slicing or enter the optimizer or loss.
        source_images = cache["images"][selected].astype(np.float32)
        source_gt = cache["gt"][selected].astype(np.float32)
    x = torch.from_numpy(source_images).to(device)
    gt = torch.from_numpy(source_gt).to(device)
    validate_image(x)
    if min(x.shape[-2:]) < 32 or not torch.isfinite(gt).all() or not (gt > 0).all():
        raise ValueError("Source requires >=32px images and finite positive illuminant labels")
    local_train, local_val = (
        np.searchsorted(selected, ix["train"]),
        np.searchsorted(selected, ix["val"]),
    )
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out.mkdir(parents=True, exist_ok=False)
    identity, snapshot = source_snapshot(out)
    config = {
        "schema_version": SCHEMA,
        "data": str(data),
        "out": str(out),
        "mode": args.mode,
        "backbone": args.backbone,
        "protocol": args.protocol,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch": args.batch,
        "gain_aug": args.gain_aug,
        "device": str(device),
        "precision": "FP32",
        "resolution": list(x.shape[-2:]),
        "optimizer": {
            "name": "AdamW",
            "lr": 0.001,
            "weight_decay": 0.0001,
            "cosine_min_lr": 0.00002,
            "gradient_clip_norm": 5,
        },
        "augmentation": "TRANSFORMED REAL: diagonal gains with transformed illuminant labels; scalar exposure and horizontal flips; no physical new GT",
        "selection": "lowest unaugmented source-validation mean reproduction error; gain stress diagnostic only",
        "data_hashes": hashes,
        "split_counts": {key: len(value) for key, value in ix.items()},
        "all_split_ids": {key: [rows[i]["id"] for i in value] for key, value in ix.items()},
        "source_snapshot": snapshot,
        **identity,
    }
    write_json(out / "config.json", config)
    model = CompactResidualCC(args.mode, args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00002)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start, best, history = time.perf_counter(), math.inf, []
    try:
        for epoch in range(args.epochs):
            model.train()
            order = rng.permutation(local_train)
            total, count = 0.0, 0
            for offset in range(0, len(order), args.batch):
                batch_ix = order[offset : offset + args.batch]
                if len(batch_ix) == 1:
                    for layer in model.modules():
                        if isinstance(layer, torch.nn.BatchNorm2d):
                            layer.eval()
                image, target, _ = gain_augment(x[batch_ix], gt[batch_ix], args.gain_aug)
                pred, _ = model(image)
                loss = reproduction_loss(pred, target)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach()) * len(batch_ix)
                count += len(batch_ix)
            schedule.step()
            val_pred, _ = predict_tensor(model, x[local_val], args.batch)
            val = float(reproduction(val_pred.cpu().numpy(), source_gt[local_val]).mean())
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": total / count,
                    "train_images": count,
                    "val_reproduction": val,
                }
            )
            if val < best:
                best = val
                torch.save(
                    {
                        "state": {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        },
                        "epoch": epoch + 1,
                    },
                    out / "model.pt",
                )
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(
                    json.dumps(
                        {"epoch": epoch + 1, "val_reproduction": val, "best_validation": best}
                    ),
                    flush=True,
                )
        saved = torch.load(out / "model.pt", weights_only=True, map_location=device)
        model.load_state_dict(saved["state"])
        stress = []
        for gain in STRESS_GAINS:
            gains = torch.tensor(gain, dtype=x.dtype, device=device)
            pred, _ = predict_tensor(model, x[local_val] * gains[None, :, None, None], args.batch)
            value = reproduction(pred.cpu().numpy(), source_gt[local_val] * np.asarray(gain)).mean()
            stress.append(
                {
                    "gains": gain,
                    "mean_reproduction": float(value),
                    "images": len(local_val),
                    "use": "secondary validation diagnostic; never checkpoint selection",
                }
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        summary = {
            "status": "COMPLETED",
            "history": history,
            "best_validation": best,
            "checkpoint_epoch": saved["epoch"],
            "validation_gain_stress": stress,
            "seconds": time.perf_counter() - start,
            "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_bytes": (out / "model.pt").stat().st_size,
            "peak_allocated_mb_including_source_cache": torch.cuda.max_memory_allocated() / 2**20
            if device.type == "cuda"
            else None,
            "peak_reserved_mb_including_source_cache": torch.cuda.max_memory_reserved() / 2**20
            if device.type == "cuda"
            else None,
            "source_cache_mb": (x.numel() * x.element_size() + gt.numel() * gt.element_size())
            / 2**20,
            "invalid_source_images": int((~input_validity(x)).sum()),
            "test_or_external_errors_computed": False,
        }
        write_json(out / "training.json", summary)
        write_json(
            out / "checkpoint_manifest.json",
            {
                "schema_version": SCHEMA,
                "checkpoint_file": "model.pt",
                "checkpoint_sha256": sha256(out / "model.pt"),
                "config_sha256": sha256(out / "config.json"),
                "source_hash": config["source_hash"],
            },
        )
    except Exception as exc:
        write_json(
            out / "failure.json", {"status": "FAILED", "reason": str(exc), "history": history}
        )
        raise
    return out


def verify_run(out, data):
    out = Path(out)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA or manifest.get("schema_version") != SCHEMA:
        raise ValueError("Expected completed v2 checkpoint manifest")
    if (
        manifest.get("checkpoint_file") != "model.pt"
        or manifest.get("checkpoint_sha256") != sha256(out / "model.pt")
        or manifest.get("config_sha256") != sha256(out / "config.json")
        or manifest.get("source_hash") != config["source_hash"]
    ):
        raise ValueError("checkpoint/config binding mismatch")
    verify_data(data, config["data_hashes"])
    snapshot = config["source_snapshot"]
    base = (out / "source_snapshot").resolve()
    digest = hashlib.sha256()
    for name, expected in snapshot["files"].items():
        path = (base / name).resolve()
        if not path.is_relative_to(base) or sha256(path) != expected:
            raise ValueError("Source snapshot changed")
        digest.update(name.encode())
        digest.update(path.read_bytes())
    if (
        digest.hexdigest() != config["source_hash"]
        or snapshot["source_hash"] != config["source_hash"]
    ):
        raise ValueError("Source snapshot hash mismatch")
    if source_identity()["source_hash"] != config["source_hash"]:
        raise ValueError(
            "Live source differs from frozen training snapshot; restore snapshot before inference"
        )
    return config, manifest


@torch.no_grad()
def _predict_cache(model, npz_path, manifest_path, batch, device):
    rows = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Prediction manifest requires unique image IDs")
    result = {key: [] for key in ("pred", "context", "cheap_features", "valid")}
    with np.load(npz_path, allow_pickle=False) as cache:
        images = cache["images"]  # Labels deliberately never accessed in predict.
        if len(images) != len(rows):
            raise ValueError("Prediction cache/manifest row mismatch")
        for start in range(0, len(rows), batch):
            x = torch.from_numpy(images[start : start + batch].astype(np.float32)).to(device)
            pred, context = model(x)
            features = risk_features_invariant(x, pred, context)
            for key, value in (
                ("pred", pred),
                ("context", context),
                ("cheap_features", features["cheap"]),
                ("valid", features["valid"]),
            ):
                result[key].append(value.cpu().numpy())
    return {
        **{key: np.concatenate(value) for key, value in result.items()},
        "ids": np.array([r["id"] for r in rows]),
    }


def predict(args):
    out, data = Path(args.out).resolve(), Path(args.data).resolve()
    config, checkpoint = verify_run(out, data)
    if args.batch < 1:
        raise ValueError("Prediction batch must be positive")
    device = _device(args.device)
    sources = [("predictions.npz", data / "cube.npz", data / "cube_manifest.json")]
    if "sony.npz" in config["data_hashes"]:
        sources.append(("sony_predictions.npz", data / "sony.npz", data / "sony_manifest.json"))
    if args.external is not None:
        sources.append(
            (
                "external_predictions.npz",
                Path(args.external[0]).resolve(),
                Path(args.external[1]).resolve(),
            )
        )
    targets = [out / name for name, _, _ in sources] + [out / "predictions_manifest.json"]
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Prediction diagnostics exist; use --overwrite explicitly to replace")
    torch.set_num_threads(4)
    model = CompactResidualCC(config["mode"], config["backbone"]).to(device).eval()
    model.load_state_dict(
        torch.load(out / "model.pt", weights_only=True, map_location=device)["state"]
    )
    outputs = {}
    for name, npz_path, manifest_path in sources:
        source_hashes = {"npz": sha256(npz_path), "manifest": sha256(manifest_path)}
        values = _predict_cache(model, npz_path, manifest_path, args.batch, device)
        temp = (out / name).with_suffix(".npz.tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **values)
        temp.replace(out / name)
        outputs[name] = {
            "sha256": sha256(out / name),
            "images": len(values["pred"]),
            "invalid_images": int((~values["valid"]).sum()),
            "input_npz": str(npz_path),
            "input_manifest": str(manifest_path),
            "input_hashes": source_hashes,
        }
    write_json(
        out / "predictions_manifest.json",
        {
            "schema_version": SCHEMA,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "config_sha256": checkpoint["config_sha256"],
            "training_source_hash": config["source_hash"],
            "prediction_source_identity": source_identity(),
            "outputs": outputs,
            "context_columns": [f"context_{i:02d}" for i in range(64)],
            "cheap_feature_columns": CHEAP_FEATURE_COLUMNS,
            "cheap_feature_units": FEATURE_UNITS,
            "invariance_scope": "anchored modes, nondegenerate inputs, diagonal channel gains; direct mode has no invariance promise",
            "invalid_policy": "valid=False requires rejection independently of any learned risk score",
            "fitting_or_error_evaluation": False,
        },
    )
    return out / "predictions_manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "predict"])
    parser.add_argument("--data", default="data/processed/cc128")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["direct", "gw", "sog"], default="direct")
    parser.add_argument("--backbone", choices=["small", "large"], default="small")
    parser.add_argument("--protocol", choices=["official", "camera"], default="official")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--gain-aug", type=float, choices=[0, 0.7], default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace v2 prediction diagnostics only; never training",
    )
    parser.add_argument(
        "--external",
        nargs=2,
        metavar=("NPZ", "MANIFEST"),
        help="Additional frozen evaluation-only image cache; prediction action only",
    )
    args = parser.parse_args()
    if args.action == "train" and (args.overwrite or args.external is not None):
        parser.error("--overwrite and --external apply only to predict")
    result = train(args) if args.action == "train" else predict(args)
    print(json.dumps({"output": str(result)}), flush=True)


if __name__ == "__main__":
    main()
