"""Local experiment lifecycle. Frozen manifest binding and separate fit roles."""

import json
import random
import time
from pathlib import Path

import numpy as np

from luma_skin_vision.calibration import Calibrator
from luma_skin_vision.color import (
    delta_e00,
    lab_to_srgb,
    linear_to_srgb,
    srgb_to_lab,
    srgb_to_linear,
)
from luma_skin_vision.data import dataset_identity, sha256, validate_records
from luma_skin_vision.environment import inspect_environment
from luma_skin_vision.evaluation import risk_coverage, summarize
from luma_skin_vision.experiment import create_run, write_json
from luma_skin_vision.gates import measurement_gate
from luma_skin_vision.photometry import apply_ccm, fit_ccm, hypotheses
from luma_skin_vision.preprocessing import decode_image, region_tensor
from luma_skin_vision.roi import ambiguity_features, cheek_masks, robust_rgb
from luma_skin_vision.uncertainty import error_features, fit_error, predict_error

LEARNED = ("baseline_c", "baseline_c_plus", "proposed_v1")
METHODS = ("baseline_a0", "baseline_a1", "baseline_a2") + LEARNED


def validate_config(config):
    required = {
        "schema_version",
        "dataset",
        "method",
        "seed",
        "resolution",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_accumulation",
        "precision",
        "device",
        "oof_folds",
        "correction",
    }
    if set(config) != required or config["schema_version"] != "1.0":
        raise ValueError(f"config keys/version invalid; expected {sorted(required)}")
    if config["method"] not in METHODS or config["precision"] not in ("fp32", "fp16"):
        raise ValueError("unsupported method/precision")
    if config["correction"] not in ("none", "gray_world", "shades_of_gray"):
        raise ValueError("invalid correction")
    for key in ("resolution", "epochs", "batch_size", "gradient_accumulation", "oof_folds"):
        if not isinstance(config[key], int) or config[key] < (
            2 if key in ("batch_size", "oof_folds") else 1
        ):
            raise ValueError(f"invalid {key}")
    if config["resolution"] < 32 or config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("invalid training parameters")
    return dict(config)


def prepare(manifest, rows, config):
    image_tensors, features, observed = [], [], []
    cache = {}
    for r in rows:
        if r.image_id not in cache:
            rgb = decode_image(Path(manifest).parent / r.image_path, mirrored=r.image_mirrored)
            masks = cheek_masks(rgb.shape, r.face_bbox)
            aux = ambiguity_features(rgb, masks)
            corrected = hypotheses(rgb)[config["correction"]]
            tensors = [region_tensor(corrected, mask, config["resolution"]) for mask in masks]
            colors = [robust_rgb(corrected, [mask]) for mask in masks]
            cache[r.image_id] = tensors, colors, aux
            del rgb, masks, corrected
        tensors, colors, aux = cache[r.image_id]
        region_index = 0 if r.reference_region == "left_cheek" else 1
        image_tensors.append(tensors[region_index])
        features.append(aux)
        observed.append(colors[region_index])
    return {
        "images": np.asarray(image_tensors),
        "aux": np.asarray(features),
        "observed": np.asarray(observed),
        "target": np.asarray([r.target for r in rows], dtype=np.float32),
    }


def _seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _device(config):
    import torch

    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("requested training device unavailable")
    if config["precision"] == "fp16" and device != "cuda":
        raise ValueError("fp16 training requires CUDA")
    return device


def _fit_model(data, fit_indices, config, validation_indices=None, seed=None):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from luma_skin_vision.models import ColorRegressor

    _seed(config["seed"] if seed is None else seed)
    device = _device(config)
    y = data["target"][fit_indices]
    model = ColorRegressor(config["method"], y.mean(axis=0), np.maximum(y.std(axis=0), 5)).to(
        device
    )
    model.aux_mean.copy_(torch.as_tensor(data["aux"][fit_indices].mean(axis=0), device=device))
    model.aux_scale.copy_(
        torch.as_tensor(np.maximum(data["aux"][fit_indices].std(axis=0), 0.01), device=device)
    )
    ds = TensorDataset(
        *(torch.from_numpy(data[k][fit_indices]).float() for k in ("images", "aux", "target"))
    )
    # Keep all subjects/records; singleton BatchNorm batches handled by eval-only BN.
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    amp = config["precision"] == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state, history = float("inf"), None, []
    for epoch in range(config["epochs"]):
        model.train()
        total = 0.0
        optimizer.zero_grad(set_to_none=True)
        accumulation = config["gradient_accumulation"]
        for step, (x, aux, target) in enumerate(loader):
            if len(x) == 1:
                for layer in model.modules():
                    if isinstance(layer, torch.nn.BatchNorm2d):
                        layer.eval()
            x, aux, target = x.to(device), aux.to(device), target.to(device)
            # Correct scaling for the final partial accumulation group.
            group_start = (step // accumulation) * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            with torch.autocast(device_type=device, enabled=amp, dtype=torch.float16):
                pred = model(x, aux)
                raw_loss = torch.nn.functional.smooth_l1_loss(
                    (pred - target) / model.target_scale, torch.zeros_like(target)
                )
                loss = raw_loss / group_size
            scaler.scale(loss).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total += float(raw_loss.detach()) * len(x)
        val_error = None
        if validation_indices is not None:
            val_prediction = _predict_model(model, data, validation_indices, config)
            val_error = float(delta_e00(val_prediction, data["target"][validation_indices]).mean())
            if val_error < best:
                best = val_error
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(
            {
                "epoch": epoch + 1,
                "train_normalized_huber": total / len(ds),
                "validation_delta_e00": val_error,
            }
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def _predict_model(model, data, indices, config):
    import torch

    device = next(model.parameters()).device
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), config["batch_size"]):
            ix = indices[start : start + config["batch_size"]]
            x = torch.from_numpy(data["images"][ix]).float().to(device)
            aux = torch.from_numpy(data["aux"][ix]).float().to(device)
            result.append(model(x, aux).cpu().numpy())
    return np.concatenate(result)


def _fit_deterministic(data, indices, method):
    if method == "baseline_a2":
        coeff = fit_ccm(
            srgb_to_linear(data["observed"][indices]),
            srgb_to_linear(lab_to_srgb(data["target"][indices])),
            split="train",
        )
        return {
            "method": method,
            "ccm": coeff.tolist(),
            "target_mapping": "Lab->in-gamut linear sRGB; clipped targets prohibited in real work",
        }
    return {"method": method}


def _predict_deterministic(model, data, indices):
    rgb = data["observed"][indices]
    if model["method"] == "baseline_a2":
        rgb = np.clip(
            linear_to_srgb(apply_ccm(srgb_to_linear(rgb), np.asarray(model["ccm"]))), 0, 1
        )
    return srgb_to_lab(rgb)


def train(config, output_root="experiments/runs"):
    config = validate_config(config)
    config["dataset"] = str(Path(config["dataset"]).resolve())
    rows = validate_records(config["dataset"])
    gate = measurement_gate(config["dataset"], rows)
    regions_by_image = {}
    for row in rows:
        regions_by_image.setdefault(row.image_id, set()).add(row.reference_region)
    if any(regions != {"left_cheek", "right_cheek"} for regions in regions_by_image.values()):
        raise ValueError("Bilateral experiment requires both cheek targets for every image")
    if {r.split for r in rows} != {"train", "validation", "calibration", "test"}:
        raise ValueError("All four splits are required for an experiment")
    run, meta = create_run(output_root, config)
    meta.update(dataset_identity(config["dataset"], rows))
    meta["measurement_gate"] = gate
    meta.update(
        status="IN PROGRESS",
        config=config,
        checkpoint_source="random initialization or classical fit",
        pretrained_license="N/A: no pretrained weights",
        augmentation="none; synthetic generator is separate",
        optimizer="AdamW" if config["method"] in LEARNED else "ridge CCM or none",
        hardware=inspect_environment(),
        notes="SYNTHETIC is engineering validation only; no deployment approval",
    )
    write_json(run / "run.json", meta)
    start = time.perf_counter()
    try:
        data = prepare(config["dataset"], rows, config)
        train_ix = np.array([i for i, r in enumerate(rows) if r.split == "train"])
        val_ix = np.array([i for i, r in enumerate(rows) if r.split == "validation"])
        subjects = np.array([r.subject_id for r in rows])
        train_subjects = np.unique(subjects[train_ix])
        if len(train_subjects) < config["oof_folds"] or config["oof_folds"] < 2:
            raise ValueError("insufficient subjects for OOF folds")
        if config["method"] == "baseline_a2":
            gamut = lab_to_srgb(data["target"][train_ix], clip=False)
            if np.any((gamut < -1e-5) | (gamut > 1 + 1e-5)):
                raise ValueError(
                    "A2 target outside sRGB gamut; select a different calibrated target model"
                )
        learned = config["method"] in LEARNED
        if learned:
            import torch

            if _device(config) == "cuda":
                torch.cuda.reset_peak_memory_stats()
        rng = np.random.default_rng(config["seed"])
        rng.shuffle(train_subjects)
        oof_prediction = np.zeros_like(data["target"], dtype=np.float64)
        audit = []
        for fold, held_subjects in enumerate(np.array_split(train_subjects, config["oof_folds"])):
            held = train_ix[np.isin(subjects[train_ix], held_subjects)]
            fit = train_ix[~np.isin(subjects[train_ix], held_subjects)]
            if not len(fit) or not len(held):
                raise ValueError("empty OOF fold")
            audit.append(
                {
                    "fold": fold,
                    "fit_subjects": sorted(set(subjects[fit])),
                    "held_out_subjects": sorted(set(subjects[held])),
                    "fit_records": len(fit),
                    "held_out_records": len(held),
                }
            )
            if learned:
                fold_model, _ = _fit_model(data, fit, config, seed=config["seed"] + fold)
                oof_prediction[held] = _predict_model(fold_model, data, held, config)
                del fold_model
            else:
                fold_model = _fit_deterministic(data, fit, config["method"])
                oof_prediction[held] = _predict_deterministic(fold_model, data, held)
        error = fit_error(
            error_features(config["method"], oof_prediction[train_ix], data["aux"][train_ix]),
            delta_e00(oof_prediction[train_ix], data["target"][train_ix]),
            provenance="subject_out_of_fold",
        )
        write_json(run / "error_model.json", error)
        write_json(run / "oof_audit.json", audit)
        np.savez_compressed(
            run / "oof_predictions.npz",
            prediction=oof_prediction[train_ix],
            target=data["target"][train_ix],
            subject_id=subjects[train_ix],
            record_id=np.array([rows[i].key for i in train_ix]),
        )
        if learned:
            model, history = _fit_model(data, train_ix, config, validation_indices=val_ix)
            torch.save(model.cpu().state_dict(), run / "model.pt")
            meta["parameters"] = sum(p.numel() for p in model.parameters())
            meta["peak_vram_bytes"] = (
                torch.cuda.max_memory_allocated() if _device(config) == "cuda" else None
            )
            meta["training_device"] = _device(config)
            checkpoint = "model.pt"
        else:
            model = _fit_deterministic(data, train_ix, config["method"])
            write_json(run / "model.json", model)
            history, checkpoint = [], "model.json"
            meta.update(parameters=0, peak_vram_bytes=None, training_device="cpu")
        write_json(run / "history.json", history)
        meta.update(
            status="COMPLETED",
            training_duration_seconds=time.perf_counter() - start,
            checkpoint=checkpoint,
            checkpoint_sha256=sha256(run / checkpoint),
            error_model_sha256=sha256(run / "error_model.json"),
        )
        write_json(run / "run.json", meta)
    except Exception as exc:
        meta.update(
            status="FAILED", failure=str(exc), training_duration_seconds=time.perf_counter() - start
        )
        write_json(run / "run.json", meta)
        raise
    return run


def load_run(run):
    run = Path(run)
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if meta["status"] != "COMPLETED":
        raise ValueError("run is not complete")
    if (
        sha256(run / meta["checkpoint"]) != meta["checkpoint_sha256"]
        or sha256(run / "error_model.json") != meta["error_model_sha256"]
    ):
        raise ValueError("model artifact changed")
    if meta["config"]["method"] in LEARNED:
        import torch

        from luma_skin_vision.models import ColorRegressor

        model = ColorRegressor(meta["config"]["method"])
        model.load_state_dict(
            torch.load(run / meta["checkpoint"], map_location="cpu", weights_only=True)
        )
        model.eval()
    else:
        model = json.loads((run / meta["checkpoint"]).read_text())
    return model, meta


def predictions(run, split):
    run = Path(run)
    model, meta = load_run(run)
    config = meta["config"]
    manifest = Path(config["dataset"])
    if sha256(manifest) != meta["dataset_hash"]:
        raise ValueError("dataset changed since training")
    all_rows = validate_records(manifest)
    rows = [r for r in all_rows if r.split == split]
    if not rows:
        raise ValueError("requested split empty")
    data = prepare(manifest, rows, config)
    ix = np.arange(len(rows))
    prediction = (
        _predict_model(model, data, ix, config)
        if config["method"] in LEARNED
        else _predict_deterministic(model, data, ix)
    )
    error_model = json.loads((run / "error_model.json").read_text())
    score = predict_error(error_model, error_features(config["method"], prediction, data["aux"]))
    return rows, prediction, score, meta


def calibrate_run(run, alpha=0.1, tolerance=5):
    rows, pred, score, meta = predictions(run, "calibration")
    calibration = Calibrator.fit(
        score,
        delta_e00(pred, [r.target for r in rows]),
        [r.subject_id for r in rows],
        split="calibration",
        alpha=alpha,
        tolerance=tolerance,
        model_hash=meta["checkpoint_sha256"] + ":" + meta["error_model_sha256"],
        data_kind=meta["data_kind"],
    )
    calibration.save(Path(run) / "calibration.json")
    return calibration


def evaluate_run(run, split="test"):
    if split not in ("test", "validation"):
        raise ValueError("evaluate uses validation or locked test; not training/calibration")
    rows, pred, score, meta = predictions(run, split)
    target = np.array([r.target for r in rows])
    subjects = np.array([r.subject_id for r in rows])
    errors = delta_e00(pred, target)
    image_ids = np.array([r.image_id for r in rows])
    images = np.unique(image_ids)
    # Selection operates on the whole photograph: mean cheek error, max predicted error.
    image_error = np.array([errors[image_ids == k].mean() for k in images])
    image_score = np.array([score[image_ids == k].max() for k in images])
    report = {
        "data_kind": meta["data_kind"],
        "split": split,
        "dataset_hash": meta["dataset_hash"],
        "method": meta["config"]["method"],
        "metrics": summarize(pred, target, subjects),
        "risk_coverage_unit": "image; region error averaged, worst region score selects entire image",
        "risk_coverage": risk_coverage(image_error, image_score, images),
        "full_risk_coverage": risk_coverage(
            image_error, image_score, images, np.arange(1, len(images) + 1) / len(images)
        ),
        "image_count": len(images),
        "deployment_approved": False,
        "groups": {},
    }
    for name in ("device_model", "lighting_id", "front_or_rear"):
        values = np.array([getattr(r, name) for r in rows])
        report["groups"][name] = {
            str(k): summarize(pred[values == k], target[values == k], subjects[values == k])
            for k in np.unique(values)
        }
    bands = np.digitize(target[:, 0], [30, 50, 70])
    report["groups"]["measured_L_band"] = {
        str(k): summarize(pred[bands == k], target[bands == k], subjects[bands == k])
        for k in np.unique(bands)
    }
    variability = {}
    for domain in ("device_model", "lighting_id"):
        group_variance = []
        for subject in np.unique(subjects):
            for region in ("left_cheek", "right_cheek"):
                indices = [
                    i
                    for i, r in enumerate(rows)
                    if r.subject_id == subject and r.reference_region == region
                ]
                domains = sorted({getattr(rows[i], domain) for i in indices})
                if len(domains) > 1:
                    means = [
                        pred[[i for i in indices if getattr(rows[i], domain) == d]].mean(axis=0)
                        for d in domains
                    ]
                    group_variance.append(np.var(means, axis=0).tolist())
        variability[domain] = np.mean(group_variance, axis=0).tolist() if group_variance else None
    report["cross_domain_variance_lab"] = variability
    calpath = Path(run) / "calibration.json"
    if calpath.exists():
        cal = Calibrator.load(calpath)
        if cal.model_hash != meta["checkpoint_sha256"] + ":" + meta["error_model_sha256"]:
            raise ValueError("calibration model binding mismatch")
        accepted = cal.upper_error(image_score) <= cal.tolerance
        report["frozen_policy"] = {
            "coverage": float(accepted.mean()),
            "accepted_images": int(accepted.sum()),
            "mean_delta_e00": float(image_error[accepted].mean()) if accepted.any() else None,
            "calibration_subjects": cal.subject_count,
            "finite_bound": cal.residual_quantile is not None,
            "domain_validated": cal.domain_validated,
            "warning": "Subject-marginal upper bound is not a conditional selected-risk guarantee",
        }
    write_json(Path(run) / f"evaluation_{split}.json", report)
    write_json(
        Path(run) / f"predictions_{split}.json",
        [
            dict(
                record_id=r.key,
                image_id=r.image_id,
                subject_id=r.subject_id,
                prediction=p.tolist(),
                target=r.target.tolist(),
                predicted_error=float(s),
                observed_error=float(e),
            )
            for r, p, s, e in zip(rows, pred, score, errors, strict=True)
        ],
    )
    return report
