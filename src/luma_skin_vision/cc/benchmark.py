import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from luma_skin_vision.experiment import source_identity, write_json

from .core import EXPERT_NAMES, angular, reproduction, selective_curve, summarize
from .model import CompactCC, reproduction_loss


def data_hashes(data):
    hashes = {}
    for name in ["cube.npz", "cube_manifest.json", "sony.npz", "sony_manifest.json"]:
        h = hashlib.sha256()
        with (Path(data) / name).open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                h.update(chunk)
        hashes[name] = h.hexdigest()
    return hashes


def load(data, name="cube"):
    cache = np.load(Path(data) / (name + ".npz"))
    rows = json.loads((Path(data) / (name + "_manifest.json")).read_text())
    return cache, rows


def indices(rows, protocol):
    if protocol == "official":
        return {
            part: np.array([i for i, r in enumerate(rows) if r["subset"] == part])
            for part in ["train", "val", "risk", "cal", "test"]
        }
    # Use only official training images from 550D for fitting; 600D entirely excluded.
    result = {
        part: np.array(
            [
                i
                for i, r in enumerate(rows)
                if r["camera"] == "Canon EOS 550D" and r["subset"] == part
            ]
        )
        for part in ["train", "val", "risk", "cal"]
    }
    # Dates shared with the source fitting population are excluded from held-out camera.
    source_days = {rows[i]["group"] for part in result for i in result[part]}
    result["test"] = np.array(
        [
            i
            for i, r in enumerate(rows)
            if r["camera"] == "Canon EOS 600D" and r["group"] not in source_days
        ]
    )
    return result


@torch.no_grad()
def predict(model, x, ex, batch=64):
    model.eval()
    pred, features = [], []
    for start in range(0, len(x), batch):
        p, f = model(x[start : start + batch], ex[start : start + batch])
        pred.append(p.cpu().numpy())
        features.append(f.cpu().numpy())
    return np.concatenate(pred), np.concatenate(features)


def train(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cache, rows = load(args.data)
    ix = indices(rows, args.protocol)
    if any(len(v) == 0 for v in ix.values()):
        raise ValueError({k: len(v) for k, v in ix.items()})
    # Actual raw test arrays never participate in optimizer/selection/calibration.
    x = torch.tensor(cache["images"].astype(np.float32), device="cuda")
    gt = torch.tensor(cache["gt"], dtype=torch.float32, device="cuda")
    ex = torch.tensor(cache["experts"], dtype=torch.float32, device="cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        **source_identity(),
        "data_hashes": data_hashes(args.data),
        "split_counts": {k: len(v) for k, v in ix.items()},
        "manifest_sha256": hashlib.sha256(
            (Path(args.data) / "cube_manifest.json").read_bytes()
        ).hexdigest(),
        "train_ids": [rows[i]["id"] for i in ix["train"]],
        "all_split_ids": {k: [rows[i]["id"] for i in v] for k, v in ix.items()},
    }
    write_json(out / "config.json", config)
    model = CompactCC(args.method == "proposed").cuda()
    optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.00002)
    best, history = float("inf"), []
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(ix["train"])
        losses = []
        for offset in range(0, len(order), args.batch):
            idx = order[offset : offset + args.batch]
            if len(idx) < 2:
                continue
            image = x[idx].clone()
            if torch.rand(()) < 0.5:
                image = image.flip(-1)
            # Common exposure augmentation; does not alter illuminant chromaticity.
            image *= torch.exp(torch.empty(len(idx), 1, 1, 1, device="cuda").uniform_(-0.5, 0.5))
            pred, _ = model(image, ex[idx])
            loss = reproduction_loss(pred, gt[idx])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optim.step()
            losses.append(float(loss.detach()))
        schedule.step()
        p, _ = predict(model, x[ix["val"]], ex[ix["val"]])
        val = float(reproduction(p, cache["gt"][ix["val"]]).mean())
        history.append(
            {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_reproduction": val}
        )
        if val < best:
            best = val
            torch.save(
                {"state": model.state_dict(), "mixture": model.mixture, "epoch": epoch + 1},
                out / "model.pt",
            )
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                json.dumps({"method": args.method, "epoch": epoch + 1, "val": val, "best": best}),
                flush=True,
            )
    torch.cuda.synchronize()
    write_json(
        out / "training.json",
        {
            "history": history,
            "seconds": time.perf_counter() - start,
            "peak_allocated_mb_including_cache": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mb_including_cache": torch.cuda.max_memory_reserved() / 2**20,
            "cache_gpu_mb": (x.numel() * 4 + gt.numel() * 4 + ex.numel() * 4) / 2**20,
            "parameters": sum(p.numel() for p in model.parameters()),
            "gradient_active_parameters": sum(
                p.numel() for p in model.parameters() if p.grad is not None
            ),
            "checkpoint_bytes": (out / "model.pt").stat().st_size,
            "best_validation": best,
        },
    )


def risk_features(context, pred, experts, mode):
    standard = np.concatenate([context, np.log(np.maximum(pred, 1e-8))], axis=1)
    distances = np.stack([angular(pred, experts[:, i]) for i in range(4)], axis=1)
    disagreement = np.stack(
        [angular(experts[:, i], experts[:, j]) for i in range(4) for j in range(i + 1, 4)], axis=1
    )
    special = np.concatenate([distances, disagreement, experts.reshape(len(pred), -1)], axis=1)
    return (
        standard
        if mode == "context"
        else special
        if mode == "disagreement"
        else np.concatenate([standard, special], axis=1)
    )


def fit_risk(features, errors, risk_idx, cal_idx):
    # Frozen base model never trained on risk/cal groups. Exact held-out residuals.
    center = features[risk_idx].mean(0)
    scale = np.maximum(features[risk_idx].std(0), 0.01)
    design = np.column_stack(
        [np.ones(len(features)), np.clip((features - center) / scale, -10, 10)]
    )
    penalty = np.eye(design.shape[1]) * 10
    penalty[0, 0] = 0
    coef = np.linalg.solve(
        design[risk_idx].T @ design[risk_idx] + penalty,
        design[risk_idx].T @ np.log1p(errors[risk_idx]),
    )
    raw = np.expm1(np.clip(design @ coef, 0, np.log(181)))
    offset = float(np.mean(errors[cal_idx] - raw[cal_idx]))
    return {"center": center, "scale": scale, "coef": coef, "offset": offset}


def apply_risk(features, state):
    design = np.column_stack(
        [np.ones(len(features)), np.clip((features - state["center"]) / state["scale"], -10, 10)]
    )
    return np.maximum(
        0, np.expm1(np.clip(design @ state["coef"], 0, np.log(181))) + state["offset"]
    )


def result(pred, gt, scores, ids, calibration_scores):
    recovery, rep = angular(pred, gt), reproduction(pred, gt)
    thresholds = {}
    for coverage in [1, 0.95, 0.9, 0.8, 0.7, 0.6]:
        threshold = float(np.quantile(calibration_scores, coverage)) if coverage < 1 else None
        accepted = np.ones(len(scores), dtype=bool) if threshold is None else scores <= threshold
        thresholds[str(int(coverage * 100))] = {
            "threshold": threshold,
            "n": int(accepted.sum()),
            "coverage": float(accepted.mean()),
            "mean_reproduction": float(rep[accepted].mean()) if accepted.any() else None,
        }
    return {
        "recovery": summarize(recovery),
        "reproduction": summarize(rep),
        "selective": selective_curve(rep, scores, ids),
        "frozen_source_thresholds": thresholds,
        "pred": pred.tolist(),
        "gt": gt.tolist(),
        "scores": scores.tolist(),
        "errors": rep.tolist(),
        "ids": ids,
    }


def evaluate(args):
    torch.set_num_threads(4)
    out = Path(args.out)
    config = json.loads((out / "config.json").read_text())
    if config["data_hashes"] != data_hashes(args.data):
        raise ValueError("Dataset cache/manifest changed since training")
    cache, rows = load(args.data)
    ix = indices(rows, config["protocol"])
    saved = torch.load(out / "model.pt", weights_only=True, map_location="cuda")
    model = CompactCC(saved["mixture"]).cuda()
    model.load_state_dict(saved["state"])
    x = torch.tensor(cache["images"].astype(np.float32), device="cuda")
    ex = torch.tensor(cache["experts"], dtype=torch.float32, device="cuda")
    pred, context = predict(model, x, ex)
    gt = cache["gt"]
    evaluations = {"source": {}, "sony_pilot": {}}
    sony, sony_rows = load(args.data, "sony")
    sony_pred, sony_context = predict(
        model,
        torch.tensor(sony["images"].astype(np.float32), device="cuda"),
        torch.tensor(sony["experts"], dtype=torch.float32, device="cuda"),
    )
    methods = [
        (config["method"] + "_" + mode, pred, sony_pred, mode)
        for mode in ["context", "disagreement", "combined"]
    ]
    if config["method"] == "baseline":
        methods += [
            (name, cache["experts"][:, i], sony["experts"][:, i], "combined")
            for i, name in enumerate(EXPERT_NAMES)
        ]
    risk_states = {}
    for name, p, sp, mode in methods:
        features = risk_features(context, p, cache["experts"], mode)
        errors = reproduction(p, gt)
        state = fit_risk(features, errors, ix["risk"], ix["cal"])
        scores = apply_risk(features, state)
        sf = risk_features(sony_context, sp, sony["experts"], mode)
        ss = apply_risk(sf, state)
        evaluations["source"][name] = result(
            p[ix["test"]],
            gt[ix["test"]],
            scores[ix["test"]],
            [rows[i]["id"] for i in ix["test"]],
            scores[ix["cal"]],
        )
        evaluations["sony_pilot"][name] = result(
            sp, sony["gt"], ss, [r["id"] for r in sony_rows], scores[ix["cal"]]
        )
        risk_states[name] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in state.items()
        }
    write_json(out / "risk_heads.json", risk_states)
    write_json(
        out / "evaluation.json",
        {"protocol": config["protocol"], "checkpoint_epoch": saved["epoch"], **evaluations},
    )
    # Release dataset tensors before inference memory benchmark.
    single, single_ex = x[:1].clone(), ex[:1].clone()
    del x, ex
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    with torch.inference_mode():
        for i in range(120):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(single, single_ex)
            torch.cuda.synchronize()
            if i >= 20:
                timings.append((time.perf_counter() - start) * 1000)
    write_json(
        out / "inference.json",
        {
            "device": torch.cuda.get_device_name(),
            "input": [1, 3, 128, 128],
            "model_only_median_ms": float(np.median(timings)),
            "model_only_p95_ms": float(np.percentile(timings, 95)),
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
            "excluded": "PNG decode, black subtraction, full-resolution classical experts, risk-head CPU and host/device transfer",
        },
    )
    print(
        json.dumps(
            {
                k: {
                    m: {
                        "mean": v["reproduction"]["mean"],
                        "risk80": v["selective"]["fixed"]["80"]["mean"],
                    }
                    for m, v in val.items()
                }
                for k, val in evaluations.items()
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "evaluate"])
    parser.add_argument("--data", default="data/processed/cc128")
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=["baseline", "proposed"], default="baseline")
    parser.add_argument("--protocol", choices=["official", "camera"], default="official")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    train(args) if args.action == "train" else evaluate(args)


if __name__ == "__main__":
    main()
