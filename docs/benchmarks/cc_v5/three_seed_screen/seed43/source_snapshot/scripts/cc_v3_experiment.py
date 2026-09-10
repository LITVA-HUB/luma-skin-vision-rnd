"""Immutable source-only V3 graph/posterior screen. No test rows are decoded."""

import argparse
import json
import math
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v3_model import ColorFramePosteriorNet, camera_posterior_nll, posterior_nll
from torch.nn import functional as F

from luma_skin_vision.cc.core import angular, reproduction, summarize
from luma_skin_vision.cc.v2_experiment import source_snapshot
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT = Path(__file__).resolve().parents[1]
DATA_HASHES = {
    "cube.npz": "8323048ad50deb5aa7a0f5c8b9787ed311e4c8779a8ff2d87160068a07e92128",
    "cube_manifest.json": "912927e16f32a55b0289910f32790be0ddaf34062ce71a1bf3fb98ae75a7edee",
}
LOSS = {"reproduction_degrees": 1.0, "canonical_nll": 0.02, "positivity": 10.0}


def source_rows(rows):
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Unique nonempty source IDs required")
    roles = {
        k: np.array([i for i, r in enumerate(rows) if r["subset"] == k], dtype=np.int64)
        for k in ("train", "val")
    }
    if any(len(v) == 0 for v in roles.values()):
        raise ValueError("Both fitting roles required")
    groups = [{rows[i]["group"] for i in roles[k]} for k in ("train", "val")]
    if groups[0] & groups[1]:
        raise ValueError("Training/validation capture group overlap")
    selected = np.sort(np.concatenate(list(roles.values())))
    return (
        selected,
        np.searchsorted(selected, roles["train"]),
        np.searchsorted(selected, roles["val"]),
    )


def point_objective(raw, gt):
    """All-row clipped raw point surrogate plus normalized negativity penalty.

    Evaluation uses explicit valid/fallback output. Clipping here only supplies
    a finite training surrogate; the separate penalty gives negative channels
    a repair gradient. It never turns an invalid output into an accepted one.
    """
    unit = F.normalize(raw, dim=-1)
    ratio = F.normalize(gt / unit.clamp_min(1e-6), dim=-1)
    neutral = torch.ones_like(ratio) / math.sqrt(3)
    angle = torch.atan2(
        torch.linalg.cross(ratio, neutral, dim=-1).norm(dim=-1), (ratio * neutral).sum(-1)
    ) * (180 / math.pi)
    return angle.mean(), F.relu(1e-4 - unit).mean()


def validation_key(mean_error, valid_fraction):
    return float(mean_error) if valid_fraction >= 0.99 and math.isfinite(mean_error) else math.inf


def training_summary(sums, seen):
    return {
        key: float(value / seen) if seen else None
        for key, value in zip(("train_reproduction", "train_nll", "train_positivity"), sums)
    }


@torch.no_grad()
def evaluate(model, x, gt, batch):
    model.eval()
    arrays = {
        k: []
        for k in ("pred", "valid", "transport_risk", "invalid_posterior_mass", "frame_condition")
    }
    density = []
    for start in range(0, len(x), batch):
        output = model(x[start : start + batch])
        for k in arrays:
            arrays[k].append(output[k].cpu().numpy())
        density.append(
            camera_posterior_nll(output, gt[start : start + batch], "none").cpu().numpy()
        )
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    label = gt.cpu().numpy().astype(np.float64)
    errors = reproduction(arrays["pred"].astype(np.float64), label)
    arrays["reproduction"] = errors
    arrays["recovery"] = angular(arrays["pred"].astype(np.float64), label)
    arrays["camera_nll"] = np.concatenate(density)
    result = {
        "reproduction": summarize(errors),
        "recovery": summarize(arrays["recovery"]),
        "valid_fraction": float(arrays["valid"].mean()),
        "camera_nll_mean": float(np.nanmean(arrays["camera_nll"])),
        "transport_risk_mean": float(arrays["transport_risk"].mean()),
    }
    return result, arrays


def train(args):
    if args.epochs < 1 or args.batch < 2 or args.mode not in ("direct", "diagonal", "frame"):
        raise ValueError("Invalid budget/mode")
    out, data = Path(args.out).resolve(), Path(args.data).resolve()
    if out.exists():
        raise FileExistsError("Use a fresh immutable run directory")
    actual = {name: sha256(data / name) for name in DATA_HASHES}
    if actual != DATA_HASHES:
        raise ValueError("Source cache/manifest fingerprints changed")
    rows = json.loads((data / "cube_manifest.json").read_text(encoding="utf-8"))
    selected, train_ix, val_ix = source_rows(rows)
    if (len(rows), len(train_ix), len(val_ix)) != (2234, 1126, 119):
        raise ValueError("Unexpected source population")
    out.mkdir(parents=True)
    identity, snapshot = source_snapshot(out)
    scripts = {}
    for name in ("cc_v3_experiment.py", "cc_v3_model.py", "cc_v2_statistics.py"):
        destination = out / "source_snapshot" / "scripts" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "scripts" / name, destination)
        scripts[name] = sha256(destination)
    config = {
        "schema": "cc-v3-source-screen-1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "loss_weights": LOSS,
        "data_hashes": actual,
        "source_identity": identity,
        "snapshot": snapshot,
        "scripts_sha256": scripts,
        "train_ids": [rows[selected[i]]["id"] for i in train_ix],
        "validation_ids": [rows[selected[i]]["id"] for i in val_ix],
        "selection": "Minimum all-population validation reproduction mean among >=99% valid epochs, including untrained epoch0. No test/risk/cal/INTEL decoding.",
        "augmentation": "Common exposure exp(U[-0.5,0.5]); independent image horizontal flip p0.5; no channel gains/camera metadata.",
        "optimizer": "AdamW lr0.001 weight_decay0.0001; cosine epoch schedule eta_min0.00002; clip_norm5; FP32 neural/FP64 frame; no AMP",
        "width": 192,
        "layers": 4,
        "hypotheses": 8,
        "torch": torch.__version__,
        "cudnn_benchmark": False,
        "tf32": False,
        "limitation": "Source validation is reused for development and checkpoint selection, not an independent test result. CUDA index_add may be nondeterministic.",
    }
    write_json(out / "config.json", config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    images = read_npz_rows(data / "cube.npz", "images", selected, 2234).astype(np.float32)
    labels = read_npz_rows(data / "cube.npz", "gt", selected, 2234).astype(np.float32)
    x, gt = torch.from_numpy(images).to(device), torch.from_numpy(labels).to(device)
    if (
        not torch.isfinite(x).all()
        or (x < 0).any()
        or not torch.isfinite(gt).all()
        or (gt <= 0).any()
    ):
        raise ValueError("Invalid source pixels/GT")
    model = ColorFramePosteriorNet(mode=args.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=2e-5)
    train_ix = torch.as_tensor(train_ix, device=device)
    val_ix = torch.as_tensor(val_ix, device=device)
    best, best_epoch, history = math.inf, None, []
    started = time.perf_counter()
    for epoch in range(args.epochs + 1):
        sums, seen = np.zeros(3), 0
        epoch_start = time.perf_counter()
        if epoch:
            model.train()
            order = train_ix[torch.randperm(len(train_ix), device=device)]
            for batch_ix in order.split(args.batch):
                xb = x[batch_ix]
                flip = torch.rand(len(xb), 1, 1, 1, device=device) < 0.5
                xb = torch.where(flip, xb.flip(-1), xb)
                xb = xb * torch.empty(len(xb), 1, 1, 1, device=device).uniform_(-0.5, 0.5).exp()
                optimizer.zero_grad(set_to_none=True)
                output = model(xb)
                task, positivity = point_objective(output["raw_pred"], gt[batch_ix])
                density = posterior_nll(output, gt[batch_ix])
                loss = task + LOSS["canonical_nll"] * density + LOSS["positivity"] * positivity
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at epoch{epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                sums += np.array([task.item(), density.item(), positivity.item()]) * len(batch_ix)
                seen += len(batch_ix)
            schedule.step()
        metrics, arrays = evaluate(model, x[val_ix], gt[val_ix], args.batch)
        key = validation_key(metrics["reproduction"]["mean"], metrics["valid_fraction"])
        if key < best:
            best, best_epoch = key, epoch
            torch.save(model.state_dict(), out / "best.pt")
            np.savez_compressed(out / "best_validation.npz", **arrays)
            write_json(out / "best_metrics.json", {"epoch": epoch, **metrics})
        record = {
            "epoch": epoch,
            **training_summary(sums, seen),
            "validation": metrics,
            "seconds": time.perf_counter() - epoch_start,
            "best_epoch": best_epoch,
        }
        history.append(record)
        with (out / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if epoch % 5 == 0 or epoch == args.epochs:
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "epoch": epoch,
                        "val_repro": key if math.isfinite(key) else None,
                        "valid": metrics["valid_fraction"],
                        "best_epoch": best_epoch,
                        "elapsed_s": round(time.perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )
    torch.save(model.state_dict(), out / "last.pt")
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = {
        "status": "complete",
        "best_epoch": best_epoch,
        "best_validation_reproduction": best,
        "parameters": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
        "training_peak_allocated_mib_including_source_cache": torch.cuda.max_memory_allocated()
        / 2**20
        if device.type == "cuda"
        else None,
        "source_cache_mib": (x.numel() * x.element_size() + gt.numel() * gt.element_size()) / 2**20,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "checkpoint_bytes": (out / "best.pt").stat().st_size if best_epoch is not None else None,
        "checkpoint_sha256": sha256(out / "best.pt") if best_epoch is not None else None,
    }
    write_json(out / "result.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["direct", "diagonal", "frame"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default="data/processed/cc128")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    existed_before = Path(args.out).exists()
    try:
        train(args)
    except Exception as exc:
        out = Path(args.out)
        if (
            not existed_before
            and out.is_dir()
            and not (out / "result.json").exists()
            and not (out / "failure.json").exists()
        ):
            write_json(out / "failure.json", {"exception": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
