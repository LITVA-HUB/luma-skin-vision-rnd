"""Source-development evidence routing screen; never reads test/camera GT rows."""

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
from cc_v3_experiment import DATA_HASHES, source_rows

from luma_skin_vision.cc.core import angular, reproduction, summarize
from luma_skin_vision.cc.v2_experiment import source_snapshot
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT = Path(__file__).resolve().parents[1]
LOSS_WEIGHTS = {"point_degrees": 1., "angular_mse_degrees": .05, "sin2_mse": 25.}


def sample_actions(point, generator=None):
    """33 actions; sampling sees no GT. Proposal is detached, RNG independent."""
    options = {"device": point.device, "dtype": point.dtype, "generator": generator}
    local = point.detach()[:, None] + (torch.rand(len(point), 16, 2, **options) * 2 - 1) * .4
    global_ = (torch.rand(len(point), 16, 2, **options) * 2 - 1) * 1.5
    return torch.cat([point.detach()[:, None], local, global_], 1).clamp(-2, 2)


def field_weight(epoch):
    return min(1., max(0., (epoch - 20) / 20))


def action_rgb(action):
    logits = np.stack([action[..., 0], np.zeros(action.shape[:-1]), action[..., 1]], -1)
    value = np.exp(logits - logits.max(-1, keepdims=True))
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def risk_summary(errors, risk):
    errors, risk = np.asarray(errors), np.asarray(risk)
    if errors.ndim != 1 or errors.shape != risk.shape or len(errors) == 0 or not np.isfinite([errors, risk]).all():
        raise ValueError("Nonempty finite aligned errors/risk required")
    order = np.argsort(risk, kind="stable")
    curve = np.cumsum(errors[order]) / np.arange(1, len(order) + 1)
    fixed = {}
    for coverage in (100, 95, 90, 80, 70, 60):
        k = max(1, int(len(order) * coverage / 100))
        accepted = errors[order[:k]]
        fixed[str(coverage)] = {"accepted": k, "coverage": k / len(order), "mean": float(accepted.mean()),
                                "p95": float(np.percentile(accepted, 95)), "over10": int((accepted > 10).sum())}
    return {"fixed": fixed, "curve": curve.tolist(), "aurc_discrete_mean": float(curve.mean()),
            "status": "Raw model ranking on reused development validation; uncalibrated"}


def refinement_summary(errors, risks):
    result = {}
    for left, right in ((0, 1), (1, 3), (0, 3)):
        delta = errors[:, right] - errors[:, left]
        result[f"{left+1}_to_{right+1}"] = {
            "mean_true_delta": float(delta.mean()),
            "true_error_worsened": int((delta > 1e-6).sum()),
            "true_error_improved": int((delta < -1e-6).sum()),
            "predicted_risk_increased": int((risks[:, right] > risks[:, left] + 1e-5).sum()),
        }
    return result


def oracle_errors(point, trajectory, gt):
    """Oracle diagnostic over the actual policy's 103 evaluated candidates."""
    offsets = np.array([(a, b) for a in np.linspace(-1, 1, 5) for b in np.linspace(-1, 1, 5)])
    candidates = []
    for step, radius in enumerate((.24, .06, .03, .015)):
        center = point if step == 0 else trajectory[:, step - 1]
        candidates.append(np.clip(center[:, None] + radius * offsets, -2, 2))
        if step:
            candidates.append(point[:, None])
    candidate = np.concatenate(candidates, 1)
    pred = action_rgb(candidate)
    target = np.broadcast_to(gt[:, None], pred.shape)
    errors = reproduction(pred.reshape(-1, 3), target.reshape(-1, 3)).reshape(len(gt), -1)
    return errors[:, :51].min(-1), errors.min(-1)


@torch.no_grad()
def evaluate(model, x, gt, batch):
    model.eval()
    arrays = {key: [] for key in ("point_action", "trajectory_actions", "trajectory_risk", "valid")}
    observed_query_counts = set()
    for start in range(0, len(x), batch):
        cache = model.encode(x[start:start + batch])
        selected = model.select(cache, steps=4)
        observed_query_counts.add(selected["query_count"])
        arrays["point_action"].append(cache["point_action"].cpu().numpy())
        for key in ("trajectory_actions", "trajectory_risk", "valid"):
            arrays[key].append(selected[key].cpu().numpy())
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    label = gt.cpu().numpy().astype(np.float64)
    point = action_rgb(arrays["point_action"].astype(np.float64))
    trajectory = action_rgb(arrays["trajectory_actions"].astype(np.float64))
    arrays["base_pred"] = point
    arrays["base_reproduction"] = reproduction(point, label)
    arrays["trajectory_reproduction"] = np.stack([reproduction(trajectory[:, step], label) for step in range(4)], -1)
    arrays["trajectory_recovery"] = np.stack([angular(trajectory[:, step], label) for step in range(4)], -1)
    arrays["oracle2"], arrays["oracle4"] = oracle_errors(arrays["point_action"], arrays["trajectory_actions"], label)
    arrays["pred"] = trajectory[:, 1]
    arrays["reproduction"] = arrays["trajectory_reproduction"][:, 1]
    arrays["risk"] = arrays["trajectory_risk"][:, 1]
    if observed_query_counts != {103}:
        raise ValueError("Four-stage policy changed its observed query budget")
    metrics = {"base_reproduction": summarize(arrays["base_reproduction"]),
               "observed_four_stage_queries": sorted(observed_query_counts),
               "valid_fraction": float(arrays["valid"].mean()), "stages": {},
               "refinement": refinement_summary(arrays["trajectory_reproduction"], arrays["trajectory_risk"]),
               "oracle2_mean": float(arrays["oracle2"].mean()), "oracle4_mean": float(arrays["oracle4"].mean())}
    for stage in (1, 2, 4):
        err = arrays["trajectory_reproduction"][:, stage - 1]
        metrics["stages"][str(stage)] = {"reproduction": summarize(err),
            "recovery": summarize(arrays["trajectory_recovery"][:, stage - 1]),
            "risk": risk_summary(err, arrays["trajectory_risk"][:, stage - 1])}
    return metrics, arrays


def train(args):
    from cc_v4_model import CorrectionEvidenceNet, analytic_costs

    if args.mode not in {"posterior", "action", "transport"} or args.epochs < 1 or args.batch < 2 or args.gradient_weight < 0:
        raise ValueError("Invalid experimental mode/budget")
    out, data = Path(args.out).resolve(), Path(args.data).resolve()
    if out.exists():
        raise FileExistsError("Immutable run output already exists")
    actual = {name: sha256(data / name) for name in DATA_HASHES}
    if actual != DATA_HASHES:
        raise ValueError("Frozen source cache changed")
    rows = json.loads((data / "cube_manifest.json").read_text(encoding="utf-8"))
    selected, train_ix, val_ix = source_rows(rows)
    if (len(rows), len(train_ix), len(val_ix)) != (2234, 1126, 119):
        raise ValueError("Unexpected source roles")
    out.mkdir(parents=True)
    identity, snapshot = source_snapshot(out)
    scripts = {}
    for name in ("cc_v4_experiment.py", "cc_v4_model.py", "cc_v4_geometry.py", "cc_v3_experiment.py", "cc_v3_model.py", "cc_v2_statistics.py"):
        dest = out / "source_snapshot" / "scripts" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "scripts" / name, dest)
        scripts[name] = sha256(dest)
    for name in ("cc_v4_spec.md", "cc_v4_source_lock.md"):
        shutil.copyfile(ROOT / "docs/research" / name, out / name)
    write_json(out / "config.json", {
        "schema": "cc-v4-source-screen-1", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "is_frozen_primary": (args.epochs, args.batch, args.seed, args.gradient_weight) == (120, 32, 17, 0.),
        "arguments": vars(args), "data_hashes": actual, "source_identity": identity, "snapshot": snapshot,
        "scripts_sha256": scripts, "train_ids": [rows[selected[i]]["id"] for i in train_ix],
        "validation_ids": [rows[selected[i]]["id"] for i in val_ix], "loss_weights": LOSS_WEIGHTS,
        "warmup": "point-only epochs1-20; field fraction (epoch-20)/20 clamped0..1 thereafter",
        "action_sampling": "33 GT-independent actions: detachedpoint,16 point+U[-.4,.4],16 U[-1.5,1.5];clamp[-2,2]",
        "augmentation": "Common exposure exp(U[-.5,.5]), independent horizontal flip p.5; no channel/color perturbation",
        "optimizer": "AdamW .001 wd.0001 cosinefloor.00002 clip5; FP32 noAMP/noTF32",
        "checkpoint_selection": "Lowest step2 mean reproduction on development val at valid>=.99; also preserve independent best direct-point epoch",
        "test_time_compute": "One cached encoder; steps1/2/4=25/51/103 router queries; GT never enters policy",
        "limitations": "Reused source-development validation, one seed, no calibrated error head; no independent test/camera/skin result",
        "torch": torch.__version__, "inactive_router_weights": 128 if args.mode == "posterior" else 0,
    })
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
    x = torch.from_numpy(read_npz_rows(data / "cube.npz", "images", selected, 2234).astype(np.float32)).to(device)
    gt = torch.from_numpy(read_npz_rows(data / "cube.npz", "gt", selected, 2234).astype(np.float32)).to(device)
    if not torch.isfinite(x).all() or (x < 0).any() or not torch.isfinite(gt).all() or (gt <= 0).any():
        raise ValueError("Invalid fitting rows")
    model = CorrectionEvidenceNet(mode=args.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.00002)
    train_ix, val_ix = torch.as_tensor(train_ix, device=device), torch.as_tensor(val_ix, device=device)
    best, best_base, best_epoch, base_epoch = math.inf, math.inf, None, None
    started = time.perf_counter()
    for epoch in range(args.epochs + 1):
        sums, seen = np.zeros(4), 0
        epoch_started = time.perf_counter()
        if epoch:
            model.train()
            order = train_ix[torch.randperm(len(train_ix), device=device)]
            for ix in order.split(args.batch):
                xb = x[ix]
                flip = torch.rand(len(ix), 1, 1, 1, device=device) < .5
                xb = torch.where(flip, xb.flip(-1), xb)
                xb = xb * torch.empty(len(ix), 1, 1, 1, device=device).uniform_(-.5, .5).exp()
                loggt = gt[ix].log()
                loggt = loggt[:, [0, 2]] - loggt[:, 1:2]
                optimizer.zero_grad(set_to_none=True)
                cache = model.encode(xb)
                point = analytic_costs(loggt[:, None], cache["point_action"][:, None])["angular"].mean()
                actions = sample_actions(cache["point_action"])
                ang_loss = point.new_zeros(())
                cost_loss, grad_loss = ang_loss, ang_loss
                weight = field_weight(epoch)
                if weight:
                    if args.gradient_weight:
                        actions.requires_grad_(True)
                    field = model.query(cache, actions)
                    targets = analytic_costs(loggt[:, None], actions)
                    ang_loss = (field["angular_risk"] - targets["angular"].squeeze(-1).detach()).square().mean()
                    cost_loss = (field["sin2_risk"] - targets["sin2"].squeeze(-1).detach()).square().mean()
                    if args.gradient_weight:
                        desired = torch.autograd.grad(targets["sin2"].sum(), actions, retain_graph=True)[0].detach()
                        actual_gradient = torch.autograd.grad(field["sin2_risk"].sum(), actions, create_graph=True)[0]
                        grad_loss = (actual_gradient - desired).square().mean()
                loss = point + weight * (.05 * ang_loss + 25 * cost_loss + args.gradient_weight * grad_loss)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss epoch{epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                sums += np.array([point.item(), ang_loss.item(), cost_loss.item(), grad_loss.item()]) * len(ix)
                seen += len(ix)
            schedule.step()
        metrics, arrays = evaluate(model, x[val_ix], gt[val_ix], args.batch)
        valid = metrics["valid_fraction"] >= .99
        key = metrics["stages"]["2"]["reproduction"]["mean"] if valid else math.inf
        base_key = metrics["base_reproduction"]["mean"] if valid else math.inf
        if key < best:
            best, best_epoch = key, epoch
            torch.save(model.state_dict(), out / "best.pt")
            np.savez_compressed(out / "best_validation.npz", **arrays)
            write_json(out / "best_metrics.json", {"epoch": epoch, **metrics})
        if base_key < best_base:
            best_base, base_epoch = base_key, epoch
            torch.save(model.state_dict(), out / "best_point.pt")
            np.savez_compressed(out / "best_point_validation.npz", **arrays)
            write_json(out / "best_point_metrics.json", {"epoch": epoch, **metrics})
        record = {"epoch": epoch, "training": dict(zip(("point", "angular_mse", "sin2_mse", "gradient_mse"),
                       [float(v / seen) for v in sums] if seen else [None] * 4)),
                  "field_weight": field_weight(epoch), "validation": metrics,
                  "seconds": time.perf_counter() - epoch_started, "best_epoch": best_epoch}
        with (out / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if epoch % 5 == 0 or epoch == args.epochs:
            print(json.dumps({"mode": args.mode, "epoch": epoch, "val_repro": key,
                              "base_repro": base_key, "best_epoch": best_epoch,
                              "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
    torch.save(model.state_dict(), out / "last.pt")
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = {"status": "complete", "best_epoch": best_epoch, "best_reproduction": best,
              "best_point_epoch": base_epoch, "best_point_reproduction": best_base,
              "parameters": sum(p.numel() for p in model.parameters()), "elapsed_seconds": time.perf_counter() - started,
              "training_peak_allocated_mib_including_cache": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
              "source_cache_mib": (x.numel() * x.element_size() + gt.numel() * gt.element_size()) / 2**20,
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
              "checkpoint_bytes": (out / "best.pt").stat().st_size, "checkpoint_sha256": sha256(out / "best.pt")}
    write_json(out / "result.json", result)
    manifest = {str(path.relative_to(out)).replace("\\", "/"): sha256(path)
                for path in sorted(out.rglob("*")) if path.is_file()}
    write_json(out / "artifact_manifest.json", {"sha256": manifest,
        "note": "All run files before this manifest, including point-selected/last checkpoints, predictions, histories, configs and source/lock snapshots."})
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["posterior", "action", "transport"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default="data/processed/cc128")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--gradient-weight", type=float, default=0.)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    existed = Path(args.out).exists()
    try:
        train(args)
    except Exception as exc:
        out = Path(args.out)
        if not existed and out.is_dir() and not (out / "result.json").exists() and not (out / "failure.json").exists():
            write_json(out / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
