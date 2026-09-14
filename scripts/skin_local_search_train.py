"""Frozen TRAIN-only experiment. No legacy validation/test loaders are imported."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from skin_local_search_core import (
    greedy_ridge_indices,
    kernel_ridge_fit,
    rbf_features,
    ridge_solve,
)

from luma_skin_vision.color import delta_e00

CACHE_HASH = "d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0"
ROLE_HASH = "7fa14adc41525a868d737539571c9d8cb9e954a680e457f828d7aab35a2bc064"
METHODS = ("ridge", "krr", "random_rbf", "guided_rbf", "mlp")
SEEDS = (17, 29, 43)
GRIDS = {m: (0.1, 1.0, 10.0) for m in METHODS}
GRIDS["mlp"] = (0.0003, 0.001, 0.003)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def weights_for(person, site):
    w = np.zeros(len(person), dtype=np.float64)
    for p in np.unique(person):
        ps = np.unique(site[person == p])
        for s in ps:
            mask = (person == p) & (site == s)
            w[mask] = 1.0 / (len(ps) * mask.sum())
    return w / w.mean()


def roles(person, camera):
    fixed = np.random.default_rng(20260911)
    held = []
    for c, n, expected in (("SLR", 2, 8), ("ipod", 4, 16)):
        people = np.unique(person[camera == c])
        if len(people) != expected:
            raise ValueError("Unexpected original TRAIN population")
        held.extend(fixed.permutation(people)[:n])
    b = np.isin(person, held)
    a = ~b
    digest = hashlib.sha256(np.packbits(a).tobytes() + np.packbits(b).tobytes()).hexdigest()
    if digest != ROLE_HASH:
        raise ValueError("Historical mixed role digest mismatch")
    result = {"mixed": (a, b), "slr_to_ipod": (camera == "SLR", camera == "ipod"),
              "ipod_to_slr": (camera == "ipod", camera == "SLR")}
    for fit, held in result.values():
        if np.any(fit & held) or not np.all(fit | held):
            raise ValueError("Invalid outer row roles")
        if set(person[fit]) & set(person[held]):
            raise ValueError("Outer person leakage")
    return result


def folds_for(person, camera):
    rng = np.random.default_rng(917031)
    folds = np.full(len(person), -1, dtype=np.int64)
    for c in sorted(np.unique(camera)):
        ids = rng.permutation(np.unique(person[camera == c]))
        for k, group in enumerate(np.array_split(ids, 3)):
            folds[np.isin(person, group)] = k
    if np.any(folds < 0):
        raise ValueError("Unassigned fold")
    for k in range(3):
        if set(person[folds == k]) & set(person[folds != k]):
            raise ValueError("Inner person leakage")
    return folds


def metrics(pred, target, person, site, camera=None):
    de = delta_e00(pred, target)
    person_values = [float(de[person == p].mean()) for p in np.unique(person)]
    site_person = []
    for p in np.unique(person):
        site_person.append(np.mean([de[(person == p) & (site == s)].mean()
                                    for s in np.unique(site[person == p])]))
    result = {"person_mean": float(np.mean(person_values)), "image_mean": float(de.mean()),
              "site_person_mean": float(np.mean(site_person)), "median": float(np.median(de)),
              "p90": float(np.quantile(de, .9)), "gt5": float(np.mean(de > 5)),
              "gt10": float(np.mean(de > 10)), "n_images": len(de),
              "n_people": len(person_values)}
    if camera is not None:
        result["camera"] = {str(c): metrics(pred[camera == c], target[camera == c],
                                           person[camera == c], site[camera == c])
                            for c in np.unique(camera)}
    return result


def center_indices(x, person, site, seed, count=256):
    """One geometric medoid per site first; round robin gives people coverage."""
    rng = np.random.default_rng(seed)
    groups = []
    for p in rng.permutation(np.unique(person)):
        medoids = []
        for s in rng.permutation(np.unique(site[person == p])):
            idx = np.flatnonzero((person == p) & (site == s))
            distance = ((x[idx] - x[idx].mean(axis=0)) ** 2).sum(axis=1)
            medoids.append(int(idx[np.argmin(distance)]))
        groups.append(medoids)
    chosen = []
    while any(groups) and len(chosen) < count:
        for group in groups:
            if group and len(chosen) < count:
                chosen.append(group.pop(0))
    if len(chosen) < min(count, len(x)):
        remaining = rng.permutation(np.setdiff1d(np.arange(len(x)), chosen))
        chosen.extend(remaining[:count - len(chosen)].tolist())
    return np.asarray(chosen, dtype=np.int64)


def tensor(a, device, dtype=torch.float64):
    return torch.as_tensor(a, dtype=dtype, device=device)


def predict(model, x):
    """Canonical deployed float32 payload, CPU NumPy implementation."""
    z = (np.asarray(x, dtype=np.float32) - model["x_mean"]) / model["x_std"]
    method = str(model["method"])
    if method == "mlp":
        h = z @ model["hidden_w"].T + model["hidden_b"]
        h = h / (1 + np.exp(-np.clip(h, -80, 80)))
        out = h @ model["out_w"].T + model["out_b"] + z @ model["skip_w"].T
    elif method == "ridge":
        out = np.column_stack((np.ones(len(z), dtype=np.float32), z)) @ model["beta"]
    else:
        c = model["centers"]
        # The squared-distance identity avoids N x centers x 36 temporary arrays.
        d2 = np.maximum((z * z).sum(1)[:, None] + (c * c).sum(1)[None, :] - 2 * z @ c.T, 0) / z.shape[1]
        h = np.exp(-.5 * d2 / model["widths"] ** 2)
        if method == "krr":
            out = h @ model["beta"]
        else:
            out = np.column_stack((np.ones(len(z), dtype=np.float32), z, h)) @ model["beta"]
    return out * model["y_std"] + model["y_mean"]


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(36, 64)
        self.out = nn.Linear(64, 3)
        self.skip = nn.Linear(36, 3, bias=False)

    def forward(self, x):
        return self.out(torch.nn.functional.silu(self.hidden(x))) + self.skip(x)


def fit_model(x, y, person, site, method, parameter, seed, device, steps=512):
    synchronize(device)
    started = time.perf_counter()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    xm, xs = x.mean(0), np.maximum(x.std(0), 1e-6)
    ym, ys = y.mean(0), np.maximum(y.std(0), 1e-6)
    xn, yn = (x - xm) / xs, (y - ym) / ys
    wn = weights_for(person, site)
    model = {"method": np.asarray(method), "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys}
    aux = {}
    if method == "mlp":
        torch.manual_seed(seed)
        net = SmallMLP().to(device)
        xt, yt, wt = tensor(xn, device, torch.float32), tensor(yn, device, torch.float32), tensor(wn, device, torch.float32)
        optimizer = torch.optim.AdamW(net.parameters(), lr=parameter, weight_decay=.01)
        generator = torch.Generator(device=device).manual_seed(seed + 9001)
        # Sample all minibatches once so Python/RNG overhead is not charged per step.
        indices = torch.multinomial(wt, steps * 64, replacement=True, generator=generator).reshape(steps, 64)
        losses = []
        for step in range(steps):
            idx = indices[step]
            optimizer.zero_grad(set_to_none=True)
            loss = (net(xt[idx]) - yt[idx]).square().mean()
            loss.backward()
            optimizer.step()
            if step + 1 in (64, 128, 256, 512):
                losses.append([step + 1, float(loss.detach())])
        state = net.state_dict()
        for key, source in (("hidden_w", "hidden.weight"), ("hidden_b", "hidden.bias"),
                            ("out_w", "out.weight"), ("out_b", "out.bias"), ("skip_w", "skip.weight")):
            model[key] = state[source].detach().cpu().numpy()
        aux["sampled_minibatch_loss_trace"] = losses
        aux["optimizer_updates"] = steps
    else:
        xt, yt, wt = tensor(xn, device), tensor(yn, device), tensor(wn, device)
        base = torch.cat((torch.ones((len(xt), 1), dtype=xt.dtype, device=device), xt), dim=1)
        with torch.no_grad():
            if method == "ridge":
                beta = ridge_solve(base, yt, parameter, wt)
            else:
                dist = torch.pdist(xt) / np.sqrt(xt.shape[1])
                width = max(float(dist[dist > 1e-10].median()), 1e-6)
                if method == "krr":
                    beta = kernel_ridge_fit(xt, yt, width, parameter, wt)
                    model["centers"] = xn
                    model["widths"] = np.asarray(width)
                else:
                    ids = center_indices(xn, person, site, seed)
                    centers = np.repeat(xn[ids], 3, axis=0)
                    widths = np.tile(np.asarray([.5, 1., 2.]) * width, len(ids))
                    z = rbf_features(xt, tensor(centers, device), tensor(widths, device))
                    count = min(64, z.shape[1])
                    if method == "guided_rbf":
                        selected, gains = greedy_ridge_indices(base, z, yt, parameter, count, wt)
                        aux["exact_objective_gains"] = [float(g) for g in gains]
                    else:
                        selected = np.random.default_rng(seed + 101).choice(z.shape[1], count, replace=False).tolist()
                    beta = ridge_solve(torch.cat((base, z[:, selected]), dim=1), yt, parameter, wt)
                    model["centers"] = centers[selected]
                    model["widths"] = widths[selected]
                    aux["candidate_atoms"] = int(z.shape[1])
                    aux["selected_atoms"] = len(selected)
                    aux["unique_centers"] = len(set(int(i) // 3 for i in selected))
                aux["base_width"] = width
            model["beta"] = beta.detach().cpu().numpy()
    synchronize(device)
    fit_seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else 0
    model = {k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in model.items()}
    pred = predict(model, x)
    sse = float(np.sum(wn[:, None] * ((pred - y) / ys) ** 2))
    aux.update({"fit_seconds": fit_seconds, "peak_allocated_bytes": peak,
                "weighted_train_normalized_sse": sse,
                "numeric_scalars": sum(v.size for v in model.values() if v.dtype.kind == "f"),
                "numeric_bytes": sum(v.nbytes for v in model.values() if v.dtype.kind == "f")})
    return model, aux


def load_model(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def role_summary(person, camera, fit, held, folds):
    return {"fit_images": int(fit.sum()), "held_images": int(held.sum()),
            "fit_people": len(np.unique(person[fit])), "held_people": len(np.unique(person[held])),
            "inner": [{"fold": k, "images": int(np.sum(folds == k)),
                       "people": len(np.unique(person[fit][folds == k]))} for k in range(3)]}


def lock_sources(args, run):
    source_files = [Path(__file__), ROOT / "scripts/skin_local_search_core.py",
                    ROOT / "src/luma_skin_vision/color.py", ROOT / "docs/research/skin_local_search_v1_protocol.md"]
    binding = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    if sha(args.cache) != CACHE_HASH:
        raise ValueError("Unauthorized cache or changed data")
    lock = {"cache_sha256": CACHE_HASH, "source_hashes": binding, "device": args.device,
            "steps": 512, "seeds": list(SEEDS), "grids": GRIDS,
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "threads": torch.get_num_threads(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    dest = run / "source_lock.json"
    # JSON normalize tuples before comparing a resumed run.
    lock = json.loads(json.dumps(lock))
    if dest.exists():
        if json.loads(dest.read_text(encoding="utf-8")) != lock:
            raise ValueError("Frozen run binding changed: use a new run directory")
    else:
        write_json(dest, lock)
    return lock


def fit_phase(data, run, device):
    x = data["color"].astype(np.float64)
    person, site, camera = data["patient"], data["site"], data["device"]
    outer_roles = roles(person, camera)
    selected = {}
    all_records = []
    for protocol, (fit, held) in outer_roles.items():
        # Only fitting labels are used during this protocol's selection/refit phase.
        xx, yy, pp, ss, cc = x[fit], data["target"][fit], person[fit], site[fit], camera[fit]
        folds = folds_for(pp, cc)
        folder = run / protocol
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "roles.npz", fit=fit, held=held, folds=folds)
        write_json(folder / "role_summary.json", role_summary(person, camera, fit, held, folds))
        choices = {}
        for method in METHODS:
            seed_list = (17,) if method in ("ridge", "krr") else SEEDS
            scores = []
            for config, parameter in enumerate(GRIDS[method]):
                oof = {seed: np.empty_like(yy) for seed in seed_list}
                for seed in seed_list:
                    for fold in range(3):
                        tag = f"{method}_c{config}_s{seed}_f{fold}"
                        receipt = folder / "inner" / (tag + ".json")
                        artifact = receipt.with_suffix(".npz")
                        train, valid = folds != fold, folds == fold
                        if receipt.exists() and artifact.exists():
                            record = json.loads(receipt.read_text(encoding="utf-8"))
                            if sha(artifact) != record["artifact_sha256"]:
                                raise ValueError("Cached inner model changed")
                            model = load_model(artifact)
                        else:
                            model, aux = fit_model(xx[train], yy[train], pp[train], ss[train],
                                                   method, parameter, seed, device)
                            artifact.parent.mkdir(parents=True, exist_ok=True)
                            np.savez(artifact, **model)
                            pred = predict(model, xx[valid])
                            record = {"protocol": protocol, "method": method, "parameter": parameter,
                                      "seed": seed, "fold": fold, **aux,
                                      "inner_metrics": metrics(pred, yy[valid], pp[valid], ss[valid]),
                                      "artifact_sha256": sha(artifact), "serialized_bytes": artifact.stat().st_size}
                            write_json(receipt, record)
                        oof[seed][valid] = predict(model, xx[valid])
                        all_records.append(record)
                seed_scores = [metrics(oof[s], yy, pp, ss)["person_mean"] for s in seed_list]
                score = float(np.mean(seed_scores))
                scores.append(score)
                print(json.dumps({"stage": "inner", "protocol": protocol, "method": method,
                                  "parameter": parameter, "person_delta_e": score}, ensure_ascii=False), flush=True)
            best = int(np.argmin(scores))
            choices[method] = {"parameter": GRIDS[method][best], "inner_person_mean": scores[best],
                               "all_config_scores": scores, "seeds": list(seed_list)}
        write_json(folder / "selection.json", choices)
        selected[protocol] = choices
        for method, choice in choices.items():
            for seed in choice["seeds"]:
                tag = f"{method}_s{seed}"
                receipt = folder / "final" / (tag + ".json")
                artifact = receipt.with_suffix(".npz")
                if receipt.exists() and artifact.exists():
                    continue
                model, aux = fit_model(xx, yy, pp, ss, method, choice["parameter"], seed, device)
                artifact.parent.mkdir(parents=True, exist_ok=True)
                np.savez(artifact, **model)
                write_json(receipt, {"protocol": protocol, "method": method, "seed": seed,
                                     "parameter": choice["parameter"], **aux,
                                     "train_metrics": metrics(predict(model, xx), yy, pp, ss),
                                     "artifact_sha256": sha(artifact), "serialized_bytes": artifact.stat().st_size})
    write_json(run / "inner_records.json", all_records)
    write_json(run / "frozen_selections.json", selected)
    print("All selections and final weights frozen; outer evaluation has not run.", flush=True)


def cpu_latency(model, x):
    row = x[:1]
    for _ in range(30):
        predict(model, row)
    timings = []
    for _ in range(200):
        start = time.perf_counter_ns()
        predict(model, row)
        timings.append((time.perf_counter_ns() - start) / 1000)
    return {"batch1_cpu_us_p50": float(np.median(timings)), "batch1_cpu_us_p95": float(np.quantile(timings, .95))}


def evaluation_phase(data, run):
    frozen = run / "frozen_selections.json"
    if not frozen.exists():
        raise ValueError("Complete all fit selections before evaluating any outer roles")
    selection = json.loads(frozen.read_text(encoding="utf-8"))
    records = []
    for protocol, methods in selection.items():
        folder = run / protocol
        with np.load(folder / "roles.npz", allow_pickle=False) as role_data:
            held = role_data["held"]
        x, y = data["color"][held], data["target"][held]
        p, s, c = data["patient"][held], data["site"][held], data["device"][held]
        for method, choice in methods.items():
            for seed in choice["seeds"]:
                path = folder / "final" / f"{method}_s{seed}.npz"
                fit_record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                if sha(path) != fit_record["artifact_sha256"]:
                    raise ValueError("Final model changed after fitting")
                model = load_model(path)
                pred = predict(model, x)
                record = {**fit_record, "outer_metrics": metrics(pred, y, p, s, c),
                          "inner_person_mean": choice["inner_person_mean"], **cpu_latency(model, x)}
                records.append(record)
                np.savez(folder / "final" / f"{method}_s{seed}_outer.npz", prediction=pred, target=y,
                         person=p, site=s, camera=c)
                print(json.dumps({"stage": "outer", "protocol": protocol, "method": method,
                                  "seed": seed, "person_delta_e": record["outer_metrics"]["person_mean"],
                                  "image_delta_e": record["outer_metrics"]["image_mean"],
                                  "fit_seconds": record["fit_seconds"], "numeric_bytes": record["numeric_bytes"]}), flush=True)
    result = {"evidence": "Exploratory original-TRAIN-only, overlapping protocols; old outer people reused",
              "frozen_selection_sha256": sha(frozen), "models": records}
    write_json(run / "evaluation.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=ROOT / "experiments/runs/skin_local_search_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=("fit", "evaluate"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.run.mkdir(parents=True, exist_ok=True)
    lock_sources(args, args.run)
    # NPZ handle is confined to the explicit, hash-verified original TRAIN file.
    with np.load(args.cache, allow_pickle=False) as archive:
        data = {k: archive[k] for k in ("color", "target", "patient", "site", "device")}
    if data["color"].shape != (966, 36) or not np.isfinite(data["target"]).all():
        raise ValueError("Invalid data")
    started = time.perf_counter()
    if args.stage == "fit":
        fit_phase(data, args.run, args.device)
    else:
        evaluation_phase(data, args.run)
    write_json(args.run / f"{args.stage}_completion.json", {"seconds_this_invocation": time.perf_counter() - started,
                                                            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


if __name__ == "__main__":
    main()
