"""Independent analytic geometry, augmented-SVD and selection audit for P."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from chromaseed_kernel_audit import direct_kernel, independent_predict, js, norm, nz, unpack
from chromaseed_perceptual_reference import analytic_tensor, augmented_svd
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json

from luma_skin_vision.color import delta_e00

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("norm_mse", "constant_de2", "local_de2", "local_irls", "midpoint_irls")
SEEDS, ALPHAS, WIDTHS = (17, 29, 43), (.1, 1., 10.), (.5, 1., 2.)


def name_for(family, seed, wi, ai, step):
    family = "norm_mse" if family.endswith("irls") and step == 0 else family
    return f"{family}_s{seed}_w{wi}_a{ai}_t{step}"


def balanced(person, site):
    weight = np.empty(len(person))
    people = np.unique(person)
    for p in people:
        sites = np.unique(site[person == p])
        for s in sites:
            mask = (person == p) & (site == s)
            weight[mask] = 1 / (len(people) * len(sites) * mask.sum())
    return weight / weight.mean()


def reference_readouts(model, x, y, weights, family, alpha, steps):
    standardized = norm(model, x)
    columns = direct_kernel(standardized, model["centers"], float(model["width"]))
    sub = direct_kernel(model["centers"], model["centers"], float(model["width"]))
    u, spectrum, _ = np.linalg.svd(sub, full_matrices=False)
    keep = spectrum > spectrum.max() * 1e-8
    whitening = u[:, keep] / np.sqrt(spectrum[keep])
    z = columns @ whitening
    target = (y.astype(np.float64) - model["y_mean"]) / model["y_std"]
    identity = np.broadcast_to(np.eye(3), (len(y), 3, 3))
    theta = augmented_svd(z, target, identity, weights, alpha)
    checkpoints = {0: (whitening @ theta).astype(np.float32)}
    if family == "norm_mse" or steps == 0:
        return checkpoints, []
    raw = analytic_tensor(y)
    std = model["y_std"].astype(np.float64)
    metric = raw * std[None, :, None] * std[None, None, :]
    scale = float(np.average(np.trace(metric, axis1=1, axis2=2), weights=weights) / 3)
    metric /= scale
    if family in ("constant_de2", "local_de2"):
        if family == "constant_de2":
            metric = np.broadcast_to(np.average(metric, axis=0, weights=weights), metric.shape)
        theta = augmented_svd(z, target, metric, weights, alpha)
        return {1: (whitening @ theta).astype(np.float32)}, []

    def loss(beta):
        residual = (z @ beta - target) * std
        distance2 = (np.einsum("ni,nij,nj->n", residual, raw, residual) if family == "local_irls"
                     else delta_e00(y + residual, y) ** 2)
        return float(weights @ (4 * (np.sqrt(np.maximum(distance2, 0) + 4) - 2)) / scale + alpha * np.sum(beta ** 2))

    value = loss(theta)
    history = [value]
    stopped = False
    for step in range(1, steps + 1):
        if not stopped:
            residual = (z @ theta - target) * std
            geometry = analytic_tensor(y + .5 * residual) if family == "midpoint_irls" else raw
            normalized = geometry * std[None, :, None] * std[None, None, :] / scale
            d2 = np.einsum("ni,nij,nj->n", residual, geometry, residual)
            factor = 2 / np.sqrt(np.maximum(d2, 0) + 4)
            proposal = augmented_svd(z, target, normalized, weights * factor, alpha)
            accepted = False
            for length in 2. ** -np.arange(7):
                trial = theta + length * (proposal - theta)
                new = loss(trial)
                if np.isfinite(new) and new < value:
                    theta, value, accepted = trial, new, True
                    break
            stopped = not accepted
        history.append(value)
        if step in (1, 4, 16, steps):
            checkpoints[step] = (whitening @ theta).astype(np.float32)
    return checkpoints, history


def audit(run, cache, parent, output):
    started = time.perf_counter()
    lock = js(run / "source_lock.json")
    if sha(cache) != CACHE_HASH or lock["cache_sha256"] != CACHE_HASH:
        raise ValueError("cache changed")
    for rel, expected in lock["sources"].items():
        assert sha(ROOT / rel) == expected, rel
    for rel, expected in lock["parent_bindings"].items():
        assert sha(parent / rel) == expected, rel
    with np.load(cache, allow_pickle=False) as z:
        data = {k: z[k] for k in ("color", "target", "patient", "site", "device")}
    source = sha(run / "source_lock.json")
    selection = js(run / "selections.json")
    results = js(run / "results.json")
    assert selection["source_lock_sha256"] == source and results["selection_sha256"] == sha(run / "selections.json")
    checks = {"banks": 0, "stored_readouts": 0, "oof_query_rows": 0, "choices": 0, "iterative_trajectories": 0,
              "selected_models": 0, "selected_query_rows": 0, "independent_refits": 0, "positive_step_probe_readouts": 0}
    maxima = {"oof_prediction_drift": 0., "selected_prediction_drift": 0., "svd_prediction_drift": 0.,
              "positive_step_probe_drift": 0., "objective_increase": 0., "normal_equation_residual": 0.}
    probe_records, final_scores = [], {}
    for role, (fit, held) in roles(data["patient"], data["device"]).items():
        base_rows = np.flatnonzero(fit)
        folds = folds_for(data["patient"][fit], data["device"][fit])
        oof_files = []
        for fold in (0, 1, 2, None):
            directory = run / ("final" if fold is None else "inner") / role / ("bank" if fold is None else f"fold{fold}")
            receipt = js(directory / "receipt.json")
            assert receipt["source_lock_sha256"] == source
            for filename, expected in receipt["files"].items():
                assert sha(directory / filename) == expected
            rows = base_rows if fold is None else base_rows[folds != fold]
            query_rows = np.flatnonzero(held) if fold is None else base_rows[folds == fold]
            assert not set(data["patient"][rows]) & set(data["patient"][query_rows])
            x, y = data["color"][rows], data["target"][rows]
            arrays = nz(directory / "models.npz")
            expected_prep = {"x_mean": x.astype(float).mean(0).astype(np.float32), "x_std": np.maximum(x.astype(float).std(0), 1e-6).astype(np.float32),
                             "y_mean": y.mean(0).astype(np.float32), "y_std": np.maximum(y.std(0), 1e-6).astype(np.float32)}
            for name in receipt["models"]:
                model = unpack(arrays, name)
                for field, value in expected_prep.items():
                    np.testing.assert_array_equal(model[field], value)
                checks["stored_readouts"] += 1
            for operation in receipt["operations"]:
                if "normal_equation_relative_residual" in operation:
                    maxima["normal_equation_residual"] = max(maxima["normal_equation_residual"], operation["normal_equation_relative_residual"])
                if operation["kind"] == "iterative":
                    values = [h["objective"] for h in operation["trajectory"]]
                    maxima["objective_increase"] = max(maxima["objective_increase"], float(np.max(np.diff(values))))
                    assert all(np.isfinite(values)) and np.max(np.diff(values)) <= 1e-9
                    for item in operation["trajectory"]:
                        maxima["normal_equation_residual"] = max(maxima["normal_equation_residual"], item.get("normal_equation_relative_residual", 0.))
                    checks["iterative_trajectories"] += 1
            checks["banks"] += 1
            if fold is None:
                continue
            saved = nz(directory / "oof.npz")
            np.testing.assert_array_equal(saved["row_indices"], query_rows)
            kernels = {}
            for name, config in receipt["models"].items():
                model = unpack(arrays, name)
                group = config["seed"], config["width_index"]
                if group not in kernels:
                    kernels[group] = direct_kernel(norm(model, data["color"][query_rows]), model["centers"], model["width"])
                prediction = kernels[group] @ model["coefficient"].astype(float) * model["y_std"] + model["y_mean"]
                difference = float(np.abs(prediction - saved[f"pred__{name}"]).max())
                assert difference < 1e-7
                maxima["oof_prediction_drift"] = max(maxima["oof_prediction_drift"], difference)
                checks["oof_query_rows"] += len(query_rows)
            oof_files.append(saved)
            if fold == 0:
                for family in ("local_irls", "midpoint_irls"):
                    base = unpack(arrays, name_for("norm_mse", 17, 1, 0, 0))
                    coefficients, _ = reference_readouts(base, x, y, balanced(data["patient"][rows], data["site"][rows]), family, .1, 16)
                    for step in (1, 4, 16):
                        reference = {**base, "coefficient": coefficients[step]}
                        prediction = independent_predict(reference, data["color"][query_rows])
                        difference = float(np.abs(prediction - saved[f"pred__{name_for(family, 17, 1, 0, step)}"]).max())
                        assert difference <= .002, (role, family, step, difference)
                        maxima["positive_step_probe_drift"] = max(maxima["positive_step_probe_drift"], difference)
                        checks["positive_step_probe_readouts"] += 1
                        probe_records.append({"role": role, "family": family, "step": step, "max_prediction_drift": difference})
            print(f"AUDIT {role}/fold{fold} hashes, all predictions and trajectory checks", flush=True)
        rows = np.concatenate([o["row_indices"] for o in oof_files])
        order = np.argsort(rows)
        rows = rows[order]
        np.testing.assert_array_equal(rows, base_rows)
        oof = {name: np.concatenate([o[name] for o in oof_files])[order] for name in oof_files[0] if name != "row_indices"}
        for family in FAMILIES:
            candidates = []
            step_choices = (0, 1, 4, 16) if family.endswith("irls") else ((0,) if family == "norm_mse" else (1,))
            for wi, width in enumerate(WIDTHS):
                for ai, alpha in enumerate(ALPHAS):
                    for step in step_choices:
                        means = [error_summary(oof[f"pred__{name_for(family, seed, wi, ai, step)}"], data["target"][rows], data["patient"][rows], data["site"][rows])[0]["person_mean"] for seed in SEEDS]
                        candidates.append({"width_index": wi, "alpha_index": ai, "steps": step, "person_mean": float(np.mean(means)), "alpha": alpha, "width_factor": width})
            best = min(candidates, key=lambda c: (c["person_mean"], c["steps"], c["alpha"], c["width_factor"]))
            chosen = selection["roles"][role][family]["selected"]
            for field in ("width_index", "alpha_index", "steps"):
                assert best[field] == chosen[field]
            assert abs(best["person_mean"] - chosen["person_mean"]) < 1e-10
            checks["choices"] += 1
        fit_rows, query_rows = base_rows, np.flatnonzero(held)
        for record in (r for r in results["records"] if r["role"] == role):
            name = f"{record['family']}_s{record['seed']}.npz"
            mp, pp = run / "selected" / role / name, run / "evaluated" / role / name
            assert sha(mp) == record["model_sha256"] and sha(pp) == record["prediction_sha256"]
            model, saved = nz(mp), nz(pp)
            np.testing.assert_array_equal(saved["row_indices"], query_rows)
            prediction = independent_predict(model, data["color"][query_rows])
            difference = float(np.abs(prediction - saved["prediction"]).max())
            assert difference < 1e-7
            maxima["selected_prediction_drift"] = max(maxima["selected_prediction_drift"], difference)
            metric, per_person = error_summary(prediction, data["target"][query_rows], data["patient"][query_rows], data["site"][query_rows])
            for field, value in metric.items():
                assert abs(record["metrics"][field] - value) < 1e-9
            final_scores[(role, record["family"], record["seed"])] = per_person
            coefficients, _ = reference_readouts(model, data["color"][fit_rows], data["target"][fit_rows],
                                                 balanced(data["patient"][fit_rows], data["site"][fit_rows]), record["family"], ALPHAS[record["alpha_index"]], record["steps"])
            reference = {**model, "coefficient": coefficients[record["steps"]]}
            difference = float(np.abs(independent_predict(reference, data["color"][query_rows]) - prediction).max())
            assert difference <= .002, (role, record["family"], difference)
            maxima["svd_prediction_drift"] = max(maxima["svd_prediction_drift"], difference)
            checks["selected_models"] += 1
            checks["selected_query_rows"] += len(query_rows)
            checks["independent_refits"] += 1
        print(f"AUDIT {role} choices and15 selected SVD refits", flush=True)
    assert checks["banks"] == 12 and checks["stored_readouts"] == 2916 and checks["choices"] == 15 and checks["selected_models"] == 45
    paired = []
    rng = np.random.default_rng(392417)
    for role in roles(data["patient"], data["device"]):
        for family in FAMILIES[1:]:
            delta = np.mean([np.asarray(final_scores[(role, family, seed)]) - np.asarray(final_scores[(role, "norm_mse", seed)]) for seed in SEEDS], axis=0)
            boot = np.mean(delta[rng.integers(0, len(delta), size=(20000, len(delta)))], axis=1)
            paired.append({"role": role, "family": family, "person_mean_difference": float(delta.mean()), "descriptive_bootstrap_interval": np.quantile(boot, [.025, .975]).tolist(), "n_people": len(delta)})
    dependency_paths = ("scripts/chromaseed_perceptual_reference.py", "scripts/chromaseed_kernel_audit.py", "scripts/chromaseed_refine_audit.py", "scripts/skin_local_search_train.py", "src/luma_skin_vision/color.py")
    result = {"passed": True, "source_lock_sha256": source, "selection_sha256": sha(run / "selections.json"), "audit_source_sha256": sha(Path(__file__)),
              "dependencies": {p: sha(ROOT / p) for p in dependency_paths}, "checks": checks, "maxima": maxima, "positive_step_probes": probe_records, "paired": paired,
              "seconds": time.perf_counter() - started, "evidence": "independent local algebra audit; analytic tensors/augmented SVD. Descriptive reused-TRAIN comparisons, not external confirmation."}
    write_json(output / "audit.json", result)
    print(f"AUDIT PASS {checks}; maxima {maxima}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for option in ("run", "cache", "parent", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.cache, args.parent, args.output)
