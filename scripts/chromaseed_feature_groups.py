"""Fixed input subsets with matched A readouts and exact raw/X references."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_affine import color_metrics, design_matrix, export_model, gram_stats, solve_stats
from chromaseed_gated import fit_gate
from chromaseed_gated import predict as gated_predict
from chromaseed_kernel import coordinates
from chromaseed_perceptual import make_basis, prepare
from chromaseed_projection import exact_width, projected_basis
from chromaseed_projection import predict as x_predict
from skin_local_search_train import weights_for

GROUPS = {
    "raw36": list(range(36)),
    "mean3": [27, 28, 29],
    "median3": [12, 13, 14],
    "central9": list(range(9, 18)),
    "mean_std6": list(range(27, 33)),
    "quant27": list(range(27)),
    "no_corr33": list(range(33)),
}
ALL_GROUPS = (*GROUPS, "projected16")
FAMILIES = ("norm_static", "norm_joint_soft", "perceptual_static", "perceptual_joint_soft")
SEEDS, ALPHAS, POLICIES = (17, 29, 43), (0.1, 1.0, 10.0), ("quality", "compact")


def gather(x, group):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != 36 or not np.isfinite(x).all() or group not in GROUPS:
        raise ValueError("finite color36 matrix and registered group required")
    return x if group == "raw36" else x[:, GROUPS[group]].copy()


def state_for(x, y, weights, group):
    xx = gather(x, group)
    if group == "raw36":
        return prepare(xx, y, weights)
    y = np.asarray(y, np.float64)
    if not len(xx) or y.shape != (len(xx), 3) or not np.isfinite(y).all():
        raise ValueError("nonempty matching native Lab targets required")
    prep = {}
    for prefix, a in (("x", xx.astype(np.float64)), ("y", y)):
        prep[prefix + "_mean"] = a.mean(0).astype(np.float32)
        prep[prefix + "_std"] = np.maximum(a.std(0), 1e-6).astype(np.float32)
    z = coordinates(prep, xx)
    width, info = exact_width(z)
    return dict(
        prep=prep,
        normalized=z,
        base=width,
        width_info=info,
        y=y,
        yn=(y - prep["y_mean"]) / prep["y_std"],
        weights=weights,
    )


def basis_for(state, seed, rank, group):
    if group == "raw36":
        basis = make_basis(state, seed, 1, rank, (0.1,))
        info = dict(**basis["info"], fit_center_indices=basis["ids"].tolist())
    else:
        basis, info = projected_basis(
            state["normalized"], state["weights"], float(np.float32(state["base"])), seed, rank
        )
    return basis, {**info, "effective_rank": basis["whitening"].shape[1]}


def export(state, basis, theta, beta, joint, group):
    model = export_model(state, basis, theta, beta, joint)
    if group != "raw36":
        model["feature_indices"] = np.asarray(GROUPS[group], np.uint8)
    return model


def predict(model, x):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != 36 or not np.isfinite(x).all():
        raise ValueError("finite color36 matrix required")
    if "feature_indices" not in model:
        return x_predict(model, x)
    base = {k: v for k, v in model.items() if k != "feature_indices"}
    return gated_predict(base, x[:, model["feature_indices"]])


def gate_score(model, x):
    if "gate_beta" not in model:
        return None
    x = np.asarray(x, np.float32)
    if "feature_indices" in model:
        x = x[:, model["feature_indices"]]
    z = coordinates(model, x)
    beta = model["gate_beta"].astype(np.float64)
    return z @ beta[1:] + beta[0]


def model_id(family, seed, group, alpha):
    return f"{family}_{group}_s{seed}_a{ALPHAS.index(alpha)}"


def choose_alpha(rows):
    if not rows:
        raise ValueError("nonempty candidates required")
    return min(rows, key=lambda r: (r["clean"], r["p90"], -r["alpha"]))


def choose_policy(rows, policy):
    if not rows or policy not in POLICIES:
        raise ValueError("nonempty groups and registered policy required")
    if policy == "quality":
        return min(rows, key=lambda r: (r["clean"], r["numeric_bytes"], r["p90"], r["order"]))
    raw = next(r for r in rows if r["group"] == "raw36")
    eligible = [
        r for r in rows if r["clean"] <= raw["clean"] + 0.05 and r["p90"] <= raw["p90"] + 0.1
    ]
    return min(eligible, key=lambda r: (r["numeric_bytes"], r["clean"], r["p90"], r["order"]))


def fit_bank(x, y, person, site, camera, rank=128, groups=None, seeds=SEEDS):
    start = time.perf_counter()
    groups = tuple(GROUPS) if groups is None else tuple(groups)
    if not groups or len(set(groups)) != len(groups) or any(g not in GROUPS for g in groups):
        raise ValueError("distinct registered groups required")
    weights = weights_for(person, site)
    first = state_for(x, y, weights, groups[0])
    metrics = color_metrics(first)
    models, metadata, operations, bases, states = {}, {}, [], [], []
    gate_count = 0
    for group in groups:
        state = first if group == groups[0] else state_for(x, y, weights, group)
        beta = fit_gate(state["normalized"], person, site, camera)
        gate_count += int(beta is not None)
        signal = (
            None
            if beta is None
            else state["normalized"] @ beta[1:].astype(np.float64) + float(beta[0])
        )
        structures = (False, True) if beta is not None else (False,)
        states.append(
            dict(
                group=group,
                dimension=len(GROUPS[group]),
                base_width=state["base"],
                width_info=state["width_info"],
            )
        )
        for seed in seeds:
            basis, info = basis_for(state, seed, rank, group)
            bases.append(dict(group=group, seed=seed, **info))
            for joint in structures:
                design = design_matrix(basis["z"], signal, joint)
                solutions, detail = solve_stats(
                    gram_stats(design, state["yn"], weights), metrics, ALPHAS
                )
                operations.append(dict(group=group, seed=seed, joint=joint, **detail))
                for (loss, alpha), theta in solutions.items():
                    family = loss + ("_joint_soft" if joint else "_static")
                    name = model_id(family, seed, group, alpha)
                    models[name] = export(state, basis, theta, beta, joint, group)
                    metadata[name] = dict(
                        family=family, seed=seed, group=group, alpha=alpha, fallback=False
                    )
            if beta is None:
                for loss in ("norm", "perceptual"):
                    for alpha in ALPHAS:
                        family = loss + "_joint_soft"
                        name = model_id(family, seed, group, alpha)
                        models[name] = dict(models[model_id(loss + "_static", seed, group, alpha)])
                        metadata[name] = dict(
                            family=family, seed=seed, group=group, alpha=alpha, fallback=True
                        )
    return models, dict(
        models=metadata,
        operations=operations,
        bases=bases,
        groups=states,
        new_coefficient_solutions=sum(len(o["solutions"]) for o in operations),
        gram_decompositions=len(operations),
        basis_preparations=len(bases),
        exact_width_calculations=len(groups),
        gate_fits=gate_count,
        auxiliary_baseline_solves=len(seeds) * int("raw36" in groups),
        auxiliary_theta_solves=len(seeds) * int("raw36" in groups),
        target_metric_preparations=1,
        n_fit_rows=len(x),
        original_weight_mass=float(weights.sum()),
        fit_bank_seconds=time.perf_counter() - start,
    )


def fit_single(x, y, person, site, camera, family, seed, group, alpha, rank=128):
    start = time.perf_counter()
    if family not in FAMILIES or seed not in SEEDS or group not in GROUPS or alpha not in ALPHAS:
        raise ValueError("registered family, seed, group and alpha required")
    weights = weights_for(person, site)
    state = state_for(x, y, weights, group)
    joint = family.endswith("joint_soft")
    beta = fit_gate(state["normalized"], person, site, camera) if joint else None
    active = beta is not None
    signal = (
        None if beta is None else state["normalized"] @ beta[1:].astype(np.float64) + float(beta[0])
    )
    basis, info = basis_for(state, seed, rank, group)
    solutions, detail = solve_stats(
        gram_stats(design_matrix(basis["z"], signal, active), state["yn"], weights),
        color_metrics(state, (family.split("_")[0],)),
        (alpha,),
    )
    model = export(state, basis, solutions[(family.split("_")[0], alpha)], beta, active, group)
    return model, dict(
        fit_seconds=time.perf_counter() - start, basis=info, active_gate=active, **detail
    )
