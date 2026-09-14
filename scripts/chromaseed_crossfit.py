"""Fixed-H compact corrections with matched/included and excluded-person targets."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_affine import color_metrics, design_matrix, gram_stats, solve_stats
from chromaseed_gated import fit_gate
from chromaseed_hybrid import package, predict, projection_fit, raw_fit, shared_basis, support
from chromaseed_kernel import gaussian_kernel
from chromaseed_perceptual import prepare
from chromaseed_projection import export
from skin_local_search_train import weights_for

ARMS, LOSSES, SEEDS = ("in_matched", "out_person"), ("norm", "perceptual"), (17, 29, 43)


def routing(person, camera):
    person, camera = np.asarray(person), np.asarray(camera)
    if person.ndim != 1 or camera.shape != person.shape or not len(person):
        raise ValueError("nonempty person/camera vectors required")
    people, own = np.unique(person, return_inverse=True)
    group = []
    for p in people:
        c = np.unique(camera[person == p])
        if len(c) != 1:
            raise ValueError("one camera group per person required")
        group.append(c[0])
    group = np.array(group)
    next_person = np.empty(len(people), np.int64)
    for c in np.unique(group):
        ix = np.flatnonzero(group == c)
        if len(ix) < 2:
            raise ValueError("at least two fit people per camera required")
        next_person[ix] = np.roll(ix, -1)
    return people, own, next_person[own]


def route(native_predictions, own, matched):
    a = np.asarray(native_predictions, np.float64)
    own, matched = np.asarray(own, np.int64), np.asarray(matched, np.int64)
    if a.ndim != 3 or a.shape[1:] != (len(own), 3) or matched.shape != own.shape:
        raise ValueError("teacher x original-row x native-Lab table required")
    if (
        not np.isfinite(a).all()
        or np.any(own == matched)
        or np.any(own < 0)
        or np.any(matched < 0)
        or np.any(own >= len(a))
        or np.any(matched >= len(a))
    ):
        raise ValueError("finite table and distinct valid teacher routing required")
    rows = np.arange(len(own))
    return a[own, rows].copy(), a[matched, rows].copy()


def teacher_fit(x, y, person, site, camera, excluded, seeds=SEEDS, losses=LOSSES, rank=128):
    fit = np.flatnonzero(person != excluded)
    excluded_rows = np.flatnonzero(person == excluded)
    if not len(fit) or not len(excluded_rows):
        raise ValueError("nonempty fit and whole excluded person required")
    state = prepare(x[fit], y[fit], weights_for(person[fit], site[fit]))
    beta = fit_gate(state["normalized"], person[fit], site[fit], camera[fit])
    metric = color_metrics(state, losses)
    models, operations, bases = {}, [], []
    for seed in seeds:
        raw, b, _, info = raw_fit(state, beta, metric, seed, rank)
        operations.append(dict(seed=seed, **info))
        bases.append(
            dict(
                seed=seed,
                center_local_rows=fit[b["ids"]].tolist(),
                effective_rank=b["whitening"].shape[1],
            )
        )
        for loss in losses:
            models[f"{loss}_s{seed}"] = raw[loss]
    return models, dict(
        fit_local_rows=fit.tolist(),
        excluded_local_rows=excluded_rows.tolist(),
        fit_person_values=np.unique(person[fit]).tolist(),
        n_fit_rows=len(fit),
        n_fit_people=len(np.unique(person[fit])),
        active_gate=beta is not None,
        bases=bases,
        operations=operations,
    )


def teacher_pool(x, y, person, site, camera, seeds=SEEDS, losses=LOSSES, rank=128):
    people, own, matched = routing(person, camera)
    models, tables, subsets = {}, {}, []
    for j, excluded in enumerate(people):
        current, info = teacher_fit(x, y, person, site, camera, excluded, seeds, losses, rank)
        subsets.append(dict(excluded_index=j, **info))
        for name, m in current.items():
            models[f"exclude{j}_{name}"] = m
            key = "native__" + name
            if key not in tables:
                tables[key] = np.empty((len(people), len(x), 3), np.float64)
            tables[key][j] = predict(m, x)
    for loss in losses:
        for seed in seeds:
            name = f"{loss}_s{seed}"
            out, included = route(tables["native__" + name], own, matched)
            tables["out_person__" + name] = out
            tables["in_matched__" + name] = included
    tables.update(own_teacher=own.astype(np.int64), matched_teacher=matched.astype(np.int64))
    return (
        models,
        tables,
        dict(
            people=people.tolist(),
            subsets=subsets,
            teacher_subsets=len(people),
            teacher_models=len(models),
            teacher_prediction_rows=len(people) * len(x) * len(losses) * len(seeds),
            routed_prediction_rows=2 * len(x) * len(losses) * len(seeds),
            teacher_gate_fits=sum(s["active_gate"] for s in subsets),
        ),
    )


def students(
    x, y, person, site, camera, settings, tables, seeds=SEEDS, losses=LOSSES, arms=ARMS, rank=128
):
    state = prepare(x, y, weights_for(person, site))
    beta = fit_gate(state["normalized"], person, site, camera)
    metric = color_metrics(state, losses)
    payload, q, width, pi = projection_fit(state, x)
    models, operations, bases = {}, [], []
    for seed in seeds:
        raw, b, signal, info = raw_fit(state, beta, metric, seed, rank)
        operations.append(dict(seed=seed, kind="raw", **info))
        pb = shared_basis(q, b["ids"], width)
        bases.append(
            dict(
                seed=seed,
                center_local_rows=b["ids"].tolist(),
                raw_rank=b["whitening"].shape[1],
                projected_rank=pb["whitening"].shape[1],
            )
        )
        d = design_matrix(pb["z"], signal, beta is not None)
        for loss in losses:
            models[f"{loss}_raw_s{seed}"] = raw[loss]
            tables[f"full__{loss}_s{seed}"] = predict(raw[loss], x)
            k = gaussian_kernel(
                state["normalized"], raw[loss]["centers"], float(raw[loss]["width"])
            )
            for kind, setting in settings[loss].items():
                design = d if kind == "uniform" else support(k, setting["power"])[:, None] * d
                for arm in arms:
                    native = tables[f"{arm}__{loss}_s{seed}"]
                    residual = (np.asarray(y, np.float64) - native) / state["prep"]["y_std"]
                    solved, info = solve_stats(
                        gram_stats(design, residual, state["weights"]),
                        {loss: metric[loss]},
                        (setting["alpha"],),
                    )
                    operations.append(dict(seed=seed, loss=loss, arm=arm, **setting, **info))
                    branch = export(
                        state, payload, pb, solved[(loss, setting["alpha"])], beta, beta is not None
                    )
                    models[f"{arm}_{loss}_{kind}_s{seed}"] = package(
                        raw[loss], branch, kind, setting["rho"], setting["power"]
                    )
    return models, dict(
        student_operations=operations,
        student_bases=bases,
        projection_info=pi,
        student_gate_fits=int(beta is not None),
        n_fit_rows=len(x),
        original_weight_mass=float(state["weights"].sum()),
        new_correction_solutions=sum(len(o["solutions"]) for o in operations if o["kind"] != "raw"),
    )


def fit_bank(x, y, person, site, camera, settings, rank=128):
    start = time.perf_counter()
    teachers, tables, info = teacher_pool(x, y, person, site, camera, rank=rank)
    models, si = students(x, y, person, site, camera, settings, tables, rank=rank)
    return (
        models,
        teachers,
        tables,
        dict(**info, **si, fit_bank_seconds=time.perf_counter() - start),
    )


def fit_single(x, y, person, site, camera, loss, seed, arm, setting, rank=128):
    if (
        loss not in LOSSES
        or seed not in SEEDS
        or arm not in ARMS
        or setting["kind"] not in ("uniform", "support")
    ):
        raise ValueError("registered C arm and H correction setting required")
    start = time.perf_counter()
    _, tables, info = teacher_pool(
        x, y, person, site, camera, seeds=(seed,), losses=(loss,), rank=rank
    )
    models, _ = students(
        x,
        y,
        person,
        site,
        camera,
        {loss: {setting["kind"]: setting}},
        tables,
        seeds=(seed,),
        losses=(loss,),
        arms=(arm,),
        rank=rank,
    )
    return models[f"{arm}_{loss}_{setting['kind']}_s{seed}"], dict(
        fit_seconds=time.perf_counter() - start, teacher_subsets=info["teacher_subsets"]
    )
