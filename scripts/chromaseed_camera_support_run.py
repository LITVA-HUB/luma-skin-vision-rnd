"""Fit registered camera classifiers and measure input/target coverage, TRAIN only."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy
from chromaseed_camera_support import (
    CALIPERS,
    METHODS,
    VIEWS,
    aggregate_people,
    classification_metrics,
    classify,
    describe,
    match_cost,
    nearest,
    row_weights,
    standardize,
    views,
    weighted_quantile,
)
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json

from luma_skin_vision.color import delta_e00

ROOT = Path(__file__).resolve().parents[1]


def js(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lock_sources(run, cache):
    if cache.name != "train.npz" or sha(cache) != CACHE_HASH:
        raise ValueError("only original TRAIN allowed")
    parent = ROOT / "experiments/runs/chromaseed_selection_stability_v1"
    inherited = js(parent / "source_lock.json")
    sources = dict(inherited["sources"])
    for rel, h in sources.items():
        assert sha(ROOT / rel) == h, rel
    own = (
        "scripts/chromaseed_camera_support.py",
        "scripts/chromaseed_camera_support_run.py",
        "tests/test_chromaseed_camera_support.py",
        "docs/research/chromaseed_camera_support_v1_protocol.md",
        "scripts/chromaseed.py",
        "tests/test_chromaseed.py",
    )
    sources.update({rel: sha(ROOT / rel) for rel in own})
    inputs = [
        parent / "source_lock.json",
        ROOT / "docs/benchmarks/chromaseed_selection_stability_v1/verification.json",
    ]
    assert js(inputs[1])["passed"]
    lock = dict(
        sources=sources,
        input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        cache_sha256=CACHE_HASH,
        views=list(VIEWS),
        methods=list(METHODS),
        calipers=list(CALIPERS),
        numpy=np.__version__,
        scipy=scipy.__version__,
        threads=1,
        loaded_keys=["color", "target", "patient", "site", "device"],
        purpose="camera/support diagnostic; no color-model promotion",
    )
    path = run / "source_lock.json"
    if path.exists() and js(path) != lock:
        raise ValueError("source lock changed")
    if not path.exists():
        write_json(path, lock)
    return sha(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    started = time.perf_counter()
    lock = lock_sources(run, args.cache)
    if (run / "results.json").exists():
        raise ValueError("completed run exists; use fresh directory")
    write_json(
        run / "progress.json", dict(status="diagnosing", pid=os.getpid(), updated_unix=time.time())
    )
    with np.load(args.cache, allow_pickle=False) as z:
        x, y, person, site, camera = (
            z[k] for k in ("color", "target", "patient", "site", "device")
        )
    assert x.shape == (966, 36) and y.shape == (966, 3)
    label = np.where(camera == "SLR", 1.0, -1.0)
    folds = folds_for(person, camera)
    people = np.unique(person)
    plabel = np.array([np.unique(label[person == p]).item() for p in people])
    pfold = np.array([np.unique(folds[person == p]).item() for p in people])
    centroid = aggregate_people(y, person, site)
    predictions = {f"{v}__{m}": np.empty(len(x)) for v in VIEWS for m in METHODS}
    fit_records = []
    for fold in range(3):
        a, b = folds != fold, folds == fold
        assert not set(person[a]) & set(person[b])
        w = row_weights(person[a], site[a], camera[a])
        fit_views, query_views = views(x[a], y[a], x[b], y[b], w)
        for v in VIEWS:
            for m in METHODS:
                scores, info = classify(fit_views[v], query_views[v], label[a], w, m)
                predictions[f"{v}__{m}"][b] = scores
                fit_records.append(
                    dict(
                        fold=fold,
                        view=v,
                        method=m,
                        fit_people=len(np.unique(person[a])),
                        query_people=len(np.unique(person[b])),
                        fit_rows=int(a.sum()),
                        query_rows=int(b.sum()),
                        **info,
                    )
                )
        print(f"FIT fold{fold}: eight fixed camera classifiers, person-disjoint", flush=True)
    pscores = {name: aggregate_people(values, person, site) for name, values in predictions.items()}
    metrics = []
    for v in VIEWS:
        for m in METHODS:
            score = pscores[f"{v}__{m}"]
            metrics.append(
                dict(
                    view=v,
                    method=m,
                    all_people=classification_metrics(score, plabel),
                    folds=[
                        dict(
                            fold=f, **classification_metrics(score[pfold == f], plabel[pfold == f])
                        )
                        for f in range(3)
                    ],
                )
            )
    pairs, matched = {}, []
    for caliper in CALIPERS:
        chunks = []
        for fold in range(3):
            a = np.flatnonzero((pfold == fold) & (plabel == 1))
            b = np.flatnonzero((pfold == fold) & (plabel == -1))
            cost = delta_e00(centroid[a, None, :], centroid[b][None, :, :])
            i, j = match_cost(cost, caliper)
            pairing = np.column_stack([a[i], b[j]])
            pairs[f"c{int(caliper)}__f{fold}"] = pairing
            chunks.append(pairing)
        pairing = np.concatenate(chunks)
        unique = pairing.ravel()
        distance = delta_e00(centroid[pairing[:, 0]], centroid[pairing[:, 1]])
        scores = []
        for v in VIEWS:
            for m in METHODS:
                s = pscores[f"{v}__{m}"]
                gap = s[pairing[:, 0]] - s[pairing[:, 1]]
                scores.append(
                    dict(
                        view=v,
                        method=m,
                        metrics=classification_metrics(s[unique], plabel[unique]),
                        within_pair_slr_above=float(np.mean((gap > 0) + 0.5 * (gap == 0)))
                        if len(gap)
                        else None,
                    )
                )
        matched.append(
            dict(
                caliper=caliper,
                pairs=len(pairing),
                matched_people=len(unique),
                unmatched_slr=8 - len(pairing),
                unmatched_ipod=16 - len(pairing),
                matched_images=int(np.isin(person, people[unique]).sum()),
                pair_de00=describe(distance, np.ones(len(distance))) if len(distance) else None,
                scores=scores,
            )
        )
    print("MATCHED all five fixed target-color calipers within held folds", flush=True)
    support, distances = [], {}
    for role, (fit, held) in roles(person, camera).items():
        fw, qw = row_weights(person[fit], site[fit]), row_weights(person[held], site[held])
        zx, zq, _ = standardize(x[fit], x[held], fw)
        for name, fx, qx, kind in [
            ("color36", zx, zq, "rms"),
            ("native_lab", y[fit], y[held], "de00"),
        ]:
            reference = nearest(fx, fx, kind, person[fit], person[fit])
            query = nearest(qx, fx, kind)
            threshold = weighted_quantile(reference, fw, 0.95)
            distances[f"{role}__{name}__fit"] = reference
            distances[f"{role}__{name}__query"] = query
            support.append(
                dict(
                    role=role,
                    view=name,
                    fit_people=len(np.unique(person[fit])),
                    query_people=len(np.unique(person[held])),
                    fit_reference=describe(reference, fw),
                    query=describe(query, qw),
                    fit_q95_radius=threshold,
                    query_mass_above_radius=float(np.sum(qw[query > threshold])),
                )
            )
    target_summary = []
    for c in np.unique(camera):
        mask = camera == c
        w = row_weights(person[mask], site[mask])
        cent = aggregate_people(y[mask], person[mask], site[mask])
        target_summary.append(
            dict(
                camera=str(c),
                people=len(cent),
                rows=int(mask.sum()),
                lab_quantiles_05_50_95=[
                    [weighted_quantile(y[mask, k], w, q) for k in range(3)]
                    for q in (0.05, 0.5, 0.95)
                ],
                person_centroid_min=cent.min(axis=0).tolist(),
                person_centroid_max=cent.max(axis=0).tolist(),
            )
        )
    np.savez_compressed(run / "predictions.npz", **predictions)
    np.savez_compressed(run / "pairs.npz", **pairs)
    np.savez_compressed(run / "support.npz", **distances)
    write_json(
        run / "results.json",
        dict(
            source_lock_sha256=lock,
            classifier_fits=fit_records,
            classifier_metrics=metrics,
            matched=matched,
            support=support,
            target_summary=target_summary,
            files={n: sha(run / n) for n in ("predictions.npz", "pairs.npz", "support.npz")},
            elapsed_seconds=time.perf_counter() - started,
            evidence="Repeated original TRAIN; diagnostic classifiers, no new skin-color predictor",
        ),
    )
    write_json(
        run / "progress.json",
        dict(status="diagnostic_complete_audit_pending", pid=os.getpid(), updated_unix=time.time()),
    )


if __name__ == "__main__":
    main()
