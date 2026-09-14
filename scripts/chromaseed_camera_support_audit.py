"""Independent weights, SVD pipelines, exhaustive matches and support calculations."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from luma_skin_vision.color import delta_e00

ROOT = Path(__file__).resolve().parents[1]


def js(path):
    return json.loads(path.read_text(encoding="utf-8"))


def weights(person, site, camera=None):
    labels = np.zeros(len(person)) if camera is None else camera
    result = []
    for p, s, c in zip(person, site, labels, strict=True):
        peers = (person == p) & (labels == c)
        result.append(
            1
            / (
                len(set(labels))
                * len(set(person[labels == c]))
                * len(set(site[peers]))
                * np.sum(peers & (site == s))
            )
        )
    result = np.array(result)
    return result / result.sum()


def norm(x, q, w):
    mean = np.array([np.sum(w * column) for column in x.T])
    std = np.maximum(
        np.array(
            [np.sqrt(np.sum(w * (column - m) ** 2)) for column, m in zip(x.T, mean, strict=True)]
        ),
        1e-6,
    )
    return (x - mean) / std, (q - mean) / std


def aggregate(value, person, site):
    return np.array(
        [
            np.mean(
                [
                    np.mean(value[(person == p) & (site == s)], axis=0)
                    for s in sorted(set(site[person == p]))
                ],
                axis=0,
            )
            for p in sorted(set(person))
        ]
    )


def metric(score, label):
    if len(set(label)) != 2:
        return None
    return dict(
        auc=float(roc_auc_score(label == 1, score)),
        balanced_accuracy=float(balanced_accuracy_score(label == 1, score >= 0)),
        slr_correct=int(np.sum(score[label == 1] >= 0)),
        slr_people=int(np.sum(label == 1)),
        ipod_correct=int(np.sum(score[label == -1] < 0)),
        ipod_people=int(np.sum(label == -1)),
    )


def close(actual, expected, atol=1e-8):
    if actual is None or expected is None:
        assert actual is expected
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            close(actual[key], expected[key], atol)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            close(a, b, atol)
    elif isinstance(expected, (float, int, np.number)):
        assert abs(actual - expected) < atol, (actual, expected)
    else:
        assert actual == expected


def quantile(values, w, q):
    mass = 0.0
    for i in sorted(range(len(values)), key=lambda i: values[i]):
        mass += w[i] / sum(w)
        if mass >= q:
            return float(values[i])
    return float(max(values))


def describe(values, w):
    return dict(
        mean=float(sum(values * w) / sum(w)),
        median=quantile(values, w, 0.5),
        p95=quantile(values, w, 0.95),
    )


def direct_dist(q, fit, kind):
    if kind == "color36":
        return np.sqrt(np.mean((q[:, None, :] - fit[None, :, :]) ** 2, axis=-1))
    return delta_e00(q[:, None, :], fit[None, :, :])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    run = args.run
    lock, result = js(run / "source_lock.json"), js(run / "results.json")
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH == lock["cache_sha256"]
    assert result["source_lock_sha256"] == sha(run / "source_lock.json")
    for field in ("sources", "input_sha256"):
        for rel, h in lock[field].items():
            assert sha(ROOT / rel) == h, rel
    for name, h in result["files"].items():
        assert sha(run / name) == h, name
    with np.load(args.cache, allow_pickle=False) as z:
        x, y, person, site, camera = (
            z[k] for k in ("color", "target", "patient", "site", "device")
        )
    with np.load(run / "predictions.npz", allow_pickle=False) as z:
        saved = {k: z[k] for k in z.files}
    with np.load(run / "pairs.npz", allow_pickle=False) as z:
        pairs = {k: z[k] for k in z.files}
    with np.load(run / "support.npz", allow_pickle=False) as z:
        support = {k: z[k] for k in z.files}
    label = np.where(camera == "SLR", 1.0, -1.0)
    folds = folds_for(person, camera)
    people = np.unique(person)
    plabel = np.array([np.unique(label[person == p]).item() for p in people])
    pfold = np.array([np.unique(folds[person == p]).item() for p in people])
    centroid = aggregate(y, person, site)
    counters = dict(
        classifier_refits=0,
        prediction_rows=0,
        all_person_metrics=0,
        fold_metrics=0,
        exhaustive_matching_problems=0,
        matched_metric_groups=0,
        support_roles_views=0,
        support_row_distances=0,
        target_camera_summaries=0,
    )
    maxima = dict(classifier_score_drift=0.0, support_distance_drift=0.0, kernel_width_drift=0.0)
    for fold in range(3):
        a, b = folds != fold, folds == fold
        assert not set(person[a]) & set(person[b])
        w = weights(person[a], site[a], camera[a])
        z, q = norm(x[a], x[b], w)
        t, tq = norm(y[a], y[b], w)
        d, dq = np.column_stack([np.ones(sum(a)), t]), np.column_stack([np.ones(sum(b)), tq])
        root = np.sqrt(w)
        beta = lstsq(root[:, None] * d, root[:, None] * z, cond=1e-12, lapack_driver="gelsd")[0]
        residual, qr = norm(z - d @ beta, q - dq @ beta, w)
        fit_views = dict(color36=z, rgb_mean=z[:, 27:30], lab3=t, lab_residual=residual)
        query_views = dict(color36=q, rgb_mean=q[:, 27:30], lab3=tq, lab_residual=qr)
        for record in (r for r in result["classifier_fits"] if r["fold"] == fold):
            v, method = record["view"], record["method"]
            f, qv = fit_views[v], query_views[v]
            if method == "linear":
                design = np.column_stack([np.ones(len(f)), f])
                penalty = np.column_stack([np.zeros(f.shape[1]), np.sqrt(0.1) * np.eye(f.shape[1])])
                coeff = lstsq(
                    np.vstack([root[:, None] * design, penalty]),
                    np.r_[root * label[a], np.zeros(f.shape[1])],
                    lapack_driver="gelsd",
                )[0]
                pred = np.column_stack([np.ones(len(qv)), qv]) @ coeff
            else:
                distance = np.sqrt(np.mean((f[:, None, :] - f[None, :, :]) ** 2, axis=-1))
                d = distance[np.triu_indices(len(f), 1)]
                positive = sorted(d[d > 1e-10])
                width = max(positive[(len(positive) - 1) // 2], 1e-6) if len(positive) else 1e-6
                maxima["kernel_width_drift"] = max(
                    maxima["kernel_width_drift"], abs(width - record["width"])
                )
                k = np.exp(-0.5 * (distance / width) ** 2)
                h = root[:, None] * k * root[None, :] + 0.01 * np.eye(len(f))
                coeff = lstsq(h, root * label[a], lapack_driver="gelsd")[0] * root
                kq = np.exp(
                    -0.5 * np.mean((qv[:, None, :] - f[None, :, :]) ** 2, axis=-1) / (width * width)
                )
                pred = kq @ coeff
            drift = float(np.max(np.abs(pred - saved[f"{v}__{method}"][b])))
            assert drift < 1e-8, (fold, v, method, drift)
            maxima["classifier_score_drift"] = max(maxima["classifier_score_drift"], drift)
            assert record["fit_people"] == len(set(person[a])) and record["query_people"] == len(
                set(person[b])
            )
            assert record["fit_rows"] == sum(a) and record["query_rows"] == sum(b)
            counters["classifier_refits"] += 1
            counters["prediction_rows"] += sum(b)
        print(f"AUDITED fold{fold}: eight independent SVD camera fits", flush=True)
    pscores = {name: aggregate(value, person, site) for name, value in saved.items()}
    for record in result["classifier_metrics"]:
        s = pscores[f"{record['view']}__{record['method']}"]
        close(record["all_people"], metric(s, plabel))
        counters["all_person_metrics"] += 1
        for f in record["folds"]:
            close(
                {k: v for k, v in f.items() if k != "fold"},
                metric(s[pfold == f["fold"]], plabel[pfold == f["fold"]]),
            )
            counters["fold_metrics"] += 1
    for record in result["matched"]:
        caliper = record["caliper"]
        chunks = []
        for fold in range(3):
            a = np.flatnonzero((pfold == fold) & (plabel == 1))
            b = np.flatnonzero((pfold == fold) & (plabel == -1))
            cost = delta_e00(centroid[a, None, :], centroid[b][None, :, :])
            best = (1, np.inf)
            for assignment in itertools.product(range(-1, len(b)), repeat=len(a)):
                used = [i for i in assignment if i >= 0]
                if len(set(used)) != len(used) or any(
                    cost[i, j] > caliper for i, j in enumerate(assignment) if j >= 0
                ):
                    continue
                objective = (
                    -len(used),
                    sum(cost[i, j] for i, j in enumerate(assignment) if j >= 0),
                )
                best = min(best, objective)
            pairing = pairs[f"c{int(caliper)}__f{fold}"]
            assert len(set(pairing[:, 0])) == len(pairing) and len(set(pairing[:, 1])) == len(
                pairing
            )
            assert all(i in a and j in b for i, j in pairing)
            dist = delta_e00(centroid[pairing[:, 0]], centroid[pairing[:, 1]])
            assert (
                np.all(dist <= caliper)
                and -len(pairing) == best[0]
                and abs(sum(dist) - best[1]) < 1e-10
            )
            chunks.append(pairing)
            counters["exhaustive_matching_problems"] += 1
        pairing = np.concatenate(chunks)
        distance = delta_e00(centroid[pairing[:, 0]], centroid[pairing[:, 1]])
        assert record["pairs"] == len(pairing) and record["matched_people"] == 2 * len(pairing)
        assert record["unmatched_slr"] == 8 - len(pairing) and record["unmatched_ipod"] == 16 - len(
            pairing
        )
        assert record["matched_images"] == int(np.isin(person, people[pairing.ravel()]).sum())
        close(
            record["pair_de00"],
            describe(distance, np.ones(len(distance))) if len(distance) else None,
        )
        for s in record["scores"]:
            values = pscores[f"{s['view']}__{s['method']}"]
            close(s["metrics"], metric(values[pairing.ravel()], plabel[pairing.ravel()]))
            gap = values[pairing[:, 0]] - values[pairing[:, 1]]
            close(
                s["within_pair_slr_above"],
                float(np.mean((gap > 0) + 0.5 * (gap == 0))) if len(gap) else None,
            )
            counters["matched_metric_groups"] += 1
    for role, (fit, held) in roles(person, camera).items():
        fw, qw = weights(person[fit], site[fit]), weights(person[held], site[held])
        z, q = norm(x[fit], x[held], fw)
        for view, f, query in [("color36", z, q), ("native_lab", y[fit], y[held])]:
            ds = direct_dist(f, f, view)
            ds[person[fit, None] == person[fit][None, :]] = np.inf
            reference = ds.min(axis=1)
            query_distance = direct_dist(query, f, view).min(axis=1)
            for name, value in [("fit", reference), ("query", query_distance)]:
                drift = float(np.max(np.abs(support[f"{role}__{view}__{name}"] - value)))
                maxima["support_distance_drift"] = max(maxima["support_distance_drift"], drift)
                assert drift < 1e-9
                counters["support_row_distances"] += len(value)
            threshold = quantile(reference, fw, 0.95)
            actual = next(r for r in result["support"] if r["role"] == role and r["view"] == view)
            close(actual["fit_reference"], describe(reference, fw))
            close(actual["query"], describe(query_distance, qw))
            close(actual["fit_q95_radius"], threshold)
            close(actual["query_mass_above_radius"], sum(qw[query_distance > threshold]))
            counters["support_roles_views"] += 1
    for record in result["target_summary"]:
        mask = camera == record["camera"]
        w = weights(person[mask], site[mask])
        cent = aggregate(y[mask], person[mask], site[mask])
        assert record["people"] == len(cent) and record["rows"] == sum(mask)
        close(
            record["lab_quantiles_05_50_95"],
            [[quantile(y[mask, k], w, q) for k in range(3)] for q in (0.05, 0.5, 0.95)],
        )
        close(record["person_centroid_min"], cent.min(axis=0).tolist())
        close(record["person_centroid_max"], cent.max(axis=0).tolist())
        counters["target_camera_summaries"] += 1
    expected = dict(
        classifier_refits=24,
        prediction_rows=7728,
        all_person_metrics=8,
        fold_metrics=24,
        exhaustive_matching_problems=15,
        matched_metric_groups=40,
        support_roles_views=6,
        support_row_distances=5796,
        target_camera_summaries=2,
    )
    assert counters == expected
    write_json(
        args.output / "audit.json",
        dict(
            passed=True,
            source_lock_sha256=sha(run / "source_lock.json"),
            results_sha256=sha(run / "results.json"),
            audit_source_sha256=sha(Path(__file__)),
            dependencies={
                r: sha(ROOT / r)
                for r in ("scripts/skin_local_search_train.py", "src/luma_skin_vision/color.py")
            },
            checks={k: int(v) for k, v in counters.items()},
            maxima={k: float(v) for k, v in maxima.items()},
            elapsed_seconds=time.perf_counter() - start,
            limits="Independent numerical pipelines; shared verified CIEDE2000 and frozen folds. Not camera causality or new skin-color accuracy.",
        ),
    )
    print(
        "AUDIT PASS: all predictions, exhaustive caliper objectives, metrics and support/target summaries",
        flush=True,
    )


if __name__ == "__main__":
    main()
