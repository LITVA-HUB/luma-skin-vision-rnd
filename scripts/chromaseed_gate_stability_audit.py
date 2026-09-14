"""Independent feature algebra, NumPy batch-one predictions and boundary checks for GS."""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
from chromaseed_gated_numpy import Predictor
from chromaseed_kernel_audit import direct_kernel, js, nz
from skin_local_search_train import CACHE_HASH, sha, write_json

from luma_skin_vision.color import delta_e00

ROOT = Path(__file__).resolve().parents[1]
ANCHORS = np.array(list(itertools.product((0.0, 1.0), repeat=3)))
DOSES = np.array([1, 4, 16, 64], dtype=np.float64) / 255
EPS = 1e-4


def transformed(x, amount, color):
    source = np.asarray(x, np.float64)
    alpha = np.broadcast_to(np.asarray(amount), (len(x),))
    anchor = np.broadcast_to(np.asarray(color), (len(x), 3))
    result = source.copy()
    for channel in range(3):
        indices = np.arange(channel, 30, 3)
        result[:, indices] = source[:, indices] + alpha[:, None] * (
            anchor[:, channel, None] - source[:, indices]
        )
        result[:, 30 + channel] = source[:, 30 + channel] - alpha * source[:, 30 + channel]
    return result.astype(np.float32)


def normalized(m, x):
    return ((np.asarray(x, np.float32) - m["x_mean"]) / m["x_std"]).astype(np.float64)


def score(m, x):
    z = normalized(m, x)
    return np.sum(z * m["gate_beta"][None, 1:].astype(np.float64), axis=1) + float(
        m["gate_beta"][0]
    )


def metric(v, person, camera):
    if len(v) == 0:
        return None
    v = np.asarray(v, np.float64)
    group = {}
    for c in np.unique(camera):
        people = np.unique(person[camera == c])
        group[str(c)] = float(
            sum(float(v[(person == p) & (camera == c)].mean()) for p in people) / len(people)
        )
    people = np.unique(person)
    ordered = np.sort(v)
    return dict(
        person_mean=float(sum(float(v[person == p].mean()) for p in people) / len(people)),
        image_mean=float(v.sum() / len(v)),
        p90=float(np.interp(0.9 * (len(v) - 1), np.arange(len(v)), ordered)),
        maximum=float(max(v)),
        people=len(people),
        rows=len(v),
        camera_person_mean=group,
    )


def close(actual, expected, tol=2e-8):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            close(actual[key], expected[key], tol)
    elif expected is None:
        assert actual is None
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            close(a, b, tol)
    else:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tol)


def feasible(x):
    flags = []
    for row in x:
        ok = np.isfinite(row).all()
        ok &= all(-1e-6 <= v <= 1 + 1e-6 for v in row[:30])
        ok &= all(row[k + 3] >= row[k] - 1e-6 for k in range(24))
        ok &= all(
            row[30 + c] >= -1e-6 and row[30 + c] ** 2 <= row[27 + c] * (1 - row[27 + c]) + 1e-6
            for c in range(3)
        )
        ok &= all(abs(v) <= 1 + 1e-6 for v in row[33:])
        flags.append(ok)
    return np.array(flags)


def main():
    parser = argparse.ArgumentParser()
    for key in ("run", "cache", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument(
        "--parent", type=Path, default=ROOT / "experiments/runs/chromaseed_gated_v1"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    run, out, parent = args.run, args.output, args.parent
    result, lock = js(run / "results.json"), js(run / "source_lock.json")
    assert sha(run / "source_lock.json") == result["source_lock_sha256"]
    assert sha(args.cache) == lock["cache_sha256"] == CACHE_HASH
    for path, value in {**lock["sources"], **lock["input_sha256"]}.items():
        assert sha(ROOT / path) == value, path
    for path, value in result["numeric_sha256"].items():
        assert sha(run / path) == value
    with np.load(args.cache, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("color", "target", "patient", "device")}
    counts = dict(
        models=0,
        transform_cases=0,
        transformed_query_rows=0,
        dose_cases=0,
        boundary_models=0,
        legal_boundary_pairs=0,
        boundary_control_query_rows=0,
        projected_rows=0,
    )
    maxima = dict(
        ordinary_prediction=0.0, gate_score=0.0, boundary_prediction=0.0, root=0.0, projection=0.0
    )
    settings = [(0.0, np.zeros(3), -1)] + [(t, a, j) for t in DOSES for j, a in enumerate(ANCHORS)]
    for role in ("mixed", "slr_to_ipod", "ipod_to_slr"):
        for rec in (r for r in result["records"] if r["role"] == role):
            name = f"{rec['family']}_s{rec['seed']}.npz"
            m, saved = nz(parent / "selected" / role / name), nz(run / rec["numeric_file"])
            original = nz(parent / "evaluated" / role / name)
            np.testing.assert_array_equal(saved["row_indices"], original["row_indices"])
            rows = saved["row_indices"]
            x, target, person, camera = (
                data[k][rows] for k in ("color", "target", "patient", "device")
            )
            p = Predictor(m)
            output = []
            gates = []
            for i, (t, anchor, ai) in enumerate(settings):
                xx = transformed(x, t, anchor)
                pred = np.array([p(row) for row in xx])
                drift = float(np.abs(pred - saved["prediction"][i]).max())
                maxima["ordinary_prediction"] = max(maxima["ordinary_prediction"], drift)
                assert drift <= 2e-8
                output.append(pred)
                if "gate_beta" in m:
                    s = score(m, xx)
                    maxima["gate_score"] = max(
                        maxima["gate_score"], float(np.abs(s - saved["scores"][i]).max())
                    )
                    np.testing.assert_allclose(s, saved["scores"][i], rtol=0, atol=2e-10)
                    gates.append(s)
                close(rec["transforms"][i]["dose"], t)
                assert rec["transforms"][i]["anchor_index"] == ai
                np.testing.assert_array_equal(rec["transforms"][i]["anchor"], anchor)
                counts["transform_cases"] += 1
                counts["transformed_query_rows"] += len(rows)
            output = np.array(output)
            np.testing.assert_allclose(output[0], original["prediction"], atol=2e-8, rtol=0)
            err = delta_e00(output, target[None])
            drift = delta_e00(output, output[0][None])
            flips = None if not gates else (np.array(gates) >= 0) != (np.array(gates)[0:1] >= 0)
            for i, rec_t in enumerate(rec["transforms"]):
                for key, values in (
                    ("error", err[i]),
                    ("drift", drift[i]),
                    ("error_change", err[i] - err[0]),
                ):
                    close(rec_t[key], metric(values, person, camera))
                close(
                    rec_t["sign_flip"], None if flips is None else metric(flips[i], person, camera)
                )
            for i, dose in enumerate(DOSES):
                indices = np.arange(1 + 8 * i, 9 + 8 * i)
                rec_d = rec["doses"][i]
                close(rec_d["dose"], dose)
                worst = np.max(err[indices], axis=0)
                for key, values in (
                    ("worst_error", worst),
                    ("worst_drift", np.max(drift[indices], axis=0)),
                    ("worst_error_change", worst - err[0]),
                ):
                    close(rec_d[key], metric(values, person, camera))
                close(
                    rec_d["any_sign_flip"],
                    None
                    if flips is None
                    else metric(np.any(flips[indices], axis=0), person, camera),
                )
                counts["dose_cases"] += 1
            counts["models"] += 1
        print(f"AUDIT {role}:all24x33 outputs and person/camera metrics pass", flush=True)
    for rec in result["boundaries"]:
        base, seed = rec["base"], rec["seed"]
        saved = nz(run / "boundary" / f"{base}_hard_s{seed}.npz")
        models = {
            key: nz(parent / "selected/mixed" / f"{base}_{key}_s{seed}.npz")
            for key in ("base", "uniform", "soft", "hard")
        }
        m = models["hard"]
        np.testing.assert_array_equal(m["gate_beta"], models["soft"]["gate_beta"])
        rows = saved["row_indices"]
        np.testing.assert_array_equal(
            rows, nz(parent / "evaluated/mixed" / f"{base}_hard_s{seed}.npz")["row_indices"]
        )
        x, person, camera = (data[k][rows] for k in ("color", "patient", "device"))
        z = normalized(m, x)
        beta = m["gate_beta"].astype(np.float64)
        scores = np.sum(z * beta[None, 1:], axis=1) + beta[0]
        direction = beta[1:] / np.linalg.norm(beta[1:])
        signed_distance = scores / np.linalg.norm(beta[1:])
        projected = z - signed_distance[:, None] * direction[None]
        distance = np.abs(signed_distance) / 6
        maxima["projection"] = max(
            maxima["projection"], float(np.abs(projected - saved["projected_z"]).max())
        )
        np.testing.assert_allclose(projected, saved["projected_z"], atol=2e-12, rtol=0)
        np.testing.assert_allclose(projected @ beta[1:] + beta[0], 0, atol=1e-12)
        np.testing.assert_allclose(distance, saved["distance"], atol=1e-12, rtol=0)
        raw = projected * m["x_std"] + m["x_mean"]
        np.testing.assert_allclose(raw, saved["projected_raw"], atol=2e-12, rtol=0)
        legal = feasible(raw)
        np.testing.assert_array_equal(legal, saved["basic_legal"])
        k = direct_kernel(projected, m["centers"], float(m["width"]))
        plus = (
            k
            @ (
                m["coefficient"].astype(np.float64)
                + float(m["rho"]) * m["correction"].astype(np.float64)
            )
        ) * m["y_std"] + m["y_mean"]
        minus = (
            k
            @ (
                m["coefficient"].astype(np.float64)
                - float(m["rho"]) * m["correction"].astype(np.float64)
            )
        ) * m["y_std"] + m["y_mean"]
        for name, value in (("plus", plus), ("minus", minus)):
            np.testing.assert_allclose(value, saved[name], atol=2e-9, rtol=0)
        close(rec["unconstrained_distance"], metric(distance, person, camera))
        close(rec["unconstrained_jump"], metric(delta_e00(plus, minus), person, camera))
        close(rec["unconstrained_basic_bounds"], metric(legal, person, camera))
        locations, ais, roots = [], [], []
        for j, a in enumerate(ANCHORS):
            d = np.zeros_like(x, dtype=np.float64)
            for channel in range(3):
                indices = np.arange(channel, 30, 3)
                d[:, indices] = a[channel] - x[:, indices].astype(np.float64)
                d[:, 30 + channel] = -x[:, 30 + channel].astype(np.float64)
            slope = np.sum(d * (beta[None, 1:] / m["x_std"]), axis=1)
            root = np.divide(-scores, slope, out=np.full(len(x), np.nan), where=slope != 0)
            ix = np.flatnonzero((root > EPS) & (root < DOSES[-1] - EPS))
            locations.extend(ix.tolist())
            ais.extend([j] * len(ix))
            roots.extend(root[ix].tolist())
        loc, ai, roots = np.array(locations, int), np.array(ais, int), np.array(roots)
        np.testing.assert_array_equal(loc, saved["query_ordinal"])
        np.testing.assert_array_equal(ai, saved["anchor_index"])
        maxima["root"] = max(
            maxima["root"], float(np.max(np.abs(roots - saved["root_t"]), initial=0))
        )
        np.testing.assert_allclose(roots, saved["root_t"], atol=2e-12, rtol=0)
        lo, hi = (
            transformed(x[loc], roots - EPS, ANCHORS[ai]),
            transformed(x[loc], roots + EPS, ANCHORS[ai]),
        )
        np.testing.assert_allclose(lo, saved["lower_x"], atol=8e-8, rtol=0)
        np.testing.assert_allclose(hi, saved["upper_x"], atol=8e-8, rtol=0)
        pair_scores = np.stack((score(m, lo), score(m, hi)))
        np.testing.assert_allclose(pair_scores, saved["pair_scores"], atol=2e-6, rtol=0)
        crossed = (pair_scores[0] >= 0) != (pair_scores[1] >= 0)
        np.testing.assert_array_equal(crossed, saved["crossed"])
        assert (
            len(loc) == rec["legal_boundary_pairs"]
            and int(crossed.sum()) == rec["actual_sign_flips"]
        )
        assert len(np.unique(person[loc])) == rec["legal_boundary_people"]
        close(rec["actual_pair_rgb_distance_bound"], 2 * EPS)
        for key, model in models.items():
            pred = Predictor(model)
            values = np.array([[pred(row) for row in xx] for xx in (lo, hi)])
            maxima["boundary_prediction"] = max(
                maxima["boundary_prediction"],
                float(np.max(np.abs(values - saved[f"prediction_{key}"]), initial=0)),
            )
            np.testing.assert_allclose(values, saved[f"prediction_{key}"], atol=0.001, rtol=0)
            close(
                rec["jumps"][key],
                metric(delta_e00(values[0], values[1]), person[loc], camera[loc]),
                tol=0.001,
            )
            counts["boundary_control_query_rows"] += 2 * len(loc)
        counts["boundary_models"] += 1
        counts["legal_boundary_pairs"] += len(loc)
        counts["projected_rows"] += len(rows)
        print(
            f"AUDIT boundary {base} seed{seed}: {len(loc)} legal pairs +{len(rows)} unconstrained points",
            flush=True,
        )
    assert counts == dict(
        models=72,
        transform_cases=2376,
        transformed_query_rows=948816,
        dose_cases=288,
        boundary_models=6,
        legal_boundary_pairs=3204,
        boundary_control_query_rows=25632,
        projected_rows=1392,
    )
    dependencies = (
        "scripts/chromaseed_gated_numpy.py",
        "scripts/chromaseed_kernel_audit.py",
        "scripts/skin_local_search_train.py",
        "src/luma_skin_vision/color.py",
    )
    write_json(
        out / "audit.json",
        dict(
            passed=True,
            source_lock_sha256=sha(run / "source_lock.json"),
            results_sha256=sha(run / "results.json"),
            audit_source_sha256=sha(Path(__file__)),
            dependencies={p: sha(ROOT / p) for p in dependencies},
            checks=counts,
            maxima=maxima,
            elapsed_seconds=time.perf_counter() - started,
            scope="Independent feature algebra, actual NumPy batch-one consumer, all person/camera metrics and boundary constructions. Shared verified CIEDE2000. Synthetic sensitivity only; no new fits or target-image measurements.",
        ),
    )


if __name__ == "__main__":
    main()
