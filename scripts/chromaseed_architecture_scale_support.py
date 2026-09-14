"""Inner-only diagnostic of the fixed +/-one-target-std residual output range."""

from __future__ import annotations

import numpy as np
from chromaseed_architecture_scale_run import OUT, ROOT, RUN, TIMES, VARIANTS, bank_path, load_data
from chromaseed_gated_audit import model_from
from chromaseed_kernel_audit import js, nz
from chromaseed_neural_prefix_numpy import predict as warm_predict
from chromaseed_widen_run import bank_path as warm_bank
from skin_local_search_train import roles, sha, write_json

MARGIN = 1e-4
NEAR = 0.95


def person_average(values, person):
    """Equal-person means, then equal means over the three separate seeds."""
    return np.mean([values[:, person == p].mean(axis=1) for p in np.unique(person)], axis=(0, 1))


def distance_to_box(target, anchor, scale):
    normalized = (target - anchor) / scale
    excess = np.maximum(np.abs(normalized) - 1.0, 0.0) * scale
    return normalized, np.sqrt(np.sum(excess**2, axis=-1))


def main():
    # Exact Euclidean projection on a box, not a CIEDE2000 lower bound.
    _, sanity = distance_to_box(np.array([[0.0, 2.0, -3.0]]), np.zeros((1, 3)), np.ones((1, 3)))
    np.testing.assert_allclose(sanity, [np.sqrt(5.0)], atol=1e-15, rtol=0)
    source = sha(RUN / "source_lock.json")
    lock = js(RUN / "source_lock.json")
    assert (
        sha(ROOT / "scripts/chromaseed_architecture_scale.py")
        == lock["sources"]["scripts/chromaseed_architecture_scale.py"]
    )
    data = load_data()
    inputs, records, ranges = {}, [], []
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        fit = np.flatnonzero(mask)
        fold_rows, anchors, scales, needed, distances = [], [], [], [], []
        for fold in range(3):
            path = warm_bank(role, "m31", fold)
            for name in ("rows.npz", "warm_models.npz"):
                inputs[(path / name).relative_to(ROOT).as_posix()] = sha(path / name)
            query = nz(path / "rows.npz")["query_rows"]
            warm = nz(path / "warm_models.npz")
            ms = [model_from(warm, str(si)) for si in range(3)]
            anchor = np.stack([warm_predict(m, data["color"][query]) for m in ms])
            scale = np.stack([m["y_std"] for m in ms]).astype(float)[:, None, :]
            norm, distance = distance_to_box(data["target"][query][None], anchor, scale)
            fold_rows.append(query)
            anchors.append(anchor)
            scales.append(scale)
            needed.append(norm)
            distances.append(distance)
        query = np.concatenate(fold_rows)
        order = np.argsort(query)
        np.testing.assert_array_equal(query[order], fit)
        person = data["patient"][fit]
        required = np.concatenate(needed, axis=1)[:, order]
        euclidean = np.concatenate(distances, axis=1)[:, order]
        outside = np.abs(required) > 1 + MARGIN
        ranges.append(
            dict(
                role=role,
                rows=len(fit),
                people=len(np.unique(person)),
                outside_any_person_fraction=float(person_average(outside.any(-1), person)),
                outside_component_person_fractions=person_average(outside, person).tolist(),
                mean_person_euclidean_distance_to_box=float(person_average(euclidean, person)),
                max_required_normalized_correction=float(np.abs(required).max()),
                interpretation="Known inner targets beyond a necessary output box; Euclidean Lab distance is not DeltaE00",
            )
        )
        for variant in VARIANTS:
            paths = [bank_path(role, variant, f) for f in range(3)]
            if not all((p / "receipt.json").exists() for p in paths):
                continue
            receipts = [js(p / "receipt.json") for p in paths]
            assert all(
                r["source_lock_sha256"] == source and r["selection_sha256"] is None
                for r in receipts
            )
            for step in TIMES:
                residuals = []
                for fold, path in enumerate(paths):
                    p = path / f"oof_{step}.npz"
                    assert sha(p) == receipts[fold]["files"][p.name]
                    inputs[p.relative_to(ROOT).as_posix()] = sha(p)
                    saved = nz(p)
                    np.testing.assert_array_equal(saved["row_indices"], fold_rows[fold])
                    residuals.append(
                        (saved["predictions"] - np.repeat(anchors[fold], 2, axis=0))
                        / np.repeat(scales[fold], 2, axis=0)
                    )
                residual = np.concatenate(residuals, axis=1)[:, order]
                assert np.max(np.abs(residual)) <= 1.002, (
                    "exported response exceeds registered correction range"
                )
                for ri, rate in enumerate((1e-5, 1e-4)):
                    r = residual[[2 * si + ri for si in range(3)]]
                    near = np.abs(r) >= NEAR
                    records.append(
                        dict(
                            role=role,
                            variant=variant,
                            step=step,
                            rate=rate,
                            near_boundary_person_fraction=float(
                                person_average(near.any(-1), person)
                            ),
                            near_component_person_fractions=person_average(near, person).tolist(),
                            max_normalized_correction=float(np.abs(r).max()),
                        )
                    )
    value = dict(
        classification="provisional inner-only range diagnostic; no training, final-role evaluation or selector change",
        source_lock_sha256=source,
        margin=MARGIN,
        near_boundary_threshold=NEAR,
        structural_cap="NP anchor +/- fit-only target standard deviation per Lab coordinate; tanh or four0.25*tanh increments",
        target_ranges=ranges,
        learned_residuals=records,
        input_sha256=inputs,
        analysis_sha256=sha(ROOT / "scripts/chromaseed_architecture_scale_support.py"),
    )
    path = OUT / "inner_reviews" / f"residual_support_{len(records)}.json"
    assert not path.exists(), "preserve this snapshot; do not replace a range diagnostic"
    write_json(path, value)
    for r in ranges:
        print("AS INNER TARGET RANGE", r)
    for role in {r["role"] for r in records}:
        for variant in [
            v for v in VARIANTS if any(r["role"] == role and r["variant"] == v for r in records)
        ]:
            rr = [r for r in records if r["role"] == role and r["variant"] == variant]
            print(
                "AS INNER LEARNED RANGE",
                role,
                variant,
                "max near-boundary fraction",
                max(r["near_boundary_person_fraction"] for r in rr),
                "max normalized correction",
                max(r["max_normalized_correction"] for r in rr),
            )
    print("AS RANGE SNAPSHOT", path)


if __name__ == "__main__":
    main()
