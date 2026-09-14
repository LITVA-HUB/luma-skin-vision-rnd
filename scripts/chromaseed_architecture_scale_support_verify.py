"""CPU geometry check of the provisional output-range diagnosis, not AS model acceptance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from chromaseed_architecture_scale_run import OUT, ROOT, RUN, load_data
from chromaseed_gated_audit import model_from
from chromaseed_kernel_audit import js, nz
from chromaseed_widen_audit import direct
from chromaseed_widen_run import bank_path, check_map
from skin_local_search_train import roles, sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    source = js(args.snapshot)
    assert source["source_lock_sha256"] == sha(RUN / "source_lock.json")
    assert source["margin"] == 1e-4 and source["near_boundary_threshold"] == 0.95
    check_map(source["input_sha256"])
    data = load_data()
    checked = []
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        fit = np.flatnonzero(mask)
        samples = {s: [] for s in range(3)}
        for fold in range(3):
            path = bank_path(role, "m31", fold)
            ix = nz(path / "rows.npz")["query_rows"]
            packed = nz(path / "warm_models.npz")
            for seed in range(3):
                model = model_from(packed, str(seed))
                anchor = direct(model, data["color"][ix], data["tokens"][ix])
                scale = model["y_std"].astype(float)
                lower, upper = anchor - scale, anchor + scale
                target = data["target"][ix]
                projection = np.clip(target, lower, upper)
                outside = (target < lower - 1e-4 * scale) | (target > upper + 1e-4 * scale)
                distance = np.linalg.norm(target - projection, axis=1)
                samples[seed].extend(
                    (int(row), outside[j], distance[j]) for j, row in enumerate(ix)
                )
        seed_components, seed_any, seed_distance = [], [], []
        for rows in samples.values():
            rows.sort(key=lambda r: r[0])
            np.testing.assert_array_equal([r[0] for r in rows], fit)
            person = data["patient"][fit]
            outside = np.stack([r[1] for r in rows])
            distances = np.array([r[2] for r in rows])
            seed_components.append(
                np.mean([outside[person == p].mean(0) for p in np.unique(person)], axis=0)
            )
            seed_any.append(
                np.mean([outside[person == p].any(1).mean() for p in np.unique(person)])
            )
            seed_distance.append(
                np.mean([distances[person == p].mean() for p in np.unique(person)])
            )
        rec = next(r for r in source["target_ranges"] if r["role"] == role)
        np.testing.assert_allclose(
            rec["outside_component_person_fractions"],
            np.mean(seed_components, axis=0),
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            rec["outside_any_person_fraction"], np.mean(seed_any), atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            rec["mean_person_euclidean_distance_to_box"], np.mean(seed_distance), atol=1e-12, rtol=0
        )
        checked.append(
            dict(
                role=role,
                rows=len(fit),
                seeds=3,
                component_and_any_fractions_verified=True,
                independent_direct_NP_and_box_projection=True,
            )
        )
    result = dict(
        passed=True,
        classification="target-range geometry verified; AS learned-model audit still pending",
        snapshot_sha256=sha(args.snapshot),
        checks=checked,
        source_sha256=sha(ROOT / "scripts/chromaseed_architecture_scale_support_verify.py"),
    )
    path = OUT / "inner_reviews" / (args.snapshot.stem + "_geometry_verified.json")
    assert not path.exists()
    write_json(path, result)
    print("AS TARGET-RANGE GEOMETRY VERIFIED", checked)


if __name__ == "__main__":
    main()
