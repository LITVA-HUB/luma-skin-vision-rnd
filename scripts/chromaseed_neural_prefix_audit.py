"""Independent NP pruning, conditional selection and all-consumer audit."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_affine_audit import summaries
from chromaseed_gate_stability_audit import transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_audit import SETTINGS, actual_consumer
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_audit import close, direct, full_metrics, verify_normalizers
from chromaseed_neural_prefix_numpy import Predictor, export_prefix, predict
from chromaseed_neural_prefix_run import ND, ND_OUT, OUT, ROOT, RUN, check_map, load_data
from chromaseed_refine_audit import error_summary
from skin_local_search_train import folds_for, roles, sha, write_json

SIZES = {
    "plain": (1, 36, 69),
    "local2": (2, 39, 32),
    "local4": (4, 39, 16),
    "blind4": (4, 39, 16),
    "e2e4": (4, 39, 16),
}
SEEDS = (17, 29, 43)


def inspect_export(parent, model, prefix):
    """Account for every retained array without calling the NP exporter."""
    family = str(parent["family"])
    k, d, h = SIZES[family]
    assert int(model["original_k"]) == k and int(model["prefix"]) == prefix
    assert str(model["family"]) == family
    assert model["original_k"].dtype == model["prefix"].dtype == np.uint8
    indices = [prefix - 1] if family == "blind4" else list(range(prefix))
    expected = {key: parent[key] for key in ("family", "x_mean", "x_std", "y_mean", "y_std")}
    for i, original_index in enumerate(indices):
        row = parent["theta"][original_index]
        expected[f"w{i}"] = row[: d * h].reshape(d, h)[: 36 if i == 0 else 39]
        expected[f"b{i}"] = row[d * h : (d + 1) * h]
        expected[f"v{i}"] = row[(d + 1) * h : -3].reshape(h, 3)
        expected[f"c{i}"] = row[-3:]
    assert set(model) == set(expected) | {"original_k", "prefix"}
    for key, value in expected.items():
        np.testing.assert_array_equal(model[key], value)
        assert model[key].dtype == value.dtype
    blocks = len(indices)
    parameters = (36 + 1) * h + (h + 1) * 3 + (blocks - 1) * ((39 + 1) * h + (h + 1) * 3)
    return dict(
        parameters=parameters, numeric_bytes=parameters * 4 + 78 * 4 + 2, executed_blocks=blocks
    )


def inner(data, selection):
    counts = dict(
        inner_base_models=0, inner_exports=0, inner_prefix_rows=0, candidates=0, choices=0
    )
    maximum = 0.0
    previous = js(ND / "selections.json")
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        ix = np.flatnonzero(mask)
        fold_id = folds_for(data["patient"][ix], data["device"][ix])
        for family, (k, _, _) in SIZES.items():
            setting = previous["roles"][role][family]["selected"]
            entry = selection["roles"][role][family]
            for key in ("step", "lr", "lr_index"):
                assert entry[key] == setting[key]
            step, li = setting["step"], setting["lr_index"]
            parts, caps = [], {}
            for fold in range(3):
                path = ND / "inner" / role / family / f"fold{fold}"
                rows = nz(path / "rows.npz")
                query, fit_rows = ix[fold_id == fold], ix[fold_id != fold]
                np.testing.assert_array_equal(rows["fit_rows"], fit_rows)
                np.testing.assert_array_equal(rows["query_rows"], query)
                assert not set(data["patient"][fit_rows]) & set(data["patient"][query])
                original, saved = nz(path / f"models_{step}.npz"), nz(path / f"oof_{step}.npz")
                np.testing.assert_array_equal(saved["row_indices"], query)
                output = []
                for si, _ in enumerate(SEEDS):
                    parent = model_from(original, str(2 * si + li))
                    verify_normalizers(parent, data["color"][fit_rows], data["target"][fit_rows])
                    expected = direct(parent, data["color"][query])
                    np.testing.assert_allclose(
                        saved["stages"][2 * si + li], expected, rtol=0, atol=2e-8
                    )
                    for j in range(1, k + 1):
                        model = export_prefix(parent, j)
                        caps[j] = inspect_export(parent, model, j)
                        actual = predict(model, data["color"][query])
                        maximum = max(maximum, float(np.max(abs(actual - expected[:, j - 1]))))
                        np.testing.assert_allclose(actual, expected[:, j - 1], rtol=0, atol=2e-8)
                        counts["inner_exports"] += 1
                        counts["inner_prefix_rows"] += len(query)
                    output.append(expected)
                    counts["inner_base_models"] += 1
                parts.append((query, np.stack(output)))
            rows = np.concatenate([p[0] for p in parts])
            order = np.argsort(rows)
            np.testing.assert_array_equal(rows[order], ix)
            expected = np.concatenate([p[1] for p in parts], axis=1)[:, order]
            candidates = []
            for j in range(1, k + 1):
                mm = [
                    error_summary(
                        expected[si, :, j - 1],
                        data["target"][ix],
                        data["patient"][ix],
                        data["site"][ix],
                    )[0]
                    for si in range(3)
                ]
                candidates.append(
                    dict(
                        prefix=j,
                        clean=float(np.mean([m["person_mean"] for m in mm])),
                        p90=float(np.mean([m["p90"] for m in mm])),
                        seed_metrics=mm,
                        **caps[j],
                    )
                )
            close(entry["candidates"], candidates)
            quality = sorted(
                candidates,
                key=lambda c: (
                    c["clean"],
                    c["p90"],
                    c["numeric_bytes"],
                    c["executed_blocks"],
                    c["prefix"],
                ),
            )[0]
            compact = sorted(
                [c for c in candidates if c["clean"] <= quality["clean"] + 0.10],
                key=lambda c: (
                    c["numeric_bytes"],
                    c["executed_blocks"],
                    c["clean"],
                    c["p90"],
                    c["prefix"],
                ),
            )[0]
            close(entry["policies"], dict(quality=quality, compact=compact))
            counts["candidates"] += k
            counts["choices"] += 2
    assert counts == dict(
        inner_base_models=135, inner_exports=405, inner_prefix_rows=76500, candidates=45, choices=30
    ), counts
    return counts, maximum


def final(data, selection, result, artifacts):
    counts = dict(final_models=0, exact_references=0, consumer_calls=0, transforms=0, doses=0)
    maximum, batch_max = 0.0, 0.0
    errors, cache = {}, {}
    parent_records = {(r["role"], r["name"]): r for r in js(ND / "results.json")["records"]}
    for rec in result["records"]:
        role, family, name, j = rec["role"], rec["family"], rec["name"], rec["prefix"]
        key = (role, rec["parent_name"])
        old = parent_records[key]
        assert rec["seed"] == old["seed"] and rec["step"] == old["step"] and rec["lr"] == old["lr"]
        parentpath = ND / "selected" / role / f"{old['name']}.npz"
        assert sha(parentpath) == rec["parent_model_sha256"] == old["model_sha256"]
        parent = nz(parentpath)
        fitmask, held = roles(data["patient"], data["device"])[role]
        query = np.flatnonzero(held)
        path, pp = RUN / "selected" / role / f"{name}.npz", RUN / "evaluated" / role / f"{name}.npz"
        assert sha(path) == rec["model_sha256"] and sha(pp) == rec["prediction_sha256"]
        model, saved = nz(path), nz(pp)
        np.testing.assert_array_equal(saved["row_indices"], query)
        if key not in cache:
            xx = [transformed(data["color"][query], amount, anchor) for amount, anchor in SETTINGS]
            outputs = np.stack([direct(parent, x) for x in xx])
            old_pred = nz(ND / "evaluated" / role / f"{old['name']}.npz")
            np.testing.assert_allclose(outputs, old_pred["stages"], rtol=0, atol=2e-8)
            cache[key] = (xx, outputs)
        xx, outputs = cache[key]
        if j is None:
            assert set(model) == set(parent) and rec["origin"] == "exact_ND_reference"
            for field in model:
                np.testing.assert_array_equal(model[field], parent[field])
            consumer = actual_consumer(model)
            expected = outputs[:, :, 0]
            counts["exact_references"] += 1
        else:
            cap = inspect_export(parent, model, j)
            close({field: rec[field] for field in cap}, cap)
            assert rec["original_k"] == SIZES[family][0] and rec["full_prefix"] == (
                j == SIZES[family][0]
            )
            assert rec["policies"] == [
                p
                for p, choice in selection["roles"][role][family]["policies"].items()
                if choice["prefix"] == j
            ]
            verify_normalizers(model, data["color"][fitmask], data["target"][fitmask])
            consumer = Predictor(model)
            expected = outputs[:, :, j - 1]
        assert rec["numeric_bytes"] == sum(
            v.nbytes for v in model.values() if v.dtype.kind in "biufc"
        )
        assert rec["archive_bytes"] == path.stat().st_size
        batch_max = max(batch_max, float(np.max(abs(saved["predictions"] - expected))))
        np.testing.assert_allclose(saved["predictions"], expected, rtol=0, atol=2e-8)
        for x, target in zip(xx, expected, strict=True):
            actual = np.stack([consumer(row) for row in x])
            maximum = max(maximum, float(np.max(abs(actual - target))))
            np.testing.assert_allclose(actual, target, rtol=0, atol=2e-8)
            counts["consumer_calls"] += len(query)
        close(rec["metrics"], full_metrics(expected[0], data, query))
        summaries(
            rec,
            expected,
            data["target"][query],
            data["patient"][query],
            data["device"][query],
            None,
        )
        errors[(role, family, j, rec["seed"])] = error_summary(
            expected[0], data["target"][query], data["patient"][query], data["site"][query]
        )[1]
        for p in (path, pp):
            artifacts[p.relative_to(ROOT).as_posix()] = sha(p)
        counts["final_models"] += 1
        counts["transforms"] += 33
        counts["doses"] += 4
        if counts["final_models"] % 17 == 0:
            print("NP AUDIT", counts["final_models"], "/153", flush=True)
    assert counts == dict(
        final_models=153, exact_references=18, consumer_calls=2016234, transforms=5049, doses=612
    ), counts
    return counts, batch_max, maximum, errors


def paired(errors, selection):
    rows = []
    for role, families in selection["roles"].items():
        for family, entry in families.items():
            full_j = SIZES[family][0]
            for policy, choice in entry["policies"].items():
                j = choice["prefix"]
                value = np.mean([errors[(role, family, j, s)] for s in SEEDS], axis=0)
                n = len(value)
                draws = np.random.default_rng(771031).integers(n, size=(20000, n))
                for ref, ref_j in (
                    (family, full_j),
                    ("fg_norm_static", None),
                    ("random_head", None),
                ):
                    baseline = np.mean([errors[(role, ref, ref_j, s)] for s in SEEDS], axis=0)
                    difference = value - baseline
                    rows.append(
                        dict(
                            role=role,
                            family=family,
                            policy=policy,
                            prefix=j,
                            reference=ref,
                            reference_prefix=ref_j,
                            mean_difference=float(difference.mean()),
                            people=n,
                            people_improved=int(np.sum(difference < 0)),
                            descriptive_95=np.quantile(
                                difference[draws].mean(1), [0.025, 0.975]
                            ).tolist(),
                        )
                    )
    assert len(rows) == 90
    return rows


def main():
    assert not (OUT / "verification.json").exists(), (
        "sealed measurement: use report read-only verifier"
    )
    start = time.perf_counter()
    lock, selection, result = [
        js(RUN / name) for name in ("source_lock.json", "selections.json", "results.json")
    ]
    check_map({**lock["sources"], **lock["input_sha256"]})
    source = sha(RUN / "source_lock.json")
    assert selection["source_lock_sha256"] == result["source_lock_sha256"] == source
    assert result["selection_sha256"] == sha(RUN / "selections.json")
    data, artifacts = load_data(), {}
    a, amax = inner(data, selection)
    print("NP INNER PASSED", a, flush=True)
    b, bmax, cmax, errors = final(data, selection, result, artifacts)
    for name in ("source_lock.json", "selections.json", "results.json", "progress.json"):
        path = RUN / name
        artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
    dependencies = {
        f"scripts/{name}.py": sha(ROOT / f"scripts/{name}.py")
        for name in (
            "chromaseed_neural_prefix_audit",
            "chromaseed_local_denoise_audit",
            "chromaseed_gaussian_audit",
            "chromaseed_affine_audit",
            "chromaseed_gate_stability_audit",
            "chromaseed_refine_audit",
        )
    }
    output = dict(
        passed=True,
        source_lock_sha256=source,
        selection_sha256=sha(RUN / "selections.json"),
        results_sha256=sha(RUN / "results.json"),
        counts={**a, **b},
        max_inner_native_lab=amax,
        max_batch_native_lab=bmax,
        max_consumer_native_lab=cmax,
        paired=paired(errors, selection),
        artifact_sha256=artifacts,
        dependencies=dependencies,
        seconds=time.perf_counter() - start,
    )
    write_json(OUT / "audit.json", output)
    print("NP AUDIT PASSED", output["counts"], flush=True)


if __name__ == "__main__":
    main()
