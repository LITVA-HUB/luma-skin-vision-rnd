"""Independent ND payload, inner selection, stage and actual-consumer audit."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_affine_audit import summaries
from chromaseed_gate_stability_audit import transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_audit import SETTINGS, actual_consumer
from chromaseed_gaussian_audit import direct as reference_direct
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_numpy import Predictor
from chromaseed_local_denoise_train import CACHE, ROOT, RUN, load_data
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, weights_for, write_json

OUT = ROOT / "docs/benchmarks/chromaseed_local_denoise_v1"
FAMILIES = ("plain", "local2", "local4", "blind4", "e2e4")
SEEDS = (17, 29, 43)


def close(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b), (set(a), set(b))
        for key in a:
            close(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            close(x, y)
    elif isinstance(a, (float, int, np.number)):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=2e-8)
    else:
        assert a == b, (a, b)


def direct(model, x):
    """Separate neuron reduction and algebraic recurrence; no ND forward helper."""
    if "theta" not in model:
        return reference_direct(model, x)[:, None, :]
    family = str(model["family"])
    k, dimension, hidden = {
        "plain": (1, 36, 69),
        "local2": (2, 39, 32),
        "local4": (4, 39, 16),
        "blind4": (4, 39, 16),
        "e2e4": (4, 39, 16),
    }[family]
    xx = ((np.asarray(x, np.float32) - model["x_mean"]) / model["x_std"]).astype(float)
    latent = np.zeros((len(xx), 3))
    curve = []
    for block in range(k):
        row = model["theta"][block].astype(float)
        state = np.zeros_like(latent) if family == "blind4" else latent
        z = xx if family == "plain" else np.column_stack((xx, state))
        weight = row[: dimension * hidden].reshape(dimension, hidden)
        activation = np.maximum(
            np.sum(z[:, :, None] * weight[None], axis=1)
            + row[dimension * hidden : (dimension + 1) * hidden],
            0,
        )
        output_weight = row[(dimension + 1) * hidden : -3].reshape(hidden, 3)
        clean = np.sum(activation[:, :, None] * output_weight[None], axis=1) + row[-3:]
        curve.append(clean * model["y_std"] + model["y_mean"])
        current = np.pi * block / (2 * k)
        following = np.pi * (block + 1) / (2 * k)
        ratio = 0.0 if block + 1 == k else np.cos(following) / np.cos(current)
        alpha_next = 1.0 if block + 1 == k else np.sin(following)
        latent = ratio * latent + (alpha_next - ratio * np.sin(current)) * clean
    return np.stack(curve, axis=1)


def verify_normalizers(model, x, y):
    for prefix, value in (("x", x.astype(float)), ("y", y.astype(float))):
        np.testing.assert_array_equal(model[prefix + "_mean"], value.mean(0).astype(np.float32))
        np.testing.assert_array_equal(
            model[prefix + "_std"], np.maximum(value.std(0), 1e-6).astype(np.float32)
        )


def full_metrics(pred, data, rows):
    value, _ = error_summary(pred, data["target"][rows], data["patient"][rows], data["site"][rows])
    value["camera"] = {
        str(c): error_summary(
            pred[data["device"][rows] == c],
            data["target"][rows][data["device"][rows] == c],
            data["patient"][rows][data["device"][rows] == c],
            data["site"][rows][data["device"][rows] == c],
        )[0]
        for c in np.unique(data["device"][rows])
    }
    return value


def audit_inner(data, source, selection, artifacts):
    counts = dict(inner_banks=0, inner_models=0, inner_stage_rows=0, candidates=0, choices=0)
    maximum = 0.0
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        ix = np.flatnonzero(mask)
        folds = folds_for(data["patient"][ix], data["device"][ix])
        for family in FAMILIES:
            reconstructed = {step: [] for step in (512, 2048, 8192)}
            for fold in range(3):
                path = RUN / "inner" / role / family / f"fold{fold}"
                rec = js(path / "receipt.json")
                assert rec["source_lock_sha256"] == source and rec["selection_sha256"] is None
                fitrows, query = ix[folds != fold], ix[folds == fold]
                rr = nz(path / "rows.npz")
                np.testing.assert_array_equal(rr["fit_rows"], fitrows)
                np.testing.assert_array_equal(rr["query_rows"], query)
                assert not set(data["patient"][fitrows]) & set(data["patient"][query])
                np.testing.assert_allclose(
                    weights_for(data["patient"][fitrows], data["site"][fitrows]),
                    balanced(data["patient"][fitrows], data["site"][fitrows]),
                    rtol=1e-12,
                    atol=1e-12,
                )
                assert (
                    rec["steps"] == 8192
                    and rec["engine"] == "cuda_graph"
                    and rec["max_torch_numpy_native_lab"] <= 0.002
                )
                for name, digest in rec["files"].items():
                    assert sha(path / name) == digest
                    artifacts[(path / name).relative_to(ROOT).as_posix()] = digest
                artifacts[(path / "receipt.json").relative_to(ROOT).as_posix()] = sha(
                    path / "receipt.json"
                )
                for step in (512, 2048, 8192):
                    models = nz(path / f"models_{step}.npz")
                    saved = nz(path / f"oof_{step}.npz")
                    np.testing.assert_array_equal(saved["row_indices"], query)
                    predictions = []
                    for slot in range(6):
                        m = model_from(models, str(slot))
                        verify_normalizers(m, data["color"][fitrows], data["target"][fitrows])
                        actual = direct(m, data["color"][query])
                        maximum = max(maximum, float(np.max(abs(actual - saved["stages"][slot]))))
                        np.testing.assert_allclose(actual, saved["stages"][slot], rtol=0, atol=2e-8)
                        predictions.append(actual)
                        counts["inner_models"] += 1
                        counts["inner_stage_rows"] += len(query) * actual.shape[1]
                    reconstructed[step].append((query, np.stack(predictions)))
                counts["inner_banks"] += 1
            candidates = []
            for step in (512, 2048, 8192):
                rows = np.concatenate([v[0] for v in reconstructed[step]])
                order = np.argsort(rows)
                np.testing.assert_array_equal(rows[order], ix)
                output = np.concatenate([v[1] for v in reconstructed[step]], axis=1)[:, order]
                for li, lr in enumerate((0.001, 0.003)):
                    mm = [
                        error_summary(
                            output[2 * si + li, :, -1],
                            data["target"][ix],
                            data["patient"][ix],
                            data["site"][ix],
                        )[0]
                        for si in range(3)
                    ]
                    candidates.append(
                        dict(
                            step=step,
                            lr=lr,
                            lr_index=li,
                            clean=float(np.mean([v["person_mean"] for v in mm])),
                            p90=float(np.mean([v["p90"] for v in mm])),
                            seed_metrics=mm,
                        )
                    )
            chosen = min(candidates, key=lambda c: (c["clean"], c["p90"], c["step"], c["lr_index"]))
            recorded = selection["roles"][role][family]
            close(recorded["candidates"], candidates)
            close(recorded["selected"], chosen)
            counts["candidates"] += 6
            counts["choices"] += 1
    assert counts == dict(
        inner_banks=45, inner_models=810, inner_stage_rows=459000, candidates=90, choices=15
    ), counts
    return counts, maximum


def audit_final(data, source, selection, result, artifacts):
    counts = dict(
        final_models=0, exact_NS_references=0, stage_rows=0, consumer_calls=0, transforms=0, doses=0
    )
    maximum, consumer_max = 0.0, 0.0
    person_errors = {}
    for rec in result["records"]:
        role, family, seed = rec["role"], rec["family"], rec["seed"]
        fitmask, held = roles(data["patient"], data["device"])[role]
        query = np.flatnonzero(held)
        modelpath = RUN / "selected" / role / f"{rec['name']}.npz"
        predpath = RUN / "evaluated" / role / f"{rec['name']}.npz"
        assert sha(modelpath) == rec["model_sha256"] and sha(predpath) == rec["prediction_sha256"]
        model, saved = nz(modelpath), nz(predpath)
        np.testing.assert_array_equal(saved["row_indices"], query)
        if family in FAMILIES:
            assert str(model["family"]) == family
            setting = selection["roles"][role][family]["selected"]
            assert rec["step"] == setting["step"] and rec["lr"] == setting["lr"]
            bank = RUN / "final" / role / family / f"s{seed}"
            receipt = js(bank / "receipt.json")
            assert receipt["selection_sha256"] == sha(RUN / "selections.json")
            assert receipt["source_lock_sha256"] == source
            assert receipt["slots"] == [dict(seed=seed, lr=setting["lr"])]
            assert (
                receipt["steps"] == setting["step"]
                and receipt["max_torch_numpy_native_lab"] <= 0.002
            )
            for name, digest in receipt["files"].items():
                assert sha(bank / name) == digest
                artifacts[(bank / name).relative_to(ROOT).as_posix()] = digest
            artifacts[(bank / "receipt.json").relative_to(ROOT).as_posix()] = sha(
                bank / "receipt.json"
            )
            original = model_from(nz(bank / f"models_{setting['step']}.npz"), "0")
            for key in model:
                np.testing.assert_array_equal(model[key], original[key])
            verify_normalizers(model, data["color"][fitmask], data["target"][fitmask])
            rows = nz(bank / "rows.npz")
            np.testing.assert_array_equal(rows["fit_rows"], np.flatnonzero(fitmask))
            assert rows["query_rows"].size == 0
            assert model["theta"].size == rec["parameters"]
            predictor = Predictor(model)
        else:
            original = (
                ROOT
                / "experiments/runs/chromaseed_neural_shrinkage_v1/selected"
                / role
                / f"{rec['ns_name']}.npz"
            )
            assert sha(original) == rec["ns_model_sha256"]
            old = nz(original)
            assert set(old) == set(model)
            for key in model:
                np.testing.assert_array_equal(old[key], model[key])
            predictor = actual_consumer(model)
            counts["exact_NS_references"] += 1
        assert rec["numeric_bytes"] == sum(
            v.nbytes for v in model.values() if v.dtype.kind in "biufc"
        )
        assert rec["archive_bytes"] == modelpath.stat().st_size
        all_outputs = []
        for index, (amount, anchor) in enumerate(SETTINGS):
            xx = transformed(data["color"][query], amount, anchor)
            predicted = direct(model, xx)
            maximum = max(maximum, float(np.max(abs(predicted - saved["stages"][index]))))
            np.testing.assert_allclose(predicted, saved["stages"][index], rtol=0, atol=2e-8)
            single = np.stack([predictor(row) for row in xx])
            consumer_max = max(consumer_max, float(np.max(abs(single - predicted[:, -1]))))
            np.testing.assert_allclose(single, predicted[:, -1], rtol=0, atol=2e-8)
            all_outputs.append(predicted)
            counts["consumer_calls"] += len(query)
            counts["stage_rows"] += len(query) * predicted.shape[1]
        output = np.stack(all_outputs)
        close(rec["metrics"], full_metrics(output[0, :, -1], data, query))
        mm = [
            error_summary(
                output[0, :, j], data["target"][query], data["patient"][query], data["site"][query]
            )[0]
            for j in range(output.shape[2])
        ]
        close(rec["stage_metrics"], mm)
        summaries(
            rec,
            output[:, :, -1],
            data["target"][query],
            data["patient"][query],
            data["device"][query],
            None,
        )
        person_errors[(role, family, seed)] = error_summary(
            output[0, :, -1], data["target"][query], data["patient"][query], data["site"][query]
        )[1]
        counts["final_models"] += 1
        counts["transforms"] += 33
        counts["doses"] += 4
        artifacts[modelpath.relative_to(ROOT).as_posix()] = sha(modelpath)
        artifacts[predpath.relative_to(ROOT).as_posix()] = sha(predpath)
        if counts["final_models"] % 7 == 0:
            print("AUDIT final", counts["final_models"], "/63", flush=True)
    assert counts == dict(
        final_models=63,
        exact_NS_references=18,
        stage_rows=2016234,
        consumer_calls=830214,
        transforms=2079,
        doses=252,
    ), counts
    return counts, maximum, consumer_max, person_errors


def paired(errors):
    result = []
    for role in ("mixed", "slr_to_ipod", "ipod_to_slr"):
        values = {
            family: np.mean([errors[(role, family, seed)] for seed in SEEDS], axis=0)
            for family in (*FAMILIES, "fg_norm_static", "random_head")
        }
        n = len(values["plain"])
        draws = np.random.default_rng(771031).integers(n, size=(20000, n))
        pairs = [
            (family, reference)
            for family in FAMILIES
            for reference in ("plain", "fg_norm_static", "random_head")
            if family != reference
        ]
        pairs += [("local4", "blind4"), ("local4", "e2e4")]
        for family, reference in pairs:
            difference = values[family] - values[reference]
            result.append(
                dict(
                    role=role,
                    family=family,
                    reference=reference,
                    mean_difference=float(difference.mean()),
                    people=n,
                    people_improved=int(np.sum(difference < 0)),
                    descriptive_95=np.quantile(difference[draws].mean(1), [0.025, 0.975]).tolist(),
                )
            )
    assert len(result) == 48
    return result


def main():
    started = time.perf_counter()
    lock, selection, result = [
        js(RUN / name) for name in ("source_lock.json", "selections.json", "results.json")
    ]
    source = sha(RUN / "source_lock.json")
    for p, h in {**lock["sources"], **lock["input_sha256"]}.items():
        assert sha(ROOT / p) == h, p
    assert sha(CACHE) == CACHE_HASH
    assert selection["source_lock_sha256"] == result["source_lock_sha256"] == source
    assert result["selection_sha256"] == sha(RUN / "selections.json")
    data = load_data()
    artifacts = {}
    a, amax = audit_inner(data, source, selection, artifacts)
    print("AUDIT inner complete", a, flush=True)
    b, bmax, cmax, errors = audit_final(data, source, selection, result, artifacts)
    for name in (
        "source_lock.json",
        "selections.json",
        "results.json",
        "progress_io_recovery.json",
    ):
        path = RUN / name
        artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
    dependencies = {
        f"scripts/{name}.py": sha(ROOT / f"scripts/{name}.py")
        for name in (
            "chromaseed_local_denoise_audit",
            "chromaseed_local_denoise_numpy",
            "chromaseed_refine_audit",
            "chromaseed_affine_audit",
            "chromaseed_gate_stability_audit",
            "chromaseed_gaussian_audit",
            "chromaseed_gated_audit",
        )
    }
    output = dict(
        passed=True,
        source_lock_sha256=source,
        selection_sha256=sha(RUN / "selections.json"),
        results_sha256=sha(RUN / "results.json"),
        counts={**a, **b},
        max_inner_native_lab=amax,
        max_final_native_lab=bmax,
        max_consumer_native_lab=cmax,
        paired=paired(errors),
        artifact_sha256=artifacts,
        dependencies=dependencies,
        seconds=time.perf_counter() - started,
    )
    write_json(OUT / "audit.json", output)
    print("ND AUDIT PASSED", output["counts"], flush=True)


if __name__ == "__main__":
    main()
