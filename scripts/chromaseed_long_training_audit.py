"""Independent LT data/lineage/selection/prediction and actual-consumer audit."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
from chromaseed_affine_audit import summaries
from chromaseed_gate_stability_audit import transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_audit import SETTINGS
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_audit import close, full_metrics, verify_normalizers
from chromaseed_long_training_run import ND, NP, OUT, ROOT, RUN, bank_path, check_map, load_data
from chromaseed_neural_prefix_audit import inspect_export
from chromaseed_neural_prefix_numpy import Predictor
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from chromaseed_refine_train import setup
from skin_local_search_train import folds_for, roles, sha, write_json

SEEDS = (17, 29, 43)
STEPS = (0, 512, 2048, 8192, 32768, 131072)
MODES = (0, 16, 256)
RATES = (0.0001, 0.0003)
SLOTS = [dict(seed=s, variants=v, lr=r) for s in SEEDS for v in MODES for r in RATES]


def direct(model, x):
    xx = ((np.asarray(x, np.float32) - model["x_mean"]) / model["x_std"]).astype(float)
    hidden = np.maximum(
        np.sum(xx[:, :, None] * model["w0"].astype(float)[None], axis=1) + model["b0"], 0
    )
    normalized = np.sum(hidden[:, :, None] * model["v0"].astype(float)[None], axis=1) + model["c0"]
    return normalized * model["y_std"] + model["y_mean"]


def framework(models, x, expected):
    xx = (np.asarray(x, np.float32) - models[0]["x_mean"]) / models[0]["x_std"]
    xt = torch.as_tensor(np.repeat(xx[None], 18, axis=0), device="cuda")
    w, b, v, c = [
        torch.as_tensor(np.stack([m[k] for m in models]), device="cuda")
        for k in ("w0", "b0", "v0", "c0")
    ]
    with torch.no_grad():
        output = torch.bmm(torch.relu(torch.bmm(xt, w) + b[:, None]), v) + c[:, None]
        output = output * torch.as_tensor(models[0]["y_std"], device="cuda") + torch.as_tensor(
            models[0]["y_mean"], device="cuda"
        )
    actual = output.cpu().numpy()
    maximum = float(np.max(abs(actual - expected)))
    assert maximum <= 0.002, maximum
    return maximum


def validate_data(data, ix, receipt):
    x = data["color"][ix]
    pool = np.repeat(x[:, None], 257, axis=1)
    for n, row in enumerate(ix):
        draws = np.random.default_rng(880003 + 1009 * int(row)).random((256, 4))
        t = draws[:, 3, None] * (4 / 255)
        values = np.repeat(x[n : n + 1].astype(float), 256, axis=0)
        values[:, :30] += t * (np.tile(draws[:, :3], (1, 10)) - values[:, :30])
        values[:, 30:33] += t * (-values[:, 30:33])
        pool[n, 1:] = values.astype(np.float32)
    assert hashlib.sha256(pool.tobytes()).hexdigest() == receipt["pool_sha256"]
    probability = balanced(data["patient"][ix], data["site"][ix])
    probability /= probability.sum()
    index = np.stack(
        [
            np.random.default_rng(s + 9001).choice(len(ix), (131072, 64), p=probability)
            for s in SEEDS
        ]
    )
    u = np.stack(
        [np.random.default_rng(s + 910003).random((131072, 64), dtype=np.float32) for s in SEEDS]
    )
    assert hashlib.sha256(index.tobytes()).hexdigest() == receipt["sampling_indices_sha256"]
    assert hashlib.sha256(u.tobytes()).hexdigest() == receipt["uniforms_sha256"]
    inventory = []
    for si, seed in enumerate(SEEDS):
        for count in MODES:
            chosen = np.zeros_like(index[si])
            if count:
                augmented = u[si] >= 0.5
                chosen[augmented] = 1 + np.floor((u[si][augmented] - 0.5) * (2 * count)).astype(
                    np.int64
                )
            occupied = np.bincount(
                (index[si] * 257 + chosen).ravel(), minlength=len(ix) * 257
            ).reshape(len(ix), 257)
            inventory.append(
                dict(
                    seed=seed,
                    variants=count,
                    presentations=int(chosen.size),
                    augmented_presentations=int(np.count_nonzero(chosen)),
                    distinct_observed_variants=int(np.count_nonzero(occupied)),
                    distinct_source_rows=int(np.sum(occupied.sum(1) > 0)),
                )
            )
    close(receipt["sample_inventory"], inventory)
    np.testing.assert_array_equal(
        np.array(receipt["final_learning_rates"], np.float32),
        np.array([s["lr"] for s in SLOTS], np.float32) * np.float32(0.1),
    )


def bank(data, role, fold, source, artifacts):
    path = bank_path(role, fold)
    receipt = js(path / "receipt.json")
    assert receipt["source_lock_sha256"] == source and receipt["slots"] == SLOTS
    assert receipt["steps"] == receipt["schedule_horizon"] == 131072 and receipt[
        "checkpoints"
    ] == list(STEPS)
    assert (
        receipt["trajectory_count"] == 18
        and receipt["engine"] == "cuda_graph"
        and receipt["device"] == "cuda"
    )
    mask, held = roles(data["patient"], data["device"])[role]
    ix = np.flatnonzero(mask)
    if fold is not None:
        assignment = folds_for(data["patient"][ix], data["device"][ix])
        query, ix = ix[assignment == fold], ix[assignment != fold]
    else:
        query = np.flatnonzero(held)
    savedrows = nz(path / "rows.npz")
    np.testing.assert_array_equal(savedrows["fit_rows"], ix)
    np.testing.assert_array_equal(
        savedrows["query_rows"], query if fold is not None else np.array([], np.int64)
    )
    assert not set(data["patient"][ix]) & set(data["patient"][query])
    assert receipt["pool_rows"] == len(ix) * 257 and receipt["original_rows"] == len(ix)
    assert receipt["selection_sha256"] == (
        None if fold is not None else sha(RUN / "selections.json")
    )
    for name, digest in receipt["files"].items():
        assert sha(path / name) == digest
        artifacts[(path / name).relative_to(ROOT).as_posix()] = digest
    artifacts[(path / "receipt.json").relative_to(ROOT).as_posix()] = sha(path / "receipt.json")
    prior = js(NP / "selections.json")["roles"][role]["blind4"]
    j = prior["policies"]["quality"]["prefix"]
    warm = nz(path / "warm_models.npz")
    baseline = nz(path / "models_0.npz")
    for si, seed in enumerate(SEEDS):
        w = model_from(warm, str(si))
        verify_normalizers(w, data["color"][ix], data["target"][ix])
        if fold is None:
            expected = nz(NP / "selected" / role / f"blind4_j{j}_s{seed}.npz")
            for k in expected:
                np.testing.assert_array_equal(w[k], expected[k])
        else:
            old = ND / "inner" / role / "blind4" / f"fold{fold}"
            rows = nz(old / "rows.npz")
            np.testing.assert_array_equal(rows["fit_rows"], ix)
            np.testing.assert_array_equal(rows["query_rows"], query)
            original = model_from(
                nz(old / f"models_{prior['step']}.npz"), str(2 * si + prior["lr_index"])
            )
            inspect_export(original, w, j)
        for offset in range(6):
            b = model_from(baseline, str(si * 6 + offset))
            assert set(b) == set(w)
            for k in w:
                np.testing.assert_array_equal(b[k], w[k])
    validate_data(data, ix, receipt)
    return ix, query


def inner(data, source, selection, artifacts):
    counts = dict(inner_banks=0, inner_models=0, inner_prediction_rows=0, candidates=0, choices=0)
    maximum, gpu_max = 0.0, 0.0
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        collection = {s: [] for s in STEPS}
        for fold in range(3):
            _, query = bank(data, role, fold, source, artifacts)
            path = bank_path(role, fold)
            for step in STEPS:
                packed = nz(path / f"models_{step}.npz")
                models = [model_from(packed, str(i)) for i in range(18)]
                expected = np.stack([direct(m, data["color"][query]) for m in models])
                saved = nz(path / f"oof_{step}.npz")
                np.testing.assert_array_equal(saved["row_indices"], query)
                maximum = max(maximum, float(np.max(abs(expected - saved["predictions"]))))
                np.testing.assert_allclose(expected, saved["predictions"], rtol=0, atol=2e-8)
                gpu_max = max(gpu_max, framework(models, data["color"][query], expected))
                collection[step].append((query, expected))
                counts["inner_models"] += 18
                counts["inner_prediction_rows"] += len(query) * 18
            counts["inner_banks"] += 1
            print("LT AUDIT inner", role, fold, flush=True)
        ix = np.flatnonzero(mask)
        values = {}
        for step, parts in collection.items():
            order = np.argsort(np.concatenate([p[0] for p in parts]))
            np.testing.assert_array_equal(np.concatenate([p[0] for p in parts])[order], ix)
            values[step] = np.concatenate([p[1] for p in parts], axis=1)[:, order]
        candidates = []
        for mi, mode in enumerate(MODES):
            for step in STEPS:
                for li, rate in enumerate((None,) if step == 0 else RATES):
                    mm = [
                        error_summary(
                            values[step][si * 6 + mi * 2 + li],
                            data["target"][ix],
                            data["patient"][ix],
                            data["site"][ix],
                        )[0]
                        for si in range(3)
                    ]
                    candidates.append(
                        dict(
                            variants=mode,
                            step=step,
                            lr=rate,
                            clean=float(np.mean([m["person_mean"] for m in mm])),
                            p90=float(np.mean([m["p90"] for m in mm])),
                            seed_metrics=mm,
                        )
                    )
        chosen = {
            str(v): min(
                [c for c in candidates if c["variants"] == v],
                key=lambda c: (c["clean"], c["p90"], c["step"], c["lr"] or 0),
            )
            for v in MODES
        }
        overall = min(
            candidates, key=lambda c: (c["clean"], c["p90"], c["step"], c["variants"], c["lr"] or 0)
        )
        close(
            selection["roles"][role], dict(candidates=candidates, per_mode=chosen, overall=overall)
        )
        counts["candidates"] += 33
        counts["choices"] += 4
    assert counts == dict(
        inner_banks=9, inner_models=972, inner_prediction_rows=183600, candidates=99, choices=12
    ), counts
    return counts, maximum, gpu_max


def final(data, source, selection, result, artifacts):
    counts = dict(
        final_banks=0,
        final_models=0,
        final_prediction_rows=0,
        consumer_calls=0,
        transforms=0,
        doses=0,
    )
    maximum, gpu_max = 0.0, 0.0
    record_map = {(r["role"], r["step"], r["slot"]): r for r in result["records"]}
    for role in selection["roles"]:
        _, query = bank(data, role, None, source, artifacts)
        for step in STEPS:
            packed = nz(bank_path(role) / f"models_{step}.npz")
            models = [model_from(packed, str(i)) for i in range(18)]
            predicted = []
            for amount, anchor in SETTINGS:
                xx = transformed(data["color"][query], amount, anchor)
                expected = np.stack([direct(m, xx) for m in models])
                gpu_max = max(gpu_max, framework(models, xx, expected))
                predicted.append(expected)
            predicted = np.stack(predicted, axis=1)
            for slot, model in enumerate(models):
                rec = record_map[(role, step, slot)]
                assert all(rec[k] == SLOTS[slot][k] for k in ("seed", "variants", "lr"))
                assert rec["numeric_bytes"] == 2886 and rec["parameters"] == 643
                path, pp = (
                    RUN / "selected" / role / f"{rec['name']}.npz",
                    RUN / "evaluated" / role / f"{rec['name']}.npz",
                )
                assert sha(path) == rec["model_sha256"] and sha(pp) == rec["prediction_sha256"]
                exported, saved = nz(path), nz(pp)
                for key in model:
                    np.testing.assert_array_equal(exported[key], model[key])
                np.testing.assert_array_equal(saved["row_indices"], query)
                expected = predicted[slot]
                maximum = max(maximum, float(np.max(abs(expected - saved["predictions"]))))
                np.testing.assert_allclose(expected, saved["predictions"], rtol=0, atol=2e-8)
                consumer = Predictor(model)
                for i, (amount, anchor) in enumerate(SETTINGS):
                    xx = transformed(data["color"][query], amount, anchor)
                    actual = np.stack([consumer(row) for row in xx])
                    maximum = max(maximum, float(np.max(abs(actual - expected[i]))))
                    np.testing.assert_allclose(actual, expected[i], rtol=0, atol=2e-8)
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
                entry = selection["roles"][role]

                def matches(c):
                    return (
                        step == c["step"]
                        and rec["variants"] == c["variants"]
                        and (slot % 2 == 0 if step == 0 else rec["lr"] == c["lr"])
                    )

                assert rec["selected_per_mode"] == matches(entry["per_mode"][str(rec["variants"])])
                assert rec["selected_overall"] == matches(entry["overall"])
                counts["final_models"] += 1
                counts["final_prediction_rows"] += len(query) * 33
                counts["transforms"] += 33
                counts["doses"] += 4
                for p in (path, pp):
                    artifacts[p.relative_to(ROOT).as_posix()] = sha(p)
            print("LT AUDIT final", role, step, flush=True)
        counts["final_banks"] += 1
    assert counts == dict(
        final_banks=3,
        final_models=324,
        final_prediction_rows=4269672,
        consumer_calls=4269672,
        transforms=10692,
        doses=1296,
    ), counts
    return counts, maximum, gpu_max


def main():
    assert not (OUT / "verification.json").exists(), "sealed LT: report verifier only"
    setup("cuda")
    start = time.perf_counter()
    lock, selection, result = [
        js(RUN / n) for n in ("source_lock.json", "selections.json", "results.json")
    ]
    check_map({**lock["sources"], **lock["input_sha256"]})
    source = sha(RUN / "source_lock.json")
    assert selection["source_lock_sha256"] == result["source_lock_sha256"] == source
    assert result["selection_sha256"] == sha(RUN / "selections.json")
    data, artifacts = load_data(), {}
    a, amax, agpu = inner(data, source, selection, artifacts)
    b, bmax, bgpu = final(data, source, selection, result, artifacts)
    for name in ("source_lock.json", "selections.json", "results.json", "progress.json"):
        path = RUN / name
        artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
    deps = {
        f"scripts/{name}.py": sha(ROOT / f"scripts/{name}.py")
        for name in (
            "chromaseed_long_training_audit",
            "chromaseed_local_denoise_audit",
            "chromaseed_neural_prefix_audit",
            "chromaseed_affine_audit",
            "chromaseed_gate_stability_audit",
            "chromaseed_perceptual_audit",
            "chromaseed_refine_audit",
        )
    }
    value = dict(
        passed=True,
        source_lock_sha256=source,
        selection_sha256=sha(RUN / "selections.json"),
        results_sha256=sha(RUN / "results.json"),
        counts={**a, **b},
        maximum_numpy_audit_drift=max(amax, bmax),
        maximum_cuda_numpy_drift=max(agpu, bgpu),
        artifact_sha256=artifacts,
        dependencies=deps,
        seconds=time.perf_counter() - start,
    )
    write_json(OUT / "audit.json", value)
    print("LT AUDIT PASSED", value["counts"], flush=True)


if __name__ == "__main__":
    main()
