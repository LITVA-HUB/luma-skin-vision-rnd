"""Independent WA scalar means, inner selection and actual final consumers."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_affine_audit import summaries
from chromaseed_gate_stability_audit import transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_audit import SETTINGS
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_audit import close, full_metrics
from chromaseed_long_training_audit import direct
from chromaseed_neural_prefix_numpy import Predictor
from chromaseed_refine_audit import error_summary
from chromaseed_weight_average_run import LT, OUT, ROOT, RUN, check_map, load_data
from skin_local_search_train import folds_for, roles, sha, write_json

SEEDS = (17, 29, 43)
TIMES = (0, 512, 2048, 8192, 32768, 131072)


def definitions():
    result = [dict(name="base", method="last", variants=0, lr=0.0001, steps=[0])]
    for v in (0, 16, 256):
        for i, r in enumerate((0.0001, 0.0003)):
            parts = [("last", [x]) for x in TIMES[1:]]
            parts += [("pair", [a, b]) for a, b in zip(TIMES[:-1], TIMES[1:], strict=True)]
            parts += [
                ("prefix", [512, 2048, 8192]),
                ("prefix", [512, 2048, 8192, 32768]),
                ("prefix", [512, 2048, 8192, 32768, 131072]),
                ("tail3", [2048, 8192, 32768]),
                ("tail3", [8192, 32768, 131072]),
            ]
            for method, steps in parts:
                result.append(
                    dict(
                        name=f"{method}_v{v}_r{i}_t{steps[-1]}",
                        method=method,
                        variants=v,
                        lr=r,
                        steps=steps,
                    )
                )
    return result


def parents(role, fold):
    path = (
        LT
        / ("final" if fold is None else "inner")
        / role
        / ("bank" if fold is None else f"fold{fold}")
    )
    return {s: nz(path / f"models_{s}.npz") for s in TIMES}


def scalar_average(spec, seed_index, bank):
    slot = (
        seed_index * 6
        + (0, 16, 256).index(spec["variants"]) * 2
        + (0.0001, 0.0003).index(spec["lr"])
    )
    components = [model_from(bank[s], str(slot)) for s in spec["steps"]]
    result = {k: v.copy() for k, v in components[0].items()}
    for k in ("w0", "b0", "v0", "c0"):
        result[k] = np.array(
            [
                sum(float(m[k].flat[j]) for m in components) / len(components)
                for j in range(result[k].size)
            ],
            np.float32,
        ).reshape(result[k].shape)
    return result, components


def exact(a, b):
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def rank(c):
    return (
        c["clean"],
        c["p90"],
        c["steps"][-1],
        len(c["steps"]),
        c["variants"],
        c["lr"],
        ["last", "pair", "prefix", "tail3"].index(c["method"]),
    )


def associations(entry):
    result = {"base": dict(policies=[], endpoint_for=[], baseline=True)}
    for key, c in entry["policies"].items():
        result.setdefault(c["name"], dict(policies=[], endpoint_for=[], baseline=False))[
            "policies"
        ].append(key)
        end = (
            "base"
            if c["steps"] == [0]
            else f"last_v{c['variants']}_r{(0.0001, 0.0003).index(c['lr'])}_t{c['steps'][-1]}"
        )
        result.setdefault(end, dict(policies=[], endpoint_for=[], baseline=False))[
            "endpoint_for"
        ].append(key)
    return result


def main():
    assert not (OUT / "verification.json").exists(), "sealed WA: report verifier only"
    began = time.perf_counter()
    lock, selection, result = [
        js(RUN / n) for n in ("source_lock.json", "selections.json", "results.json")
    ]
    check_map({**lock["sources"], **lock["input_sha256"]})
    source = sha(RUN / "source_lock.json")
    assert selection["source_lock_sha256"] == result["source_lock_sha256"] == source
    assert result["selection_sha256"] == sha(RUN / "selections.json")
    specs = definitions()
    assert specs == lock["recipes"] and len(specs) == 91
    counts = dict(
        inner_models=0,
        inner_prediction_rows=0,
        candidates=0,
        policies=0,
        final_models=0,
        final_prediction_rows=0,
        consumer_calls=0,
        singleton_controls=0,
    )
    data, artifacts = load_data(), {}
    maximum = 0.0
    for role, (mask, held) in roles(data["patient"], data["device"]).items():
        full = np.flatnonzero(mask)
        folds = folds_for(data["patient"][full], data["device"][full])
        parts = []
        for fold in range(3):
            fit, query = full[folds != fold], full[folds == fold]
            assert not set(data["patient"][fit]) & set(data["patient"][query])
            path = RUN / "inner" / role / f"fold{fold}"
            receipt = js(path / "receipt.json")
            assert receipt["source_lock_sha256"] == source and receipt["models"] == 273
            for n, d in receipt["files"].items():
                assert sha(path / n) == d
                artifacts[(path / n).relative_to(ROOT).as_posix()] = d
            artifacts[(path / "receipt.json").relative_to(ROOT).as_posix()] = sha(
                path / "receipt.json"
            )
            rows = nz(path / "rows.npz")
            np.testing.assert_array_equal(rows["fit_rows"], fit)
            np.testing.assert_array_equal(rows["query_rows"], query)
            old = nz(LT / "inner" / role / f"fold{fold}" / "rows.npz")
            np.testing.assert_array_equal(old["fit_rows"], fit)
            np.testing.assert_array_equal(old["query_rows"], query)
            bank = parents(role, fold)
            packed, saved = nz(path / "models.npz"), nz(path / "oof.npz")
            np.testing.assert_array_equal(saved["row_indices"], query)
            names, expected = [], []
            for spec in specs:
                for si, seed in enumerate(SEEDS):
                    name = f"{spec['name']}_s{seed}"
                    model, _ = scalar_average(spec, si, bank)
                    exact(model, model_from(packed, name))
                    names.append(name)
                    expected.append(direct(model, data["color"][query]))
                    counts["inner_models"] += 1
                    counts["inner_prediction_rows"] += len(query)
            expected = np.stack(expected)
            np.testing.assert_array_equal(saved["names"], names)
            np.testing.assert_allclose(expected, saved["predictions"], rtol=0, atol=2e-8)
            maximum = max(maximum, float(np.max(abs(expected - saved["predictions"]))))
            parts.append((query, expected))
            print("WA AUDIT inner", role, fold, flush=True)
        order = np.argsort(np.concatenate([p[0] for p in parts]))
        np.testing.assert_array_equal(np.concatenate([p[0] for p in parts])[order], full)
        values = np.concatenate([p[1] for p in parts], axis=1)[:, order]
        candidates = []
        for k, spec in enumerate(specs):
            mm = [
                error_summary(
                    values[k * 3 + si],
                    data["target"][full],
                    data["patient"][full],
                    data["site"][full],
                )[0]
                for si in range(3)
            ]
            candidates.append(
                dict(
                    **spec,
                    clean=float(np.mean([m["person_mean"] for m in mm])),
                    p90=float(np.mean([m["p90"] for m in mm])),
                    seed_metrics=mm,
                )
            )
        policies = {
            m: min([c for c in candidates if c["method"] == m], key=rank)
            for m in ("last", "pair", "prefix", "tail3")
        }
        policies["overall"] = min(candidates, key=rank)
        close(selection["roles"][role], dict(candidates=candidates, policies=policies))
        old = js(LT / "selections.json")["roles"][role]["overall"]
        last = policies["last"]
        assert (last["variants"], last["steps"][-1], last["lr"] if last["steps"][-1] else None) == (
            old["variants"],
            old["step"],
            old["lr"],
        )
        close(last["clean"], old["clean"])
        counts["candidates"] += 91
        counts["policies"] += 5
        query = np.flatnonzero(held)
        bank = parents(role, None)
        union = associations(selection["roles"][role])
        records = {r["name"]: r for r in result["records"] if r["role"] == role}
        assert len(records) == len(union) * 3
        lookup = {s["name"]: s for s in specs}
        for recipe_name, flags in union.items():
            spec = lookup[recipe_name]
            for si, seed in enumerate(SEEDS):
                name = f"{recipe_name}_s{seed}"
                rec = records[name]
                for k, v in {**{k: v for k, v in spec.items() if k != "name"}, **flags}.items():
                    assert rec[k] == v
                model, components = scalar_average(spec, si, bank)
                path, pp = (
                    RUN / "selected" / role / f"{name}.npz",
                    RUN / "evaluated" / role / f"{name}.npz",
                )
                exact(model, nz(path))
                assert sha(path) == rec["model_sha256"] and sha(pp) == rec["prediction_sha256"]
                assert (
                    rec["numeric_bytes"]
                    == sum(a.nbytes for a in model.values() if a.dtype.kind in "biufc")
                    == 2886
                )
                assert rec["parameters"] == 643 and rec["archive_bytes"] == path.stat().st_size
                saved = nz(pp)
                np.testing.assert_array_equal(saved["row_indices"], query)
                output = []
                consumer = Predictor(model)
                for dose, anchor in SETTINGS:
                    xx = transformed(data["color"][query], dose, anchor)
                    pred = direct(model, xx)
                    actual = np.stack([consumer(row) for row in xx])
                    np.testing.assert_allclose(actual, pred, rtol=0, atol=2e-8)
                    maximum = max(maximum, float(np.max(abs(actual - pred))))
                    output.append(pred)
                    counts["consumer_calls"] += len(query)
                output = np.stack(output)
                np.testing.assert_allclose(output, saved["predictions"], rtol=0, atol=2e-8)
                maximum = max(maximum, float(np.max(abs(output - saved["predictions"]))))
                close(rec["metrics"], full_metrics(output[0], data, query))
                summaries(
                    rec,
                    output,
                    data["target"][query],
                    data["patient"][query],
                    data["device"][query],
                    None,
                )
                ensemble = np.mean([direct(c, data["color"][query]) for c in components], axis=0)
                close(
                    rec["weight_vs_prediction_mean_max_lab"],
                    float(np.max(abs(output[0] - ensemble))),
                )
                if len(spec["steps"]) == 1:
                    p = (
                        LT
                        / "evaluated"
                        / role
                        / f"v{spec['variants']}_r{(0.0001, 0.0003).index(spec['lr'])}_t{spec['steps'][0]}_s{seed}.npz"
                    )
                    np.testing.assert_array_equal(saved["predictions"], nz(p)["predictions"])
                    counts["singleton_controls"] += 1
                counts["final_models"] += 1
                counts["final_prediction_rows"] += len(query) * 33
                for p in (path, pp):
                    artifacts[p.relative_to(ROOT).as_posix()] = sha(p)
        print("WA AUDIT final", role, len(records), flush=True)
    assert counts["inner_models"] == 2457 and counts["inner_prediction_rows"] == 464100
    assert counts["candidates"] == 273 and counts["policies"] == 15
    expected_count = sum(3 * len(associations(e)) for e in selection["roles"].values())
    assert counts["final_models"] == result["final_models"] == expected_count
    assert counts["consumer_calls"] == counts["final_prediction_rows"]
    for name in ("source_lock.json", "selections.json", "results.json", "progress.json"):
        path = RUN / name
        artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
    deps = {
        f"scripts/{n}.py": sha(ROOT / f"scripts/{n}.py")
        for n in (
            "chromaseed_weight_average_audit",
            "chromaseed_long_training_audit",
            "chromaseed_gated_audit",
            "chromaseed_affine_audit",
            "chromaseed_gate_stability_audit",
            "chromaseed_refine_audit",
            "chromaseed_local_denoise_audit",
            "chromaseed_neural_prefix_numpy",
        )
    }
    write_json(
        OUT / "audit.json",
        dict(
            passed=True,
            source_lock_sha256=source,
            selection_sha256=sha(RUN / "selections.json"),
            results_sha256=sha(RUN / "results.json"),
            counts=counts,
            maximum_native_lab_drift=maximum,
            artifact_sha256=artifacts,
            dependencies=deps,
            seconds=time.perf_counter() - began,
        ),
    )
    print("WA AUDIT PASSED", counts, flush=True)


if __name__ == "__main__":
    main()
