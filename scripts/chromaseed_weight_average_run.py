"""WA inner-only selection, then final means and matched endpoint controls."""

from __future__ import annotations

import os
import time

import numpy as np
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_run import OUT as LT_OUT
from chromaseed_long_training_run import (
    ROOT,
    bank_path,
    check_map,
    context,
    load_data,
    save_npz,
    slot_index,
)
from chromaseed_long_training_run import RUN as LT
from chromaseed_neural_prefix_numpy import predict
from chromaseed_weight_average import METHODS, STEPS, average, choose, final_union, recipes
from skin_local_search_train import metrics, roles, sha, write_json

RUN = ROOT / "experiments/runs/chromaseed_weight_average_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_weight_average_v1"
SEEDS = (17, 29, 43)
PARENT = "18b9127d82ca32e854be7efbf3906dd4f1fa73a749bf9d41ab7a78eae7df795b"


def freeze():
    assert sha(LT_OUT / "verification.json") == PARENT
    old, seal = js(LT / "source_lock.json"), js(LT_OUT / "verification.json")
    sources = {**old["sources"], **seal["postprocess_sources"]}
    inputs = {**old["input_sha256"], **seal["artifact_sha256"]}
    inputs[(LT_OUT / "verification.json").relative_to(ROOT).as_posix()] = PARENT
    for name in (
        "scripts/chromaseed_weight_average.py",
        "scripts/chromaseed_weight_average_run.py",
        "tests/test_chromaseed_weight_average.py",
        "docs/research/chromaseed_weight_average_v1_protocol.md",
    ):
        sources[name] = sha(ROOT / name)
    check_map({**sources, **inputs})
    assert all(
        os.environ.get(k) == "1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    value = dict(
        sources=sources,
        input_sha256=inputs,
        parent_verification_sha256=PARENT,
        recipes=recipes(),
        inner_banks=9,
        inner_models=2457,
        inner_prediction_rows=464100,
        candidates=273,
        policies=15,
        previous_turn_classification="progress",
        numpy=np.__version__,
        primary_new_gradient_fits=0,
    )
    path = RUN / "source_lock.json"
    if path.exists():
        assert js(path) == value
    else:
        write_json(path, value)
    return sha(path)


def load_parents(role, fold=None):
    return {s: nz(bank_path(role, fold) / f"models_{s}.npz") for s in STEPS}


def construct(spec, si, parents, identity):
    slot = slot_index(si, spec["variants"], spec["lr"])
    components = [unpack(parents[s], str(slot)) for s in spec["steps"]]
    provenance = [
        dict(context=identity, seed=SEEDS[si], variants=spec["variants"], lr=spec["lr"], step=s)
        for s in spec["steps"]
    ]
    return average(components, provenance), components


def inner(data, source):
    for role in roles(data["patient"], data["device"]):
        for fold in range(3):
            fit, query, _, _ = context(data, role, fold)
            parents = load_parents(role, fold)
            models, predictions, names = {}, [], []
            for spec in recipes():
                for si, seed in enumerate(SEEDS):
                    name = f"{spec['name']}_s{seed}"
                    model, _ = construct(spec, si, parents, f"{role}/fold{fold}")
                    models[name] = model
                    names.append(name)
                    predictions.append(predict(model, data["color"][query]))
            path = RUN / "inner" / role / f"fold{fold}"
            save_npz(path / "models.npz", flatten(models))
            save_npz(
                path / "oof.npz",
                dict(row_indices=query, names=np.array(names), predictions=np.stack(predictions)),
            )
            save_npz(path / "rows.npz", dict(fit_rows=fit, query_rows=query))
            receipt = dict(
                source_lock_sha256=source,
                role=role,
                fold=fold,
                models=len(models),
                files={n: sha(path / n) for n in ("models.npz", "oof.npz", "rows.npz")},
            )
            write_json(path / "receipt.json", receipt)
            print("WA INNER", role, fold, len(models), flush=True)


def select(data, source):
    result = dict(source_lock_sha256=source, roles={})
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        parts = [nz(RUN / "inner" / role / f"fold{f}" / "oof.npz") for f in range(3)]
        rows = np.concatenate([p["row_indices"] for p in parts])
        order = np.argsort(rows)
        np.testing.assert_array_equal(rows[order], np.flatnonzero(mask))
        for p in parts:
            np.testing.assert_array_equal(p["names"], parts[0]["names"])
        values = np.concatenate([p["predictions"] for p in parts], axis=1)[:, order]
        index = {str(n): i for i, n in enumerate(parts[0]["names"])}
        candidates = []
        for spec in recipes():
            mm = [
                metrics(
                    values[index[f"{spec['name']}_s{s}"]],
                    data["target"][mask],
                    data["patient"][mask],
                    data["site"][mask],
                )
                for s in SEEDS
            ]
            candidates.append(
                dict(
                    **spec,
                    clean=float(np.mean([m["person_mean"] for m in mm])),
                    p90=float(np.mean([m["p90"] for m in mm])),
                    seed_metrics=mm,
                )
            )
        policies = {m: choose([c for c in candidates if c["method"] == m]) for m in METHODS}
        policies["overall"] = choose(candidates)
        old = js(LT / "selections.json")["roles"][role]["overall"]
        last = policies["last"]
        assert (last["variants"], last["steps"][-1], last["lr"] if last["steps"][-1] else None) == (
            old["variants"],
            old["step"],
            old["lr"],
        )
        assert last["clean"] == old["clean"] and last["p90"] == old["p90"]
        result["roles"][role] = dict(candidates=candidates, policies=policies)
    write_json(RUN / "selections.json", result)
    print("WA SELECTED", sha(RUN / "selections.json"), flush=True)
    return result


def evaluate(data, source, selection):
    records = []
    lookup = {s["name"]: s for s in recipes()}
    for role, entry in selection["roles"].items():
        _, query, _, _ = context(data, role)
        parents = load_parents(role)
        union = final_union(entry)
        for recipe_name, association in union.items():
            spec = lookup[recipe_name]
            for si, seed in enumerate(SEEDS):
                model, components = construct(spec, si, parents, f"{role}/final")
                name = f"{recipe_name}_s{seed}"
                path, pp = (
                    RUN / "selected" / role / f"{name}.npz",
                    RUN / "evaluated" / role / f"{name}.npz",
                )
                save_npz(path, model)
                output = np.stack(
                    [
                        predict(
                            model, affine_features(data["color"][query], s["dose"], s["anchor"])
                        )
                        for s in grid()
                    ]
                )
                save_npz(pp, dict(row_indices=query, predictions=output))
                transforms, doses = summaries(
                    output,
                    data["target"][query],
                    data["patient"][query],
                    data["device"][query],
                    None,
                )
                ensemble = np.mean([predict(c, data["color"][query]) for c in components], axis=0)
                records.append(
                    dict(
                        **{k: v for k, v in spec.items() if k != "name"},
                        recipe_name=recipe_name,
                        name=name,
                        role=role,
                        seed=seed,
                        **association,
                        parameters=643,
                        numeric_bytes=2886,
                        archive_bytes=path.stat().st_size,
                        model_sha256=sha(path),
                        prediction_sha256=sha(pp),
                        metrics=metrics(
                            output[0],
                            data["target"][query],
                            data["patient"][query],
                            data["site"][query],
                            data["device"][query],
                        ),
                        transforms=transforms,
                        doses=doses,
                        weight_vs_prediction_mean_max_lab=float(np.max(abs(output[0] - ensemble))),
                    )
                )
        print("WA FINAL", role, "recipes", len(union), flush=True)
    expected = sum(3 * len(final_union(e)) for e in selection["roles"].values())
    assert len(records) == expected
    value = dict(
        source_lock_sha256=source,
        selection_sha256=sha(RUN / "selections.json"),
        records=records,
        final_models=expected,
        primary_new_gradient_fits=0,
    )
    write_json(RUN / "results.json", value)
    return value


def main():
    assert not (OUT / "verification.json").exists(), "sealed WA: report read-only verifier only"
    started = time.perf_counter()
    data = load_data()
    source = freeze()
    inner(data, source)
    selection = select(data, source)
    result = evaluate(data, source, selection)
    write_json(
        RUN / "progress.json",
        dict(
            status="complete",
            pid=os.getpid(),
            seconds=time.perf_counter() - started,
            source_lock_sha256=source,
            results_sha256=sha(RUN / "results.json"),
        ),
    )
    print(
        "WA COMPLETE",
        dict(models=result["final_models"], results_sha256=sha(RUN / "results.json")),
        flush=True,
    )


if __name__ == "__main__":
    main()
