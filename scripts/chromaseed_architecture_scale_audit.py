"""Independent NumPy export, lineage and metric audit for the AS screen."""

from __future__ import annotations

import hashlib
import time

import numpy as np
from chromaseed_architecture_scale import Bank, Predictor, base_model, predict
from chromaseed_architecture_scale_run import OUT, ROOT, RUN, WE, bank_path, check_map, load_data
from chromaseed_gated_audit import model_from
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_audit import close, verify_normalizers
from chromaseed_neural_prefix_numpy import predict as warm_predict
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from chromaseed_refine_train import setup
from chromaseed_widen_audit import exact
from chromaseed_widen_run import bank_path as old_bank
from skin_local_search_train import folds_for, roles, sha, write_json

PARAMETERS = dict(
    patch_small=17374,
    patch5m=4962566,
    soft_small=15246,
    soft5m=4846822,
    dynamic_small=15246,
    dynamic5m=4846822,
    pool5m=4851846,
)
SEEDS = (17, 29, 43)
RATES = (0.00001, 0.0001)
TIMES = (128, 512, 2048)


def rank(candidates):
    return min(
        candidates,
        key=lambda c: (c["clean"], c["p90"], c["numeric_bytes"], c["step"], c["lr"] or 0),
    )


def compare(actual, expected):
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, atol=0.002, rtol=1e-6)
    return float(np.max(np.abs(actual - expected)))


def inspect(data, source, role, variant, fold, artifacts, steps=2048):
    mask, held = roles(data["patient"], data["device"])[role]
    ix, query = np.flatnonzero(mask), np.flatnonzero(held)
    if fold is not None:
        assignments = folds_for(data["patient"][ix], data["device"][ix])
        query, ix = ix[assignments == fold], ix[assignments != fold]
    assert not set(data["patient"][ix]) & set(data["patient"][query])
    path = bank_path(role, variant, fold)
    receipt = js(path / "receipt.json")
    assert receipt["source_lock_sha256"] == source
    assert receipt["selection_sha256"] == (
        None if fold is not None else sha(RUN / "selections.json")
    )
    assert receipt["variant"] == variant and receipt["role"] == role and receipt["fold"] == fold
    assert receipt["steps"] == steps and receipt["horizon"] == 8192 and receipt["batch_size"] == 64
    assert receipt["slots"] == [[s, r] for s in SEEDS for r in RATES]
    assert receipt["checkpoints"] == list(TIMES if fold is not None else (steps,))
    assert receipt["deployed_parameters"] == PARAMETERS[variant]
    assert receipt["trainable_parameters"] == PARAMETERS[variant] - 643
    assert receipt["engine"] == "cuda_graph" and receipt["device"] == "cuda"
    check_map(receipt["warm_parent_sha256"])
    for name, digest in receipt["files"].items():
        assert sha(path / name) == digest
        artifacts[(path / name).relative_to(ROOT).as_posix()] = digest
    artifacts[(path / "receipt.json").relative_to(ROOT).as_posix()] = sha(path / "receipt.json")
    rows = nz(path / "rows.npz")
    np.testing.assert_array_equal(rows["fit_rows"], ix)
    np.testing.assert_array_equal(
        rows["query_rows"], query if fold is not None else np.array([], np.int64)
    )
    old = nz(old_bank(role, "m31", fold) / "warm_models.npz")
    warm = nz(path / "warm_models.npz")
    exact(old, warm)
    parents = [model_from(warm, str(i)) for i in range(3)]
    for m in parents:
        verify_normalizers(m, data["color"][ix], data["target"][ix])
    t = data["tokens"][ix]
    assert hashlib.sha256(t.tobytes()).hexdigest() == receipt["fit_tokens_sha256"]
    tm = t.astype(float).mean((0, 1)).astype(np.float32)
    ts = np.maximum(t.astype(float).std((0, 1)), 1e-6).astype(np.float32)
    weights = balanced(data["patient"][ix], data["site"][ix])
    weights /= weights.sum()
    draws = np.stack(
        [np.random.default_rng(s + 830003).choice(len(ix), (steps, 64), p=weights) for s in SEEDS]
    )
    assert hashlib.sha256(draws.tobytes()).hexdigest() == receipt["sampling_sha256"]
    factor = np.float32(0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (steps - 1) / 8192)))
    np.testing.assert_array_equal(
        np.asarray(receipt["final_learning_rates"], np.float32),
        np.tile(np.array(RATES, np.float32), 3) * factor,
    )
    return ix, query, parents, tm, ts


def check_model(model, variant, parent, tm, ts):
    assert str(model["variant"]) == variant
    assert model["theta"].dtype == np.float32 and model["theta"].shape == (
        PARAMETERS[variant] - 643,
    )
    assert np.isfinite(model["theta"]).all()
    exact(base_model(model), parent)
    np.testing.assert_array_equal(model["t_mean"], tm)
    np.testing.assert_array_equal(model["t_std"], ts)
    assert (
        sum(v.nbytes for v in model.values() if v.dtype.kind in "biufc")
        == 4 * PARAMETERS[variant] + 458
    )


def main():
    assert not (OUT / "verification.json").exists(), "sealed AS; use report verifier"
    assert js(RUN / "job.json")["status"] == "complete", "primary must be terminal"
    setup("cpu")
    started = time.perf_counter()
    lock, selection, results = [
        js(RUN / n) for n in ("source_lock.json", "selections.json", "results.json")
    ]
    source = sha(RUN / "source_lock.json")
    assert selection["source_lock_sha256"] == results["source_lock_sha256"] == source
    assert results["selection_sha256"] == sha(RUN / "selections.json")
    check_map({**lock["sources"], **lock["input_sha256"]})
    data = load_data()
    oldsel = js(WE / "selections.json")
    oldresults = js(WE / "results.json")
    artifacts = {
        (RUN / name).relative_to(ROOT).as_posix(): sha(RUN / name)
        for name in ("source_lock.json", "selections.json", "results.json", "job.json")
    }
    counts = dict(
        inner_banks=0,
        inner_models=0,
        inner_vectors=0,
        candidates=0,
        choices=0,
        final_banks=0,
        final_models=0,
        final_vectors=0,
        actual_single_calls=0,
        imported_controls=0,
    )
    maximum = 0.0
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        fullrows = np.flatnonzero(mask)
        candidates = []
        for variant in PARAMETERS:
            parts = {step: [] for step in TIMES}
            rows = []
            for fold in range(3):
                ix, query, warm, tm, ts = inspect(data, source, role, variant, fold, artifacts)
                rows.append(query)
                path = bank_path(role, variant, fold)
                zero = Bank(variant)
                for si in range(3):
                    initial = zero.export(2 * si, warm, data["tokens"][ix])
                    np.testing.assert_allclose(
                        predict(initial, data["color"][query[:3]], data["tokens"][query[:3]]),
                        warm_predict(warm[si], data["color"][query[:3]]),
                        atol=2e-8,
                        rtol=0,
                    )
                del zero
                for step in TIMES:
                    packed = nz(path / f"models_{step}.npz")
                    saved = nz(path / f"oof_{step}.npz")
                    np.testing.assert_array_equal(saved["row_indices"], query)
                    assert saved["predictions"].shape == (6, len(query), 3)
                    for slot in range(6):
                        model = model_from(packed, str(slot))
                        check_model(model, variant, warm[slot // 2], tm, ts)
                        out = predict(model, data["color"][query], data["tokens"][query])
                        maximum = max(maximum, compare(out, saved["predictions"][slot]))
                        counts["inner_models"] += 1
                        counts["inner_vectors"] += len(query)
                    parts[step].append(saved["predictions"])
                counts["inner_banks"] += 1
                print("AS AUDIT INNER", role, variant, fold, "maxLab", maximum, flush=True)
            rows = np.concatenate(rows)
            order = np.argsort(rows)
            np.testing.assert_array_equal(rows[order], fullrows)
            for ri, rate in enumerate(RATES):
                for step in TIMES:
                    pred = np.concatenate(parts[step], axis=1)[:, order][
                        [2 * i + ri for i in range(3)]
                    ]
                    mm = [
                        error_summary(
                            p,
                            data["target"][fullrows],
                            data["patient"][fullrows],
                            data["site"][fullrows],
                        )[0]
                        for p in pred
                    ]
                    candidates.append(
                        dict(
                            variant=variant,
                            step=step,
                            lr=rate,
                            parameters=PARAMETERS[variant],
                            numeric_bytes=4 * PARAMETERS[variant] + 458,
                            clean=float(np.mean([m["person_mean"] for m in mm])),
                            p90=float(np.mean([m["p90"] for m in mm])),
                            seed_metrics=mm,
                        )
                    )
        for variant, parent in (
            ("np", next(c for c in oldsel["roles"][role]["candidates"] if c["variant"] == "np")),
            ("we", oldsel["roles"][role]["policies"]["overall"]),
        ):
            candidates.append(dict(parent, variant=variant, parent_variant=parent["variant"]))
        close(selection["roles"][role]["candidates"], candidates)
        policies = {v: rank([c for c in candidates if c["variant"] == v]) for v in PARAMETERS}
        policies["overall"] = rank(candidates)
        close(selection["roles"][role]["policies"], policies)
        counts["candidates"] += len(candidates)
        counts["choices"] += len(policies)
        for variant in PARAMETERS:
            choice = policies[variant]
            _, query, warm, tm, ts = inspect(
                data, source, role, variant, None, artifacts, choice["step"]
            )
            path = bank_path(role, variant) / f"models_{choice['step']}.npz"
            packed = nz(path)
            for slot in range(6):
                check_model(model_from(packed, str(slot)), variant, warm[slot // 2], tm, ts)
            for si, seed in enumerate(SEEDS):
                rec = next(
                    r
                    for r in results["records"]
                    if r["role"] == role and r["variant"] == variant and r["seed"] == seed
                )
                slot = 2 * si + RATES.index(choice["lr"])
                assert (
                    rec["slot"] == slot and rec["source_path"] == path.relative_to(ROOT).as_posix()
                )
                assert rec["source_sha256"] == sha(path)
                assert rec["step"] == choice["step"] and rec["lr"] == choice["lr"]
                assert rec["overall"] == (policies["overall"]["variant"] == variant)
                for key in ("model", "output"):
                    p = ROOT / rec[key]
                    assert sha(p) == rec[key + "_sha256"]
                    artifacts[rec[key]] = sha(p)
                model = nz(ROOT / rec["model"])
                exact(model, model_from(packed, str(slot)))
                saved = nz(ROOT / rec["output"])
                np.testing.assert_array_equal(saved["row_indices"], query)
                np.testing.assert_array_equal(saved["predictions"], saved["passes"][:, -1])
                consumer = Predictor(model)
                # Actual one-example deployment executes all needed layers/passes.
                for i, row in enumerate(query):
                    out = consumer(data["color"][row], data["tokens"][row], all_passes=True)
                    maximum = max(maximum, compare(out, saved["passes"][i]))
                    counts["actual_single_calls"] += 1
                m, _ = error_summary(
                    saved["predictions"],
                    data["target"][query],
                    data["patient"][query],
                    data["site"][query],
                )
                close(rec["metrics"], m)
                assert (
                    rec["parameters"] == PARAMETERS[variant]
                    and rec["numeric_bytes"] == 4 * PARAMETERS[variant] + 458
                )
                counts["final_models"] += 1
                counts["final_vectors"] += len(query)
            counts["final_banks"] += 1
            print("AS AUDIT FINAL", role, variant, flush=True)
        for variant in ("np", "we"):
            for rec in [
                r for r in results["records"] if r["role"] == role and r["variant"] == variant
            ]:
                old = rec["imported_we_record"]
                assert (
                    old in oldresults["records"]
                    and old["role"] == role
                    and old["seed"] == rec["seed"]
                )
                assert old["kind"] == "np" if variant == "np" else "overall" in old["policies"]
                close(rec["metrics"], old["metrics"])
                assert rec["overall"] == (policies["overall"]["variant"] == variant)
                counts["imported_controls"] += 1
    assert counts == dict(
        inner_banks=63,
        inner_models=1134,
        inner_vectors=214200,
        candidates=132,
        choices=24,
        final_banks=21,
        final_models=63,
        final_vectors=25158,
        actual_single_calls=25158,
        imported_controls=18,
    )
    check_map({**lock["sources"], **lock["input_sha256"], **artifacts})
    value = dict(
        passed=True,
        results_sha256=sha(RUN / "results.json"),
        counts=counts,
        maximum_native_lab=maximum,
        artifact_sha256=artifacts,
        dependencies={
            "scripts/chromaseed_architecture_scale_audit.py": sha(
                ROOT / "scripts/chromaseed_architecture_scale_audit.py"
            )
        },
        seconds=time.perf_counter() - started,
    )
    write_json(OUT / "audit.json", value)
    print("AS AUDIT PASSED", counts, "maxLab", maximum, flush=True)


if __name__ == "__main__":
    main()
