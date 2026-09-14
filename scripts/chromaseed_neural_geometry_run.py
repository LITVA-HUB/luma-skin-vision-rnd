"""Frozen inner-only NR geometry and independent reconstruction."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from chromaseed_fast_kernel_train import row_hash
from chromaseed_gated import unpack
from chromaseed_gaussian_audit import scoring
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from chromaseed_neural_geometry import degrees, source_design, source_metric, spectrum
from chromaseed_neural_geometry_reference import reconstruct
from chromaseed_neural_readout import BASES, FAMILIES, GROUPS, SEEDS, name_for
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, weights_for, write_json

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "experiments/runs/chromaseed_neural_readout_v1"
RECEIPT = ROOT / "docs/benchmarks/chromaseed_neural_readout_v1/verification.json"
RECEIPT_HASH = "3e4079df5d688136ef59be6b599633b0bc7c44d9d21f410eaeefc2aef0699c65"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def freeze(run, cache):
    assert cache.name == "train.npz" and sha(cache) == CACHE_HASH
    assert all(
        os.environ.get(k) == "1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    assert sha(RECEIPT) == RECEIPT_HASH
    parent, receipt = js(PARENT / "source_lock.json"), js(RECEIPT)
    assert receipt["passed"] and sha(PARENT / "source_lock.json") == receipt["source_lock_sha256"]
    sources = dict(parent["sources"])
    for name in (
        "chromaseed_neural_geometry.py",
        "chromaseed_neural_geometry_reference.py",
        "chromaseed_neural_geometry_run.py",
    ):
        path = "scripts/" + name
        sources[path] = sha(ROOT / path)
    for path in (
        "tests/test_chromaseed_neural_geometry.py",
        "docs/research/chromaseed_neural_geometry_v1_protocol.md",
    ):
        sources[path] = sha(ROOT / path)
    for path, digest in {
        **parent["sources"],
        **parent["input_sha256"],
        **receipt["postprocess_sources"],
    }.items():
        assert sha(ROOT / path) == digest, path
    paths = [
        RECEIPT,
        PARENT / "source_lock.json",
        PARENT / "selections.json",
        ROOT / "docs/research/chromaseed_neural_readout_next_decision.md",
        ROOT / "docs/research/chromaseed_weak_ridge_next_decision.md",
        ROOT / "scripts/chromaseed_gaussian_audit.py",
        ROOT / "scripts/chromaseed_refine_audit.py",
    ]
    for directory in sorted((PARENT / "inner").glob("*/fold*")):
        paths += [directory / f for f in ("bases.npz", "receipt.json", "oof.npz")]
        bank = js(directory / "receipt.json")
        for name, digest in bank["files"].items():
            assert sha(directory / name) == digest, directory / name
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    value = dict(
        sources=sources,
        input_sha256=inputs,
        cache_sha256=CACHE_HASH,
        parent_verification_sha256=RECEIPT_HASH,
        previous_turn_classification="progress",
        alphas=list(ALPHAS),
        expected_designs=756,
        expected_candidate_scores=252,
    )
    path = run / "source_lock.json"
    if path.exists():
        assert js(path) == value, "frozen geometry inputs changed"
    else:
        write_json(path, value)
    return sha(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    source = freeze(args.run, args.cache)
    assert not (args.run / "results.json").exists(), "completed geometry must not be overwritten"
    data = nz(args.cache)
    old = js(PARENT / "selections.json")
    assert sha(PARENT / "selections.json") == js(RECEIPT)["selection_sha256"]
    records, spectra, curves = [], {}, []
    maxima = dict(spectrum_scaled_error=0.0, degrees_error=0.0, score_error=0.0)
    write_json(
        args.run / "progress.json",
        dict(status="running", pid=os.getpid(), source_lock_sha256=source),
    )
    for role, (outer_fit, _) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(outer_fit)
        folds = folds_for(data["patient"][rows], data["device"][rows])
        merged = {}
        for fold in range(3):
            fit, query = rows[folds != fold], rows[folds == fold]
            path = PARENT / "inner" / role / f"fold{fold}"
            rec, arrays, oof = (
                js(path / "receipt.json"),
                nz(path / "bases.npz"),
                nz(path / "oof.npz"),
            )
            assert rec["fit_rows_sha256"] == row_hash(fit)
            assert rec["query_rows_sha256"] == row_hash(query)
            np.testing.assert_array_equal(oof["row_indices"], query)
            assert not set(data["patient"][fit]) & set(data["patient"][query])
            weights = weights_for(data["patient"][fit], data["site"][fit])
            x, y = data["color"][fit], data["target"][fit]
            for base_name, meta in rec["bases"].items():
                model = unpack(arrays, base_name)
                design, constant = source_design(model, x)
                for family in FAMILIES:
                    metric = None if family == "norm" else source_metric(model, y, weights)
                    value = spectrum(design, weights, metric)
                    reference = reconstruct(model, x, y, weights, family)
                    largest = max(1.0, float(value[0][0]))
                    np.testing.assert_allclose(
                        value[0], reference[0], rtol=2e-6, atol=2e-7 * largest
                    )
                    e = float(np.max(abs(value[0] - reference[0])) / largest)
                    maxima["spectrum_scaled_error"] = max(maxima["spectrum_scaled_error"], e)
                    capacity = [degrees(value, a) for a in ALPHAS]
                    check = [degrees(reference, a) for a in ALPHAS]
                    np.testing.assert_allclose(capacity, check, rtol=0, atol=3e-5)
                    maxima["degrees_error"] = max(
                        maxima["degrees_error"], float(np.max(abs(np.array(capacity) - check)))
                    )
                    key = f"{role}_f{fold}_{family}_{base_name}"
                    spectra[key] = value[0]
                    records.append(
                        dict(
                            role=role,
                            fold=fold,
                            family=family,
                            **meta,
                            key=key,
                            rows=len(fit),
                            constant_columns=constant,
                            largest_eigenvalue=float(value[0][0]),
                            total_df=capacity,
                            equivalent_per_output_df=[v / 3 for v in capacity],
                        )
                    )
            for group in GROUPS:
                for family in FAMILIES:
                    for basis in BASES:
                        for seed in SEEDS:
                            for ai in range(3):
                                name = name_for(family, basis, group, seed, ai)
                                if name not in merged:
                                    merged[name] = np.empty((len(rows), 3))
                                merged[name][folds == fold] = oof["pred__" + name]
            print(f"GEOMETRY {role}/fold{fold}:84 designs independently checked", flush=True)
        for group in GROUPS:
            for family in FAMILIES:
                for basis in BASES:
                    entry = old["roles"][role][group][family]["bases"][basis]
                    verified = []
                    for ai, candidate in enumerate(entry["candidates"]):
                        seed_scores = [
                            scoring(
                                merged[name_for(family, basis, group, seed, ai)],
                                data["target"][rows],
                                data["patient"][rows],
                            )
                            for seed in SEEDS
                        ]
                        clean = float(np.mean([s["clean"] for s in seed_scores]))
                        p90 = float(np.mean([s["p90"] for s in seed_scores]))
                        err = max(abs(clean - candidate["clean"]), abs(p90 - candidate["p90"]))
                        assert err <= 2e-8, (role, group, family, basis, err)
                        maxima["score_error"] = max(maxima["score_error"], err)
                        verified.append(clean)
                    curves.append(
                        dict(
                            role=role,
                            group=group,
                            family=family,
                            basis=basis,
                            old_alpha=entry["selected"]["alpha"],
                            clean=verified,
                            change_1_to_10=verified[2] - verified[1],
                        )
                    )
    assert len(records) == 756 and len(curves) == 84
    improvements = sum(c["change_1_to_10"] < 0 for c in curves)
    reduction = float(
        np.median(
            [r["equivalent_per_output_df"][2] - r["equivalent_per_output_df"][3] for r in records]
        )
    )
    decision = dict(
        improved_old_curves=improvements,
        total_old_curves=84,
        median_equivalent_df_reduction_10_to_100=reduction,
        bounded_extension_justified=improvements >= 56 and reduction >= 1,
    )
    atomic_npz(args.run / "spectra.npz", spectra)
    write_json(
        args.run / "results.json",
        dict(
            source_lock_sha256=source,
            records=records,
            curves=curves,
            maxima=maxima,
            decision=decision,
            alphas=list(ALPHAS),
            independent_designs_checked=756,
            degrees_values_checked=3780,
            candidate_scores_checked=252,
            spectra_sha256=sha(args.run / "spectra.npz"),
            new_model_fits=0,
            outer_predictions_evaluated=0,
        ),
    )
    write_json(
        args.run / "workflow.json",
        dict(
            pid=os.getpid(),
            exit_code=0,
            wall_seconds=time.perf_counter() - started,
            source_lock_sha256=source,
        ),
    )
    write_json(args.run / "progress.json", dict(status="calculation_complete", pid=os.getpid()))
    print(
        dict(decision=decision, maxima=maxima, wall_seconds=time.perf_counter() - started),
        flush=True,
    )


if __name__ == "__main__":
    main()
