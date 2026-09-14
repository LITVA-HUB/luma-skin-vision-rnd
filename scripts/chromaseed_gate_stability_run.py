"""Run frozen G sensitivity and boundary diagnostics on original TRAIN only."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from chromaseed_gate_stability import (
    DOSES,
    affine_features,
    boundaries,
    gate_score,
    grid,
    summaries,
)
from chromaseed_gated import predict
from chromaseed_kernel_audit import js, nz
from skin_local_search_train import CACHE_HASH, sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def bind(run, parent, cache):
    assert sha(cache) == CACHE_HASH
    gp = ROOT / "docs/benchmarks/chromaseed_gated_v1/verification.json"
    assert sha(gp) == "9b73c11efed3d4fae6e6e41f4987377a68fc841af37a88984194f879a1d00065"
    previous = js(gp)
    assert previous["passed"]
    for name, field in (
        ("source_lock.json", "source_lock_sha256"),
        ("selections.json", "selection_sha256"),
        ("results.json", "results_sha256"),
    ):
        assert sha(parent / name) == previous[field]
    for path, value in previous["postprocess_sources"].items():
        assert sha(ROOT / path) == value, path
    sources = dict(js(parent / "source_lock.json")["sources"])
    for path in (
        "scripts/chromaseed_gate_stability.py",
        "scripts/chromaseed_gate_stability_run.py",
        "scripts/chromaseed_gated_numpy.py",
        "tests/test_chromaseed_gate_stability.py",
        "docs/research/chromaseed_gate_stability_v1_protocol.md",
    ):
        sources[path] = sha(ROOT / path)
    for p, value in sources.items():
        assert sha(ROOT / p) == value, p
    inputs = {str(gp.relative_to(ROOT)): sha(gp)}
    for p in (parent / "source_lock.json", parent / "selections.json", parent / "results.json"):
        inputs[str(p.relative_to(ROOT))] = sha(p)
    for p, value in previous["selected_numeric_file_hashes_checked"].items():
        path = parent / p
        assert sha(path) == value
        inputs[str(path.relative_to(ROOT))] = value
    lock = dict(
        cache_sha256=CACHE_HASH,
        sources=sources,
        input_sha256=inputs,
        doses=DOSES,
        transforms=33,
        epsilon=1e-4,
        numpy=np.__version__,
        threads=1,
        loaded_keys=["color", "target", "patient", "site", "device"],
        purpose="fixed-model synthetic color and gate stability; no fitting/selection",
    )
    run.mkdir(parents=True, exist_ok=True)
    path = run / "source_lock.json"
    if path.exists():
        assert js(path) == lock
    else:
        write_json(path, lock)
    return sha(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--parent", type=Path, default=ROOT / "experiments/runs/chromaseed_gated_v1"
    )
    args = parser.parse_args()
    run, parent = args.run.resolve(), args.parent.resolve()
    started = time.perf_counter()
    lock = bind(run, parent, args.cache)
    assert not (run / "results.json").exists(), "completed diagnostic must not be overwritten"
    write_json(
        run / "progress.json", dict(status="running", pid=os.getpid(), updated_unix=time.time())
    )
    with np.load(args.cache, allow_pickle=False) as archive:
        data = {k: archive[k] for k in ("color", "target", "patient", "site", "device")}
    source_result = js(parent / "results.json")
    records, boundary_records, numeric = [], [], {}
    for role in ("mixed", "slr_to_ipod", "ipod_to_slr"):
        for record in (r for r in source_result["records"] if r["role"] == role):
            family, seed = record["family"], record["seed"]
            name = f"{family}_s{seed}.npz"
            model, original = (
                nz(parent / "selected" / role / name),
                nz(parent / "evaluated" / role / name),
            )
            rows = original["row_indices"]
            x, y, person, camera = (data[k][rows] for k in ("color", "target", "patient", "device"))
            xx = [affine_features(x, s["dose"], s["anchor"]) for s in grid()]
            pred = np.stack([predict(model, v) for v in xx])
            np.testing.assert_allclose(pred[0], original["prediction"], atol=2e-8, rtol=0)
            scores = np.stack([gate_score(model, v) for v in xx]) if "gate_beta" in model else None
            tt, dd = summaries(pred, y, person, camera, scores)
            path = run / "perturbed" / role / name
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                prediction=pred,
                row_indices=rows,
                **({"scores": scores} if scores is not None else {}),
            )
            numeric[str(path.relative_to(run))] = sha(path)
            records.append(
                dict(
                    role=role,
                    family=family,
                    seed=seed,
                    active_gate=scores is not None,
                    transforms=tt,
                    doses=dd,
                    numeric_file=str(path.relative_to(run)),
                )
            )
        print(f"PERTURBED {role}:24 fixed models x33 transforms", flush=True)
    for base in ("norm", "perceptual"):
        for seed in (17, 29, 43):
            name = f"{base}_hard_s{seed}.npz"
            original = nz(parent / "evaluated/mixed" / name)
            rows = original["row_indices"]
            models = {
                kind: nz(parent / "selected/mixed" / f"{base}_{kind}_s{seed}.npz")
                for kind in ("base", "uniform", "soft", "hard")
            }
            arrays, summary = boundaries(
                models, data["color"][rows], data["patient"][rows], data["device"][rows]
            )
            path = run / "boundary" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays, row_indices=rows)
            numeric[str(path.relative_to(run))] = sha(path)
            boundary_records.append(dict(base=base, seed=seed, **summary))
            print(
                f"BOUNDARY {base} seed{seed}: {summary['legal_boundary_pairs']} pairs", flush=True
            )
    write_json(
        run / "results.json",
        dict(
            source_lock_sha256=lock,
            records=records,
            boundaries=boundary_records,
            numeric_sha256=numeric,
            elapsed_seconds=time.perf_counter() - started,
            evidence="synthetic perturbations of historically reused TRAIN; no model fitting or real camera validation",
        ),
    )
    write_json(
        run / "progress.json",
        dict(status="diagnostic_complete_audit_pending", pid=os.getpid(), updated_unix=time.time()),
    )


if __name__ == "__main__":
    main()
