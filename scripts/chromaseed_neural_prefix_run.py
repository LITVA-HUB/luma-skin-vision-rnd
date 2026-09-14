"""Immutable conditional NP inner selection followed by all-prefix evaluation."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import unpack
from chromaseed_gaussian_train import infer
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_train import CACHE, ROOT, SEEDS, load_data, oof, save_npz
from chromaseed_local_denoise_train import RUN as ND
from chromaseed_neural_prefix_numpy import (
    SPECS,
    TOLERANCE,
    capacity,
    choose,
    export_prefix,
    predict,
)
from skin_local_search_train import CACHE_HASH, metrics, roles, sha, write_json

RUN = ROOT / "experiments/runs/chromaseed_neural_prefix_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_neural_prefix_v1"
ND_OUT = ROOT / "docs/benchmarks/chromaseed_local_denoise_v1"
PARENT_HASH = "36442da6dabe927b4056ef5532fc7a6dd3956c14eb8ed35736a3f161ac3e2376"


def check_map(mapping):
    for path, digest in mapping.items():
        assert sha(ROOT / path) == digest, path


def freeze():
    assert sha(CACHE) == CACHE_HASH
    assert all(
        os.environ.get(k) == "1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    parent = js(ND_OUT / "verification.json")
    assert sha(ND_OUT / "verification.json") == PARENT_HASH and parent["passed"]
    inherited, audit = js(ND / "source_lock.json"), js(ND_OUT / "audit.json")
    sources = {**inherited["sources"], **audit["dependencies"]}
    inputs = {
        **inherited["input_sha256"],
        **audit["artifact_sha256"],
        **parent["artifact_sha256"],
        **parent["postprocess_sources"],
    }
    check_map({**sources, **inputs})
    for path in (
        ND_OUT / "verification.json",
        ROOT / "docs/research/chromaseed_local_denoise_next_decision.md",
        ROOT / "docs/research/chromaseed_refine_precision_protocol.md",
    ):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in (
        "scripts/chromaseed_neural_prefix_numpy.py",
        "scripts/chromaseed_neural_prefix_run.py",
        "tests/test_chromaseed_neural_prefix.py",
        "docs/research/chromaseed_neural_prefix_v1_protocol.md",
    ):
        sources[path] = sha(ROOT / path)
    value = dict(
        sources=sources,
        input_sha256=inputs,
        cache_sha256=CACHE_HASH,
        parent_verification_sha256=PARENT_HASH,
        compact_tolerance=TOLERANCE,
        candidates=45,
        policies=30,
        exports=135,
        references=18,
        neural_fits_in_primary=0,
        previous_turn_classification="progress",
    )
    path = RUN / "source_lock.json"
    if path.exists():
        assert js(path) == value, "NP source or parent changed"
    else:
        write_json(path, value)
    return sha(path)


def select(data, source):
    parent = js(ND / "selections.json")
    result = dict(
        source_lock_sha256=source, parent_selection_sha256=sha(ND / "selections.json"), roles={}
    )
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        result["roles"][role] = {}
        for family, (k, _, _) in SPECS.items():
            setting = parent["roles"][role][family]["selected"]
            step, li = setting["step"], setting["lr_index"]
            rows, saved = oof(role, family, step)
            np.testing.assert_array_equal(rows, np.flatnonzero(mask))
            model = unpack(
                nz(ND / "inner" / role / family / "fold0" / f"models_{step}.npz"), str(li)
            )
            candidates = []
            for j in range(1, k + 1):
                mm = [
                    metrics(
                        saved[2 * si + li, :, j - 1],
                        data["target"][rows],
                        data["patient"][rows],
                        data["site"][rows],
                    )
                    for si in range(3)
                ]
                candidates.append(
                    dict(
                        prefix=j,
                        clean=float(np.mean([m["person_mean"] for m in mm])),
                        p90=float(np.mean([m["p90"] for m in mm])),
                        seed_metrics=mm,
                        **capacity(export_prefix(model, j)),
                    )
                )
            result["roles"][role][family] = dict(
                step=step,
                lr=setting["lr"],
                lr_index=li,
                candidates=candidates,
                policies=choose(candidates),
            )
    path = RUN / "selections.json"
    if path.exists():
        assert js(path) == result
    else:
        write_json(path, result)
    return result


def evaluate(data, source, selection):
    records = []
    for old in js(ND / "results.json")["records"]:
        role, family, seed = old["role"], old["family"], old["seed"]
        _, held = roles(data["patient"], data["device"])[role]
        query = np.flatnonzero(held)
        parentpath = ND / "selected" / role / f"{old['name']}.npz"
        parent = nz(parentpath)
        prefixes = range(1, SPECS[family][0] + 1) if family in SPECS else [None]
        for j in prefixes:
            name = old["name"] if j is None else f"{family}_j{j}_s{seed}"
            model = parent if j is None else export_prefix(parent, j)
            policies = (
                []
                if j is None
                else [
                    key
                    for key, c in selection["roles"][role][family]["policies"].items()
                    if c["prefix"] == j
                ]
            )
            path, predpath = (
                RUN / "selected" / role / f"{name}.npz",
                RUN / "evaluated" / role / f"{name}.npz",
            )
            save_npz(path, model)
            output = []
            for setting in grid():
                xx = affine_features(data["color"][query], setting["dose"], setting["anchor"])
                output.append(infer(model, xx) if j is None else predict(model, xx))
            output = np.stack(output)
            save_npz(predpath, dict(row_indices=query, predictions=output))
            transforms, doses = summaries(
                output, data["target"][query], data["patient"][query], data["device"][query], None
            )
            cap = (
                dict(
                    parameters=old["parameters"],
                    numeric_bytes=old["numeric_bytes"],
                    executed_blocks=None,
                )
                if j is None
                else capacity(model)
            )
            records.append(
                dict(
                    role=role,
                    name=name,
                    family=family,
                    seed=seed,
                    prefix=j,
                    original_k=None if j is None else SPECS[family][0],
                    full_prefix=j is not None and j == SPECS[family][0],
                    policies=policies,
                    step=old["step"],
                    lr=old["lr"],
                    origin="exact_ND_reference" if j is None else "ND_prefix_export",
                    parent_name=old["name"],
                    parent_model_sha256=sha(parentpath),
                    **cap,
                    archive_bytes=path.stat().st_size,
                    model_sha256=sha(path),
                    prediction_sha256=sha(predpath),
                    metrics=metrics(
                        output[0],
                        data["target"][query],
                        data["patient"][query],
                        data["site"][query],
                        data["device"][query],
                    ),
                    transforms=transforms,
                    doses=doses,
                )
            )
        print("EXPORTED", role, old["name"], flush=True)
    assert len(records) == 153
    value = dict(
        source_lock_sha256=source,
        selection_sha256=sha(RUN / "selections.json"),
        records=records,
        evidence="Conditional fixed prefixes; repeated TRAIN/person-camera confounding; no phone-face validation",
    )
    path = RUN / "results.json"
    if path.exists():
        assert js(path) == value
    else:
        write_json(path, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "all"), default="all")
    args = parser.parse_args()
    data = load_data()
    assert data["color"].shape == (966, 36)
    if args.stage == "preflight":
        assert not (RUN / "source_lock.json").exists()
        assert sha(ND_OUT / "verification.json") == PARENT_HASH
        print(
            dict(
                stage="preflight", candidates=45, choices=30, exports=135, references=18, new_fits=0
            )
        )
        return
    start = time.perf_counter()
    source = freeze()
    selection = select(data, source)
    print("SELECTION FROZEN", sha(RUN / "selections.json"), flush=True)
    evaluate(data, source, selection)
    write_json(
        RUN / "progress.json",
        dict(
            status="complete",
            pid=os.getpid(),
            seconds=time.perf_counter() - start,
            source_lock_sha256=source,
            results_sha256=sha(RUN / "results.json"),
        ),
    )
    print("NP COMPLETE", sha(RUN / "results.json"), flush=True)


if __name__ == "__main__":
    main()
