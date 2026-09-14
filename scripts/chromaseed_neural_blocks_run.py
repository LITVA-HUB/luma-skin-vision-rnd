"""Matched full/subset training banks to8192; immutable NP policies unchanged."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from chromaseed_fast_kernel_train import row_hash
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_fit import fit as full_fit
from chromaseed_local_denoise_train import CACHE, ROOT, load_data, save_npz
from chromaseed_neural_blocks_fit import fit as subset_fit
from chromaseed_neural_blocks_fit import to_prefix
from chromaseed_neural_prefix_numpy import SPECS, capacity, export_prefix, predict
from chromaseed_neural_prefix_run import RUN as NP
from chromaseed_neural_prefix_run import check_map
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_train import setup
from skin_local_search_train import CACHE_HASH, metrics, roles, sha, weights_for, write_json

RUN = ROOT / "experiments/runs/chromaseed_neural_blocks_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_neural_blocks_v1"
NP_OUT = ROOT / "docs/benchmarks/chromaseed_neural_prefix_v1"
SEEDS, CHECKPOINTS = (17, 29, 43), (512, 2048, 8192)
FAMILIES = ("local2", "local4", "blind4")
PARENT_HASH = "f9bd98ddad578678278581ce2fde786ced806e13e1db3a15cea1aa0fd33fe024"


def settings():
    value = []
    for role, families in js(NP / "selections.json")["roles"].items():
        for family in FAMILIES:
            entry = families[family]
            prefixes = sorted(
                {SPECS[family][0], *(c["prefix"] for c in entry["policies"].values())}
            )
            assert len(prefixes) == 2
            value.append(
                dict(
                    role=role,
                    family=family,
                    step=entry["step"],
                    lr=entry["lr"],
                    prefixes=prefixes,
                    policies={p: c["prefix"] for p, c in entry["policies"].items()},
                )
            )
    assert len(value) == 9
    return value


def freeze():
    assert sha(CACHE) == CACHE_HASH and sha(NP_OUT / "verification.json") == PARENT_HASH
    assert all(
        os.environ.get(k) == "1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    inherited, parent = js(NP / "source_lock.json"), js(NP_OUT / "verification.json")
    sources = {**inherited["sources"], **parent["postprocess_sources"]}
    inputs = {**inherited["input_sha256"], **parent["artifact_sha256"]}
    check_map({**sources, **inputs})
    inputs[(NP_OUT / "verification.json").relative_to(ROOT).as_posix()] = PARENT_HASH
    for name in (
        "scripts/chromaseed_neural_blocks_fit.py",
        "scripts/chromaseed_neural_blocks_run.py",
        "tests/test_chromaseed_neural_blocks.py",
        "docs/research/chromaseed_neural_blocks_v1_protocol.md",
    ):
        sources[name] = sha(ROOT / name)
    value = dict(
        sources=sources,
        input_sha256=inputs,
        cache_sha256=CACHE_HASH,
        settings=settings(),
        seeds=list(SEEDS),
        checkpoints=list(CHECKPOINTS),
        parent_verification_sha256=PARENT_HASH,
        parent_selection_sha256=sha(NP / "selections.json"),
        banks=27,
        trajectories=81,
        checkpoint_payloads=243,
        device="cuda",
        engine="cuda_graph",
        torch=torch.__version__,
        numpy=np.__version__,
        gpu=torch.cuda.get_device_name(),
        previous_turn_classification="progress",
    )
    path = RUN / "source_lock.json"
    if path.exists():
        assert js(path) == value, "NB source or inputs changed after freeze"
    else:
        write_json(path, value)
    return sha(path)


def bank_path(setting, prefix):
    return (
        RUN
        / "banks"
        / setting["role"]
        / setting["family"]
        / ("full" if prefix is None else f"j{prefix}")
    )


def train_bank(data, setting, prefix, source):
    path = bank_path(setting, prefix)
    mask, _ = roles(data["patient"], data["device"])[setting["role"]]
    ix = np.flatnonzero(mask)
    if (path / "receipt.json").exists():
        old = js(path / "receipt.json")
        assert old["source_lock_sha256"] == source and old["fit_rows_sha256"] == row_hash(ix)
        for file, digest in old["files"].items():
            assert sha(path / file) == digest
        print("REUSE", path.relative_to(RUN).as_posix(), flush=True)
        return
    x, y = data["color"][ix], data["target"][ix]
    w = weights_for(data["patient"][ix], data["site"][ix])
    np.testing.assert_allclose(
        w, balanced(data["patient"][ix], data["site"][ix]), rtol=1e-12, atol=1e-12
    )

    def progress(entry):
        if entry["step"] in CHECKPOINTS:
            print(
                "NB BANK",
                path.relative_to(RUN).as_posix(),
                entry["step"],
                round(entry["seconds"], 3),
                flush=True,
            )

    slots = [(seed, setting["lr"]) for seed in SEEDS]
    if prefix is None:
        models, info = full_fit(
            x, y, w, setting["family"], slots, 8192, CHECKPOINTS, "cuda", "cuda_graph", progress
        )
    else:
        models, info = subset_fit(
            x,
            y,
            w,
            setting["family"],
            slots,
            prefix,
            8192,
            CHECKPOINTS,
            "cuda",
            "cuda_graph",
            progress,
        )
    files = {}
    for step, payloads in models.items():
        rawpath = path / f"raw_{step}.npz"
        save_npz(rawpath, flatten({str(i): p for i, p in enumerate(payloads)}))
        files[rawpath.name] = sha(rawpath)
        if prefix is not None:
            exportpath = path / f"models_{step}.npz"
            save_npz(exportpath, flatten({str(i): to_prefix(p) for i, p in enumerate(payloads)}))
            files[exportpath.name] = sha(exportpath)
    save_npz(path / "rows.npz", dict(fit_rows=ix))
    files["rows.npz"] = sha(path / "rows.npz")
    info.update(
        role=setting["role"],
        prefix=prefix,
        mode="full" if prefix is None else "subset",
        source_lock_sha256=source,
        parent_selection_sha256=sha(NP / "selections.json"),
        fit_rows_sha256=row_hash(ix),
        files=files,
    )
    write_json(path / "receipt.json", info)


def compare_arrays(actual, expected):
    assert set(actual) == set(expected)
    exact, maximum = True, 0.0
    for key in actual:
        exact &= np.array_equal(actual[key], expected[key])
        if actual[key].dtype.kind in "biufc":
            maximum = max(
                maximum, float(np.max(abs(actual[key].astype(float) - expected[key].astype(float))))
            )
            np.testing.assert_allclose(actual[key], expected[key], rtol=2e-6, atol=2e-6)
        else:
            np.testing.assert_array_equal(actual[key], expected[key])
    return bool(exact), maximum


def outputs(model, x):
    return np.stack([predict(model, affine_features(x, s["dose"], s["anchor"])) for s in grid()])


def evaluate(data, source):
    records = []
    for setting in settings():
        role, family = setting["role"], setting["family"]
        _, held = roles(data["patient"], data["device"])[role]
        query = np.flatnonzero(held)
        for step in CHECKPOINTS:
            full_raw = nz(bank_path(setting, None) / f"raw_{step}.npz")
            for prefix in (None, *setting["prefixes"]):
                raw = nz(bank_path(setting, prefix) / f"raw_{step}.npz")
                mode = "full" if prefix is None else "subset"
                j = SPECS[family][0] if prefix is None else prefix
                for si, seed in enumerate(SEEDS):
                    actual_raw, parent_raw = unpack(raw, str(si)), unpack(full_raw, str(si))
                    model = (
                        export_prefix(actual_raw, j) if prefix is None else to_prefix(actual_raw)
                    )
                    matched = export_prefix(parent_raw, j)
                    payload_exact, weight_drift = compare_arrays(model, matched)
                    raw_exact, raw_drift = True, 0.0
                    if prefix is not None:
                        ids = actual_raw["block_indices"]
                        target = parent_raw["theta"][ids]
                        raw_exact = np.array_equal(actual_raw["theta"], target)
                        raw_drift = float(np.max(abs(actual_raw["theta"] - target)))
                        np.testing.assert_allclose(
                            actual_raw["theta"], target, rtol=2e-6, atol=2e-6
                        )
                    name = f"{family}_{mode}{j}_t{step}_s{seed}"
                    path, pp = (
                        RUN / "selected" / role / f"{name}.npz",
                        RUN / "evaluated" / role / f"{name}.npz",
                    )
                    save_npz(path, model)
                    pred, reference = (
                        outputs(model, data["color"][query]),
                        outputs(matched, data["color"][query]),
                    )
                    maximum = float(np.max(abs(pred - reference)))
                    assert maximum <= 0.002
                    mm = metrics(
                        pred[0],
                        data["target"][query],
                        data["patient"][query],
                        data["site"][query],
                        data["device"][query],
                    )
                    rm = metrics(
                        reference[0],
                        data["target"][query],
                        data["patient"][query],
                        data["site"][query],
                        data["device"][query],
                    )
                    error_drift = mm["person_mean"] - rm["person_mean"]
                    assert abs(error_drift) <= 0.002
                    save_npz(pp, dict(row_indices=query, predictions=pred))
                    transforms, doses = summaries(
                        pred,
                        data["target"][query],
                        data["patient"][query],
                        data["device"][query],
                        None,
                    )
                    np_check = None
                    if step == setting["step"]:
                        parentpath = NP / "selected" / role / f"{family}_j{j}_s{seed}.npz"
                        previous = nz(parentpath)
                        np_exact, np_max = compare_arrays(model, previous)
                        prev_prediction = outputs(previous, data["color"][query])
                        drift = float(np.max(abs(pred - prev_prediction)))
                        prev_metrics = metrics(
                            prev_prediction[0],
                            data["target"][query],
                            data["patient"][query],
                            data["site"][query],
                            data["device"][query],
                        )
                        difference = mm["person_mean"] - prev_metrics["person_mean"]
                        assert drift <= 0.002 and abs(difference) <= 0.002
                        np_check = dict(
                            parent_name=parentpath.stem,
                            parent_model_sha256=sha(parentpath),
                            bitwise_equal=np_exact,
                            max_weight_drift=np_max,
                            max_prediction_drift=drift,
                            error_difference=difference,
                        )
                    records.append(
                        dict(
                            role=role,
                            family=family,
                            name=name,
                            mode=mode,
                            prefix=j,
                            seed=seed,
                            step=step,
                            lr=setting["lr"],
                            selected_step=step == setting["step"],
                            policies=[]
                            if prefix is None or step != setting["step"]
                            else [p for p, value in setting["policies"].items() if value == j],
                            **capacity(model),
                            archive_bytes=path.stat().st_size,
                            model_sha256=sha(path),
                            prediction_sha256=sha(pp),
                            metrics=mm,
                            transforms=transforms,
                            doses=doses,
                            full_control_comparison=dict(
                                raw_bitwise_equal=bool(raw_exact),
                                raw_max_weight_drift=raw_drift,
                                payload_bitwise_equal=payload_exact,
                                max_weight_drift=weight_drift,
                                max_prediction_drift=maximum,
                                error_difference=error_drift,
                            ),
                            np_comparison=np_check,
                        )
                    )
            print("NB EVALUATED", role, family, step, flush=True)
    assert len(records) == 243 and sum(r["mode"] == "subset" for r in records) == 162
    result = dict(
        source_lock_sha256=source,
        parent_selection_sha256=sha(NP / "selections.json"),
        records=records,
        evidence="Matched implementation/cost study on reused TRAIN; longer checkpoints do not reopen NP selection",
    )
    path = RUN / "results.json"
    if path.exists():
        assert js(path) == result
    else:
        write_json(path, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "all"), default="all")
    args = parser.parse_args()
    assert not (OUT / "verification.json").exists(), "sealed NB: report verifier only"
    setup("cuda")
    data = load_data()
    if args.stage == "preflight":
        assert not (RUN / "source_lock.json").exists()
        print(
            dict(
                settings=settings(),
                banks=27,
                trajectories=81,
                checkpoints=243,
                gpu=torch.cuda.get_device_name(),
            )
        )
        return
    started = time.perf_counter()
    source = freeze()
    for setting in settings():
        for prefix in (None, *setting["prefixes"]):
            train_bank(data, setting, prefix, source)
    evaluate(data, source)
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
    print("NB COMPLETE", sha(RUN / "results.json"), flush=True)


if __name__ == "__main__":
    main()
