"""Synthetic P3 training gates using actual P2 encoders; never native quality."""

from __future__ import annotations

import argparse
import gc
import sys
import time

import numpy as np
import torch
from chromaseed_head_range import Predictor, predict_torch
from chromaseed_head_range_fit import fit as original_fit
from chromaseed_head_range_verification import (
    check_hashes,
    digest,
    read,
    require_quiet_host,
    write_once,
)
from chromaseed_palette_transfer import ARMS, VARIANTS
from chromaseed_palette_transfer_fit import fit
from chromaseed_palette_transfer_run import (
    ROOT,
    RUN,
    encoders_for,
    require_hr_sealed,
    verify_registration,
)
from chromaseed_refine_train import setup
from chromaseed_widen_audit import exact
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    device = args.device
    if device == "cuda":
        require_hr_sealed()
        require_quiet_host()
    else:
        assert not torch.cuda.is_initialized()
    registration = verify_registration()
    output = RUN / f"preflight_{device}.json"
    if output.exists():
        value = read(output)
        assert value["passed"] and value["registration_sha256"] == digest(RUN / "registration.json")
        print("P3 EXISTING PREFLIGHT VERIFIED", device, digest(output), flush=True)
        return
    setup(device)
    sys.path.insert(0, str(ROOT / "tests"))
    from test_chromaseed_architecture_scale import fixture

    x, tokens, y, warm = fixture()
    weights = np.linspace(0.25, 2, len(y))
    steps, half = (2, 1) if device == "cpu" else (8, 4)
    engine = "eager" if device == "cpu" else "cuda_graph"
    checkpoints = (half, steps)
    originals = prefixes = predictions = 0
    maximum, peak, records = 0.0, 0, []
    began = time.perf_counter()
    with threadpool_limits(limits=1):
        for variant in VARIANTS:
            for mode in ("unit", "wide", "linear"):
                expected, oldinfo = original_fit(
                    x, tokens, y, weights, warm, variant, mode, steps, checkpoints, device, engine
                )
                for arm in ARMS:
                    encoders = encoders_for(arm)
                    fitted, info = fit(
                        x,
                        tokens,
                        y,
                        weights,
                        warm,
                        variant,
                        mode,
                        arm,
                        encoders,
                        steps,
                        checkpoints,
                        device,
                        engine,
                    )
                    assert info["sampling_sha256"] == oldinfo["sampling_sha256"]
                    assert info["final_learning_rates"] == oldinfo["final_learning_rates"]
                    if device == "cuda":
                        peak = max(
                            peak,
                            info["cuda_peak_allocated_bytes"],
                            oldinfo["cuda_peak_allocated_bytes"],
                        )
                        assert peak <= 6.5e9
                    if arm == "original":
                        for step in checkpoints:
                            for a, b in zip(expected[step], fitted[step], strict=True):
                                exact(a, b)
                                originals += 1
                    else:
                        short, shortinfo = fit(
                            x,
                            tokens,
                            y,
                            weights,
                            warm,
                            variant,
                            mode,
                            arm,
                            encoders,
                            half,
                            (half,),
                            device,
                            engine,
                        )
                        assert info["initial_theta_sha256"] == shortinfo["initial_theta_sha256"]
                        for a, b in zip(short[half], fitted[half], strict=True):
                            exact(a, b)
                            prefixes += 1
                        del short
                    if device == "cpu":
                        assert not torch.cuda.is_initialized()
                    local = 0.0
                    hashes = []
                    for step in checkpoints:
                        for slot, model in enumerate(fitted[step]):
                            a = Predictor(model)(x[:3], tokens[:3], all_passes=True)
                            b = predict_torch(
                                model, x[:3], tokens[:3], device=device, all_passes=True
                            )
                            assert np.isfinite(a).all() and np.isfinite(b).all()
                            np.testing.assert_allclose(a, b, atol=0.002, rtol=1e-6)
                            local = max(local, float(np.max(np.abs(a - b))))
                            predictions += 3
                            from chromaseed_palette_transfer import encoder_digest

                            hashes.append(
                                dict(step=step, slot=slot, payload_digest=encoder_digest(model))
                            )
                    maximum = max(maximum, local)
                    records.append(
                        dict(
                            variant=variant,
                            mode=mode,
                            arm=arm,
                            fit=info,
                            maximum_native_lab=local,
                            payloads=hashes,
                        )
                    )
                    print(
                        "P3 SYNTHETIC PREFLIGHT",
                        device,
                        variant,
                        mode,
                        arm,
                        "maxLab",
                        local,
                        flush=True,
                    )
                    del fitted
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
                del expected
    assert originals == prefixes == 108 and predictions == 972 and len(records) == 27
    if device == "cpu":
        assert not torch.cuda.is_initialized()
    check_hashes(registration["bindings"])
    result = dict(
        passed=True,
        registration_sha256=digest(RUN / "registration.json"),
        synthetic=True,
        actual_p2_encoders=True,
        native_finetuning_performed=False,
        native_accuracy_claim=False,
        device=device,
        engine=engine,
        steps=steps,
        original_exact_payloads=originals,
        transferred_prefix_exact_payloads=prefixes,
        synthetic_example_predictions=predictions,
        maximum_native_lab=maximum,
        cuda_peak_allocated_bytes=peak if device == "cuda" else None,
        records=records,
        seconds=time.perf_counter() - began,
    )
    write_once(output, result)
    print("P3 PREFLIGHT PASSED", device, digest(output), "seconds", result["seconds"], flush=True)


if __name__ == "__main__":
    main()
