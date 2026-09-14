"""CPU-only lossless-storage estimate from completed AS checkpoints.

This probe never rewrites an AS archive and is not a production checkpoint
format. Reconstructed bytes must be identical, including all FP32 bits.
"""

from __future__ import annotations

import hashlib
import shutil
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from chromaseed_architecture_scale import VARIANTS, Bank, capacity
from chromaseed_architecture_scale_run import ROOT, RUN, bank_path
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json

OUT = ROOT / "docs/benchmarks/chromaseed_lossless_storage_probe"


def trial(theta, initial, method):
    original = np.ascontiguousarray(theta, dtype="<f4").view("<u4")
    reference = np.ascontiguousarray(initial, dtype="<f4").view("<u4")
    assert original.shape == reference.shape and original.ndim == 1
    start = time.perf_counter()
    value = np.bitwise_xor(original, reference) if method == "xor_shuffle" else original
    if method != "raw":
        value = value.view(np.uint8).reshape(-1, 4).T.copy()
    packed = zlib.compress(value.tobytes(), level=3)
    encode_seconds = time.perf_counter() - start
    start = time.perf_counter()
    decoded = np.frombuffer(zlib.decompress(packed), dtype=np.uint8)
    if method != "raw":
        decoded = decoded.reshape(4, -1).T.copy().reshape(-1)
    decoded = decoded.view("<u4")
    if method == "xor_shuffle":
        decoded = np.bitwise_xor(decoded, reference)
    assert decoded.tobytes() == original.tobytes(), "lossless reconstruction failed"
    return dict(
        method=method,
        original_bytes=original.nbytes,
        compressed_bytes=len(packed),
        ratio=len(packed) / original.nbytes,
        encode_seconds=encode_seconds,
        decode_seconds=time.perf_counter() - start,
        exact_sha256=hashlib.sha256(decoded.tobytes()).hexdigest(),
    )


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    lock = js(RUN / "source_lock.json")
    for path, digest in lock["sources"].items():
        if "architecture_scale" in path:
            assert sha(ROOT / path) == digest
    output = OUT / "probe.json"
    assert not output.exists(), "retain existing measured probe"
    methods = ("raw", "shuffle", "xor_shuffle")
    bits = np.array([0, 0x80000000, 0x7FC00001, 0x7F800000, 0xFF800000, 1, 0xFFFFFFFF], "<u4")
    for method in methods:
        trial(bits.view("<f4"), np.zeros(len(bits), "<f4"), method)
    inputs, records = {}, []
    for variant in ("patch5m", "soft5m", "dynamic5m", "pool5m"):
        initial = Bank(variant, seeds=(17,)).theta.detach().numpy()[0].copy()
        path = bank_path("mixed", variant, 0)
        receipt = js(path / "receipt.json")
        assert receipt["source_lock_sha256"] == sha(RUN / "source_lock.json")
        inputs[(path / "receipt.json").relative_to(ROOT).as_posix()] = sha(path / "receipt.json")
        for step in (128, 2048):
            source = path / f"models_{step}.npz"
            digest = sha(source)
            assert digest == receipt["files"][source.name]
            inputs[source.relative_to(ROOT).as_posix()] = digest
            with np.load(source, allow_pickle=False) as archive:
                for slot in (0, 1):
                    theta = archive[f"{slot}__theta"]
                    for method in methods:
                        record = dict(
                            variant=variant, step=step, slot=slot, **trial(theta, initial, method)
                        )
                        records.append(record)
            print("LOSSLESS PROBE", variant, step, "verified", flush=True)
    # Two changed heads, nine inner folds per architecture, three archived
    # checkpoints with six models each, plus final banks and selected copies.
    new_models_per_architecture = 2 * (9 * 3 * 6 + 3 * 6 + 3 * 3)
    raw_theta_bytes = sum(4 * (capacity(v) - 643) for v in VARIANTS) * new_models_per_architecture
    estimates = {}
    for method in methods:
        ratios = {
            v: max(r["ratio"] for r in records if r["variant"] == v and r["method"] == method)
            for v in ("patch5m", "soft5m", "dynamic5m", "pool5m")
        }
        estimates[method] = dict(
            largest_observed_ratio_by_architecture=ratios,
            theta_only_projected_bytes=sum(
                4 * (capacity(v) - 643) * ratios.get(v, 1.0) for v in VARIANTS
            )
            * new_models_per_architecture,
        )
    assert not torch.cuda.is_initialized()
    value = dict(
        classification="lossless storage feasibility only; no new training and no production codec",
        passed=True,
        source_sha256=sha(Path(__file__)),
        as_source_lock_sha256=sha(RUN / "source_lock.json"),
        input_sha256=inputs,
        records=records,
        exceptional_bit_patterns_exact=True,
        numpy=np.__version__,
        torch=torch.__version__,
        zlib=zlib.ZLIB_RUNTIME_VERSION,
        cuda_context_initialized=False,
        proposed_new_heads=2,
        planned_model_payloads_per_architecture=new_models_per_architecture,
        raw_theta_projected_bytes=raw_theta_bytes,
        estimates=estimates,
        free_bytes_at_probe=shutil.disk_usage(ROOT).free,
        limitations=[
            "Estimates cover theta arrays only, excluding metadata, predictions and any extra controls.",
            "Measurements use mixed fold0, seed17, both rates, checkpoints128/2048 from four completed AS architectures.",
            "New head trajectories may compress differently; projections are not storage guarantees.",
            "XOR requires the exact deterministic initial weights and bound implementation to reconstruct.",
            "Timings are individual CPU observations during a GPU experiment, not deployment performance claims.",
        ],
    )
    write_json(output, value)
    print("RAW PROJECTED GB", round(raw_theta_bytes / 1e9, 3), flush=True)
    for method, estimate in estimates.items():
        print(method, round(estimate["theta_only_projected_bytes"] / 1e9, 3), "GB", flush=True)


if __name__ == "__main__":
    main()
