"""Lossless modular integer differences of FP32 bit patterns; CPU probe only."""

from __future__ import annotations

import hashlib
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from chromaseed_architecture_scale import VARIANTS, Bank, capacity
from chromaseed_architecture_scale_run import ROOT, bank_path
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json

OUT = ROOT / "docs/benchmarks/chromaseed_lossless_storage_probe"


def trial(theta, initial):
    original = np.ascontiguousarray(theta, dtype="<f4").view("<u4")
    reference = np.ascontiguousarray(initial, dtype="<f4").view("<u4")
    start = time.perf_counter()
    difference = np.subtract(original, reference, dtype="<u4").view("<i4").astype(np.int64)
    zigzag = ((difference << 1) ^ (difference >> 31)).astype("<u4")
    planes = zigzag.view(np.uint8).reshape(-1, 4).T.copy()
    packed = zlib.compress(planes.tobytes(), level=3)
    encode_seconds = time.perf_counter() - start
    start = time.perf_counter()
    decoded = np.frombuffer(zlib.decompress(packed), dtype=np.uint8)
    words = decoded.reshape(4, -1).T.copy().reshape(-1).view("<u4").astype(np.int64)
    delta = ((words >> 1) ^ -(words & 1)).astype("<u4")
    reconstructed = np.add(delta, reference, dtype="<u4")
    assert reconstructed.tobytes() == original.tobytes()
    return dict(
        original_bytes=original.nbytes,
        compressed_bytes=len(packed),
        ratio=len(packed) / original.nbytes,
        encode_seconds=encode_seconds,
        decode_seconds=time.perf_counter() - start,
        exact_sha256=hashlib.sha256(reconstructed.tobytes()).hexdigest(),
    )


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    parent_path = OUT / "probe.json"
    parent = js(parent_path)
    assert parent["source_sha256"] == sha(ROOT / "scripts/chromaseed_lossless_storage_probe.py")
    for path, digest in parent["input_sha256"].items():
        assert sha(ROOT / path) == digest
    output = OUT / "arithmetic_probe.json"
    assert not output.exists()
    r = np.random.default_rng(73985)
    a = r.integers(0, 2**32, 100_000, dtype="<u4")
    b = r.integers(0, 2**32, 100_000, dtype="<u4")
    a[:7] = [0, 0x80000000, 0x7FC00001, 0x7F800000, 0xFF800000, 1, 0xFFFFFFFF]
    trial(a.view("<f4"), b.view("<f4"))
    records = []
    for variant in ("patch5m", "soft5m", "dynamic5m", "pool5m"):
        initial = Bank(variant, seeds=(17,)).theta.detach().numpy()[0].copy()
        for step in (128, 2048):
            with np.load(
                bank_path("mixed", variant, 0) / f"models_{step}.npz", allow_pickle=False
            ) as archive:
                for slot in (0, 1):
                    record = dict(
                        variant=variant,
                        step=step,
                        slot=slot,
                        **trial(archive[f"{slot}__theta"], initial),
                    )
                    matched = next(
                        v
                        for v in parent["records"]
                        if (v["variant"], v["step"], v["slot"]) == (variant, step, slot)
                    )
                    assert record["exact_sha256"] == matched["exact_sha256"]
                    records.append(record)
        print("ARITHMETIC LOSSLESS", variant, "verified", flush=True)
    ratios = {
        v: max(r["ratio"] for r in records if r["variant"] == v)
        for v in {r["variant"] for r in records}
    }
    projected = (
        sum(4 * (capacity(v) - 643) * ratios.get(v, 1) for v in VARIANTS)
        * parent["planned_model_payloads_per_architecture"]
    )
    assert not torch.cuda.is_initialized()
    write_json(
        output,
        dict(
            classification="CPU lossless storage probe, not a production format or guarantee",
            passed=True,
            source_sha256=sha(Path(__file__)),
            parent_sha256=sha(parent_path),
            method="modular IEEE754 bit-pattern subtraction, signed zigzag, byte shuffle, zlib3",
            records=records,
            random_bit_patterns_exact=100000,
            largest_observed_ratio_by_architecture=ratios,
            theta_only_projected_bytes=projected,
            cuda_context_initialized=False,
            limitations=parent["limitations"],
        ),
    )
    print("ARITHMETIC PROJECTED GB", round(projected / 1e9, 3), flush=True)


if __name__ == "__main__":
    main()
