"""Reproduce the largest FP16 drift without changing any exit threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from chromaseed_refine_audit import read_npz
from chromaseed_refine_numpy import NumpyRefiner
from chromaseed_refine_precision import decode
from skin_local_search_train import CACHE_HASH, roles, sha, write_json

from luma_skin_vision.color import delta_e00


def diagnose(source_run, precision_run, cache, output):
    if cache.name != "train.npz" or sha(cache) != CACHE_HASH:
        raise ValueError("only original TRAIN allowed")
    records = json.loads((precision_run / "results.json").read_text(encoding="utf-8"))["records"]
    worst = max((r for r in records if r["precision"] == "fp16_storage"), key=lambda r: r["max_prediction_delta_e00"])
    role, family, seed = worst["role"], worst["family"], worst["seed"]
    a_path = source_run / "final" / role / family / f"seed{seed}.npz"
    b_path = precision_run / role / family / f"seed{seed}_fp16_storage.npz"
    a, b = NumpyRefiner(read_npz(a_path)), NumpyRefiner(decode(read_npz(b_path)))
    with np.load(cache, allow_pickle=False) as z:
        x, patches, person, device = z["color"], z["tokens"], z["patient"], z["device"]
    idx = np.flatnonzero(roles(person, device)[role][1])
    pairs = []
    for row in idx:
        pa, na, _ = a.predict(x[row], patches[row])
        pb, nb, _ = b.predict(x[row], patches[row])
        pairs.append((float(delta_e00(pa[None], pb[None])[0]), row, na, nb))
    drift, row, na, nb = max(pairs)
    ta, ca = a.predict_trace(x[row], patches[row], threshold=0.)
    tb, cb = b.predict_trace(x[row], patches[row], threshold=0.)
    result = {"type": "post-hoc diagnosis, no threshold adjustment", "role": role, "family": family, "seed": seed,
              "diagnosis_source_sha256": sha(Path(__file__)), "cache_sha256": CACHE_HASH,
              "original_weight_sha256": sha(a_path), "encoded_weight_sha256": sha(b_path),
              "maximum_adaptive_prediction_delta_e00": drift, "original_executed_passes": na, "fp16_executed_passes": nb,
              "locked_threshold_native_lab": a.threshold,
              "original_native_lab_increment_norms": np.linalg.norm(np.diff(ta, axis=0), axis=1).tolist(),
              "fp16_native_lab_increment_norms": np.linalg.norm(np.diff(tb, axis=0), axis=1).tolist(),
              "matched_pass_prediction_delta_e00": delta_e00(ta, tb).tolist(), "original_patch_counts": ca.tolist(),
              "fp16_patch_counts": cb.tolist(), "changed_exits_count": sum(aa != bb for _, _, aa, bb in pairs), "n_rows": len(idx)}
    write_json(output / "precision_exit_diagnosis.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--precision-run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.source_run, args.precision_run, args.cache, args.output)
