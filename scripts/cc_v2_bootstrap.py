"""Paired dataset uncertainty conditional on frozen trained seeds and selectors."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json


def records(root, group, domain):
    run, block = group.split("::")
    values = []
    for path in sorted((root / "runs").glob("*.json")):
        if re.sub(r"_s\d+$", "", path.stem) == run:
            values.append(json.loads(path.read_text(encoding="utf-8"))["domains"][domain][block])
    if not values:
        raise ValueError("Unknown comparison: " + group)
    return values


def compare(root, a, b, domain, *, draws=2000, seed=1209, cluster=False):
    first, second = records(root, a, domain), records(root, b, domain)
    ids = first[0]["ids"]
    if any(r["ids"] != ids for r in first + second):
        raise ValueError("Paired comparison requires identical image IDs/order")
    ties = np.array(
        [int(hashlib.sha256(i.encode()).hexdigest()[:16], 16) for i in ids], dtype=np.uint64
    )
    if len(np.unique(ties)) != len(ids):
        raise ValueError("Truncated tie hashes collide")
    errors = [np.asarray(r["errors"]) for r in first + second]
    scores = [np.asarray(r["scores"]) for r in first + second]
    if domain.startswith("fresh"):
        manifest = json.loads(
            Path("data/processed/cc_v2_fresh128/fresh_manifest.json").read_text(encoding="utf-8")
        )
        by_id = {r["id"]: r for r in manifest}
        strata = np.array([by_id[i]["camera"] for i in ids])
        grouping = np.array(
            [by_id[i]["camera"] + ":" + by_id[i]["reference_hash"] if cluster else i for i in ids]
        )
    else:
        strata = np.array(["source"] * len(ids))
        grouping = np.array(first[0]["groups"] if cluster else ids)
    strata_groups = []
    for camera in np.unique(strata):
        mask = strata == camera
        strata_groups.append(
            [np.flatnonzero(mask & (grouping == g)) for g in np.unique(grouping[mask])]
        )
    rng = np.random.default_rng(seed)

    def difference(idx):
        values = []
        for e, s in zip(errors, scores, strict=True):
            order = np.lexsort((ties[idx], s[idx]))
            n = max(1, int(np.floor(len(idx) * 0.8)))
            ranked = e[idx[order]]
            aurc = np.mean(np.cumsum(ranked) / np.arange(1, len(ranked) + 1))
            values.append([e[idx].mean(), ranked[:n].mean(), aurc])
        return np.mean(values[: len(first)], axis=0) - np.mean(values[len(first) :], axis=0)

    boot = []
    for _ in range(draws):
        idx = np.concatenate(
            [
                group
                for groups in strata_groups
                for group in [groups[i] for i in rng.integers(0, len(groups), len(groups))]
            ]
        )
        boot.append(difference(idx))
    boot = np.asarray(boot)
    observed = difference(np.arange(len(ids)))
    return {
        "first": a,
        "second": b,
        "domain": domain,
        "n": len(ids),
        "runs_first": len(first),
        "runs_second": len(second),
        "draws": draws,
        "seed": seed,
        "script_sha256": sha256(Path(__file__)),
        "grouping": "camera-stratified exact-reference-hash proxy clusters"
        if domain.startswith("fresh") and cluster
        else "capture-date clusters"
        if cluster
        else "camera-stratified image resampling"
        if domain.startswith("fresh")
        else "image resampling",
        "clusters": sum(len(v) for v in strata_groups),
        "interpretation": "First minus second; negative favors first. Percentile95% interval, conditional on trained seeds. Referencehash is a sensitivity proxy, not verified scene independence. No population or deployment guarantee.",
        "mean": {
            "difference": float(observed[0]),
            "ci95": np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
        },
        "risk80": {
            "difference": float(observed[1]),
            "ci95": np.percentile(boot[:, 1], [2.5, 97.5]).tolist(),
        },
        "aurc": {
            "difference": float(observed[2]),
            "ci95": np.percentile(boot[:, 2], [2.5, 97.5]).tolist(),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("docs/benchmarks/cc_v2"))
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = [
        compare(args.root, args.first, other, domain, cluster=cluster)
        for other in args.second
        for domain in ("source_regression", "fresh_all")
        for cluster in (False, True)
    ]
    write_json(args.output, result)
    print(json.dumps(result), flush=True)
