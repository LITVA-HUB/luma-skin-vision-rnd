"""Equal checkpoint-weight means with explicit same-trajectory provenance."""

from __future__ import annotations

import numpy as np
from chromaseed_neural_prefix_numpy import Predictor

STEPS = (0, 512, 2048, 8192, 32768, 131072)
METHODS = ("last", "pair", "prefix", "tail3")
WEIGHTS = ("w0", "b0", "v0", "c0")


def average(models, provenance):
    if not models or len(models) != len(provenance):
        raise ValueError("matching nonempty models and provenance required")
    fields = ("context", "seed", "variants", "lr", "step")
    if any(set(p) != set(fields) for p in provenance):
        raise ValueError("complete provenance required")
    if any(any(p[k] != provenance[0][k] for k in fields[:-1]) for p in provenance):
        raise ValueError("checkpoints must come from the same trajectory")
    steps = [p["step"] for p in provenance]
    if steps != sorted(set(steps)) or any(s not in STEPS for s in steps):
        raise ValueError("distinct ascending registered checkpoints required")
    first = models[0]
    for m in models:
        Predictor(m)
        if str(m["family"]) != "blind4" or m["w0"].shape != (36, 16) or set(m) != set(first):
            raise ValueError("matching NP643 head required")
        for k in m:
            if m[k].dtype != first[k].dtype or m[k].shape != first[k].shape:
                raise ValueError("matching array types/shapes required")
            if k not in WEIGHTS and not np.array_equal(m[k], first[k]):
                raise ValueError("normalizers and lineage metadata must match")
        if any(m[k].dtype != np.float32 or not np.isfinite(m[k]).all() for k in WEIGHTS):
            raise ValueError("finite FP32 weights required")
    result = {k: v.copy() for k, v in first.items()}
    if len(models) > 1:
        for k in WEIGHTS:
            result[k] = (
                np.stack([m[k].astype(np.float64) for m in models]).sum(0) / len(models)
            ).astype(np.float32)
    return result


def recipes():
    out = [dict(name="base", method="last", variants=0, lr=0.0001, steps=[0])]
    positive = STEPS[1:]
    for variants in (0, 16, 256):
        for li, lr in enumerate((0.0001, 0.0003)):
            groups = {
                "last": [[s] for s in positive],
                "pair": [list(STEPS[i : i + 2]) for i in range(5)],
                "prefix": [list(positive[:i]) for i in (3, 4, 5)],
                "tail3": [list(positive[i - 3 : i]) for i in (4, 5)],
            }
            for method, parts in groups.items():
                for steps in parts:
                    out.append(
                        dict(
                            name=f"{method}_v{variants}_r{li}_t{steps[-1]}",
                            method=method,
                            variants=variants,
                            lr=lr,
                            steps=steps,
                        )
                    )
    return out


def choose(candidates):
    return min(
        candidates,
        key=lambda c: (
            c["clean"],
            c["p90"],
            max(c["steps"]),
            len(c["steps"]),
            c["variants"],
            c["lr"],
            METHODS.index(c["method"]),
        ),
    )


def endpoint_name(spec):
    return (
        "base"
        if spec["steps"][-1] == 0
        else f"last_v{spec['variants']}_r{(0.0001, 0.0003).index(spec['lr'])}_t{spec['steps'][-1]}"
    )


def final_union(entry):
    associations = {"base": dict(policies=[], endpoint_for=[], baseline=True)}
    for policy, choice in entry["policies"].items():
        associations.setdefault(choice["name"], dict(policies=[], endpoint_for=[], baseline=False))[
            "policies"
        ].append(policy)
        associations.setdefault(
            endpoint_name(choice), dict(policies=[], endpoint_for=[], baseline=False)
        )["endpoint_for"].append(policy)
    return associations
