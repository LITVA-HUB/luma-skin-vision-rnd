"""Independent scalar atan2 audit of locked real-camera benchmark reporting."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from cc_v7_external_data import CACHE
from cc_v7_external_lock import BENCH, ROOT, checked_lock, load

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json


def angle(a, b):
    cross = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    return math.degrees(
        math.atan2(
            math.sqrt(math.fsum(x * x for x in cross)), math.fsum(x * y for x, y in zip(a, b))
        )
    )


def percentile(values, fraction):
    ordered = sorted(values)
    location = (len(ordered) - 1) * fraction
    low, high = math.floor(location), math.ceil(location)
    return ordered[low] + (ordered[high] - ordered[low]) * (location - low)


def statistics(values):
    ordered = sorted(values)
    n = len(values)
    tail = max(1, math.ceil(n / 4))

    def mean(x):
        return math.fsum(x) / len(x)

    return {
        "n": n,
        "mean": mean(values),
        "median": percentile(values, 0.5),
        "trimean": (
            percentile(values, 0.25) + 2 * percentile(values, 0.5) + percentile(values, 0.75)
        )
        / 4,
        "best25": mean(ordered[:tail]),
        "worst25": mean(ordered[-tail:]),
        "p90": percentile(values, 0.9),
        "p95": percentile(values, 0.95),
        "max": ordered[-1],
        "over10_fraction": sum(x > 10 for x in values) / n,
        "over20_fraction": sum(x > 20 for x in values) / n,
    }


def run(digest):
    lock = checked_lock(digest)
    out = BENCH / "independent_metric_audit.json"
    if out.exists():
        raise ValueError("Immutable audit exists")
    prep = load(CACHE / "preparation.json")
    predictions = load(BENCH / "predictions/manifest.json")
    evaluation = load(BENCH / "evaluation/results.json")
    report_manifest = load(BENCH / "report/manifest.json")
    assert (
        prep["method_lock_sha256"]
        == predictions["method_lock_sha256"]
        == evaluation["method_lock_sha256"]
        == report_manifest["method_lock_sha256"]
        == digest
    )
    assert predictions["preparation_sha256"] == sha256(CACHE / "preparation.json")
    assert evaluation["prediction_manifest_sha256"] == sha256(BENCH / "predictions/manifest.json")
    assert report_manifest["evaluation_sha256"] == sha256(BENCH / "evaluation/results.json")
    for name, expected in prep["sha256"].items():
        assert sha256(CACHE / name) == expected
    for name, expected in predictions["outputs_sha256"].items():
        assert sha256(BENCH / "predictions" / name) == expected
    for name, expected in report_manifest["sha256"].items():
        assert sha256(BENCH / "report" / name) == expected
    methods = {m["id"]: m for m in lock["methods"]}
    raw = load(ROOT / lock["raw_calibration"])["methods"]
    source = {}
    for dataset in ("external", "source_test"):
        rows = load(CACHE / (dataset + "_manifest.json"))
        with (
            np.load(CACHE / (dataset + ".npz")) as values,
            np.load(BENCH / "evaluation" / (dataset + "_references.npz")) as refs,
        ):
            np.testing.assert_array_equal(values["gt"], refs["gt"])
            assert refs["ids"].tolist() == [r["id"] for r in rows]
            source[dataset] = (rows, refs["gt"].tolist())
    differences, audits = [], []

    def compare(expected, observed):
        for key, value in expected.items():
            if key == "n" or key.endswith("fraction"):
                assert observed[key] == value, (key, observed[key], value)
            else:
                differences.append(abs(value - observed[key]))

    assert len(evaluation["records"]) == 29 * 9
    for record in evaluation["records"]:
        method = methods[record["method"]]
        rows, gt = source[record["dataset"]]
        selected = record["indices"]
        name = record["dataset"] + "__" + record["method"] + ".npz"
        with np.load(BENCH / "predictions" / name) as pred:
            assert pred["ids"].tolist() == [r["id"] for r in rows]
            p, s, v = (pred[k][selected].tolist() for k in ("pred", "scores", "valid"))
        g = [gt[i] for i in selected]
        ids = [rows[i]["id"] for i in selected]
        errors = [angle([b / a for a, b in zip(pi, gi)], [1, 1, 1]) for pi, gi in zip(p, g)]
        recovery = [angle(pi, gi) for pi, gi in zip(p, g)]
        metrics = record["metrics"]
        assert metrics["valid"] == v
        assert metrics["invalid_n"] == len(v) - sum(v)
        compare(statistics(errors), metrics["reproduction"])
        compare(statistics(recovery), metrics["recovery"])
        order = sorted(
            (i for i in range(len(v)) if v[i]),
            key=lambda i: (s[i], hashlib.sha256(ids[i].encode()).hexdigest()),
        )
        curve = metrics["selective"]
        assert curve["order"] == order
        risks = [math.fsum(errors[i] for i in order[:n]) / n for n in range(1, len(order) + 1)]
        differences.extend(abs(a - b) for a, b in zip(risks, curve["risk"]))
        assert len(risks) == len(curve["risk"]) == len(curve["coverage"])
        differences.extend(abs(n / len(v) - c) for n, c in enumerate(curve["coverage"], 1))
        if risks:
            differences.append(abs(math.fsum(risks) / len(risks) - curve["aurc"]))
        else:
            assert curve["aurc"] is None
        cal = (
            load(ROOT / method["selection"])["heads"]["combined"]["cal_scores"]
            if "selection" in method
            else raw[method["id"]]["cal_scores"]
        )
        for coverage in (100, 95, 90, 80, 70, 60):
            n = max(1, len(v) * coverage // 100)
            fixed = curve["fixed"][str(coverage)]
            if len(order) >= n:
                compare(statistics([errors[i] for i in order[:n]]), fixed)
                assert fixed["coverage"] == n / len(v)
            else:
                assert fixed["mean"] is None and fixed["attainable"] is False
            threshold = None if coverage == 100 else percentile(cal, coverage / 100)
            frozen = metrics["frozen_source_thresholds"][str(coverage)]
            if threshold is None:
                assert frozen["threshold"] is None
            else:
                differences.append(abs(frozen["threshold"] - threshold))
            accepted = [
                i for i in range(len(v)) if v[i] and (threshold is None or s[i] <= threshold)
            ]
            assert frozen["n"] == len(accepted)
            assert frozen["coverage"] == len(accepted) / len(v)
            if accepted:
                differences.append(
                    abs(
                        math.fsum(errors[i] for i in accepted) / len(accepted)
                        - frozen["mean_reproduction"]
                    )
                )
            else:
                assert frozen["mean_reproduction"] is None
        audits.append(
            {
                "method": record["method"],
                "population": record["population"],
                "n": len(v),
                "supported": sum(v),
            }
        )
    maximum = max(differences)
    if maximum > 1e-7:
        raise ValueError(f"Independent scalar metric mismatch: {maximum}")
    write_json(
        out,
        {
            "status": "PASS: independent scalar atan2 metrics, all rankings, curves and source thresholds",
            "method_population_records": len(audits),
            "fixed_coverage_and_threshold_cases": len(audits) * 6,
            "maximum_difference": maximum,
            "method_lock_sha256": digest,
            "evaluation_sha256": sha256(BENCH / "evaluation/results.json"),
            "script_sha256": sha256(Path(__file__)),
            "records": audits,
        },
    )
    print(json.dumps({"records": len(audits), "maximum_difference": maximum}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--digest", required=True)
    run(parser.parse_args().digest)
