"""Aggregate every frozen evaluation; never select a model from target errors."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json


def collect(root):
    groups = defaultdict(list)
    rows = []
    inputs = {}
    for path in sorted((root / "runs").glob("*.json")):
        values = json.loads(path.read_text(encoding="utf-8"))
        inputs[path.name] = sha256(path)
        run = re.sub(r"_s\d+$", "", path.stem)
        for domain, methods in values["domains"].items():
            for method, result in methods.items():
                key = (run + "::" + method, domain)
                groups[key].append({"file": path.name, "record": result})
                row = {
                    "run": path.stem,
                    "method": method,
                    "domain": domain,
                    "n": result["reproduction"]["n"],
                    "recovery_mean": result["recovery"]["mean"],
                    "reproduction_mean": result["reproduction"]["mean"],
                    "median": result["reproduction"]["median"],
                    "p95": result["reproduction"]["p95"],
                    "over10": result["reproduction"]["over10_fraction"],
                    "over20": result["reproduction"]["over20_fraction"],
                    "aurc": result["selective"]["aurc"],
                    "invalid_n": result.get("invalid_n", 0),
                    "frozen_source80_coverage": result["frozen_source_thresholds"]["80"][
                        "coverage"
                    ],
                    "frozen_source80_error": result["frozen_source_thresholds"]["80"][
                        "mean_reproduction"
                    ],
                }
                for percent in (100, 95, 90, 80, 70, 60):
                    row["risk" + str(percent)] = result["selective"]["fixed"][str(percent)]["mean"]
                rows.append(row)
    summary = {}
    for (group, domain), items in groups.items():
        records = [v["record"] for v in items]
        result = {
            "runs": len(records),
            "source_files": [v["file"] for v in items],
            "n_per_run": [r["reproduction"]["n"] for r in records],
        }
        metrics = {
            "mean": [r["reproduction"]["mean"] for r in records],
            "median": [r["reproduction"]["median"] for r in records],
            "p95": [r["reproduction"]["p95"] for r in records],
            "aurc": [r["selective"]["aurc"] for r in records],
            "source80_actual_coverage": [
                r["frozen_source_thresholds"]["80"]["coverage"] for r in records
            ],
            "source80_error": [
                r["frozen_source_thresholds"]["80"]["mean_reproduction"] for r in records
            ],
        }
        for percent in (100, 95, 90, 80, 70, 60):
            metrics["risk" + str(percent)] = [
                r["selective"]["fixed"][str(percent)]["mean"] for r in records
            ]
        for key, values in metrics.items():
            result[key] = {
                "mean": float(np.mean(values)) if None not in values else None,
                "std_across_runs": float(np.std(values, ddof=1))
                if len(values) > 1 and None not in values
                else None,
                "values": values,
            }
        summary.setdefault(domain, {})[group] = result
    write_json(
        root / "aggregate.json",
        {
            "scope": "All reproduced methods, source-selected configurations; seed means not pooled predictions; no target selection",
            "script_sha256": sha256(Path(__file__)),
            "inputs": inputs,
            "domains": summary,
        },
    )
    if rows:
        with (root / "all_results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for domain in ("source_regression", "fresh_all"):
        print(
            json.dumps(
                {
                    "domain": domain,
                    "results": {
                        k: {m: v[m]["mean"] for m in ("mean", "risk80", "aurc")}
                        for k, v in summary.get(domain, {}).items()
                    },
                }
            ),
            flush=True,
        )
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("docs/benchmarks/cc_v2"))
    collect(parser.parse_args().root)
