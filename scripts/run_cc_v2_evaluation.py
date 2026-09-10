"""Execute the prespecified matrix only after checking the immutable method lock."""

import json
import subprocess
import sys
from pathlib import Path

from luma_skin_vision.data import sha256

base = Path("docs/benchmarks/cc_v2")
lock_path = base / "final_head_lock.json"
lock = json.loads(lock_path.read_text(encoding="utf-8"))
if lock["target_errors_observed"] is not False:
    raise ValueError("Expected pre-evaluation method lock")
for name, digest in lock["script_hashes"].items():
    if sha256(Path("scripts", name)) != digest:
        raise ValueError("Evaluation implementation changed since method lock")
fresh = [
    "data/processed/cc_v2_fresh128/fresh.npz",
    "data/processed/cc_v2_fresh128/fresh_manifest.json",
]
for key, path in zip(("cache", "manifest"), fresh, strict=True):
    if sha256(Path(path)) != lock["fresh_data"][key]:
        raise ValueError("Fresh evaluation input changed")
output = base / "runs"
output.mkdir(exist_ok=True)
logs = Path("artifacts/cc_v2_evaluation")
logs.mkdir(parents=True, exist_ok=True)
for category in ("cnn_selectors", "legacy_selectors"):
    for name, digest in lock[category].items():
        run = Path("experiments/runs", name)
        selection = run / (
            "risk_v2/selection.json" if category == "cnn_selectors" else "selection.json"
        )
        if sha256(selection) != digest:
            raise ValueError("Selector changed since lock")
        target = output / (name + ".json")
        if target.exists():
            raise FileExistsError("Evaluation output already exists: " + str(target))
        if category == "cnn_selectors":
            command = [
                sys.executable,
                "scripts/cc_v2_select.py",
                "evaluate",
                "--runs",
                str(run),
                "--external",
                *fresh,
            ]
        else:
            legacy = json.loads(selection.read_text(encoding="utf-8"))["legacy_run"]
            command = [
                sys.executable,
                "scripts/cc_v2_legacy.py",
                "evaluate",
                "--run",
                legacy,
                "--out",
                str(run),
                "--external",
                *fresh,
                "--destination",
                str(target),
            ]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        (logs / (name + ".log")).write_text(
            completed.stdout + "\n" + completed.stderr, encoding="utf-8"
        )
        if completed.returncode:
            raise RuntimeError(name + ": " + completed.stderr[-3000:])
        result = json.loads(target.read_text(encoding="utf-8"))
        block = "combined" if category == "cnn_selectors" else "upgraded_combined"
        print(
            json.dumps(
                {
                    "completed": name,
                    "source": {
                        "mean": result["domains"]["source_regression"][block]["reproduction"][
                            "mean"
                        ],
                        "risk80": result["domains"]["source_regression"][block]["selective"][
                            "fixed"
                        ]["80"]["mean"],
                    },
                    "fresh": {
                        "mean": result["domains"]["fresh_all"][block]["reproduction"]["mean"],
                        "risk80": result["domains"]["fresh_all"][block]["selective"]["fixed"]["80"][
                            "mean"
                        ],
                    },
                }
            ),
            flush=True,
        )
