"""Preserve exact training/selection metadata and code bytes without copying datasets/weights."""

import importlib.metadata
import json
from pathlib import Path

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json


def main():
    destination = Path("docs/benchmarks/cc_v2/reproducibility")
    records = []

    def copy(source, relative):
        target = destination / relative
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError("Refusing to overwrite changed evidence: " + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        records.append(
            {
                "original": source.as_posix(),
                "archived": target.as_posix(),
                "sha256": sha256(target),
                "bytes": target.stat().st_size,
            }
        )

    for run in sorted(Path("experiments/runs").glob("ccv2_*_s*")):
        for relative in (
            "config.json",
            "training.json",
            "checkpoint_manifest.json",
            "predictions_manifest.json",
            "selection.json",
            "risk_v2/selection.json",
        ):
            source = run / relative
            if source.is_file():
                copy(source, Path("runs") / run.name / relative)
    for run_name in ("ccv2_statistics", "ccv2_statistics_fixed"):
        run = Path("experiments/runs") / run_name
        copy(run / "screen.json", Path("statistics") / run_name / "screen.json")
        copy(run / "script_snapshot.py", Path("statistics") / run_name / "script_snapshot.py.txt")
    for run in sorted(Path("experiments/runs/ccv2_statistics_eval").iterdir()):
        if run.is_dir():
            copy(run / "selection.json", Path("runs") / run.name / "selection.json")
            for source in run.glob("*.manifest.json"):
                copy(source, Path("runs") / run.name / source.name)
    snapshot = Path("experiments/runs/ccv2_sog_large_g0_s17/source_snapshot")
    for source in sorted(snapshot.rglob("*")):
        if source.is_file():
            relative = source.relative_to(snapshot)
            # Text suffix keeps archived Python source out of active imports/lint.
            copy(source, Path("source_snapshot") / (relative.as_posix() + ".txt"))
    licenses = {}
    for name in ("scikit-learn", "scipy", "joblib", "threadpoolctl"):
        dist = importlib.metadata.distribution(name)
        licenses[name] = {
            "version": dist.version,
            "license_expression": dist.metadata.get("License-Expression"),
            "license": dist.metadata.get("License"),
            "files": [],
        }
        for entry in dist.files or []:
            if ".dist-info/" in entry.as_posix() and any(
                token in entry.name.lower() for token in ("license", "copying", "notice")
            ):
                source = Path(dist.locate_file(entry))
                relative = Path("licenses") / name / entry.name
                copy(source, relative)
                licenses[name]["files"].append(str(destination / relative))
    write_json(destination / "installed_research_licenses.json", licenses)
    write_json(
        destination / "archive_manifest.json",
        {
            "scope": "Exact-byte metadata/source only; original local weights/data remain excluded from Git. No external publication.",
            "source_hash": "598a44f190d624db58fd4d2a60368f59bf41b1f3e8620c8c49c55d6e1e680398",
            "files": records,
        },
    )
    print(json.dumps({"files": len(records), "bytes": sum(r["bytes"] for r in records)}))


if __name__ == "__main__":
    main()
