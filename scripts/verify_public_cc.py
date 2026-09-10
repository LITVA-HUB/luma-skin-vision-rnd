"""Verify the public milestone without overwriting historical synthetic evidence."""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from luma_skin_vision.cc.benchmark import data_hashes
from luma_skin_vision.experiment import source_identity, write_json

root = Path(__file__).resolve().parents[1]
env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
commands = [
    [sys.executable, "-m", "pytest", "-q"],
    [sys.executable, "-m", "ruff", "check", "."],
    [sys.executable, "-m", "ruff", "format", "--check", "."],
    ["git", "diff", "--check"],
    ["uv", "lock", "--check"],
]
results = []
for cmd in commands:
    run = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, encoding="utf-8")
    results.append(
        {"command": cmd, "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    )
    print(" ".join(cmd), run.returncode, flush=True)
hashes = data_hashes(root / "data/processed/cc128")
configs = list((root / "docs/benchmarks/public_runs").glob("*/config.json"))
assert len(configs) == 8
for file in configs:
    assert json.loads(file.read_text())["data_hashes"] == hashes
record = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    **source_identity(),
    "checks": results,
    "dataset_binding": "8/8 run manifests match current immutable caches",
    "data_hashes": hashes,
    "all_passed": all(r["returncode"] == 0 for r in results),
}
write_json(root / "docs/research/public_verification.json", record)
if not record["all_passed"]:
    print(json.dumps(results, indent=2))
    raise SystemExit(1)
