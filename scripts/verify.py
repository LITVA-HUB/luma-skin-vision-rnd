"""Record local verification evidence; exit nonzero on any failed check."""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from luma_skin_vision.experiment import source_identity


def main():
    root = Path(__file__).resolve().parents[1]
    checks = [
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        ["git", "diff", "--check"],
        ["uv", "lock", "--check"],
    ]
    outputs = []
    for command in checks:
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        outputs.append(
            {
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        print(f"{command[1:]}: exit {result.returncode}", flush=True)
    match = re.search(r"(\d+) passed", outputs[0]["stdout"])
    evidence = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **source_identity(),
        "all_passed": all(x["exit_code"] == 0 for x in outputs),
        "tests_passed": int(match.group(1)) if match else None,
        "checks": outputs,
    }
    path = root / "docs/research/verification.json"
    path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "evidence": str(path),
                "all_passed": evidence["all_passed"],
                "tests_passed": evidence["tests_passed"],
            }
        )
    )
    if not evidence["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
