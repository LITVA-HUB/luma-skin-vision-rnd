"""Fixed repetitions and held-out-camera protocol; sequential to avoid GPU contention."""

import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
transcript = root / "artifacts/public_cc_replications.jsonl"
env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
for protocol, seed in [("official", 29), ("official", 43), ("camera", 17)]:
    for method in ["baseline", "proposed"]:
        out = f"experiments/runs/cc_{protocol}_{method}_s{seed}"
        for action in ["train", "evaluate"]:
            command = [
                sys.executable,
                "-m",
                "luma_skin_vision.cc.benchmark",
                action,
                "--out",
                out,
                "--method",
                method,
                "--protocol",
                protocol,
                "--seed",
                str(seed),
                "--epochs",
                "60",
            ]
            print(f"RUN {protocol} {method} seed{seed} {action}", flush=True)
            result = subprocess.run(
                command, cwd=root, env=env, capture_output=True, text=True, encoding="utf-8"
            )
            with transcript.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "command": command,
                            "returncode": result.returncode,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                        }
                    )
                    + "\n"
                )
            print(result.stdout[-1600:], flush=True)
            if result.returncode:
                print(result.stderr, flush=True)
                raise SystemExit(result.returncode)
