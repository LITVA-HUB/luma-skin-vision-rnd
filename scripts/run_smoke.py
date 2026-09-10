"""Run actual CLI entry points; preserve commands and outputs under artifacts."""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "artifacts" / ("smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    commands = []

    def run(script, *args):
        cmd = [sys.executable, str(ROOT / "scripts" / f"{script}.py"), *map(str, args)]
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        commands.append(
            {
                "command": cmd,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        (output / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{script} failed: {result.stderr}")
        return json.loads(result.stdout)

    run("inspect_environment")
    dataset = run("generate_synthetic_demo", "--subjects", 60)
    validated = run("validate_dataset", "--manifest", dataset["manifest"])
    repeat = run("repeatability", "--manifest", dataset["manifest"])
    runs, results = {}, {}
    for method in (
        "baseline_a0",
        "baseline_a1",
        "baseline_a2",
        "baseline_c",
        "baseline_c_plus",
        "proposed_v1",
    ):
        trained = run("train", "--config", f"configs/experiments/{method}.yaml")
        runs[method] = trained["experiment"]
        run("calibrate", "--experiment", trained["experiment"])
        results[method] = run("evaluate", "--experiment", trained["experiment"])
        print(f"{method}: train/calibrate/evaluate completed (SYNTHETIC only)", flush=True)
    deployment = {}
    for method in ("baseline_c", "baseline_c_plus", "proposed_v1"):
        experiment = runs[method]
        deployment[method] = {
            "export": run("export_onnx", "--experiment", experiment),
            "cpu": run("benchmark", "--experiment", experiment, "--device", "cpu"),
            "onnx_cpu": run("benchmark_onnx", "--experiment", experiment),
        }
        import torch

        if torch.cuda.is_available():
            deployment[method]["cuda"] = run(
                "benchmark", "--experiment", experiment, "--device", "cuda"
            )
    summary = {
        "evidence_kind": "SYNTHETIC",
        "scientific_gates_passed": False,
        "validation": validated,
        "repeatability": repeat,
        "runs": runs,
        "results": results,
        "deployment": deployment,
        "command_log": str(output / "commands.json"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ROOT / "artifacts" / "latest_smoke.json").write_text(
        json.dumps({"summary": str(output / "summary.json")}, indent=2)
    )
    print(json.dumps({"summary": str(output / "summary.json"), "all_commands_exit_zero": True}))


if __name__ == "__main__":
    main()
