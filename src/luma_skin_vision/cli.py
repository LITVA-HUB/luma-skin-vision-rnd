import argparse
import json
from pathlib import Path

import yaml

from luma_skin_vision.data import Record, assign_splits, validate_records
from luma_skin_vision.experiment import write_json


def main(command):
    parser = argparse.ArgumentParser(description=f"Luma R&D: {command}; synthetic != real evidence")
    if command == "inspect_environment":
        parser.add_argument("--output", default="docs/research/environment.json")
    elif command == "generate_synthetic_demo":
        parser.add_argument("--output", default="data/synthetic_demo")
        parser.add_argument("--subjects", type=int, default=20)
        parser.add_argument("--seed", type=int, default=42)
    elif command in ("validate_dataset", "prepare_dataset", "repeatability"):
        parser.add_argument("--manifest", required=True)
        if command == "prepare_dataset":
            parser.add_argument("--output", required=True)
            parser.add_argument("--seed", type=int, default=42)
    elif command == "train":
        parser.add_argument("--config", required=True)
        parser.add_argument("--output-root", default="experiments/runs")
    else:
        parser.add_argument("--experiment", required=True)
        if command == "evaluate":
            parser.add_argument("--split", choices=["validation", "test"], default="test")
        if command == "calibrate":
            parser.add_argument("--alpha", type=float, default=0.1)
            parser.add_argument("--tolerance", type=float, default=5)
        if command in ("benchmark", "benchmark_onnx"):
            parser.add_argument("--iterations", type=int, default=50)
            parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if command == "inspect_environment":
        from luma_skin_vision.environment import inspect_environment

        result = inspect_environment()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    elif command == "generate_synthetic_demo":
        from luma_skin_vision.synthetic import generate

        result = {
            "manifest": str(generate(args.output, args.subjects, args.seed)),
            "data_kind": "SYNTHETIC",
        }
    elif command == "validate_dataset":
        rows = validate_records(args.manifest)
        result = {
            "valid": True,
            "records": len(rows),
            "images": len({r.image_id for r in rows}),
            "subjects": len({r.subject_id for r in rows}),
            "data_kind": rows[0].data_kind,
            "splits": {
                s: len({r.subject_id for r in rows if r.split == s})
                for s in ("train", "validation", "calibration", "test")
            },
        }
    elif command == "prepare_dataset":
        # Import JSONL unassigned capture rows; reject existing split labels to prevent accidental reshuffle.
        source, destination = Path(args.manifest), Path(args.output)
        if destination.exists():
            raise ValueError("refusing to overwrite an existing split manifest")
        raw = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any("split" in row for row in raw):
            raise ValueError(
                "Input already contains split labels; locked splits must not be regenerated"
            )
        assignment = assign_splits([r["subject_id"] for r in raw], args.seed)
        if destination.parent.resolve() != source.parent.resolve():
            raise ValueError("output must share dataset root with source manifest")
        rows = [
            Record.model_validate({**row, "split": assignment[row["subject_id"]]}) for row in raw
        ]
        destination.write_text(
            "\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8"
        )
        validate_records(destination)
        result = {"manifest": str(destination), "subjects": len(assignment)}
    elif command == "repeatability":
        from luma_skin_vision.evaluation import repeatability

        rows = validate_records(args.manifest)
        refs = {r.reference_measurement_id: r.reference_repeats_lab for r in rows}
        result = {
            **repeatability(list(refs.values())),
            "data_kind": rows[0].data_kind,
            "gate_1_passed": False,
        }
    elif command == "train":
        from luma_skin_vision.training import train

        config = yaml.safe_load(Path(args.config).read_text())
        result = {"experiment": str(train(config, args.output_root))}
    elif command == "evaluate":
        from luma_skin_vision.training import evaluate_run

        report = evaluate_run(args.experiment, args.split)
        result = {
            k: report[k] for k in ("data_kind", "method", "split", "metrics", "risk_coverage")
        }
    elif command == "calibrate":
        from dataclasses import asdict

        from luma_skin_vision.training import calibrate_run

        result = asdict(calibrate_run(args.experiment, args.alpha, args.tolerance))
    elif command == "export_onnx":
        from luma_skin_vision.export import export_run

        result = export_run(args.experiment)
    elif command in ("benchmark", "benchmark_onnx"):
        from luma_skin_vision.export import benchmark_run

        result = benchmark_run(
            args.experiment,
            iterations=args.iterations,
            device=args.device,
            onnx=command == "benchmark_onnx",
        )
    else:
        raise ValueError(f"unknown command {command}")
    print(json.dumps(result, indent=2, allow_nan=False))
