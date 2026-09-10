import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Read transparent local experiment registry")
parser.add_argument("--root", default="experiments/runs")
args = parser.parse_args()
rows = []
for path in sorted(Path(args.root).glob("*/run.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        {
            key: record.get(key)
            for key in (
                "experiment_id",
                "timestamp",
                "status",
                "data_kind",
                "git_commit",
                "source_hash",
                "dataset_hash",
                "parameters",
                "training_duration_seconds",
            )
        }
    )
print(json.dumps(rows, indent=2))
