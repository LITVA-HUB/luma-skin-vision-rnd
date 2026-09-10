import json
from pathlib import Path

from luma_skin_vision.data import Record

root = Path(__file__).resolve().parents[1]
destination = root / "data" / "schemas" / "record-v1.schema.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(Record.model_json_schema(), indent=2), encoding="utf-8")
print(destination)
