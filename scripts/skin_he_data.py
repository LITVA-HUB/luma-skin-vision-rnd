"""Read the untouched author workbook into role-separated numerical intermediates."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/skin_he_xyz_v1"
BENCH = ROOT / "docs/benchmarks/skin_he_xyz_v1"
WORKBOOK_SHA = "e3ad5b30a828c542ba2d23dd7fe57cddf0c3de2a163f1815cd2e482a816484a8"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(role, lock_digest=None):
    if role not in ("train", "test"):
        raise ValueError("Explicit role required")
    if role == "test":
        lock = BENCH / "model_lock.json"
        if not lock_digest or digest(lock) != lock_digest:
            raise ValueError("Locked source models required before test decoding")
        for filename, expected in json.loads(lock.read_text())["files"].items():
            if digest(ROOT / filename) != expected:
                raise ValueError("Frozen source/model changed")
    destination = DATA / (role + ".json")
    if destination.exists():
        raise ValueError("Immutable extracted role exists")
    path = next((ROOT / "data/public/he_skin_2021").glob("*.xlsx"))
    if digest(path) != WORKBOOK_SHA:
        raise ValueError("Original workbook changed")
    book = openpyxl.load_workbook(path, read_only=True, data_only=False)
    sheet = book["FSCD" if role == "train" else "Testing"]
    count, offset = (200, 0) if role == "train" else (100, 40)
    expected_header = ["X", "Y", "Z", "Raw_R", "Raw_G", "Raw_B", "R", "G", "B"]
    if [c.value for c in sheet["G2:O2"][0]] != expected_header:
        raise ValueError("Column meaning changed")
    records = []
    for row in sheet.iter_rows(min_row=3, max_row=count+2, min_col=3, max_col=15):
        subject, site = row[0].value, row[1].value
        expected_subject = f"Obs.{offset+len(records)//5+1}"
        if subject != expected_subject or site != ("FH", "CBR", "CBL", "NT", "CH")[len(records) % 5]:
            raise ValueError("Author subject/site order changed")
        values = []
        for cell in row[4:]:
            if cell.data_type == "f" or not isinstance(cell.value, (int, float)) or not math.isfinite(cell.value):
                raise ValueError("Missing, formula or invalid measured value")
            values.append(float(cell.value))
        if any(v <= 0 for v in values[:6]) or any(v < 0 or v > 255 for v in values[6:]):
            raise ValueError("Nonpositive XYZ/RAW or out-of-range JPG")
        records.append({"id": subject+"/"+site, "subject": subject, "site": site,
                        "xyz": values[:3], "raw": values[3:6], "jpg": [v/255 for v in values[6:]],
                        "source_row": row[0].row})
    if len({r["id"] for r in records}) != count:
        raise ValueError("Duplicate reference identity")
    DATA.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"role": role, "workbook_sha256": WORKBOOK_SHA,
                                      "sheet": sheet.title, "rows": records,
                                      "model_lock_sha256": lock_digest}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"role": role, "sites": count, "people": count//5, "sha256": digest(destination)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("train", "test"))
    parser.add_argument("--lock-digest")
    args = parser.parse_args()
    prepare(args.role, args.lock_digest)
