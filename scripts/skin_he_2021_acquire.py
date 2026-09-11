"""Acquire the original CC BY 4.0 paired skin RGB/XYZ workbook, without editing it."""
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "docs/data/provenance/skin_public_2026_09_11"
record_path = PROVENANCE / "he_2021_zenodo_record.json"
record = json.loads(record_path.read_bytes())
if record["metadata"]["license"]["id"] != "cc-by-4.0" or len(record["files"]) != 1:
    raise ValueError("Original license or inventory differs")
item = record["files"][0]
if item["size"] != 395693 or item["checksum"] != "md5:0217b7d83c2a3883605ba722a8025c8a":
    raise ValueError("Original file changed")
out = ROOT / "data/public/he_skin_2021"
out.mkdir(parents=True, exist_ok=False)
url = item["links"]["self"]
with urllib.request.urlopen(url, timeout=30) as response:
    data = response.read(item["size"] + 1)
if len(data) != item["size"] or hashlib.md5(data).hexdigest() != item["checksum"].split(":")[1]:
    raise ValueError("Publisher byte count/checksum mismatch")
destination = out / item["key"]
destination.write_bytes(data)
receipt = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "source": "https://zenodo.org/records/5532176", "url": url,
           "license": "CC BY 4.0 (original author deposit)",
           "record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
           "file": str(destination.relative_to(ROOT)), "bytes": len(data),
           "sha256": hashlib.sha256(data).hexdigest(), "publisher_md5": item["checksum"],
           "scope": "Tabulated measured skin XYZ and corresponding image-derived RGB. No full images in deposit. No accuracy result yet."}
(PROVENANCE / "he_2021_acquisition.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
print(json.dumps(receipt))
