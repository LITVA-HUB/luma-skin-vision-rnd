"""Author-redistributed Intel-TAU pilot; upstream CC BY-SA 4.0, evaluation only here."""

import hashlib
import json
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "data/public/intel_tau_c5_sony_pilot"
out.mkdir(parents=True, exist_ok=True)
prov = root / "docs/data/provenance/intel_tau"
prov.mkdir(parents=True, exist_ok=True)
api = "https://api.github.com/repos/mahmoudnafifi/C5"
commit = json.load(urllib.request.urlopen(api + "/commits/main"))["sha"]
files = json.load(urllib.request.urlopen(api + "/contents/images?ref=" + commit))
files = [f for f in files if f["name"].endswith((".png", "_metadata.json"))]
if sum(f["size"] for f in files) > 25_000_000:
    raise ValueError("Pilot exceeds approved bound")
records = []
for f in files:
    content = urllib.request.urlopen(f["download_url"], timeout=30).read()
    if len(content) != f["size"]:
        raise ValueError("Size mismatch")
    (out / f["name"]).write_bytes(content)
    records.append(
        {
            "file": f["name"],
            "url": f["download_url"],
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    )
record = {
    "source_commit": commit,
    "upstream_license": "CC-BY-SA-4.0",
    "original_license_record": "https://research.fi/en/results/dataset/f0570a3f-3d77-4f44-9ef1-99ab4878f17c",
    "usage": "Evaluation only. No training, tuning, calibration or adaptation on these images.",
    "limitation": "30 author-selected Sony examples; not the complete INTEL-TAU benchmark.",
    "files": records,
}
(prov / "pilot_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
(prov / "C5_README.md").write_bytes(
    urllib.request.urlopen(
        f"https://raw.githubusercontent.com/mahmoudnafifi/C5/{commit}/README.md"
    ).read()
)
print(f"Downloaded {len(records)} files, {sum(r['bytes'] for r in records)} bytes, commit {commit}")
