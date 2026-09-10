"""Retrieve/verify the original Sony30 regression pilot from its committed pinned manifest."""

import hashlib
import json
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "docs/data/provenance/intel_tau/pilot_manifest.json").read_text(encoding="utf-8")
    )
    rows = manifest["files"]
    if len(rows) != 60 or sum(r["bytes"] for r in rows) > 25_000_000:
        raise ValueError("Expected original bounded60-file pilot")
    out = root / "data/public/intel_tau_c5_sony_pilot"
    out.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for row in rows:
        name, url = row["file"], row["url"]
        parsed = urlparse(url)
        if (
            Path(name).name != name
            or "/" in name
            or "\\" in name
            or parsed.scheme != "https"
            or parsed.netloc != "raw.githubusercontent.com"
            or f"/mahmoudnafifi/C5/{manifest['source_commit']}/images/" not in parsed.path
        ):
            raise ValueError("Unexpected pinned pilot location")
        target = out / name
        existed = target.exists()
        if existed:
            content = target.read_bytes()
        else:
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read(row["bytes"] + 1)
        if len(content) != row["bytes"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("Pilot size/hash mismatch: " + name)
        if not existed:
            with target.open("xb") as stream:
                stream.write(content)
            downloaded += len(content)
    print(
        json.dumps(
            {
                "verified_files": len(rows),
                "downloaded_bytes": downloaded,
                "upstream_license": "CC-BY-SA-4.0",
                "use": "evaluation-only; original manifest unchanged",
            }
        )
    )


if __name__ == "__main__":
    main()
