"""Download only explicitly approved original Cube++ artifacts, with publisher MD5."""

import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/public/cube"
PROVENANCE = ROOT / "docs/data/provenance/cube"
FILES = {
    "SimpleCube++.zip": (2113441199, "d3438223e5b4ca27be874db690bef822"),
    "markup.zip": (19957846, "8c3c8344cfd16f131d0d632569d799e9"),
}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    sources = {
        "README.md": "https://raw.githubusercontent.com/Visillect/CubePlusPlus/master/README.md",
        "description.md": "https://raw.githubusercontent.com/Visillect/CubePlusPlus/master/description/description.md",
        "zenodo_record.json": "https://zenodo.org/api/records/4153431",
    }
    for name, url in sources.items():
        (PROVENANCE / name).write_bytes(urllib.request.urlopen(url, timeout=60).read())
    for name, (size, expected) in FILES.items():
        dest = DATA / name
        url = "https://zenodo.org/records/4153431/files/" + urllib.parse.quote(name) + "?download=1"
        if not dest.exists():
            partial = dest.with_suffix(".partial")
            start = time.monotonic()
            total = 0
            last = start
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
                while chunk := response.read(4 * 1024 * 1024):
                    total += len(chunk)
                    if total > size:
                        raise ValueError("Download exceeds approved size")
                    out.write(chunk)
                    if time.monotonic() - last > 20:
                        print(f"{name}: {total / size:.1%}, {total / 1e6:.0f} MB", flush=True)
                        last = time.monotonic()
            if total != size:
                raise ValueError("Truncated download")
            partial.replace(dest)
        md5, sha = hashlib.md5(), hashlib.sha256()
        with dest.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                md5.update(chunk)
                sha.update(chunk)
        if dest.stat().st_size != size or md5.hexdigest() != expected:
            raise ValueError(f"Publisher checksum mismatch: {name}")
        record = {
            "file": name,
            "bytes": size,
            "md5": md5.hexdigest(),
            "sha256": sha.hexdigest(),
            "url": url,
            "license": "CC-BY-4.0",
            "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (PROVENANCE / (name + ".json")).write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
