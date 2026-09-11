"""Original MSKCC joins and patient roles. No endpoint values in manifest."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/public/mskcc_skin_v1"
PROV = ROOT / "docs/data/provenance/mskcc_skin_v1"
PROTOCOL = ROOT / "docs/research/skin_mskcc_protocol_v1.md"
MANIFEST = ROOT / "data/processed/skin_mskcc_v1/manifest.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    with (RAW / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def patient_roles(devices):
    roles = {}
    for device, ntest, expected in [("SLR", 4, 18), ("ipod", 6, 28)]:
        people = sorted((p for p, d in devices.items() if d == device),
                        key=lambda p: hashlib.sha256(("LumaMSKCCv1|" + p).encode()).hexdigest())
        if len(people) != expected:
            raise ValueError("Unexpected instrument-paired cohort")
        for i, p in enumerate(people):
            roles[p] = ("test" if i < ntest else "validation" if i < ntest + 3
                        else "calibration" if i < ntest + 6 else "train")
    if set(roles) != set(devices):
        raise ValueError("Unrecognized camera or unassigned person")
    return roles


def prepare():
    meta = {r["isic_id"]: r for r in read("mskcc-skin-tone-labeling-dataset.csv")}
    sites = {r["tag_id"]: r for r in read("s2.csv")}
    people = {r["patient_id"]: r for r in read("s1.csv")}
    rows, excluded = [], []
    for r in read("s7.csv"):
        if r["isic_id"] == "not-available":
            excluded.append("not-available")
            continue
        m, site = meta[r["isic_id"]], sites[r["tag_id"]]
        p = site["patient_id"]
        if m["patient_id"] != p or m["lesion_id"] != site["lesion_id"]:
            raise ValueError("Conflicting image/site/patient join")
        if m["copyright_license"] != "CC-BY" or r["type"] != "normal-skin":
            raise ValueError("Wrong license or tissue cohort")
        rows.append({"image": r["isic_id"], "patient": p, "site": r["tag_id"],
                     "device": people[p]["dermatoscope"], "mode": r["dermoscopic_type"],
                     "image_type": m["image_type"], "anatomic_site": r["anatomic_site"]})
    if len(rows) != 1838 or len({r["image"] for r in rows}) != 1838 or len(excluded) != 3:
        raise ValueError("Release inventory changed")
    roles = patient_roles({r["patient"]: r["device"] for r in rows})
    for r in rows:
        r["role"] = roles[r["patient"]]
    files = ["mskcc-skin-tone-labeling-dataset.csv", "s1.csv", "s2.csv", "s4.csv", "s7.csv"]
    out = {"protocol_sha256": sha(PROTOCOL), "source_sha256": {n: sha(RAW / n) for n in files},
           "rows": sorted(rows, key=lambda r: r["image"])}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(out, indent=2) + "\n"
    if MANIFEST.exists() and MANIFEST.read_text(encoding="utf8") != payload:
        raise ValueError("Refusing to overwrite different manifest")
    MANIFEST.write_text(payload, encoding="utf8")
    summary = {"manifest_sha256": sha(MANIFEST), "protocol_sha256": sha(PROTOCOL),
               "source_sha256": out["source_sha256"], "excluded_missing_images": len(excluded),
               "roles": {role: {"people": len({r['patient'] for r in rows if r['role'] == role}),
                                  "sites": len({r['site'] for r in rows if r['role'] == role}),
                                  "images": sum(r['role'] == role for r in rows),
                                  "devices": dict(Counter(r['device'] for r in rows if r['role'] == role))}
                         for role in ["train", "validation", "calibration", "test"]},
               "numeric_endpoints_decoded": False}
    (PROV / "split_receipt.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    print(json.dumps(summary, indent=2))


def manifest():
    data = json.loads(MANIFEST.read_bytes())
    if data["protocol_sha256"] != sha(PROTOCOL):
        raise ValueError("Protocol hash changed")
    for n, digest in data["source_sha256"].items():
        if sha(RAW / n) != digest:
            raise ValueError("Source file changed")
    return data


def load_source(role):
    if role not in {"train", "validation"}:
        raise ValueError("This pilot loader cannot open calibration or test endpoints")
    import numpy as np
    rows = [r for r in manifest()["rows"] if r["role"] == role]
    allowed = {r["image"] for r in rows}
    values = {r["isic_id"]: r for r in read("s7.csv") if r["isic_id"] in allowed}
    x, y, repetitions = [], [], []
    for row in rows:
        r = values[row["image"]]
        reps = np.array([[float(r[f"{c}_{i}"]) for c in "lab"] for i in (1, 2, 3)])
        target = np.array([float(r["average_" + c]) for c in "lab"])
        if not np.isfinite(reps).all() or not np.allclose(reps.mean(0), target, atol=0.051, rtol=0):
            raise ValueError("Invalid or inconsistent instrument repetitions")
        x.append([float(r["img_" + c]) for c in "lab"])
        y.append(reps.mean(0))
        repetitions.append(reps)
    x, y = np.array(x), np.array(y)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid color values")
    return rows, x, y, np.array(repetitions)


if __name__ == "__main__":
    prepare()
