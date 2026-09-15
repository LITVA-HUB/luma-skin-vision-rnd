"""Restore the original 19 TRAIN cubes without depending on P1/P2 model artifacts.

The historical source lock, not a new random split, defines the allowed files.
No held-out cube is downloaded or decoded. Source arrays stay in ignored data/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE_LOCK = ROOT / "docs/archive/2026-09-14/evidence/chromaseed_palette_pretrain_v1/source_lock.json"
EXPECTED_SOURCE_LOCK_SHA = "db5e5c8d9eef8f7a066a0fac8296956507ceb59522b8ffb95b7f86ab8b5b2cf7"
ORIGINAL_MANIFEST_SHA = "3d3b7dc7256eb04250dee2f5ef76880f4101d019313d97ba051480f8d320b5c7"
API_URL = "https://api.figshare.com/v2/articles/25598670"
RAW = ROOT / "data/public/uminho_access"
REPORT = ROOT / "docs/data/access_2026_09_15/uminho"


def digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def save_once(path: Path, value: dict) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Refusing to replace existing artifact: {path}")
    else:
        path.write_text(text, encoding="utf-8")


def original_train() -> list[dict]:
    if digest(SOURCE_LOCK) != EXPECTED_SOURCE_LOCK_SHA:
        raise ValueError("Historical source-lock bytes differ")
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    records = []
    for number, row in enumerate(lock["sources"], 1):
        name = PureWindowsPath(row["path"]).name
        if row["role"] != "train" or not name.endswith("_reflectance.mat"):
            raise ValueError("Unexpected original source")
        if lock["bindings"][row["path"]] != row["sha256"]:
            raise ValueError("Source hash disagrees with historical binding")
        records.append({"source_alias": f"U{number:02d}", "name": name,
                        "bytes": row["bytes"], "sha256": row["sha256"], "role": "train"})
    if len(records) != 19 or len({r["name"] for r in records}) != 19:
        raise ValueError("Exactly 19 original TRAIN faces required")
    return records


def inventory() -> dict:
    rows = original_train()
    manifest = {"status": "HISTORICAL_TRAIN_MEMBERSHIP_RECOVERED",
                "source_lock_sha256": EXPECTED_SOURCE_LOCK_SHA,
                "original_29_face_manifest_sha256": ORIGINAL_MANIFEST_SHA,
                "all_29_face_manifest_bytes_available": False,
                "allocation": "The exact 19 historical TRAIN members; no new split",
                "held_out_policy": "Other 10 original faces remain unacquired and unopened",
                "rows": rows}
    save_once(RAW / "train_manifest.private.json", manifest)
    files = []
    for row in rows:
        path = RAW / row["name"]
        present = path.is_file()
        verified = present and path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]
        if present and not verified:
            raise ValueError("Existing original cube does not match its historical SHA256")
        files.append({k: row[k] for k in ("source_alias", "bytes", "sha256", "role")} |
                     {"present": present, "verified": verified})
    return {"dataset": "UMINHO-HSFD", "source_lock_sha256": EXPECTED_SOURCE_LOCK_SHA,
            "private_train_manifest_sha256": digest(RAW / "train_manifest.private.json"),
            "historical_train_cubes": 19, "historical_train_bytes": sum(r["bytes"] for r in rows),
            "current_verified_cubes": sum(r["verified"] for r in files),
            "current_people_represented": sum(r["verified"] for r in files),
            "independent_camera_rgb_images": 0, "held_out_cubes_opened": 0,
            "exact_old_P1_requires_missing_Seg1_and_annotations": True,
            "new_preparation_requires_P2_checkpoint": False, "files": files}


def download() -> dict:
    rows = original_train()
    RAW.mkdir(parents=True, exist_ok=True)
    metadata_path = RAW / "official_article_metadata.json"
    if not metadata_path.exists():
        try:
            with urllib.request.urlopen(API_URL, timeout=30) as response:
                payload = response.read(2_000_001)
            if len(payload) > 2_000_000:
                raise ValueError("Unexpected metadata size")
            metadata = json.loads(payload)
            metadata_path.write_bytes(payload)
        except urllib.error.HTTPError as error:
            body = error.read(100_000)
            return {"status": "BLOCKED_SOURCE_HTTP", "checked_utc": datetime.now(timezone.utc).isoformat(),
                    "url": API_URL, "http_status": error.code,
                    "response_server": error.headers.get("Server"),
                    "response_content_type": error.headers.get("Content-Type"),
                    "response_bytes": len(body), "response_sha256": hashlib.sha256(body).hexdigest(),
                    "login_or_agreement_page_observed": False,
                    "scope": "This environment request failed; not proof of global unavailability",
                    "known_official_file_urls_in_preserved_metadata": 0,
                    "downloaded_cubes_this_run": 0}
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    license_ = metadata["license"]
    if license_["name"] != "CC BY 4.0" or license_["url"] != "https://creativecommons.org/licenses/by/4.0/":
        raise ValueError("Official terms differ from historical terms; do not accept new terms automatically")
    if metadata.get("download_disabled") or metadata.get("is_embargoed"):
        raise ValueError("Author download is disabled or embargoed")
    lookup = {row["name"]: row for row in metadata["files"]}
    completed = 0
    for row in rows:
        current = lookup[row["name"]]
        if current["size"] != row["bytes"]:
            raise ValueError("Current source length differs from historical source")
        path = RAW / row["name"]
        if not path.exists():
            url = current["download_url"]
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.hostname not in {"ndownloader.figshare.com", "api.figshare.com"}:
                raise ValueError("Unexpected official download host; inspect before proceeding")
            partial = path.with_suffix(".mat.part")
            if partial.exists():
                raise ValueError("Partial file preserved; inspect it before retrying")
            total = 0
            with urllib.request.urlopen(url, timeout=60) as source, partial.open("xb") as sink:
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > row["bytes"]:
                        raise ValueError("Download larger than historical source")
                    sink.write(chunk)
            if total != row["bytes"] or digest(partial) != row["sha256"] or digest(partial, "md5") != current["computed_md5"]:
                raise ValueError("Source failed original size/MD5 or historical SHA256 verification")
            partial.rename(path)
            completed += 1
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Existing source failed historical checksum")
        print(f"Verified {row['source_alias']} / 19", flush=True)
    return {"status": "RESTORED_ORIGINAL_TRAIN", "downloaded_cubes_this_run": completed,
            "metadata_sha256": digest(metadata_path), "license": license_, "url": API_URL}


def color_contract() -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    from skin_spectral_palette import original_cie, integration_matrix, xyz_lab, CIE_FILES
    wave, cmf2, cmf10, spd = original_cie()  # Also verifies original CIE checksums.
    knots = np.arange(400., 721., 10.)
    arrays = {"wavelength_nm": knots, "integration_wavelength_nm": wave, "d65_spd": spd}
    summary = {}
    for suffix, cmf in (("2deg", cmf2), ("10deg", cmf10)):
        matrix = integration_matrix(knots, wave, cmf, spd)
        white = matrix.sum(0)
        reference = xyz_lab(np.ones((1, 33)) @ matrix, white)[0]
        if not np.allclose(reference, [100, 0, 0], rtol=0, atol=1e-10):
            raise AssertionError("Perfect reflector check failed")
        arrays["matrix_" + suffix], arrays["white_xyz_y1_" + suffix] = matrix, white
        summary[suffix] = {"white_xyz_y1": white.tolist(), "perfect_reflector_lab": reference.tolist()}
    REPORT.mkdir(parents=True, exist_ok=True)
    matrix_path = REPORT / "canonical_integration_matrices.npz"
    if matrix_path.exists():
        with np.load(matrix_path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], arrays[k]) for k in arrays):
                raise ValueError("Stored colorimetric matrices differ")
    else:
        np.savez_compressed(matrix_path, **arrays)
    return {"source": "Measured directional reflectance datao; dimensionless, no clipping",
            "expected_cube": "MATLAB float64 H x W x 33; background all-zero",
            "measured_wavelength_nm": list(range(400, 721, 10)),
            "derived_target_conventions": ["D65/CIE1931 2deg", "D65/CIE1964 10deg"],
            "integration": "1nm sum 360..830nm, linear interpolation between measured bands, constant endpoint extrapolation",
            "xyz_units": "Relative XYZ with perfect-reflector Y=1; multiply all XYZ by100 for Y=100",
            "source_rgb_convention": "Author-rendered D65/CIE2006 10deg; distinct from our two explicit conventions",
            "uncertainty": "400..720nm truncation and endpoint extrapolation; finite filter bandwidth; no physical accuracy inferred from numerical checks",
            "foreground_is_not_verified_skin": True, "derived_rgb_is_independent_camera_photo": False,
            "color_checks": summary, "matrix_sha256": digest(matrix_path),
            "source_table_hashes": {p.name: digest(p) for pair in CIE_FILES for p in pair},
            "versions": {"python": platform.python_version(), "numpy": np.__version__}}


def audit() -> dict:
    """Decode existing verified original cubes; never accept a missing file or fake cube."""
    from scipy.io import loadmat
    records = []
    for row in original_train():
        path = RAW / row["name"]
        if not path.is_file():
            continue
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Cube does not match exact historical source")
        cube = loadmat(path, variable_names=["datao"], verify_compressed_data_integrity=True)["datao"]
        if cube.ndim != 3 or cube.shape[2] != 33 or cube.dtype != np.float64 or not np.isfinite(cube).all():
            raise ValueError("Unexpected source cube contract")
        foreground = np.any(cube != 0, axis=-1)
        values = cube[foreground]
        records.append({"source_alias": row["source_alias"], "sha256": row["sha256"],
                        "shape": list(cube.shape), "foreground_spectra": int(foreground.sum()),
                        "negative_channels": int((values < 0).sum()), "above_one_channels": int((values > 1).sum()),
                        "minimum": float(values.min()), "maximum": float(values.max()),
                        "verified_skin_mask": False})
        del cube, foreground, values
    return {"status": "ORIGINAL_CUBES_DECODED" if records else "NO_SOURCE_CUBES",
            "actual_decoded_cubes": len(records), "actual_foreground_spectra": sum(r["foreground_spectra"] for r in records),
            "held_cubes_decoded": 0, "independent_camera_rgb_images": 0, "files": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "download", "audit", "contract"])
    parser.add_argument("--receipt", type=Path, help="New receipt path when repeating a later access attempt; old reports are immutable")
    args = parser.parse_args()
    if args.action == "inventory":
        result = inventory()
    elif args.action == "download":
        inventory()
        result = download()
    elif args.action == "audit":
        result = audit()
    else:
        result = color_contract()
    save_once(args.receipt or REPORT / f"{args.action}.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
