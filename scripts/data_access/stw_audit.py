#!/usr/bin/env python3
"""Audit an original STW release locally; never infer Lab from MST.

Participant names, paths and image hashes are retained only in --private-root.
No source processing code is executed. No model is trained. The ZIP remains intact.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from PIL import Image


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def person(row: dict) -> tuple[str, str]:
    # Camera must not be added to identity. Underlying cross-dataset identity
    # linkage remains unknown and is explicitly not claimed here.
    return row["dataset"], row["tokens"]


def count_rows(rows: list[dict]) -> dict:
    counts = collections.Counter(person(row) for row in rows)
    return {"rows": len(rows), "source_person_keys": len(counts),
            "people_with_repeats": sum(n > 1 for n in counts.values()),
            "min_images_per_person": min(counts.values(), default=0),
            "max_images_per_person": max(counts.values(), default=0)}


def audit(root: Path, public: Path) -> dict:
    metadata = root / "all_annotated_data.csv"
    archive = root / "images.zip"
    rows = load_rows(metadata)
    lookup = collections.defaultdict(list)
    for idx, row in enumerate(rows):
        stem = PurePosixPath(row["paths"].replace("\\", "/")).stem
        lookup[("all_cropped_images_by_class", row["new_tokens"] + stem)].append(idx)
        lookup[("all_cropped_skin_only_images", row["new_tokens"] + "_" + stem)].append(idx)
    record_rows = []
    failures = []
    variant_counts = collections.Counter()
    sizes = collections.Counter()
    modes = collections.Counter()
    formats = collections.Counter()
    matched = collections.defaultdict(list)
    digest_rows = collections.defaultdict(list)
    total_bytes = 0
    with zipfile.ZipFile(archive) as z:
        members = [entry for entry in z.infolist() if not entry.is_dir()]
        for i, entry in enumerate(members):
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or len(path.parts) != 4:
                raise ValueError("Unsafe or unexpected archive member structure")
            variant = path.parts[1]
            variant_counts[variant] += 1
            blob = z.read(entry)
            total_bytes += len(blob)
            key = re.sub(r"_face_\d+$", "", path.stem)
            candidates = lookup.get((variant, key), [])
            if len(candidates) != 1:
                failures.append({"member": entry.filename,
                                 "reason": "ambiguous_or_unmatched_metadata",
                                 "candidate_count": len(candidates)})
                continue
            idx = candidates[0]
            row = rows[idx]
            if int(row["class"]) != int(path.parent.name):
                failures.append({"member": entry.filename, "reason": "MST_folder_conflict"})
                continue
            try:
                with Image.open(io.BytesIO(blob)) as image:
                    image.load()  # Decode actual pixels, not only a JPEG header.
                    sizes[str(image.size)] += 1
                    modes[image.mode] += 1
                    formats[image.format] += 1
                    width, height = image.size
                    mode, fmt = image.mode, image.format
            except Exception as exc:
                failures.append({"member": entry.filename,
                                 "reason": "decode_failed", "error": str(exc)})
                continue
            digest = hashlib.sha256(blob).hexdigest()
            digest_rows[digest].append((variant, idx))
            matched[variant].append(idx)
            record_rows.append({"member": entry.filename, "csv_row": idx,
                                "dataset": row["dataset"], "tokens": row["tokens"],
                                "new_tokens": row["new_tokens"], "source_path": row["paths"],
                                "MST": int(row["class"]), "variant": variant,
                                "width": width, "height": height, "mode": mode,
                                "format": fmt, "sha256": digest})
            if (i + 1) % 5000 == 0:
                print(f"decoded {i + 1}/{len(members)}", flush=True)
    full = "all_cropped_images_by_class"
    skin = "all_cropped_skin_only_images"
    full_indices = set(matched[full])
    skin_indices = set(matched[skin])
    full_rows = [rows[idx] for idx in sorted(full_indices)]
    unique_people = sorted({person(row) for row in rows})
    aliases = {key: f"STW_P{i + 1:04d}" for i, key in enumerate(unique_people)}
    write_json(root / "image_pairing.private.json", record_rows)
    write_json(root / "pairing_failures.private.json", failures)
    write_json(root / "person_mapping.private.json",
               [{"source": k[0], "source_person": k[1], "pseudonym": v}
                for k, v in aliases.items()])
    annotations_per_person = collections.defaultdict(set)
    for row in rows:
        annotations_per_person[person(row)].add(row["class"])
    summary = {
        "dataset": "STW", "checked_at_utc_date": "2026-09-15",
        "status": "PARTIAL_RELEASE_DOWNLOADED_AND_PAIRED",
        "target": "Human apparent-tone MST integer 1..10; per-person annotation",
        "not_targets": ["instrument Lab", "reflectance", "pseudo-Lab"],
        "metadata": {**count_rows(rows), "unique_paths": len({r["paths"] for r in rows}),
                     "unique_new_tokens": len({r["new_tokens"] for r in rows}),
                     "MST_conflicting_source_person_keys": sum(len(v) > 1 for v in annotations_per_person.values())},
        "actual_download": {"archive_bytes": archive.stat().st_size,
                            "archive_sha256": sha256(archive),
                            "metadata_bytes": metadata.stat().st_size,
                            "metadata_sha256": sha256(metadata),
                            "jpeg_files": sum(variant_counts.values()),
                            "uncompressed_bytes": total_bytes,
                            "decoded_and_uniquely_paired_files": len(record_rows),
                            "decode_or_pairing_failures": len(failures),
                            "failure_reasons": dict(collections.Counter(x["reason"] for x in failures))},
        "full_face": count_rows(full_rows),
        "skin_only": count_rows([rows[idx] for idx in sorted(skin_indices)]),
        "full_face_without_skin_only": len(full_indices - skin_indices),
        "skin_only_without_full_face": len(skin_indices - full_indices),
        "sources_metadata": {d: count_rows([r for r in rows if r["dataset"] == d])
                             for d in sorted({r["dataset"] for r in rows})},
        "sources_paired_full_face": {d: count_rows([r for r in full_rows if r["dataset"] == d])
                                     for d in sorted({r["dataset"] for r in full_rows})},
        "MST_distribution_full_face": dict(sorted(collections.Counter(r["class"] for r in full_rows).items(), key=lambda x: int(x[0]))),
        "image_dimensions": dict(sizes), "image_modes": dict(modes),
        "image_formats": dict(formats),
        "identical_byte_groups": sum(len(v) > 1 for v in digest_rows.values()),
        "identical_byte_groups_across_different_source_images": sum(len({idx for _, idx in v}) > 1 for v in digest_rows.values()),
        "image_camera_exposure_illuminant_session_metadata": "NOT_RELEASED in supplied six-column annotation CSV",
        "physical_identity_linkage_across_source_datasets": "UNKNOWN; source-person key is an annotation grouping, not proven global identity",
        "author_split_audit": {},
    }
    split_root = root / "splits" / "individuals_train_test_splits"
    split_sets = {}
    split_paired_sets = {}
    split_rows = {}
    main_by_path = {r["paths"]: r for r in rows}
    paired_source_paths = {r["paths"] for r in full_rows}
    for path in sorted(split_root.glob("*.csv")):
        data = load_rows(path)
        split_rows[path.stem] = data
        paired_rows = [r for r in data if r["paths"] in paired_source_paths]
        split_sets[path.stem] = {person(r) for r in data}
        split_paired_sets[path.stem] = {person(r) for r in paired_rows}
        summary["author_split_audit"][path.name] = {
            "sha256": sha256(path), "metadata": count_rows(data),
            "paired_full_face": count_rows(paired_rows),
            "rows_absent_from_main_csv": sum(r["paths"] not in main_by_path for r in data),
            "MST_conflicts_vs_main_csv": sum(r["paths"] in main_by_path and r["class"] != main_by_path[r["paths"]]["class"] for r in data),
            "source_person_conflicts_vs_main_csv": sum(r["paths"] in main_by_path and person(r) != person(main_by_path[r["paths"]]) for r in data)}
    if "train" in split_rows and "test" in split_rows:
        combined = split_rows["train"] + split_rows["test"]
        summary["author_split_audit"]["train_test_union"] = {
            **count_rows(combined), "unique_source_paths": len({r["paths"] for r in combined}),
            "source_rows": dict(collections.Counter(r["dataset"] for r in combined)),
            "paired_full_face_rows_unassigned": len(paired_source_paths - {r["paths"] for r in combined})}
    for a, b in [("train", "test"), ("holdout_train", "holdout_val")] + [
        (f"SKF_TRAIN_Fold_{i}", f"SKF_VAL_Fold_{i}") for i in range(1, 6)
    ]:
        if a in split_sets and b in split_sets:
            summary["author_split_audit"][f"overlap_{a}__{b}"] = {
                "source_person_keys_all_metadata": len(split_sets[a] & split_sets[b]),
                "source_person_keys_paired_full_face": len(split_paired_sets[a] & split_paired_sets[b])}
    write_json(public / "inventory.json", summary)
    write_json(root / "inventory.json", summary)
    receipt_files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in {"manifest.json", "inventory.json"} and p.suffix != ".log":
            receipt_files.append({"local_path": str(p.relative_to(root)),
                                  "bytes": p.stat().st_size, "sha256": sha256(p)})
    write_json(public / "file_receipts.json", {"asset_root": "data/public/stw_access",
                                              "participant_assets_gitignored": True,
                                              "files": receipt_files})
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, default=Path("data/public/stw_access"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/data/access_2026_09_15/stw"))
    args = parser.parse_args()
    audit(args.private_root, args.report_dir)
