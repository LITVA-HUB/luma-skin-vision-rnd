"""Local cross-source image-content screen; TRAIN/VALIDATION only, no biometrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from PIL import Image
from PIL import __version__ as pillow_version
from scipy.fft import dctn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("D:/Luma-RnD/data_growth_2026_09_14")
OUT = DATA / "cross_source_similarity_v1"
SEG2 = Path("D:/Luma-RnD/skin_face_transfer_v1/registration.json")
SEG2_SHA = "c92d12e53e42325cec76fd6ad61dc94a8c685fcba9a48b97ccc0a791d108da88"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"preserve existing artifact: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def describe(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 16:
        raise ValueError("uint8 RGB image required")
    thumb = np.asarray(Image.fromarray(rgb).resize((64, 64), Image.Resampling.LANCZOS))
    gray = np.asarray(Image.fromarray(thumb).convert("L").resize((32, 32), Image.Resampling.LANCZOS), float)
    hashes = []
    for view in (gray, gray[:, ::-1]):
        coefficients = dctn(view, type=2, norm="ortho", workers=1)[:8, :8].ravel()
        bits = np.concatenate(([False], coefficients[1:] > np.median(coefficients[1:])))
        hashes.append(int.from_bytes(np.packbits(bits, bitorder="big").tobytes(), "big"))
    return thumb, np.asarray(hashes, np.uint64), float(gray.std())


def candidates(left, right, max_distance=8, block=512):
    left, right = np.asarray(left), np.asarray(right)
    if (left.ndim != 1 or right.ndim != 1 or left.dtype != np.uint64 or right.dtype != np.uint64
            or type(max_distance) is not int or not 0 <= max_distance <= 64
            or type(block) is not int or block < 1):
        raise ValueError("uint64 hashes and valid search bounds required")
    for begin in range(0, len(left), block):
        distance = np.bitwise_count(left[begin:begin+block, None] ^ right[None])
        ii, jj = np.nonzero(distance <= max_distance)
        for i, j in zip(ii, jj, strict=True):
            yield begin+int(i), int(j), int(distance[i, j])


def similarity(a, b):
    if a.shape != (64, 64, 3) or b.shape != a.shape:
        raise ValueError("matching 64x64 RGB thumbnails required")
    u = np.asarray(Image.fromarray(a).convert("L"), float)
    v = np.asarray(Image.fromarray(b).convert("L"), float)
    u, v = u-u.mean(), v-v.mean()
    scale = float(np.sqrt(np.sum(u*u)*np.sum(v*v)))
    return dict(gray_correlation=float(np.sum(u*v)/scale) if scale > 0 else None,
                rgb_mae=float(np.abs(a.astype(float)-b.astype(float)).mean()))


def sources():
    if digest(SEG2) != SEG2_SHA:
        raise ValueError("Seg2 registration changed")
    expected = {Path(k): v for k, v in read(SEG2)["bindings"].items()}
    result, bindings = [], {str(SEG2): SEG2_SHA}
    for source, role, count in (("lapa", "train", 15914), ("lapa", "validation", 1692),
                                ("celeba", "train", 24112), ("celeba", "validation", 2992)):
        if source == "lapa":
            name = "val" if role == "validation" else role
            base = DATA / "lapa/prepared_192"
            rgb_path, metadata = base / f"{name}_rgb.npy", base / f"{name}_order.json"
        else:
            rgb_path = DATA / "celeba_mask_hq/prepared_192_v1/images.npy"
            metadata = DATA / "celeba_mask_hq/split_v1" / f"{role}_indices.npy"
        for path in (rgb_path, metadata):
            if path not in expected or digest(path) != expected[path]:
                raise ValueError(f"source does not match Seg2: {path}")
            bindings[str(path)] = expected[path]
        images = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
        ids = np.arange(count) if source == "lapa" else np.load(metadata, allow_pickle=False)
        if (images.dtype != np.uint8 or images.shape[1:] != (192, 192, 3) or ids.shape != (count,)
                or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= len(images))
                or not np.array_equal(ids, np.unique(ids))
                or (source == "lapa" and (len(images) != count or len(read(metadata)) != count))):
            raise ValueError("source shape or permitted global indices mismatch")
        result.append(dict(source=source, role=role, ids=ids, images=images))
    if np.intersect1d(result[2]["ids"], result[3]["ids"]).size:
        raise ValueError("CelebA TRAIN/VALIDATION overlap")
    for name in ("scripts/skin_image_similarity.py", "tests/test_skin_image_similarity.py",
                 "docs/data/skin_cross_source_similarity_v1_protocol.md"):
        path = ROOT / name
        bindings[str(path)] = digest(path)
    return result, bindings


def search(rows, thumbs, hashes, spread):
    left = np.asarray([i for i, row in enumerate(rows) if row["source"] == "lapa" and spread[i] >= 8], np.int64)
    right = np.asarray([i for i, row in enumerate(rows) if row["source"] == "celeba" and spread[i] >= 8], np.int64)
    found = []
    for i, j, distance in candidates(hashes[left, 0], hashes[right].reshape(-1)):
        a, b, flip = int(left[i]), int(right[j//2]), j % 2
        second = thumbs[b, :, ::-1] if flip else thumbs[b]
        score = similarity(thumbs[a], second)
        strong = (score["gray_correlation"] is not None and score["gray_correlation"] >= 0.995
                  and score["rgb_mae"] <= 8)
        found.append(dict(left=a, right=b, horizontal_flip=bool(flip), hamming=distance,
                          **score, strong_candidate=strong,
                          cross_role=rows[a]["role"] != rows[b]["role"]))
    strong = [p for p in found if p["strong_candidate"]]
    return dict(candidate_views=found, searched_image_pairs=int(len(left)*len(right)),
                searched_view_pairs=int(2*len(left)*len(right)),
                candidate_pairs=len({(p["left"], p["right"]) for p in found}),
                strong_pairs=len({(p["left"], p["right"]) for p in strong}),
                strong_cross_role_pairs=len({(p["left"], p["right"]) for p in strong if p["cross_role"]}),
                low_contrast_rows=int(np.sum(spread < 8)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    args = parser.parse_args()
    if args.command == "run" and (OUT / "thumbs.npy").exists():
        raise FileExistsError("preserve existing screen; use verify")
    source_sets, bindings = sources()
    registration = dict(bindings=bindings, rows=44710, max_hamming=8, min_gray_std=8,
                        strong_gray_correlation=0.995, strong_rgb_mae=8,
                        versions=dict(numpy=np.__version__, scipy=scipy.__version__, pillow=pillow_version),
                        roles=["train", "validation"], test_feature_rows=0, mask_rows_read=0,
                        uses_identity_features=False)
    if args.command == "run":
        write_once(OUT / "registration.json", registration)
        thumbs = np.lib.format.open_memmap(OUT / "thumbs.npy", mode="w+", dtype=np.uint8, shape=(44710, 64, 64, 3))
        hashes, spread = np.empty((44710, 2), np.uint64), np.empty(44710, float)
        rows = []
    else:
        if read(OUT / "registration.json") != registration:
            raise ValueError("registration did not reproduce")
        old = read(OUT / "results.json")
        for name, expected in old["files"].items():
            if digest(OUT / name) != expected:
                raise ValueError(f"saved output changed: {name}")
        thumbs = np.load(OUT / "thumbs.npy", mmap_mode="r", allow_pickle=False)
        with np.load(OUT / "features.npz", allow_pickle=False) as archive:
            hashes, spread = archive["hashes"], archive["spread"]
        rows = read(OUT / "rows.json")
    offset = 0
    with threadpool_limits(limits=1):
        for source in source_sets:
            for index in source["ids"]:
                thumb, h, s = describe(source["images"][index])
                row = dict(source=source["source"], role=source["role"], global_index=int(index))
                if args.command == "run":
                    thumbs[offset], hashes[offset], spread[offset] = thumb, h, s
                    rows.append(row)
                else:
                    if rows[offset] != row:
                        raise ValueError("row mapping mismatch")
                    np.testing.assert_array_equal(thumbs[offset], thumb)
                    np.testing.assert_array_equal(hashes[offset], h)
                    if spread[offset] != s:
                        raise ValueError("contrast statistic mismatch")
                offset += 1
            print("SIMILARITY", args.command, source["source"], source["role"], len(source["ids"]), flush=True)
        if offset != 44710:
            raise ValueError("incomplete source coverage")
        outcome = search(rows, thumbs, hashes, spread)
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError("input changed during screen")
    if "torch" in sys.modules:
        raise ValueError("image-content screen unexpectedly imported Torch")
    if args.command == "run":
        thumbs.flush()
        with (OUT / "features.npz").open("xb") as output:
            np.savez(output, hashes=hashes, spread=spread)
        write_once(OUT / "rows.json", rows)
        files = {name: digest(OUT / name) for name in ("thumbs.npy", "features.npz", "rows.json")}
        result = dict(registration_sha256=digest(OUT / "registration.json"), files=files, outcome=outcome,
                      counts=dict(Counter(row["source"]+"/"+row["role"] for row in rows)),
                      confirmed_duplicates=None, visual_review_pending=True,
                      test_feature_rows=0, neural_inference=False, changed_dataset_rows=0,
                      limitations=["No crop/rotation/large-edit invariance or identity matching.",
                                   "No within-source search or TEST feature comparison.",
                                   "Strong candidates require image-content review; a negative screen does not prove independence."])
        write_once(OUT / "results.json", result)
    else:
        if old["registration_sha256"] != digest(OUT / "registration.json") or old["outcome"] != outcome:
            raise ValueError("candidate screen did not reproduce")
        write_once(OUT / "verification.json", dict(passed=True, results_sha256=digest(OUT / "results.json"),
                   registration_sha256=digest(OUT / "registration.json"), all_source_thumbnails_and_candidates_replayed=True,
                   verified_image_rows=44710, duplicate_recall_validated=False))
    print(json.dumps(dict(command=args.command, result_sha256=digest(OUT / "results.json"),
                          image_pairs=outcome["searched_image_pairs"], candidates=outcome["candidate_pairs"],
                          strong_pairs=outcome["strong_pairs"], strong_cross_role_pairs=outcome["strong_cross_role_pairs"],
                          low_contrast_rows=outcome["low_contrast_rows"])), flush=True)


if __name__ == "__main__":
    main()
