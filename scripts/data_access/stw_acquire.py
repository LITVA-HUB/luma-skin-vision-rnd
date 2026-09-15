#!/usr/bin/env python3
"""Acquire the author's public STW partial release, without account/EULA actions.

Only files linked from https://github.com/vitorpmh/STW are used. The author does
not redistribute CASIA or CelebA images in this archive. Those sources are not
downloaded here. Existing files are reused only if their SHA256 matches.
"""
from __future__ import annotations

import argparse
import hashlib
from html import unescape
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

REPO_COMMIT = "172e4906de9e91d7cee1090e63a34db95fe03f69"
DRIVE_ROOT = "https://drive.google.com/drive/folders/1jPVDyY0m_WH9VRwS6uaEtLyAhiWKF7ye"
# filename, original author Drive ID, expected bytes, SHA256 on 2026-09-15.
FILES = [
    ("images.zip", "1_HVULb5ubHXsu0VqSDkr6TX8xwSSTNO8", 636672248, "6bca3b5074c115d7f871a58414f0b24654e9ff2c90461ed901d4090dcd5d0e95"),
    ("all_annotated_data.csv", "1zP221CHHj5g7ZjelxeTevoITtdyEDxFG", 3818026, "ef98d1a0a23ca338b150542a2b596146f76308e58708276c0cf07314fd9a0d57"),
    ("holdout_train.csv", "1esJ1rfaQdwk8iBxgDupEacyklaRHLMoI", 1800178, "94a7836f44224e15eef9402c86db7950bf24936b18ca909a69d6010a301564db"),
    ("holdout_val.csv", "1LJLO9pGMnl-2VgkGDiMk_e54t81EqPOX", 406390, "47c710d304b5438cfd79421020d2b588e5980b6799583d2b590a0929e39d236f"),
    ("SKF_TRAIN_Fold_1.csv", "1MZG-gh7wzHmJtYzJqj3rmVtrxHRqhnh3", 1734909, "2794a0658b96aa10b9fbaf146720adfec0c0bf1e3150882c172f555a0b96652c"),
    ("SKF_TRAIN_Fold_2.csv", "1hJ8ZGBKBvTFvRuNkjnbj0szPldhgBKLe", 1754094, "9493ec661aa5dbb5b325427d9b4663466c6a3c1a8a855df5b62edbe2eb1d288a"),
    ("SKF_TRAIN_Fold_3.csv", "1aQkmLDfhg2t_qaJGJB2955xnffhv5Q8C", 1776245, "3bf462230910d70c796e6d50ad0dbd177fafb2208aea2663ccfa5393d5a93fca"),
    ("SKF_TRAIN_Fold_4.csv", "1unP-Nc8e_TR_x47YZ9-CuOxMCKDlZ6CK", 1788091, "3aadc1f969768e31f38761476d80ab854495a3a79ca06220f1f4c3c1d783f4e6"),
    ("SKF_TRAIN_Fold_5.csv", "1s547mP5UABA9OWOwxCenBA1OwRmjyfAa", 1772765, "15d776fa2e6492439d31d9df96f000e16b6de2c956ea8711215b7f86d186e0dd"),
    ("SKF_VAL_Fold_1.csv", "1gwOaDkrRZibx5_1IQ8tskOEYo9EI4oVA", 471659, "07bc7b371b381e972133afd1363520159043772d6efa893ca7a43d8fbb130d5b"),
    ("SKF_VAL_Fold_2.csv", "1NHTamFj6YbW2nEFcHWZpej3OXPSCEST8", 452474, "e69dac0dec3aa4ca9003b530d6c1d95b0ac6aa02a05f14cb3125c728afdf8d52"),
    ("SKF_VAL_Fold_3.csv", "18v8g9EwKqD-bMncq9VBF25jJ7dPpM65v", 430323, "260f39d75a464c8db87a2348e52b9273624893cc06645b91601771afd5118451"),
    ("SKF_VAL_Fold_4.csv", "1qGv9s7aTAkGzUqrsgiBaHyEPspmogk5t", 418477, "de8aac737d5595da5bf73f0af6a6c26ef314523e965ba0facc49cf5b85541694"),
    ("SKF_VAL_Fold_5.csv", "1KXfRoXOge3_xtj0ozaYAAzQ2UZ0Y5GnF", 433803, "8c4a7a3013eae89ac1497076ec6da7829843a2cec025ecbdbcc430bfa2e43782"),
    ("test.csv", "1nl_BFfadmEKaqDdfUxy6U6SKPaVZ4bP-", 545680, "d12ccde6114785cd4c0620402c7ae410c8c6a4b9774f82e8128834dbcfd9f1e9"),
    ("train.csv", "1F7EK_T2qdlJDjZxCJJKd5U799I7_tD3P", 2206512, "f8c40f5074561f8b792692d2369fcdba5ccb808db86e73af3e59d342585d2ca1"),
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def open_download(file_id: str):
    url = "https://drive.google.com/uc?" + urllib.parse.urlencode(
        {"export": "download", "id": file_id})
    response = urllib.request.urlopen(url, timeout=60)
    if "text/html" not in response.headers.get("Content-Type", ""):
        return response
    page = response.read(2 * 1024 * 1024).decode("utf-8")
    response.close()
    # Follow only Google's ordinary large-file virus-scan warning. This is not
    # an account, permission, license, consent or captcha workflow.
    if "too large for Google to scan for viruses" not in page:
        raise RuntimeError("Google Drive returned a non-download page; owner action may be needed")
    action = re.search(r'<form[^>]*action="([^"]+)"', page)
    if not action or unescape(action.group(1)) != "https://drive.usercontent.google.com/download":
        raise RuntimeError("Unexpected download confirmation endpoint")
    params = {unescape(k): unescape(v) for k, v in re.findall(
        r'<input type="hidden" name="([^"]+)" value="([^"]*)"', page)}
    if params.get("id") != file_id or params.get("export") != "download":
        raise RuntimeError("Unexpected download confirmation identity")
    response = urllib.request.urlopen(action.group(1) + "?" + urllib.parse.urlencode(params), timeout=60)
    if "text/html" in response.headers.get("Content-Type", ""):
        response.close()
        raise RuntimeError("Download still unavailable; no further access actions will be attempted")
    return response


def acquire(root: Path) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, file_id, expected_bytes, expected_sha in FILES:
        path = root / filename if filename in {"images.zip", "all_annotated_data.csv"} else root / "splits" / "individuals_train_test_splits" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        cached = path.exists()
        if cached:
            if path.stat().st_size != expected_bytes or digest(path) != expected_sha:
                raise RuntimeError(f"Existing {filename} has a different hash; preserved without overwrite")
        else:
            part = path.with_suffix(path.suffix + ".part")
            with open_download(file_id) as response, part.open("wb") as f:
                size = 0
                for b in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    size += len(b)
                    if size > expected_bytes:
                        raise RuntimeError(f"{filename} exceeds frozen release size")
                    f.write(b)
            if part.stat().st_size != expected_bytes or digest(part) != expected_sha:
                raise RuntimeError(f"{filename} does not match frozen release; retained as .part")
            part.rename(path)
        record = {"path": str(path.relative_to(root)), "cached": cached,
                  "bytes": expected_bytes, "sha256": expected_sha,
                  "source_url": "https://drive.google.com/file/d/" + file_id + "/view"}
        records.append(record)
        print(json.dumps(record), flush=True)
    (root / "acquisition_receipt.json").write_text(json.dumps({
        "official_repo": "https://github.com/vitorpmh/STW",
        "official_repo_commit": REPO_COMMIT, "author_drive_root": DRIVE_ROOT,
        "source_licenses_separate": True, "MST_is_not_Lab": True,
        "files": records}, indent=2) + "\n")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/public/stw_access"))
    acquire(parser.parse_args().root)
