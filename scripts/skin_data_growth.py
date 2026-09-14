"""Traceable local data acquisition, grouping and unlabeled-photo intake."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

DATA_ROOT = Path('D:/Luma-RnD/data_growth_2026_09_14')
UCI_PAGE = 'https://archive.ics.uci.edu/dataset/229/skin%2Bsegmentation'
SALT = b'LumaUCIColorGroups20260914|'


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2) + '\n'
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.replace(path)


def validate_uci(raw):
    a = np.asarray(raw)
    if a.ndim != 2 or a.shape[1] != 4 or len(a) == 0:
        raise ValueError('expected nonempty B,G,R,class table')
    if not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
        raise ValueError('non-finite or nonnumeric table')
    if not np.equal(a, np.floor(a)).all():
        raise ValueError('noninteger pixels or labels')
    if np.any((a[:, :3] < 0) | (a[:, :3] > 255)):
        raise ValueError('BGR outside 0..255')
    if not np.isin(a[:, 3], [1, 2]).all():
        raise ValueError('unknown class; author skin=1, nonskin=2')
    return a[:, :3].astype(np.uint8), (a[:, 3] == 1).astype(np.uint8)


def color_partitions(bgr):
    a = np.asarray(bgr)
    if a.ndim != 2 or a.shape[1] != 3 or a.dtype != np.uint8:
        raise ValueError('partitions require validated uint8 BGR')
    unique, inverse = np.unique(a, axis=0, return_inverse=True)
    buckets = np.array([int.from_bytes(hashlib.sha256(SALT + c.tobytes()).digest()[:8], 'big') % 100
                        for c in unique])
    return np.where(buckets < 70, 0, np.where(buckets < 85, 1, 2)).astype(np.uint8)[inverse]


def profile_uci(bgr, labels, partition):
    unique, inverse, count = np.unique(bgr, axis=0, return_inverse=True, return_counts=True)
    positives = np.bincount(inverse, weights=labels, minlength=len(unique))
    return dict(rows=len(bgr), unique_colors=len(unique), duplicate_color_rows=int(len(bgr)-len(unique)),
                conflicting_colors=int(np.sum((positives > 0) & (positives < count))),
                skin_rows=int(labels.sum()), non_skin_rows=int(len(labels)-labels.sum()),
                partitions={name:dict(rows=int(np.sum(partition == code)),
                                      skin=int(labels[partition == code].sum()),
                                      unique_colors=int(len(np.unique(bgr[partition == code], axis=0))))
                            for code, name in enumerate(['train', 'validation', 'test'])},
                evaluation_unit='exact-color groups; source image/person identifiers unavailable',
                ground_truth_type='binary skin/nonskin pixel; no instrument Lab')


def ingest_photos(sources, directory):
    sources, directory = [Path(p) for p in sources], Path(directory).resolve()
    prepared = []
    for source in sources:
        raw = source.read_bytes()
        try:
            with Image.open(io.BytesIO(raw)) as im:
                im.verify()
            with Image.open(io.BytesIO(raw)) as im:
                size, fmt, mode = im.size, im.format, im.mode
        except Exception as exc:
            raise ValueError(f'image decode failed: {source.name}') from exc
        prepared.append((sha_bytes(raw), source, raw, size, fmt, mode))
    hashes = sorted(set(p[0] for p in prepared))
    batch_group = 'provided_batch_' + sha_bytes('|'.join(hashes).encode())[:16]
    records, seen = [], set()
    directory.mkdir(parents=True, exist_ok=True)
    for digest, source, raw, (width, height), fmt, mode in prepared:
        if digest in seen:
            continue
        seen.add(digest)
        extension = '.jpg' if fmt == 'JPEG' else '.png' if fmt == 'PNG' else '.img'
        dest = directory / (digest + extension)
        if dest.exists() and sha_bytes(dest.read_bytes()) != digest:
            raise ValueError('existing private image hash mismatch')
        if not dest.exists():
            with dest.open('xb') as stream:
                stream.write(raw)
        records.append(dict(id=digest[:16], sha256=digest, path=str(dest), source_basename=source.name,
                            width=width, height=height, format=fmt, mode=mode, bytes=len(raw),
                            role='unlabeled_diagnostic', leakage_group=batch_group,
                            ground_truth_lab=None, eligible_supervised_color=False,
                            subject_id=None, identity_grouping='not inferred; entire batch kept together',
                            rights='user supplied for local task; no public redistribution grant assumed'))
    manifest = dict(source='user_provided_2026_09_14', supplied_files=len(sources),
                    batch_group=batch_group, records=records,
                    reference_status='User confirmed: photographs only; no colorimeter or color card')
    write_json(directory / 'manifest.json', manifest)
    return manifest


def fetch_original(url, dest, max_bytes=20000000):
    dest = Path(dest)
    receipt = dest.with_suffix(dest.suffix + '.source.json')
    if dest.exists():
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        if saved['url'] != url or sha_bytes(dest.read_bytes()) != saved['sha256']:
            raise ValueError('existing original provenance/hash mismatch')
        return saved
    request = Request(url, headers={'User-Agent': 'LumaResearch/1.0 (dataset acquisition)'})
    with urlopen(request, timeout=45) as response:
        payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError('download exceeds bounded size')
        saved = dict(url=url, final_url=response.url, bytes=len(payload), sha256=sha_bytes(payload),
                     content_type=response.headers.get('Content-Type'), etag=response.headers.get('ETag'))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('xb') as stream:
        stream.write(payload)
    write_json(receipt, saved)
    return saved


def acquire_uci(directory):
    from html.parser import HTMLParser

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []

        def handle_starttag(self, tag, attrs):
            if tag == 'a':
                self.links.extend(value for name, value in attrs if name == 'href' and value)

    directory = Path(directory)
    landing = directory / 'original_landing.html'
    fetch_original(UCI_PAGE, landing)
    html = landing.read_text(encoding='utf-8')
    if 'Creative Commons Attribution 4.0' not in html:
        raise ValueError('original dataset license not found')
    parser = Links()
    parser.feed(html)
    matches = [urljoin(UCI_PAGE, url) for url in parser.links if '/229/' in url and '.zip' in url]
    if len(set(matches)) != 1:
        raise ValueError('unambiguous original UCI download link not found')
    archive = directory / 'original.zip'
    receipt = fetch_original(matches[0], archive)
    with zipfile.ZipFile(archive) as z:
        matches = [name for name in z.namelist() if Path(name).name == 'Skin_NonSkin.txt']
        if len(matches) != 1 or z.getinfo(matches[0]).file_size > 10000000:
            raise ValueError('unexpected UCI archive')
        raw = z.read(matches[0])
    bgr, labels = validate_uci(np.loadtxt(io.BytesIO(raw)))
    if len(bgr) != 245057 or int(labels.sum()) != 50859:
        raise ValueError('original author counts disagree')
    text_path = directory / 'Skin_NonSkin.txt'
    if text_path.exists() and text_path.read_bytes() != raw:
        raise ValueError('existing UCI original differs')
    text_path.write_bytes(raw)
    partition = color_partitions(bgr)
    arrays = directory / 'grouped_pixels.npz'
    np.savez_compressed(arrays, bgr=bgr, skin=labels, partition=partition)
    profile = profile_uci(bgr, labels, partition)
    profile.update(archive=receipt, original_text_sha256=sha_bytes(raw),
                   arrays_sha256=sha_bytes(arrays.read_bytes()), salt=SALT.decode(),
                   license='CC BY 4.0', attribution='Bhatt, R. & Dhall, A. (2009). Skin Segmentation. UCI. DOI 10.24432/C5T30C',
                   source_code_sha256=sha_bytes(Path(__file__).read_bytes()))
    write_json(directory / 'profile.json', profile)
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uci', action='store_true')
    parser.add_argument('--photos', nargs='*', type=Path)
    args = parser.parse_args()
    if args.uci:
        print(json.dumps(acquire_uci(DATA_ROOT / 'uci_skin_segmentation'), ensure_ascii=True))
    if args.photos:
        result = ingest_photos(args.photos, DATA_ROOT / 'private_user_faces')
        print(json.dumps(dict(supplied=result['supplied_files'], unique=len(result['records']),
                              manifest=str(DATA_ROOT / 'private_user_faces/manifest.json')), ensure_ascii=True))


if __name__ == '__main__':
    main()
