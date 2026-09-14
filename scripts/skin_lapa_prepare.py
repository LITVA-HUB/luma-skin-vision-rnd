"""Validate LaPa originals and prepare train/validation only; never decode test images."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
from skin_data_growth import DATA_ROOT, sha_bytes, write_json


def member_target(root, name):
    root = Path(root).resolve()
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name or not p.parts or p.parts[0] != 'LaPa':
        raise ValueError('unsafe archive path')
    target = (root / Path(*p.parts[1:])).resolve()
    if not target.is_relative_to(root):
        raise ValueError('archive path escapes output')
    return target


def select_rows(records):
    groups = defaultdict(dict)
    for item in records:
        key = (item['split'], item['stem'])
        if item['kind'] in groups[key]:
            raise ValueError('duplicate dataset component')
        groups[key][item['kind']] = item
    rows = []
    for (split, stem), items in sorted(groups.items()):
        if set(items) != {'images', 'labels', 'landmarks'}:
            raise ValueError('missing image/label/landmark triplet')
        rows.append(dict(split=split, stem=stem, components=items,
                         conservative_source_group=stem.rsplit('_', 1)[0]))
    # Keep official test intact. Exclude lower-priority exact-byte or filename-prefix overlaps.
    priority = {'train':0, 'val':1, 'test':2}
    prefix_roles, hash_roles = defaultdict(set), defaultdict(set)
    for row in rows:
        prefix_roles[row['conservative_source_group']].add(row['split'])
        hash_roles[row['components']['images']['sha256']].add(row['split'])
    kept, dropped = [], []
    for row in rows:
        other = prefix_roles[row['conservative_source_group']] | hash_roles[row['components']['images']['sha256']]
        if any(priority[s] > priority[row['split']] for s in other):
            dropped.append(dict(split=row['split'], stem=row['stem'], reason='higher-priority split shares image bytes or conservative source prefix'))
        else:
            kept.append(row)
    return kept, dropped


def file_sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def extract(root):
    archive, destination = root/'LaPa.tar.gz', root/'originals'
    receipt = json.loads((root/'archive_source.json').read_text())
    if file_sha(archive) != receipt['sha256']:
        raise ValueError('archive checksum mismatch')
    if (root/'file_manifest.json').exists():
        manifest = json.loads((root/'file_manifest.json').read_text())
        assert manifest['archive_sha256'] == receipt['sha256']
        return manifest
    records, seen, count = [], set(), 0
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            target = member_target(destination, member.name)
            if member.isdir():
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError('non-regular dataset member')
            parts = PurePosixPath(member.name).parts
            if (len(parts) != 4 or parts[1] not in ['train','val','test'] or
                    parts[2] not in ['images','labels','landmarks'] or
                    target.suffix.lower() not in ['.jpg','.png','.txt']):
                raise ValueError('unexpected dataset member: '+member.name)
            if member.name in seen or member.size > 20_000_000:
                raise ValueError('duplicate/oversized archive member')
            seen.add(member.name)
            count += member.size
            if count > 10_000_000_000:
                raise ValueError('expanded archive exceeds bound')
            raw = tar.extractfile(member).read()
            if len(raw) != member.size:
                raise ValueError('truncated archive member')
            digest = sha_bytes(raw)
            if target.exists() and file_sha(target) != digest:
                raise ValueError('existing original differs')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(raw)
            records.append(dict(split=parts[1],kind=parts[2],stem=target.stem,
                                bytes=len(raw),sha256=digest,path=str(target)))
            if len(records) % 6000 == 0:
                print('LAPA EXTRACT',len(records),count,flush=True)
    rows, dropped = select_rows(records)
    counts = dict(Counter(item['split'] for item in records if item['kind'] == 'images'))
    # The actual author-linked archive contains 18168 train triplets, eight fewer
    # than the 18176 often cited for LaPa. Preserve this discrepancy; invent no files.
    if counts != {'train':18168,'val':2000,'test':2000}:
        raise ValueError('original archive image counts changed: '+str(counts))
    manifest = dict(archive_sha256=receipt['sha256'], original_counts=counts,
                    file_count=len(records),expanded_bytes=count,records=records,
                    usable_counts=dict(Counter(row['split'] for row in rows)), exclusions=dropped,
                    commonly_reported_train_count=18176, observed_train_count_difference=-8,
                    identity_disjointness='not established; official partitions plus exact-byte/source-prefix exclusion',
                    test_decoded=False,code_sha256=sha_bytes(Path(__file__).read_bytes()))
    write_json(root/'file_manifest.json',manifest)
    write_json(root/'usable_triplets.json',rows)
    print('LAPA ORIGINALS VALIDATED',json.dumps({k:v for k,v in manifest.items() if k != 'records'}),flush=True)
    return manifest


def prepare(root, size):
    rows = json.loads((root/'usable_triplets.json').read_text())
    out = root/f'prepared_{size}'
    profile_path = out/'profile.json'
    if profile_path.exists():
        raise RuntimeError('prepared dataset exists; preserve its receipt')
    out.mkdir(parents=True,exist_ok=True)
    profiles = {}
    for split in ['train','val']:
        subset = [row for row in rows if row['split'] == split]
        images_path, labels_path = out/f'{split}_rgb.npy',out/f'{split}_labels.npy'
        if images_path.exists() or labels_path.exists():
            raise RuntimeError('incomplete prepared arrays exist; inspect them first')
        images = np.lib.format.open_memmap(images_path,mode='w+',dtype=np.uint8,shape=(len(subset),size,size,3))
        labels = np.lib.format.open_memmap(labels_path,mode='w+',dtype=np.uint8,shape=(len(subset),size,size))
        hist, shapes = np.zeros(11,np.int64),Counter()
        for i,row in enumerate(subset):
            c = row['components']
            if file_sha(c['images']['path']) != c['images']['sha256'] or file_sha(c['labels']['path']) != c['labels']['sha256']:
                raise ValueError('original image/label checksum changed')
            with Image.open(c['images']['path']) as im, Image.open(c['labels']['path']) as label:
                im.load()
                label.load()
                lab = np.asarray(label)
                if im.size != label.size or lab.ndim != 2 or not np.isin(lab,np.arange(11)).all():
                    raise ValueError('invalid image/semantic-label pair')
                hist += np.bincount(lab.reshape(-1),minlength=11)
                shapes[str(im.size)] += 1
                images[i] = np.asarray(im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR))
                labels[i] = np.asarray(label.resize((size,size),Image.Resampling.NEAREST))
            if (i+1) % 2000 == 0:
                print('LAPA PREPARE',split,i+1,'of',len(subset),flush=True)
        images.flush()
        labels.flush()
        del images, labels
        profiles[split] = dict(rows=len(subset),label_pixel_counts=hist.tolist(),
                               original_shapes=dict(shapes),rgb_sha256=file_sha(images_path),labels_sha256=file_sha(labels_path))
        write_json(out/f'{split}_order.json',[row['stem'] for row in subset])
    result = dict(size=size,splits=profiles,test_decoded=False,
                   original_manifest_sha256=file_sha(root/'file_manifest.json'),
                   triplets_sha256=file_sha(root/'usable_triplets.json'),
                   code_sha256=sha_bytes(Path(__file__).read_bytes()),
                   color_ground_truth=None,label_type='11-class semantic pixel masks, not measured color',
                   resampling='bilinear RGB, nearest neighbor labels; no color normalization or augmentation at acquisition')
    write_json(profile_path,result)
    print('LAPA PREPARED',json.dumps({k:v for k,v in result.items() if k != 'splits'}),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract',action='store_true')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--size',type=int,default=192)
    args = parser.parse_args()
    root = DATA_ROOT/'lapa'
    if args.extract:
        extract(root)
    if args.prepare:
        if not 64 <= args.size <= 512:
            raise ValueError('invalid image size')
        prepare(root,args.size)


if __name__ == '__main__':
    main()
