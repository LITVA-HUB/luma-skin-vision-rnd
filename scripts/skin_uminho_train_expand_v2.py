"""Acquire all original TRAIN reflectance cubes; preserve the old held allocation."""
import hashlib
import json
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path('C:/Users/dimal/Documents/просто/luma-skin-vision-rnd/data/public/uminho_hsfd_v1')
OUT = Path('D:/Luma-RnD/data_growth_2026_09_14/uminho_train_v2')
MANIFEST_SHA = '3d3b7dc7256eb04250dee2f5ef76880f4101d019313d97ba051480f8d320b5c7'


def training_rows(rows):
    chosen = [r for r in rows if r['role'] == 'train']
    if len(chosen) != 19:
        raise ValueError('Exactly 19 original TRAIN cubes required')
    for row in chosen:
        name = row['name']
        if not name or any(c in name for c in '/\\:') or name in ('.', '..'):
            raise ValueError('Unsafe source filename')
    if len({r['name'] for r in chosen}) != 19:
        raise ValueError('Duplicate source filename')
    return sorted(chosen, key=lambda r: r['name'])


def digest(path, algorithm='sha256'):
    state = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            state.update(block)
    return state.hexdigest()


def save_once(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Preserve existing artifact: ' + str(path))
        return
    path.write_text(payload, encoding='utf-8')


def checked_file(row, target):
    if not target.exists():
        partial = target.with_suffix(target.suffix + '.part')
        with partial.open('xb') as sink, urlopen(row['download_url'], timeout=60) as source:
            size = 0
            md5 = hashlib.md5()
            while block := source.read(1024 * 1024):
                size += len(block)
                if size > row['size']:
                    raise ValueError('Source larger than original metadata')
                md5.update(block)
                sink.write(block)
        if size != row['size'] or md5.hexdigest() != row['computed_md5']:
            raise ValueError('Downloaded source size/MD5 mismatch')
        partial.rename(target)
    if target.stat().st_size != row['size'] or digest(target, 'md5') != row['computed_md5']:
        raise ValueError('Existing source size/MD5 mismatch')
    return dict(path=str(target), bytes=target.stat().st_size, sha256=digest(target), role='train')


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = LEGACY / 'manifest.json'
    if digest(manifest) != MANIFEST_SHA:
        raise ValueError('Original allocation changed')
    rows = training_rows(json.loads(manifest.read_text())['rows'])
    metadata = OUT / 'source_metadata.json'
    if not metadata.exists():
        with urlopen('https://api.figshare.com/v2/articles/25598670', timeout=60) as response:
            metadata.write_bytes(response.read())
    current = json.loads(metadata.read_text())
    lookup = {f['name']: f for f in current['files']}
    for row in rows:
        for key in ['size', 'computed_md5', 'download_url']:
            if row[key] != lookup[row['name']][key]:
                raise ValueError('Current author metadata differs from frozen source')
    plan_path = OUT / 'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if plan['source_sha256'] != digest(__file__) or plan['rows'] != rows:
            raise ValueError('Frozen acquisition plan changed')
    else:
        paths = [str(LEGACY / r['name'] if (LEGACY / r['name']).exists() else OUT / r['name']) for r in rows]
        plan = dict(scope='All 19 original TRAIN cubes; old VAL/TEST remain untouched',
                    manifest_sha256=MANIFEST_SHA, source_metadata_sha256=digest(metadata),
                    source_sha256=digest(__file__), rows=rows, paths=paths,
                    existing_originals=sum(Path(p).parent == LEGACY for p in paths),
                    dataset_license=current['license'], camera_photographs=False,
                    supervision='measured reflectance; any RGB rendering is a derived input')
        save_once(plan_path, plan)
    records = []
    for i, (row, path) in enumerate(zip(rows, plan['paths'], strict=True)):
        expected = LEGACY / row['name'] if Path(path).parent == LEGACY else OUT / row['name']
        if Path(path) != expected:
            raise ValueError('Frozen target path differs')
        record = checked_file(row, expected)
        record['new_in_this_extension'] = expected.parent == OUT
        records.append(record)
        print('UMINHO TRAIN', i + 1, 'of', len(rows), 'verified bytes', record['bytes'], flush=True)
    result = dict(plan_sha256=digest(plan_path), source_sha256=digest(__file__),
                  total_cubes=len(records), new_cubes=sum(r['new_in_this_extension'] for r in records),
                  total_bytes=sum(r['bytes'] for r in records),
                  new_bytes=sum(r['bytes'] for r in records if r['new_in_this_extension']),
                  files=records, original_md5_and_size_verified=True,
                  held_cubes_acquired=0, independent_rgb_photographs=0,
                  source='https://api.figshare.com/v2/articles/25598670',
                  attribution='Gomes, Linhares, Nascimento (2024), UMINHO-HSFD',
                  license=current['license'], seconds=time.perf_counter()-started)
    if (OUT / 'profile.json').exists():
        old = json.loads((OUT / 'profile.json').read_text(encoding='utf-8'))
        if old['files'] != records or old['plan_sha256'] != result['plan_sha256']:
            raise ValueError('Previous acquisition receipt differs')
        print('Existing complete acquisition verified; receipt preserved', flush=True)
        return
    save_once(OUT / 'profile.json', result)
    print('UMINHO COMPLETE', json.dumps({k: v for k, v in result.items() if k not in ['files', 'license']}), flush=True)


if __name__ == '__main__':
    main()
