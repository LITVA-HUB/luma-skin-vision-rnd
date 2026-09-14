"""Role-safe Seg2 sources and matched sampling; no model or CUDA initialization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

DATA_ROOT = Path('D:/Luma-RnD/data_growth_2026_09_14')
ROLES = ('train', 'validation', 'test')
SOURCES = ('lapa', 'celeba')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Changed source file: {path}')


def integer_ids(values, size=None, unique=False):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) == 0 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError('Nonempty integer row IDs required')
    ids = values.astype(np.int64, copy=True)
    if np.any(ids < 0) or (size is not None and np.any(ids >= size)):
        raise ValueError('Row ID outside source array')
    if unique and len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate allowed row ID')
    return ids


def targets(raw, encoding):
    raw = np.asarray(raw)
    if raw.dtype != np.uint8:
        raise ValueError('Expected uint8 labels')
    if encoding == 'lapa11':
        if np.any(raw > 10):
            raise ValueError('Invalid LaPa label')
        return ((raw == 1) | (raw == 6)).astype(np.uint8)
    if encoding == 'binary':
        if np.any(raw > 1):
            raise ValueError('Invalid binary label')
        return raw.copy()
    raise ValueError('Unknown label encoding')


class ArraySplit:
    def __init__(self, source, role, rgb, labels, indices, encoding, metadata=None):
        if source not in SOURCES or role not in ROLES or encoding not in ('lapa11', 'binary'):
            raise ValueError('Unknown source, role or encoding')
        if (rgb.ndim != 4 or rgb.shape[-1] != 3 or labels.shape != rgb.shape[:-1]
                or rgb.dtype != np.uint8 or labels.dtype != np.uint8
                or min(rgb.shape[1:3]) < 16 or any(n % 16 for n in rgb.shape[1:3])):
            raise ValueError('Expected matching uint8 RGB/label arrays with spatial multiples of 16')
        self.source, self.role, self.encoding = source, role, encoding
        self.rgb, self.labels = rgb, labels
        self.indices = integer_ids(indices, len(rgb), unique=True)
        self.indices.setflags(write=False)
        self.allowed = np.zeros(len(rgb), dtype=bool)
        self.allowed[self.indices] = True
        self.metadata = metadata
        if metadata is not None and set(metadata) != set(self.indices):
            raise ValueError('Metadata and permitted rows differ')

    def get(self, global_ids):
        ids = integer_ids(global_ids, len(self.rgb))
        if not self.allowed[ids].all():
            raise ValueError(f'Row outside allowed {self.source}/{self.role} split')
        return np.array(self.rgb[ids], copy=True), targets(self.labels[ids], self.encoding)


class ImageFilesSplit:
    """Original LaPa TEST files, opened only by an explicitly authorized evaluator."""
    def __init__(self, records):
        self.source, self.role = 'lapa', 'test'
        self.records = records
        self.indices = np.arange(len(records), dtype=np.int64)
        self.indices.setflags(write=False)
        self.metadata = {i:dict(source='lapa', role='test', original_id=r['stem'],
                               group='lapa:'+r['conservative_source_group'],
                               source_sha256=r['components']['images']['sha256'])
                         for i,r in enumerate(records)}

    def get(self, global_ids):
        ids = integer_ids(global_ids, len(self.records))
        rgb, label = [], []
        for index in ids:
            components = self.records[index]['components']
            for kind in ('images', 'labels'):
                checked(components[kind]['path'], components[kind]['sha256'])
            with Image.open(components['images']['path']) as image, Image.open(components['labels']['path']) as mask:
                if image.size != mask.size:
                    raise ValueError('Original image and mask dimensions differ')
                rgb.append(np.asarray(image.convert('RGB').resize((192, 192), Image.Resampling.BILINEAR)))
                label.append(np.asarray(mask.resize((192, 192), Image.Resampling.NEAREST)))
        return np.stack(rgb), targets(np.stack(label), 'lapa11')


def load_split(source, role, allow_test=False, data_root=DATA_ROOT):
    if role == 'test' and not allow_test:
        raise ValueError('Explicit post-selection test access required')
    if source not in SOURCES or role not in ROLES:
        raise ValueError('Unknown source or role')
    data_root = Path(data_root)
    if source == 'lapa':
        root = data_root / 'lapa'
        pool = root / 'prepared_192'
        profile = read(pool / 'profile.json')
        checked(root / 'usable_triplets.json', profile['triplets_sha256'])
        name = 'val' if role == 'validation' else role
        rows = [r for r in read(root / 'usable_triplets.json') if r['split'] == name]
        if role == 'test':
            if len(rows) != 2000:
                raise ValueError('Unexpected LaPa TEST count')
            return ImageFilesSplit(rows)
        if read(pool / f'{name}_order.json') != [r['stem'] for r in rows]:
            raise ValueError('LaPa row order no longer matches original mapping')
        arrays = {}
        for kind in ('rgb', 'labels'):
            path = pool / f'{name}_{kind}.npy'
            checked(path, profile['splits'][name][kind+'_sha256'])
            arrays[kind] = np.load(path, mmap_mode='r', allow_pickle=False)
        if len(rows) != profile['splits'][name]['rows'] or len(rows) != len(arrays['rgb']):
            raise ValueError('LaPa source count mismatch')
        metadata = {i:dict(source=source, role=role, original_id=r['stem'],
                           group='lapa:'+r['conservative_source_group'],
                           source_sha256=r['components']['images']['sha256']) for i,r in enumerate(rows)}
        return ArraySplit(source, role, arrays['rgb'], arrays['labels'], np.arange(len(rows)), 'lapa11', metadata)
    root = data_root / 'celeba_mask_hq'
    pool, split = root / 'prepared_192_v1', root / 'split_v1'
    profile, split_profile = read(pool / 'profile.json'), read(split / 'profile.json')
    checked(pool / 'profile.json', split_profile['pool_profile_sha256'])
    for filename in ('rows.json', role+'_indices.npy'):
        checked(split / filename, split_profile['files'][filename]['sha256'])
    rows = [r for r in read(split / 'rows.json') if r['role'] == role]
    indices = np.load(split / (role+'_indices.npy'), allow_pickle=False)
    if indices.tolist() != [r['index'] for r in rows] or len(rows) != split_profile['kept'][role]['images']:
        raise ValueError('CelebA cleaned split/index mismatch')
    arrays = {}
    for filename in ('images.npy', 'face_skin.npy'):
        checked(pool / filename, profile['files'][filename]['sha256'])
        arrays[filename] = np.load(pool / filename, mmap_mode='r', allow_pickle=False)
    metadata = {r['index']:dict(source=source, role=role, original_id=r['original_file'],
                                group='celeba:'+str(r['conservative_group']),
                                person_code=r['person_code'], source_sha256=r['source_sha256']) for r in rows}
    return ArraySplit(source, role, arrays['images.npy'], arrays['face_skin.npy'], indices, 'binary', metadata)


class _Cycle:
    def __init__(self, ids, seed, stream):
        self.ids = integer_ids(ids, unique=True)
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
        self.order, self.offset = self.rng.permutation(self.ids), 0

    def take(self, count):
        chunks = []
        while count:
            if self.offset == len(self.order):
                self.order, self.offset = self.rng.permutation(self.ids), 0
            length = min(count, len(self.order)-self.offset)
            chunks.append(self.order[self.offset:self.offset+length])
            self.offset += length
            count -= length
        return np.concatenate(chunks)


class PairedSampler:
    def __init__(self, lapa_ids, celeba_ids, seed, batch_size=32):
        if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
            raise ValueError('Nonnegative 32-bit seed required')
        if not isinstance(batch_size, int) or batch_size < 2 or batch_size % 2:
            raise ValueError('Positive even batch size required')
        self.half = batch_size // 2
        self.anchor = _Cycle(lapa_ids, seed, 0)
        self.extra = {'lapa_only': _Cycle(lapa_ids, seed, 1), 'lapa_celeba': _Cycle(celeba_ids, seed, 2)}
        self.step, self.cross_stream_duplicate_presentations = 0, 0

    def next(self, arm):
        if arm not in self.extra:
            raise ValueError('Unregistered data arm')
        anchor, extra = self.anchor.take(self.half), self.extra[arm].take(self.half)
        if arm == 'lapa_only':
            self.cross_stream_duplicate_presentations += int(np.isin(extra, anchor).sum())
        self.step += 1
        sources = np.zeros(self.half*2, dtype=np.uint8)
        if arm == 'lapa_celeba':
            sources[self.half:] = 1
        return sources, np.concatenate((anchor, extra))


def assemble_batch(splits, source_ids, global_ids):
    sources, ids = np.asarray(source_ids), np.asarray(global_ids)
    if (sources.ndim != 1 or sources.shape != ids.shape or not len(sources)
            or not np.issubdtype(sources.dtype, np.integer) or not np.isin(sources, [0, 1]).all()):
        raise ValueError('Aligned valid source and global row IDs required')
    x = y = None
    for source in np.unique(sources):
        if source not in splits or splits[source].role != 'train' or splits[source].source != SOURCES[source]:
            raise ValueError('Training batch requires the correct TRAIN source views')
        positions = np.flatnonzero(sources == source)
        a, b = splits[source].get(ids[positions])
        if x is None:
            x = np.empty((len(ids), *a.shape[1:]), dtype=np.uint8)
            y = np.empty((len(ids), *b.shape[1:]), dtype=np.uint8)
        if x.shape[1:] != a.shape[1:] or y.shape[1:] != b.shape[1:]:
            raise ValueError('Source spatial dimensions differ')
        x[positions], y[positions] = a, b
    return x, y
