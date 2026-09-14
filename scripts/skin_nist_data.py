"""Parse original NIST triplicate spectra without inventing paired photographs."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
from skin_data_growth import DATA_ROOT, sha_bytes, write_json


def parse_nist(text):
    rows = list(csv.reader(io.StringIO(text)))
    header = next((i for i, row in enumerate(rows) if row and row[0].strip() == 'Wavelength (nm)'), None)
    if header is None:
        raise ValueError('missing wavelength header')
    columns = rows[header]
    while columns and not columns[-1].strip():
        columns = columns[:-1]
    if len(columns) < 5 or (len(columns)-1) % 4:
        raise ValueError('unexpected triplicate layout')
    labels = [columns[k:k+4] for k in range(1, len(columns), 4)]
    if any(group[3] != 'Average' or len(set(group[:3])) != 3 or
           any(not re.fullmatch(r'R[1-9][0-9]*', label) for label in group[:3])
           for group in labels):
        raise ValueError('unexpected triplicate layout')
    matrix = np.asarray([[float(v) for v in row[:len(columns)]]
                         for row in rows[header+1:] if row and row[0].strip()], dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError('nonfinite reflectance')
    wave = matrix[:, 0]
    if not np.all(np.diff(wave) > 0):
        raise ValueError('wavelength must strictly increase')
    count = (len(columns)-1)//4
    blocks = matrix[:, 1:].reshape(len(wave), count, 4).transpose(1, 2, 0)
    repeats, averages = blocks[:, :3], blocks[:, 3]
    if np.max(np.abs(repeats.mean(1)-averages)) > .000101:
        raise ValueError('reported average disagrees beyond rounding')
    # Partition subject columns as units; all three repeats remain with their subject.
    order = sorted(range(count), key=lambda i: hashlib.sha256(f'LumaNIST20260914|Subject {i+1}'.encode()).digest())
    partition = np.zeros(count, np.uint8)
    n = max(1, round(count*.15)) if count >= 3 else 0
    partition[order[:n]], partition[order[n:2*n]] = 2, 1
    return dict(wavelength_nm=wave, repeats=repeats, average=averages, partition=partition,
                repeat_labels=np.asarray([group[:3] for group in labels]),
                source_subject=np.arange(1, count+1))


def main():
    root = DATA_ROOT / 'nist_reflectance'
    raw = (root / 'original_spectra.txt').read_bytes()
    arrays = parse_nist(raw.decode('cp1252'))
    assert arrays['repeats'].shape[0] == 100
    assert arrays['wavelength_nm'][0] == 250 and arrays['wavelength_nm'][-1] == 2500
    out = root / 'grouped_spectra.npz'
    np.savez_compressed(out, **arrays)
    p = dict(subjects=len(arrays['repeats']), repeat_spectra=int(np.prod(arrays['repeats'].shape[:2])),
             average_spectra=len(arrays['average']), wavelength_samples=len(arrays['wavelength_nm']),
             wavelength_min=float(arrays['wavelength_nm'][0]), wavelength_max=float(arrays['wavelength_nm'][-1]),
             minimum=float(arrays['repeats'].min()), maximum=float(arrays['repeats'].max()),
             negative_values=int(np.sum(arrays['repeats'] < 0)), above_one_values=int(np.sum(arrays['repeats'] > 1)),
             max_average_rounding_difference=float(np.max(np.abs(arrays['repeats'].mean(1)-arrays['average']))),
             partitions={name:int(np.sum(arrays['partition'] == i)) for i,name in enumerate(['train','validation','test'])},
             original_sha256=sha_bytes(raw), arrays_sha256=sha_bytes(out.read_bytes()),
             code_sha256=sha_bytes(Path(__file__).read_bytes()), input_encoding='cp1252',
             nonstandard_repeat_labels={str(i+1):row.tolist() for i,row in enumerate(arrays['repeat_labels'])
                                        if row.tolist() != ['R1','R2','R3']},
             citation='Barnes, Allen, Tsai (2017). Reference Data Set of Human Skin Reflectance. NIST. DOI 10.18434/M38597',
             terms='Original NIST catalog links NIST non-SRD data terms; attribution retained in original_terms.html',
             labels='measured reflectance factor, not photographic RGB or instrument Lab pairs',
             augmentation_policy='future rendered camera observations must be labeled derived/simulated',
             no_values_clipped=True)
    write_json(root / 'profile.json', p)
    print(json.dumps(p))


if __name__ == '__main__':
    main()
