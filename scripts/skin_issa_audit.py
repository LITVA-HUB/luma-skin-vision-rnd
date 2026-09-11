"""Audit native formula/cache consistency on explicitly enabled ISSA roles."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from luma_skin_vision.color import delta_e00
from skin_issa_data import ROOT, OUT, PRIVATE, RAW, NS, constants, endpoint_rows, sha, write_json, column_name
from skin_issa_color import spectral_xyz, source_lab


def run(role):
    if role == 'validation':
        lock = json.loads((OUT/'material_lock.json').read_text(encoding='utf-8-sig'))
        for path, value in lock['bindings'].items():
            assert sha(ROOT/path) == value, path
    header = constants()
    columns = [column_name(i) for i in range(14,57)]
    wavelengths = np.array([float(header[2][c]) for c in columns])
    cmf = np.array([[float(header[r][c]) for r in (3,4,5)] for c in columns])
    spd = np.array([float(header[6][c]) for c in columns])
    white = np.array([float(header[6][c]) for c in ('BF','BG','BH')])
    common = (wavelengths>=400)&(wavelengths<=700)
    white_gap = float(np.max(np.abs(spectral_xyz(np.full((1,31),100.),cmf[common],spd[common])[0]-white)))
    records, spectra, xyz, labs, declared, formula_patterns = [], [], [], [], [], Counter()
    missing_in_support = outside_support = formulas = 0
    for meta, cells, row in endpoint_rows(role):
        spec = np.array([float(cells[c]) if c in cells else np.nan for c in columns])
        mask = (wavelengths>=float(meta['J']))&(wavelengths<=float(meta['K']))
        missing_in_support += int(np.sum(~np.isfinite(spec[mask])))
        outside_support += int(np.sum(np.isfinite(spec[~mask])))
        records.append(meta); spectra.append(spec); declared.append(mask)
        xyz.append([float(cells[c]) for c in ('BF','BG','BH')])
        labs.append([float(cells[c]) for c in ('BK','BL','BM')])
        for cell in row:
            formula = cell.find(NS+'f')
            if formula is not None:
                col = ''.join(c for c in cell.attrib['r'] if c.isalpha())
                pattern = (re.sub(r'(\$?[A-Z]+\$?)\d+',r'\1#',formula.text) if formula.text
                    else '[shared formula reference; numerical replay checked separately]')
                formula_patterns[f'{col}: {pattern}'] += 1
                formulas += 1
    spectra=np.stack(spectra);xyz=np.asarray(xyz);labs=np.asarray(labs);declared=np.asarray(declared)
    if missing_in_support or outside_support:
        raise ValueError(f'Support mismatch: {missing_in_support} missing, {outside_support} unexpected')
    reproduction=np.empty_like(xyz)
    for bounds in sorted({(r['J'],r['K']) for r in records}):
        rows=np.array([(r['J'],r['K'])==bounds for r in records])
        mask=declared[rows][0]
        reproduction[rows]=spectral_xyz(spectra[rows][:,mask],cmf[mask],spd[mask])
    source_lab_replay=source_lab(xyz,white)
    # Independent scalar math.fsum XYZ replay, separate from matrix dot above.
    scalar_gap=0.
    for index in range(len(records)):
        use=np.flatnonzero(declared[index])
        denominator=math.fsum(float(cmf[j,1]*spd[j]) for j in use)
        for c in range(3):
            scalar=math.fsum(float(spectra[index,j])*float(cmf[j,c])*float(spd[j]) for j in use)/denominator
            scalar_gap=max(scalar_gap,abs(scalar-xyz[index,c]))
    common_lab=source_lab(spectral_xyz(spectra[:,common],cmf[common],spd[common]),white)
    common_error=delta_e00(common_lab,labs)
    by_support={}
    for support in sorted({f"{r['J']}-{r['K']}" for r in records}):
        mask=np.array([f"{r['J']}-{r['K']}"==support for r in records])
        e=common_error[mask]
        by_support[support]={'n':int(mask.sum()),'common31_to_supplied_Lab_mean_delta_e00':float(e.mean()),
            'p95':float(np.quantile(e,.95)),'max':float(e.max())}
    PRIVATE.mkdir(parents=True,exist_ok=True)
    cache=PRIVATE/f'{role}.npz'
    np.savez_compressed(cache,spectra_percent=spectra,declared_support=declared,xyz=xyz,lab=labs,
        wavelength=wavelengths,cmf=cmf,spd=spd,white=white,
        subject=np.array([r['C'] for r in records]),origin=np.array([r['B'] for r in records]),
        record=np.array([r['A'] for r in records]),body=np.array([r['G'] for r in records]))
    report={'role':role,'records':len(records),'subjects':len({r['C'] for r in records}),
        'reflectance_unit':'percent; native XYZ formula has no factor 100',
        'measured_reflectance_percent_min':float(np.nanmin(spectra)),
        'measured_reflectance_percent_max':float(np.nanmax(spectra)),
        'common31_outside_open_unit_interval_records':int(np.any((spectra[:,common]<=0)|(spectra[:,common]>=100),axis=1).sum()),
        'missing_declared_samples':missing_in_support,'numbers_outside_declared_support':outside_support,
        'white_reproduction_max_abs':white_gap,'xyz_reproduction_max_abs':float(np.abs(reproduction-xyz).max()),
        'xyz_scalar_fsum_max_abs':scalar_gap,'xyz_scalar_comparisons':len(records)*3,
        'native_lab_formula_reproduction_max_abs':float(np.abs(source_lab_replay-labs).max()),
        'common31_identity_to_native_reference':by_support,
        'formula_patterns':dict(formula_patterns),'formula_cells_inspected':formulas,
        'scope':'Original measured spectra and cached colorimetry; no camera input or accuracy measured',
        'reserved_numerical_endpoints_read':False,
        'bindings':{p:sha(ROOT/p) for p in ['scripts/skin_issa_audit.py','scripts/skin_issa_color.py',
            'tests/test_skin_issa_color.py','docs/benchmarks/skin_issa_v1/split_lock.json',
            'src/luma_skin_vision/color.py',str(cache.relative_to(ROOT)).replace('\\','/')]}}
    write_json(OUT/f'{role}_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'formula_patterns','bindings'}}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--role',choices=['train','validation'],default='train')
    run(parser.parse_args().role)
