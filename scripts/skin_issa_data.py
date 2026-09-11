"""Read-only original ISSA extraction; numerical endpoint access is role gated."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/public/issa_v4/ISSA_17_Jan_2025_Yan_Lu.xlsx'
RAW_SHA = '7396aa7608f1a6f91ec70bf8f8421252a9d81f71a2bca066eb7e98155b9d1161'
PRIVATE = ROOT / 'data/processed/skin_issa_v1'
OUT = ROOT / 'docs/benchmarks/skin_issa_v1'
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf8')


def column_name(index):
    if index < 1:
        raise ValueError('Column index must be positive')
    name = ''
    while index:
        index, remainder = divmod(index-1, 26)
        name = chr(65+remainder) + name
    return name


def selected_cells(row, strings, columns):
    result = {}
    for cell in row:
        column = ''.join(x for x in cell.attrib['r'] if x.isalpha())
        if column not in columns:
            continue  # Do not decode a value outside the requested columns.
        value = cell.find(NS+'v')
        if value is None:
            continue
        result[column] = strings[int(value.text)] if cell.attrib.get('t') == 's' else value.text
    return result


def original_xml(path=RAW):
    if sha(path) != RAW_SHA:
        raise ValueError('Original ISSA workbook hash mismatch')
    with zipfile.ZipFile(path) as archive:
        strings = [''.join(x.itertext()) for x in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        tree = ET.fromstring(archive.read('xl/worksheets/sheet2.xml'))
    return strings, tree.find(NS+'sheetData')


def metadata(path=RAW):
    strings, sheet = original_xml(path)
    rows = []
    for row in sheet:
        if int(row.attrib['r']) < 13:
            continue
        values = selected_cells(row, strings, set('ABCDEFGHIJKL'))
        if not values:
            continue
        if not set('ABCDGHIJKL') <= set(values) or not values['A'].isdigit():
            raise ValueError('Unexpected metadata row')
        values['row'] = int(row.attrib['r'])
        rows.append(values)
    if len(rows) != 15256 or len({r['A'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate record identifiers')
    return rows


def assign_roles(rows):
    by_origin = defaultdict(set)
    subject_origins = defaultdict(set)
    for row in rows:
        by_origin[row['B']].add(row['C'])
        subject_origins[row['C']].add(row['B'])
    if any(len(v) != 1 for v in subject_origins.values()):
        raise ValueError('Subject label occurs in multiple origins')
    roles = {}
    for origin, subjects in by_origin.items():
        if not 1 <= int(origin) <= 11:
            raise ValueError('Unknown source origin')
        ordered = sorted(subjects, key=lambda subject: hashlib.sha256(
            f'LumaISSA1|{origin}|{subject}'.encode('utf8')).digest())
        n_train, n_val = int(.70*len(ordered)), int(.15*len(ordered))
        for index, subject in enumerate(ordered):
            if int(origin) >= 9:
                role = f'reserved_origin_{origin}'
            elif index < n_train:
                role = 'train'
            elif index < n_train + n_val:
                role = 'validation'
            else:
                role = 'known_source_test'
            roles[(origin, subject)] = role
    return roles


def checked_manifest():
    lock = json.loads((OUT/'split_lock.json').read_text(encoding='utf-8-sig'))
    for name, expected in lock['bindings'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'ISSA split binding changed: {name}')
    return json.loads((PRIVATE/'roles.json').read_text(encoding='utf-8-sig'))


def endpoint_rows(role, path=RAW):
    if role not in {'train', 'validation'}:
        raise ValueError('Reserved numerical endpoints are not enabled')
    manifest = checked_manifest()
    selected = {r['row']: r for r in manifest if r['role'] == role}
    strings, sheet = original_xml(path)
    columns = {column_name(i) for i in range(14,70)}
    for row in sheet:
        index = int(row.attrib['r'])
        if index not in selected:
            continue  # Filter role before numerical endpoint values are decoded.
        yield selected[index], selected_cells(row, strings, columns), row


def constants(path=RAW):
    strings, sheet = original_xml(path)
    columns = {column_name(i) for i in range(14,70)}
    return {int(r.attrib['r']): selected_cells(r, strings, columns)
            for r in sheet if int(r.attrib['r']) in {2,3,4,5,6,12}}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if (OUT/'split_lock.json').exists():
        checked_manifest()
        print('Existing split verified; no endpoints read.')
        return
    rows = metadata()
    roles = assign_roles(rows)
    manifest = [dict(r, role=roles[(r['B'],r['C'])]) for r in rows]
    subjects = defaultdict(list)
    for r in rows:
        subjects[r['C']].append(r)
    origins = {}
    for origin in sorted({r['B'] for r in rows}, key=int):
        rr = [r for r in manifest if r['B'] == origin]
        origins[origin] = {
            'records': len(rr), 'subject_labels': len({r['C'] for r in rr}),
            'record_roles': dict(Counter(r['role'] for r in rr)),
            'declared_support': dict(Counter(f"{r['J']}-{r['K']}/{r['L']}nm" for r in rr)),
            'instrument': dict(Counter(r['H'] for r in rr)),
            'specular_mode': dict(Counter(r['I'] for r in rr)),
            'body_locations': dict(Counter(r['G'] for r in rr)),
        }
    report = {'scope':'metadata only, numeric participant endpoints not decoded',
        'records': len(rows), 'subject_labels': len(subjects),
        'author_reported_subjects':2113, 'subject_count_discrepancy':6,
        'identity_limitation':'Disjoint original labels, not independently verified identities; do not repair ambiguous codes.',
        'missing_demographic_cells':{k:sum(k not in r for r in rows) for k in ('E','F')},
        'subject_metadata_conflicts': {k:sum(len({r.get(k) for r in rr})>1 for rr in subjects.values()) for k in ('B','D','E','F')},
        'origins':origins,
        'record_roles':dict(Counter(r['role'] for r in manifest)),
        'subject_roles':dict(Counter(roles.values()))}
    write_json(PRIVATE/'roles.json', manifest)
    write_json(OUT/'metadata.json', report)
    bindings = ['scripts/skin_issa_data.py','tests/test_skin_issa_data.py',
        'docs/research/skin_issa_material_protocol_v1.md',
        'data/public/issa_v4/ISSA_17_Jan_2025_Yan_Lu.xlsx',
        'data/processed/skin_issa_v1/roles.json', 'docs/benchmarks/skin_issa_v1/metadata.json']
    write_json(OUT/'split_lock.json', {'bindings':{p:sha(ROOT/p) for p in bindings},
        'numeric_participant_endpoints_read':False,'reserved_roles_enabled':False})
    print(json.dumps({k:report[k] for k in ('records','subject_labels','record_roles','subject_roles')}))


if __name__ == '__main__':
    main()
