"""Bounded TRAIN-only extension: the three smallest source cubes, fixed by bytes."""
import json
from pathlib import Path
from skin_uminho_acquire import RAW, OUT, download
from skin_mskcc_data import sha


def main():
    receipt = json.loads((OUT / 'acquisition.json').read_bytes())
    manifest = RAW / 'manifest.json'
    assert sha(manifest) == receipt['manifest_sha256']
    rows = sorted((r for r in json.loads(manifest.read_bytes())['rows'] if r['role'] == 'train'),
                  key=lambda r: (r['size'], r['name']))[:3]
    plan = {'rule': 'Three smallest original TRAIN cubes by bytes; no VAL/TEST access', 'rows': rows}
    frozen = RAW / 'train_expand_v1.json'
    if frozen.exists():
        assert json.loads(frozen.read_bytes()) == plan
    else:
        frozen.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf8')
    records = []
    for row in rows:
        if Path(row['name']).name != row['name'] or not (RAW / row['name']).resolve().is_relative_to(RAW.resolve()):
            raise ValueError('Invalid original path')
        path = download(row)
        records.append({'bytes': path.stat().st_size, 'sha256': sha(path), 'role': 'train'})
    result = {'scope': plan['rule'], 'original_manifest_sha256': sha(manifest),
              'local_plan_sha256': sha(frozen), 'license': 'Original CC BY 4.0',
              'script_sha256': sha(Path(__file__)), 'files': records,
              'total_cube_bytes': sum(r['bytes'] for r in records)}
    target = OUT / 'train_expand_v1.json'
    if target.exists():
        assert json.loads(target.read_bytes()) == result
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
