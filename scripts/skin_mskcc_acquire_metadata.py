"""Reproduce the small original MSKCC metadata acquisition from pinned receipts."""
import hashlib
import json
import urllib.request
from skin_mskcc_data import RAW, PROV


def main():
    receipt = json.loads((PROV / 'acquisition.json').read_bytes())
    allowed = {'mskcc-skin-tone-labeling-dataset.csv', 's1.csv', 's2.csv',
               's4.csv', 's7.csv', 'supplemental-data-dictionary.txt'}
    if {f['file'] for f in receipt['files']} != allowed:
        raise ValueError('Unexpected metadata inventory')
    RAW.mkdir(exist_ok=True, parents=True)
    for item in receipt['files']:
        if not item['url'].startswith('https://isic-archive.s3.amazonaws.com/dois/10.34970-962049/'):
            raise ValueError('Unrecognized original host/path')
        if not 0 < item['bytes'] < 2_000_000:
            raise ValueError('Metadata acquisition size cap')
        path = RAW / item['file']
        if path.exists():
            data = path.read_bytes()
        else:
            with urllib.request.urlopen(item['url'], timeout=30) as response:
                data = response.read(item['bytes'] + 1)
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('Original metadata differs from pinned receipt')
        if not path.exists():
            path.write_bytes(data)
    print(json.dumps({'verified_original_metadata_files': len(allowed),
                      'license': 'Original CC-BY; version unspecified on landing page',
                      'numeric_endpoints_decoded': False}))


if __name__ == '__main__':
    main()
