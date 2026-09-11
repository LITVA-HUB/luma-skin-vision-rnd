"""Acquire original licensed annotations; no additional image/cube access."""
import json
from pathlib import Path
from skin_mskcc_data import sha
from skin_uminho_acquire import RAW, OUT, download


def main():
    original = json.loads((OUT / 'acquisition.json').read_bytes())
    records = []
    for identifier in [25599159, 25599150, 25599153, 25638543]:
        metadata = RAW / 'metadata' / f'{identifier}.json'
        assert sha(metadata) == original['metadata_sha256'][str(identifier)]
        article = json.loads(metadata.read_bytes())
        assert article['license']['url'] == 'https://creativecommons.org/licenses/by/4.0/'
        for file in article['files']:
            name = file['name']
            if Path(name).name != name or not (RAW / name).resolve().is_relative_to(RAW.resolve()):
                raise ValueError('Original filename escapes data root')
            path = download(file)
            records.append({'article': article['url'], 'file_name': name,
                            'bytes': path.stat().st_size, 'sha256': sha(path),
                            'original_md5': file['computed_md5']})
    result = {'scope': 'Original annotations and rendering source, not executed; no image/cube acquisition',
              'license': 'Original CC BY 4.0', 'files': records, 'script_sha256': sha(Path(__file__))}
    target = OUT / 'support_v2.json'
    if target.exists():
        assert json.loads(target.read_bytes()) == result
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
