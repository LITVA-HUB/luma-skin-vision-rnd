"""Original licensed UMINHO metadata and one preassigned TRAIN reflectance cube."""
import hashlib,json,urllib.request
from pathlib import Path
from skin_mskcc_data import ROOT,sha

RAW=ROOT/'data/public/uminho_hsfd_v1'
OUT=ROOT/'docs/data/provenance/uminho_hsfd_v1'
IDS=[25598670,25594026,25599159,25599150,25599156,25599153,25637814,25638543,25638546]


def fetch(url):
    with urllib.request.urlopen(url,timeout=60) as response:return response.read()


def download(file):
    path=RAW/file['name']
    if path.exists():
        if path.stat().st_size!=file['size'] or hashlib.md5(path.read_bytes()).hexdigest()!=file['computed_md5']:
            raise ValueError('Existing original file differs')
        return path
    partial=path.with_suffix(path.suffix+'.part')
    if partial.exists():raise ValueError('Inspect incomplete download before resuming')
    md5=hashlib.md5();size=0
    with urllib.request.urlopen(file['download_url'],timeout=60) as source,partial.open('wb') as sink:
        while block:=source.read(1024*1024):
            size+=len(block)
            if size>file['size']:raise ValueError('Download larger than original metadata')
            sink.write(block);md5.update(block)
    if size!=file['size'] or md5.hexdigest()!=file['computed_md5']:raise ValueError('Original size/MD5 mismatch')
    partial.rename(path);return path


def main():
    RAW.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    metadata=RAW/'metadata';metadata.mkdir(exist_ok=True);articles={}
    for identifier in IDS:
        p=metadata/f'{identifier}.json'
        if not p.exists():p.write_bytes(fetch(f'https://api.figshare.com/v2/articles/{identifier}'))
        d=json.loads(p.read_bytes())
        if d['license']['name']!='CC BY 4.0' or d['license']['url']!='https://creativecommons.org/licenses/by/4.0/':
            raise ValueError('Original item license differs; no dataset download')
        articles[identifier]=d
    collection=metadata/'collection.json'
    if not collection.exists():collection.write_bytes(fetch('https://api.figshare.com/v2/collections/7163569'))
    readme=download(articles[25638546]['files'][0]);text=readme.read_bytes().decode('mac_roman')
    if 'Creative Commons Attribution 4.0 International License (CC-BY 4.0)' not in text:
        raise ValueError('Author README does not confirm the license')
    files=articles[25598670]['files'];assert len(files)==29
    ordered=sorted(files,key=lambda f:hashlib.sha256(('LumaUMINHOv1|'+f['name']).encode()).hexdigest())
    rows=[{'role':'test' if i<5 else 'validation' if i<10 else 'train',**f} for i,f in enumerate(ordered)]
    manifest=RAW/'manifest.json';payload=json.dumps({'split_rule':'SHA256 LumaUMINHOv1|original filename; first5test,next5validation,remaining19train','rows':rows},indent=2)+'\n'
    if manifest.exists() and manifest.read_text(encoding='utf8')!=payload:raise ValueError('Frozen spectral split changed')
    manifest.write_text(payload,encoding='utf8')  # Before any cube is opened/downloaded.
    docs=[download(articles[i]['files'][0]) for i in [25599156,25637814]]
    pilot=min((r for r in rows if r['role']=='train'),key=lambda r:r['size']);cube=download(pilot)
    receipt={'original_collection_doi':'10.6084/m9.figshare.c.7163569.v1','original_article':'https://api.figshare.com/v2/articles/25598670',
        'attribution':'Gomes, Andreia E.; Linhares, Joao M. M.; Nascimento, Sergio M. C. (2024), University of Minho Hyperspectral Faces Database: UMINHO-HSFD',
        'license':'CC BY 4.0; verified every original item and author README','license_url':'https://creativecommons.org/licenses/by/4.0/',
        'metadata_sha256':{str(i):sha(metadata/f'{i}.json') for i in IDS},'manifest_sha256':sha(manifest),
        'roles':{'train':19,'validation':5,'test':5},'reflectance_files_total':29,'reflectance_files_total_bytes':sum(f['size'] for f in files),
        'downloaded_cube_count':1,'pilot_role':'train','pilot_cube_bytes':cube.stat().st_size,'pilot_cube_sha256':sha(cube),
        'original_md5_and_size_verified':True,'support_file_bytes':sum(p.stat().st_size for p in [readme,*docs]),
        'support_sha256':{p.name:sha(p) for p in [readme,*docs]},
        'spectral_cube_decoded':False,'rgb_files_downloaded':False,'scope':'Measured facial reflectance; supplied RGB images are rendered, not independent camera captures',
        'acquisition_script_sha256':sha(Path(__file__))}
    (OUT/'acquisition.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8');print(json.dumps(receipt))


if __name__=='__main__':main()
