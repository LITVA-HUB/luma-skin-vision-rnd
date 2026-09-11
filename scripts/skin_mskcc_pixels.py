"""Source-only verified pixel cache and deterministic color/patch features."""
import concurrent.futures
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from skin_mskcc_data import ROOT, RAW, MANIFEST, sha, load_source

PROTOCOL = ROOT/'docs/research/skin_mskcc_pixel_protocol_v1.md'
CACHE = ROOT/'data/processed/skin_mskcc_pixels_v1'
BENCH = ROOT/'docs/benchmarks/skin_mskcc_pixels_v1'


def features(rgb):
    z = np.asarray(rgb, dtype=np.float64)/255.
    if z.shape != (128,128,3):
        raise ValueError('Expected128x128RGB')
    flat = z.reshape(-1,3)
    median = np.median(flat,axis=0)
    std = flat.std(0)
    constant = np.ptp(flat,axis=0) == 0
    std[constant] = 0
    covariance = (flat-flat.mean(0)).T@(flat-flat.mean(0))/len(flat)
    corr = covariance / np.maximum(std[:,None]*std[None,:],1e-12)
    corr[constant,:] = 0; corr[:,constant] = 0
    corr = np.clip(corr,-1,1)
    color = np.concatenate([np.quantile(flat,[.01,.05,.1,.25,.5,.75,.9,.95,.99],axis=0).ravel(),
                            flat.mean(0), std, corr[np.triu_indices(3,1)]])
    bins = np.minimum((flat*8).astype(int),7)
    hist = np.bincount(bins[:,0]*64+bins[:,1]*8+bins[:,2],minlength=512)/len(flat)
    patches = z.reshape(8,16,8,16,3).transpose(0,2,1,3,4).reshape(64,16,16,3)
    p = patches.reshape(64,256,3)
    quant = np.quantile(p,[.1,.5,.9],axis=1).transpose(1,0,2).reshape(64,9)
    gradient = .5*np.abs(np.diff(patches,axis=1)).mean((1,2))+.5*np.abs(np.diff(patches,axis=2)).mean((1,2))
    tokens = np.concatenate([quant,p.mean(1),p.std(1),gradient],axis=1)
    return {'median':median, 'color':color, 'hist':hist,
            'color_hist':np.concatenate([color,hist]), 'tokens':tokens}


def decode(row, expected):
    path = RAW/'images'/(row['image']+'.jpg')
    if sha(path) != expected[row['image']]['sha256']:
        raise ValueError('Original JPEG hash mismatch')
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        w,h = im.size; side=int(.8*min(w,h)); left=(w-side)//2; top=(h-side)//2
        rgb=np.asarray(im.crop((left,top,left+side,top+side)).resize((128,128),Image.Resampling.LANCZOS))
    return rgb, features(rgb)


def main():
    CACHE.mkdir(exist_ok=False,parents=True);BENCH.mkdir(exist_ok=True,parents=True)
    expected={r['image']:r for r in json.loads((RAW/'image_receipts.json').read_bytes())}
    receipt={'protocol_sha256':sha(PROTOCOL),'manifest_sha256':sha(MANIFEST),
             'script_sha256':sha(Path(__file__)),'roles':{},'test_and_calibration_decoded':False}
    for role in ['train','validation']:
        rows,_,y,rep=load_source(role)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            decoded=list(pool.map(lambda row:decode(row,expected),rows))
        values={'rgb':np.stack([r[0] for r in decoded]),'target':y,'repetitions':rep,
                **{k:np.stack([r[1][k] for r in decoded]).astype(np.float32) for k in decoded[0][1]},
                **{k:np.array([r[k] for r in rows]) for k in ['image','patient','site','device','mode','image_type']}}
        for k in ['median','color','hist','color_hist','tokens','target']:
            if not np.isfinite(values[k]).all(): raise ValueError('Invalid feature/target')
        path=CACHE/(role+'.npz');np.savez(path,**values)
        receipt['roles'][role]={'n':len(rows),'sha256':sha(path),'bytes':path.stat().st_size,
                                'patients':len(set(values['patient']))}
        print(json.dumps({'cache_role':role,**receipt['roles'][role]}),flush=True)
    (BENCH/'cache.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')


def load(role):
    if role not in {'train','validation'}: raise ValueError('Source-only cache')
    receipt=json.loads((BENCH/'cache.json').read_bytes());path=CACHE/(role+'.npz')
    if receipt['protocol_sha256'] != sha(PROTOCOL) or receipt['script_sha256'] != sha(Path(__file__)) or receipt['roles'][role]['sha256'] != sha(path):
        raise ValueError('Cache/protocol/code changed')
    return dict(np.load(path))


if __name__ == '__main__':main()
