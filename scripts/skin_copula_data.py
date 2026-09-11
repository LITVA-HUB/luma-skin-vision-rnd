"""Source-only rank-dependence context, keeping absolute RGB in the main arm."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load as load_source_cache,features,CACHE as SOURCE_CACHE

CACHE=ROOT/'data/processed/skin_copula_v1'
OUT=ROOT/'docs/benchmarks/skin_copula_v1'


def rank_channels(x):
    x=np.asarray(x,dtype=np.float64)
    if x.shape[-1]!=3 or not np.isfinite(x).all():raise ValueError('Finite three-channel input required')
    flat=x.reshape(-1,3)
    return ((rankdata(flat,method='average',axis=0)-.5)/len(flat)).reshape(x.shape)


def joint_hist(x):
    x=np.asarray(x,dtype=np.float64).reshape(-1,3)
    if not np.isfinite(x).all() or np.any(x<0) or np.any(x>1):raise ValueError('Histogram domain is [0,1]')
    bins=np.minimum((x*8).astype(int),7)
    return np.bincount(bins[:,0]*64+bins[:,1]*8+bins[:,2],minlength=512)/len(x)


def pack(data,arm):
    if arm not in ['none','rgb_hist','copula','rank_only']:raise ValueError(arm)
    tokens=data['rank_tokens'] if arm=='rank_only' else data['tokens']
    hist=data['hist'] if arm=='rgb_hist' else data['rank_hist']
    if arm=='none':hist=np.zeros_like(hist)
    return np.concatenate([tokens.reshape(len(tokens),-1),hist],1).astype(np.float32)


def load(role):
    if role not in ['train','validation']:raise ValueError('Source-only roles')
    receipt=json.loads((OUT/'feature_receipt.json').read_bytes())
    assert sha(Path(__file__))==receipt['script_sha256']
    assert sha(SOURCE_CACHE/f'{role}.npz')==receipt['roles'][role]['original_cache_sha256']
    file=CACHE/f'{role}.npz';assert sha(file)==receipt['roles'][role]['derived_sha256']
    data=load_source_cache(role)
    with np.load(file) as a:
        for key in a.files:data[key]=a[key].copy()
    assert len(data['rank_hist'])==len(data['tokens'])==len(data['rank_tokens'])
    return data


def main():
    CACHE.mkdir(parents=True,exist_ok=False);OUT.mkdir(parents=True,exist_ok=True)
    receipt={'scope':'Only source TRAIN/VALIDATION cached real JPEG pixels; no new labels/pixels/endpoints',
             'script_sha256':sha(Path(__file__)),'roles':{},'invariance_probe':[]}
    for role in ['train','validation']:
        data=load_source_cache(role);histograms=[];tokens=[]
        for rgb in data['rgb']:
            x=rgb.astype(float)/255.;rank=rank_channels(x)
            np.testing.assert_array_equal(joint_hist(x),features(rgb)['hist'])
            histograms.append(joint_hist(rank));tokens.append(features(rank*255.)['tokens'])
        file=CACHE/f'{role}.npz';np.savez(file,rank_hist=np.array(histograms,dtype=np.float32),rank_tokens=np.array(tokens,dtype=np.float32))
        receipt['roles'][role]={'n':len(histograms),'original_cache_sha256':sha(SOURCE_CACHE/f'{role}.npz'),'derived_sha256':sha(file),'bytes':file.stat().st_size}
        if role=='train':
            matrix=np.array([[.8,.1,.1],[.15,.7,.15],[.1,.2,.7]])
            for i,rgb in enumerate(data['rgb'][:16]):
                x=rgb.astype(float)/255.;reference=joint_hist(rank_channels(x))
                variants={'strict_monotone':.03+.85*x**np.array([.6,1.4,2.2]),'channel_mixing':x@matrix.T,
                          'clipping':np.clip(x*np.array([1.3,.8,1.1])+np.array([.02,.05,-.03]),0,1)}
                for kind,y in variants.items():
                    tv=float(.5*np.abs(joint_hist(rank_channels(y))-reference).sum())
                    if kind=='strict_monotone':assert tv==0
                    receipt['invariance_probe'].append({'source_order_index':i,'derived_transform':kind,'rank_hist_total_variation':tv})
    (OUT/'feature_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'roles':receipt['roles'],'invariance_cases':len(receipt['invariance_probe']),'input_transform_probe_is_not_accuracy':True}))


if __name__=='__main__':main()
