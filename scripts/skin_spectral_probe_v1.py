"""TRAIN-only oracle material representation and declared illuminant stress probe."""
import json
from pathlib import Path
import numpy as np


def project_spectra(train, query, rank, space):
    if space not in ('linear', 'log'):
        raise ValueError(space)
    if space == 'log':
        if np.any(train <= 0) or np.any(query <= 0):
            raise ValueError('Log spectrum requires positive original measurements')
        train, query = np.log(train), np.log(query)
    mean = train.mean(axis=0)
    _, _, vt = np.linalg.svd(train-mean, full_matrices=False)
    basis = vt[:rank]
    result = mean + ((query-mean) @ basis.T) @ basis
    return np.exp(result) if space == 'log' else result


def smooth_illumination_match(first, second, degree):
    if np.any(first <= 0) or np.any(second <= 0):
        raise ValueError('Original spectra must be positive')
    x = np.linspace(-1, 1, len(first))
    design = np.polynomial.polynomial.polyvander(x, degree)
    coefficients = np.linalg.lstsq(design, np.log(first/second), rcond=None)[0]
    illuminant = np.exp(design @ coefficients)
    return illuminant, second * illuminant


def metrics(prediction, target):
    relative = np.linalg.norm(prediction-target, axis=1)/np.linalg.norm(target, axis=1)
    return {'reflectance_rmse': float(np.sqrt(np.mean((prediction-target)**2))),
            'mean_relative_l2': float(relative.mean()), 'p95_relative_l2': float(np.quantile(relative,.95)),
            'negative_predictions': int((prediction<0).sum()), 'above_one_predictions': int((prediction>1).sum())}


def main():
    from scipy.io import loadmat
    from scipy.spatial import cKDTree
    from skin_mskcc_data import ROOT, sha
    from skin_uminho_acquire import RAW, OUT
    bench = ROOT/'docs/benchmarks/skin_spectral_probe_v1'
    lock = json.loads((bench/'source_lock.json').read_bytes())
    for name, digest in lock['sources'].items():
        assert sha(ROOT/name) == digest, name
    receipt = json.loads((OUT/'train_expand_v1.json').read_bytes())
    plan_path = RAW/'train_expand_v1.json'
    assert sha(plan_path) == receipt['local_plan_sha256']
    plan = json.loads(plan_path.read_bytes())
    assert sha(RAW/'manifest.json') == receipt['original_manifest_sha256']
    roi_path = RAW/'skin_regions_v1.json'
    assert sha(roi_path) == lock['local_roi_sha256']
    rois = json.loads(roi_path.read_bytes())
    samples=[]; means=[]; quality=[]
    for i,(row,record,regions) in enumerate(zip(plan['rows'],receipt['files'],rois,strict=True)):
        assert row['role'] == record['role'] == 'train'
        path = (RAW/row['name']).resolve()
        assert path.is_relative_to(RAW.resolve()) and sha(path) == record['sha256']
        cube = loadmat(path)['datao']
        regions_data=[]
        for x0,y0,x1,y1 in regions:
            assert 0<=x0<x1<=cube.shape[1] and 0<=y0<y1<=cube.shape[0]
            a=cube[y0:y1,x0:x1].reshape(-1,33).copy()
            assert len(a)==2500 and np.isfinite(a).all() and (a>0).all()
            regions_data.append(a); means.append(a.mean(axis=0))
        a=np.concatenate(regions_data);samples.append(a)
        quality.append({'source_index':i,'spectra':len(a),'minimum':float(a.min()),
                        'maximum':float(a.max()),'above_one_values':int((a>1).sum())})
        del cube
    rows=[]; predictions={}
    for held in range(3):
        train=np.concatenate([s for i,s in enumerate(samples) if i!=held]); target=samples[held]
        controls={'mean': np.broadcast_to(train.mean(axis=0),target.shape),
                  'nearest_spectrum':train[cKDTree(train).query(target)[1]]}
        for space in ('linear','log'):
            for rank in [1,2,3,5,8]:
                controls[f'{space}_pca{rank}']=project_spectra(train,target,rank,space)
        for name,p in controls.items():
            rows.append({'held_source_index':held,'method':name,**metrics(p,target)})
            predictions[f'fold{held}_{name}']=p
        predictions[f'fold{held}_target']=target
    ambiguities=[]
    for i in range(9):
        for j in range(i+1,9):
            first,second=means[i],means[j]
            for degree in [0,1,2,3]:
                illuminant,corrected=smooth_illumination_match(first,second,degree)
                ambiguities.append({'region_pair':[i,j], 'different_source_faces': i//3!=j//3,
                    'degree':degree,'original_relative_l2':float(np.linalg.norm(first-second)/np.linalg.norm(first)),
                    'mean_reflectance_ratio':float(first.mean()/second.mean()),
                    'radiance_relative_l2':float(np.linalg.norm(first-corrected)/np.linalg.norm(first)),
                    'illuminant_min':float(illuminant.min()),'illuminant_max':float(illuminant.max()),
                    'illuminant_max_min_ratio':float(illuminant.max()/illuminant.min())})
    predictions['region_mean_spectra']=np.array(means)
    local=ROOT/'data/processed/skin_spectral_probe_v1';local.mkdir(parents=True,exist_ok=True)
    cache=local/'oracle_arrays.npz'
    if cache.exists():
        with np.load(cache) as old:
            assert set(old.files)==set(predictions)
            for key,p in predictions.items():np.testing.assert_array_equal(old[key],p)
    else:np.savez_compressed(cache,**predictions)
    result={'scope':'MEASURED TRAIN SPECTRA; oracle compression and DERIVED RADIANCE stress, not RGB accuracy',
            'source_lock_sha256':sha(bench/'source_lock.json'),'quality':quality,'representations':rows,
            'ambiguities':ambiguities,'local_arrays_sha256':sha(cache)}
    target=bench/'summary.json'
    if target.exists():assert json.loads(target.read_bytes())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    for row in rows: print(json.dumps(row))
    print(json.dumps({'spectra':22500,'oracle_rows':len(rows),'ambiguity_rows':len(ambiguities),
                      'result_sha256':sha(target),'arrays_replayed_or_saved':len(predictions)}))


if __name__=='__main__':main()
