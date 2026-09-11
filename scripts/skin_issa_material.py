"""Fixed oracle compression controls for real measured skin reflectance."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.skin_issa_data import OUT, PRIVATE, sha, write_json, checked_manifest
from scripts.skin_issa_color import spectral_xyz, source_lab
from luma_skin_vision.color import delta_e00

METHODS=('reflectance','density','logit')
WIDTHS=(2,3,4,6,8)


def fit_basis(x,weight):
    w=np.asarray(weight,dtype=np.float64);w=w/w.sum()
    mean=np.sum(x*w[:,None],axis=0)
    centered=x-mean
    covariance=centered.T@(centered*w[:,None])
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    basis=eigenvectors[:,np.argsort(eigenvalues)[::-1]]
    signs=np.sign(basis[np.argmax(np.abs(basis),axis=0),np.arange(basis.shape[1])])
    return mean,basis*signs


def subject_weights(people):
    count=Counter(people)
    return np.array([1./count[p] for p in people],dtype=np.float64)


def encode(x,method):
    if not np.isfinite(x).all() or np.any((x<=0)|(x>=1)):
        raise ValueError('Material transforms require measured reflectance strictly inside (0,1)')
    if method=='reflectance':return x
    if method=='density':return -np.log(x)
    if method=='logit':return np.log(x)-np.log1p(-x)
    raise ValueError(method)


def decode(x,method):
    if method=='reflectance':return x
    if method=='density':return np.exp(-x)
    if method=='logit':return np.exp(-np.logaddexp(0.,-x))
    raise ValueError(method)


def load_cache(role):
    checked_manifest()
    receipt=json.loads((OUT/f'{role}_audit.json').read_text(encoding='utf-8-sig'))
    for path,h in receipt['bindings'].items():assert sha(ROOT/path)==h,path
    assert receipt['xyz_reproduction_max_abs']<1e-10
    assert receipt['native_lab_formula_reproduction_max_abs']<1e-10
    data=dict(np.load(PRIVATE/f'{role}.npz',allow_pickle=False))
    common=(data['wavelength']>=400)&(data['wavelength']<=700)
    r=data['spectra_percent'][:,common]/100.
    valid=np.all(np.isfinite(r)&(r>0)&(r<1),axis=1)
    return data,common,r,valid


def train():
    if (OUT/'material_lock.json').exists():
        verify_lock();print('Existing material fits verified; no retraining.');return
    data,common,r,valid=load_cache('train')
    weight=subject_weights(data['subject'][valid])
    state={};started=time.perf_counter()
    for method in METHODS:
        state[method+'_mean'],state[method+'_basis']=fit_basis(encode(r[valid],method),weight)
    state['mean_control']=np.average(r[valid],axis=0,weights=weight)
    model=PRIVATE/'material_bases.npz';np.savez_compressed(model,**state)
    receipt={'role':'TRAIN only','fitting_records':int(valid.sum()),'excluded_records':int((~valid).sum()),
        'fitting_subjects':len(set(data['subject'][valid])),'methods':METHODS,'widths':WIDTHS,
        'fits':15,'additional_mean_control':True,'seconds':time.perf_counter()-started,
        'validation_decoded':False,'model_bytes':model.stat().st_size,'model_sha256':sha(model)}
    write_json(OUT/'material_fit.json',receipt)
    names=['scripts/skin_issa_material.py','tests/test_skin_issa_material.py',
        'scripts/skin_issa_audit.py','scripts/skin_issa_color.py','tests/test_skin_issa_color.py',
        'docs/research/skin_issa_material_audit_amendment_v1.md',
        'docs/benchmarks/skin_issa_v1/split_lock.json','docs/benchmarks/skin_issa_v1/train_audit.json',
        'data/processed/skin_issa_v1/train.npz','data/processed/skin_issa_v1/material_bases.npz',
        'docs/benchmarks/skin_issa_v1/material_fit.json','src/luma_skin_vision/color.py']
    write_json(OUT/'material_lock.json',{'bindings':{n:sha(ROOT/n) for n in names},
        'validation_numerical_endpoints_read':False,'reserved_endpoints_enabled':False})
    print(json.dumps(receipt))


def verify_lock():
    lock=json.loads((OUT/'material_lock.json').read_text(encoding='utf-8-sig'))
    for p,h in lock['bindings'].items():assert sha(ROOT/p)==h,p


def stats(x):
    return {'mean':float(np.mean(x)),'median':float(np.median(x)),
        'p95':float(np.quantile(x,.95)),'max':float(np.max(x))}


def evaluate():
    verify_lock()
    data,common,r,valid=load_cache('validation')
    train_data,_,train_r,train_valid=load_cache('train')
    assert not set(data['subject'])&set(train_data['subject'])
    assert data['origin'].astype(int).max()<=8
    state=dict(np.load(PRIVATE/'material_bases.npz',allow_pickle=False))
    r=r[valid];people=data['subject'][valid];origins=data['origin'][valid]
    # Only records whose entire original measured support is this common domain
    # have supplied Lab exactly corresponding to the 31-band reconstruction.
    color_mask=np.all(data['declared_support'][valid]==common[None,:],axis=1)
    target=data['lab'][valid][color_mask]
    cmf=data['cmf'][common];spd=data['spd'][common];white=data['white']
    identity=source_lab(spectral_xyz(r[color_mask]*100,cmf,spd),white)
    assert np.max(delta_e00(identity,target))<1e-9
    models=[('mean',0)]+[(m,k) for m in METHODS for k in WIDTHS]
    records=[];saved={};svd_max=0.
    # Independent weighted SVD checks covariance-eigenvector projectors.
    svd_projectors={}
    w=subject_weights(train_data['subject'][train_valid]);w=w/w.sum()
    for m in METHODS:
        x=encode(train_r[train_valid],m)
        center=np.average(x,axis=0,weights=w)
        _,_,vt=np.linalg.svd((x-center)*np.sqrt(w[:,None]),full_matrices=False)
        for k in WIDTHS:svd_projectors[(m,k)]=vt[:k].T@vt[:k]
    for method,k in models:
        name=f'{method}_{k}'
        if method=='mean':
            raw=np.broadcast_to(state['mean_control'],r.shape).copy()
        else:
            basis=state[method+'_basis'][:,:k];mean=state[method+'_mean']
            x=encode(r,method)
            raw=decode((x-mean)@basis@basis.T+mean,method)
            alt=decode((x-mean)@svd_projectors[(method,k)]+mean,method)
            svd_max=max(svd_max,float(np.max(np.abs(raw-alt))))
        pred=np.clip(raw,0,1)
        rmse=np.sqrt(np.mean((pred-r)**2,axis=1))
        rel=np.linalg.norm(pred-r,axis=1)/np.linalg.norm(r,axis=1)
        colors=source_lab(spectral_xyz(pred[color_mask]*100,cmf,spd),white)
        error=delta_e00(colors,target)
        color_people=people[color_mask];color_origins=origins[color_mask]
        person_means=[float(error[color_people==p].mean()) for p in sorted(set(color_people))]
        by_origin={str(o):stats(error[color_origins==o]) for o in sorted(set(color_origins),key=int)}
        rec={'method':method,'width':k,'validation_spectra':len(r),'color_records':int(color_mask.sum()),
            'color_subjects':len(set(color_people)),'spectral_rmse_reflectance_fraction':stats(rmse),
            'spectral_relative_l2':stats(rel),'common_support_native_delta_e00':stats(error),
            'mean_subject_delta_e00':float(np.mean(person_means)),
            'native_delta_e00_by_origin':by_origin,
            'worst_origin_mean_delta_e00':max(v['mean'] for v in by_origin.values()),
            'clipped_output_fraction':float(np.mean(pred!=raw)),
            'unclipped_spectral_rmse':stats(np.sqrt(np.mean((raw-r)**2,axis=1))),
            'decoder_coefficients':31 if method=='mean' else 31*(k+1)}
        records.append(rec);saved[name+'_spectra']=pred;saved[name+'_lab']=colors;saved[name+'_error']=error
    assert svd_max<1e-10
    prediction_file=PRIVATE/'material_validation_predictions.npz';np.savez_compressed(prediction_file,**saved)
    result={'scope':'ORACLE full-measured-spectrum compression; NOT camera/RGB/phone skin accuracy',
        'records':records,'excluded_validation_records':int((~valid).sum()),
        'independent_weighted_svd_max_spectrum_gap':svd_max,
        'reserved_numerical_endpoints_read':False,'no_validation_model_selection':True,
        'bindings':{p:sha(ROOT/p) for p in ['docs/benchmarks/skin_issa_v1/material_lock.json',
            'docs/benchmarks/skin_issa_v1/validation_audit.json',
            'data/processed/skin_issa_v1/material_validation_predictions.npz']}}
    write_json(OUT/'material_results.json',result)
    print(json.dumps({'scope':result['scope'],'svd_gap':svd_max,'records':[
        {k:rec[k] for k in ('method','width','color_records','common_support_native_delta_e00','mean_subject_delta_e00')}
        for rec in records]}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['train','evaluate'])
    if parser.parse_args().action=='train':train()
    else:evaluate()
