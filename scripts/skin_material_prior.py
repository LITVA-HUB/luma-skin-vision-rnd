"""Freeze a measured-material prior and explicit observer/tail assumptions."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from scripts.skin_issa_material import load_cache,subject_weights,verify_lock
from scripts.skin_issa_data import sha,write_json
from luma_skin_vision.color import delta_e00

OUT=ROOT/'docs/benchmarks/skin_material_image_v1'
PRIOR=ROOT/'artifacts/skin_material_image_v1/prior.npz'
CIE=ROOT/'docs/data/provenance/cie_material_v1'


def integration_matrix(knots,wave,cmf,spd):
    identity=np.eye(len(knots))
    interpolation=np.stack([np.interp(wave,knots,row) for row in identity])
    weights=cmf*spd[:,None]
    return interpolation@weights/weights[:,1].sum()


def lab_from_xyz(xyz,white):
    q=np.asarray(xyz)/np.asarray(white);d=6/29
    f=np.where(q>d**3,np.cbrt(q),q/(3*d*d)+4/29)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],axis=-1)


def material_value_jacobian(mu,basis,matrix,white):
    r=1/(1+np.exp(-mu));xyz=r@matrix;q=xyz/white;d=6/29
    fprime=np.where(q>d**3,1/(3*np.cbrt(q)**2),1/(3*d*d))/white
    color_jac=np.array([[0.,500*fprime[0],0.],[116*fprime[1],-500*fprime[1],200*fprime[1]],[0.,0.,-200*fprime[2]]])
    return lab_from_xyz(xyz,white),basis.T@((r*(1-r))[:,None]*matrix)@color_jac


def original_cie():
    for name,meta_name in [('CIE_xyz_1964_10deg.csv','CIE_xyz_1964_10deg.csv_metadata.json'),
                           ('CIE_std_illum_D65.csv','CIE_std_illum_D65.csv_metadata_v2.json')]:
        meta=json.loads((CIE/meta_name).read_text(encoding='utf-8-sig'))
        for item in meta['checksums']:
            assert hashlib.new(item['hashMethod'],(CIE/name).read_bytes()).hexdigest()==item['checksum']
        assert meta['rightsList'][0]['rightsIdentifier']=='CC BY-SA 4.0'
    observer=np.genfromtxt(CIE/'CIE_xyz_1964_10deg.csv',delimiter=',')
    d65=np.genfromtxt(CIE/'CIE_std_illum_D65.csv',delimiter=',')
    wave=observer[:,0];cmf=observer[:,1:].copy()
    assert np.array_equal(wave,np.arange(360,831))
    assert np.array_equal(np.isnan(cmf[:,2]),wave>559)
    cmf[wave>559,2]=0.  # Column support ends559nm; metadata extrapolation is zero.
    assert np.isfinite(cmf).all() and (cmf>=0).all()
    spd=np.interp(wave,d65[:,0],d65[:,1])
    return wave,cmf,spd


def build():
    if PRIOR.exists():raise ValueError('Prior already built; verify and reuse instead of replacing')
    verify_lock();data,common,r,valid=load_cache('train')
    assert valid.all()
    basis_file=ROOT/'data/processed/skin_issa_v1/material_bases.npz'
    state=np.load(basis_file,allow_pickle=False)
    mu=state['logit_mean'];v=state['logit_basis'][:,:8]
    scores=(np.log(r)-np.log1p(-r)-mu)@v
    w=subject_weights(data['subject']);w/=w.sum()
    mean=np.sum(scores*w[:,None],axis=0)
    scale=np.sqrt(np.sum((scores-mean)**2*w[:,None],axis=0))
    assert np.abs(mean).max()<1e-12
    basis=v*scale
    wave,cmf,spd=original_cie();knots=data['wavelength'][common]
    matrix=integration_matrix(knots,wave,cmf,spd);white=matrix.sum(0)
    base,jacobian=material_value_jacobian(mu,basis,matrix,white)
    # Tail sensitivity from already-read TRAIN records; neither branch is a
    # newly measured10degree target. Full integration assumes flat residual tails.
    have39=np.sum(data['declared_support'],axis=1)==39
    knots39=data['wavelength'][:39]
    matrix39=integration_matrix(knots39,wave,cmf,spd)
    lab39=lab_from_xyz(data['spectra_percent'][have39,:39]/100@matrix39,white)
    lab31=lab_from_xyz(r[have39]@matrix,white)
    difference=delta_e00(lab31,lab39)
    full_weights=cmf*spd[:,None];outside=(wave<400)|(wave>700)
    missing_mass=full_weights[outside].sum(0)/full_weights[:,1].sum()
    PRIOR.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(PRIOR,mu=mu,basis=basis,matrix=matrix,white=white,base=base,jacobian=jacobian,latent_scale=scale)
    names=['scripts/skin_material_prior.py','tests/test_skin_material_prior.py',
        'docs/research/skin_material_image_protocol_v1.md',
        'data/processed/skin_issa_v1/train.npz','data/processed/skin_issa_v1/material_bases.npz',
        'docs/benchmarks/skin_issa_v1/material_lock.json',str(PRIOR.relative_to(ROOT)).replace('\\','/')]
    names += [str(p.relative_to(ROOT)).replace('\\','/') for p in sorted(CIE.glob('*')) if p.is_file()]
    report={'scope':'ISSA TRAIN-only prior, modelled D65/10degree color interface; MSKCC labels unchanged',
        'prior_source_records':len(r),'prior_subject_codes':len(set(data['subject'])),
        'prior_bytes':PRIOR.stat().st_size,'stored_float64_values':int(sum(a.size for a in np.load(PRIOR).values())),
        'white_xyz_y1':white.tolist(),'zero_extrapolated_cmf_zbar_values':int((wave>559).sum()),
        'missing_reflectance_extrapolation':'constant endpoints, model assumption only',
        'outside400_700_full_xyz_weight_mass':missing_mass.tolist(),
        'derived_tail_sensitivity':{'records':int(have39.sum()),'mean_delta_e00':float(difference.mean()),
            'p95':float(np.quantile(difference,.95)),'max':float(difference.max()),
            'not_a_measured10degree_accuracy_endpoint':True},
        'landing_page_cmf_md5':'cd6135a724480eb8c5e7668bae914445',
        'actual_and_metadata_cmf_md5':'6140e032f9326d88c5a0959b29b4d8f3',
        'table_license':'CC BY-SA4.0; adapted integration constants retain attribution/share-alike',
        'no_reserved_or_new_image_endpoints_read':True,
        'bindings':{p:sha(ROOT/p) for p in names}}
    write_json(OUT/'prior_receipt.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='bindings'}))


if __name__=='__main__':build()
