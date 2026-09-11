"""Target-informed single-material feasibility; no image prediction claim."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
from skin_material_prior import PRIOR,material_value_jacobian
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_material_train import OUT
from skin_pair_train import write
from luma_skin_vision.color import delta_e00


def main():
    tr=load('train');p=dict(np.load(PRIOR,allow_pickle=False));refs=[]
    for site in sorted(set(tr['site'])):
        values=tr['target'][tr['site']==site]
        assert np.all(values==values[0])
        refs.append(values[0])
    refs=np.stack(refs);assert len(refs)==248
    def value_jac(z):return material_value_jacobian(p['mu']+p['basis']@z,p['basis'],p['matrix'],p['white'])
    records=[];saved={}
    with threadpool_limits(limits=1):
        for bound in (1.,3.,6.):
            predictions=[];codes=[];success=[];evaluations=[]
            for target in refs:
                initial=np.linalg.lstsq(p['jacobian'].T,target-p['base'],rcond=None)[0]
                starts=[np.zeros(8),np.clip(initial,-bound+1e-8,bound-1e-8)]
                fits=[least_squares(lambda z:value_jac(z)[0]-target,start,jac=lambda z:value_jac(z)[1].T,
                    bounds=(-bound,bound),max_nfev=400,ftol=1e-10,xtol=1e-10,gtol=1e-10) for start in starts]
                best=min(fits,key=lambda f:np.dot(f.fun,f.fun))
                predictions.append(value_jac(best.x)[0]);codes.append(best.x);success.append(bool(best.success));evaluations.append(best.nfev)
            prediction=np.stack(predictions);codes=np.stack(codes);e=delta_e00(prediction,refs)
            record={'coefficient_bound':bound,'sites':len(refs),'mean_achieved_delta_e00':float(e.mean()),
                'median':float(np.median(e)),'p95':float(np.quantile(e,.95)),'max':float(e.max()),
                'fraction_below_0_01':float(np.mean(e<.01)),'solver_success_fraction':float(np.mean(success)),
                'maximum_nfev':max(evaluations),'mean_code_l2':float(np.linalg.norm(codes,axis=1).mean()),
                'fraction_any_coordinate_at_bound':float(np.mean(np.any(np.abs(codes)>bound-1e-5,axis=1)))}
            records.append(record);saved[f'b{int(bound)}_prediction']=prediction;saved[f'b{int(bound)}_code']=codes;saved[f'b{int(bound)}_error']=e
            print(json.dumps(record),flush=True)
    output=ROOT/'experiments/runs/skin_material_image_v1/feasibility_train.npz'
    np.savez_compressed(output,**saved)
    result={'scope':'POST-HOC TRAIN known-target oracle; no image accuracy, no global infeasibility proof',
        'records':records,'validation_or_reserved_targets_fit':False,'network_weights_modified':False,
        'bindings':{str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),
            ROOT/'docs/research/skin_material_feasibility_protocol_v1.md',PRIOR,
            ROOT/'scripts/skin_material_prior.py',ROOT/'data/processed/skin_mskcc_pixels_v1/train.npz',output]}}
    write(OUT/'feasibility_train.json',result)


if __name__=='__main__':main()
