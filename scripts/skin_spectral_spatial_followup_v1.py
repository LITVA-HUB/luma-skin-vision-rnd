"""Declared post-screen attack: can shared multi-region evidence resolve ambiguity?"""
import json
from pathlib import Path
import numpy as np
from skin_mskcc_data import ROOT, sha


def main():
    bench=ROOT/'docs/benchmarks/skin_spectral_probe_v1'
    summary=json.loads((bench/'summary.json').read_bytes())
    cache=ROOT/'data/processed/skin_spectral_probe_v1/oracle_arrays.npz'
    assert sha(cache)==summary['local_arrays_sha256']
    with np.load(cache) as data: means=data['region_mean_spectra'].reshape(3,3,33)
    rows=[];maximum=0.
    for first,second in [(0,1),(0,2),(1,2)]:
        a,b=means[first],means[second]
        x=np.linspace(-1,1,33)
        y=np.log(a/b).reshape(-1)
        for degree in [0,1,2,3]:
            p=np.polynomial.polynomial.polyvander(x,degree)
            designs={'shared':np.tile(p,(3,1)),
                     'shared_with_local_exposure':np.column_stack([np.repeat(np.eye(3),33,axis=0),np.tile(p[:,1:],(3,1))]),
                     'independent_regions':np.kron(np.eye(3),p)}
            for mode,design in designs.items():
                coefficient=np.linalg.lstsq(design,y,rcond=None)[0]
                independent=np.linalg.solve(design.T@design,design.T@y)
                maximum=max(maximum,float(np.max(np.abs(design@(coefficient-independent)))))
                illuminants=np.exp(design@coefficient).reshape(3,33)
                corrected=b*illuminants
                rows.append({'source_pair':[first,second],'degree':degree,'model':mode,
                    'parameters':int(design.shape[1]),
                    'pooled_radiance_relative_l2':float(np.linalg.norm(a-corrected)/np.linalg.norm(a)),
                    'per_region_radiance_relative_l2':(np.linalg.norm(a-corrected,axis=1)/np.linalg.norm(a,axis=1)).tolist(),
                    'gain_min':float(illuminants.min()),'gain_max':float(illuminants.max())})
    assert maximum<1e-12
    result={'scope':'POST-SCREEN TRAIN follow-up; same measured regional means; no new labels or pixels',
        'protocol':'All3source-face pairs x degree0/1/2/3 x shared/shared+local-exposure/independent; log-ratio least squares, pooled radianceL2 scoring. Correspondence is manually selected forehead and image-left/right cheeks, not registered physical surface points.',
        'not_a_real_illuminant_or_RGB_benchmark':True,'parent_summary_sha256':sha(bench/'summary.json'),
        'script_sha256':sha(Path(__file__)),'normal_equation_maximum_log_gain_difference':maximum,'rows':rows}
    target=bench/'spatial_followup.json'
    if target.exists():assert json.loads(target.read_bytes())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    for row in rows:
        if row['degree']==2:print(json.dumps(row))


if __name__=='__main__':main()
