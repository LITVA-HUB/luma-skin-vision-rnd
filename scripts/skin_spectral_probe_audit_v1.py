"""Independent scalar metrics, eigensystem projection and normal-equation checks."""
import json
import math
from pathlib import Path
import numpy as np
from skin_mskcc_data import ROOT, sha


def main():
    bench=ROOT/'docs/benchmarks/skin_spectral_probe_v1'
    summary=json.loads((bench/'summary.json').read_bytes())
    cache=ROOT/'data/processed/skin_spectral_probe_v1/oracle_arrays.npz'
    assert sha(cache)==summary['local_arrays_sha256']
    assert sha(bench/'source_lock.json')==summary['source_lock_sha256']
    maximum_metric=0.;maximum_projection=0.;maximum_illumination=0.
    with np.load(cache) as a:
        for row in summary['representations']:
            held=row['held_source_index'];method=row['method']
            target=a[f'fold{held}_target'];prediction=a[f'fold{held}_{method}']
            errors=[];squared=[]
            for p,t in zip(prediction,target,strict=True):
                numerator=math.fsum((float(x)-float(y))**2 for x,y in zip(p,t,strict=True))
                denominator=math.fsum(float(y)**2 for y in t)
                squared.append(numerator);errors.append(math.sqrt(numerator/denominator))
            independent={'reflectance_rmse':math.sqrt(math.fsum(squared)/(len(squared)*33)),
                         'mean_relative_l2':math.fsum(errors)/len(errors),
                         'p95_relative_l2':float(np.quantile(errors,.95))}
            for key,v in independent.items():maximum_metric=max(maximum_metric,abs(v-row[key]))
            train=np.concatenate([a[f'fold{i}_target'] for i in range(3) if i!=held])
            if '_pca' in method:
                space,rank=method.split('_pca');rank=int(rank)
                fit=np.log(train) if space=='log' else train
                query=np.log(target) if space=='log' else target
                mean=fit.mean(0);center=fit-mean
                eigenvalues,basis=np.linalg.eigh(center.T@center)
                basis=basis[:,np.argsort(eigenvalues)[-rank:]]
                p=mean+(query-mean)@basis@basis.T
                if space=='log':p=np.exp(p)
                maximum_projection=max(maximum_projection,float(np.max(np.abs(p-prediction))))
            elif method=='nearest_spectrum':
                for idx in [0,2499,2500,4999,5000,7499]:
                    distances=((train-target[idx])**2).sum(1)
                    np.testing.assert_allclose(((prediction[idx]-target[idx])**2).sum(),distances.min(),atol=1e-15)
        means=a['region_mean_spectra'];x=np.linspace(-1,1,33)
        for row in summary['ambiguities']:
            first,second=means[row['region_pair']];degree=row['degree']
            design=np.array([[float(t)**j for j in range(degree+1)] for t in x])
            y=np.array([math.log(float(f)/float(s)) for f,s in zip(first,second,strict=True)])
            coef=np.linalg.solve(design.T@design,design.T@y)
            illuminant=np.exp(design@coef)
            corrected=second*illuminant
            r=math.sqrt(math.fsum((float(f)-float(c))**2 for f,c in zip(first,corrected,strict=True))/math.fsum(float(f)**2 for f in first))
            for actual,expected in [(r,row['radiance_relative_l2']),
                                    (float(illuminant.min()),row['illuminant_min']),
                                    (float(illuminant.max()),row['illuminant_max'])]:
                maximum_illumination=max(maximum_illumination,abs(actual-expected))
    assert maximum_metric<1e-12 and maximum_projection<1e-9 and maximum_illumination<1e-12
    result={'status':'PASS','summary_sha256':sha(bench/'summary.json'),'audit_script_sha256':sha(Path(__file__)),
            'independent_scalar_spectrum_cases':36*7500,'metric_rows':36,'eigensystem_projection_checks':30,
            'brute_force_nearest_checks':18,'normal_equation_illumination_checks':144,
            'maximum_scalar_metric_difference':maximum_metric,'maximum_projection_difference':maximum_projection,
            'maximum_illumination_difference':maximum_illumination}
    (bench/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':main()
