"""Explicit exclusion matrix, scalar color replay and residual energy identity."""
import json,math
from pathlib import Path
import numpy as np
from skin_offset_diagnostic import OUT,RUN,SOURCE_RUN,bindings,excluded_person_offsets
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_support_curve_verify import check_summary
from skin_pair_train import write


def main():
    assert json.loads((OUT/'source_lock.json').read_bytes())['bindings']==bindings()
    record=json.loads((OUT/'results.json').read_bytes());assert len(record['results'])==90
    scalar_cases=0;coverage=0;folds=0;identities=0;gap=0.;matrix_gap=0.;energy_gap=0.;energies=[]
    for r in record['results']:
        file=SOURCE_RUN/r['model']/(r['domain']+'.npz');assert sha(file)==r['input_sha256']
        output=RUN/(r['name']+'.npz');assert sha(output)==r['output_sha256']
        z=dict(np.load(file));saved=dict(np.load(output));people=sorted(set(z['patient']));n=len(z['prediction']);k=len(people)
        assert r['people']==k and r['reference_people_per_exclusion']==k-1
        matrix=np.zeros((n,n));weights=np.zeros(n);person_means=[]
        residual=z['target']-z['prediction']
        for person in people:
            mask=z['patient']==person;sites=sorted(set(z['site'][mask]));local=[]
            for site in sites:
                ix=np.flatnonzero(z['site']==site);assert np.all(z['patient'][ix]==person)
                weights[ix]=1/(k*len(sites)*len(ix));local.append(np.array([math.fsum(residual[ix,j])/len(ix) for j in range(3)]))
            person_means.append(np.mean(local,0))
        for person in people:
            held=np.flatnonzero(z['patient']==person);cal=np.flatnonzero(z['patient']!=person)
            matrix[np.ix_(held,cal)]=weights[cal]*k/(k-1)
            assert np.all(matrix[np.ix_(held,held)]==0);folds+=1
            changed=z['target'].copy();changed[held]+=np.array([10.,-20.,30.])
            got=excluded_person_offsets(z['prediction'],changed,z['patient'],z['site'])
            np.testing.assert_array_equal(got[held],saved['offset'][held])
        np.testing.assert_allclose(matrix.sum(1),1,rtol=1e-14,atol=1e-14)
        offset=matrix@residual;matrix_gap=max(matrix_gap,float(np.max(abs(offset-saved['offset']))));assert matrix_gap<1e-11
        np.testing.assert_array_equal(excluded_person_offsets(z['prediction'],z['target'],z['patient'],z['site']),saved['offset'])
        m=np.array(person_means);mu=m.mean(0);shared=float(np.dot(mu,mu));spread=float(np.mean(np.sum((m-mu)**2,1)))
        baseline_energy=float(weights@np.sum(residual**2,1));row={'name':r['name'],'shared_residual_energy':shared,'between_person_residual_energy':spread,'original_squared_Lab_energy':baseline_energy,'changes':{}}
        order=np.argsort(z['risk'],kind='stable')
        for strength,key in [(0.,'original'),(.5,'half'),(1.,'full')]:
            p=z['prediction'] if strength==0 else z['prediction']+strength*saved['offset']
            if strength:np.testing.assert_array_equal(p,saved[key+'_prediction'])
            e=np.array([scalar_de(a,b) for a,b in zip(p,z['target'])]);scalar_cases+=n
            expected=z['error'] if strength==0 else saved[key+'_error'];gap=max(gap,float(np.max(abs(e-expected))));assert gap<1e-10
            met=r['metrics'][key];gap=max(gap,check_summary(e,z['patient'],z['site'],met['full']))
            assert abs(np.mean([e[z['site']==s].mean() for s in np.unique(z['site'])])-met['site_balanced_mean'])<1e-10
            for c,rec in zip((1,.95,.9,.8,.7,.6),met['coverage']):
                nn=math.ceil(c*n);ix=order[:nn];assert rec['accepted']==nn and rec['requested_coverage']==c;coverage+=1
                gap=max(gap,check_summary(e[ix],z['patient'][ix],z['site'][ix],rec))
            if strength:
                np.testing.assert_allclose([math.fsum(e[order[:j]])/j for j in range(1,n+1)],saved[key+'_curve'],atol=1e-12,rtol=1e-12)
                actual=float(weights@np.sum((z['target']-p)**2,1)-baseline_energy)
                formula=(-2*strength+strength**2)*shared+(2*strength/(k-1)+strength**2/(k-1)**2)*spread
                energy_gap=max(energy_gap,abs(actual-formula));assert energy_gap<1e-9;identities+=1
                row['changes'][key]={'direct':actual,'formula':formula}
        energies.append(row)
    write(OUT/'energy_diagnostic.json',{'scope':'Descriptive squared Euclidean native-Lab identity, not DeltaE00 accuracy','rows':energies,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'docs/research/skin_offset_energy_diagnostic.md',Path(__file__),OUT/'results.json']}})
    audit={'status':'PASS','input_arrays':90,'exact_corrected_arrays':180,'excluded_person_folds_and_own_reference_interventions':folds,
        'scalar_color_cases':scalar_cases,'fixed_coverage_rows':coverage,'squared_Lab_energy_identities':identities,'maximum_color_gap':gap,
        'maximum_exclusion_matrix_gap':matrix_gap,'maximum_squared_Lab_identity_gap':energy_gap,'reserved_endpoint_access':False,
        'privileged_reference_comparator':True,'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'source_lock.json',OUT/'results.json',ROOT/'scripts/skin_mskcc_audit.py',ROOT/'scripts/skin_support_curve_verify.py']}}
    write(OUT/'audit.json',audit);print(json.dumps(audit))


if __name__=='__main__':main()
