"""Audit person-mass controls independently from their sampler implementation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from skin_color_sampling_mass_train import OUT,RUN,bindings
from skin_color_sampling_verify import independent_probability,indices_for,refit
from skin_support_curve_verify import check_summary
from skin_support_curve_train import predict
from skin_support_curve import SkinRepresentation,patient_roles
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_mskcc_train_pixels import digest
from skin_pair_train import subset,write


def main():
    assert json.loads((OUT/'source_lock.json').read_bytes())['bindings']==bindings()
    assert json.loads((ROOT/'docs/benchmarks/skin_color_sampling_v1/audit.json').read_bytes())['status']=='PASS'
    data=load('train');paths=sorted(OUT.glob('*/result.json'));assert len(paths)==6
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    gap=0.;prob_gap=0.;cases=0;density_cases=0;identities=0;refits=0
    for path in paths:
        r=json.loads(path.read_bytes());seed=r['seed'];arm=r['arm'];folder=RUN/path.parent.name
        a,b=patient_roles(data['patient'],data['device'],18,seed);tr,ev=subset(data,a),subset(data,b)
        assert set(tr['patient']).isdisjoint(ev['patient']) and set(tr['site']).isdisjoint(ev['site'])
        assert hashlib.sha256(np.packbits(a).tobytes()+np.packbits(b).tobytes()).hexdigest()==r['role_sha256']
        q,_,rho,_,n=independent_probability(tr,'color');density_cases+=n;original=q.copy()
        sites=sorted(set(tr['site']));owners=[tr['patient'][np.flatnonzero(tr['site']==s)[0]] for s in sites]
        mass=np.array([math.fsum(q[tr['site']==s]) for s in sites]);before=mass.copy();rng=np.random.default_rng(seed+314159)
        for person in sorted(set(owners)):
            ix=np.array([i for i,p in enumerate(owners) if p==person]);values=mass[ix].copy()
            if arm=='person_mass':mass[ix]=math.fsum(values)/len(values)
            else:mass[ix]=rng.permutation(values)
        for value,site in zip(mass,sites):q[tr['site']==site]=value/sum(tr['site']==site)
        for person in np.unique(tr['patient']):
            mask=tr['patient']==person;assert abs(q[mask].sum()-original[mask].sum())<1e-12;identities+=1
        if arm=='within_person_shuffle':np.testing.assert_allclose(np.sort(before),np.sort(mass),atol=1e-14,rtol=1e-12)
        else:
            for person in set(owners):assert np.ptp(mass[np.array(owners)==person])<1e-14
        for file,key in [('final.pt','checkpoint_sha256'),('evaluation.npz','evaluation_sha256'),('sampling.npz','sampling_sha256')]:assert sha(folder/file)==r[key]
        sample=dict(np.load(folder/'sampling.npz'));prob_gap=max(prob_gap,float(np.max(abs(q-sample['proposal']))));assert prob_gap<1e-12
        np.testing.assert_array_equal(sample['correction'],np.ones(len(q)));np.testing.assert_allclose(mass,sample['site_probability'],rtol=1e-12,atol=1e-14)
        trace=hashlib.sha256();counts=np.zeros(len(q),np.int64)
        for epoch in range(1,31):
            ix=indices_for(q,arm,seed,epoch);trace.update(ix.tobytes());counts+=np.bincount(ix.ravel(),minlength=len(q))
        assert trace.hexdigest()==r['draw_sha256'];np.testing.assert_array_equal(counts,sample['counts']);assert counts.sum()==29760
        saved=torch.load(folder/'final.pt',weights_only=True,map_location='cpu');z=dict(np.load(folder/'evaluation.npz'))
        for key in ('target','patient','site'):np.testing.assert_array_equal(z[key],ev[key])
        np.testing.assert_array_equal(saved['target_mean'].numpy(),tr['target'].mean(0).astype(np.float32))
        np.testing.assert_array_equal(saved['target_std'].numpy(),tr['target'].std(0).astype(np.float32))
        torch.manual_seed(seed);model=SkinRepresentation('baseline');assert digest(model.state_dict())==r['initial_sha256']
        model.load_state_dict(saved['state']);model=model.cuda()
        p=predict(model,torch.from_numpy(ev['tokens']).cuda(),torch.from_numpy(ev['rgb']).cuda().float()/255,saved['target_mean'].cuda(),saved['target_std'].cuda())
        np.testing.assert_array_equal(p,z['prediction']);e=np.array([scalar_de(x,y) for x,y in zip(p,ev['target'])]);cases+=len(e)
        gap=max(gap,float(np.max(abs(e-z['error']))),check_summary(e,ev['patient'],ev['site'],r['scores']['full']));assert gap<1e-10
        old=np.load(ROOT/'experiments/runs/skin_color_sampling_v1'/f'color__s{seed}'/'evaluation.npz')
        np.testing.assert_array_equal(z['risk'],old['risk']);order=np.argsort(z['risk'],kind='stable')
        for c,rec in zip((1,.95,.9,.8,.7,.6),r['scores']['coverage']):
            k=math.ceil(c*len(e));ix=order[:k];assert rec['accepted']==k and rec['requested_coverage']==c
            gap=max(gap,check_summary(e[ix],ev['patient'][ix],ev['site'][ix],rec))
        np.testing.assert_allclose([math.fsum(e[order[:k]])/k for k in range(1,len(e)+1)],z['curve'],atol=1e-12,rtol=1e-12)
        assert abs(np.mean([e[ev['site']==s].mean() for s in np.unique(ev['site'])])-r['scores']['site_balanced_mean'])<1e-10
        del model;torch.cuda.empty_cache()
        if seed==17:refit(tr,arm,seed,q,np.ones(len(q)),saved);refits+=1
    result={'status':'PASS','fits':6,'exact_prediction_arrays':6,'exact_full_state_refits':refits,'scalar_color_cases':cases,
        'scalar_training_density_pairs':density_cases,'person_mass_identities':identities,'coverage_rows':36,'curve_points':cases,
        'maximum_color_gap':gap,'maximum_probability_gap':prob_gap,'reserved_endpoint_access':False,'only_original_TRAIN_loaded':True,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'source_lock.json',ROOT/'scripts/skin_color_sampling_verify.py']}}
    write(OUT/'audit.json',result);print(json.dumps(result))


if __name__=='__main__':main()
