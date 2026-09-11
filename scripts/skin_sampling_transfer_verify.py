"""Known/unseen source split, native-color sampling and final-state audit."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from skin_sampling_transfer_train import OUT,RUN,bindings,predict
from skin_color_sampling_verify import independent_probability,indices_for,refit
from skin_support_curve_verify import check_summary
from skin_support_curve import SkinRepresentation
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_mskcc_train_pixels import digest
from skin_pair_train import subset,write


def main():
    assert json.loads((OUT/'source_lock.json').read_bytes())['bindings']==bindings()
    train,validation=load('train'),load('validation');paths=sorted(OUT.glob('*/result.json'));assert len(paths)==54
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    probability={};rankings={};initials={};draws={};density_pairs=0;identities=0;arrays=0;cases=0;coverage=0;refits=0
    gap=0.;prob_gap=0.
    for path in paths:
        r=json.loads(path.read_bytes());protocol=r['protocol'];arm=r['arm'];seed=r['seed'];folder=RUN/path.parent.name
        if protocol=='mixed':tr=train;evaluations={'known':validation}
        else:
            camera=protocol.removeprefix('from_');assert camera in ('SLR','ipod')
            tr=subset(train,train['device']==camera)
            evaluations={'known':subset(validation,validation['device']==camera),'unseen':subset(validation,validation['device']!=camera)}
        assert set(r['evaluations'])==set(evaluations)
        assert r['train_people']==len(set(tr['patient'])) and r['train_images']==len(tr['target']) and r['train_cameras']==np.unique(tr['device']).tolist()
        base_arm='color' if arm=='person_color' else arm
        if (protocol,base_arm) not in probability:
            q,w,rho,site_q,n=independent_probability(tr,base_arm);density_pairs+=n
            probability[protocol,base_arm]=(q,w,rho,site_q)
        q,w,rho,site_q=[v.copy() for v in probability[protocol,base_arm]]
        if arm=='person_color':
            people=np.unique(tr['patient']);original=q.copy()
            for person in people:
                ix=tr['patient']==person;q[ix]=q[ix]/math.fsum(q[ix])/len(people)
                assert abs(q[ix].sum()-1/len(people))<1e-12
                np.testing.assert_allclose(q[ix]/q[ix].sum(),original[ix]/original[ix].sum(),rtol=1e-12,atol=1e-14);identities+=1
            w=np.ones(len(q))
        elif arm=='color_ipw':
            np.testing.assert_allclose(q*w,site_q,rtol=1e-12,atol=1e-14);assert w.max()<=2
        for file,key in [('final.pt','checkpoint_sha256'),('sampling.npz','sampling_sha256')]:assert sha(folder/file)==r[key]
        sample=dict(np.load(folder/'sampling.npz'));prob_gap=max(prob_gap,float(np.max(abs(q-sample['proposal']))),float(np.max(abs(w-sample['correction']))));assert prob_gap<1e-12
        np.testing.assert_allclose(rho,sample['density'],rtol=1e-12,atol=1e-14)
        site_mass=np.array([math.fsum(q[tr['site']==s]) for s in sorted(set(tr['site']))]);np.testing.assert_allclose(site_mass,sample['site_probability'],rtol=1e-12,atol=1e-14)
        assert abs(1/np.sum(q*q)-r['effective_image_mass'])<1e-9
        trace=hashlib.sha256();counts=np.zeros(len(q),np.int64)
        for epoch in range(1,31):
            ix=indices_for(q,arm,seed,epoch);trace.update(ix.tobytes());counts+=np.bincount(ix.ravel(),minlength=len(q))
        assert trace.hexdigest()==r['draw_sha256'];np.testing.assert_array_equal(counts,sample['counts']);assert counts.sum()==r['sampled_images']==29760
        assert r['optimizer_updates']==930 and r['rounds']==30 and r['validation_used_for_fitting_or_selection'] is False
        history=json.loads((path.parent/'history.json').read_bytes());assert len(history)==30 and all(set(row)=={'round','loss'} for row in history)
        draws[protocol,seed,arm]=r['draw_sha256']
        saved=torch.load(folder/'final.pt',weights_only=True,map_location='cpu');assert saved['arm']==arm and saved['protocol']==protocol and saved['seed']==seed
        np.testing.assert_array_equal(saved['target_mean'].numpy(),tr['target'].mean(0).astype(np.float32));np.testing.assert_array_equal(saved['target_std'].numpy(),tr['target'].std(0).astype(np.float32))
        torch.manual_seed(seed);model=SkinRepresentation('baseline');assert digest(model.state_dict())==r['initial_sha256']
        initials.setdefault(seed,set()).add(r['initial_sha256']);assert sum(p.numel() for p in model.parameters())==r['parameters']==929297
        model.load_state_dict(saved['state']);model=model.cuda()
        for domain,ev in evaluations.items():
            assert set(tr['patient']).isdisjoint(ev['patient']) and set(tr['site']).isdisjoint(ev['site'])
            if domain=='unseen':assert set(tr['device']).isdisjoint(ev['device'])
            else:assert set(ev['device'])<=set(tr['device'])
            rec=r['evaluations'][domain];file=folder/(domain+'.npz');assert sha(file)==rec['array_sha256']
            assert rec['people']==len(set(ev['patient'])) and rec['images']==len(ev['target']) and rec['cameras']==np.unique(ev['device']).tolist()
            z=dict(np.load(file))
            for key in ('target','patient','site'):np.testing.assert_array_equal(z[key],ev[key])
            p=predict(model,torch.from_numpy(ev['tokens']).cuda(),saved['target_mean'].cuda(),saved['target_std'].cuda());np.testing.assert_array_equal(p,z['prediction']);arrays+=1
            e=np.array([scalar_de(x,y) for x,y in zip(p,ev['target'])]);cases+=len(e);gap=max(gap,float(np.max(abs(e-z['error']))));assert gap<1e-10
            if (protocol,domain) not in rankings:
                aa=tr['tokens'].mean(1).astype(float);bb=ev['tokens'].mean(1).astype(float);scale=np.maximum(aa.std(0),1e-6)
                risk=np.array([min(np.linalg.norm((x-y)/scale) for y in aa) for x in bb]);rankings[protocol,domain]=risk
            np.testing.assert_allclose(rankings[protocol,domain],z['risk'],rtol=1e-12,atol=1e-12);order=np.argsort(z['risk'],kind='stable')
            gap=max(gap,check_summary(e,ev['patient'],ev['site'],rec['scores']['full']))
            assert abs(np.mean([e[ev['site']==s].mean() for s in np.unique(ev['site'])])-rec['scores']['site_balanced_mean'])<1e-10
            for c,cc in zip((1,.95,.9,.8,.7,.6),rec['scores']['coverage']):
                k=math.ceil(c*len(e));ix=order[:k];assert cc['accepted']==k and cc['requested_coverage']==c;coverage+=1
                gap=max(gap,check_summary(e[ix],ev['patient'][ix],ev['site'][ix],cc))
            np.testing.assert_allclose([math.fsum(e[order[:k]])/k for k in range(1,len(e)+1)],z['curve'],rtol=1e-12,atol=1e-12)
        del model;torch.cuda.empty_cache()
        if seed==17 and arm in ('image','person_color'):refit(tr,arm,seed,q,w,saved);refits+=1
    for seed in (17,29,43):assert len(initials[seed])==1
    for protocol in ('mixed','from_SLR','from_ipod'):
        for seed in (17,29,43):assert draws[protocol,seed,'color']==draws[protocol,seed,'color_ipw']
    assert arrays==90 and coverage==540
    audit={'status':'PASS','fits':54,'exact_known_unseen_prediction_arrays':arrays,'exact_full_930_step_refits':refits,'independent_scalar_color_cases':cases,
        'independent_scalar_TRAIN_density_pairs':density_pairs,'hybrid_person_mass_and_ratio_identities':identities,'coverage_rows':coverage,'curve_points':cases,
        'maximum_color_gap':gap,'maximum_probability_gap':prob_gap,'all_draw_counts_and_shared_initialization_checked':True,
        'known_unseen_cameras_and_disjoint_people_sites_checked':True,'validation_used_for_fitting_or_selection':False,'reserved_endpoint_access':False,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'source_lock.json',ROOT/'scripts/skin_color_sampling_verify.py',ROOT/'scripts/skin_support_curve_verify.py',ROOT/'scripts/skin_mskcc_audit.py']}}
    write(OUT/'audit.json',audit);print(json.dumps(audit))


if __name__=='__main__':main()
