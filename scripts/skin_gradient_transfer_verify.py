"""Independent batch reduction, scalar skin color and finite-update replay."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,math
from pathlib import Path
import numpy as np
import torch
from skin_gradient_transfer_run import OUT,RUN,SOURCE,MODELS,bindings,setup,values
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de
from skin_pair_train import write


def independent_grad(model,x,y,mode,indices):
    params=tuple(model.parameters());totals=[torch.zeros_like(p) for p in params]
    auxiliary=[torch.zeros_like(p) for p in params]
    for start in range(0,len(indices),17):
        ix=torch.tensor(indices[start:start+17],device=x.device)
        pred,logit,_=model(x[ix],None)
        color=((pred-y[ix])**2).sum()/(3*len(indices))
        ce=-torch.log_softmax(logit,1)[torch.arange(len(ix),device=x.device),mode[ix]].sum()/len(indices)
        for objective,dst,retain in [(color,totals,True),(ce,auxiliary,False)]:
            gs=torch.autograd.grad(objective,params,retain_graph=retain,allow_unused=True)
            for total,g in zip(dst,gs):
                if g is not None:total.add_(g)
    return torch.stack([torch.cat([p.flatten() for p in totals]),torch.cat([p.flatten() for p in auxiliary])])


def independent_step(model,g,length):
    theta=torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    torch.nn.utils.vector_to_parameters(theta-length*g/torch.linalg.vector_norm(g),model.parameters())


def main():
    assert json.loads((OUT/'source_lock.json').read_bytes())['bindings']==bindings()
    records=json.loads((OUT/'results.json').read_bytes())['models'];assert [r['model'] for r in records]==MODELS
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);train=load('train')
    scalar_cases=0;finite_cases=0;replayed=0;grams=0;restored=0;scalar_gap=0.;gradient_gap=0.;arithmetic_gap=0.;first_gap=0.;replay_gap=0.
    for r in records:
        name=r['model'];folder=RUN/name
        assert sha(SOURCE/name/'final.pt')==r['original_checkpoint_sha256']
        for f,h in r['arrays'].items():assert sha(folder/f)==h
        tr,model,x,y,mode,mean,std=setup(name,train)
        original={k:v.clone() for k,v in model.state_dict().items()};assert digest(original)==r['fp64_state_sha256']
        base=dict(np.load(folder/'base.npz'));np.testing.assert_array_equal(base['target'],tr['target'])
        np.testing.assert_array_equal(base['standardized_target'],y.cpu().numpy())
        bp,bv=values(model,x,y,mode,mean,std,tr['target'])
        np.testing.assert_array_equal(bp,base['prediction']);np.testing.assert_array_equal(bv,base['values'])
        pooled=independent_grad(model,x,y,mode,np.arange(len(x)))
        stored=np.load(folder/'pooled.npz')['gradients']
        gradient_gap=max(gradient_gap,float(np.max(abs(pooled.cpu().numpy()-stored))))
        np.testing.assert_allclose(pooled.cpu().numpy(),stored,atol=1e-10,rtol=1e-10)
        pooled_gram=(pooled@pooled.T).cpu().numpy()
        np.testing.assert_allclose(pooled_gram,r['pooled_component_gram'],atol=1e-10,rtol=1e-10)
        counts_expected=np.unique(tr['patient'],return_counts=True)[1]
        for partition in ('person','shuffle51871','shuffle51872','shuffle51873'):
            z=dict(np.load(folder/(partition+'__gram.npz')));group=z['group'];k=len(counts_expected)
            np.testing.assert_array_equal(np.bincount(group),counts_expected)
            labels=np.unique(tr['patient'],return_inverse=True)[1]
            expected=labels if partition=='person' else np.random.default_rng(int(partition[7:])).permutation(labels)
            np.testing.assert_array_equal(group,expected)
            weights=np.array([math.fsum(group==j)/len(x) for j in range(k)])
            np.testing.assert_array_equal(weights,z['weights'])
            gm=np.array([[math.fsum(base['values'][group==j,c])/np.sum(group==j) for c in range(3)] for j in range(k)])
            np.testing.assert_allclose(gm,z['base_group'],atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(weights@gm,base['values'].mean(0),atol=1e-12,rtol=1e-12)
            if partition in ('person','shuffle51871'):
                gradients=torch.stack([independent_grad(model,x,y,mode,np.flatnonzero(group==j)) for j in range(k)])
                stacked=torch.cat([gradients[:,0],gradients[:,1]])
                replayed_gram=(stacked@stacked.T).cpu().numpy();grams+=1
                gradient_gap=max(gradient_gap,float(np.max(abs(replayed_gram-z['gram']))))
                np.testing.assert_allclose(replayed_gram,z['gram'],atol=1e-9,rtol=1e-10)
                for c in range(2):np.testing.assert_allclose(weights@gradients[:,c].cpu().numpy(),stored[c],atol=1e-10,rtol=1e-10)
            for objective,coefficient in [('color',0.),('joint',.1)]:
                case=next(c for c in r['cases'] if c['partition']==partition and c['objective']==objective)
                gram=z['gram'];m=gram[:k,:k]+coefficient*(gram[:k,k:]+gram[k:,:k])+coefficient**2*gram[k:,k:]
                cosine=[];negative=0
                for i in range(k):
                    for j in range(k):
                        if i!=j:
                            cosine.append(m[i,j]/math.sqrt(m[i,i]*m[j,j]));negative+=int(m[i,j]<0)
                st=case['statistics']
                assert abs(math.fsum(cosine)/len(cosine)-st['mean_cosine'])<1e-12
                assert negative/(k*(k-1))==st['negative_pair_fraction']
                assert abs(math.sqrt(max(0.,weights@m@weights))-st['pooled_norm'])<1e-10
                for step in case['steps']:
                    s=dict(np.load(folder/step['file']));finite_cases+=1
                    p=s['prediction'];errs=np.array([scalar_de(a,b) for a,b in zip(p,tr['target'])]);scalar_cases+=len(p)
                    scalar_gap=max(scalar_gap,float(np.max(abs(errs-s['values'][:,2]))))
                    mse=np.mean(((p-mean.cpu().numpy())/std.cpu().numpy()-base['standardized_target'])**2,1)
                    arithmetic_gap=max(arithmetic_gap,float(np.max(abs(mse-s['values'][:,0]))))
                    v=np.column_stack([mse,s['values'][:,1],errs]);assert (v[:,1]>=0).all()
                    change=v-base['values'];gc=np.array([change[group==j].mean(0) for j in range(k)])
                    np.testing.assert_allclose(gc,s['group_changes'],atol=1e-11,rtol=1e-9)
                    source=step['source_index'];h=step['length'];predicted=-h*m[source]/math.sqrt(m[source,source])
                    first_gap=max(first_gap,float(np.max(abs(predicted-s['first_order']))))
                    actual=gc[:,0]+coefficient*gc[:,1]
                    assert abs(float(abs(actual-predicted).max())-step['max_first_order_remainder'])<1e-11
                    assert abs(change[:,2].mean()-step['skin_delta_e00_change_image_mean'])<1e-11
                    assert abs(change[:,0].mean()-step['color_mse_change_image_mean'])<1e-11
                    assert abs(weights@actual-step['objective_change_image_mean'])<1e-11
                    if partition=='person' and source==0 and h==.001:
                        g=gradients[source,0]+coefficient*gradients[source,1]
                        independent_step(model,g,h)
                        try:rp,rv=values(model,x,y,mode,mean,std,tr['target'])
                        finally:model.load_state_dict(original)
                        replay_gap=max(replay_gap,float(abs(rp-p).max()),float(abs(rv-s['values']).max()))
                        np.testing.assert_allclose(rp,p,atol=1e-10,rtol=1e-12)
                        np.testing.assert_allclose(rv,s['values'],atol=1e-10,rtol=1e-12)
                        assert digest(model.state_dict())==r['fp64_state_sha256'];replayed+=1
            if partition in ('person','shuffle51871'):del gradients,stacked
        be=np.array([scalar_de(a,b) for a,b in zip(base['prediction'],tr['target'])])
        scalar_gap=max(scalar_gap,float(abs(be-base['values'][:,2]).max()))
        scalar_cases+=len(base['prediction'])
        assert r['exact_restorations']==48;restored+=r['exact_restorations']
        assert digest(model.state_dict())==r['fp64_state_sha256']
        print(json.dumps({'verified_model':name,'scalar_cases':scalar_cases,'gradient_gap':gradient_gap}),flush=True)
        del model,x,y,pooled;torch.cuda.empty_cache()
    assert finite_cases==288 and replayed==12 and grams==12 and restored==288
    assert scalar_gap<1e-10 and gradient_gap<1e-9 and arithmetic_gap<1e-10 and first_gap<1e-12 and replay_gap<1e-10
    assert json.loads((OUT/'source_lock.json').read_bytes())['bindings']==bindings()
    audit={'status':'PASS','six_original_checkpoints_unchanged':True,'pooled_gradient_recomputations':6,
           'independent_group_gram_recomputations':grams,'finite_interventions':finite_cases,'finite_intervention_replays':replayed,
           'recorded_exact_restorations':restored,'scalar_color_cases':scalar_cases,'max_scalar_delta_e00_gap':scalar_gap,
           'max_gradient_or_gram_gap':gradient_gap,'max_mse_arithmetic_gap':arithmetic_gap,'max_first_order_gap':first_gap,
           'max_finite_replay_gap':replay_gap,'train_only':True,'independent_accuracy_changed':False,
           'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'results.json',OUT/'source_lock.json',ROOT/'scripts/skin_mskcc_audit.py']}}
    write(OUT/'audit.json',audit);print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
