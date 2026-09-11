"""Replay source transfer, independent routing/numerics and complete sample refits."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
from pathlib import Path
import numpy as np
import torch
from skin_correction_transfer import OUT,RUN,bindings,source_file,active_lock
from skin_crossfit_correction import StableColorModel,features,ARMS
from skin_crossfit_correction_train import predict,fit_core,fit_head
from skin_crossfit_correction_verify import numpy_head
from skin_neural_reference_verify import numeric_metrics,compare_metrics
from skin_support_curve import SkinRepresentation
from skin_local_reference_transfer import banks
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    results=json.loads((OUT/'results.json').read_bytes());assert results['status']=='COMPLETE'
    assert results['bindings']==bindings()==json.loads(active_lock().read_bytes())['bindings']
    assert json.loads(active_lock().read_bytes())['original_source_lock_sha256']==sha(OUT/'source_lock.json')
    train,va=load('train'),load('validation');protocols={p:(t,d) for p,t,d in banks(train,va)}
    counts={'inner_core_replays':0,'original_core_domain_replays':0,'person_scale_exclusions':0,'routing_rows':0,
            'complete_core_refits':0,'complete_head_refits':0,'numpy_heads':0,'exact_arrays':0,'scalar_colors':0,'coverage_rows':0}
    gaps={'numpy_native_prediction':0.,'scalar_color':0.}
    for record in results['records']:
        protocol,seed=record['protocol'],record['seed'];name=f'{protocol}__s{seed}';folder=RUN/name;t,domains=protocols[protocol]
        for file,h in record['private_files'].items():assert sha(folder/file)==h
        assert set(t['patient']).isdisjoint(va['patient']) and set(t['site']).isdisjoint(va['site'])
        if 'unseen' in domains:assert set(t['device']).isdisjoint(va['device'][domains['unseen']])
        table=dict(np.load(folder/'tables.npz'));folds=np.full(len(t['target']),-1,np.int64)
        for camera,width,seed_role in [('SLR',2,917031),('ipod',4,917032)]:
            ids=np.random.default_rng(seed_role).permutation(np.unique(t['patient'][t['device']==camera]))
            for k in range(4):folds[np.isin(t['patient'],ids[k*width:(k+1)*width])]=k
        np.testing.assert_array_equal(folds,table['folds']);assert (folds>=0).all();x=torch.from_numpy(t['tokens']).cuda();inner_predictions=[]
        for fold,row in enumerate(record['folds']):
            included=folds!=fold;inner=subset(t,included)
            assert set(inner['patient']).isdisjoint(t['patient'][~included]) and len(set(inner['patient']))==row['train_people']
            s=torch.load(folder/f'fold{fold}.pt',map_location='cpu',weights_only=True)
            np.testing.assert_array_equal(s['target_mean'].numpy(),inner['target'].mean(0).astype(np.float32))
            np.testing.assert_array_equal(s['target_std'].numpy(),inner['target'].std(0).astype(np.float32))
            altered=t['target'].copy();altered[~included]+=10000;np.testing.assert_array_equal(altered[included],inner['target'])
            counts['person_scale_exclusions']+=1
            model=SkinRepresentation('baseline').cuda().eval();model.load_state_dict(s['state'])
            assert digest(model.state_dict())==row['final_state_sha256']
            pred=predict(model,x,s['target_mean'].cuda(),s['target_std'].cuda());inner_predictions.append(pred)
            np.testing.assert_array_equal(pred,table['all_inner_predictions'][fold]);counts['inner_core_replays']+=1
            if seed==17 and fold==0:
                refit,_,_,r=fit_core(inner,seed)
                assert digest(refit.state_dict())==row['final_state_sha256'] and r['draw_sha256']==row['draw_sha256']
                counts['complete_core_refits']+=1;del refit
            del model
        inner_predictions=np.stack(inner_predictions)
        for i,k in enumerate(folds):
            np.testing.assert_array_equal(table['out_person'][i],inner_predictions[k,i])
            np.testing.assert_array_equal(table['in_matched'][i],inner_predictions[(k+1)%4,i])
            assert k!=(k+1)%4;counts['routing_rows']+=2
        saved=torch.load(source_file(protocol,seed),map_location='cpu',weights_only=True)
        core=SkinRepresentation('baseline').cuda().eval();core.load_state_dict(saved['state']);mean=saved['target_mean'].cuda();std=saved['target_std'].cuda()
        np.testing.assert_array_equal(predict(core,x,mean,std),table['in_full'])
        vx=torch.from_numpy(va['tokens']).cuda();base=np.empty_like(va['target'],dtype=np.float32)
        for mask in domains.values():base[mask]=predict(core,torch.from_numpy(va['tokens'][mask]).cuda(),mean,std)
        for domain,mask in domains.items():
            np.testing.assert_array_equal(base[mask],np.load(source_file(protocol,seed).parent/(domain+'.npz'))['prediction'])
            counts['original_core_domain_replays']+=1
        cm=t['color'].mean(0).astype(np.float32);cs=np.maximum(t['color'].std(0),1e-6).astype(np.float32);ym=mean.cpu().numpy();ys=std.cpu().numpy()
        for key,value in [('color_mean',cm),('color_std',cs),('target_mean',ym),('target_std',ys)]:np.testing.assert_array_equal(table[key],value)
        scale=np.maximum(t['color'].astype(np.float64).std(0),1e-6)
        risk=np.sqrt(np.min(np.mean(((va['color'][:,None]-t['color'][None])/scale)**2,axis=2),axis=1))
        for arm,row in record['arms'].items():
            if arm=='base':pred=base
            else:
                s=torch.load(folder/(arm+'.pt'),map_location='cpu',weights_only=True);deployed=StableColorModel().cuda().eval();deployed.load_state_dict(s['state'])
                assert sum(v.numel() for v in deployed.parameters())==row['parameters']==1117332<=1129297
                assert sum(v.numel() for v in deployed.head.parameters())==row['head_parameters']==188035
                assert digest(deployed.base.state_dict())==record['core_state_sha256']
                assert (folder/(arm+'.pt')).stat().st_size==row['checkpoint_bytes']
                for key,value in [('color_mean',cm),('color_std',cs),('target_mean',ym),('target_std',ys)]:np.testing.assert_array_equal(s['state'][key].numpy(),value)
                vc=torch.from_numpy(va['color']).cuda()
                pred=np.empty_like(base)
                with torch.no_grad():
                    for mask in domains.values():
                        ix=torch.from_numpy(np.flatnonzero(mask)).cuda();tx,cx=vx[ix],vc[ix]
                        pred[mask]=torch.cat([deployed(v,c) for v,c in zip(tx.split(32),cx.split(32))]).cpu().numpy()
                independent=base+numpy_head(features(va['color'],base,cm,cs,ym,ys),s['state'])*ys
                gap=float(abs(independent-pred).max());gaps['numpy_native_prediction']=max(gaps['numpy_native_prediction'],gap)
                np.testing.assert_allclose(independent,pred,atol=2e-5,rtol=1e-6);counts['numpy_heads']+=1
                errors=np.array([scalar_de(a,b) for a,b in zip(table[arm],t['target'])])
                compare_metrics(numeric_metrics(errors,t['patient'],t['site']),record['table_training_errors'][arm]);counts['scalar_colors']+=len(errors)
                if seed==17 and arm=='in_matched':
                    inp=torch.from_numpy(features(t['color'],table[arm],cm,cs,ym,ys)).cuda();target=torch.tensor((t['target'].astype(np.float32)-ym)/ys,device='cuda')
                    refit,hr=fit_head(arm,seed,inp,inp[:,36:],target)
                    assert digest(refit.state_dict())==row['final_state_sha256'] and hr['draw_sha256']==row['draw_sha256']
                    counts['complete_head_refits']+=1;del refit
                del deployed
            for domain,d in row['evaluations'].items():
                mask=domains[domain];ev=subset(va,mask);array=dict(np.load(folder/f'{arm}__{domain}.npz'))
                np.testing.assert_array_equal(array['prediction'],pred[mask]);counts['exact_arrays']+=1
                for key in ('target','patient','site'):np.testing.assert_array_equal(array[key],ev[key])
                np.testing.assert_array_equal(array['risk'],risk[mask]);order=np.argsort(risk[mask],kind='stable');np.testing.assert_array_equal(array['order'],order)
                errors=np.array([scalar_de(a,b) for a,b in zip(array['prediction'],ev['target'])])
                gaps['scalar_color']=max(gaps['scalar_color'],float(abs(errors-array['error']).max()))
                np.testing.assert_allclose(errors,array['error'],atol=2e-12,rtol=2e-12);counts['scalar_colors']+=len(errors)
                compare_metrics(numeric_metrics(errors,ev['patient'],ev['site']),d['full'])
                np.testing.assert_allclose(np.cumsum(errors[order])/np.arange(1,len(errors)+1),array['curve'],atol=2e-12,rtol=2e-12)
                for c in d['coverage']:
                    n=int(np.ceil(c['coverage']*len(errors)));assert n==c['accepted'];ix=order[:n]
                    compare_metrics(numeric_metrics(errors[ix],ev['patient'][ix],ev['site'][ix]),c);counts['coverage_rows']+=1
        print(json.dumps({'verified':name}),flush=True)
    assert counts['inner_core_replays']==36 and counts['exact_arrays']==60 and counts['coverage_rows']==360
    files=[Path(__file__),OUT/'source_lock.json',active_lock(),OUT/'results.json',ROOT/'scripts/skin_crossfit_correction_verify.py',ROOT/'scripts/skin_neural_reference_verify.py',ROOT/'scripts/skin_mskcc_audit.py']
    write(OUT/'audit.json',{'status':'PASS','counts':counts,'max_gaps':gaps,'bindings':{str(p.relative_to(ROOT)):sha(p) for p in files},'new_independent_accuracy':False})
    print(json.dumps({'status':'PASS','counts':counts,'max_gaps':gaps}),flush=True)


if __name__=='__main__':main()
