"""Encoder-role/scaling audit, numeric replay and complete fit reproducibility."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from skin_crossfit_correction_train import OUT,RUN,bindings,predict,source_file,fit_core,fit_head
from skin_crossfit_correction import StableColorModel,features,ARMS
from skin_support_curve import patient_roles,SkinRepresentation
from skin_pair_train import subset,write
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de
from skin_neural_reference_verify import numeric_metrics,compare_metrics


def numpy_head(x,state):
    h=x.astype(np.float64)
    for i in (0,2,4,6):
        h=h@state[f'head.net.{i}.weight'].numpy().astype(np.float64).T+state[f'head.net.{i}.bias'].numpy()
        h=np.tanh(h) if i==6 else h/(1+np.exp(-h))
    return h


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    result=json.loads((OUT/'results.json').read_bytes());assert result['status']=='COMPLETE'
    assert bindings()==result['bindings']==json.loads((OUT/'source_lock.json').read_bytes())['bindings']
    data=load('train');counts={'inner_checkpoint_replays':0,'outer_checkpoint_replays':0,'scale_exclusion_checks':0,
                             'prediction_routing_rows':0,'full_core_refits':0,'full_head_refits':0,'numpy_heads':0,'scalar_colors':0,'coverage_rows':0,'exact_arrays':0}
    gaps={'numpy_native_prediction':0.,'scalar_color':0.}
    for record in result['records']:
        seed=record['seed'];folder=RUN/f's{seed}'
        for name,h in record['private_files'].items():assert sha(folder/name)==h
        saved_table=dict(np.load(folder/'tables.npz'));a,b=patient_roles(data['patient'],data['device'],18,seed)
        np.testing.assert_array_equal(a,saved_table['train_mask']);np.testing.assert_array_equal(b,saved_table['evaluation_mask'])
        tr,ev=subset(data,a),subset(data,b);assert set(tr['patient']).isdisjoint(ev['patient'])
        folds=np.full(len(tr['target']),-1,np.int64);rng=np.random.default_rng(917031)
        for camera,width in [('SLR',2),('ipod',4)]:
            ids=rng.permutation(np.unique(tr['patient'][tr['device']==camera]))
            for k in range(3):folds[np.isin(tr['patient'],ids[k*width:(k+1)*width])]=k
        np.testing.assert_array_equal(folds,saved_table['folds']);x=torch.from_numpy(tr['tokens']).cuda();inner_predictions=[]
        for fold,row in enumerate(record['folds']):
            included=folds!=fold;inner=subset(tr,included);assert len(set(inner['patient']))==12
            assert set(inner['patient']).isdisjoint(tr['patient'][~included])
            s=torch.load(folder/f'fold{fold}.pt',map_location='cpu',weights_only=True)
            np.testing.assert_array_equal(s['target_mean'].numpy(),inner['target'].mean(0).astype(np.float32))
            np.testing.assert_array_equal(s['target_std'].numpy(),inner['target'].std(0).astype(np.float32))
            altered=tr['target'].copy();altered[~included]+=10000
            np.testing.assert_array_equal(altered[included],inner['target']);counts['scale_exclusion_checks']+=1
            model=SkinRepresentation('baseline').cuda().eval();model.load_state_dict(s['state'])
            assert digest(model.state_dict())==row['final_state_sha256']
            p=predict(model,x,s['target_mean'].cuda(),s['target_std'].cuda());inner_predictions.append(p)
            np.testing.assert_array_equal(p,saved_table['all_inner_predictions'][fold]);counts['inner_checkpoint_replays']+=1
            if seed==17 and fold==0:
                refit,mean,std,r=fit_core(inner,seed)
                assert digest(refit.state_dict())==row['final_state_sha256'] and r['draw_sha256']==row['draw_sha256']
                counts['full_core_refits']+=1;del refit
            del model
        inner_predictions=np.stack(inner_predictions)
        for i,k in enumerate(folds):
            np.testing.assert_array_equal(saved_table['out_person'][i],inner_predictions[k,i])
            np.testing.assert_array_equal(saved_table['in_matched'][i],inner_predictions[(k+1)%3,i])
            assert folds[i]!=(k+1)%3
            counts['prediction_routing_rows']+=2
        saved=torch.load(source_file(seed),map_location='cpu',weights_only=True)
        core=SkinRepresentation('baseline').cuda().eval();core.load_state_dict(saved['state']);mean=saved['target_mean'].cuda();std=saved['target_std'].cuda()
        np.testing.assert_array_equal(predict(core,x,mean,std),saved_table['in_full'])
        vx=torch.from_numpy(ev['tokens']).cuda();base=predict(core,vx,mean,std)
        np.testing.assert_array_equal(base,np.load(source_file(seed).parent/'evaluation.npz')['prediction']);counts['outer_checkpoint_replays']+=1
        cm=tr['color'].mean(0).astype(np.float32);cs=np.maximum(tr['color'].std(0),1e-6).astype(np.float32);ym=mean.cpu().numpy();ys=std.cpu().numpy()
        for key,val in [('color_mean',cm),('color_std',cs),('target_mean',ym),('target_std',ys)]:np.testing.assert_array_equal(saved_table[key],val)
        scale=np.maximum(tr['color'].astype(np.float64).std(0),1e-6)
        risk=np.sqrt(np.min(np.mean(((ev['color'][:,None]-tr['color'][None])/scale)**2,axis=2),axis=1))
        for arm,row in record['arms'].items():
            if arm=='base':pred=base
            else:
                s=torch.load(folder/(arm+'.pt'),map_location='cpu',weights_only=True);deployed=StableColorModel().cuda().eval();deployed.load_state_dict(s['state'])
                assert sum(p.numel() for p in deployed.parameters())==1117332<=1129297
                assert sum(p.numel() for p in deployed.head.parameters())==188035
                assert digest(deployed.base.state_dict())==record['core_state_sha256']
                for key,val in [('color_mean',cm),('color_std',cs),('target_mean',ym),('target_std',ys)]:np.testing.assert_array_equal(s['state'][key].numpy(),val)
                vc=torch.from_numpy(ev['color']).cuda()
                with torch.no_grad():pred=torch.cat([deployed(t,c) for t,c in zip(vx.split(32),vc.split(32))]).cpu().numpy()
                independent=base+numpy_head(features(ev['color'],base,cm,cs,ym,ys),s['state'])*ys
                gap=float(abs(independent-pred).max());gaps['numpy_native_prediction']=max(gaps['numpy_native_prediction'],gap)
                np.testing.assert_allclose(independent,pred,atol=2e-5,rtol=1e-6);counts['numpy_heads']+=1
                e=np.array([scalar_de(p,y) for p,y in zip(saved_table[arm],tr['target'])])
                compare_metrics(numeric_metrics(e,tr['patient'],tr['site']),record['table_training_errors'][arm]);counts['scalar_colors']+=len(e)
                if seed==17:
                    inp=torch.from_numpy(features(tr['color'],saved_table[arm],cm,cs,ym,ys)).cuda()
                    target=torch.tensor((tr['target'].astype(np.float32)-ym)/ys,device='cuda')
                    head,hr=fit_head(arm,seed,inp,inp[:,36:],target)
                    assert digest(head.state_dict())==row['final_state_sha256'] and hr['draw_sha256']==row['draw_sha256']
                    counts['full_head_refits']+=1;del head
                del deployed
            got=dict(np.load(folder/(arm+'.npz')));np.testing.assert_array_equal(got['prediction'],pred);counts['exact_arrays']+=1
            for key in ('target','patient','site'):np.testing.assert_array_equal(got[key],ev[key])
            np.testing.assert_array_equal(got['risk'],risk);order=np.argsort(risk,kind='stable');np.testing.assert_array_equal(got['order'],order)
            e=np.array([scalar_de(p,y) for p,y in zip(pred,ev['target'])]);gaps['scalar_color']=max(gaps['scalar_color'],float(abs(e-got['error']).max()))
            np.testing.assert_allclose(e,got['error'],atol=2e-12,rtol=2e-12);counts['scalar_colors']+=len(e)
            compare_metrics(numeric_metrics(e,ev['patient'],ev['site']),row['evaluation']['full'])
            np.testing.assert_allclose(np.cumsum(e[order])/np.arange(1,len(e)+1),got['curve'],atol=2e-12,rtol=2e-12)
            for r in row['evaluation']['coverage']:
                n=int(np.ceil(r['coverage']*len(e)));assert n==r['accepted'];ix=order[:n]
                compare_metrics(numeric_metrics(e[ix],ev['patient'][ix],ev['site'][ix]),r);counts['coverage_rows']+=1
        print(json.dumps({'verified_seed':seed}),flush=True)
    assert counts['inner_checkpoint_replays']==9 and counts['exact_arrays']==12 and counts['coverage_rows']==72
    files=[Path(__file__),OUT/'results.json',OUT/'source_lock.json',ROOT/'scripts/skin_mskcc_audit.py',ROOT/'scripts/skin_neural_reference_verify.py']
    write(OUT/'audit.json',{'status':'PASS','counts':counts,'max_gaps':gaps,'bindings':{str(p.relative_to(ROOT)):sha(p) for p in files},'only_original_TRAIN':True})
    print(json.dumps({'status':'PASS','counts':counts,'max_gaps':gaps}),flush=True)


if __name__=='__main__':main()
