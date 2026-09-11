"""Exact replay, reference boundary and fixed-coverage audit of localteacher fits and frozen feature normalization."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from skin_mskcc_data import ROOT,sha
from skin_local_teacher_features import load,pack
from skin_mskcc_audit import scalar_de
from skin_capture_model import MODES
from skin_local_teacher_model import LocalTeacherColor
from skin_local_teacher_train import OUT,RUN,PROTOCOL,prediction,subset


def verify_precision_amendment():
    import ast
    archive=OUT/'pooling_precision_amendment'
    a=json.loads((archive/'amendment.json').read_bytes())
    assert sha(archive/'original_features.py')==a['original_script_sha256']
    assert sha(archive/'original_receipt.json')==a['original_receipt_sha256']
    assert sha(ROOT/'scripts/skin_local_teacher_features.py')==a['updated_script_sha256']
    assert sha(OUT/'feature_receipt.json')==a['updated_receipt_sha256']
    old=json.loads((archive/'original_receipt.json').read_bytes());new=json.loads((OUT/'feature_receipt.json').read_bytes())
    changed=[k for k in old['bindings'] if old['bindings'][k]!=new['bindings'][k]]
    assert changed==[str(Path('scripts/skin_local_teacher_features.py'))]
    old['bindings']=new['bindings'];assert old==new and a['training_started'] is False
    class RemoveManualCheck(ast.NodeTransformer):
        def visit_If(self,node):
            if any(isinstance(v,ast.Name) and v.id=='verify_pooling' for v in ast.walk(node.test)):
                node.body=[ast.Pass()]
            return self.generic_visit(node)
    before=RemoveManualCheck().visit(ast.parse((archive/'original_features.py').read_text(encoding='utf-8-sig')))
    after=RemoveManualCheck().visit(ast.parse((ROOT/'scripts/skin_local_teacher_features.py').read_text(encoding='utf-8-sig')))
    assert ast.dump(before)==ast.dump(after)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fits',type=int,choices=[12,36],default=36);args=parser.parse_args()
    torch.set_num_threads(4);tr,va=load('train'),load('validation')
    lock=json.loads((OUT/'source_lock.json').read_bytes())
    for name,digest in lock['bindings'].items():assert sha(ROOT/name)==digest,name
    verify_precision_amendment()
    records=sorted(OUT.glob('*/result.json'));assert len(records)==args.fits
    gap=0.;replays=0;comparisons=0;initials={};control_weights=0;coverage_cases=0
    for path in records:
        r=json.loads(path.read_bytes());folder=RUN/path.parent.name
        if r['protocol']=='mixed':train,selection,evaluation=tr,va,va
        else:
            c=r['protocol'][5:];train=subset(tr,tr['device']==c);selection=subset(va,va['device']==c);evaluation=subset(va,va['device']!=c)
            assert set(train['device']).isdisjoint(evaluation['device']) and set(selection['device']).isdisjoint(evaluation['device'])
        assert set(train['patient']).isdisjoint(selection['patient']) and set(train['patient']).isdisjoint(evaluation['patient'])
        initials.setdefault(r['seed'],set()).add(r['initial_state_sha256'])
        history=json.loads((path.parent/'history.json').read_bytes());assert len(history)==80
        assert int(np.argmin([h['selection_patient_mean'] for h in history]))+1==r['best_epoch']
        for label in ['best','final']:
            file=folder/(label+'.pt');assert sha(file)==r[label+'_sha256'];s=torch.load(file,weights_only=True,map_location='cpu')
            assert s['protocol_sha256']==sha(PROTOCOL)
            if label=='final' and r['arch']=='plain':
                previous=ROOT/'experiments/runs/skin_capture_v1'/f"{r['protocol']}__mixture_mse__s{r['seed']}"/'final.pt'
                original_record=ROOT/'docs/benchmarks/skin_capture_v1'/previous.parent.name/'result.json'
                assert sha(previous)==json.loads(original_record.read_bytes())['final_sha256']
                original=torch.load(previous,weights_only=True,map_location='cpu')
                assert all(torch.equal(s['state'][k],v) for k,v in original['state'].items())
                control_weights+=1
            np.testing.assert_array_equal(s['target_mean'].numpy(),train['target'].mean(0).astype(np.float32))
            np.testing.assert_array_equal(s['target_std'].numpy(),train['target'].std(0).astype(np.float32))
            teacher_values=train['teacher'].astype(np.float64)
            tm=teacher_values.mean((0,1)).astype(np.float32);ts=teacher_values.std((0,1));ts=np.where(ts<1e-8,1.,ts).astype(np.float32)
            np.testing.assert_array_equal(s['state']['teacher_mean'].numpy(),tm);np.testing.assert_array_equal(s['state']['teacher_std'].numpy(),ts)
            model=LocalTeacherColor(r['arch']).cuda().eval();model.load_state_dict(s['state']);mean=s['target_mean'].cuda();std=s['target_std'].cuda()
            assert sum(p.numel() for p in model.parameters())==r['stored_parameters'] and model.active_parameters()==r['active_head_parameters']
            assert r['required_teacher_parameters']==(0 if r['arch']=='plain' else 22056576)
            endpoints=[(selection,folder/(label+'_selection.npz'))]
            if label=='best':endpoints.append((evaluation,folder/'evaluation.npz'))
            for data,file in endpoints:
                x=torch.from_numpy(pack(data,r['arch'])).cuda();p,g,h=prediction(model,x,mean,std);saved=np.load(file)
                np.testing.assert_array_equal(saved['target'],data['target']);np.testing.assert_array_equal(saved['prediction'],p);replays+=1
                if file.name=='evaluation.npz':
                    np.testing.assert_array_equal(saved['gate'],g);np.testing.assert_array_equal(saved['hypotheses'],h)
                    e=np.array([scalar_de(a,b) for a,b in zip(p,data['target'])]);gap=max(gap,float(abs(e-saved['error']).max()),abs(float(e.mean())-r['full']['mean']));comparisons+=len(e)
                    accuracy=float((g.argmax(1)==np.array([MODES.index(str(v)) for v in data['mode']])).mean())
                    assert accuracy==r['mode_accuracy']
                    risk=np.sqrt(np.mean(np.sum((h-h.mean(1,keepdims=True))**2,axis=2),axis=1))
                    np.testing.assert_array_equal(saved['risk'],risk)
                    order=sorted(range(len(risk)),key=lambda i:(float(risk[i]),i))
                    for cr in r['uncalibrated_disagreement_coverage']:
                        n=int(np.ceil(len(e)*cr['requested_coverage']));assert n==cr['accepted']
                        ee=e[order[:n]]
                        for key,value in [('mean',ee.mean()),('median',np.median(ee)),('p95',np.quantile(ee,.95)),('above_5_fraction',(ee>5).mean()),('above_10_fraction',(ee>10).mean())]:
                            gap=max(gap,abs(float(value)-cr[key]))
                        coverage_cases+=1
            del model,x,mean,std;torch.cuda.empty_cache()
    assert all(len(v)==1 for v in initials.values()) and gap<1e-10
    result={'status':'PASS','fits':len(records),'exact_prediction_array_replays':replays,'gate_and_hypothesis_replays':len(records),
        'independent_scalar_delta_e00_comparisons':comparisons,'maximum_metric_gap':gap,
        'fitting_only_target_scales_and_same_camera_selection_checked':True,'matched_initializations':True,
        'identical_historical_control_final_weights':control_weights,'independent_coverage_rows':coverage_cases,
        'audit_script_sha256':sha(Path(__file__)),'reserved_test_or_calibration_read':False,'source_lock_sha256':sha(OUT/'source_lock.json')}
    (OUT/('audit_mixed.json' if args.fits==12 else 'audit.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':main()
