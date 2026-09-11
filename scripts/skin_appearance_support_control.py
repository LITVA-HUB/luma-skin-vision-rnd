"""Frozen direct-color support restriction, with independent scalar audit."""
import argparse,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from skin_appearance_inverse_train import OUT,RUN,ALPHAS
from skin_color_hull_control import make_hull,project_hull
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write,rows
from skin_distribution_train import score
from skin_mskcc_summary_pilot import summarize
from luma_skin_vision.color import delta_e00

PROTOCOL=ROOT/'docs/research/skin_appearance_support_control_v1.md'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','run'));args=parser.parse_args()
    paths=[p for p in sorted(OUT.glob('*/result.json')) if json.loads(p.read_bytes())['direction']=='direct'];assert len(paths)==36
    files=[Path(__file__),PROTOCOL,ROOT/'scripts/skin_color_hull_control.py',ROOT/'tests/test_skin_color_hull_control.py',OUT/'source_lock.json',OUT/'audit.json',ROOT/'scripts/skin_mskcc_audit.py']
    files+=paths
    for p in paths:files.extend(RUN/p.parent.name/f'{name}.npz' for name in ('selection','evaluation'))
    bindings={str(p.relative_to(ROOT)):sha(p) for p in files};lock=OUT/'support_control_lock.json'
    if args.stage=='prepare':
        if lock.exists():assert json.loads(lock.read_bytes())['bindings']==bindings
        else:write(lock,{'bindings':bindings,'control_arms':['nearest_atom','hull']})
        print('Support restriction frozen before evaluation');return
    assert json.loads(lock.read_bytes())['bindings']==bindings
    train,val=load('train'),load('validation');records=[];curves=coverage=scalar=nearest_cases=0;gap=0.;max_feasibility=max_stationarity=0.;projected=0
    with threadpool_limits(1):
        for path in paths:
            r=json.loads(path.read_bytes());protocol=r['protocol'];camera=protocol.removeprefix('from_')
            t=train if protocol=='mixed' else subset(train,train['device']==camera)
            v=val if protocol=='mixed' else subset(val,val['device']==camera)
            ev=val if protocol=='mixed' else subset(val,val['device']!=camera)
            _,first=np.unique(t['site'],return_index=True);palette=t['target'][first];hull=make_hull(palette)
            saved={};result={'name':path.parent.name,'protocol':protocol,'kind':r['kind'],'degree':r['degree'],'alpha':r['alpha'],'selection':{},'metrics':{},'projection':{}}
            for role,data in (('selection',v),('evaluation',ev)):
                old=dict(np.load(RUN/path.parent.name/f'{role}.npz'));p=old['direct']
                distance=delta_e00(p[:,None],palette[None]);nearest=palette[distance.argmin(1)]
                independent=np.array([[scalar_de(a,b) for b in palette] for a in p]);nearest_cases+=independent.size
                np.testing.assert_array_equal(nearest,palette[independent.argmin(1)])
                bound,receipt=project_hull(p,hull);result['projection'][role]=receipt
                max_feasibility=max(max_feasibility,receipt['max_feasibility_violation']);max_stationarity=max(max_stationarity,receipt['max_relative_stationarity']);projected+=receipt['projected_images']
                for control,pred in (('nearest_atom',nearest),('hull',bound)):
                    saved[role+'_'+control]=pred;e=np.array([scalar_de(a,b) for a,b in zip(pred,data['target'])]);scalar+=len(e)
                    if role=='selection':result['selection'][control]=summarize(e,rows(data));continue
                    metric,errors,order=score(pred,old['common_risk'],data);metric.pop('predicted_error_mae_or_dispersion_mae');result['metrics'][control]=metric
                    saved[control+'_error']=e;gap=max(gap,float(np.max(abs(errors-e))))
                    independent_order=sorted(range(len(e)),key=lambda i:(float(old['common_risk'][i]),i))
                    for c in metric['coverage']:
                        n=int(np.ceil(c['requested_coverage']*len(e)));assert n==c['accepted'];accepted=e[independent_order[:n]]
                        for key,value in [('mean',accepted.mean()),('median',np.median(accepted)),('p95',np.quantile(accepted,.95)),('above_5_fraction',(accepted>5).mean()),('above_10_fraction',(accepted>10).mean())]:gap=max(gap,abs(float(value)-c[key]))
                        coverage+=1
                    saved[control+'_curve']=np.cumsum(e[independent_order])/np.arange(1,len(e)+1);curves+=len(e)
            file=RUN/path.parent.name/'support_control.npz';np.savez(file,**saved);result['arrays_sha256']=sha(file);records.append(result)
            print(json.dumps({'completed_control':path.parent.name}),flush=True)
    selected=[]
    for protocol in ('mixed','from_SLR','from_ipod'):
        for kind in ('rgb','stats'):
            for degree in (1,2):
                group=[r for r in records if (r['protocol'],r['kind'],r['degree'])==(protocol,kind,degree)]
                for control in ('nearest_atom','hull'):
                    r=min(group,key=lambda r:(r['selection'][control]['patient_balanced_mean'],ALPHAS.index(r['alpha'])))
                    selected.append({'protocol':protocol,'kind':kind,'degree':degree,'control':control,'name':r['name'],'alpha':r['alpha'],
                        'selection_patient_mean':r['selection'][control]['patient_balanced_mean'],'metrics':r['metrics'][control]})
    assert gap<1e-8
    write(OUT/'support_control.json',{'scope':'Post-hoc source-only support restriction; no new independent result','records':records,'selected':selected,
        'audit':{'status':'PASS','direct_models':36,'selected_endpoints':24,'independent_nearest_atom_costs':nearest_cases,'independent_scalar_color_cases':scalar,
            'coverage_rows':coverage,'curve_points':curves,'maximum_gap':gap,'projected_points_with_kkt_check':projected,
            'max_feasibility_violation':max_feasibility,'max_relative_stationarity':max_stationarity,'reserved_endpoint_access':False},
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),lock]}})
    print(json.dumps({'selected':[{k:v for k,v in r.items() if k!='metrics'}|{'mean':r['metrics']['full']['mean']} for r in selected]}))


if __name__=='__main__':main()
