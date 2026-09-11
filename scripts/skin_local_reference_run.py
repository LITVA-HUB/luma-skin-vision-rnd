"""Six frozen systems on original-TRAIN leave-one-person-out skin color."""
import argparse,json,time
from pathlib import Path
import numpy as np
from luma_skin_vision.color import delta_e00
from skin_local_reference import METHODS,predict_bank
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_pair_train import write

OUT=ROOT/'docs/benchmarks/skin_local_reference_v1';RUN=ROOT/'experiments/runs/skin_local_reference_v1'
PROTOCOL=ROOT/'docs/research/skin_local_reference_protocol_v1.md'
PREVIOUS=ROOT/'experiments/runs/skin_relational_probe_v1'


def bindings():
    paths=[PROTOCOL,Path(__file__),ROOT/'scripts/skin_local_reference.py',ROOT/'tests/test_skin_local_reference.py',
           ROOT/'scripts/skin_mskcc_pixels.py',ROOT/'scripts/skin_mskcc_data.py',ROOT/'src/luma_skin_vision/color.py',
           ROOT/'docs/research/skin_mskcc_pixel_protocol_v1.md',ROOT/'docs/benchmarks/skin_mskcc_pixels_v1/cache.json',
           ROOT/'data/processed/skin_mskcc_pixels_v1/train.npz',PREVIOUS/'ridge36.npz',PREVIOUS/'mean_lab.npz',
           ROOT/'docs/benchmarks/skin_relational_probe_v1/results.json']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def summarize(e,person,site):
    return {'mean':float(e.mean()),'median':float(np.median(e)),'p95':float(np.quantile(e,.95)),
            'fraction_above10':float((e>10).mean()),'person_mean':float(np.mean([e[person==p].mean() for p in np.unique(person)])),
            'site_mean':float(np.mean([e[site==s].mean() for s in np.unique(site)]))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'source_lock.json';bound=bindings()
    if args.stage=='prepare':
        if lock.exists():assert json.loads(lock.read_bytes())['bindings']==bound
        else:write(lock,{'bindings':bound,'systems':METHODS,'folds':24,'train_only':True})
        print('Frozen six-system local-reference experiment');return
    assert json.loads(lock.read_bytes())['bindings']==bound
    if (OUT/'results.json').exists():raise ValueError('Results exist; preserve and verify them')
    RUN.mkdir(parents=True,exist_ok=True);z=load('train');pred={k:np.empty_like(z['target']) for k in METHODS};risk=np.empty(len(z['target']));folds=[];start=time.perf_counter()
    for j,person in enumerate(np.unique(z['patient'])):
        tr=np.flatnonzero(z['patient']!=person);ev=np.flatnonzero(z['patient']==person)
        result=predict_bank(z['color'][tr],z['target'][tr],z['color'][ev])
        for name,p in result.pop('predictions').items():pred[name][ev]=p
        risk[ev]=result['risk'];details={}
        for affinity in ('appearance','color'):
            w=result['weights_'+affinity];mass=w/len(tr)
            photo=1/(mass**2).sum(1)
            person_mass=np.column_stack([mass[:,z['patient'][tr]==p].sum(1) for p in np.unique(z['patient'][tr])])
            people=1/(person_mass**2).sum(1)
            details[affinity]={'photo_effective_min':float(photo.min()),'photo_effective_median':float(np.median(photo)),
                'people_effective_min':float(people.min()),'people_effective_median':float(np.median(people)),
                'largest_person_mass':float(person_mass.max())}
        file=RUN/f'fold{j}.npz';np.savez(file,train_indices=tr,held_indices=ev,**result)
        folds.append({'fold':j,'train_images':len(tr),'held_images':len(ev),'array_sha256':sha(file),'uniform_limit_max_gap':result['uniform_limit_max_gap'],'support':details})
    np.testing.assert_array_equal(pred['global_ridge'],np.load(PREVIOUS/'ridge36.npz')['features'])
    np.testing.assert_array_equal(pred['global_mean'],np.load(PREVIOUS/'mean_lab.npz')['features'])
    np.savez(RUN/'predictions.npz',**pred,risk=risk,target=z['target'],patient=z['patient'],site=z['site'])
    error={name:delta_e00(p,z['target']) for name,p in pred.items()};order=np.argsort(risk,kind='stable');methods=[]
    for name in METHODS:
        e=error[name];curve=np.cumsum(e[order])/np.arange(1,len(e)+1);coverage=[]
        for c in (1.,.95,.9,.8,.7,.6):
            n=int(np.ceil(c*len(e)));ix=order[:n]
            coverage.append({'coverage':c,'accepted':n,**summarize(e[ix],z['patient'][ix],z['site'][ix])})
        diff=e-error['global_ridge'];pd=np.array([diff[z['patient']==p].mean() for p in np.unique(z['patient'])])
        file=RUN/(name+'.npz');np.savez(file,error=e,curve=curve,order=order,person_differences=pd)
        row={'method':name,'full':summarize(e,z['patient'],z['site']),'coverage':coverage,'discrete_mean_risk':float(curve.mean()),
            'person_improved_vs_ridge':int((pd<0).sum()),'person_delta_mean':float(pd.mean()),'mean_delta_vs_ridge':float(diff.mean()),
            'predicted_L_outside0_100':int(((pred[name][:,0]<0)|(pred[name][:,0]>100)).sum()),'array_sha256':sha(file)}
        methods.append(row);print(json.dumps({'method':name,'mean':row['full']['mean'],'p95':row['full']['p95'],'at80':coverage[3]['mean'],'improved_people':row['person_improved_vs_ridge']}),flush=True)
    assert bindings()==bound
    record={'methods':methods,'folds':folds,'images':len(risk),'people':len(folds),'global_ridge_fits':24,'local_affine_solves':2*len(risk),
            'weighted_means':2*len(risk),'exact_previous_control_replays':2,'seconds':time.perf_counter()-start,'train_only':True,
            'independent_accuracy_changed':False,'source_lock_sha256':sha(lock),'arrays':{p.name:sha(p) for p in RUN.glob('*.npz')}}
    write(OUT/'results.json',record);print(json.dumps({'complete':True,'seconds':record['seconds']}),flush=True)


if __name__=='__main__':main()
