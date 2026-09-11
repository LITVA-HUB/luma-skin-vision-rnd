"""Privileged source-reference calibration diagnostic; never a production score."""
import argparse,json
from pathlib import Path
import numpy as np
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write
from skin_distribution_train import score

SOURCE=ROOT/'docs/benchmarks/skin_sampling_transfer_v1';SOURCE_RUN=ROOT/'experiments/runs/skin_sampling_transfer_v1'
OUT=ROOT/'docs/benchmarks/skin_offset_diagnostic_v1';RUN=ROOT/'experiments/runs/skin_offset_diagnostic_v1'
PROTOCOL=ROOT/'docs/research/skin_offset_diagnostic_protocol_v1.md'


def excluded_person_offsets(prediction,target,patient,site):
    p=np.asarray(prediction,dtype=float);y=np.asarray(target,dtype=float);people=np.unique(patient)
    if p.shape!=y.shape or p.shape!=(len(patient),3) or len(site)!=len(patient) or len(people)<2:raise ValueError('Invalid calibration population')
    if not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('Invalid colors')
    residual=y-p;person_means={}
    for person in people:
        ix=patient==person;values=[]
        for s in np.unique(site[ix]):
            mask=site==s
            if np.any(patient[mask]!=person):raise ValueError('Site has multiple owners')
            values.append(residual[mask].mean(0))
        person_means[person]=np.mean(values,0)
    offsets=np.empty_like(p)
    for held in people:offsets[patient==held]=np.mean([person_means[other] for other in people if other!=held],0)
    return offsets


def bindings():
    paths=[PROTOCOL,Path(__file__),ROOT/'tests/test_skin_offset_diagnostic.py',SOURCE/'audit.json',SOURCE/'source_lock.json',
        ROOT/'scripts/skin_distribution_train.py',ROOT/'scripts/skin_mskcc_summary_pilot.py',ROOT/'src/luma_skin_vision/color.py']
    paths+=sorted(SOURCE.glob('*/result.json'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'source_lock.json';bound=bindings()
    if args.stage=='prepare':
        if lock.exists():assert json.loads(lock.read_bytes())['bindings']==bound
        else:write(lock,{'bindings':bound,'input_arrays':90,'corrected_arrays':180,'strengths':[.5,1.],'privileged_reference_comparator':True})
        print('Privileged diagnostic frozen; no corrections evaluated');return
    assert json.loads(lock.read_bytes())['bindings']==bound and json.loads((SOURCE/'audit.json').read_bytes())['status']=='PASS'
    RUN.mkdir(parents=True,exist_ok=True);results=[]
    for path in sorted(SOURCE.glob('*/result.json')):
        r=json.loads(path.read_bytes())
        for domain,d in r['evaluations'].items():
            file=SOURCE_RUN/path.parent.name/(domain+'.npz');assert sha(file)==d['array_sha256']
            z=dict(np.load(file));prediction=z['prediction'];offset=excluded_person_offsets(prediction,z['target'],z['patient'],z['site'])
            ev={k:z[k] for k in ('target','patient','site')}
            ev.update(image=np.arange(len(prediction)),device=np.full(len(prediction),domain),image_type=np.full(len(prediction),'existing_source_prediction'))
            arrays={'offset':offset};metrics={}
            for strength,key in [(0.,'original'),(.5,'half'),(1.,'full')]:
                p=prediction if strength==0 else prediction+strength*offset
                scores,error,order=score(p,z['risk'],ev);scores['site_balanced_mean']=float(np.mean([error[z['site']==s].mean() for s in np.unique(z['site'])]))
                metrics[key]=scores
                if strength:
                    arrays[key+'_prediction']=p;arrays[key+'_error']=error;arrays[key+'_curve']=np.cumsum(error[order])/np.arange(1,len(error)+1)
            name=path.parent.name+'__'+domain;np.savez(RUN/(name+'.npz'),**arrays)
            results.append({'name':name,'model':path.parent.name,'protocol':r['protocol'],'arm':r['arm'],'seed':r['seed'],'domain':domain,
                'people':len(set(z['patient'])),'images':len(prediction),'reference_people_per_exclusion':len(set(z['patient']))-1,
                'metrics':metrics,'offset_rms':float(np.sqrt(np.mean(offset**2))),'input_sha256':sha(file),'output_sha256':sha(RUN/(name+'.npz'))})
    assert len(results)==90
    write(OUT/'results.json',{'scope':'PRIVILEGED REFERENCE-CALIBRATION COMPARATOR; not the single-image model','results':results,
        'reserved_endpoint_access':False,'neural_weights_changed':False,'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'source_lock.json']}})
    print(json.dumps({'completed_input_arrays':len(results),'corrected_output_arrays':2*len(results),'excluded_person_folds':sum(r['people'] for r in results)}))


if __name__=='__main__':main()
