"""TRAIN same-site/color-hard controls and48 excluded-person linear fits."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,itertools,json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from luma_skin_vision.color import delta_e00
from skin_relational_probe import matched_controls,loo_ridge,excluded_standardize,paired_preference,anchor_average,complete_potential
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_gradient_transfer_run import setup,SOURCE
from skin_pair_train import write

OUT=ROOT/'docs/benchmarks/skin_relational_probe_v1';RUN=ROOT/'experiments/runs/skin_relational_probe_v1'
PROTOCOL=ROOT/'docs/research/skin_relational_probe_protocol_v1.md'
MODELS=('mixed__image__s17','mixed__person_color__s17')


def bindings():
    paths=[PROTOCOL,Path(__file__),ROOT/'scripts/skin_relational_probe.py',ROOT/'tests/test_skin_relational_probe.py',
           ROOT/'scripts/skin_gradient_transfer_run.py',ROOT/'scripts/skin_support_curve.py',ROOT/'scripts/skin_capture_model.py',
           ROOT/'scripts/skin_mskcc_pixels.py',ROOT/'scripts/skin_mskcc_data.py',ROOT/'scripts/skin_mskcc_train_pixels.py',
           ROOT/'src/luma_skin_vision/color.py',ROOT/'data/processed/skin_mskcc_pixels_v1/train.npz',
           ROOT/'docs/benchmarks/skin_mskcc_pixels_v1/cache.json',ROOT/'docs/research/skin_mskcc_pixel_protocol_v1.md']
    paths += [SOURCE/m/'final.pt' for m in MODELS]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def context_features(name,z):
    _,model,x,y,mode,mean,std=setup(name,z);before=digest(model.state_dict());context=[];pred=[]
    hook=model.core.context.register_forward_hook(lambda module,args,out:context.append(out.detach().cpu().numpy()))
    try:
        with torch.no_grad():
            for batch in x.split(32):pred.append((model(batch,None)[0]*std+mean).cpu().numpy())
    finally:hook.remove()
    assert digest(model.state_dict())==before
    return np.concatenate(context),np.concatenate(pred)


def distance(features,pairs,is_lab):
    a,b=pairs.T
    if is_lab:return delta_e00(features[a],features[b])
    return np.sqrt(np.mean((features[a]-features[b])**2,1))


def color_summary(pred,z):
    e=delta_e00(pred,z['target'])
    return {'mean':float(e.mean()),'median':float(np.median(e)),'p95':float(np.quantile(e,.95)),
            'person_mean':float(np.mean([e[z['patient']==p].mean() for p in np.unique(z['patient'])]))},e


def correlation(x,y):
    if len(x)<2 or np.ptp(x)==0 or np.ptp(y)==0:return None
    return float(spearmanr(x,y).statistic)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'source_lock.json';bound=bindings()
    if args.stage=='prepare':
        if lock.exists():assert json.loads(lock.read_bytes())['bindings']==bound
        else:write(lock,{'bindings':bound,'ridge_fits':48,'representations':10,'original_train_only':True})
        print('Frozen TRAIN relational falsifier');return
    assert json.loads(lock.read_bytes())['bindings']==bound
    if (OUT/'results.json').exists():raise ValueError('Preserve existing outputs; run verifier instead')
    RUN.mkdir(parents=True,exist_ok=True);z=load('train');controls=matched_controls(z);positive=controls['positive'];person=z['patient'][positive[:,0]]
    pairsets={'positive':positive,**{k:np.column_stack([positive[:,0],controls[k]]) for k in ('uniform','near')}}
    between=np.array([(a,b) for p in np.unique(z['patient']) for a,b in itertools.combinations(np.flatnonzero(z['patient']==p),2) if z['site'][a]!=z['site'][b]],dtype=np.int64)
    pairsets['between']=between
    truth={k:delta_e00(z['target'][v[:,0]],z['target'][v[:,1]]) for k,v in pairsets.items()}
    assert (truth['positive']==0).all()
    np.savez(RUN/'pairs.npz',**pairsets,**{k+'_truth':v for k,v in truth.items()},candidate_counts=controls['candidate_counts'])
    features={'median_rgb':(z['median'].astype(float),False,'unfitted_descriptor'),
        'color36':(excluded_standardize(z['color'],z['patient']),False,'unfitted_descriptor'),
        'patch54':(excluded_standardize(np.concatenate([z['tokens'].mean(1),z['tokens'].max(1),z['tokens'].std(1)],1),z['patient']),False,'unfitted_descriptor')}
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for name in MODELS:
        c,p=context_features(name,z);suffix='image' if '__image__' in name else 'combined'
        features['context_'+suffix]=(excluded_standardize(c,z['patient']),False,'descriptive_encoder_trained_on_all_TRAIN')
        features['frozen_lab_'+suffix]=(p,True,'descriptive_encoder_trained_on_all_TRAIN')
        np.savez(RUN/('raw_context_'+suffix+'.npz'),context=c,prediction=p)
    fit_records=[]
    for key,source in [('ridge3','median'),('ridge36','color')]:
        p,folds=loo_ridge(z[source],z['target'],z['patient']);features[key]=(p,True,'excluded_person_TRAIN_cross_validation')
        for j,fold in enumerate(folds):
            file=RUN/f'{key}__fold{j}.npz';np.savez(file,**fold)
            fit_records.append({'representation':key,'fold':j,'file':file.name,'sha256':sha(file),'train_images':len(fold['train_indices']),'held_images':len(fold['held_indices'])})
    mean=np.stack([z['target'][z['patient']!=p].mean(0) for p in z['patient']]);features['mean_lab']=(mean,True,'excluded_person_TRAIN_cross_validation')
    results=[];reference=truth['near'];thresholds=[None,.5,1.,2.,5.]
    for key,(f,is_lab,scope) in features.items():
        d={k:distance(f,v,is_lab) for k,v in pairsets.items()};records=[]
        for control in ('uniform','near'):
            for threshold in thresholds:
                keep=np.ones(len(positive),bool) if threshold is None else truth[control]<=threshold
                record={'control':control,'maximum_reference_delta_e00':threshold}
                if keep.any():record.update(paired_preference(d['positive'][keep],d[control][keep],person[keep]))
                else:record.update({'pairs':0,'pooled_preference':None,'person_mean_preference':None,'people':0})
                records.append(record)
        by_person=[]
        for identity in np.unique(z['patient']):
            ix=z['patient'][between[:,0]]==identity;v=correlation(d['between'][ix],truth['between'][ix])
            if v is not None:by_person.append(v)
        rec={'representation':key,'scope':scope,'dimensions':f.shape[1],'distance_unit':'DeltaE00 between estimates' if is_lab else 'feature RMS, not DeltaE00',
             'comparisons':records,'between_color_spearman':correlation(d['between'],truth['between']),
             'between_color_person_mean_spearman':float(np.mean(by_person)) if by_person else None,'correlation_people':len(by_person)}
        extra={}
        if scope=='excluded_person_TRAIN_cross_validation':rec['absolute_color_error'],extra['absolute_error']=color_summary(f,z)
        file=RUN/(key+'.npz');np.savez(file,features=f,**d,**extra);rec['array_sha256']=sha(file);results.append(rec)
        print(json.dumps({'representation':key,'near_preference':records[5]['person_mean_preference'],'true_color_spearman':rec['between_color_person_mean_spearman']}),flush=True)
    # Verify the additive identity on actual frozen predictor values, without using it as improved accuracy.
    f=features['frozen_lab_image'][0];ids=np.arange(32);anchors=np.arange(32,64)
    gap=float(abs(anchor_average(f[ids,None]-f[None,anchors],z['target'][anchors])-(f[ids]+(z['target'][anchors]-f[anchors]).mean(0))).max())
    potential,residual=complete_potential(f[ids,None]-f[None,ids]);cycle_gap=float(abs(residual).max())
    assert gap<1e-12 and cycle_gap<1e-12
    counts={'people':len(set(z['patient'])),'images':len(z['patient']),'sites':len(set(z['site'])),'positive_pairs':controls['total_positive_pairs'],
        'eligible_pairs':len(positive),'unmatched':controls['unmatched'],'between_site_pairs':len(between),
        'same_site_cross_camera_pairs':int(np.sum(z['device'][positive[:,0]]!=z['device'][positive[:,1]])),
        'near_reference_quantiles':np.quantile(reference,[0,.25,.5,.75,1]).tolist(),
        'near_threshold_counts':{str(t):int(np.sum(reference<=t)) for t in thresholds[1:]},
        'unique_negative_pairs':{k:len(set(tuple(sorted(pair)) for pair in pairsets[k].tolist())) for k in ('uniform','near')}}
    assert bindings()==bound
    record={'inventory':counts,'ridge_fits':fit_records,'representations':results,'additive_identity_max_gap':gap,'cycle_potential_max_gap':cycle_gap,
            'train_only':True,'independent_accuracy_changed':False,'arrays':{p.name:sha(p) for p in RUN.glob('*.npz')},'source_lock_sha256':sha(lock)}
    write(OUT/'results.json',record);print(json.dumps({'complete':True,'inventory':counts}),flush=True)


if __name__=='__main__':main()
