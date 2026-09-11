"""Aggregate the fixed-budget screen without selecting a model on holdout."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_support_curve_train import OUT,RUN
from skin_pair_train import write
from skin_mskcc_data import ROOT,sha


def main():
    audit=json.loads((OUT/'audit.json').read_bytes());assert audit['status']=='PASS'
    diag=json.loads((OUT/'interventions.json').read_bytes());assert diag['status']=='PASS'
    diagnostics={x['name']:x for x in diag['models']}
    records=[json.loads(p.read_bytes())|{'name':p.parent.name} for p in sorted(OUT.glob('*/result.json'))];assert len(records)==27
    groups={};arrays={};curves=[];differences=[]
    for n in (6,12,18):
        for arm in ('baseline','statistics','pixels'):
            rr=sorted([r for r in records if r['train_people']==n and r['arm']==arm],key=lambda r:r['seed'])
            aa=[dict(np.load(RUN/r['name']/'evaluation.npz')) for r in rr];arrays[n,arm]=aa
            dd=[diagnostics[r['name']] for r in rr]
            g={'train_people':n,'arm':arm,'model_runs':3,'mean':float(np.mean([r['scores']['full']['mean'] for r in rr])),
                'patient_mean':float(np.mean([r['scores']['full']['patient_balanced_mean'] for r in rr])),
                'median':float(np.mean([r['scores']['full']['median'] for r in rr])),'p95':float(np.mean([r['scores']['full']['p95'] for r in rr])),
                'at80':float(np.mean([r['scores']['coverage'][3]['mean'] for r in rr])),
                'individual_means':[r['scores']['full']['mean'] for r in rr],
                'train_mean':float(np.mean([d['train_scores']['mean'] for d in dd])),
                'reference_palette_diagnostic_mean':float(np.mean([r['support_diagnostic_mean'] for r in rr])),
                'train_images':[r['train_images'] for r in rr],'train_sites':[r['train_sites'] for r in rr],
                'interventions':{k:{'mean':float(np.mean([d['interventions'][k]['scores']['mean'] for d in dd])),
                    'lab_rms_change':float(np.mean([d['interventions'][k]['prediction_lab_rms_change'] for d in dd]))} for k in dd[0]['interventions']}}
            groups[f'{n}/{arm}']=g
            curve=np.mean([a['curve'] for a in aa],axis=0)
            for k,value in enumerate(curve,1):curves.append({'train_people':n,'arm':arm,'accepted':k,'coverage':k/len(curve),'mean_delta_e00':float(value)})
        for candidate,control in [('pixels','statistics'),('pixels','baseline'),('statistics','baseline')]:
            a,b=arrays[n,candidate],arrays[n,control];patient=a[0]['patient']
            for z in a+b:np.testing.assert_array_equal(patient,z['patient'])
            diff=np.mean([z['error'] for z in a],0)-np.mean([z['error'] for z in b],0)
            clusters=np.array([diff[patient==p].mean() for p in np.unique(patient)])
            rng=np.random.default_rng(51871);boot=clusters[rng.integers(0,len(clusters),(10000,len(clusters)))].mean(1)
            differences.append({'train_people':n,'candidate':candidate,'control':control,'image_mean_difference':float(diff.mean()),
                'patient_mean_difference':float(clusters.mean()),'descriptive_patient_interval':np.quantile(boot,[.025,.975]).tolist()})
    write(OUT/'summary.json',{'scope':'TRAIN-only internal holdout, exploratory; not fresh external accuracy','groups':groups,'comparisons':differences,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'audit.json',OUT/'interventions.json',OUT/'source_lock.json']}})
    with (OUT/'risk_coverage.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for arm in ('baseline','statistics','pixels'):
        gg=[groups[f'{n}/{arm}'] for n in (6,12,18)]
        line=axes[0].plot([6,12,18],[g['mean'] for g in gg],marker='o',label=arm+' holdout')[0]
        axes[0].plot([6,12,18],[g['train_mean'] for g in gg],linestyle='--',color=line.get_color(),label=arm+' training')
        cc=[c for c in curves if c['train_people']==18 and c['arm']==arm and c['coverage']>=.5]
        axes[1].plot([c['coverage']*100 for c in cc],[c['mean_delta_e00'] for c in cc],label=arm)
    axes[0].set(xlabel='Training people',ylabel='Mean skin DeltaE00',title='Same 930 updates; same 6 held-out people',xticks=[6,12,18]);axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Accepted coverage (%)',ylabel='Mean skin DeltaE00',title='18-person fit; uncalibrated common ranking');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'learning_curve.png',dpi=150);plt.close(fig)
    lines=['# Actual skin color: TRAIN-person support and representation screen','',
        '27 locally reproduced fits on original public MSKCC images and native instrument Lab.',
        'Only original TRAIN was loaded. Fixed internal holdout: six people, 232 images.',
        'Nested training subsets: 6/12/18 people. All fits use exactly 930 optimizer updates.',
        'These are exploratory internal results, not the archived independent test.',
        'Reported groups average three individual runs, not ensemble predictions.','',
        '| People / model | Training mean | Holdout mean | Median | p95 | At 80% |',
        '|---|---:|---:|---:|---:|---:|']
    assert all(r['holdout_images']==232 for r in records)
    for key,g in groups.items():lines.append(f"| {key} | {g['train_mean']:.4f} | {g['mean']:.4f} | {g['median']:.4f} | {g['p95']:.4f} | {g['at80']:.4f} |")
    lines+=['','More training people help this fixed-budget screen. Learned pixel residuals do',
        'not provide a convincing advantage over the matched statistics residual or original',
        'baseline. This does not prove data size is the only bottleneck or that other pixel',
        'models cannot work. These budgets/populations differ from historical 80-epoch fits.','',
        '![Learning and selection curves](learning_curve.png)','',
        'At a fixed person count and seed, all models share identical input-novelty accept',
        'sets. Across person counts these sets may differ. Novelty is not calibrated color',
        'error; the generic dispersion-MAE field is not a reliability guarantee. All six',
        'fixed coverages are in per-fit results; full curves are in risk_coverage.csv.','',
        '## Matched descriptive differences','',
        'Negative favors candidate. Patient intervals use six clusters and 10,000 bootstrap',
        'draws after averaging model-seed errors. They describe patient-balanced differences,',
        'not image-weighted differences, and are not multiplicity-adjusted confirmation.','',
        '| People | Candidate vs control | Image difference | Patient difference | Patient interval |',
        '|---:|---|---:|---:|---|']
    for d in differences:
        lo,hi=d['descriptive_patient_interval'];lines.append(f"| {d['train_people']} | {d['candidate']} vs {d['control']} | {d['image_mean_difference']:.4f} | {d['patient_mean_difference']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines+=['','## Frozen branch and pixel-information diagnostics','',
        'Removing a trained residual is an intervention, not a retrained baseline. RGB',
        'permutation moves triples within each patch; original statistical tokens remain',
        'untouched. Mean RGB removes within-patch variation from the added encoder only.','',
        '| People / model | Intervention | Mean skin error | Prediction Lab RMS change |',
        '|---|---|---:|---:|']
    for key,g in groups.items():
        for k,d in g['interventions'].items():lines.append(f"| {key} | {k} | {d['mean']:.4f} | {d['lab_rms_change']:.4f} |")
    lines+=['','Removing adapters degrades their trained models. Yet shuffling or replacing patch',
        'pixels by their means barely changes the pixel model. Its residual is active,',
        'but useful spatial structure has not been demonstrated. Existing statistical',
        'tokens already contain the patch mean; a branch can be necessary to its trained',
        'parameterization without contributing new information or better final accuracy.','',
        'Reference-palette nearest-color mean distance decreases 2.5713 -> 2.0602 -> 1.6696',
        'as training people increase. This uses actual held-out reference colors and is',
        'strictly a support diagnostic, not image inference, a noise floor or a risk score.','',
        '## Reproducibility and scope','',
        f"{audit['exact_prediction_arrays']} exact checkpoint prediction arrays; {audit['exact_full_930_step_refits']} complete exact state refits; {audit['independent_scalar_color_cases']} scalar color cases; {audit['coverage_rows']} fixed-coverage rows.",
        f"{audit['independent_support_color_distances']} independent reference-support distances; {audit['source_patch_provenance_instances']} source patch layout checks.",
        f"Diagnostics add {diag['training_prediction_arrays']} training arrays and {diag['intervention_prediction_arrays']} intervention arrays, {diag['independent_scalar_color_cases']} scalar color cases and {diag['actual_patch_intervention_provenance_cases']} patch provenance checks.",
        f"Parameters by model: { {arm:next(r['parameters'] for r in records if r['arm']==arm) for arm in ('baseline','statistics','pixels')} }.",
        f"Maximum fit allocation {max(r['fit_peak_allocated_mib'] for r in records):.2f} MiB; largest checkpoint {max(r['checkpoint_bytes'] for r in records):,} bytes. No new isolated latency/export measurement.",
        'No test-camera holdout here. Camera identity is not an input; both acquisition',
        'families are represented in training. Within each draw, camera proportions are',
        'fixed. Participant counts and skin-color support still co-vary.','',
        'Original dataset CC-BY; no new data/weights, cloud resources or publication.',
        'Independent MSKCC remains primary4.4570/80%4.1591 vs ordinary fusion4.3005/4.1447.',
        'No demonstrated universal smartphone/facial accuracy or proposed novelty win.',
        '[Research decision](../../research/skin_support_curve_next_decision.md).']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'groups':{k:{x:v[x] for x in ('mean','train_mean','at80')} for k,v in groups.items()}}))


if __name__=='__main__':main()
