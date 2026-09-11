"""Full known/unseen skin-color report without selecting camera-specific winners."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_sampling_transfer_train import OUT,RUN
from skin_sampling_transfer import ARMS
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write


def main():
    audit=json.loads((OUT/'audit.json').read_bytes());assert audit['status']=='PASS'
    records=[json.loads(p.read_bytes())|{'name':p.parent.name} for p in sorted(OUT.glob('*/result.json'))];assert len(records)==54
    views=[('mixed','known'),('from_SLR','known'),('from_SLR','unseen'),('from_ipod','known'),('from_ipod','unseen')]
    groups={};curves=[];comparisons=[]
    for protocol,domain in views:
        arrays={}
        for arm in ARMS:
            rr=sorted([r for r in records if r['protocol']==protocol and r['arm']==arm],key=lambda r:r['seed']);assert len(rr)==3
            aa=[dict(np.load(RUN/r['name']/(domain+'.npz'))) for r in rr];arrays[arm]=aa
            mm=[r['evaluations'][domain]['scores'] for r in rr]
            g={k:float(np.mean([m['full'][k] for m in mm])) for k in ('mean','median','p90','p95','patient_balanced_mean','above_5_fraction','above_10_fraction')}
            g.update(at80=float(np.mean([m['coverage'][3]['mean'] for m in mm])),site_balanced_mean=float(np.mean([m['site_balanced_mean'] for m in mm])),
                individual_means=[m['full']['mean'] for m in mm],individual_at80=[m['coverage'][3]['mean'] for m in mm],
                train_people=rr[0]['train_people'],train_images=rr[0]['train_images'],evaluation_people=rr[0]['evaluations'][domain]['people'],evaluation_images=rr[0]['evaluations'][domain]['images'])
            groups[f'{protocol}/{domain}/{arm}']=g
            curve=np.mean([a['curve'] for a in aa],0)
            for k,y in enumerate(curve,1):curves.append({'protocol':protocol,'domain':domain,'arm':arm,'accepted':k,'coverage':k/len(curve),'mean_delta_e00':float(y)})
        for control in ARMS[:-1]:
            a,b=arrays['person_color'],arrays[control];patient=a[0]['patient']
            for x in a+b:np.testing.assert_array_equal(patient,x['patient']);np.testing.assert_array_equal(a[0]['risk'],x['risk'])
            diff=np.mean([x['error'] for x in a],0)-np.mean([x['error'] for x in b],0)
            clusters=np.array([diff[patient==p].mean() for p in np.unique(patient)])
            rng=np.random.default_rng(51871);boot=clusters[rng.integers(0,len(clusters),(10000,len(clusters)))].mean(1)
            comparisons.append({'protocol':protocol,'domain':domain,'candidate':'person_color','control':control,'image_mean_difference':float(diff.mean()),
                'patient_mean_difference':float(clusters.mean()),'descriptive_patient_interval':np.quantile(boot,[.025,.975]).tolist(),
                'seed_image_mean_differences':[float((x['error']-y['error']).mean()) for x,y in zip(a,b)]})
    write(OUT/'summary.json',{'scope':'SOURCE exploratory, no new independent test or universal camera claim','groups':groups,'comparisons':comparisons,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'audit.json',OUT/'source_lock.json']}})
    with (OUT/'risk_coverage.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for ax,(protocol,domain) in zip(axes,[('mixed','known'),('from_SLR','unseen'),('from_ipod','unseen')]):
        for arm in ARMS:
            cc=[c for c in curves if c['protocol']==protocol and c['domain']==domain and c['arm']==arm and c['coverage']>=.5]
            ax.plot([c['coverage']*100 for c in cc],[c['mean_delta_e00'] for c in cc],label=arm)
        ax.set(title=protocol+' / '+domain,xlabel='Accepted coverage (%)',ylabel='Mean skin DeltaE00');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'risk_coverage.png',dpi=150);plt.close(fig)
    lines=['# Real skin color: allocation combination under source camera transfer','',
        '54 REPRODUCED LOCALLY fits,90 known/unseen evaluation arrays. Actual original',
        'MSKCC photographs and instrument-native Lab. No illuminant angular proxies.',
        'Original TRAIN and extensively reused source VALIDATION only; no TEST/CAL.',
        'All fits use the same core and930 optimizer updates, fixed final checkpoint.',
        'No validation-guided fitting, weights, hyperparameters or checkpoint selection.',
        'Groups average three individual model runs; they are not ensemble predictions.','',
        '| Training / evaluation / sampler | Mean | Median | p95 | Patient mean | At80% | Above10 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,g in groups.items():lines.append(f"| {key} | {g['mean']:.4f} | {g['median']:.4f} | {g['p95']:.4f} | {g['patient_balanced_mean']:.4f} | {g['at80']:.4f} | {100*g['above_10_fraction']:.2f}% |")
    lines+=['','## Main result','',
        'The combined person/color sampler gives a small mixed-source gain, but it',
        'does not carry across acquisition models. Its SLR-to-iPod error is worse than',
        'the image-uniform control, and reverse transfer loses to simpler person/site',
        'balancing. It is not a universal winner and is not promoted for deployment.',
        'Do not choose a different winning sampler per held-out camera and call the',
        'result a camera-independent system.','',
        'The combined reverse-transfer80% error is worse than full-coverage error.',
        'The shared input-novelty ranking does not provide reliable selective refusal.',
        'It is not C+ or calibrated expected color error. Full fixed100/95/90/80/70/60%',
        'coverage metrics are retained per fit and all curves in risk_coverage.csv.','',
        '![Risk and coverage](risk_coverage.png)','',
        '## Paired descriptive comparisons','',
        'Negative favors combination. Patient-balanced intervals use10,000 cluster',
        'resamples after seed averaging. Mixed has six people; single-camera domains',
        'have only three. Intervals are exploratory and not multiplicity-adjusted;',
        'they do not describe the differently weighted image-mean difference.','',
        '| Protocol / domain | Combined vs | Image difference | Patient difference | Patient interval |',
        '|---|---|---:|---:|---|']
    for d in comparisons:
        lo,hi=d['descriptive_patient_interval'];lines.append(f"| {d['protocol']}/{d['domain']} | {d['control']} | {d['image_mean_difference']:.4f} | {d['patient_mean_difference']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines+=['','## Scope and reproducibility','',
        'mixed uses24 TRAIN people/966images and6 known VALIDATION people/264images.',
        'SLR training uses8 people/323images; iPod training16people/643images. Each',
        'single-camera evaluation domain has3 people/132images. Training people/sites',
        'are disjoint from evaluation. The unseen camera is absent from all training',
        'and sampling. Camera and person composition remain confounded; these are',
        'special clinical acquisition protocols, not ordinary smartphone selfies.',
        'p95 above averages per-fit quantiles. Prior80-epoch results such as mixed3.4406,',
        'forward4.8328 and reverse4.9736 remain stronger historical references under',
        'different schedules; they are not matched930-step controls.','',
        f"{audit['exact_known_unseen_prediction_arrays']} exact color arrays; {audit['exact_full_930_step_refits']} full state refits; {audit['independent_scalar_color_cases']} independent scalar color cases; {audit['coverage_rows']} fixed-coverage rows.",
        f"{audit['independent_scalar_TRAIN_density_pairs']} scalar TRAIN density pairs and {audit['hybrid_person_mass_and_ratio_identities']} combined-mass/within-person-ratio identities.",
        f"Each model {records[0]['parameters']:,} parameters; maximum measured training allocation {max(r['fit_peak_allocated_mib'] for r in records):.2f} MiB; largest checkpoint {max(r['checkpoint_bytes'] for r in records):,} bytes.",
        'No new isolated inference latency or export claim; validation follows all fitting.',
        'Original MSKCC CC-BY. No new data/weights, cloud resources or publication.',
        'Independent MSKCC primary4.4570/80%4.1591 versus ordinary fusion4.3005/4.1447',
        'is unchanged. Skin-color measurement on arbitrary phones and innovative',
        'advantage over the strongest independent baseline remain unproven.',
        '[Research decision](../../research/skin_sampling_transfer_next_decision.md).']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'groups':{k:{z:v[z] for z in ('mean','at80')} for k,v in groups.items()}}))


if __name__=='__main__':main()
