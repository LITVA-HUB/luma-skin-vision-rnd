"""Matched allocation results and person-mass falsifier; no endpoint selection."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_color_sampling_train import OUT,RUN
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write


def main():
    extra=ROOT/'docs/benchmarks/skin_color_sampling_mass_v1';extra_run=ROOT/'experiments/runs/skin_color_sampling_mass_v1'
    audit=json.loads((OUT/'audit.json').read_bytes());mass_audit=json.loads((extra/'audit.json').read_bytes())
    assert audit['status']==mass_audit['status']=='PASS'
    records=[]
    for root,run in [(OUT,RUN),(extra,extra_run)]:
        for p in sorted(root.glob('*/result.json')):records.append(json.loads(p.read_bytes())|{'name':p.parent.name,'run':run})
    assert len(records)==21
    arms=('image','person_site','site','color','color_ipw','person_mass','within_person_shuffle')
    groups={};arrays={};curves=[];comparisons=[]
    for arm in arms:
        rr=sorted([r for r in records if r['arm']==arm],key=lambda r:r['seed']);assert len(rr)==3
        aa=[dict(np.load(r['run']/r['name']/'evaluation.npz')) for r in rr];arrays[arm]=aa
        g={k:float(np.mean([r['scores']['full'][k] for r in rr])) for k in ('mean','median','p90','p95','patient_balanced_mean','above_5_fraction','above_10_fraction')}
        g.update(at80=float(np.mean([r['scores']['coverage'][3]['mean'] for r in rr])),site_balanced_mean=float(np.mean([r['scores']['site_balanced_mean'] for r in rr])),
            individual_means=[r['scores']['full']['mean'] for r in rr],effective_image_mass=float(np.mean([r['effective_image_mass'] for r in rr])),
            proposal_vs_uniform_site_min=min(r['proposal_vs_uniform_site_min'] for r in rr),proposal_vs_uniform_site_max=max(r['proposal_vs_uniform_site_max'] for r in rr))
        groups[arm]=g
        curve=np.mean([a['curve'] for a in aa],0)
        for k,y in enumerate(curve,1):curves.append({'arm':arm,'accepted':k,'coverage':k/len(curve),'mean_delta_e00':float(y)})
    for control in ('image','person_site','site','color_ipw','person_mass','within_person_shuffle'):
        a,b=arrays['color'],arrays[control];patient=a[0]['patient']
        for x in a+b:np.testing.assert_array_equal(x['patient'],patient);np.testing.assert_array_equal(x['risk'],a[0]['risk'])
        diff=np.mean([x['error'] for x in a],0)-np.mean([x['error'] for x in b],0)
        clusters=np.array([diff[patient==p].mean() for p in np.unique(patient)])
        rng=np.random.default_rng(51871);boot=clusters[rng.integers(0,len(clusters),(10000,len(clusters)))].mean(1)
        comparisons.append({'candidate':'color','control':control,'image_mean_difference':float(diff.mean()),'patient_mean_difference':float(clusters.mean()),
            'descriptive_patient_interval':np.quantile(boot,[.025,.975]).tolist(),
            'seed_image_mean_differences':[float((x['error']-y['error']).mean()) for x,y in zip(a,b)]})
    write(OUT/'summary.json',{'scope':'21 exploratory TRAIN/internal-holdout fits; no new independent validation','groups':groups,'comparisons':comparisons,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'audit.json',extra/'audit.json',OUT/'source_lock.json',extra/'source_lock.json']}})
    with (OUT/'risk_coverage.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    axes[0].bar(np.arange(len(arms)),[groups[a]['mean'] for a in arms]);axes[0].set_xticks(np.arange(len(arms)),arms,rotation=35,ha='right')
    axes[0].set(ylabel='Mean skin DeltaE00',title='Same data, core and 930 updates',ylim=(0,7))
    for arm in arms:
        cc=[c for c in curves if c['arm']==arm and c['coverage']>=.5]
        axes[1].plot([c['coverage']*100 for c in cc],[c['mean_delta_e00'] for c in cc],label=arm)
    axes[1].set(xlabel='Accepted coverage (%)',ylabel='Mean skin DeltaE00',title='Identical uncalibrated accept sets');axes[1].legend(fontsize=7)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'risk_coverage.png',dpi=150);plt.close(fig)
    lines=['# Native skin-color allocation: matched results and confounding controls','',
        '21 REPRODUCED LOCALLY fits: initial15 plus six subsequently frozen person-mass',
        'controls. Actual original MSKCC skin images and instrument-native Lab. Same18',
        'training people and fixed six-person/232-image internal holdout;930 updates each.',
        'Only original TRAIN loaded. This cohort was TRAIN in older experiments, so these',
        'are exploratory results, not fresh independent or unseen-phone measurements.',
        'All group values average three individual model runs, not ensemble predictions.','',
        '| Sampler | Mean | Median | p95 | Patient mean | Site mean | At80% | Above10 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a,g in groups.items():lines.append(f"| {a} | {g['mean']:.4f} | {g['median']:.4f} | {g['p95']:.4f} | {g['patient_balanced_mean']:.4f} | {g['site_balanced_mean']:.4f} | {g['at80']:.4f} | {100*g['above_10_fraction']:.2f}% |")
    lines+=['','Color allocation modestly improves the image-uniform control, but its mean',
        'is almost tied with ordinary person/site balancing. Do not describe that as a',
        'strong general accuracy or novelty win. p95 is the average of per-fit quantiles.',
        'All fixed100/95/90/80/70/60% results are retained per fit and full curves in CSV.','',
        '![Skin risk and coverage](risk_coverage.png)','',
        '## Matched differences','',
        'Negative favors color allocation. Intervals describe patient-balanced differences',
        'using six patient clusters and10,000 draws after seed averaging. They do not',
        'describe image-weighted differences, and are neither independent confirmation',
        'nor multiplicity-adjusted inference.','',
        '| Color vs control | Image difference | Patient difference | Patient interval |',
        '|---|---:|---:|---|']
    for d in comparisons:
        lo,hi=d['descriptive_patient_interval'];lines.append(f"| {d['control']} | {d['image_mean_difference']:.4f} | {d['patient_mean_difference']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines+=['','## What was removed or preserved','',
        'color_ipw uses identical color-biased draws but restores the expected site-uniform',
        'loss through bounded importance weights. It does not restore the image-uniform',
        'objective or duplicate the site sampler trajectory. The gain shrinking after',
        'correction is consistent with a changed objective contributing to the result.','',
        'person_mass preserves each person\'s total color-sampler probability but equalizes',
        'their site weights. within_person_shuffle preserves those person totals and the',
        'original site-mass multiset while changing its association with actual color.',
        'These challenge a purely person-composition explanation. Both remain post-screen',
        'controls on the same small cohort, not an independent mechanism confirmation.','',
        '## Reproducibility and limits','',
        f"{audit['exact_prediction_arrays']+mass_audit['exact_prediction_arrays']} exact color arrays; {audit['exact_old_control_states']} exact historical control states; {audit['exact_weighted_and_unweighted_full_refits']+mass_audit['exact_full_state_refits']} complete additional state refits.",
        f"{audit['scalar_evaluation_color_cases']+mass_audit['scalar_color_cases']} independent scalar color cases; {audit['scalar_training_density_pairs']+mass_audit['scalar_training_density_pairs']} scalar TRAIN density pairs;126 coverage rows.",
        f"{mass_audit['person_mass_identities']} person-mass identities; all draws/counts and importance identities audited.",
        f"Every model has {records[0]['parameters']:,} parameters; maximum fit allocation {max(r['fit_peak_allocated_mib'] for r in records):.2f} MiB; largest checkpoint {max(r['checkpoint_bytes'] for r in records):,} bytes.",
        'No new isolated inference latency, export or calibrated-risk claim. Existing',
        'input novelty supplies identical accept sets; it is not an expected-color-error',
        'guarantee. No camera labels enter the sampling weights or model input. This is',
        'known acquisition/internal person holdout, not a camera-generalization experiment.','',
        'Original dataset CC-BY; no new data, pretrained weights, paid cloud or publication.',
        'Archived independent MSKCC remains primary4.4570/80%4.1591 versus ordinary',
        'fusion4.3005/4.1447. Ordinary facial smartphone accuracy remains unvalidated.',
        '[Research decision](../../research/skin_color_sampling_next_decision.md).']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'groups':{k:{z:v[z] for z in ('mean','p95','at80')} for k,v in groups.items()},'comparisons':comparisons}))


if __name__=='__main__':main()
