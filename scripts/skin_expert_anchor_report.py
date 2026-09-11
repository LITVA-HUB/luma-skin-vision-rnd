"""Aggregate actual skin color accuracy and explicitly labeled mode diagnostics."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_expert_anchor_train import OUT,RUN
from skin_expert_anchor import ARMS
from skin_capture_model import MODES
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write,subset


def main():
    audit=json.loads((OUT/'audit.json').read_bytes());assert audit['status']=='PASS' and audit['fits']==72
    records=[json.loads(p.read_bytes()) for p in sorted(OUT.glob('*/result.json'))];assert len(records)==72
    validation=load('validation');groups={};contrasts=[];curves=[];diagnostic_cases=0
    for protocol in ('mixed','from_SLR','from_ipod'):
        ev=validation if protocol=='mixed' else subset(validation,validation['device']!=protocol.removeprefix('from_'))
        true_mode=np.array([MODES.index(str(m)) for m in ev['mode']]);arrays={}
        for arm in ARMS:
            rr=sorted([r for r in records if r['protocol']==protocol and r['arm']==arm],key=lambda r:r['seed'])
            aa=[dict(np.load(RUN/f'{protocol}__{arm}__s{s}'/'evaluation.npz')) for s in (17,29,43)];arrays[arm]=aa
            labelled=[];mode_accuracy=[]
            for a in aa:
                np.testing.assert_array_equal(a['target'],ev['target'])
                if a['hypotheses'].shape[1]==4:
                    selected=a['hypotheses'][np.arange(len(true_mode)),true_mode]
                    labelled.append(float(np.mean([scalar_de(x,y) for x,y in zip(selected,ev['target'])])))
                    mode_accuracy.append(float(np.mean(a['gate'].argmax(1)==true_mode)));diagnostic_cases+=len(true_mode)
            groups[protocol+'/'+arm]={
                'mean':float(np.mean([r['scores']['full']['mean'] for r in rr])),
                'median':float(np.mean([r['scores']['full']['median'] for r in rr])),
                'p95':float(np.mean([r['scores']['full']['p95'] for r in rr])),
                'at80':float(np.mean([r['scores']['coverage'][3]['mean'] for r in rr])),
                'seed_means':[r['scores']['full']['mean'] for r in rr],
                'label_assisted_head_mean':float(np.mean(labelled)) if labelled else None,
                'mode_accuracy':float(np.mean(mode_accuracy)) if mode_accuracy else None,
                'coverage':[{'coverage':rr[0]['scores']['coverage'][i]['requested_coverage'],
                    'mean':float(np.mean([r['scores']['coverage'][i]['mean'] for r in rr])),
                    'p95':float(np.mean([r['scores']['coverage'][i]['p95'] for r in rr])),
                    'above10':float(np.mean([r['scores']['coverage'][i]['above_10_fraction'] for r in rr]))} for i in range(6)]}
            curve=np.mean([a['curve'] for a in aa],axis=0)
            for i,risk in enumerate(curve):curves.append({'protocol':protocol,'arm':arm,'coverage':(i+1)/len(curve),'mean_delta_e00':float(risk)})
        pairs=[(a,'baseline_raw') for a in ARMS if a!='baseline_raw']
        pairs += [(f'conditional_{aug}',f'uniform_anchor_{aug}') for aug in ('raw','paired')]
        pairs += [(f'{m}_paired',f'{m}_raw') for m in ('plain','uniform_anchor','conditional')]
        for candidate,control in pairs:
            aa,bb=arrays[candidate],arrays[control]
            for a,b in zip(aa,bb):
                np.testing.assert_array_equal(a['patient'],b['patient']);np.testing.assert_array_equal(a['risk'],b['risk'])
            difference=np.mean([a['error']-b['error'] for a,b in zip(aa,bb)],axis=0)
            patient=aa[0]['patient'];cluster=np.array([difference[patient==p].mean() for p in np.unique(patient)])
            rng=np.random.default_rng(51871);boot=cluster[rng.integers(0,len(cluster),(10000,len(cluster)))].mean(1)
            contrasts.append({'protocol':protocol,'candidate':candidate,'control':control,'mean_difference':float(difference.mean()),
                'descriptive_patient_interval':np.quantile(boot,[.025,.975]).tolist(),
                'all3seed_means_improve':all(a['error'].mean()<b['error'].mean() for a,b in zip(aa,bb))})
    summary={'scope':'Locally reproduced exploratory source data; no new independent test','groups':groups,'comparisons':contrasts,
        'label_assisted_scalar_cases':diagnostic_cases,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'audit.json',OUT/'source_lock.json']}}
    write(OUT/'summary.json',summary)
    with (OUT/'risk_coverage.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for ax,protocol in zip(axes,('mixed','from_SLR','from_ipod')):
        for arm in ARMS:
            c=[r for r in curves if r['protocol']==protocol and r['arm']==arm and r['coverage']>=.5]
            ax.plot([r['coverage']*100 for r in c],[r['mean_delta_e00'] for r in c],label=arm)
        ax.set(title=protocol,xlabel='Accepted coverage (%)',ylabel='Mean skin DeltaE00');ax.grid(alpha=.25)
    axes[-1].legend(fontsize=7);fig.suptitle('Actual instrument skin color; identical TRAIN-novelty accept sets; source development only')
    fig.tight_layout();fig.savefig(OUT/'risk_coverage.png',dpi=145);plt.close(fig)
    lines=['# Removal versus explicit expert supervision: real skin color','',
        '72 locally reproduced fits, original MSKCC photographs and actual instrument-native Lab.',
        'All errors are skin DeltaE00. Three seed scores are averaged, not ensembled.',
        'Source validation has been heavily reused. No new independent result or phone claim.','',
        '| Protocol | Arm | Mean | Median | p95 | Mean at 80% |',
        '|---|---|---:|---:|---:|---:|']
    for name,g in groups.items():
        protocol,arm=name.split('/');lines.append(f"| {protocol} | {arm} | {g['mean']:.4f} | {g['median']:.4f} | {g['p95']:.4f} | {g['at80']:.4f} |")
    lines += ['', 'p95 above averages each seed p95. Acceptance is identical across methods,',
        'using nearest TRAIN image distance of standardized mean patch statistics.',
        'This is uncalibrated input novelty, not a trained error bound or matched',
        'calibrated C+. Earlier dispersion curves use a different selection rule.','',
        '![Risk and coverage](risk_coverage.png)','', '## Matched descriptive comparisons','',
        'Patient-cluster intervals use 10,000 resamples after averaging per-image seed',
        'errors. They are exploratory, not multiplicity-adjusted or confirmatory.','',
        '| Protocol | Candidate vs control | Mean difference | Patient 95% interval | All 3 seeds |',
        '|---|---|---:|---|---|']
    for r in contrasts:
        lo,hi=r['descriptive_patient_interval'];lines.append(f"| {r['protocol']} | {r['candidate']} vs {r['control']} | {r['mean_difference']:.4f} | [{lo:.4f}, {hi:.4f}] | {r['all3seed_means_improve']} |")
    lines += ['', '## True-mode head diagnostic (labels unavailable at inference)','',
        'Select the corresponding color head using the real capture-mode label.',
        'This measures a labeled diagnostic, not deployment accuracy. Native Lab',
        'is used only for scoring. It does not select the best head using the target.','',
        '| Protocol / arm | Deployed prediction mean | Label-assisted head mean | Mode accuracy |',
        '|---|---:|---:|---:|']
    for name,g in groups.items():
        if g['label_assisted_head_mean'] is not None:lines.append(f"| {name} | {g['mean']:.4f} | {g['label_assisted_head_mean']:.4f} | {g['mode_accuracy']:.4f} |")
    lines += ['', '## Compute and evidence','',
        f"Plain contains 924,932 parameters; other arms 929,297. Maximum checkpoint {max(r['checkpoint_bytes'] for r in records):,} bytes.",
        f"Maximum fit allocation {max(r['fit_peak_allocated_mib'] for r in records):.2f} MiB; total recorded fit time {sum(r['fit_seconds'] for r in records):.1f} seconds.",
        'No new inference latency, inference VRAM, ONNX or deployment claim.',
        f"Audit: {audit['exact_color_replay_arrays']} exact color arrays, {audit['independent_scalar_color_cases']} scalar color cases, {audit['coverage_rows']} coverage rows and {audit['curve_points']} curve points.",
        f"{audit['exact_original_baseline_refits']} original baseline refits exactly match old best/final states and predictions.",
        f"{audit['independent_anchor_objective_cases']} anchor arithmetic checks; {diagnostic_cases} additional scalar label-assisted scoring cases.",
        'Shared backbone initialization, all epoch pair/plan digests, original patch provenance',
        'and source-only scale/selection are checked. No participant arrays or weights published.',
        'Original MSKCC CC-BY. Independent TEST/CAL archive unchanged; primary mean',
        '4.4570/80%4.1591 versus ordinary fusion 4.3005/4.1447. Skin product targets unmet.',
        '[Research decision](../../research/skin_expert_anchor_next_decision.md).']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'groups':{k:{z:v[z] for z in ('mean','at80','label_assisted_head_mean')} for k,v in groups.items()}}))


if __name__=='__main__':main()
