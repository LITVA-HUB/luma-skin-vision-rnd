"""All matched support results and separate teacher-assisted routing diagnostic."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_capture_support_train import OUT,RUN
from skin_capture_support import ARMS
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write


def main():
    audit=json.loads((OUT/'audit.json').read_bytes());assert audit['status']=='PASS' and audit['fits']==45
    records=[json.loads(p.read_bytes()) for p in sorted(OUT.glob('*/result.json'))];assert len(records)==45
    probe=json.loads((OUT/'routing_probe.json').read_bytes());groups={};comparisons=[];curves=[]
    for protocol in ('mixed','from_SLR','from_ipod'):
        arrays={}
        for arm in ARMS:
            rr=sorted([r for r in records if r['protocol']==protocol and r['arm']==arm],key=lambda r:r['seed'])
            aa=[dict(np.load(RUN/f'{protocol}__{arm}__s{s}'/'evaluation.npz')) for s in (17,29,43)];arrays[arm]=aa
            groups[protocol+'/'+arm]={'mean':float(np.mean([r['scores']['full']['mean'] for r in rr])),
                'median':float(np.mean([r['scores']['full']['median'] for r in rr])),
                'p95':float(np.mean([r['scores']['full']['p95'] for r in rr])),
                'at80':float(np.mean([r['scores']['coverage'][3]['mean'] for r in rr])),
                'seed_means':[r['scores']['full']['mean'] for r in rr],
                'coverage':[{'coverage':rr[0]['scores']['coverage'][i]['requested_coverage'],
                    'mean':float(np.mean([r['scores']['coverage'][i]['mean'] for r in rr])),
                    'p95':float(np.mean([r['scores']['coverage'][i]['p95'] for r in rr])),
                    'above10':float(np.mean([r['scores']['coverage'][i]['above_10_fraction'] for r in rr]))} for i in range(6)]}
            curve=np.mean([a['curve'] for a in aa],axis=0)
            for i,risk in enumerate(curve):curves.append({'protocol':protocol,'arm':arm,'coverage':(i+1)/len(curve),'mean_delta_e00':float(risk)})
        contrasts=[('self_bootstrap','baseline'),('soft_mode_control','baseline'),('paired_union','baseline'),
            ('paired_union','soft_mode_control'),('paired_stratified','soft_mode_control'),('paired_stratified','paired_union')]
        for a,b in contrasts:
            aa,bb=arrays[a],arrays[b]
            for x,y in zip(aa,bb):np.testing.assert_array_equal(x['patient'],y['patient'])
            diff=np.mean([x['error']-y['error'] for x,y in zip(aa,bb)],axis=0)
            patient=aa[0]['patient'];cluster=np.array([diff[patient==p].mean() for p in np.unique(patient)])
            rng=np.random.default_rng(51871);boot=cluster[rng.integers(0,len(cluster),(10000,len(cluster)))].mean(1)
            comparisons.append({'protocol':protocol,'candidate':a,'control':b,'image_mean_difference':float(diff.mean()),
                'descriptive_patient_interval':np.quantile(boot,[.025,.975]).tolist(),
                'all3seed_means_improve':all(x['error'].mean()<y['error'].mean() for x,y in zip(aa,bb))})
    summary={'scope':'REPRODUCED SOURCE; no new independent evaluation','groups':groups,'comparisons':comparisons,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'audit.json',OUT/'source_lock.json',OUT/'routing_probe.json']}}
    write(OUT/'summary.json',summary)
    with (OUT/'risk_coverage.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for ax,protocol in zip(axes,('mixed','from_SLR','from_ipod')):
        for arm in ARMS:
            cc=[c for c in curves if c['protocol']==protocol and c['arm']==arm and c['coverage']>=.5]
            ax.plot([100*c['coverage'] for c in cc],[c['mean_delta_e00'] for c in cc],label=arm)
        ax.set(title=protocol,xlabel='Accepted coverage (%)',ylabel='Mean skin DeltaE00');ax.grid(alpha=.25)
    axes[-1].legend(fontsize=8);fig.suptitle('Real source photographs; observed-patch training support; uncalibrated risk')
    fig.tight_layout();fig.savefig(OUT/'risk_coverage.png',dpi=145);plt.close(fig)
    lines=['# Observed same-site patch support experiment','',
        '45locally reproduced fits. Evaluation inputs are original source photographs;',
        'targets are actual MSKCC instrument-native Lab. All errors are skin DeltaE00.',
        'This is heavily reused source development, not independent TEST or phone validation.',
        'Prefit checkpoint: `checkpoint/skin-capture-support-pretrain-2026-09-11`.','',
        'TRAIN has1,421same-site pairs with exactly equal native references, but zero',
        'cross-camera pairs. Spatial pattern similarity does not prove registration.',
        'Paired augmentation mixes exact observed patch tokens, without RGB interpolation.',
        'Mixed bags are derived training inputs, not new real photographs or measurements.','',
        '## Actual source color errors','',
        'Cells average three separate seed scores, not an ensemble. p95 is the mean',
        'of individual seed p95 values. All arms have identical inference architecture.','',
        '| Protocol | Arm | Mean | Median | p95 | At80% |',
        '|---|---|---:|---:|---:|---:|']
    for key,g in groups.items():
        protocol,arm=key.split('/')
        lines.append(f"| {protocol} | {arm} | {g['mean']:.4f} | {g['median']:.4f} | {g['p95']:.4f} | {g['at80']:.4f} |")
    lines+=['','self_bootstrap controls within-image resampling. soft_mode_control changes only',
        'the auxiliary mode target, so paired-versus-soft comparisons isolate input mixing.',
        'paired_union samples from both bags with replacement; paired_stratified preserves',
        'one observed token at each grid index, without assuming physical pixel alignment.','',
        '## Matched contrasts','',
        'Patient resampling intervals are descriptive on reused source data, not',
        'confirmatory or multiplicity-adjusted. Camera protocols also change people',
        'and capture composition. Negative difference favors the candidate.','',
        '| Protocol | Candidate vs control | Mean difference | Patient95% interval | All3seeds |',
        '|---|---|---:|---|---|']
    for c in comparisons:
        lo,hi=c['descriptive_patient_interval']
        lines.append(f"| {c['protocol']} | {c['candidate']} vs {c['control']} | {c['image_mean_difference']:.4f} | [{lo:.4f},{hi:.4f}] | {c['all3seed_means_improve']} |")
    lines+=['','![Risk and coverage](risk_coverage.png)','',
        'All six fixed coverages, p95/tails and full curves are retained. Ranking is',
        'unmodified hypothesis dispersion, not a calibrated expected-error head or',
        'a per-image guarantee. No matched calibrated C+ improvement is established.',
        'Historical ordinary pair source means3.3680/4.8747/5.9025 and training-only',
        'graph reverse4.9736 remain relevant stronger controls, with their different',
        'inference costs explicitly recorded in prior reports.','',
        '## Separate known-source TRAIN routing diagnostic','',
        'Six frozen models process identical1,421virtual TRAIN bags. Known-mode gates',
        'use the original capture labels, which are not provided at deployment.',
        'These numbers are not image-model validation or an independent result.','',
        '| Source model | Learned global | Known global | Weighted known global | Known local | Shuffled local |',
        '|---|---:|---:|---:|---:|---:|']
    for r in probe['records']:
        s=r['scores'];lines.append('| '+r['source_model']+' | '+' | '.join(f"{s[k]['mean']:.4f}" for k in ('learned_global','known_global','known_weighted_global','known_local','shuffled_local'))+' |')
    lines+=['','Supplying true patch-source mode does not rescue these frozen experts.',
        'A color hypothesis trained only through a weighted sum is not automatically',
        'an individually correct, physically identified conditional estimator. This',
        'diagnostic does not rule out a differently trained local router. True Lab',
        'was used only for scoring, not route selection. Unit counterexamples show',
        'that varying gates and pooling need not commute, not that this causes the',
        'real-data failure. Preserve this distinction.','',
        '## Compute and verification','',
        f"All models contain{records[0]['parameters']:,}parameters; largest checkpoint{max(r['checkpoint_bytes'] for r in records):,}bytes.",
        f"Maximum fit-process GPU allocation{max(r['fit_peak_allocated_mib'] for r in records):.2f}MiB.",
        'Support changes are TRAIN-only; inference still takes one64-patch image bag.',
        'No new batch1latency/inference VRAM or export claim is made.','',
        f"{audit['exact_color_replay_arrays']}exact color arrays,{audit['gate_hypothesis_sets']}gate/hypothesis sets,",
        f"{audit['independent_scalar_color_cases']}independent scalar color cases,{audit['coverage_rows']}coverage rows and{audit['curve_points']}curve points.",
        f"{audit['observed_patch_tokens_independently_checked']}observed patch tokens checked against independent explicit source/index construction.",
        'Every epoch pair/plan digest and aggregate count replays; all same-site labels,',
        'fit-only scales, shared initialization and same-camera selection are checked.',
        f"Routing diagnostic independently checks{probe['independent_scalar_cases']}scalar color cases on identical inputs.",
        'Original MSKCC CC-BY; no new external data or weights; no participant files',
        'published. Independent result remains4.4570/80%4.1591 versus ordinary',
        'fusion4.3005/4.1447. Product skin precision and universality remain unmet.','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(groups))


if __name__=='__main__':main()
