"""Two independent source searches: branch combination and patch likelihood."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write


def comparison(a,b,key_a,key_b):
    patient=a[0]['patient']
    for x in a+b:np.testing.assert_array_equal(x['patient'],patient)
    difference=np.mean([x[key_a] for x in a],axis=0)-np.mean([x[key_b] for x in b],axis=0)
    clusters=np.array([difference[patient==p].mean() for p in np.unique(patient)])
    rng=np.random.default_rng(51871);boot=clusters[rng.integers(0,len(clusters),(10000,len(clusters)))].mean(1)
    return {'mean_difference':float(difference.mean()),'descriptive_patient_interval':np.quantile(boot,[.025,.975]).tolist()}


def report(name,patch=False):
    out=ROOT/'docs/benchmarks'/name;run=ROOT/'experiments/runs'/name
    audit=json.loads((out/'audit.json').read_bytes());assert audit['status']=='PASS'
    records=[json.loads(p.read_bytes())|{'name':p.parent.name} for p in sorted(out.glob('*/result.json'))]
    expected=48 if patch else 36;assert len(records)==expected
    groups={};curves=[];comparisons=[]
    for protocol in ('mixed','from_SLR','from_ipod'):
        families=sorted({f"{r['representation']}_d{r['degree']}_k{r['components']}" if patch else r['arm'] for r in records if r['protocol']==protocol})
        arrays={}
        for family in families:
            rr=sorted([r for r in records if r['protocol']==protocol and (f"{r['representation']}_d{r['degree']}_k{r['components']}" if patch else r['arm'])==family],key=lambda r:r['seed'])
            aa=[dict(np.load(run/r['name']/'evaluation.npz')) for r in rr];arrays[family]=aa
            if patch:
                for r,a in zip(rr,aa):
                    key=f"t{r['selected_strength']}";a['chosen_error']=a[key+'_error'];a['chosen_curve']=a[key+'_curve'];a['chosen_posterior_curve']=a[key+'_posterior_curve']
                mm=[r['metrics'][f"t{r['selected_strength']}"]['common'] for r in rr]
            else:mm=[r['scores'] for r in rr]
            group={'initializations':len(rr),'mean':float(np.mean([m['full']['mean'] for m in mm])),
                'median':float(np.mean([m['full']['median'] for m in mm])),'p95':float(np.mean([m['full']['p95'] for m in mm])),
                'at80':float(np.mean([m['coverage'][3]['mean'] for m in mm])),'individual_means':[m['full']['mean'] for m in mm],
                'coverage':[{'coverage':mm[0]['coverage'][i]['requested_coverage'],'mean':float(np.mean([m['coverage'][i]['mean'] for m in mm])),
                    'p95':float(np.mean([m['coverage'][i]['p95'] for m in mm])),'above10':float(np.mean([m['coverage'][i]['above_10_fraction'] for m in mm]))} for i in range(6)]}
            if patch:
                group.update(strengths=[r['selected_strength'] for r in rr],collapsed_mean=float(np.mean([r['metrics']['collapsed']['common']['full']['mean'] for r in rr])),
                    posterior80=float(np.mean([r['metrics'][f"t{r['selected_strength']}"]['posterior']['coverage'][3]['mean'] for r in rr])))
            groups[protocol+'/'+family]=group
            for ranking,key in ([('common','chosen_curve'),('posterior','chosen_posterior_curve')] if patch else [('common','curve')]):
                curve=np.mean([a[key] for a in aa],axis=0)
                for i,value in enumerate(curve):curves.append({'protocol':protocol,'family':family,'ranking':ranking,'coverage':(i+1)/len(curve),'mean_delta_e00':float(value)})
        pairs=[('graph_paired','plain_raw'),('graph_paired','plain_paired'),('graph_paired','graph_raw'),('graph_raw','plain_raw'),('plain_paired','plain_raw')] if not patch else [
            (f'bag_d{d}_k{k}',f'mean_d{d}_k{k}') for d in (1,2) for k in (1,3)]+[(f'bag_d{d}_k3',f'bag_d{d}_k1') for d in (1,2)]
        for a,b in pairs:
            comparisons.append({'protocol':protocol,'candidate':a,'control':b,**comparison(arrays[a],arrays[b],'chosen_error' if patch else 'error','chosen_error' if patch else 'error')})
        if patch:
            for family in families:
                if family.startswith('bag'):comparisons.append({'protocol':protocol,'candidate':family,'control':family+' collapsed inference',**comparison(arrays[family],arrays[family],'chosen_error','collapsed_error')})
    write(out/'summary.json',{'scope':'SOURCE exploratory only, no new independent test','groups':groups,'comparisons':comparisons,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),out/'audit.json',out/'source_lock.json']}})
    with (out/'risk_coverage.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for ax,protocol in zip(axes,('mixed','from_SLR','from_ipod')):
        for family in sorted({c['family'] for c in curves}):
            cc=[c for c in curves if c['protocol']==protocol and c['family']==family and c['ranking']=='common' and c['coverage']>=.5]
            ax.plot([c['coverage']*100 for c in cc],[c['mean_delta_e00'] for c in cc],label=family)
        ax.set(title=protocol,xlabel='Accepted coverage (%)',ylabel='Mean skin DeltaE00');ax.grid(alpha=.25)
    axes[-1].legend(fontsize=7);fig.suptitle('Real source skin color; '+('conditional patch likelihood' if patch else 'spatially valid graph/support combination'))
    fig.tight_layout();fig.savefig(out/'risk_coverage.png',dpi=145);plt.close(fig)
    lines=['# '+('Conditional patch-distribution results' if patch else 'Spatially valid graph/support combination'),'',
        f'{expected} locally reproduced fits on original MSKCC photographs and actual instrument-native Lab.',
        'All errors are skin DeltaE00. Source validation is extensively reused; no new',
        'independent, ordinary-phone or universal-camera result. Seed scores are averaged,',
        'not ensembled. One-Gaussian statistical fits have one deterministic initialization.','',
        '| Protocol / family | Initializations | Mean | Median | p95 | Common 80% |',
        '|---|---:|---:|---:|---:|---:|']
    for key,g in groups.items():lines.append(f"| {key} | {g['initializations']} | {g['mean']:.4f} | {g['median']:.4f} | {g['p95']:.4f} | {g['at80']:.4f} |")
    lines += ['', 'All families within a protocol use identical input-novelty accept sets.',
        'This is an uncalibrated comparison of color accuracy at fixed coverage, not C+.',
        'p95 above averages per-fit p95. All six fixed coverages and complete curves are retained.','',
        '![Risk and coverage](risk_coverage.png)','', '## Descriptive matched differences','',
        'Negative favors the candidate. Patient-cluster resampling uses 10,000 draws;',
        'intervals on the small reused source cohort are not confirmatory or multiplicity-adjusted.','',
        '| Protocol | Candidate vs control | Mean difference | Patient 95% interval |',
        '|---|---|---:|---|']
    for r in comparisons:
        lo,hi=r['descriptive_patient_interval'];lines.append(f"| {r['protocol']} | {r['candidate']} vs {r['control']} | {r['mean_difference']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    if patch:
        lines += ['', '## Distribution and strength diagnostics','',
            'Collapsed inference uses the same model and selected strength; it is an',
            'information intervention, not an independently trained baseline. The mean-trained',
            'families supply that baseline. Strength is selected on same-camera validation only.','',
            '| Protocol / family | Strengths | Full bag/model mean | Collapsed mean | Posterior 80% |',
            '|---|---|---:|---:|---:|']
        for key,g in groups.items():lines.append(f"| {key} | {g['strengths']} | {g['mean']:.4f} | {g['collapsed_mean']:.4f} | {g['posterior80']:.4f} |")
        lines += ['', f"{audit['single_gaussian_sufficiency_models']} Gaussian fits obey the sufficient-mean identity, maximum score-difference residual {audit['maximum_sufficiency_gap']:.3g}.",
            'The conditional mixture can encode non-Gaussian structure, but that is not a',
            'demonstrated improvement in general skin-color accuracy. Actual patch RGB is',
            'observed; native Lab refers to the site, not a measured per-pixel color map.',
            f"{audit['exact_color_arrays']} exact color arrays; {audit['scalar_evaluation_color_cases']} scalar color cases; {audit['coverage_rows']} coverage rows; {audit['curve_points']} curve points.",
            f"{audit['weighted_normal_equation_checks']} weighted normal equations, {audit['independent_gaussian_density_cases']} independent densities and {audit['independent_expected_color_costs']} expected-color scalar costs.",
            'This is a CPU statistical falsifier with no GPU latency/VRAM/export claim.',
            f"Stored numeric scalars range {min(r['stored_numeric_scalars'] for r in records)} to {max(r['stored_numeric_scalars'] for r in records)}; largest model archive {max(r['model_bytes'] for r in records):,} bytes."]
    else:
        lines += ['', 'Graph processing occurs only on the original first-pass training image.',
            'The second, potentially mixed bag uses the plain core. All inference is plain.',
            'The new two-pass schedule is not identical to the older graph_always experiment.',
            f"Inference parameters: {records[0]['inference_parameters']:,}; stored parameters: {records[0]['parameters']:,}.",
            f"Maximum fit allocation {max(r['fit_peak_allocated_mib'] for r in records):.2f} MiB; largest full checkpoint {max(r['checkpoint_bytes'] for r in records):,} bytes.",
            f"{audit['exact_color_replay_arrays']} exact color arrays; {audit['independent_scalar_color_cases']} scalar cases; {audit['coverage_rows']} coverage rows; {audit['curve_points']} curve points.",
            f"{audit['original_graph_input_and_disabled_mixed_inference_checks']} actual-source runtime branch checks and all epoch pair/plan digests pass.",
            'No new isolated inference latency or export claim.']
    lines += ['', 'Original MSKCC CC-BY; no new data/weights or participant publication.',
        'Historical source controls remain stronger in key protocols: mixture mixed3.4406,',
        'paired mixture forward4.8328, training-only graph reverse4.9736. Do not promote',
        'a gain over weaker new controls as a project-wide victory. Independent MSKCC',
        'primary4.4570/80%4.1591 and ordinary fusion4.3005/4.1447 remain unchanged.',
        '[Next decision](../../research/skin_graph_patch_next_decision.md).']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'experiment':name,'groups':{k:{z:v[z] for z in ('mean','at80')} for k,v in groups.items()}}))


if __name__=='__main__':
    report('skin_graph_support_v1');report('skin_patch_likelihood_v1',True)
