"""Source-selected inverse results and separate representation diagnostics."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_appearance_inverse_train import OUT,RUN,ALPHAS
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write


def main():
    audit=json.loads((OUT/'audit.json').read_bytes());assert audit['status']=='PASS' and audit['fits']==108
    records=[json.loads(p.read_bytes())|{'name':p.parent.name} for p in sorted(OUT.glob('*/result.json'))]
    train,val=load('train'),load('validation');selected=[];comparisons=[];curves=[];diagnostics={};scalar_cases=0
    for protocol in ('mixed','from_SLR','from_ipod'):
        t=train if protocol=='mixed' else subset(train,train['device']==protocol.removeprefix('from_'))
        ev=val if protocol=='mixed' else subset(val,val['device']!=protocol.removeprefix('from_'))
        _,first=np.unique(t['site'],return_index=True);palette=t['target'][first]
        distances=np.array([[scalar_de(a,b) for b in palette] for a in ev['target']]);scalar_cases+=distances.size
        oracle=distances.min(1);constant=np.array([scalar_de(palette.mean(0),y) for y in ev['target']]);scalar_cases+=len(constant)
        diagnostics[protocol]={'palette_size':len(palette),'constant_site_mean_error':float(constant.mean()),
            'known_target_nearest_palette_mean':float(oracle.mean()),'known_target_nearest_palette_p95':float(np.quantile(oracle,.95)),
            'scope':'Representation/prior diagnostic, not an image model or attained accuracy'}
        for kind in ('rgb','stats'):
            for direction in ('direct','global','mode'):
                for degree in (1,2):
                    group=[r for r in records if (r['protocol'],r['kind'],r['direction'],r['degree'])==(protocol,kind,direction,degree)]
                    for output in (('direct',) if direction=='direct' else ('mean','medoid')):
                        r=min(group,key=lambda r:(r['selection'][output]['patient_balanced_mean'],ALPHAS.index(r['alpha'])))
                        metric=r['metrics'][output];a=dict(np.load(RUN/r['name']/'evaluation.npz'))
                        family=f'{kind}/{direction}_d{degree}/{output}'
                        row={'protocol':protocol,'family':family,'kind':kind,'direction':direction,'degree':degree,'output':output,
                            'alpha':r['alpha'],'name':r['name'],'selection_patient_mean':r['selection'][output]['patient_balanced_mean'],
                            'mean':metric['common']['full']['mean'],'median':metric['common']['full']['median'],'p95':metric['common']['full']['p95'],
                            'common80':metric['common']['coverage'][3]['mean'],'posterior80':metric.get('posterior',{}).get('coverage',[None]*4)[3],
                            'inference_scalar_storage':r['inference_scalar_storage'],'full_model_archive_bytes':r['model_bytes'],
                            'posterior_diagnostic':r.get('posterior_diagnostic'), 'metrics':metric}
                        row['posterior80']=row['posterior80']['mean'] if row['posterior80'] else None
                        selected.append(row)
                        for ranking in metric:
                            curve=a[output+('_curve' if ranking=='common' else '_posterior_curve')]
                            for i,risk in enumerate(curve):curves.append({'protocol':protocol,'family':family,'ranking':ranking,'coverage':(i+1)/len(curve),'mean_delta_e00':float(risk)})
        # Compare same-kind same-degree output systems to direct controls; no equal-compute claim.
        for row in [r for r in selected if r['protocol']==protocol and r['direction']!='direct']:
            control=next(r for r in selected if r['protocol']==protocol and r['kind']==row['kind'] and r['direction']=='direct' and r['degree']==row['degree'])
            a=dict(np.load(RUN/row['name']/'evaluation.npz'));b=dict(np.load(RUN/control['name']/'evaluation.npz'))
            np.testing.assert_array_equal(a['patient'],b['patient']);np.testing.assert_array_equal(a['common_risk'],b['common_risk'])
            d=a[row['output']+'_error']-b['direct_error'];patient=a['patient']
            clusters=np.array([d[patient==p].mean() for p in np.unique(patient)])
            rng=np.random.default_rng(51871);boot=clusters[rng.integers(0,len(clusters),(10000,len(clusters)))].mean(1)
            comparisons.append({'protocol':protocol,'candidate':row['family'],'control':control['family'],'mean_difference':float(d.mean()),
                'descriptive_patient_interval':np.quantile(boot,[.025,.975]).tolist()})
    support=json.loads((OUT/'support_control.json').read_bytes());assert support['audit']['status']=='PASS'
    for r in support['selected']:
        a=dict(np.load(RUN/r['name']/'support_control.npz'))
        for i,risk in enumerate(a[r['control']+'_curve']):
            curves.append({'protocol':r['protocol'],'family':f"{r['kind']}/direct_d{r['degree']}/{r['control']}",'ranking':'common',
                'coverage':(i+1)/len(a[r['control']+'_curve']),'mean_delta_e00':float(risk)})
    summary={'scope':'SOURCE exploratory only; alpha selected on same-camera validation, never other-camera error',
        'selected':selected,'comparisons':comparisons,'diagnostics':diagnostics,'diagnostic_scalar_cases':scalar_cases,'support_controls':support['selected'],
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'audit.json',OUT/'source_lock.json',OUT/'support_control.json']}}
    write(OUT/'summary.json',summary)
    with (OUT/'risk_coverage.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    for row,kind in enumerate(('rgb','stats')):
        for col,protocol in enumerate(('mixed','from_SLR','from_ipod')):
            ax=axes[row,col]
            for s in selected:
                if s['protocol']!=protocol or s['kind']!=kind or s['output']=='medoid':continue
                cc=[c for c in curves if c['protocol']==protocol and c['family']==s['family'] and c['ranking']=='common' and c['coverage']>=.5]
                ax.plot([c['coverage']*100 for c in cc],[c['mean_delta_e00'] for c in cc],label=f"{s['direction']} d{s['degree']}")
            ax.set(title=protocol+' / '+kind,xlabel='Accepted coverage (%)',ylabel='Mean skin DeltaE00');ax.grid(alpha=.25)
    axes[0,-1].legend(fontsize=8);fig.suptitle('Conditional appearance inversion; same-camera alpha selection; common input ranking')
    fig.tight_layout();fig.savefig(OUT/'risk_coverage.png',dpi=145);plt.close(fig)
    lines=['# Conditional appearance inversion: actual skin color','',
        '108 locally reproduced closed-form fits, original MSKCC real photographs and',
        'instrument-native Lab. Source cohorts are repeatedly inspected; no new independent',
        'result. No camera or capture-mode ID enters inference. All errors are DeltaE00.','',
        'Each family selects alpha using only same-camera validation patient-mean error.',
        'Mean/direct outputs are shown below; all 60 selected endpoints including medoids,',
        'all candidate fits and both risk rankings are retained in JSON/CSV.','',
        '| Protocol | Input / family | Alpha | Mean | Median | p95 | Common 80% | Posterior 80% |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in selected:
        if r['output']=='medoid':continue
        pr='n/a' if r['posterior80'] is None else f"{r['posterior80']:.4f}"
        lines.append(f"| {r['protocol']} | {r['family']} | {r['alpha']:g} | {r['mean']:.4f} | {r['median']:.4f} | {r['p95']:.4f} | {r['common80']:.4f} | {pr} |")
    lines += ['', '![Risk and coverage](risk_coverage.png)','',
        'Common input ranking has identical accept sets within each input kind. Posterior',
        'expected color error is a separate uncalibrated hypothesis; it is not a per-image',
        'bound or a matched calibrated C+ comparison. Simple direct controls use fewer',
        'parameters/fit operations and no capture labels. Neural controls use all 64 patch',
        'tokens rather than only their mean: these are not equal-capacity architectures.','',
        '## Representation and prior controls','',
        '| Protocol | Actual TRAIN color atoms | Constant prior mean error | Known-target nearest atom mean | Oracle p95 |',
        '|---|---:|---:|---:|---:|']
    for p,d in diagnostics.items():lines.append(f"| {p} | {d['palette_size']} | {d['constant_site_mean_error']:.4f} | {d['known_target_nearest_palette_mean']:.4f} | {d['known_target_nearest_palette_p95']:.4f} |")
    lines += ['', 'The nearest-atom oracle uses true reference color and is not deployable.',
        'Its gap to the actual model assesses a representation limitation, not causal',
        'camera identifiability. A discrete empirical prior is not an independent physical',
        'skin model; posterior mean remains in the TRAIN convex hull. Gaussian forward',
        'noise includes out-of-person residual bias but is not calibrated on new devices.','',
        '## Verification and boundaries','',
        f"{audit['exact_prediction_arrays']} exact selection/evaluation arrays; {audit['independent_scalar_evaluation_cases']} scalar evaluation cases; {audit['coverage_rows']} coverage rows; {audit['curve_points']} curve points.",
        f"{audit['person_excluded_forward_folds_checked']} person-excluded forward folds and {audit['normal_equation_checks']} normal-equation checks; {audit['independent_gaussian_density_cases']} independent Gaussian densities.",
        f"{audit['independent_palette_cost_cases']} independent palette/expected-cost cases plus {scalar_cases} scalar representation/prior diagnostics.",
        'All 108 fitted models and OOF arrays replay exactly. Direct and forward alphas',
        'are selected separately with frozen candidate order. All results remain exploratory.',
        'Full model NPZ archives include OOF diagnostic arrays; they are not deployment',
        'file sizes. This is a CPU statistical falsifier; no GPU latency/VRAM/export claim.',
        'Original MSKCC CC-BY; no new external code, weights, data or participant publication.',
        'Independent MSKCC primary mean remains 4.4570/80%4.1591, ordinary fusion',
        '4.3005/4.1447. Ordinary phone facial accuracy and novelty remain unproved.',
        '[Decision](../../research/skin_appearance_inverse_next_decision.md).','',
        '## Opponent check: restrict direct predictions to the same empirical support','',
        'Post-hoc controls were frozen separately after the original source screen.',
        'They use predicted color and TRAIN atoms only. Hull projection uses standardized',
        'Lab Euclidean distance; nearest-atom uses DeltaE00. Neither uses image truth.',
        'Alpha is reselected on the same-camera source validation for each control.',
        'Their full common-ranking curves are included in the CSV.','',
        '| Protocol | Input / degree | Control | Alpha | Mean | Common 80% |',
        '|---|---|---|---:|---:|---:|']
    for r in support['selected']:
        m=r['metrics'];lines.append(f"| {r['protocol']} | {r['kind']} / {r['degree']} | {r['control']} | {r['alpha']:g} | {m['full']['mean']:.4f} | {m['coverage'][3]['mean']:.4f} |")
    a=support['audit']
    lines += ['',f"Support audit: {a['independent_nearest_atom_costs']} scalar atom distances, {a['independent_scalar_color_cases']} scalar color cases, {a['coverage_rows']} coverage rows and {a['curve_points']} curve points.",
        f"{a['projected_points_with_kkt_check']} projected selection/evaluation instances satisfy hull feasibility and KKT stationarity; unchanged interior predictions remain exact.",
        'These instance counts are computational checks, not additional people or photos.',
        'The empirical support control reduces some direct-transfer failures but does',
        'not generally match inverse regression. Neither family beats the strongest',
        'archived compact image models universally. No calibrated selective win.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'selected_endpoints':len(selected),'diagnostics':diagnostics,'scalar_diagnostics':scalar_cases}))


if __name__=='__main__':main()
