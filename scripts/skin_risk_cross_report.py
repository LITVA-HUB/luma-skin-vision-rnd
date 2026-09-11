"""Aggregate all fixed crossings with matched-compute descriptive comparisons."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_risk_cross import OUT,RUN
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write


def main():
    source=json.loads((OUT/'results.json').read_bytes());rr=source['records'];assert len(rr)==90
    methods=list(dict.fromkeys(r['method'] for r in rr));groups={};curves=[];comparisons=[]
    for protocol in ('mixed','from_SLR','from_ipod'):
        arrays=[dict(np.load(RUN/f'{protocol}__s{s}.npz')) for s in (17,29,43)]
        for method in methods:
            r=[r for r in rr if r['protocol']==protocol and r['method']==method]
            groups[protocol+'/'+method]={'mean':float(np.mean([x['scores']['full']['mean'] for x in r])),
                'p95':float(np.mean([x['scores']['full']['p95'] for x in r])),
                'at80':float(np.mean([x['scores']['coverage'][3]['mean'] for x in r])),
                'p95_at80':float(np.mean([x['scores']['coverage'][3]['p95'] for x in r])),
                'models':r[0]['models_required'],'active_parameters':r[0]['active_parameters']}
            curve=np.mean([a[method+'_curve'] for a in arrays],axis=0)
            for i,risk in enumerate(curve):curves.append({'protocol':protocol,'method':method,'coverage':(i+1)/len(curve),'mean_delta_e00':float(risk)})
        patient=arrays[0]['patient'];people=np.unique(patient);rng=np.random.default_rng(23119)
        contrasts=[('mixed_pair_expected','ordinary_pair_dispersion'),('mixed_pair_covariance','ordinary_pair_dispersion'),
            ('ordinary_gaussian_expected','ordinary_dispersion')]
        for a,b in contrasts:
            values=[]
            for _ in range(2000):
                selected=np.concatenate([np.flatnonzero(patient==p) for p in rng.choice(people,len(people),replace=True)])
                count=int(np.ceil(.8*len(selected)));diff=[]
                for v in arrays:
                    def risk(method):
                        order=np.argsort(v[method+'_risk'][selected],kind='stable')[:count]
                        return v[method+'_error'][selected[order]].mean()
                    diff.append(risk(a)-risk(b))
                values.append(np.mean(diff))
            comparisons.append({'protocol':protocol,'candidate':a,'control':b,
                'at80_difference':groups[protocol+'/'+a]['at80']-groups[protocol+'/'+b]['at80'],
                'descriptive_patient_bootstrap_interval':np.quantile(values,[.025,.975]).tolist(),
                'coverage_recomputed_inside_bootstrap':True})
    write(OUT/'summary.json',{'groups':groups,'comparisons':comparisons,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'results.json']}})
    with (OUT/'risk_coverage.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    shown=['ordinary_dispersion','ordinary_pair_dispersion','ordinary_pair_disagreement','mixed_pair_expected','mixed_pair_covariance']
    for ax,protocol in zip(axes,('mixed','from_SLR','from_ipod')):
        for method in shown:
            c=[x for x in curves if x['protocol']==protocol and x['method']==method and x['coverage']>=.5]
            ax.plot([100*x['coverage'] for x in c],[x['mean_delta_e00'] for x in c],label=method)
        ax.set(title=protocol,xlabel='Accepted coverage (%)',ylabel='Mean skin DeltaE00');ax.grid(alpha=.25)
    axes[-1].legend(fontsize=7);fig.suptitle('Post-hoc source crossing; uncalibrated risk; matched two-model controls')
    fig.tight_layout();fig.savefig(OUT/'risk_coverage.png',dpi=145);plt.close(fig)
    lines=['# Frozen color/risk crossing on real skin source data','',
        '90locally computed method/seed/protocol endpoints; no new image fitting.',
        'Actual MSKCC instrument-native Lab is the reference. Source validation is',
        'heavily reused. No independent test, ordinary-phone or calibrated C+ claim.','',
        'Entries average three separate seed/pair scores, not a three-run ensemble.','',
        '| Protocol | System | Mean | p95 | At80% | p95at80% | Models |',
        '|---|---|---:|---:|---:|---:|---:|']
    for key,g in groups.items():
        protocol,method=key.split('/')
        lines.append(f"| {protocol} | {method} | {g['mean']:.4f} | {g['p95']:.4f} | {g['at80']:.4f} | {g['p95_at80']:.4f} | {g['models']} |")
    lines+=['','Ordinary pairs cover17/29,29/43,43/17. Mixed pairs combine same-seed ordinary',
        'and Gaussian predictions. All averages are fixed50/50. Gaussian integration',
        'uses4096antithetic Sobol points; no true reference participates in ranking.',
        'Covariance-only risk is centered at the proposed answer, removing mean bias.',
        'Expected risk uses the Gaussian mean; disagreement is a cheaper comparator.','',
        '## Matched-compute comparisons','',
        'Patient bootstrap resamples source people and recomputes exact80%coverage',
        'inside each draw for each seed. Intervals are descriptive after extensive',
        'source reuse, not confirmatory or multiplicity-adjusted.','',
        '| Protocol | Candidate vs control | Difference at80% | Patient95% interval |',
        '|---|---|---:|---|']
    for c in comparisons:
        lo,hi=c['descriptive_patient_bootstrap_interval']
        lines.append(f"| {c['protocol']} | {c['candidate']} vs {c['control']} | {c['at80_difference']:.4f} | [{lo:.4f},{hi:.4f}] |")
    lines+=['','![Risk and coverage](risk_coverage.png)','',
        'Mixed-source crossing improves selected error; transfer is not consistently',
        'better than matched ordinary pairs. A partial source signal is not universality.',
        'Ordinary two-model pairs also improve color accuracy without the proposed risk.',
        'Historical training-only graph remains a stronger reverse-color control4.9736.','',
        'Ordinary pair active parameters1,858,594; mixed pair1,864,750. These results',
        'require two model forwards plus Gaussian integration where used. No new latency',
        'or inference-memory claim. This screen does not include a fitted C+ error head;',
        'any later calibration requires person-held-out residuals.','',
        f"{source['independent_scalar_integral_cases']}independent scalar integral cases; {source['independent_scalar_color_cases']}scalar color cases;",
        f"{source['coverage_rows']}coverage rows; {source['curve_points']}curve points. Maximum gap{source['maximum_gap']:.3g}.",
        'Original CC-BY source data and all contributing predictions remain hash-bound.',
        'No participant data, weights or claims are externally published.','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(comparisons))


if __name__=='__main__':main()
