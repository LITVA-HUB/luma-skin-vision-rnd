"""Privileged calibration results, explicitly separate from model accuracy."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_offset_diagnostic import OUT,RUN,SOURCE_RUN
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write


def main():
    audit=json.loads((OUT/'audit.json').read_bytes());assert audit['status']=='PASS'
    rr=json.loads((OUT/'results.json').read_bytes())['results'];energy={r['name']:r for r in json.loads((OUT/'energy_diagnostic.json').read_bytes())['rows']}
    groups={};curves=[]
    for protocol,domain in [('mixed','known'),('from_SLR','known'),('from_SLR','unseen'),('from_ipod','known'),('from_ipod','unseen')]:
        for arm in ('image','person_site','site','color','color_ipw','person_color'):
            rows=sorted([r for r in rr if r['protocol']==protocol and r['domain']==domain and r['arm']==arm],key=lambda r:r['seed']);assert len(rows)==3
            g={}
            for key in ('original','half','full'):
                g[key]={m:float(np.mean([r['metrics'][key]['full'][m] for r in rows])) for m in ('mean','median','p95','patient_balanced_mean','above_10_fraction')}
                g[key]['at80']=float(np.mean([r['metrics'][key]['coverage'][3]['mean'] for r in rows]))
                cc=[np.load(SOURCE_RUN/r['model']/(domain+'.npz'))['curve'] if key=='original' else np.load(RUN/(r['name']+'.npz'))[key+'_curve'] for r in rows]
                curve=np.mean(cc,0)
                for k,e in enumerate(curve,1):curves.append({'protocol':protocol,'domain':domain,'arm':arm,'correction':key,'accepted':k,'coverage':k/len(curve),'mean_delta_e00':float(e)})
            g['reference_people_per_exclusion']=rows[0]['reference_people_per_exclusion']
            g['energy']={k:float(np.mean([energy[r['name']][k] for r in rows])) for k in ('shared_residual_energy','between_person_residual_energy')}
            g['energy']['full_squared_Lab_change']=float(np.mean([energy[r['name']]['changes']['full']['direct'] for r in rows]))
            groups[f'{protocol}/{domain}/{arm}']=g
    write(OUT/'summary.json',{'scope':'PRIVILEGED REFERENCE-CALIBRATION COMPARATOR; no production accuracy improvement','groups':groups,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'audit.json',OUT/'results.json',OUT/'energy_diagnostic.json']}})
    with (OUT/'risk_coverage.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,protocol in zip(axes,['from_SLR','from_ipod']):
        for arm in ('image','person_color'):
            values=groups[f'{protocol}/unseen/{arm}'];ax.plot([0,.5,1],[values[k]['mean'] for k in ('original','half','full')],marker='o',label=arm)
        ax.set(title=protocol+' / unseen',xlabel='Privileged reference offset strength',ylabel='Actual skin mean DeltaE00',xticks=[0,.5,1]);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Diagnostic using OTHER evaluation people\'s instrument references');fig.tight_layout();fig.savefig(OUT/'offset_diagnostic.png',dpi=150);plt.close(fig)
    lines=['# Privileged source-reference offset diagnostic','',
        '**These are reference-calibration comparators, not improved single-image model scores.**',
        'Original90 source prediction arrays are unchanged. For each evaluation person,',
        'the offset uses other evaluation people\'s native instrument references. This',
        'violates the no-test-camera-calibration deployment condition by design and is',
        'used only to diagnose errors. Source cohorts are reused, not fresh independent',
        'validation. No TEST/CAL archive, new images or neural fitting was used.','',
        'All errors below are actual skin CIEDE2000. Group values average three separate',
        'runs. Half/full strengths were fixed before evaluation; neither is selected.','',
        '| Protocol / domain / model | Original mean | Privileged half | Privileged full | Original80% | Privileged full80% |',
        '|---|---:|---:|---:|---:|---:|']
    for k,g in groups.items():lines.append(f"| {k} | {g['original']['mean']:.4f} | {g['half']['mean']:.4f} | {g['full']['mean']:.4f} | {g['original']['at80']:.4f} | {g['full']['at80']:.4f} |")
    lines+=['','![Privileged offset diagnostic](offset_diagnostic.png)','',
        'A shared offset does not consistently explain transfer. In SLR-to-iPod, image',
        'control5.4877 becomes6.2298 with the full excluded-person correction; combination',
        '6.1545 becomes7.4325. In reverse, image6.5602 becomes6.0229 and combination',
        '6.4074 becomes5.4434. These are conditional reference-aided observations, not',
        'achieved calibration-free gains. Ordinary phone accuracy remains unvalidated.','',
        '## Why a correction can hurt','',
        'A separate algebraic diagnostic decomposes site/person-balanced squared native',
        'Lab residual energy. It is Euclidean Lab geometry, NOT CIEDE2000. If mu is the',
        'mean person residual, v the between-person residual energy and K the number',
        'of people, the excluded-person offset changes squared error by',
        '`(-2a+a²)||mu||² + (2a/(K-1)+a²/(K-1)²)v` at strength a.',
        'A shared correction can amplify person differences more than it removes bias.','',
        '| Protocol / domain / model | Shared residual energy | Between-person energy | Full squared-Lab error change |',
        '|---|---:|---:|---:|']
    for k,g in groups.items():
        e=g['energy'];lines.append(f"| {k} | {e['shared_residual_energy']:.4f} | {e['between_person_residual_energy']:.4f} | {e['full_squared_Lab_change']:.4f} |")
    lines+=['','This identity explains only the squared-Euclidean diagnostic. Primary skin',
        'accuracy remains the scalar-verified DeltaE00 table. These descriptive residual',
        'terms use labels and are not inference features, a noise floor or an analytic',
        'DeltaE00 decomposition. Between-person effects cannot be assigned causally to',
        'biology, camera, capture settings or reference noise from these data alone.','',
        '## Integrity','',
        f"{audit['exact_corrected_arrays']} exact corrected arrays; {audit['excluded_person_folds_and_own_reference_interventions']} exclusion folds with own-reference perturbation checks.",
        f"{audit['scalar_color_cases']} independent scalar color cases; {audit['fixed_coverage_rows']} fixed-coverage rows; {audit['squared_Lab_energy_identities']} exact energy identities.",
        'Explicit exclusion matrices have zero own-person blocks and unit row sums.',
        'Original image/site/person roles and array hashes are preserved. All fixed',
        'coverage levels and full curves use the original uncalibrated novelty ranking.',
        'There is no new calibrated selective-risk guarantee or latency/export result.',
        'Original MSKCC CC-BY; no new data/weights, cloud or external publication.',
        'Independent model result remains primary4.4570/80%4.1591 vs ordinary fusion',
        '4.3005/4.1447. This diagnostic must not replace those scores.',
        '[Research decision](../../research/skin_offset_diagnostic_next_decision.md).']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Privileged diagnostic report created; original model scores unchanged')


if __name__=='__main__':main()
