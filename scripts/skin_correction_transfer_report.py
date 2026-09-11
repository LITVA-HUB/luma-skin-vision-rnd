"""Audited camera-transfer negatives for unchanged stable-color correction."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_correction_transfer import OUT,RUN,active_lock
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write


def main():
    audit=json.loads((OUT/'audit.json').read_bytes());assert audit['status']=='PASS'
    records=json.loads((OUT/'results.json').read_bytes())['records'];arms=('base','in_full','in_matched','out_person');groups=[];coverage=[]
    for protocol in ('mixed','from_SLR','from_ipod'):
        rr=[r for r in records if r['protocol']==protocol]
        for domain in rr[0]['arms']['base']['evaluations']:
            for arm in arms:
                values=[r['arms'][arm]['evaluations'][domain] for r in rr]
                group={'protocol':protocol,'domain':domain,'arm':arm,**{k:float(np.mean([v['full'][k] for v in values])) for k in values[0]['full']},
                       'mean_at80':float(np.mean([v['coverage'][3]['mean'] for v in values])),
                       'seed_means':[v['full']['mean'] for v in values]}
                groups.append(group)
                for i,c in enumerate((1.,.95,.9,.8,.7,.6)):
                    coverage.append({'protocol':protocol,'domain':domain,'arm':arm,'coverage':c,'accepted':values[0]['coverage'][i]['accepted'],
                                     **{k:float(np.mean([v['coverage'][i][k] for v in values])) for k in values[0]['full']}})
    with (OUT/'risk_coverage.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(coverage[0]));writer.writeheader();writer.writerows(coverage)
    index={(r['protocol'],r['domain'],r['arm']):r for r in groups}
    changes=[]
    for protocol in ('from_SLR','from_ipod'):
        for arm in arms[1:]:
            base=index[protocol,'unseen','base']['mean'];new=index[protocol,'unseen',arm]['mean']
            changes.append({'protocol':protocol,'arm':arm,'mean_change':new-base,'relative_increase_percent':100*(new-base)/base})
    assert all(v['mean_change']>0 for v in changes)
    resources={}
    for kind,rows in [('inner',[f for r in records for f in r['folds']]),('head',[r['arms'][a] for r in records for a in arms[1:]])]:
        resources[kind]={k:[float(min(r[k] for r in rows)),float(max(r[k] for r in rows))] for k in ('fit_seconds','fit_peak_allocated_mib','checkpoint_bytes')}
    fig,axes=plt.subplots(1,3,figsize=(14,4.1),layout='constrained');colors=['#20282e','#985cac','#158d98','#dc7c31']
    for ax,(protocol,domain,title) in zip(axes,[('mixed','known','Mixed known cameras'),('from_SLR','unseen','SLR to unseen iPod'),('from_ipod','unseen','iPod to unseen SLR')]):
        visible=[]
        for arm,color in zip(arms,colors):
            curves=[np.load(RUN/f"{protocol}__s{r['seed']}"/f'{arm}__{domain}.npz')['curve'] for r in records if r['protocol']==protocol]
            y=np.mean(curves,axis=0);x=np.arange(1,len(y)+1)/len(y);visible.extend(y[x>=.6]);ax.plot(x,y,label=arm,color=color)
        ax.set(xlim=(.6,1),ylim=(min(visible)-.08,max(visible)+.08),xlabel='Accepted fraction',ylabel='Mean native skin DeltaE00',title=title)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Reused source validation: all correction arms worsen both unseen-camera means')
    fig.savefig(OUT/'risk_coverage.png',dpi=170);plt.close(fig)
    lines=['# Stable correction camera transfer: internal gain does not generalize','',
           '36 new inner-core fits and27 unchanged correction-head fits are complete. Nine original fixed-final930-step cores remain unchanged. Original MSKCC CC-BY real photographs and instrument-native Lab; source TRAIN24people/966photos and VALIDATION6/264. Independent TEST/CAL stay closed. Source validation is heavily reused; these are exploratory results, not fresh independent ordinary-phone accuracy.','',
           '## Actual skin-color results','',
           'Entries average three individual seed scores, not ensemble predictions. Median/p95 are averages of per-seed quantiles, not pooled quantiles.','',
           '| Training / evaluation | Arm | Mean DeltaE00 | Median | p95 | Error >10 | Mean at80% |','|---|---|---:|---:|---:|---:|---:|']
    for r in groups:lines.append(f"| {r['protocol']} /{r['domain']} | {r['arm']} | {r['mean']:.4f} | {r['median']:.4f} | {r['p95']:.4f} | {100*r['fraction_above10']:.2f}% | {r['mean_at80']:.4f} |")
    lines += ['', 'The previously leading in_matched head worsens unseen forward5.4877 ->5.7371 and reverse6.5602 ->7.2908. All three head arms worsen both unseen-camera means. The minor mixed-camera gain for in_full3.8277 ->3.7626 does not rescue the generalization hypothesis. Do not choose different head winners per camera and describe the result as camera-blind.', '',
              'The earlier internal mean6.1082 ->5.6560 is retained as a positive result on its different six-person cohort. This follow-up refutes its extension to a universal transfer gain. Stronger historical source recipes remain mixed3.4406 ([capture](../skin_capture_v1/report.md)), forward4.8328 ([paired expert](../skin_expert_anchor_v1/report.md)), reverse4.9736 ([training branch](../skin_train_branch_v1/report.md)). They use different budgets/selection and are contextual locally reproduced comparators; none is beaten here. No author numbers are passed off as our measurements.', '',
              '## What was actually controlled','',
              'The head is unchanged:39 stable color inputs ->384->384->64->3,188,035 parameters added to929,297 core; total1,117,332 under1,129,297 cap. One image/core/head at inference,78 scale scalars and no reference memory or camera ID. All arms start at zero correction with identical weights and receive the same300 updates/data draws per protocol/seed.', '',
              'Four inner folds exclude whole people; assignments agree between mixed and single-camera banks. The36 inner cores use the prior930-step recipe unchanged. in_full uses predictions of the core that saw the full training bank; in_matched uses a smaller core that saw each query person; out_person uses the corresponding core that excluded the query person. Matched inclusion and exclusion share exactly the same inner cores/budgets. All heads deploy on their unchanged full-bank core; the3/4-to-full training-support shift remains a limitation.', '',
              'Source unseen cameras are absent from all fitting rows and fitted scales. Person/camera/capture are confounded: this is valid device exclusion but not a causal camera-only intervention. The clinical SLR/iPod imagery is not a normal facial-selfie benchmark.', '',
              '## Risk and coverage','',
              'All four arms use identical nearest-TRAIN color36 distance rankings. All360 per-seed fixed-coverage rows and120 aggregate rows at100/95/90/80/70/60% are retained, with full risk curves. This is not calibrated expected error, matched confidence-head superiority or a reliable refusal guarantee.', '',
              '![Source risk-coverage curves](risk_coverage.png)', '',
              '## Reproducibility recovery and audit','',
              'The initial frozen run stopped after16 inner cores and9 mixed heads because concatenated validation batching changed four historical baseline values by at most3.8146973e-6 Lab. Separate known/unseen batch32 processing reproduced both original arrays bit for bit. The recovery restores these original domain batch boundaries, without changing any architecture, fit, hyperparameter, candidate or metric. The exact assertion was retained. Original source_lock.json, completed fits and the pretest Git checkpoint are preserved; recovery_lock.json records the amended runner/test and recovery note. No unseen correction result had been evaluated before the repair.', '',
              '[Recovery record](../../research/skin_correction_transfer_batch_recovery.md). Final audit passes36 inner-core replays,15 original domain replays,36 exclusion/scale checks,11592 routing rows,three complete core refits andthree complete head refits.27 NumPy heads and60 exact evaluation arrays replay;26892 scalar CIEDE2000 values and360 coverage checks pass. Max NumPy native Lab discrepancy4.12e-6; scalar color discrepancy4.89e-15.', '',
              '## Measured resources and decision','',
              f"Allocated cached training peak: inner cores{resources['inner']['fit_peak_allocated_mib'][0]:.2f}–{resources['inner']['fit_peak_allocated_mib'][1]:.2f}MiB, heads{resources['head']['fit_peak_allocated_mib'][0]:.2f}–{resources['head']['fit_peak_allocated_mib'][1]:.2f}MiB. Complete deployed checkpoints{int(resources['head']['checkpoint_bytes'][0])}–{int(resources['head']['checkpoint_bytes'][1])}bytes. These are process allocations with cached data; no isolated production VRAM or end-to-end inference timing claim. No export optimization is justified by this failed transfer screen.", '',
              'Stop promoting this frozen-core correction family as camera-general. Preserve the internal component gain and current negative transfer result. A next experiment should challenge the need for an output correction head and investigate image information or a jointly learned representation, while checking previous graph, histogram, spectral and pixel negatives to avoid repetition. This screen does not prove every such representation will fail.', '',
              'For Luma, these are measurements of genuine skin-color error. They do not establish ordinary smartphone facial accuracy, reliable cosmetic decisions or a novel mechanism. Known stacking is prior art. Independent primary4.4570/80%4.1591 versus ordinary fusion4.3005/4.1447 stays unchanged. The product aspiration median<=2,p95<=5 at>=80% on unseen ordinary phones remains unmet.', '',
              '[Frozen protocol](../../research/skin_correction_transfer_protocol_v1.md) · [Next decision](../../research/skin_correction_transfer_next_decision.md)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=[Path(__file__),OUT/'results.json',OUT/'audit.json',OUT/'source_lock.json',active_lock(),OUT/'report.md',OUT/'risk_coverage.csv']
    write(OUT/'summary.json',{'groups':groups,'unseen_changes':changes,'resources':resources,'all_heads_worsen_both_unseen_means':True,
                            'strongest_local_baseline_beaten':False,'new_independent_accuracy':False,
                            'bindings':{str(p.relative_to(ROOT)):sha(p) for p in files}})
    print(json.dumps({'unseen_changes':changes,'resources':resources}),flush=True)


if __name__=='__main__':main()
