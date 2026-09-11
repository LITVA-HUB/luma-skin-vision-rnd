"""Audited internal skin correction results; distinguish improvement from hypothesis."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_crossfit_correction_train import OUT,RUN
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write


def main():
    assert json.loads((OUT/'audit.json').read_bytes())['status']=='PASS'
    records=json.loads((OUT/'results.json').read_bytes())['records'];arms=('base','in_full','in_matched','out_person')
    rows=[];coverage=[];arrays={}
    for arm in arms:
        scores=[r['arms'][arm]['evaluation'] for r in records]
        rows.append({'arm':arm,**{k:float(np.mean([s['full'][k] for s in scores])) for k in scores[0]['full']},
                     'seed_means':[s['full']['mean'] for s in scores],'mean_at80':float(np.mean([s['coverage'][3]['mean'] for s in scores]))})
        arrays[arm]=[dict(np.load(RUN/f"s{r['seed']}"/(arm+'.npz'))) for r in records]
        for i,c in enumerate((1.,.95,.9,.8,.7,.6)):
            coverage.append({'arm':arm,'coverage':c,'accepted':scores[0]['coverage'][i]['accepted'],
                             **{k:float(np.mean([s['coverage'][i][k] for s in scores])) for k in scores[0]['full']}})
    with (OUT/'risk_coverage.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(coverage[0]));writer.writeheader();writer.writerows(coverage)
    base_error=np.mean([a['error'] for a in arrays['base']],axis=0);people=arrays['base'][0]['patient'];diff=[]
    for arm in arms[1:]:
        e=np.mean([a['error'] for a in arrays[arm]],axis=0);pd=np.array([(e-base_error)[people==p].mean() for p in np.unique(people)])
        diff.append({'arm':arm,'people_improved_vs_base':int(np.sum(pd<0)),'people':len(pd),'person_mean_delta':float(pd.mean()),
                     'person_delta_min':float(pd.min()),'person_delta_max':float(pd.max()),
                     'seeds_improved_vs_base':sum(r['arms'][arm]['evaluation']['full']['mean']<r['arms']['base']['evaluation']['full']['mean'] for r in records)})
    improvement=(rows[0]['mean']-rows[2]['mean'])/rows[0]['mean']*100
    resources={}
    for kind,entries in [('inner_cores',[f for r in records for f in r['folds']]),('heads',[r['arms'][a] for r in records for a in arms[1:]])]:
        resources[kind]={k:[float(min(v[k] for v in entries)),float(max(v[k] for v in entries))] for k in ('fit_seconds','fit_peak_allocated_mib','checkpoint_bytes')}
    table_errors={a:float(np.mean([r['table_training_errors'][a]['mean'] for r in records])) for a in arms[1:]}
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained');colors=['#20282e','#985cac','#158d98','#dc7c31'];visible=[]
    for arm,color in zip(arms,colors):
        curve=np.mean([a['curve'] for a in arrays[arm]],axis=0);x=np.arange(1,len(curve)+1)/len(curve)
        axes[0].plot(x,curve,label=arm,color=color);visible.extend(curve[x>=.6])
    axes[0].set(xlim=(.6,1),ylim=(min(visible)-.1,max(visible)+.1),xlabel='Accepted fraction',ylabel='Mean native skin DeltaE00',title='Same six held people; common ranking');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    for i,(row,color) in enumerate(zip(rows,colors)):
        axes[1].scatter(np.full(3,i),row['seed_means'],color=color,label=None);axes[1].plot([i-.18,i+.18],[row['mean']]*2,color=color,lw=3)
    axes[1].set(xticks=np.arange(4),xticklabels=arms,ylabel='Mean native skin DeltaE00',title='Three individual seeds; line = mean');axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Exploratory original-TRAIN screen: improvement, but OOF-superiority hypothesis fails')
    fig.savefig(OUT/'correction_results.png',dpi=170);plt.close(fig)
    text=['# Stable-color correction improves this internal screen; OOF is not the winner','',
          'Original MSKCC CC-BY real photographs and instrument-native skin Lab. Only original TRAIN was loaded.18 people/734 photographs train the final core/head;6 people/232 photographs are internal evaluation. These same people were used in earlier exploratory screens. Source VALIDATION and independent CAL/TEST remain unopened. Both cameras occur in support and evaluation: this is not unseen-camera evidence.','',
          '## Actual skin color results','',
          'Means and quantiles below average three individual seed results, not ensemble predictions or pooled quantiles.','',
          '| Correction training | Mean DeltaE00 | Median | p95 | Error >10 | Mean at80% |','|---|---:|---:|---:|---:|---:|']
    for r in rows:text.append(f"| {r['arm']} | {r['mean']:.4f} | {r['median']:.4f} | {r['p95']:.4f} | {100*r['fraction_above10']:.2f}% | {r['mean_at80']:.4f} |")
    text += ['',f'The strongest local arm in_matched lowers mean error by{improvement:.2f}% versus its unchanged core, from6.1082 to5.6560. Its p95 decreases14.9960 ->12.9644 and at80%5.6609 ->5.2608. All three correction arms improve the core in all three seeds. This is a useful source component result, not an independent universal-model victory.','',
             'The tested explanatory hypothesis fails: out_person5.8945 is worse than in_matched5.6560 and in_full5.8579 on the seed-averaged endpoint. Correct OOF construction is necessary when claiming excluded-person supervision, but did not produce the best color correction here. More realistic base error targets alone are not sufficient.','',
             '| Arm | People improved vs core, after seed averaging | Mean person error change |','|---|---:|---:|']
    for d in diff:text.append(f"| {d['arm']} | {d['people_improved_vs_base']}/{d['people']} | {d['person_mean_delta']:.4f} |")
    text += ['', 'These are six reused people and three correlated seed runs; no significance or population-generalization claim follows. The later choice of a leading arm must be followed by a separately frozen camera-transfer experiment, not selection of different winners per camera.', '',
             '## Mechanism and matched controls','',
             'The reference bank and512-dimensional neural context were removed. A stable39-vector combines36 fixed color descriptors with3 native base-color estimates. A188,035-parameter39->384->384->64->3 residual head corrects the frozen929,297-parameter core. Total1,117,332 parameters, under the authorized1,129,297 cap. One image, one core and one head at inference; no camera ID, query reference or reference-memory payload.', '',
             'Nine new cores were fitted from scratch:three inner folds x three seeds. Each inner core saw12 support people; six inner-query people were absent from its training and its target scales. Each fit used930 updates, the same fixed recipe as the existing18-person final cores. No new epoch selection occurred.', '',
             'in_full trains its correction on predictions of the18-person core that saw each support person. in_matched uses a12-person core that saw that person; out_person uses the corresponding12-person core that excluded the person. The latter two reuse exactly the same inner cores, with one encoder per training image and no averaging. Their roles are routed cyclically to keep core data size/budget equal. At evaluation every head uses the same18-person core. Thus the12-to18 support-size deployment shift remains shared by the two inner-core arms.', '',
             'Heads share initialization, zero initial correction,300 steps, image draws, target scales, capacity and standardized Lab MSE loss. Original full-core predictions replay exactly. Learned hidden coordinates from separately fitted encoders are never mixed. A changed representation and training recipe distinguish this screen from the previous551-input head screen; it is not an isolated ablation proving that context removal alone caused the gain.', '',
             '## Supervision diagnostic','',
             '| Prediction table on support people | Mean base error DeltaE00 |','|---|---:|']
    for a,e in table_errors.items():text.append(f'| {a} | {e:.4f} |')
    text += ['', 'The out_person table has larger base error, as expected for genuinely excluded-person queries. Its head still does not beat the matched inclusion control. These table errors describe supervision, not final head accuracy on new people. The head itself is trained on all18 support people; only its out_person base predictions are OOF.', '',
             '## Risk, compute and previous evidence','',
             'Fixed100/95/90/80/70/60% coverage uses identical nearest-support color36 distance for all arms.72 individual fixed-coverage checks and24 aggregate rows are retained; full per-image curves remain in private reproducible arrays. This is not calibrated expected error or a refusal guarantee. Older reports used a different mean-patch novelty ranking; do not mix their80% figures with this ranking.', '',
             '![Risk curves and individual-seed results](correction_results.png)', '',
             'The same internal cohort previously had ordinary person/site balancing6.0084 and color allocation6.0012 ([allocation report](../skin_color_sampling_v1/report.md)); current in_matched5.6560 is lower, with extra head capacity/optimization. These are historical contextual comparators, not equal-total-budget refits. The independent4.3005 fusion result concerns a different endpoint and is not directly comparable to this internal6.1082 baseline.', '',
             f"Inner-core allocated CUDA training peak{resources['inner_cores']['fit_peak_allocated_mib'][0]:.2f}–{resources['inner_cores']['fit_peak_allocated_mib'][1]:.2f}MiB; cached-head peak{resources['heads']['fit_peak_allocated_mib'][0]:.2f}–{resources['heads']['fit_peak_allocated_mib'][1]:.2f}MiB. Complete core+head checkpoints{int(resources['heads']['checkpoint_bytes'][0])}–{int(resources['heads']['checkpoint_bytes'][1])}bytes, including78 scale scalars and no bank. Training peaks are process allocations with caches; no new isolated inference VRAM, latency or export measurement. Previous phase latency is not transferred to this model.", '',
             '## Verification, prior art and Luma boundary','',
             '9 inner and3 original core replays;9 exclusion/scale checks;4404 prediction routing rows;one complete core andthree complete head refits with exact final state hashes.9 NumPy head replays;12 exact evaluation arrays;9390 scalar CIEDE2000 values and72 coverage checks. Largest NumPy native Lab gap4.02e-6; scalar color gap3.56e-15.', '',
             '[Wolpert,Stacked Generalization,1992](https://www.sciencedirect.com/science/article/pii/S0893608005800231) already describes learning corrections from base predictions on withheld data. The publisher abstract was verified in fresh search; no paper code/weights or author result numbers were adopted. This is a known-method falsifier and measured component improvement, not an invented stacking architecture.', '',
             'For Luma, the result concerns actual instrument-referenced skin color, not illuminant angle. It does not prove ordinary smartphone facial accuracy, unseen-camera reliability, cosmetic decisions or novelty. Independent evidence remains primary4.4570/80%4.1591 versus ordinary fusion4.3005/4.1447. The practical goal of median<=2,p95<=5 at>=80% on unseen ordinary phones is unmet.', '',
             '[Frozen protocol](../../research/skin_crossfit_correction_protocol_v1.md) · [Next decision](../../research/skin_crossfit_correction_next_decision.md)']
    (OUT/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    bound=[Path(__file__),OUT/'source_lock.json',OUT/'results.json',OUT/'audit.json',OUT/'report.md',OUT/'risk_coverage.csv',ROOT/'docs/benchmarks/skin_color_sampling_v1/report.md']
    write(OUT/'summary.json',{'rows':rows,'paired_differences':diff,'resources':resources,'supervision_errors':table_errors,
                            'best_local_mean_gain_percent':improvement,'oof_superiority_confirmed':False,'new_independent_accuracy':False,
                            'bindings':{str(p.relative_to(ROOT)):sha(p) for p in bound}})
    print(json.dumps({'mean_gain_percent':improvement,'paired':diff,'resources':resources}),flush=True)


if __name__=='__main__':main()
