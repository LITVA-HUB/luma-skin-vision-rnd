"""Aggregate audited adapter fits, full risk curves and in-sample residual diagnostic."""
import csv,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_neural_reference_train import OUT,RUN
from skin_neural_reference import ColorAdapter
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write


def main():
    audit=json.loads((OUT/'audit.json').read_bytes());assert audit['status']=='PASS'
    result=json.loads((OUT/'results.json').read_bytes());records=result['records'];arms=('base','residual','mean','affine')
    groups=[];coverage=[];diagnostics=[]
    for protocol in ('mixed','from_SLR','from_ipod'):
        rr=[r for r in records if r['protocol']==protocol]
        for domain in rr[0]['arms']['base']['evaluations']:
            for arm in arms:
                scores=[r['arms'][arm]['evaluations'][domain] for r in rr]
                means=[s['full']['mean'] for s in scores]
                row={'protocol':protocol,'domain':domain,'arm':arm,'mean':float(np.mean(means)),
                     'seed_std':float(np.std(means,ddof=1)),'seed_means':means,
                     'median':float(np.mean([s['full']['median'] for s in scores])),
                     'p95':float(np.mean([s['full']['p95'] for s in scores])),
                     'fraction_above10':float(np.mean([s['full']['fraction_above10'] for s in scores])),
                     'mean_at80':float(np.mean([s['coverage'][3]['mean'] for s in scores]))}
                groups.append(row)
                for i,c in enumerate((1.,.95,.9,.8,.7,.6)):
                    coverage.append({'protocol':protocol,'domain':domain,'arm':arm,'coverage':c,'accepted':scores[0]['coverage'][i]['accepted'],
                        'mean':float(np.mean([s['coverage'][i]['mean'] for s in scores])),
                        'p95':float(np.mean([s['coverage'][i]['p95'] for s in scores]))})
    with (OUT/'risk_coverage.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(coverage[0]));writer.writeheader();writer.writerows(coverage)
    # TRAIN-only explanatory diagnostic. Encoders saw these people; NOT OOF accuracy.
    torch.set_num_threads(4)
    for r in records:
        folder=RUN/f"{r['protocol']}__s{r['seed']}";cache=dict(np.load(folder/'training_cache.npz'))
        x=torch.from_numpy(cache['features']);p=torch.from_numpy(cache['base']);y=torch.from_numpy(cache['target']);g=cache['groups']
        base_mse=float((p-y).square().mean());by_arm={}
        for arm in arms[1:]:
            saved=torch.load(folder/(arm+'.pt'),map_location='cpu',weights_only=True);head=ColorAdapter(arm).eval()
            head.load_state_dict({k.removeprefix('adapter.'):v for k,v in saved['state'].items() if k.startswith('adapter.')})
            corrections=[]
            with torch.no_grad():
                for start in range(0,len(x),64):
                    keep=torch.from_numpy(g[start:start+64,None]!=g[None])
                    corrections.append(head(x[start:start+64],x,y-p,keep))
            cp=torch.cat(corrections);last=json.loads((OUT/f"{r['protocol']}__s{r['seed']}"/(arm+'_history.json')).read_bytes())['loss']
            by_arm[arm]={'person_excluded_bank_train_mse':float((p+cp-y).square().mean()),
                         'mean_standardized_correction_norm':float(cp.norm(dim=1).mean()),
                         'first30_step_loss':float(np.mean(last[:30])),'last30_step_loss':float(np.mean(last[-30:]))}
        diagnostics.append({'protocol':r['protocol'],'seed':r['seed'],'base_train_mse':base_mse,'arms':by_arm})
    allnew=[r['arms'][a] for r in records for a in arms[1:]]
    resources={}
    for a in arms[1:]:
        rows=[r['arms'][a] for r in records]
        resources[a]={k:[float(min(v[k] for v in rows)),float(max(v[k] for v in rows))] for k in ('fit_seconds','fit_peak_allocated_mib','checkpoint_bytes','bank_scalars')}
        for k in ('batch1_median_ms','inference_process_peak_allocated_mib','inference_incremental_peak_mib'):
            resources[a][k]=[float(min(v['runtime'][k] for v in rows)),float(max(v['runtime'][k] for v in rows))]
    trained_lower=sum(v['person_excluded_bank_train_mse']<d['base_train_mse'] for d in diagnostics for v in d['arms'].values())
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    colors={'base':'#222222','residual':'#a761cf','mean':'#1a929c','affine':'#de7431'}
    for ax,(protocol,domain,title) in zip(axes,[('mixed','known','Known cameras'),('from_SLR','unseen','SLR to unseen iPod'),('from_ipod','unseen','iPod to unseen SLR')]):
        visible=[]
        for arm in arms:
            curves=[]
            for r in records:
                if r['protocol']==protocol:curves.append(np.load(RUN/f"{protocol}__s{r['seed']}"/f'{arm}__{domain}.npz')['curve'])
            c=np.mean(curves,axis=0);xx=np.arange(1,len(c)+1)/len(c)
            visible.extend(c[xx>=.6])
            ax.plot(xx,c,label=arm,color=colors[arm])
        lo,hi=min(visible),max(visible);pad=max(.03,(hi-lo)*.08)
        ax.set(xlim=(.6,1),ylim=(lo-pad,hi+pad),xlabel='Accepted fraction',ylabel='Mean native skin DeltaE00',title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Exploratory source validation: mean of 3 individual seeds; common distance ranking')
    fig.savefig(OUT/'risk_coverage.png',dpi=170);plt.close(fig)
    lines=['# Neural-reference skin adapters: added capacity does not give a universal gain','',
           '27 matched adapter fits and9 unchanged compact cores, on original MSKCC CC-BY real photographs with instrument-native skin Lab. Independent TEST/CAL were not opened. Source validation has been heavily reused, including historical selection of these core checkpoints. These are exploratory results; no new independent facial-phone accuracy claim.','',
           '## Measured skin color','',
           'Each entry aggregates three individual seeds, not an ensemble. SD describes seed variability only; median/p95 are means of seed-specific quantiles, not pooled quantiles or confidence intervals. Smaller DeltaE00 is better.','',
           '| Training / domain | System | Mean ± seed SD | Median | p95 | Error >10 | Mean at80% |','|---|---|---:|---:|---:|---:|---:|']
    for r in groups:
        lines.append(f"| {r['protocol']} /{r['domain']} | {r['arm']} | {r['mean']:.4f} ±{r['seed_std']:.4f} | {r['median']:.4f} | {r['p95']:.4f} | {100*r['fraction_above10']:.2f}% | {r['mean_at80']:.4f} |")
    lines += ['', 'The local-affine mechanism beats the ordinary residual C+ head in mixed and forward transfer, but loses on reverse unseen-camera mean. Among the mixed and two unseen-camera endpoints, it improves its unchanged strong core only forward:5.0193 ->4.9658 (about1.07%). Mixed3.4406 ->3.5182 and reverse5.9635 ->6.4782 worsen. Reference mean is better than affine on reverse unseen transfer, but still loses the unchanged core. No universal capacity or mechanism win.', '',
              'Previously locally reproduced stronger transfer recipes remain forward4.8328 ([paired expert](../skin_expert_anchor_v1/report.md)) and reverse4.9736 ([training branch](../skin_train_branch_v1/report.md)). These historical means use different training recipes and are contextual comparators, not new matched refits. Neither is beaten by this phase. No author-reported numbers appear as local results.', '',
              '## What changed and what was held equal','',
              'All arms share a frozen929,297-parameter CaptureColor core and a193,795-parameter head:551->256->192->16->3. Total1,123,092, within the authorized1,129,297 cap. Inputs are512 neural context features,36 image descriptors and3 base color values. All heads start with exactly zero correction and identical weights; each receives the same300 AdamW steps, image draws and standardized Lab MSE supervision. Final adapters are used without validation epoch selection.', '',
              'The ordinary residual head directly predicts a bounded3-vector. Reference heads learn16-dimensional affinities, retrieve residuals from TRAIN skin references and gate either their weighted mean or their locally affine prediction. The affine solver uses normalized weights and0.01 slope regularization. Every training query masks every image of its person from reference memory. Validation people are absent from core training and the bank. One-image inference uses no camera ID or query color reference.', '',
              'This is an implementation of known building blocks. [MetaOptNet,CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Meta-Learning_With_Differentiable_Convex_Optimization_CVPR_2019_paper.html) and [differentiable closed-form solvers,ICLR2019](https://arxiv.org/abs/1805.08136) precede learning embeddings through regression/optimization. No claim of inventing these mechanisms, no imported third-party implementation or weights.', '',
              '## Training residual diagnostic','',
              f'{trained_lower}/27 adapters reduce full TRAIN standardized MSE relative to the frozen core, when the reference bank excludes the query person. But the core itself saw those TRAIN people. This is NOT OOF error and cannot supervise a calibrated uncertainty head. Lower training residuals together with worse source validation show that extra capacity can fit errors without improving transfer. This is consistent with residual overfitting/acquisition bias, but does not identify either as the sole cause.', '',
              'The reference bank contains in-sample residuals of the core. Excluding a query person from the bank does not make that core an excluded-person encoder. The next experiment must investigate this mismatch directly before increasing the head or repeating correction passes. Full diagnostic values are in summary.json.', '',
              '## Risk and coverage','',
              'All arms share nearest-bank standardized descriptor-distance ranking. The report retains360 seed-specific fixed-coverage checks and120 aggregate rows at100/95/90/80/70/60%. Curves average individual-seed risks. This common ranking isolates color changes; it is not an error-calibrated C+ confidence head or a refusal guarantee.', '',
              '![Full risk-coverage curves, displayed from60%](risk_coverage.png)', '',
              '## Measured compute and payload','',
              'CUDA measurements on RTX4060. Cached-head training includes the frozen core and source tensors resident in the process. Batch1 latency includes neural core and reference correction on already prepared GPU tokens/descriptors; JPEG decoding and image-feature extraction are excluded. Process VRAM includes training caches and copies, so it is not isolated production VRAM.', '',
              '| Adapter | Parameters | Head train peak MiB | Batch1 median ms range | Complete checkpoint bytes | Extra reference scalars |','|---|---:|---:|---:|---:|---:|']
    for arm,v in resources.items():
        fmt=lambda k:f"{v[k][0]:.3f}–{v[k][1]:.3f}"
        lines.append(f"| {arm} | 1,123,092 | {fmt('fit_peak_allocated_mib')} | {fmt('batch1_median_ms')} | {int(v['checkpoint_bytes'][0])}–{int(v['checkpoint_bytes'][1])} | {int(v['bank_scalars'][0])}–{int(v['bank_scalars'][1])} |")
    lines += ['', 'Reference memory stores16 embedding plus3 residual scalars per TRAIN photograph; it is included in checkpoint bytes and separately counted. Fixed preprocessing/target scales add1102 scalars. Private original features/identifiers are not part of the deployed checkpoint. No ONNX/TensorRT export or end-to-end phone latency claim.', '',
              '## Verification and Luma boundary','',
              'All9 source feature/core replays match.27 NumPy head replays include2376 independent augmented least-squares solves;36 excluded-person perturbations pass. Three complete affine optimization refits reproduce exact final head hashes.60 deployment prediction arrays reproduce exactly;9504 independent scalar CIEDE2000 cases and360 coverage rows pass. Max NumPy native Lab discrepancy4.77e-6; scalar color discrepancy5.78e-15.', '',
              'The larger model and reference correction are implemented and measured on genuine instrument-referenced skin color. They have not established novel-model superiority or ordinary smartphone facial accuracy. Independent evidence remains primary mean4.4570/80%4.1591 versus ordinary fusion4.3005/4.1447. The desired median<=2 and p95<=5 at>=80% on unseen ordinary phones remains unmet.', '',
              '[Frozen protocol](../../research/skin_neural_reference_protocol_v1.md) · [Next decision](../../research/skin_neural_reference_next_decision.md)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    paths=[Path(__file__),OUT/'results.json',OUT/'audit.json',OUT/'source_lock.json',OUT/'risk_coverage.csv',OUT/'report.md']
    write(OUT/'summary.json',{'groups':groups,'training_diagnostic':diagnostics,'train_mse_improved_fits':trained_lower,'resources':resources,
                            'strongest_neural_baseline_beaten':False,'new_independent_accuracy':False,'fits':27,
                            'bindings':{str(p.relative_to(ROOT)):sha(p) for p in paths}})
    print(json.dumps({'report':str(OUT/'report.md'),'train_mse_improved_fits':trained_lower,'resources':resources}),flush=True)


if __name__=='__main__':main()
