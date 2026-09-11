"""Positive person-excluded color signal and failed strong camera comparison."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_local_reference_run import OUT as SCREEN,RUN as SCREEN_RUN
from skin_local_reference_transfer import OUT as TRANSFER,RUN as TRANSFER_RUN
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write


def main():
    s=json.loads((SCREEN/'results.json').read_bytes());t=json.loads((TRANSFER/'results.json').read_bytes())
    sa=json.loads((SCREEN/'audit.json').read_bytes());ta=json.loads((TRANSFER/'audit.json').read_bytes())
    for audit in (sa,ta):
        assert audit['status']=='PASS'
        for p,h in audit['bindings'].items():assert sha(ROOT/p)==h
    a=next(m for m in s['methods'] if m['method']=='global_ridge');b=next(m for m in s['methods'] if m['method']=='color_affine')
    improvement=100*(a['full']['mean']-b['full']['mean'])/a['full']['mean']
    lines=['# Local-reference skin color: useful component, no universal win',
        '', 'All errors below are actual CIEDE2000 against original MSKCC instrument-native skin Lab. '
        'Original release CC-BY; no new datasets, third-party code/weights, proprietary collection or cloud. '
        'The independent400-image TEST and208-image CAL were not opened. Original source data have been '
        'extensively reused, so these are exploratory results, not new independent facial-phone accuracy.',
        '', '## Mechanism and exact controls',
        '', 'One input image supplies36 color descriptors. A bank of other training people supplies '
        'fixed descriptors and genuine reference Lab. Appearance weights use Gaussian RMS descriptor '
        'distance, bandwidth1. Color weights use Gaussian DeltaE00 distance between global-ridge '
        'estimated query and bank colors, bandwidth5. No actual query color, camera identity or '
        'test-time calibration enters the method. Both affinities use identical weighted-mean and '
        'local-affine controls. All parameters/configurations were frozen before outputs.',
        '', 'Local affine regression estimates a color relationship near each query; it is not '
        'merely copying similar reference colors. Weights sum to bank size and ridge penalty stays1. '
        'Uniform weights recover ordinary global ridge. This is established '
        '[locally weighted learning](https://link.springer.com/article/10.1023/A:1006559212014), '
        'not a claimed new architecture or a reproduction of an author benchmark table.',
        '', '## Person-excluded TRAIN screen',
        '', '24 folds; each evaluated person and every one of their sites is excluded from the '
        'bank, scales and all fits.966 real photographs. The original global ridge and mean '
        'controls replay exactly from the preceding relational probe.',
        '', '| System | Mean DeltaE00 | Median | p95 | Error>10 | Mean at80% | People improved vs ridge |',
        '|---|---:|---:|---:|---:|---:|---:|']
    csvrows=[]
    for m in s['methods']:
        f=m['full'];lines.append(f"| {m['method']} | {f['mean']:.4f} | {f['median']:.4f} | {f['p95']:.4f} | {100*f['fraction_above10']:.2f}% | {m['coverage'][3]['mean']:.4f} | {m['person_improved_vs_ridge']}/24 |")
        for c in m['coverage']:csvrows.append({'phase':'TRAIN_LOPO','protocol':'person_excluded','domain':'held_person','method':m['method'],**c})
    lines += ['', f"Color-affine improves the mean by{improvement:.2f}% versus global ridge and improves21/24 "
        'people. The mean-only color bank is worse than ridge, so this local result supports '
        'the affine mapping component rather than reference averaging alone. The source folds '
        'share training people and have been explored historically; this is not a significance '
        'claim or victory over the separate independent-test neural models.',
        '', '## Unchanged method on explicit source camera-held-out banks',
        '', 'After the screen passed its independent numeric audit, all six methods were frozen '
        'unchanged for source transfer. Mixed uses24TRAIN/6VALIDATION people; SLR-only8TRAIN '
        'and iPod-only16TRAIN each have3 same-camera and3 unseen-camera validation people. '
        'No unseen device occurs in its training bank. Device and person/capture effects are '
        'confounded; these clinical SLR/iPod photographs are not ordinary uncalibrated phone selfies.',
        '', '| Training bank / domain | System | Mean DeltaE00 | p95 | Mean at80% |',
        '|---|---|---:|---:|---:|']
    for r in t['results']:
        lines.append(f"| {r['protocol']} /{r['domain']} | {r['method']} | {r['full']['mean']:.4f} | {r['full']['p95']:.4f} | {r['coverage'][3]['mean']:.4f} |")
        for c in r['coverage']:csvrows.append({'phase':'source_transfer','protocol':r['protocol'],'domain':r['domain'],'method':r['method'],**c})
    lines += ['', 'Color-affine improves global ridge in both unseen directions, but appearance-affine '
        'worsens both. On reverse transfer, appearance-weighted mean6.6968 is stronger than '
        'color-affine7.4895; retain that result rather than selecting only the favored mechanism.',
        '', '## Stronger previously locally reproduced neural comparators',
        '', '| Source protocol | Best system in this six-method phase | Earlier strong compact neural result |',
        '|---|---:|---:|',
        '| Mixed known cameras | color_affine3.9343 | mixture3.4406 |',
        '| SLR to unseen iPod | color_affine7.1177 | paired mixture4.8328 |',
        '| iPod to unseen SLR | appearance_mean6.6968 | training-only graph4.9736 |',
        '', 'Historical figures are means of three individual seed scores, reproduced locally '
        'in [capture](../skin_capture_v1/report.md), [paired expert](../skin_expert_anchor_v1/report.md) '
        'and [training branch](../skin_train_branch_v1/report.md) experiments. They use the same '
        'source people but different capacity, optimization and source checkpoint selection. '
        'These are contextual strong comparators, not newly refitted or capacity/budget-matched C+. '
        '**No strongest-neural-baseline win is established.** No author-reported accuracy is '
        'being inserted as a local reproduction.',
        '', '## Risk and coverage',
        '', 'All six systems use identical accepted-image sets per protocol, ranked only by '
        'nearest-bank descriptor distance. This tests the color mapping at fixed coverage; '
        'it is not a calibrated expected-error head or a per-image guarantee. All216 coverage '
        'rows and full curves are retained. For color-affine, unseen forward80%7.3607 exceeds '
        'full7.1177; reverse80%7.9720 exceeds full7.4895. Novelty ranking therefore fails to '
        'provide reliable selective improvement under these shifts.',
        '', '![Screen and camera-transfer risks](local_reference_results.png)',
        '', '## Verification, compute and deployment scope',
        '', f"Both audits PASS. Combined{sa['weighted_augmented_solves']+ta['weighted_augmented_solves']} "
        f"independent weighted augmented solves,{sa['uniform_limit_checks']+ta['uniform_limit_checks']} "
        f"uniform-limit checks,{sa['scalar_color_cases']+ta['scalar_color_cases']:,} scalar color cases "
        'and216 coverage checks. Two previous full controls replay exactly; two held-person '
        'reference perturbations cannot affect predictions. Thirty source method/domain arrays '
        'and camera exclusions replay. All original caches/checkpoints remain unchanged.',
        '', 'No new neural parameters were trained in these linear-reference experiments.27 '
        'global ridge fits were used across the screen/follow-up, with3516 nonuniform local '
        'affine solves,3516 weighted means and1758 additional uniform-limit diagnostic solves. '
        'Reference-bank scalars, base coefficients and normalizations must count toward a '
        'future complete model payload. No new RTX VRAM, batch1 latency, ONNX or TensorRT '
        'claim follows from CPU algebra or cached descriptors.',
        '', '## Next decision and Luma limits',
        '', 'Keep the positive local-affine component, but combine it with a strong compact '
        'neural representation before repeating correction on the weak raw-descriptor mapping. '
        'The user has authorized up to200,000 additional neural parameters. The next single '
        'prototype uses the929,297-parameter base and at most1,129,297 neural parameters total, '
        'with an ordinary equal-capacity residual-head C+ control and full accounting of any '
        'reference bank. This permission is a compute budget, not a result.',
        '', 'Independent MSKCC remains primary4.4570/80%4.1591 versus ordinary fusion4.3005/4.1447. '
        'No new independent skin accuracy, universal camera behavior, ordinary-phone facial '
        'validation or defensible novel-method advantage. Goal active and unmet.']
    with (TRANSFER/'risk_coverage.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.3),layout='constrained')
    for name in ('global_ridge','appearance_affine','color_affine'):
        z=np.load(SCREEN_RUN/(name+'.npz'));c=np.arange(1,len(z['curve'])+1)/len(z['curve']);keep=c>=.6
        axes[0].plot(c[keep],z['curve'][keep],label=name)
    axes[0].set_title('TRAIN person-excluded');axes[0].set_ylabel('Mean actual skin DeltaE00');axes[0].legend(fontsize=7)
    for ax,protocol in zip(axes[1:],('from_SLR','from_ipod')):
        for name in ('global_ridge','appearance_mean','appearance_affine','color_affine'):
            z=np.load(TRANSFER_RUN/f'{protocol}__unseen__{name}.npz');c=np.arange(1,len(z['curve'])+1)/len(z['curve']);keep=c>=.6
            ax.plot(c[keep],z['curve'][keep],label=name)
        ax.set_title(protocol+' / unseen source camera');ax.legend(fontsize=7)
    for ax in axes:ax.set_xlabel('Accepted fraction (shared ranking)');ax.grid(alpha=.15)
    fig.suptitle('Local skin-color improvement does not yet transfer competitively to unseen cameras',fontsize=12)
    fig.savefig(TRANSFER/'local_reference_results.png',dpi=160);plt.close(fig)
    (TRANSFER/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (SCREEN/'report.md').write_text('# Person-excluded local reference screen\n\nThe complete screen, unchanged camera follow-up and both positive/negative results are in the [combined report](../skin_local_reference_transfer_v1/report.md).\n\nTRAIN-only mean5.42585 to4.58597 DeltaE00;21/24 people improve. This does not beat strong neural camera-transfer comparators or change independent skin accuracy.\n',encoding='utf8')
    write(TRANSFER/'summary.json',{'screen_mean_improvement_percent':improvement,'screen_people_improved':21,'strong_neural_baseline_beaten':False,
        'additional_neural_parameters_authorized':200000,'next_base_parameters':929297,'next_neural_parameter_cap':1129297,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),SCREEN/'results.json',SCREEN/'audit.json',TRANSFER/'results.json',TRANSFER/'audit.json',
            ROOT/'docs/benchmarks/skin_capture_v1/report.md',ROOT/'docs/benchmarks/skin_expert_anchor_v1/report.md',ROOT/'docs/benchmarks/skin_train_branch_v1/report.md']}})
    print(str(TRANSFER/'report.md'))


if __name__=='__main__':main()
