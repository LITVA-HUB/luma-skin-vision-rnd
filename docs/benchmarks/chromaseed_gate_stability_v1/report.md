# ChromaSeed-GS: frozen-model color and gate stability

**Hard routing adds a real numerical discontinuity, while overall color sensitivity affects every control.** On534 deliberately constructed legal boundary pairs, the perceptual hard model's person-mean jump is1.9279 ΔE00 versus0.0110 for soft, with a maximum7.8176 across seeds. The pair separation in encoded RGB is at most0.0002. However, most boundaries require a larger initial transformation: median15.69% color mixture from the original input. This is a mechanism stress test, not an estimate of everyday failure probability.

No models were fitted or selected. All72 [G models](../chromaseed_gated_v1/report.md) were frozen before33 transformations each. Original TRAIN only,24 people overall, mixed held subset232 rows/six people; SLR→iPod643/16 and reverse323/8. These reused roles overlap. Three seed results average errors/statistics and do not create new people or an inference ensemble.

## Transformation and interpretation

Apply v'=(1−t)v+t*a to encoded RGB pixels, where a is one of the eight cube corners and t is1/255,4/255,16/255 or64/255, plus one identity. Every pixel stays in[0,1] without clipping. Quantiles/means receive the same affine map, standard deviations scale by1−t, correlations stay unchanged. Tests compare this feature update with explicit synthetic pixel transformations, including constant channels. Actual photos are not decoded or modified here.

For each dose, take the worst error or output drift across eight anchors separately for each row, then average within people and across people. Different rows may have different worst anchors. The original instrument target stays fixed under synthetic corruption; that is a stress-test assumption, not new instrument measurements of transformed photographs.25.098% is an especially strong stress condition. All doses and controls are retained, including when hard remains better on a dose-wise mean.

![Stability diagnosis](gate_stability.png)

## All families: clean and worst-case reference error

Values are mean per-person ΔE00 averaged across seeds. Smaller is better. Synthetic worst-case scores must not be compared as if they were ordinary held-photo accuracy. Routed transfer models have no gate and are exact base aliases because the original fit had a single camera group.

| Role | Family | Original | 1/255 | 4/255 | 16/255 | 64/255 |
|---|---|---:|---:|---:|---:|---:|
| Mixed | norm_base | 5.43865 | 5.66528 | 6.39770 | 9.75930 | 24.40036 |
| Mixed | norm_uniform | 5.38439 | 5.62068 | 6.38762 | 9.92017 | 25.28959 |
| Mixed | norm_soft | 5.34902 | 5.57924 | 6.32850 | 9.78452 | 24.88204 |
| Mixed | norm_hard | 5.32267 | 5.55550 | 6.30010 | 9.76665 | 25.19345 |
| Mixed | perceptual_base | 5.39007 | 5.62063 | 6.36767 | 9.80445 | 24.65591 |
| Mixed | perceptual_uniform | 5.36166 | 5.60088 | 6.37782 | 9.94506 | 25.26538 |
| Mixed | perceptual_soft | 5.29512 | 5.52850 | 6.29059 | 9.80635 | 25.10125 |
| Mixed | perceptual_hard | 5.28510 | 5.52014 | 6.27800 | 9.80814 | 25.40082 |
| SLR → iPod | norm_base | 8.59700 | 8.73759 | 9.16801 | 10.98200 | 18.91509 |
| SLR → iPod | norm_uniform | 8.81846 | 8.97763 | 9.46406 | 11.48746 | 20.06838 |
| SLR → iPod | norm_soft | 8.59700 | 8.73759 | 9.16801 | 10.98200 | 18.91509 |
| SLR → iPod | norm_hard | 8.59700 | 8.73759 | 9.16801 | 10.98200 | 18.91509 |
| SLR → iPod | perceptual_base | 8.65480 | 8.79748 | 9.23448 | 11.07693 | 19.04825 |
| SLR → iPod | perceptual_uniform | 8.68422 | 8.84375 | 9.32990 | 11.33489 | 19.57247 |
| SLR → iPod | perceptual_soft | 8.65480 | 8.79748 | 9.23448 | 11.07693 | 19.04825 |
| SLR → iPod | perceptual_hard | 8.65480 | 8.79748 | 9.23448 | 11.07693 | 19.04825 |
| iPod → SLR | norm_base | 8.70502 | 8.89857 | 9.51912 | 12.42321 | 21.96572 |
| iPod → SLR | norm_uniform | 8.74229 | 8.95003 | 9.61525 | 12.72508 | 22.92760 |
| iPod → SLR | norm_soft | 8.70502 | 8.89857 | 9.51912 | 12.42321 | 21.96572 |
| iPod → SLR | norm_hard | 8.70502 | 8.89857 | 9.51912 | 12.42321 | 21.96572 |
| iPod → SLR | perceptual_base | 8.38689 | 8.58254 | 9.20950 | 12.14965 | 22.07537 |
| iPod → SLR | perceptual_uniform | 8.58575 | 8.81311 | 9.53732 | 12.87690 | 23.88457 |
| iPod → SLR | perceptual_soft | 8.38689 | 8.58254 | 9.20950 | 12.14965 | 22.07537 |
| iPod → SLR | perceptual_hard | 8.38689 | 8.58254 | 9.20950 | 12.14965 | 22.07537 |

## All families: worst-case answer drift from original

Same aggregation; these values measure changing answers, not their correctness. A constant predictor would have zero drift but could be useless. Therefore stability cannot replace the original target-error controls.

| Role | Family | 1/255 | 4/255 | 16/255 | 64/255 |
|---|---|---:|---:|---:|---:|
| Mixed | norm_base | 0.33780 | 1.36375 | 5.60423 | 21.66107 |
| Mixed | norm_uniform | 0.35323 | 1.42568 | 5.85892 | 22.62223 |
| Mixed | norm_soft | 0.34807 | 1.40519 | 5.77925 | 22.40378 |
| Mixed | norm_hard | 0.34985 | 1.40453 | 5.77496 | 22.74650 |
| Mixed | perceptual_base | 0.34267 | 1.38297 | 5.67998 | 21.83040 |
| Mixed | perceptual_uniform | 0.35528 | 1.43357 | 5.88230 | 22.52323 |
| Mixed | perceptual_soft | 0.35091 | 1.41728 | 5.83504 | 22.53903 |
| Mixed | perceptual_hard | 0.35218 | 1.41772 | 5.83676 | 22.88771 |
| SLR → iPod | norm_base | 0.23768 | 0.95697 | 3.90437 | 15.30820 |
| SLR → iPod | norm_uniform | 0.27562 | 1.11069 | 4.55146 | 18.20195 |
| SLR → iPod | norm_soft | 0.23768 | 0.95697 | 3.90437 | 15.30820 |
| SLR → iPod | norm_hard | 0.23768 | 0.95697 | 3.90437 | 15.30820 |
| SLR → iPod | perceptual_base | 0.24508 | 0.98568 | 4.00503 | 15.33701 |
| SLR → iPod | perceptual_uniform | 0.28036 | 1.12885 | 4.60698 | 18.01882 |
| SLR → iPod | perceptual_soft | 0.24508 | 0.98568 | 4.00503 | 15.33701 |
| SLR → iPod | perceptual_hard | 0.24508 | 0.98568 | 4.00503 | 15.33701 |
| iPod → SLR | norm_base | 0.31533 | 1.28275 | 5.35820 | 16.97143 |
| iPod → SLR | norm_uniform | 0.34119 | 1.38711 | 5.79884 | 18.37424 |
| iPod → SLR | norm_soft | 0.31533 | 1.28275 | 5.35820 | 16.97143 |
| iPod → SLR | norm_hard | 0.31533 | 1.28275 | 5.35820 | 16.97143 |
| iPod → SLR | perceptual_base | 0.32211 | 1.30884 | 5.45167 | 17.13445 |
| iPod → SLR | perceptual_uniform | 0.36875 | 1.49766 | 6.23259 | 19.67748 |
| iPod → SLR | perceptual_soft | 0.32211 | 1.30884 | 5.45167 | 17.13445 |
| iPod → SLR | perceptual_hard | 0.32211 | 1.30884 | 5.45167 | 17.13445 |

At16/255, the perceptual base worst-case error is9.8045, soft9.8064 and hard9.8081. Soft's continuity removes the routing jump but does not solve the common color-shift problem. At1/255 and4/255, hard's worst-case person means remain slightly better than soft's; no blanket measured dominance is claimed.

## Fixed-dose sign changes

The active soft and hard gates share the same frozen score. Their sign-flip indicator is identical, but soft predictions vary continuously. There is no gate in the other role payloads; absence of gate metrics is not proof of unfamiliar-camera robustness.

| Strength | Query rows flipping under any anchor | Equal-person mean | SLR / iPod person means | Boundary roots below strength |
|---|---:|---:|---:|---:|
| 1/255 | 1/232 | 0.379% | 1.136% / 0.000% | 2 |
| 4/255 | 2/232 | 0.758% | 2.273% / 0.000% | 5 |
| 16/255 | 24/232 | 9.414% | 16.667% / 5.788% | 65 |
| 64/255 | 185/232 | 79.327% | 67.929% / 85.025% | 534 |

## Legal boundary pairs

Find the affine gate-score root along every registered anchor direction, then query t−0.0001 and t+0.0001 only when both lie within the registered0–64/255 path. All534 pairs cross the actual FP32 gate. All six model/seed input-pair arrays match exactly, so3,204 pair evaluations represent534 unique query/anchor pairs,185 original rows and six people. Directions and rows repeat people; no independent-sample uncertainty claim is made.

Root mixture quantiles minimum/5th/median/95th/maximum: 0.258%, 3.161%, 15.694%, 23.954%, 24.969%. Only two roots are below1/255 and five below4/255. The tiny separation of the pair is not its distance from the original image, and the continuous feature construction is not a tested8-bit/JPEG/camera pipeline.

| Base loss | Output control | Person-mean jump | Image-mean jump | Mean seed p90 | Maximum across seeds |
|---|---|---:|---:|---:|---:|
| norm | base | 0.010794 | 0.010913 | 0.018766 | 0.025667 |
| norm | uniform | 0.011311 | 0.011389 | 0.019808 | 0.026554 |
| norm | soft | 0.010841 | 0.010966 | 0.019027 | 0.025754 |
| norm | hard | 2.026345 | 1.712969 | 3.359925 | 8.818099 |
| perceptual | base | 0.010894 | 0.010916 | 0.018967 | 0.025599 |
| perceptual | uniform | 0.011256 | 0.011269 | 0.019643 | 0.026368 |
| perceptual | soft | 0.010986 | 0.010996 | 0.019098 | 0.025728 |
| perceptual | hard | 1.927893 | 1.646498 | 3.230529 | 7.817570 |

Boundary summaries average pair errors within each person, then across people, then seeds. All four controls receive identical input pairs. No target label or output-error maximization was used to find these roots.

## Unconstrained feature-space boundaries

Projection to the nearest normalized color36 hyperplane is a separate diagnostic. A projected feature vector can violate quantile/bounds/variance requirements, and passing those necessary checks still does not prove a realizable image. The RMS distance is not a certified pixel robustness radius. Hard's analytic two-sided output limit is evaluated at one fixed projected kernel vector.

| Loss | Seed | Mean person RMS distance | Rows passing basic checks | Mean person limit jump | Max limit jump |
|---|---:|---:|---:|---:|---:|
| norm | 17 | 0.134749 | 185/232 | 1.038901 | 3.103398 |
| norm | 29 | 0.134749 | 185/232 | 1.036260 | 2.996774 |
| norm | 43 | 0.134749 | 185/232 | 1.036175 | 3.079604 |
| perceptual | 17 | 0.134749 | 185/232 | 1.080845 | 3.199779 |
| perceptual | 29 | 0.134749 | 185/232 | 1.099353 | 3.022444 |
| perceptual | 43 | 0.134749 | 185/232 | 1.085113 | 2.993824 |

## Decision

Retain the soft gate as the continuous research option alongside its shared base; keep hard as a measured comparison rather than declaring it the default because of a0.0100 clean-error advantage on six people. This is an engineering continuity choice, not a new claim of superior phone accuracy. Numeric payload21,973B and prior~0.012ms prepared-feature response are unchanged; GS does not refit weights or remeasure production latency.

The next actual learning experiment should address sensitivity with mild fit-side affine color augmentation and an analytical compact readout. Include identical no-augmentation and stronger-ridge controls, preserve the clean-error references and compare to a constant-target baseline so merely flattening the predictor cannot count as success. Keep augmentation generation/weights/selection entirely inside person-disjoint fit folds. Soft conditional readouts and static controls should share the same128-center basis; no query-time learning or synthetic-to-real guarantee. This next series is planned, not launched.

## Verification and limits

Eight numerical tests passed in1.51s after import formatting and before the primary run. The primary diagnostic took5.474s excluding interpreter imports and auditing; this is not training or deployment time. Primary PID39468 returned exit0. Independent audit session1166 returned exit0 and took14.338s:72 models,2,376 transform cases/948,816 actual standalone query predictions,288 dose summaries with every person/camera metric,3,204 legal pair evaluations/25,632 matched control predictions and1,392 unconstrained projections.

Maximum ordinary native-Lab prediction drift 2.92e-12; boundary prediction drift 1.8e-12; root drift 7.49e-16; normalized projection drift 4.44e-16. The audit uses separate feature algebra, the verified NumPy-only consumer, direct kernels and independent metric/projection constructions, while sharing the verified ΔE00 formula and original query indexing. All33 zero/positive transformations and all registered strengths remain available in the local result archive.

Only the original TRAIN arrays and frozen G artifacts were read. No images/tokens, legacy validation/calibration/test, downloads, new measured target pairs, delegation or publication. Camera groups contain different people; synthetic transforms do not establish real acquisition causality, demographic coverage or ordinary-phone face/shade-match accuracy. The full user goal remains active.

[Protocol](../../research/chromaseed_gate_stability_v1_protocol.md) · [Reproduce](reproduce.md) · [Audit](audit.json) · [Verification](verification.json) · [Summary](summary.json) · [Next learning decision](../../research/chromaseed_gate_stability_next_decision.md)
