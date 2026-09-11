# Stable correction camera transfer: internal gain does not generalize

36 new inner-core fits and27 unchanged correction-head fits are complete. Nine original fixed-final930-step cores remain unchanged. Original MSKCC CC-BY real photographs and instrument-native Lab; source TRAIN24people/966photos and VALIDATION6/264. Independent TEST/CAL stay closed. Source validation is heavily reused; these are exploratory results, not fresh independent ordinary-phone accuracy.

## Actual skin-color results

Entries average three individual seed scores, not ensemble predictions. Median/p95 are averages of per-seed quantiles, not pooled quantiles.

| Training / evaluation | Arm | Mean DeltaE00 | Median | p95 | Error >10 | Mean at80% |
|---|---|---:|---:|---:|---:|---:|
| mixed /known | base | 3.8277 | 3.4822 | 7.9298 | 1.14% | 3.6464 |
| mixed /known | in_full | 3.7626 | 3.3775 | 7.9668 | 1.26% | 3.5790 |
| mixed /known | in_matched | 3.8184 | 3.4257 | 7.9302 | 1.39% | 3.6254 |
| mixed /known | out_person | 3.9889 | 3.6233 | 7.6789 | 1.01% | 3.8151 |
| from_SLR /known | base | 3.4210 | 3.2170 | 6.4298 | 0.00% | 3.2236 |
| from_SLR /known | in_full | 3.5002 | 3.2644 | 6.8836 | 0.51% | 3.2342 |
| from_SLR /known | in_matched | 3.5041 | 3.3133 | 6.3971 | 0.00% | 3.2683 |
| from_SLR /known | out_person | 3.5168 | 2.9690 | 7.0893 | 0.00% | 3.3021 |
| from_SLR /unseen | base | 5.4877 | 5.2429 | 10.3253 | 6.57% | 5.4344 |
| from_SLR /unseen | in_full | 5.9953 | 5.9363 | 11.3036 | 10.86% | 5.8908 |
| from_SLR /unseen | in_matched | 5.7371 | 5.4933 | 10.7327 | 7.58% | 5.6632 |
| from_SLR /unseen | out_person | 6.0910 | 5.8064 | 11.1892 | 11.11% | 6.3439 |
| from_ipod /known | base | 4.0558 | 3.2771 | 9.2260 | 2.27% | 4.0294 |
| from_ipod /known | in_full | 3.9347 | 3.2037 | 8.7666 | 1.77% | 3.9080 |
| from_ipod /known | in_matched | 3.9810 | 3.3729 | 8.7457 | 1.26% | 3.9246 |
| from_ipod /known | out_person | 3.9524 | 3.5089 | 8.4998 | 2.27% | 3.9364 |
| from_ipod /unseen | base | 6.5602 | 6.2911 | 10.7986 | 7.07% | 6.5215 |
| from_ipod /unseen | in_full | 7.2462 | 6.8153 | 12.7450 | 17.42% | 7.8330 |
| from_ipod /unseen | in_matched | 7.2908 | 7.0333 | 12.2689 | 15.91% | 7.7503 |
| from_ipod /unseen | out_person | 7.6725 | 7.2741 | 12.7549 | 21.21% | 8.1845 |

The previously leading in_matched head worsens unseen forward5.4877 ->5.7371 and reverse6.5602 ->7.2908. All three head arms worsen both unseen-camera means. The minor mixed-camera gain for in_full3.8277 ->3.7626 does not rescue the generalization hypothesis. Do not choose different head winners per camera and describe the result as camera-blind.

The earlier internal mean6.1082 ->5.6560 is retained as a positive result on its different six-person cohort. This follow-up refutes its extension to a universal transfer gain. Stronger historical source recipes remain mixed3.4406 ([capture](../skin_capture_v1/report.md)), forward4.8328 ([paired expert](../skin_expert_anchor_v1/report.md)), reverse4.9736 ([training branch](../skin_train_branch_v1/report.md)). They use different budgets/selection and are contextual locally reproduced comparators; none is beaten here. No author numbers are passed off as our measurements.

## What was actually controlled

The head is unchanged:39 stable color inputs ->384->384->64->3,188,035 parameters added to929,297 core; total1,117,332 under1,129,297 cap. One image/core/head at inference,78 scale scalars and no reference memory or camera ID. All arms start at zero correction with identical weights and receive the same300 updates/data draws per protocol/seed.

Four inner folds exclude whole people; assignments agree between mixed and single-camera banks. The36 inner cores use the prior930-step recipe unchanged. in_full uses predictions of the core that saw the full training bank; in_matched uses a smaller core that saw each query person; out_person uses the corresponding core that excluded the query person. Matched inclusion and exclusion share exactly the same inner cores/budgets. All heads deploy on their unchanged full-bank core; the3/4-to-full training-support shift remains a limitation.

Source unseen cameras are absent from all fitting rows and fitted scales. Person/camera/capture are confounded: this is valid device exclusion but not a causal camera-only intervention. The clinical SLR/iPod imagery is not a normal facial-selfie benchmark.

## Risk and coverage

All four arms use identical nearest-TRAIN color36 distance rankings. All360 per-seed fixed-coverage rows and120 aggregate rows at100/95/90/80/70/60% are retained, with full risk curves. This is not calibrated expected error, matched confidence-head superiority or a reliable refusal guarantee.

![Source risk-coverage curves](risk_coverage.png)

## Reproducibility recovery and audit

The initial frozen run stopped after16 inner cores and9 mixed heads because concatenated validation batching changed four historical baseline values by at most3.8146973e-6 Lab. Separate known/unseen batch32 processing reproduced both original arrays bit for bit. The recovery restores these original domain batch boundaries, without changing any architecture, fit, hyperparameter, candidate or metric. The exact assertion was retained. Original source_lock.json, completed fits and the pretest Git checkpoint are preserved; recovery_lock.json records the amended runner/test and recovery note. No unseen correction result had been evaluated before the repair.

[Recovery record](../../research/skin_correction_transfer_batch_recovery.md). Final audit passes36 inner-core replays,15 original domain replays,36 exclusion/scale checks,11592 routing rows,three complete core refits andthree complete head refits.27 NumPy heads and60 exact evaluation arrays replay;26892 scalar CIEDE2000 values and360 coverage checks pass. Max NumPy native Lab discrepancy4.12e-6; scalar color discrepancy4.89e-15.

## Measured resources and decision

Allocated cached training peak: inner cores104.42–109.50MiB, heads73.81–76.78MiB. Complete deployed checkpoints4477653–4477653bytes. These are process allocations with cached data; no isolated production VRAM or end-to-end inference timing claim. No export optimization is justified by this failed transfer screen.

Stop promoting this frozen-core correction family as camera-general. Preserve the internal component gain and current negative transfer result. A next experiment should challenge the need for an output correction head and investigate image information or a jointly learned representation, while checking previous graph, histogram, spectral and pixel negatives to avoid repetition. This screen does not prove every such representation will fail.

For Luma, these are measurements of genuine skin-color error. They do not establish ordinary smartphone facial accuracy, reliable cosmetic decisions or a novel mechanism. Known stacking is prior art. Independent primary4.4570/80%4.1591 versus ordinary fusion4.3005/4.1447 stays unchanged. The product aspiration median<=2,p95<=5 at>=80% on unseen ordinary phones remains unmet.

[Frozen protocol](../../research/skin_correction_transfer_protocol_v1.md) · [Next decision](../../research/skin_correction_transfer_next_decision.md)
