# TRAIN relational compatibility falsifier

This is a source-mechanism probe, not a new independent color-accuracy result. Real original MSKCC CC-BY skin images and native instrument Lab; original TRAIN only. No source VALIDATION, CAL, TEST or additional held-out dataset was loaded.

## Support and controls

24 people,966 images,248 measured sites;1,421 same-site capture pairs and18,229 within-person between-site pairs. Every same-site pair has a wrong-site control from the same person, matching the second image capture mode and image type. There are **zero same-site cross-camera pairs**. Anatomical site identity does not establish pixel registration or identical imaged tissue.

Same-site reference DeltaE00 is zero by construction. Closest wrong-site reference distances range0.2803 to11.8487, median2.4103. They do not exactly match the positive color-distance distribution. Of1,421 pairs,44/209/607/1,319 have a nearest eligible control within0.5/1/2/5 DeltaE00, from5/14/22/24 people. The closest-color control uses reference labels only for diagnostic matching; it is not an inference rule.

The uniform and closest-color control lists contain1,375 and1,222 unique unordered negative pairs. Repeated images/pairs are retained and disclosed. Large pair counts do not create more independent people.

## Does the representation distinguish the same site while preserving color?

Preference is the equal-person mean of the fraction for which the same-site pair has smaller descriptor distance than its matched wrong-site pair; ties count0.5. It is NOT skin-color accuracy. Correlation is equal-person Spearman association between representation distance and actual between-site reference DeltaE00. Arbitrary feature RMS distances are never labelled DeltaE00.

| Representation | Scope | Closest-control preference | Control within1 DeltaE00 | Control within2 DeltaE00 | True-color rank association |
|---|---|---:|---:|---:|---:|
| median_rgb | unfitted descriptor | 59.30% | 57.56% | 59.62% | 0.2278 |
| color36 | unfitted descriptor | 64.85% | 66.15% | 63.48% | 0.2516 |
| patch54 | unfitted descriptor | 63.78% | 68.29% | 64.12% | 0.2201 |
| context_image | TRAIN encoder, descriptive | 70.92% | 61.07% | 71.30% | 0.3686 |
| frozen_lab_image | TRAIN encoder, descriptive | 68.59% | 60.22% | 66.20% | 0.4962 |
| context_combined | TRAIN encoder, descriptive | 69.68% | 59.92% | 69.40% | 0.3576 |
| frozen_lab_combined | TRAIN encoder, descriptive | 70.18% | 63.27% | 69.21% | 0.4924 |
| ridge3 | LOPO linear/mean | 68.33% | 67.86% | 65.51% | 0.3427 |
| ridge36 | LOPO linear/mean | 70.08% | 72.64% | 70.09% | 0.3955 |
| mean_lab | LOPO linear/mean | 50.00% | 50.00% | 50.00% | undefined (constant) |

The frozen learned contexts were trained on all original TRAIN people, including those whose pairs are analysed. Excluding a person from feature standardization does not remove encoder label exposure. Their apparent advantage is descriptive only. The48 linear ridge fits exclude the entire evaluated person, including all their sites, from every fit and normalization. Original TRAIN has still been reused across R&D: these folds are exploratory cross-validation, not a new confirmatory cohort.

Ridge36 gives70.08% closest-control preference and0.3955 color association; the all-TRAIN learned context gives70.92% and0.3686. For controls within1 DeltaE00, ridge36 gives72.64% versus context61.07%, on209 dependent pairs from14 people. Thus this probe does not establish a special comparative representation advantage beyond a simple color predictor. It also does not prove that every relational model must fail. Texture/site recognition and color differences remain potential explanations.

## Actual direct skin-color error of the excluded-person controls

Each of24 people is predicted by a ridge model fitted on the other23. All966 predictions are scored against genuine native instrument Lab. No camera identity or capture metadata enters these regressions.

| LOPO control | Mean DeltaE00 | Median | p95 | Equal-person mean |
|---|---:|---:|---:|---:|
| ridge3 | 6.0395 | 5.3623 | 11.9183 | 6.0176 |
| ridge36 | 5.4259 | 4.7431 | 11.4364 | 5.3388 |
| mean_lab | 10.9115 | 9.8601 | 22.6492 | 11.0235 |

These numbers cannot be ranked against the separate400-image independent test as if the population/protocol were the same. No new risk-coverage or unseen-camera result is established. Good relative pair ranking is insufficient for accurate absolute skin color: ridge36 still has mean5.4259 and p9511.4364 DeltaE00 here.

## Mathematical limit and prior art

For additive comparison d(x,a)=f(x)-f(a), uniformly averaging y(a)+d(x,a) is exactly f(x)+mean(y(a)-f(a)). Repeated comparisons only add a constant offset. Complete antisymmetric zero-cycle comparisons recover an ordinary potential up to a constant. The actual frozen predictor identity checks differ by at most 1.42e-14 and7.11e-15. Tests include a pure cyclic counterexample; this is algebra, not synthetic accuracy.

[Relation Networks, CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Sung_Learning_to_Compare_CVPR_2018_paper.html) already learn comparison metrics from episodes. [Deep Kernel Learning, AISTATS2016](https://proceedings.mlr.press/v51/wilson16.html) combines learned representations and kernels. [HodgeRank](https://arxiv.org/abs/0811.1067) separates potential and cyclic comparison components. These are prior art, not our inventions or locally reproduced author methods. No third-party implementation or weights were adopted.

## Verification and next decision

Audit PASS: 13,511 independent candidate checks,48 augmented least-squares refits,4 held-target perturbations,2 context replays,100 preference checks, 250 rank checks and151,361 scalar color cases. Maximum scalar gap1.24e-14; coefficient gap2.59e-11. All original neural checkpoints and cache hashes remain unchanged.

Do not promote another pair network from same-site recognition alone. A narrower next test is ordinary query-dependent reference weighting, measured directly by excluded-person absolute skin DeltaE00 with global-ridge and weighted-mean controls. This is a standard local-regression comparator, not novelty. It tests whether compatibility weighting contributes beyond the degenerate constant offset before spending on a learned pair mechanism.

Independent MSKCC primary4.4570/80%4.1591 versus ordinary fusion4.3005/4.1447 is unchanged. No strongest independent-baseline win, ordinary-phone facial validation, novel compact architecture or calibrated refusal guarantee. No new model VRAM/latency/export claim. Goal active and unmet.

![Comparative signal and actual color error](relational_probe.png)
