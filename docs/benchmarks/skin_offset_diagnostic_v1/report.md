# Privileged source-reference offset diagnostic

**These are reference-calibration comparators, not improved single-image model scores.**
Original90 source prediction arrays are unchanged. For each evaluation person,
the offset uses other evaluation people's native instrument references. This
violates the no-test-camera-calibration deployment condition by design and is
used only to diagnose errors. Source cohorts are reused, not fresh independent
validation. No TEST/CAL archive, new images or neural fitting was used.

All errors below are actual skin CIEDE2000. Group values average three separate
runs. Half/full strengths were fixed before evaluation; neither is selected.

| Protocol / domain / model | Original mean | Privileged half | Privileged full | Original80% | Privileged full80% |
|---|---:|---:|---:|---:|---:|
| mixed/known/image | 3.8277 | 3.6541 | 3.5990 | 3.8015 | 3.5629 |
| mixed/known/person_site | 3.8120 | 3.6729 | 3.6259 | 3.7878 | 3.5943 |
| mixed/known/site | 3.7959 | 3.6345 | 3.5792 | 3.7675 | 3.5402 |
| mixed/known/color | 3.7864 | 3.6950 | 3.6712 | 3.7591 | 3.6276 |
| mixed/known/color_ipw | 3.7976 | 3.6561 | 3.6104 | 3.7747 | 3.5805 |
| mixed/known/person_color | 3.7435 | 3.6163 | 3.5808 | 3.7264 | 3.5581 |
| from_SLR/known/image | 3.4210 | 3.5995 | 3.8244 | 3.2311 | 3.6006 |
| from_SLR/known/person_site | 3.3848 | 3.5396 | 3.7422 | 3.1936 | 3.5183 |
| from_SLR/known/site | 3.3870 | 3.5640 | 3.7848 | 3.2312 | 3.5896 |
| from_SLR/known/color | 3.4000 | 3.5397 | 3.7258 | 3.2406 | 3.5214 |
| from_SLR/known/color_ipw | 3.3841 | 3.5468 | 3.7567 | 3.2011 | 3.5429 |
| from_SLR/known/person_color | 3.4478 | 3.6366 | 3.8774 | 3.2447 | 3.6410 |
| from_SLR/unseen/image | 5.4877 | 5.7066 | 6.2298 | 5.5556 | 5.8688 |
| from_SLR/unseen/person_site | 5.7261 | 5.8415 | 6.3051 | 5.4999 | 5.8817 |
| from_SLR/unseen/site | 6.0016 | 6.1897 | 6.7522 | 5.6250 | 6.2222 |
| from_SLR/unseen/color | 5.7192 | 5.7497 | 6.1300 | 5.5271 | 5.7954 |
| from_SLR/unseen/color_ipw | 5.7416 | 5.7538 | 6.1043 | 5.7681 | 5.8750 |
| from_SLR/unseen/person_color | 6.1545 | 6.6762 | 7.4325 | 5.6200 | 6.6008 |
| from_ipod/known/image | 4.0558 | 3.7594 | 3.7239 | 4.0050 | 3.5686 |
| from_ipod/known/person_site | 3.9949 | 3.6854 | 3.6186 | 3.9592 | 3.4699 |
| from_ipod/known/site | 4.0226 | 3.7639 | 3.7186 | 3.9571 | 3.5458 |
| from_ipod/known/color | 4.0722 | 3.8166 | 3.7815 | 4.0312 | 3.6060 |
| from_ipod/known/color_ipw | 4.0460 | 3.7123 | 3.6394 | 4.0037 | 3.4730 |
| from_ipod/known/person_color | 4.0492 | 3.7343 | 3.6774 | 3.9987 | 3.5076 |
| from_ipod/unseen/image | 6.5602 | 6.1725 | 6.0229 | 6.5114 | 5.6142 |
| from_ipod/unseen/person_site | 6.0825 | 5.4695 | 5.2299 | 6.3985 | 5.0854 |
| from_ipod/unseen/site | 6.1048 | 5.6020 | 5.3826 | 6.2809 | 5.1367 |
| from_ipod/unseen/color | 6.6475 | 6.2406 | 6.1306 | 6.5616 | 5.7747 |
| from_ipod/unseen/color_ipw | 6.4344 | 6.0012 | 5.8297 | 6.4994 | 5.4604 |
| from_ipod/unseen/person_color | 6.4074 | 5.7353 | 5.4434 | 6.7609 | 5.2665 |

![Privileged offset diagnostic](offset_diagnostic.png)

A shared offset does not consistently explain transfer. In SLR-to-iPod, image
control5.4877 becomes6.2298 with the full excluded-person correction; combination
6.1545 becomes7.4325. In reverse, image6.5602 becomes6.0229 and combination
6.4074 becomes5.4434. These are conditional reference-aided observations, not
achieved calibration-free gains. Ordinary phone accuracy remains unvalidated.

## Why a correction can hurt

A separate algebraic diagnostic decomposes site/person-balanced squared native
Lab residual energy. It is Euclidean Lab geometry, NOT CIEDE2000. If mu is the
mean person residual, v the between-person residual energy and K the number
of people, the excluded-person offset changes squared error by
`(-2a+a²)||mu||² + (2a/(K-1)+a²/(K-1)²)v` at strength a.
A shared correction can amplify person differences more than it removes bias.

| Protocol / domain / model | Shared residual energy | Between-person energy | Full squared-Lab error change |
|---|---:|---:|---:|
| mixed/known/image | 2.5555 | 2.3321 | -1.5294 |
| mixed/known/person_site | 2.1072 | 2.1105 | -1.1785 |
| mixed/known/site | 2.2859 | 2.1257 | -1.3506 |
| mixed/known/color | 1.5574 | 2.1911 | -0.5934 |
| mixed/known/color_ipw | 2.1742 | 2.1671 | -1.2206 |
| mixed/known/person_color | 1.9402 | 2.2250 | -0.9612 |
| from_SLR/known/image | 0.2956 | 4.4279 | 5.2393 |
| from_SLR/known/person_site | 0.4117 | 4.1252 | 4.7447 |
| from_SLR/known/site | 0.2967 | 4.4324 | 5.2438 |
| from_SLR/known/color | 0.4512 | 3.8753 | 4.3929 |
| from_SLR/known/color_ipw | 0.4072 | 4.1934 | 4.8345 |
| from_SLR/known/person_color | 0.3421 | 4.5778 | 5.3802 |
| from_SLR/unseen/image | 8.8848 | 14.3471 | 9.0491 |
| from_SLR/unseen/person_site | 10.6186 | 15.7607 | 9.0823 |
| from_SLR/unseen/site | 10.6270 | 20.6881 | 15.2331 |
| from_SLR/unseen/color | 11.2316 | 13.0405 | 5.0691 |
| from_SLR/unseen/color_ipw | 11.3555 | 13.2408 | 5.1955 |
| from_SLR/unseen/person_color | 6.2580 | 24.2008 | 23.9930 |
| from_ipod/known/image | 6.1965 | 0.7977 | -5.1994 |
| from_ipod/known/person_site | 5.8350 | 0.5684 | -5.1245 |
| from_ipod/known/site | 5.2940 | 0.6105 | -4.5309 |
| from_ipod/known/color | 5.2369 | 0.8107 | -4.2235 |
| from_ipod/known/color_ipw | 6.3395 | 0.5989 | -5.5910 |
| from_ipod/known/person_color | 6.2348 | 0.6882 | -5.3746 |
| from_ipod/unseen/image | 14.0537 | 3.1965 | -10.0581 |
| from_ipod/unseen/person_site | 16.6367 | 2.3505 | -13.6985 |
| from_ipod/unseen/site | 14.7500 | 2.3020 | -11.8725 |
| from_ipod/unseen/color | 16.0726 | 4.3444 | -10.6420 |
| from_ipod/unseen/color_ipw | 14.7394 | 3.2081 | -10.7292 |
| from_ipod/unseen/person_color | 19.0466 | 2.0845 | -16.4410 |

This identity explains only the squared-Euclidean diagnostic. Primary skin
accuracy remains the scalar-verified DeltaE00 table. These descriptive residual
terms use labels and are not inference features, a noise floor or an analytic
DeltaE00 decomposition. Between-person effects cannot be assigned causally to
biology, camera, capture settings or reference noise from these data alone.

## Integrity

180 exact corrected arrays; 324 exclusion folds with own-reference perturbation checks.
42768 independent scalar color cases; 1620 fixed-coverage rows; 180 exact energy identities.
Explicit exclusion matrices have zero own-person blocks and unit row sums.
Original image/site/person roles and array hashes are preserved. All fixed
coverage levels and full curves use the original uncalibrated novelty ranking.
There is no new calibrated selective-risk guarantee or latency/export result.
Original MSKCC CC-BY; no new data/weights, cloud or external publication.
Independent model result remains primary4.4570/80%4.1591 vs ordinary fusion
4.3005/4.1447. This diagnostic must not replace those scores.
[Research decision](../../research/skin_offset_diagnostic_next_decision.md).
