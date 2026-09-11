# Real skin color: allocation combination under source camera transfer

54 REPRODUCED LOCALLY fits,90 known/unseen evaluation arrays. Actual original
MSKCC photographs and instrument-native Lab. No illuminant angular proxies.
Original TRAIN and extensively reused source VALIDATION only; no TEST/CAL.
All fits use the same core and930 optimizer updates, fixed final checkpoint.
No validation-guided fitting, weights, hyperparameters or checkpoint selection.
Groups average three individual model runs; they are not ensemble predictions.

| Training / evaluation / sampler | Mean | Median | p95 | Patient mean | At80% | Above10 |
|---|---:|---:|---:|---:|---:|---:|
| mixed/known/image | 3.8277 | 3.4822 | 7.9298 | 3.8277 | 3.8015 | 1.14% |
| mixed/known/person_site | 3.8120 | 3.4567 | 7.9895 | 3.8120 | 3.7878 | 0.63% |
| mixed/known/site | 3.7959 | 3.4476 | 7.9459 | 3.7959 | 3.7675 | 0.63% |
| mixed/known/color | 3.7864 | 3.4865 | 7.7171 | 3.7864 | 3.7591 | 1.26% |
| mixed/known/color_ipw | 3.7976 | 3.4154 | 7.9000 | 3.7976 | 3.7747 | 1.01% |
| mixed/known/person_color | 3.7435 | 3.3686 | 7.7915 | 3.7435 | 3.7264 | 1.01% |
| from_SLR/known/image | 3.4210 | 3.2170 | 6.4298 | 3.4210 | 3.2311 | 0.00% |
| from_SLR/known/person_site | 3.3848 | 3.1132 | 6.3137 | 3.3848 | 3.1936 | 0.00% |
| from_SLR/known/site | 3.3870 | 3.2159 | 6.3447 | 3.3870 | 3.2312 | 0.00% |
| from_SLR/known/color | 3.4000 | 3.0875 | 6.5445 | 3.4000 | 3.2406 | 0.00% |
| from_SLR/known/color_ipw | 3.3841 | 3.2477 | 6.4679 | 3.3841 | 3.2011 | 0.00% |
| from_SLR/known/person_color | 3.4478 | 3.2220 | 6.6598 | 3.4478 | 3.2447 | 0.25% |
| from_SLR/unseen/image | 5.4877 | 5.2429 | 10.3253 | 5.4877 | 5.5556 | 6.57% |
| from_SLR/unseen/person_site | 5.7261 | 5.4088 | 11.5638 | 5.7261 | 5.4999 | 8.59% |
| from_SLR/unseen/site | 6.0016 | 5.3900 | 12.4886 | 6.0016 | 5.6250 | 9.85% |
| from_SLR/unseen/color | 5.7192 | 5.5148 | 10.5814 | 5.7192 | 5.5271 | 8.59% |
| from_SLR/unseen/color_ipw | 5.7416 | 5.4874 | 10.7699 | 5.7416 | 5.7681 | 7.58% |
| from_SLR/unseen/person_color | 6.1545 | 5.8536 | 12.6455 | 6.1545 | 5.6200 | 10.86% |
| from_ipod/known/image | 4.0558 | 3.2771 | 9.2260 | 4.0558 | 4.0050 | 2.27% |
| from_ipod/known/person_site | 3.9949 | 3.3238 | 8.9791 | 3.9949 | 3.9592 | 1.52% |
| from_ipod/known/site | 4.0226 | 3.3182 | 9.0430 | 4.0226 | 3.9571 | 2.27% |
| from_ipod/known/color | 4.0722 | 3.4337 | 9.3097 | 4.0722 | 4.0312 | 3.03% |
| from_ipod/known/color_ipw | 4.0460 | 3.3866 | 8.9786 | 4.0460 | 4.0037 | 1.52% |
| from_ipod/known/person_color | 4.0492 | 3.3518 | 9.1016 | 4.0492 | 3.9987 | 1.52% |
| from_ipod/unseen/image | 6.5602 | 6.2911 | 10.7986 | 6.5602 | 6.5114 | 7.07% |
| from_ipod/unseen/person_site | 6.0825 | 5.8573 | 10.3434 | 6.0825 | 6.3985 | 6.31% |
| from_ipod/unseen/site | 6.1048 | 5.9169 | 10.0853 | 6.1048 | 6.2809 | 6.82% |
| from_ipod/unseen/color | 6.6475 | 6.5098 | 11.1611 | 6.6475 | 6.5616 | 12.12% |
| from_ipod/unseen/color_ipw | 6.4344 | 6.1225 | 11.1394 | 6.4344 | 6.4994 | 8.08% |
| from_ipod/unseen/person_color | 6.4074 | 6.2723 | 11.0685 | 6.4074 | 6.7609 | 8.84% |

## Main result

The combined person/color sampler gives a small mixed-source gain, but it
does not carry across acquisition models. Its SLR-to-iPod error is worse than
the image-uniform control, and reverse transfer loses to simpler person/site
balancing. It is not a universal winner and is not promoted for deployment.
Do not choose a different winning sampler per held-out camera and call the
result a camera-independent system.

The combined reverse-transfer80% error is worse than full-coverage error.
The shared input-novelty ranking does not provide reliable selective refusal.
It is not C+ or calibrated expected color error. Full fixed100/95/90/80/70/60%
coverage metrics are retained per fit and all curves in risk_coverage.csv.

![Risk and coverage](risk_coverage.png)

## Paired descriptive comparisons

Negative favors combination. Patient-balanced intervals use10,000 cluster
resamples after seed averaging. Mixed has six people; single-camera domains
have only three. Intervals are exploratory and not multiplicity-adjusted;
they do not describe the differently weighted image-mean difference.

| Protocol / domain | Combined vs | Image difference | Patient difference | Patient interval |
|---|---|---:|---:|---|
| mixed/known | image | -0.0842 | -0.0842 | [-0.1575, -0.0277] |
| mixed/known | person_site | -0.0684 | -0.0684 | [-0.1574, 0.0200] |
| mixed/known | site | -0.0524 | -0.0524 | [-0.1476, 0.0414] |
| mixed/known | color | -0.0428 | -0.0428 | [-0.1667, 0.0802] |
| mixed/known | color_ipw | -0.0540 | -0.0540 | [-0.1498, 0.0408] |
| from_SLR/known | image | 0.0267 | 0.0267 | [-0.0105, 0.0592] |
| from_SLR/known | person_site | 0.0629 | 0.0629 | [0.0175, 0.1360] |
| from_SLR/known | site | 0.0608 | 0.0608 | [-0.0100, 0.1731] |
| from_SLR/known | color | 0.0478 | 0.0478 | [-0.1228, 0.1960] |
| from_SLR/known | color_ipw | 0.0636 | 0.0636 | [0.0029, 0.1222] |
| from_SLR/unseen | image | 0.6668 | 0.6668 | [-0.0657, 1.9885] |
| from_SLR/unseen | person_site | 0.4283 | 0.4283 | [-0.0373, 0.9416] |
| from_SLR/unseen | site | 0.1529 | 0.1529 | [-0.0378, 0.3968] |
| from_SLR/unseen | color | 0.4352 | 0.4352 | [-0.3049, 1.1922] |
| from_SLR/unseen | color_ipw | 0.4129 | 0.4129 | [-0.3919, 1.7155] |
| from_ipod/known | image | -0.0066 | -0.0066 | [-0.0273, 0.0325] |
| from_ipod/known | person_site | 0.0543 | 0.0543 | [-0.0315, 0.1426] |
| from_ipod/known | site | 0.0267 | 0.0267 | [0.0063, 0.0510] |
| from_ipod/known | color | -0.0230 | -0.0230 | [-0.1543, 0.1794] |
| from_ipod/known | color_ipw | 0.0033 | 0.0033 | [-0.0265, 0.0266] |
| from_ipod/unseen | image | -0.1528 | -0.1528 | [-0.3337, 0.0854] |
| from_ipod/unseen | person_site | 0.3249 | 0.3249 | [0.1971, 0.4194] |
| from_ipod/unseen | site | 0.3026 | 0.3026 | [0.2641, 0.3572] |
| from_ipod/unseen | color | -0.2401 | -0.2401 | [-0.3407, -0.0511] |
| from_ipod/unseen | color_ipw | -0.0270 | -0.0270 | [-0.0640, 0.0316] |

## Scope and reproducibility

mixed uses24 TRAIN people/966images and6 known VALIDATION people/264images.
SLR training uses8 people/323images; iPod training16people/643images. Each
single-camera evaluation domain has3 people/132images. Training people/sites
are disjoint from evaluation. The unseen camera is absent from all training
and sampling. Camera and person composition remain confounded; these are
special clinical acquisition protocols, not ordinary smartphone selfies.
p95 above averages per-fit quantiles. Prior80-epoch results such as mixed3.4406,
forward4.8328 and reverse4.9736 remain stronger historical references under
different schedules; they are not matched930-step controls.

90 exact color arrays; 6 full state refits; 14256 independent scalar color cases; 540 fixed-coverage rows.
476490 scalar TRAIN density pairs and 144 combined-mass/within-person-ratio identities.
Each model 929,297 parameters; maximum measured training allocation 106.23 MiB; largest checkpoint 3,722,355 bytes.
No new isolated inference latency or export claim; validation follows all fitting.
Original MSKCC CC-BY. No new data/weights, cloud resources or publication.
Independent MSKCC primary4.4570/80%4.1591 versus ordinary fusion4.3005/4.1447
is unchanged. Skin-color measurement on arbitrary phones and innovative
advantage over the strongest independent baseline remain unproven.
[Research decision](../../research/skin_sampling_transfer_next_decision.md).
