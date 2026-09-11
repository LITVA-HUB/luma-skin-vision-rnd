# Shared skin-color bias versus paired-view agreement

All 36 predeclared source fits retained. Native instrument Lab / CIEDE2000.
Every reported model prediction uses one image. Values average three seed
scores, not predictions. Source camera families have already been examined;
this is neither fresh independent confirmation nor facial-phone accuracy.

| Protocol | Objective | Mean DeltaE00 | p95 | Three seed means |
|---|---|---:|---:|---|
| mixed | individual | 3.4406 | 6.9316 | 3.3783, 3.4698, 3.4736 |
| mixed | consistency | 3.4178 | 6.9574 | 3.3449, 3.4495, 3.4590 |
| mixed | shared_half | 3.5246 | 7.2980 | 3.4837, 3.5525, 3.5376 |
| mixed | shared_only | 3.7398 | 7.8739 | 3.7515, 3.7150, 3.7529 |
| mixed | historical_plain_mse | 3.4771 | 7.0733 | 3.4957, 3.5284, 3.4072 |
| mixed | historical_mixture_mse | 3.4406 | 6.9316 | 3.3783, 3.4698, 3.4736 |
| from_SLR | individual | 5.0193 | 9.7825 | 4.9975, 5.2417, 4.8187 |
| from_SLR | consistency | 5.3455 | 10.4776 | 4.9746, 6.0530, 5.0089 |
| from_SLR | shared_half | 5.2770 | 10.3365 | 5.2087, 5.3088, 5.3136 |
| from_SLR | shared_only | 6.1655 | 11.3368 | 6.2568, 6.2203, 6.0193 |
| from_SLR | historical_plain_mse | 5.8301 | 11.2459 | 5.6802, 6.0602, 5.7499 |
| from_SLR | historical_mixture_mse | 5.0193 | 9.7825 | 4.9975, 5.2417, 4.8187 |
| from_ipod | individual | 5.9635 | 11.0896 | 5.7634, 6.4146, 5.7124 |
| from_ipod | consistency | 5.9443 | 10.0827 | 6.3482, 5.9495, 5.5351 |
| from_ipod | shared_half | 5.8533 | 11.2498 | 5.5949, 6.3378, 5.6272 |
| from_ipod | shared_only | 6.3318 | 11.7212 | 5.9044, 7.2157, 5.8753 |
| from_ipod | historical_plain_mse | 5.5089 | 9.4018 | 5.3087, 5.6826, 5.5353 |
| from_ipod | historical_mixture_mse | 5.9635 | 11.0896 | 5.7634, 6.4146, 5.7124 |

| Protocol | Objective | Mean at 80% | Stored parameters | Peak fit+selection MiB |
|---|---|---:|---:|---:|
| mixed | individual | 3.4407 | 929297 | 108.36 |
| mixed | consistency | 3.4848 | 929297 | 108.36 |
| mixed | shared_half | 3.4790 | 929297 | 108.36 |
| mixed | shared_only | 3.6172 | 929297 | 108.36 |
| from_SLR | individual | 5.1087 | 929297 | 104.91 |
| from_SLR | consistency | 5.0748 | 929297 | 104.91 |
| from_SLR | shared_half | 5.3922 | 929297 | 104.91 |
| from_SLR | shared_only | 5.9525 | 929297 | 104.91 |
| from_ipod | individual | 5.7914 | 929297 | 106.33 |
| from_ipod | consistency | 5.9729 | 929297 | 106.33 |
| from_ipod | shared_half | 5.5306 | 929297 | 106.33 |
| from_ipod | shared_only | 5.8205 | 929297 | 106.33 |

Risk is uncalibrated hypothesis RMS Lab dispersion, not predicted DeltaE00.
All six coverage levels are in run JSONs; [full curves](uncalibrated_risk_coverage.csv).
No new latency, deployment, or camera-independent accuracy claim.

Audit: 108 exact color arrays, 36 gate/hypothesis replays,
6336 independent scalar color cases, 216 coverage rows,
9 exact historical control final weights.
TRAIN-only scales and same-camera epoch selection checked. Reserved endpoints unused.

[Protocol](../../research/skin_shared_bias_protocol_v1.md), [audit](audit.json),
[decision](../../research/skin_correspondence_next_decision.md).

## Descriptive comparison with strongest historical single-model control

Post-hoc source comparisons, not confirmatory inference. Each person is one
bootstrap cluster; seed errors are averaged within person before 5,000 resamples.
Only 3 or 6 validation people, and adaptive source reuse, limit these intervals.

| Protocol | Objective | Reference | Seed wins | Person mean difference | 95% descriptive interval |
|---|---|---|---:|---:|---|
| mixed | individual | mixture_mse | 0/3 | 0.0000 | [0.0000, 0.0000] |
| mixed | consistency | mixture_mse | 3/3 | -0.0228 | [-0.0650, 0.0254] |
| mixed | shared_half | mixture_mse | 0/3 | 0.0840 | [0.0340, 0.1272] |
| mixed | shared_only | mixture_mse | 0/3 | 0.2992 | [0.2081, 0.3771] |
| from_SLR | individual | mixture_mse | 0/3 | 0.0000 | [0.0000, 0.0000] |
| from_SLR | consistency | mixture_mse | 1/3 | 0.3262 | [-0.5392, 1.2023] |
| from_SLR | shared_half | mixture_mse | 0/3 | 0.2577 | [-0.3046, 0.5762] |
| from_SLR | shared_only | mixture_mse | 0/3 | 1.1462 | [-0.6378, 3.1769] |
| from_ipod | individual | plain_mse | 0/3 | 0.4546 | [0.1312, 0.8048] |
| from_ipod | consistency | plain_mse | 1/3 | 0.4354 | [0.0151, 0.6613] |
| from_ipod | shared_half | plain_mse | 0/3 | 0.3444 | [0.1055, 0.6744] |
| from_ipod | shared_only | plain_mse | 0/3 | 0.8229 | [0.5124, 1.1705] |
