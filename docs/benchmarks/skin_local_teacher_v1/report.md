# Local descriptor and RGB correspondence

All 36 predeclared source fits retained. Native instrument Lab / CIEDE2000.
Every reported model prediction uses one image. Values average three seed
scores, not predictions. Source camera families have already been examined;
this is neither fresh independent confirmation nor facial-phone accuracy.

| Protocol | Correspondence | Mean DeltaE00 | p95 | Three seed means |
|---|---|---:|---:|---|
| mixed | plain | 3.4406 | 6.9316 | 3.3783, 3.4698, 3.4736 |
| mixed | aligned | 4.1720 | 8.2774 | 4.2477, 4.2023, 4.0660 |
| mixed | global | 4.0992 | 8.0044 | 4.1424, 4.0603, 4.0948 |
| mixed | shuffled | 4.1826 | 8.1786 | 4.2621, 4.2069, 4.0787 |
| mixed | historical_plain_mse | 3.4771 | 7.0733 | 3.4957, 3.5284, 3.4072 |
| mixed | historical_mixture_mse | 3.4406 | 6.9316 | 3.3783, 3.4698, 3.4736 |
| from_SLR | plain | 5.0193 | 9.7825 | 4.9975, 5.2417, 4.8187 |
| from_SLR | aligned | 6.0116 | 13.2466 | 6.1183, 5.8730, 6.0436 |
| from_SLR | global | 5.9560 | 12.7014 | 5.8623, 6.1725, 5.8331 |
| from_SLR | shuffled | 5.9191 | 12.8089 | 5.8949, 5.8426, 6.0199 |
| from_SLR | historical_plain_mse | 5.8301 | 11.2459 | 5.6802, 6.0602, 5.7499 |
| from_SLR | historical_mixture_mse | 5.0193 | 9.7825 | 4.9975, 5.2417, 4.8187 |
| from_ipod | plain | 5.9635 | 11.0896 | 5.7634, 6.4146, 5.7124 |
| from_ipod | aligned | 7.8197 | 13.3947 | 8.2828, 7.8802, 7.2963 |
| from_ipod | global | 7.7831 | 13.6317 | 8.4186, 7.0437, 7.8870 |
| from_ipod | shuffled | 7.9399 | 13.7294 | 8.6006, 7.9304, 7.2886 |
| from_ipod | historical_plain_mse | 5.5089 | 9.4018 | 5.3087, 5.6826, 5.5353 |
| from_ipod | historical_mixture_mse | 5.9635 | 11.0896 | 5.7634, 6.4146, 5.7124 |

| Protocol | Correspondence | Mean at 80% | Active head + required teacher parameters | Peak fit+selection MiB |
|---|---|---:|---:|---:|
| mixed | plain | 3.4407 | 929297 + 0 | 108.68 |
| mixed | aligned | 4.1590 | 1011601 + 22056576 | 232.62 |
| mixed | global | 4.0446 | 1011601 + 22056576 | 232.62 |
| mixed | shuffled | 4.2122 | 1011601 + 22056576 | 232.62 |
| from_SLR | plain | 5.1087 | 929297 + 0 | 105.23 |
| from_SLR | aligned | 6.0805 | 1011601 + 22056576 | 161.50 |
| from_SLR | global | 6.2725 | 1011601 + 22056576 | 161.50 |
| from_SLR | shuffled | 5.8207 | 1011601 + 22056576 | 161.50 |
| from_ipod | plain | 5.7914 | 929297 + 0 | 106.65 |
| from_ipod | aligned | 7.9010 | 1011601 + 22056576 | 193.51 |
| from_ipod | global | 7.8224 | 1011601 + 22056576 | 193.51 |
| from_ipod | shuffled | 7.9990 | 1011601 + 22056576 | 193.51 |

Risk is uncalibrated hypothesis RMS Lab dispersion, not predicted DeltaE00.
All six coverage levels are in run JSONs; [full curves](uncalibrated_risk_coverage.csv).
Teacher-assisted arms require22,056,576frozen teacher parameters at inference in
addition to the trained head. Cached features do not remove this cost. Plain
needs only the929,297parameter RGB core. No compact student was trained.
No new latency, deployment, or camera-independent accuracy claim.

A pre-fitting verifier-only precision amendment is archived. The original
1e-6absolute manual-pooling tolerance rejected validFP32rounding. The corrected
check uses the standardfour-term summation bound; allcachebytes are unchanged.
All1,230spatial grids replay exactly;2,048cells are manuallyFP64checked.

Audit: 108 exact color arrays, 36 gate/hypothesis replays,
6336 independent scalar color cases, 216 coverage rows,
9 exact historical control final weights.
TRAIN-only scales and same-camera epoch selection checked. Reserved endpoints unused.

[Protocol](../../research/skin_local_teacher_protocol_v1.md), [audit](audit.json),
[decision](../../research/skin_local_teacher_next_decision.md).

## Descriptive comparison with strongest historical single-model control

Post-hoc source comparisons, not confirmatory inference. Each person is one
bootstrap cluster; seed errors are averaged within person before 5,000 resamples.
Only 3 or 6 validation people, and adaptive source reuse, limit these intervals.

| Protocol | Correspondence | Reference | Seed wins | Person mean difference | 95% descriptive interval |
|---|---|---|---:|---:|---|
| mixed | plain | mixture_mse | 0/3 | 0.0000 | [0.0000, 0.0000] |
| mixed | aligned | mixture_mse | 0/3 | 0.7314 | [0.2394, 1.1096] |
| mixed | global | mixture_mse | 0/3 | 0.6586 | [0.1762, 0.9987] |
| mixed | shuffled | mixture_mse | 0/3 | 0.7420 | [0.2773, 1.0745] |
| from_SLR | plain | mixture_mse | 0/3 | 0.0000 | [0.0000, 0.0000] |
| from_SLR | aligned | mixture_mse | 0/3 | 0.9923 | [-0.8010, 2.5213] |
| from_SLR | global | mixture_mse | 0/3 | 0.9367 | [-0.7605, 2.2760] |
| from_SLR | shuffled | mixture_mse | 0/3 | 0.8998 | [-0.8244, 2.2329] |
| from_ipod | plain | plain_mse | 0/3 | 0.4546 | [0.1312, 0.8048] |
| from_ipod | aligned | plain_mse | 0/3 | 2.3109 | [1.7867, 3.1140] |
| from_ipod | global | plain_mse | 0/3 | 2.2742 | [1.5822, 3.1059] |
| from_ipod | shuffled | plain_mse | 0/3 | 2.4310 | [1.7080, 3.3812] |
