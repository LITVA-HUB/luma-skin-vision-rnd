# Conditional patch-distribution results

48 locally reproduced fits on original MSKCC photographs and actual instrument-native Lab.
All errors are skin DeltaE00. Source validation is extensively reused; no new
independent, ordinary-phone or universal-camera result. Seed scores are averaged,
not ensembled. One-Gaussian statistical fits have one deterministic initialization.

| Protocol / family | Initializations | Mean | Median | p95 | Common 80% |
|---|---:|---:|---:|---:|---:|
| mixed/bag_d1_k1 | 1 | 4.9960 | 4.4199 | 9.9673 | 4.9910 |
| mixed/bag_d1_k3 | 3 | 4.2103 | 3.8964 | 8.6981 | 4.1913 |
| mixed/bag_d2_k1 | 1 | 4.7854 | 4.2638 | 9.9554 | 4.7591 |
| mixed/bag_d2_k3 | 3 | 4.0554 | 3.4875 | 8.6304 | 3.9947 |
| mixed/mean_d1_k1 | 1 | 5.1126 | 4.4212 | 10.7429 | 5.1404 |
| mixed/mean_d1_k3 | 3 | 4.1832 | 3.6952 | 8.4315 | 4.1538 |
| mixed/mean_d2_k1 | 1 | 4.8533 | 4.1516 | 10.4523 | 4.8495 |
| mixed/mean_d2_k3 | 3 | 4.0120 | 3.4598 | 8.6074 | 3.9506 |
| from_SLR/bag_d1_k1 | 1 | 5.9690 | 5.4105 | 10.7989 | 5.6272 |
| from_SLR/bag_d1_k3 | 3 | 6.4100 | 6.0247 | 12.1073 | 6.1340 |
| from_SLR/bag_d2_k1 | 1 | 5.7480 | 5.4769 | 10.4481 | 5.4046 |
| from_SLR/bag_d2_k3 | 3 | 6.6379 | 6.0045 | 14.6197 | 5.5029 |
| from_SLR/mean_d1_k1 | 1 | 5.8589 | 5.2602 | 10.6963 | 5.5374 |
| from_SLR/mean_d1_k3 | 3 | 6.6616 | 6.0116 | 12.5591 | 6.1750 |
| from_SLR/mean_d2_k1 | 1 | 5.6521 | 5.4606 | 10.2894 | 5.3286 |
| from_SLR/mean_d2_k3 | 3 | 7.3072 | 5.7168 | 15.6043 | 5.8339 |
| from_ipod/bag_d1_k1 | 1 | 8.0489 | 6.9837 | 14.9378 | 7.2322 |
| from_ipod/bag_d1_k3 | 3 | 7.8077 | 6.8819 | 14.6904 | 6.7560 |
| from_ipod/bag_d2_k1 | 1 | 8.4358 | 7.2554 | 15.9115 | 7.7306 |
| from_ipod/bag_d2_k3 | 3 | 6.6877 | 5.7653 | 13.4961 | 6.2920 |
| from_ipod/mean_d1_k1 | 1 | 9.7372 | 8.7850 | 17.3444 | 9.0319 |
| from_ipod/mean_d1_k3 | 3 | 8.5645 | 8.0872 | 16.0283 | 7.5717 |
| from_ipod/mean_d2_k1 | 1 | 9.7187 | 8.6462 | 17.9220 | 9.2626 |
| from_ipod/mean_d2_k3 | 3 | 8.3810 | 7.7791 | 15.8246 | 7.4360 |

All families within a protocol use identical input-novelty accept sets.
This is an uncalibrated comparison of color accuracy at fixed coverage, not C+.
p95 above averages per-fit p95. All six fixed coverages and complete curves are retained.

![Risk and coverage](risk_coverage.png)

## Descriptive matched differences

Negative favors the candidate. Patient-cluster resampling uses 10,000 draws;
intervals on the small reused source cohort are not confirmatory or multiplicity-adjusted.

| Protocol | Candidate vs control | Mean difference | Patient 95% interval |
|---|---|---:|---|
| mixed | bag_d1_k1 vs mean_d1_k1 | -0.1165 | [-0.2902, 0.0570] |
| mixed | bag_d1_k3 vs mean_d1_k3 | 0.0271 | [-0.0779, 0.1561] |
| mixed | bag_d2_k1 vs mean_d2_k1 | -0.0678 | [-0.2082, 0.0725] |
| mixed | bag_d2_k3 vs mean_d2_k3 | 0.0434 | [-0.0630, 0.1528] |
| mixed | bag_d1_k3 vs bag_d1_k1 | -0.7858 | [-1.2638, -0.3339] |
| mixed | bag_d2_k3 vs bag_d2_k1 | -0.7300 | [-1.2955, -0.2848] |
| mixed | bag_d1_k1 vs bag_d1_k1 collapsed inference | 0.0000 | [0.0000, 0.0000] |
| mixed | bag_d1_k3 vs bag_d1_k3 collapsed inference | 0.0098 | [-0.0382, 0.0599] |
| mixed | bag_d2_k1 vs bag_d2_k1 collapsed inference | 0.0000 | [0.0000, 0.0000] |
| mixed | bag_d2_k3 vs bag_d2_k3 collapsed inference | -0.0030 | [-0.0391, 0.0331] |
| from_SLR | bag_d1_k1 vs mean_d1_k1 | 0.1101 | [0.0210, 0.1797] |
| from_SLR | bag_d1_k3 vs mean_d1_k3 | -0.2516 | [-0.5016, -0.0455] |
| from_SLR | bag_d2_k1 vs mean_d2_k1 | 0.0959 | [0.0095, 0.1628] |
| from_SLR | bag_d2_k3 vs mean_d2_k3 | -0.6692 | [-1.1065, -0.4003] |
| from_SLR | bag_d1_k3 vs bag_d1_k1 | 0.4410 | [-0.3774, 1.1282] |
| from_SLR | bag_d2_k3 vs bag_d2_k1 | 0.8900 | [-0.0026, 1.8394] |
| from_SLR | bag_d1_k1 vs bag_d1_k1 collapsed inference | 0.0000 | [-0.0000, 0.0000] |
| from_SLR | bag_d1_k3 vs bag_d1_k3 collapsed inference | -0.1875 | [-0.4183, -0.0575] |
| from_SLR | bag_d2_k1 vs bag_d2_k1 collapsed inference | -0.0000 | [-0.0000, 0.0000] |
| from_SLR | bag_d2_k3 vs bag_d2_k3 collapsed inference | -0.0716 | [-0.2060, 0.1193] |
| from_ipod | bag_d1_k1 vs mean_d1_k1 | -1.6883 | [-2.1146, -1.1049] |
| from_ipod | bag_d1_k3 vs mean_d1_k3 | -0.7568 | [-0.8682, -0.5481] |
| from_ipod | bag_d2_k1 vs mean_d2_k1 | -1.2830 | [-1.5004, -1.1029] |
| from_ipod | bag_d2_k3 vs mean_d2_k3 | -1.6933 | [-2.3370, -0.7110] |
| from_ipod | bag_d1_k3 vs bag_d1_k1 | -0.2412 | [-0.5414, 0.1959] |
| from_ipod | bag_d2_k3 vs bag_d2_k1 | -1.7481 | [-2.3769, -0.9551] |
| from_ipod | bag_d1_k1 vs bag_d1_k1 collapsed inference | 0.0000 | [-0.0000, 0.0000] |
| from_ipod | bag_d1_k3 vs bag_d1_k3 collapsed inference | 0.0775 | [0.0495, 0.1152] |
| from_ipod | bag_d2_k1 vs bag_d2_k1 collapsed inference | -0.0000 | [-0.0000, 0.0000] |
| from_ipod | bag_d2_k3 vs bag_d2_k3 collapsed inference | -0.0131 | [-0.0336, 0.0228] |

## Distribution and strength diagnostics

Collapsed inference uses the same model and selected strength; it is an
information intervention, not an independently trained baseline. The mean-trained
families supply that baseline. Strength is selected on same-camera validation only.

| Protocol / family | Strengths | Full bag/model mean | Collapsed mean | Posterior 80% |
|---|---|---:|---:|---:|
| mixed/bag_d1_k1 | [1] | 4.9960 | 4.9960 | 5.2028 |
| mixed/bag_d1_k3 | [1, 1, 1] | 4.2103 | 4.2005 | 4.1098 |
| mixed/bag_d2_k1 | [1] | 4.7854 | 4.7854 | 4.8944 |
| mixed/bag_d2_k3 | [1, 1, 1] | 4.0554 | 4.0584 | 3.9921 |
| mixed/mean_d1_k1 | [1] | 5.1126 | 5.1126 | 5.3385 |
| mixed/mean_d1_k3 | [1, 1, 1] | 4.1832 | 4.1832 | 4.2454 |
| mixed/mean_d2_k1 | [1] | 4.8533 | 4.8533 | 4.9936 |
| mixed/mean_d2_k3 | [1, 1, 1] | 4.0120 | 4.0120 | 3.9645 |
| from_SLR/bag_d1_k1 | [1] | 5.9690 | 5.9690 | 5.9566 |
| from_SLR/bag_d1_k3 | [1, 1, 1] | 6.4100 | 6.5974 | 5.9600 |
| from_SLR/bag_d2_k1 | [1] | 5.7480 | 5.7480 | 5.8336 |
| from_SLR/bag_d2_k3 | [1, 1, 1] | 6.6379 | 6.7095 | 6.7746 |
| from_SLR/mean_d1_k1 | [1] | 5.8589 | 5.8589 | 5.8862 |
| from_SLR/mean_d1_k3 | [1, 1, 1] | 6.6616 | 6.6616 | 6.5225 |
| from_SLR/mean_d2_k1 | [1] | 5.6521 | 5.6521 | 5.7663 |
| from_SLR/mean_d2_k3 | [1, 1, 1] | 7.3072 | 7.3072 | 7.5242 |
| from_ipod/bag_d1_k1 | [1] | 8.0489 | 8.0489 | 8.7506 |
| from_ipod/bag_d1_k3 | [1, 1, 1] | 7.8077 | 7.7301 | 8.3281 |
| from_ipod/bag_d2_k1 | [1] | 8.4358 | 8.4358 | 9.0992 |
| from_ipod/bag_d2_k3 | [1, 1, 1] | 6.6877 | 6.7007 | 6.6324 |
| from_ipod/mean_d1_k1 | [1] | 9.7372 | 9.7372 | 10.5093 |
| from_ipod/mean_d1_k3 | [1, 1, 1] | 8.5645 | 8.5645 | 9.3065 |
| from_ipod/mean_d2_k1 | [1] | 9.7187 | 9.7187 | 10.4317 |
| from_ipod/mean_d2_k3 | [1, 1, 1] | 8.3810 | 8.3810 | 8.8616 |

12 Gaussian fits obey the sufficient-mean identity, maximum score-difference residual 2.84e-14.
The conditional mixture can encode non-Gaussian structure, but that is not a
demonstrated improvement in general skin-color accuracy. Actual patch RGB is
observed; native Lab refers to the site, not a measured per-pixel color map.
432 exact color arrays; 42240 scalar color cases; 2880 coverage rows; 84480 curve points.
120 weighted normal equations, 249600 independent densities and 476160 expected-color scalar costs.
This is a CPU statistical falsifier with no GPU latency/VRAM/export claim.
Stored numeric scalars range 290 to 877; largest model archive 9,466 bytes.

Original MSKCC CC-BY; no new data/weights or participant publication.
Historical source controls remain stronger in key protocols: mixture mixed3.4406,
paired mixture forward4.8328, training-only graph reverse4.9736. Do not promote
a gain over weaker new controls as a project-wide victory. Independent MSKCC
primary4.4570/80%4.1591 and ordinary fusion4.3005/4.1447 remain unchanged.
[Next decision](../../research/skin_graph_patch_next_decision.md).
