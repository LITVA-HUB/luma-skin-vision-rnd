# Spatially valid graph/support combination

36 locally reproduced fits on original MSKCC photographs and actual instrument-native Lab.
All errors are skin DeltaE00. Source validation is extensively reused; no new
independent, ordinary-phone or universal-camera result. Seed scores are averaged,
not ensembled. One-Gaussian statistical fits have one deterministic initialization.

| Protocol / family | Initializations | Mean | Median | p95 | Common 80% |
|---|---:|---:|---:|---:|---:|
| mixed/graph_paired | 3 | 3.5143 | 3.0417 | 7.0727 | 3.4239 |
| mixed/graph_raw | 3 | 3.4621 | 2.9707 | 7.1813 | 3.3877 |
| mixed/plain_paired | 3 | 3.5438 | 3.1156 | 7.0690 | 3.4662 |
| mixed/plain_raw | 3 | 3.4855 | 2.9738 | 7.1819 | 3.4136 |
| from_SLR/graph_paired | 3 | 5.2801 | 4.9537 | 9.7880 | 5.1110 |
| from_SLR/graph_raw | 3 | 5.4616 | 5.0482 | 10.3841 | 5.6054 |
| from_SLR/plain_paired | 3 | 5.4804 | 5.0277 | 10.3911 | 5.2830 |
| from_SLR/plain_raw | 3 | 5.6063 | 5.1288 | 10.9750 | 5.7226 |
| from_ipod/graph_paired | 3 | 5.6610 | 5.5284 | 9.8796 | 5.9994 |
| from_ipod/graph_raw | 3 | 5.9890 | 5.8624 | 10.6926 | 6.4210 |
| from_ipod/plain_paired | 3 | 5.8331 | 5.7398 | 10.2474 | 6.2410 |
| from_ipod/plain_raw | 3 | 6.1399 | 5.9801 | 10.7483 | 6.5662 |

All families within a protocol use identical input-novelty accept sets.
This is an uncalibrated comparison of color accuracy at fixed coverage, not C+.
p95 above averages per-fit p95. All six fixed coverages and complete curves are retained.

![Risk and coverage](risk_coverage.png)

## Descriptive matched differences

Negative favors the candidate. Patient-cluster resampling uses 10,000 draws;
intervals on the small reused source cohort are not confirmatory or multiplicity-adjusted.

| Protocol | Candidate vs control | Mean difference | Patient 95% interval |
|---|---|---:|---|
| mixed | graph_paired vs plain_raw | 0.0288 | [-0.0754, 0.1235] |
| mixed | graph_paired vs plain_paired | -0.0295 | [-0.0812, 0.0280] |
| mixed | graph_paired vs graph_raw | 0.0522 | [-0.0331, 0.1375] |
| mixed | graph_raw vs plain_raw | -0.0234 | [-0.0669, 0.0242] |
| mixed | plain_paired vs plain_raw | 0.0583 | [-0.0289, 0.1456] |
| from_SLR | graph_paired vs plain_raw | -0.3263 | [-1.2608, 0.4712] |
| from_SLR | graph_paired vs plain_paired | -0.2004 | [-0.3579, -0.0473] |
| from_SLR | graph_paired vs graph_raw | -0.1815 | [-0.7901, 0.6997] |
| from_SLR | graph_raw vs plain_raw | -0.1448 | [-0.4707, 0.2650] |
| from_SLR | plain_paired vs plain_raw | -0.1259 | [-0.9029, 0.6671] |
| from_ipod | graph_paired vs plain_raw | -0.4790 | [-0.6524, -0.2328] |
| from_ipod | graph_paired vs plain_paired | -0.1721 | [-0.3209, -0.0135] |
| from_ipod | graph_paired vs graph_raw | -0.3281 | [-0.4861, -0.1061] |
| from_ipod | graph_raw vs plain_raw | -0.1509 | [-0.1663, -0.1266] |
| from_ipod | plain_paired vs plain_raw | -0.3068 | [-0.3697, -0.2193] |

Graph processing occurs only on the original first-pass training image.
The second, potentially mixed bag uses the plain core. All inference is plain.
The new two-pass schedule is not identical to the older graph_always experiment.
Inference parameters: 924,932; stored parameters: 993,287.
Maximum fit allocation 145.21 MiB; largest full checkpoint 3,997,915 bytes.
108 exact color arrays; 6336 scalar cases; 216 coverage rows; 6336 curve points.
36 actual-source runtime branch checks and all epoch pair/plan digests pass.
No new isolated inference latency or export claim.

Original MSKCC CC-BY; no new data/weights or participant publication.
Historical source controls remain stronger in key protocols: mixture mixed3.4406,
paired mixture forward4.8328, training-only graph reverse4.9736. Do not promote
a gain over weaker new controls as a project-wide victory. Independent MSKCC
primary4.4570/80%4.1591 and ordinary fusion4.3005/4.1447 remain unchanged.
[Next decision](../../research/skin_graph_patch_next_decision.md).
