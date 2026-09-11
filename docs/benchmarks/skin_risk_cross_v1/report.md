# Frozen color/risk crossing on real skin source data

90locally computed method/seed/protocol endpoints; no new image fitting.
Actual MSKCC instrument-native Lab is the reference. Source validation is
heavily reused. No independent test, ordinary-phone or calibrated C+ claim.

Entries average three separate seed/pair scores, not a three-run ensemble.

| Protocol | System | Mean | p95 | At80% | p95at80% | Models |
|---|---|---:|---:|---:|---:|---:|
| mixed | ordinary_dispersion | 3.4406 | 6.9316 | 3.4407 | 6.8591 | 1 |
| mixed | ordinary_gaussian_expected | 3.4406 | 6.9316 | 3.2621 | 6.5107 | 2 |
| mixed | ordinary_gaussian_covariance | 3.4406 | 6.9316 | 3.2659 | 6.7278 | 2 |
| mixed | ordinary_cross_disagreement | 3.4406 | 6.9316 | 3.2875 | 6.5671 | 2 |
| mixed | ordinary_pair_disagreement | 3.3680 | 6.9395 | 3.4018 | 6.8602 | 2 |
| mixed | ordinary_pair_dispersion | 3.3680 | 6.9395 | 3.3625 | 6.7866 | 2 |
| mixed | mixed_pair_expected | 3.3788 | 6.9971 | 3.1977 | 6.7769 | 2 |
| mixed | mixed_pair_covariance | 3.3788 | 6.9971 | 3.1954 | 6.8067 | 2 |
| mixed | mixed_pair_disagreement | 3.3788 | 6.9971 | 3.2429 | 6.6265 | 2 |
| mixed | gaussian_expected | 3.4605 | 7.5348 | 3.2593 | 7.0951 | 1 |
| from_SLR | ordinary_dispersion | 5.0193 | 9.7825 | 5.1087 | 9.6635 | 1 |
| from_SLR | ordinary_gaussian_expected | 5.0193 | 9.7825 | 4.9293 | 9.7357 | 2 |
| from_SLR | ordinary_gaussian_covariance | 5.0193 | 9.7825 | 4.9639 | 9.6683 | 2 |
| from_SLR | ordinary_cross_disagreement | 5.0193 | 9.7825 | 5.0409 | 9.9071 | 2 |
| from_SLR | ordinary_pair_disagreement | 4.8747 | 9.4964 | 4.8547 | 9.5500 | 2 |
| from_SLR | ordinary_pair_dispersion | 4.8747 | 9.4964 | 5.0339 | 9.6907 | 2 |
| from_SLR | mixed_pair_expected | 5.1263 | 10.1101 | 5.0262 | 10.0203 | 2 |
| from_SLR | mixed_pair_covariance | 5.1263 | 10.1101 | 5.0250 | 10.0203 | 2 |
| from_SLR | mixed_pair_disagreement | 5.1263 | 10.1101 | 5.2910 | 10.3901 | 2 |
| from_SLR | gaussian_expected | 5.6571 | 11.2796 | 5.4955 | 10.9031 | 1 |
| from_ipod | ordinary_dispersion | 5.9635 | 11.0896 | 5.7914 | 10.1810 | 1 |
| from_ipod | ordinary_gaussian_expected | 5.9635 | 11.0896 | 6.1737 | 11.1575 | 2 |
| from_ipod | ordinary_gaussian_covariance | 5.9635 | 11.0896 | 6.1282 | 11.2014 | 2 |
| from_ipod | ordinary_cross_disagreement | 5.9635 | 11.0896 | 6.0235 | 10.5178 | 2 |
| from_ipod | ordinary_pair_disagreement | 5.9025 | 10.7281 | 5.8333 | 10.3266 | 2 |
| from_ipod | ordinary_pair_dispersion | 5.9025 | 10.7281 | 5.6588 | 9.9522 | 2 |
| from_ipod | mixed_pair_expected | 5.7134 | 10.5926 | 5.9451 | 10.9189 | 2 |
| from_ipod | mixed_pair_covariance | 5.7134 | 10.5926 | 5.9368 | 10.9189 | 2 |
| from_ipod | mixed_pair_disagreement | 5.7134 | 10.5926 | 5.8009 | 10.4063 | 2 |
| from_ipod | gaussian_expected | 5.7474 | 10.2485 | 5.9347 | 10.3551 | 1 |

Ordinary pairs cover17/29,29/43,43/17. Mixed pairs combine same-seed ordinary
and Gaussian predictions. All averages are fixed50/50. Gaussian integration
uses4096antithetic Sobol points; no true reference participates in ranking.
Covariance-only risk is centered at the proposed answer, removing mean bias.
Expected risk uses the Gaussian mean; disagreement is a cheaper comparator.

## Matched-compute comparisons

Patient bootstrap resamples source people and recomputes exact80%coverage
inside each draw for each seed. Intervals are descriptive after extensive
source reuse, not confirmatory or multiplicity-adjusted.

| Protocol | Candidate vs control | Difference at80% | Patient95% interval |
|---|---|---:|---|
| mixed | mixed_pair_expected vs ordinary_pair_dispersion | -0.1648 | [-0.4428,0.1349] |
| mixed | mixed_pair_covariance vs ordinary_pair_dispersion | -0.1671 | [-0.4692,0.1323] |
| mixed | ordinary_gaussian_expected vs ordinary_dispersion | -0.1785 | [-0.4763,0.0949] |
| from_SLR | mixed_pair_expected vs ordinary_pair_dispersion | -0.0077 | [-0.2263,0.4876] |
| from_SLR | mixed_pair_covariance vs ordinary_pair_dispersion | -0.0089 | [-0.2523,0.4739] |
| from_SLR | ordinary_gaussian_expected vs ordinary_dispersion | -0.1794 | [-0.2693,0.1297] |
| from_ipod | mixed_pair_expected vs ordinary_pair_dispersion | 0.2863 | [-0.0252,0.5385] |
| from_ipod | mixed_pair_covariance vs ordinary_pair_dispersion | 0.2780 | [-0.0314,0.4869] |
| from_ipod | ordinary_gaussian_expected vs ordinary_dispersion | 0.3822 | [0.1345,0.6874] |

![Risk and coverage](risk_coverage.png)

Mixed-source crossing improves selected error; transfer is not consistently
better than matched ordinary pairs. A partial source signal is not universality.
Ordinary two-model pairs also improve color accuracy without the proposed risk.
Historical training-only graph remains a stronger reverse-color control4.9736.

Ordinary pair active parameters1,858,594; mixed pair1,864,750. These results
require two model forwards plus Gaussian integration where used. No new latency
or inference-memory claim. This screen does not include a fitted C+ error head;
any later calibration requires person-held-out residuals.

368640independent scalar integral cases; 15840scalar color cases;
540coverage rows; 15840curve points. Maximum gap4e-15.
Original CC-BY source data and all contributing predictions remain hash-bound.
No participant data, weights or claims are externally published.
