# Instrument reference and paired-capture correspondence

Original TRAIN: 966 images / 248 sites / 24 people; VALIDATION: 264 / 66 / 6.
Each site has three readings copied exactly across image rows. Copies count
once. No fitting, image decoding or reserved endpoint access. Native DeltaE00.

| Role | Pairwise reading mean | Reading-to-mean | Leave-one-reading-out |
|---|---:|---:|---:|
| train | 2.5757 | 1.5195 | 2.2809 |
| validation | 2.6139 | 1.5354 | 2.3043 |

These discrepancies are NOT a known sensor-noise level or an irreducible model
floor. Do not subtract them from model error; the target uses the same readings.

| Protocol | Model | Single image mean | Diagnostic multi-view mean | Shared squared-Lab fraction | Error >5 against all 3 readings |
|---|---|---:|---:|---:|---:|
| mixed | plain_mse | 3.4771 | 3.0120 | 75.26% | 10.98% |
| mixed | mixture_mse | 3.4406 | 2.8719 | 71.65% | 11.11% |
| mixed | graph_always | 3.6394 | 3.1481 | 74.79% | 14.14% |
| from_SLR | plain_mse | 5.8301 | 5.4606 | 81.31% | 43.43% |
| from_SLR | mixture_mse | 5.0193 | 4.6598 | 81.59% | 36.11% |
| from_SLR | graph_always | 5.8339 | 5.6420 | 88.29% | 47.98% |
| from_ipod | plain_mse | 5.5089 | 4.0398 | 61.05% | 42.42% |
| from_ipod | mixture_mse | 5.9635 | 4.1870 | 58.64% | 48.48% |
| from_ipod | graph_always | 4.9736 | 3.9778 | 63.33% | 34.60% |

Values average three seed scores, not predictions. Multi-view averaging uses
all site views and is NOT a single-image model result. Shared fraction is an
Euclidean Lab squared-error decomposition, NOT a DeltaE00 decomposition or
causal attribution. Camera groups contain different people.

TRAIN site-L quartiles define strata; validation has 32/16/11/7 sites per bin.
It is not distribution-balanced. All mode/device/lightness strata and all 27
endpoints remain in [summary](summary.json).

## Interpretation and opponent check

Repeat spread is material, but cannot explain away all model errors. Shared
site bias dominates in many protocols: agreement alone does not mean accuracy.
Next test weakens rather than strengthens the relative agreement penalty.
Errors may cancel under that training objective; only single-image accuracy
will decide. The multiple-view diagnostic is not an achieved deployment gain.

## Source methods and verification

The [original study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12749783/) describes
gentle-contact triplicate colorimetry and four capture settings. Contact photos
used alcohol and a glass lens; both camera setups had initial white balance
calibration. High ITA repeatability measures something different from full-Lab
DeltaE00 discrepancy. These methods differ from ordinary facial photography.

Independent scalar color checks: 30150; maximum gap 4.66e-15.
Exact reference copies, patient joins, target order and saved errors checked.
Squared-error decomposition identities pass.

[Protocol](../../research/skin_correspondence_protocol_v1.md), [bindings](source_lock.json).
