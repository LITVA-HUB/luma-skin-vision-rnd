# Real paired skin-color calibration: He 2021 pilot

Measured on 100 facial sites from20 held-out people. Training used200 sites
from40 different people. Both reference XYZ and corresponding regional RAW/JPG
RGB originate in the author's CC BY4.0 workbook. These are source-calibrated
ordinary regressors, not the V7 image model or a proposed new architecture.

All14 controls were frozen before numeric TEST extraction. A5-subject-group
cross-validation selected RAW poly3 and JPG poly2 before the test. No test
tuning, demographics, site identity, camera labels or full images were inputs.

|Input|Control|Selected by source CV|Source OOF XYZ RMSE|Test XYZ RMSE|Test80% XYZ RMSE|
|---|---|---|---:|---:|---:|
|raw|constant|False|4.346908|4.147158|3.699085|
|raw|linear3|False|2.340966|1.866550|1.851333|
|raw|affine4|False|2.290256|1.830634|1.825671|
|raw|poly2|False|2.082153|1.999538|1.937395|
|raw|poly3|True|2.072378|2.025092|1.949136|
|raw|root2|False|2.197008|1.998954|1.950271|
|raw|mlp_5_25_5|False|2.133432|1.947308|1.961226|
|jpg|constant|False|4.346908|4.147158|3.635247|
|jpg|linear3|False|2.460762|2.302701|2.133216|
|jpg|affine4|False|2.117998|1.860912|1.765333|
|jpg|poly2|True|2.047101|1.952336|1.822263|
|jpg|poly3|False|2.123525|1.980025|1.896993|
|jpg|root2|False|2.399310|2.323195|2.126804|
|jpg|mlp_5_25_5|False|2.064161|1.920980|1.895292|

XYZ RMSE is the square root of the average squared discrepancy across all
three original instrument-coordinate channels. It is neither DeltaE nor
angular illumination error, nor a percentage accuracy. It has no established
cosmetics acceptance interpretation here. The measured reference-white
convention is not numerically verified, so DeltaE00/DeltaE76 are NOT CALCULATED.

raw source-selected control: XYZ RMSE 2.025092, 95% subject-cluster bootstrap interval [1.782612, 2.245452].
Fixed100/95/90/80/70/60% XYZ RMSE: 2.025092, 1.921689, 1.924661, 1.949136, 1.960870, 1.906485.

jpg source-selected control: XYZ RMSE 1.952336, 95% subject-cluster bootstrap interval [1.728709, 2.149566].
Fixed100/95/90/80/70/60% XYZ RMSE: 1.952336, 1.921919, 1.877768, 1.822263, 1.863061, 1.857726.

The shared5-nearest-training-RGB distance is an exploratory ranking, not a
calibrated expected-error estimate. Lower coverage does not monotonically
lower measured risk. Do not infer reliable production rejection from it.

Five MLP convergence warnings across source fits are retained in source_fit.json.
No rerun or optimizer retuning followed test exposure. The MLP uses sklearn
L-BFGS and is not a reproduction of MATLAB Bayesian-regularized trainbr.

Limits: one controlled Canon6D MarkII capture setup with polarization, no
unseen phone/lighting test, no full-image normalization or region extraction
evaluation, no repeated capture noise floor and no validated cosmetic decision.
Only skin regional RGB-to-measured-XYZ calibration was tested. Skin-accuracy
claims for the Luma image pipeline and its V7 weights remain unmeasured.

[Original author data](https://zenodo.org/records/5532176),
[paper](https://doi.org/10.1002/col.22737),
[frozen protocol](../../research/skin_he_xyz_protocol_v1.md),
[all metrics](evaluation/results.json), [independent audit](independent_audit.json).
